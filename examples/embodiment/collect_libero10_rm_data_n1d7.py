"""Collect balanced GR00T-N1.7 rollouts for a shared-semantic reward expert.

The collector stores the exact token packet consumed by the DiT, including its
source frame, semantic version, episode generation, packet age, and wallclock
timestamps. It supports both coupled smoke collection and the real decoupled
semantic-server path without running a second VLM for reward inference.

One compressed NPZ is written per episode and indexed by JSONL.  Observations have
length T + 1 so the actual success/timeout frame is retained.  Transition arrays
have length T; chunk arrays additionally have length C (the GR00T action chunk).

Run one process per GPU on disjoint task sets, for example:
  CUDA_VISIBLE_DEVICES=0 python examples/embodiment/collect_libero10_rm_data_n15.py \
    --checkpoint /path/to/GR00T-N1.7 --output /data/libero10_rm \
    --tasks 0,1,2 --num-envs 16
  CUDA_VISIBLE_DEVICES=1 python examples/embodiment/collect_libero10_rm_data_n15.py \
    --checkpoint /path/to/RLinf-Gr00t-SFT-10 --output /data/libero10_rm \
    --tasks 3,4,5 --num-envs 16

The defaults target 750 successes and 750 failures for each of all 10 tasks.
Use --inference-mode train to retain Flow-SDE exploration.  If the frozen policy
does not naturally produce both outcomes, action noise/dropout can be enabled,
and the perturbation is recorded in every trajectory.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = "libero10-gr00t-n1.7-shared-semantic-rm-v2"
OUTCOMES = ("success", "failure")
MANIFEST_THREAD_LOCK = threading.Lock()


def as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """Detach a tensor and return a NumPy array."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def copy_model_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Copy observations because GR00T preprocessing mutates the state tensor."""
    copied: dict[str, Any] = {}
    for key, value in obs.items():
        if torch.is_tensor(value):
            copied[key] = value.clone()
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


def parse_tasks(spec: str) -> list[int]:
    """Parse task specifications such as 0-9 or 0,2,4-6."""
    tasks: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            begin, end = (int(x) for x in part.split("-", 1))
            tasks.update(range(begin, end + 1))
        else:
            tasks.add(int(part))
    result = sorted(tasks)
    if not result or result[0] < 0 or result[-1] > 9:
        raise ValueError(f"LIBERO-10 task ids must be in [0, 9], got {result}")
    return result


def build_model(args: argparse.Namespace):
    """Load N1.7 in coupled or semantic-server-decoupled execution mode."""
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    config_path = Path(__file__).parent / "config" / "model" / "gr00t_n1d7.yaml"
    model_cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(model_cfg, False)
    model_cfg.model_path = str(Path(args.checkpoint).resolve())
    if args.backbone_model_path:
        model_cfg.backbone_model_path = str(Path(args.backbone_model_path).resolve())
    model_cfg.model_type = "gr00t_n1d7"
    model_cfg.num_action_chunks = args.action_chunk
    model_cfg.add_value_head = False
    model_cfg.rl_head_config.add_value_head = False
    model_cfg.rl_head_config.use_vlm_value = False
    model_cfg.rl_head_config.action_noise_scale = 0.0
    model_cfg.rl_head_config.disable_dropout = True
    model_cfg.rl_head_config.semantic_feature_tokens = args.semantic_tokens
    model_cfg.rl_head_config.semantic_age_mode = "simulator"
    model_cfg.rl_head_config.semantic_control_hz = args.control_hz
    # Dataset age is measured in simulator frames; wallclock request delay only
    # reduces collection throughput and does not create additional supervision.
    model_cfg.rl_head_config.semantic_fetch_delay_fraction = 0.0
    model_cfg.rl_head_config.semantic_fetch_delay_initial_ms = 0.0
    model_cfg.rl_head_config.semantic_fetch_delay_min_ms = 0.0
    model_cfg.rl_head_config.semantic_fetch_delay_max_ms = 0.0
    model_cfg.rl_head_config.initialize_packet_age_adapter = True

    if args.semantic_server:
        model_cfg.rl_head_config.execution_mode = "decoupled"
        model_cfg.rl_head_config.semantic_server_enabled = True
        model_cfg.rl_head_config.semantic_server_non_blocking = True
        model_cfg.rl_head_config.semantic_server_central_cache = True
        model_cfg.rl_head_config.semantic_server_boundary_publish = True
        model_cfg.rl_head_config.semantic_boundary_publish_interval = 1
        model_cfg.rl_head_config.semantic_env_bootstrap_publish = False
        model_cfg.rl_head_config.semantic_publish_interval_frames = 0
        model_cfg.rl_head_config.semantic_server_host = args.semantic_server_host
        model_cfg.rl_head_config.semantic_server_port = str(args.semantic_server_port)
        model_cfg.rl_head_config.semantic_server_publish_port = str(
            args.semantic_server_publish_port
        )
        model_cfg.rl_head_config.semantic_server_timeout_ms = args.semantic_timeout_ms
        model_cfg.rl_head_config.semantic_fetch_target_age_frames = (
            args.semantic_target_age_frames
        )
        model_cfg.rl_head_config.semantic_fetch_hard_max_age_frames = (
            args.semantic_hard_max_age_frames
        )
        model_cfg.rl_head_config.semantic_fetch_max_wait_ms = args.semantic_timeout_ms
        model_cfg.rl_head_config.drop_local_backbone = True
    else:
        model_cfg.rl_head_config.execution_mode = "coupled"
        model_cfg.rl_head_config.semantic_server_enabled = False
        model_cfg.rl_head_config.drop_local_backbone = False

    model = get_model(model_cfg, torch_dtype=torch.bfloat16).to(args.device)
    model.eval()
    return model, model_cfg


def make_env(
    task_id: int,
    seed: int,
    num_envs: int,
    max_episode_steps: int,
):
    """Create vectorized LIBERO envs without auto-reset.

    Auto-reset must be disabled so the actual terminal observation is available
    for the frame-success positive label.
    """
    from omegaconf import OmegaConf

    from rlinf.envs.libero.libero_env import LiberoEnv

    env_cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": "libero_10",
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": max_episode_steps,
            "max_episode_steps": max_episode_steps,
            "use_fixed_reset_state_ids": False,
            "use_ordered_reset_state_ids": False,
            "use_rel_reward": True,
            "reward_coef": 5.0,
            "reset_gripper_open": True,
            "is_eval": False,
            "seed": seed,
            "group_size": 1,
            "task_id_filter": [task_id],
            "video_cfg": {"save_video": False, "info_on_video": False},
            "init_params": {"camera_heights": 256, "camera_widths": 256},
        }
    )
    return LiberoEnv(
        cfg=env_cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def obs_at(obs: dict[str, Any], index: int) -> dict[str, np.ndarray]:
    """Extract the raw RM observation for one vector-env slot."""
    return {
        "main_image": as_numpy(obs["main_images"][index], np.uint8).copy(),
        "wrist_image": as_numpy(obs["wrist_images"][index], np.uint8).copy(),
        "state": as_numpy(obs["states"][index], np.float32).copy(),
    }


def batch_from_records(
    records: list[dict[str, np.ndarray]],
    language: str,
) -> dict[str, Any]:
    """Build a GR00T observation batch from stored raw observations."""
    return {
        "main_images": torch.from_numpy(
            np.stack([record["main_image"] for record in records])
        ),
        "wrist_images": torch.from_numpy(
            np.stack([record["wrist_image"] for record in records])
        ),
        "states": torch.from_numpy(np.stack([record["state"] for record in records])),
        "task_descriptions": [language] * len(records),
    }


def attach_rollout_metadata(
    observation: dict[str, Any],
    *,
    env_ids: list[int],
    frame_ids: list[int],
    episode_generations: list[int],
    task_id: int,
    trial_ids: list[int],
) -> dict[str, Any]:
    """Attach the causal identity used by the central semantic cache."""
    batch_size = len(env_ids)
    if not (len(frame_ids) == len(episode_generations) == len(trial_ids) == batch_size):
        raise ValueError("rollout metadata lengths do not match")
    result = dict(observation)
    result["__rlinf_semantic_env_ids"] = torch.tensor(env_ids, dtype=torch.int64)
    result["__rlinf_semantic_frame_ids"] = torch.tensor(frame_ids, dtype=torch.int64)
    result["__rlinf_semantic_generations"] = torch.tensor(
        episode_generations, dtype=torch.int64
    )
    result["__rlinf_semantic_observation_wallclock_s"] = torch.full(
        (batch_size,), time.time(), dtype=torch.float64
    )
    result["__rlinf_task_ids"] = torch.full(
        (batch_size,), int(task_id), dtype=torch.int64
    )
    result["__rlinf_trial_ids"] = torch.tensor(trial_ids, dtype=torch.int64)
    return result


def unpack_features(
    result: dict[str, Any],
    *,
    action_frame_ids: list[int],
    episode_generations: list[int],
) -> dict[str, np.ndarray]:
    """Extract the exact semantic packet stashed for PPO replay."""
    forward_inputs = result.get("forward_inputs")
    if not isinstance(forward_inputs, dict):
        raise RuntimeError("GR00T N1.7 rollout did not return forward_inputs")
    if "semantic_backbone_features" not in forward_inputs:
        raise RuntimeError("GR00T N1.7 rollout is missing semantic backbone tokens")

    tokens = as_numpy(forward_inputs["semantic_backbone_features"], np.float16)
    mask_value = forward_inputs.get("semantic_backbone_attention_mask")
    attention_mask = (
        np.ones(tokens.shape[:-1], dtype=bool)
        if mask_value is None
        else as_numpy(mask_value, bool)
    )
    batch_size = tokens.shape[0]
    action_frames = np.asarray(action_frame_ids, dtype=np.int64)
    generations = np.asarray(episode_generations, dtype=np.int64)

    def field(name: str, default: Any, dtype: np.dtype) -> np.ndarray:
        value = forward_inputs.get(name, default)
        array = as_numpy(value, dtype).reshape(-1)
        if array.size == 1 and batch_size != 1:
            array = np.repeat(array, batch_size)
        if array.size != batch_size:
            raise ValueError(f"{name} has {array.size} rows for batch {batch_size}")
        return array

    source_frames = field("rollout_semantic_source_frame_ids", action_frames, np.int64)
    packet_age_s = field(
        "packet_age_s",
        np.maximum(action_frames - source_frames, 0) / 20.0,
        np.float32,
    )
    state = forward_inputs.get("state")
    if state is None:
        state = np.zeros((batch_size, 0), dtype=np.float32)
    action_history = forward_inputs.get("action_history")
    if action_history is None:
        action_history = np.zeros((batch_size, 0, 0), dtype=np.float32)
    embodiment_id = forward_inputs.get("embodiment_id")
    if embodiment_id is None:
        embodiment_id = np.zeros(batch_size, dtype=np.int64)
    normalized_actions = forward_inputs.get("rollout_normalized_executed_actions")
    if normalized_actions is None:
        raise RuntimeError("GR00T rollout is missing normalized executed actions")
    return {
        "semantic_tokens": tokens,
        "semantic_attention_mask": attention_mask,
        "semantic_source_frame_id": source_frames,
        "semantic_version": field("rollout_semantic_versions", source_frames, np.int64),
        "semantic_episode_generation": field(
            "rollout_semantic_episode_generations", generations, np.int64
        ),
        "semantic_source_wallclock_s": field(
            "rollout_semantic_source_wallclock_s", 0.0, np.float64
        ),
        "semantic_completed_wallclock_s": field(
            "rollout_semantic_completed_wallclock_s", 0.0, np.float64
        ),
        "action_frame_id": field("action_frame_ids", action_frames, np.int64),
        "action_wallclock_s": field("action_wallclock_s", time.time(), np.float64),
        "packet_age_s": packet_age_s,
        "action_state": as_numpy(state, np.float32),
        "action_history": as_numpy(action_history, np.float32),
        "embodiment_id": as_numpy(embodiment_id, np.int64).reshape(batch_size),
        "normalized_executed_action": as_numpy(normalized_actions, np.float32),
    }


def new_buffer(
    first_obs: dict[str, np.ndarray],
    init_state_id: int,
    corruption_seed: int,
) -> dict[str, Any]:
    """Allocate one in-flight episode buffer."""
    return {
        "observations": [first_obs],
        "semantic_tokens": [],
        "semantic_attention_mask": [],
        "semantic_source_frame_id": [],
        "semantic_version": [],
        "semantic_episode_generation": [],
        "semantic_source_wallclock_s": [],
        "semantic_completed_wallclock_s": [],
        "action_frame_id": [],
        "action_wallclock_s": [],
        "packet_age_s": [],
        "action_state": [],
        "action_history": [],
        "embodiment_id": [],
        "delay_action_state": [],
        "delay_action_history": [],
        "delay_action_frame_id": [],
        "delay_valid_mask": [],
        "delay_completion": [],
        "policy_action": [],
        "executed_action": [],
        "chunk_reward": [],
        "chunk_terminated": [],
        "chunk_truncated": [],
        "action_valid_mask": [],
        "init_state_id": init_state_id,
        "corruption_seed": corruption_seed,
    }


def canonical_returns(
    valid_mask: np.ndarray,
    success: bool,
    gamma: float,
    failure_terminal_reward: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-observation canonical RTG and env steps remaining.

    Each valid environment step costs -1.  The terminal observation is worth 0
    on success and failure_terminal_reward otherwise.  Discounting is applied
    per environment step, not per decision.
    """
    valid_counts = valid_mask.sum(axis=1).astype(np.int32)
    returns = np.empty(len(valid_counts) + 1, dtype=np.float32)
    steps_to_end = np.empty(len(valid_counts) + 1, dtype=np.int32)
    running_return = 0.0 if success else float(failure_terminal_reward)
    running_steps = 0
    returns[-1] = running_return
    steps_to_end[-1] = 0
    for index in range(len(valid_counts) - 1, -1, -1):
        for _ in range(int(valid_counts[index])):
            running_return = -1.0 + gamma * running_return
        running_steps += int(valid_counts[index])
        returns[index] = running_return
        steps_to_end[index] = running_steps
    return returns, steps_to_end


def environment_returns(
    chunk_reward: np.ndarray,
    valid_mask: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute RTG from the unmodified LIBERO reward, per env step."""
    returns = np.zeros(chunk_reward.shape[0] + 1, dtype=np.float32)
    running = 0.0
    for decision in range(chunk_reward.shape[0] - 1, -1, -1):
        valid_indices = np.flatnonzero(valid_mask[decision])
        for chunk_index in valid_indices[::-1]:
            running = float(chunk_reward[decision, chunk_index]) + gamma * running
        returns[decision] = running
    return returns


def atomic_save_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Write one trajectory atomically so interrupted files are never indexed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl_locked(path: Path, record: dict[str, Any]) -> None:
    """Append a manifest row safely across task-parallel processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_THREAD_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def next_index(directory: Path, trajectory_prefix: str = "") -> int:
    """Find the next trajectory index for resume."""
    indices = []
    pattern = f"trajectory_{trajectory_prefix}*.npz"
    for path in directory.glob(pattern):
        try:
            indices.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def process_is_alive(pid: int) -> bool:
    """Return whether a reservation-owning process is still alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def live_reservations_locked(directory: Path) -> list[Path]:
    """List live reservations and remove markers left by dead collectors."""
    reservations: list[Path] = []
    for marker in directory.glob(".trajectory_*.reserved"):
        try:
            owner_pid = int(json.loads(marker.read_text(encoding="utf-8"))["pid"])
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            marker.unlink(missing_ok=True)
            continue
        if process_is_alive(owner_pid):
            reservations.append(marker)
        else:
            marker.unlink(missing_ok=True)
    return reservations


def quota_snapshot(
    task_root: Path,
    target: dict[str, int],
) -> tuple[dict[str, int], dict[str, int], bool]:
    """Read global complete and complete-plus-reserved task counts."""
    task_root.mkdir(parents=True, exist_ok=True)
    complete: dict[str, int] = {}
    occupied: dict[str, int] = {}
    with (task_root / ".quota.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for outcome in OUTCOMES:
            directory = task_root / outcome
            directory.mkdir(parents=True, exist_ok=True)
            complete[outcome] = len(list(directory.glob("trajectory_*.npz")))
            occupied[outcome] = complete[outcome] + len(
                live_reservations_locked(directory)
            )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    full = all(occupied[outcome] >= target[outcome] for outcome in OUTCOMES)
    return complete, occupied, full


def reserve_outcome_slot(
    task_root: Path,
    outcome: str,
    target: int,
    trajectory_prefix: str,
) -> tuple[int, Path, int] | None:
    """Reserve one globally quota-limited, prefix-unique trajectory slot."""
    if target <= 0:
        return None
    directory = task_root / outcome
    directory.mkdir(parents=True, exist_ok=True)
    with (task_root / ".quota.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing_all = list(directory.glob("trajectory_*.npz"))
        reservations = live_reservations_locked(directory)
        occupied = len(existing_all) + len(reservations)
        if occupied >= target:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return None
        prefix_indices = [
            int(path.stem.rsplit("_", 1)[1])
            for path in directory.glob(f"trajectory_{trajectory_prefix}*.npz")
        ]
        prefix_indices.extend(
            int(marker.name.removesuffix(".reserved").rsplit("_", 1)[1])
            for marker in reservations
            if marker.name.startswith(f".trajectory_{trajectory_prefix}")
        )
        index = max(prefix_indices, default=-1) + 1
        marker = directory / f".trajectory_{trajectory_prefix}{index:06d}.reserved"
        with marker.open("x", encoding="utf-8") as marker_handle:
            json.dump({"pid": os.getpid()}, marker_handle)
            marker_handle.flush()
            os.fsync(marker_handle.fileno())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return index, marker, occupied + 1


def write_schema(output: Path) -> None:
    """Write a concise machine-readable schema once."""
    schema = {
        "schema_version": SCHEMA_VERSION,
        "alignment": {
            "observation_arrays": "[T+1, ...]",
            "feature_arrays": "[T+1, ...]",
            "transition_arrays": "[T, ...]",
            "chunk_arrays": "[T, action_chunk, ...]",
        },
        "inputs": {
            "observation_main_image": "uint8 [T+1,H,W,3]",
            "observation_wrist_image": "uint8 [T+1,H,W,3]",
            "observation_state": "float32 [T+1,8]",
            "feature_semantic_tokens": "float16 [T+1,N,D], exact DiT packet",
            "feature_semantic_attention_mask": "bool [T+1,N]",
            "feature_action_state": "float32 [T+1,...], normalized DiT state",
            "feature_action_history": "float32 [T+1,H,A], DiT action history",
            "feature_embodiment_id": "int64 [T+1]",
            "delay_action_state": "float32 [T+1,L,...], current state at age 0..L-1",
            "delay_action_history": "float32 [T+1,L,H,A]",
            "delay_action_frame_id": "int64 [T+1,L]",
            "delay_valid_mask": "bool [T+1,L]",
            "semantic_source_frame_id": "int64 [T+1]",
            "semantic_version": "int64 [T+1]",
            "semantic_episode_generation": "int64 [T+1]",
            "semantic_source_wallclock_s": "float64 [T+1]",
            "semantic_completed_wallclock_s": "float64 [T+1]",
            "action_frame_id": "int64 [T+1]",
            "action_wallclock_s": "float64 [T+1]",
            "packet_age_s": "float32 [T+1]",
            "action_policy": "float32 [T,C,7], GR00T output",
            "action_executed": "float32 [T,C,7], sent to LIBERO",
            "action_valid_mask": "bool [T,C]",
        },
        "labels": {
            "label_frame_success": "bool [T+1], source-aligned observation success",
            "label_delay_completion": "bool [T+1,L], current successful chunk will complete",
            "label_episode_success": "bool scalar, eventual trajectory outcome",
            "label_canonical_return_to_go": "float32 [T+1]",
            "label_env_return_to_go": "float32 [T+1]",
            "label_steps_to_end": "int32 [T+1], valid env steps",
        },
    }
    path = output / "schema.json"
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    try:
        os.replace(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)


def save_episode(
    output: Path,
    task_id: int,
    language: str,
    checkpoint: str,
    inference_mode: str,
    action_chunk: int,
    gamma: float,
    failure_terminal_reward: float,
    action_noise_std: float,
    action_dropout_prob: float,
    buffer: dict[str, Any],
    success: bool,
    outcome_index: int,
    trajectory_prefix: str,
    source_stream: str,
    reservation_path: Path | None = None,
) -> Path:
    """Finalize labels and atomically save an episode."""
    outcome = "success" if success else "failure"
    observations = buffer["observations"]
    valid_mask = np.stack(buffer["action_valid_mask"]).astype(bool)
    chunk_reward = np.stack(buffer["chunk_reward"]).astype(np.float32)
    canonical_rtg, steps_to_end = canonical_returns(
        valid_mask, success, gamma, failure_terminal_reward
    )
    env_rtg = environment_returns(chunk_reward, valid_mask, gamma)
    frame_success = np.zeros(len(observations), dtype=bool)
    frame_terminal = np.zeros(len(observations), dtype=bool)
    frame_success[-1] = success
    frame_terminal[-1] = True

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "task_name": language,
        "language_instruction": language,
        "episode_success": success,
        "checkpoint": str(Path(checkpoint).resolve()),
        "inference_mode": inference_mode,
        "action_chunk": action_chunk,
        "gamma_per_env_step": gamma,
        "failure_terminal_reward": failure_terminal_reward,
        "init_state_id": buffer["init_state_id"],
        "corruption_seed": buffer["corruption_seed"],
        "action_noise_std": action_noise_std,
        "action_dropout_prob": action_dropout_prob,
        "trajectory_prefix": trajectory_prefix,
        "source_stream": source_stream,
        "feature_source": "action_rollout_forward_inputs",
        "reward_specific_backbone_forwards": 0,
        "terminal_packet_forced_fresh": True,
        "delay_max_frames": len(buffer["delay_valid_mask"][0]) - 1,
        "delay_condition_source": "exact_action_side_state_and_history",
    }
    arrays = {
        "observation_main_image": np.stack(
            [item["main_image"] for item in observations]
        ).astype(np.uint8),
        "observation_wrist_image": np.stack(
            [item["wrist_image"] for item in observations]
        ).astype(np.uint8),
        "observation_state": np.stack([item["state"] for item in observations]).astype(
            np.float32
        ),
        "feature_semantic_tokens": np.stack(buffer["semantic_tokens"]).astype(
            np.float16
        ),
        "feature_semantic_attention_mask": np.stack(
            buffer["semantic_attention_mask"]
        ).astype(bool),
        "feature_action_state": np.stack(buffer["action_state"]).astype(np.float32),
        "feature_action_history": np.stack(buffer["action_history"]).astype(np.float32),
        "feature_embodiment_id": np.asarray(buffer["embodiment_id"], dtype=np.int64),
        "delay_action_state": np.stack(buffer["delay_action_state"]).astype(np.float32),
        "delay_action_history": np.stack(buffer["delay_action_history"]).astype(
            np.float32
        ),
        "delay_action_frame_id": np.stack(buffer["delay_action_frame_id"]).astype(
            np.int64
        ),
        "delay_valid_mask": np.stack(buffer["delay_valid_mask"]).astype(bool),
        "semantic_source_frame_id": np.asarray(
            buffer["semantic_source_frame_id"], dtype=np.int64
        ),
        "semantic_version": np.asarray(buffer["semantic_version"], dtype=np.int64),
        "semantic_episode_generation": np.asarray(
            buffer["semantic_episode_generation"], dtype=np.int64
        ),
        "semantic_source_wallclock_s": np.asarray(
            buffer["semantic_source_wallclock_s"], dtype=np.float64
        ),
        "semantic_completed_wallclock_s": np.asarray(
            buffer["semantic_completed_wallclock_s"], dtype=np.float64
        ),
        "action_frame_id": np.asarray(buffer["action_frame_id"], dtype=np.int64),
        "action_wallclock_s": np.asarray(
            buffer["action_wallclock_s"], dtype=np.float64
        ),
        "packet_age_s": np.asarray(buffer["packet_age_s"], dtype=np.float32),
        "action_policy": np.stack(buffer["policy_action"]).astype(np.float32),
        "action_executed": np.stack(buffer["executed_action"]).astype(np.float32),
        "action_valid_mask": valid_mask,
        "transition_chunk_reward": chunk_reward,
        "transition_chunk_terminated": np.stack(buffer["chunk_terminated"]).astype(
            bool
        ),
        "transition_chunk_truncated": np.stack(buffer["chunk_truncated"]).astype(bool),
        "transition_decision_reward": (chunk_reward * valid_mask).sum(axis=1),
        "label_frame_success": frame_success,
        "label_delay_completion": np.stack(buffer["delay_completion"]).astype(bool),
        "label_frame_terminal": frame_terminal,
        "label_episode_success": np.asarray(success),
        "label_canonical_return_to_go": canonical_rtg,
        "label_env_return_to_go": env_rtg,
        "label_steps_to_end": steps_to_end,
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }

    episode_length = len(buffer["policy_action"])
    if len(observations) != episode_length + 1:
        raise ValueError("Observation/transition alignment is invalid")
    if int(buffer["semantic_source_frame_id"][-1]) != int(
        buffer["action_frame_id"][-1]
    ):
        raise ValueError("Terminal success label is not source-frame aligned")
    for feature_name in (
        "semantic_tokens",
        "semantic_attention_mask",
        "semantic_source_frame_id",
        "semantic_version",
        "semantic_episode_generation",
        "semantic_source_wallclock_s",
        "semantic_completed_wallclock_s",
        "action_frame_id",
        "action_wallclock_s",
        "packet_age_s",
        "action_state",
        "action_history",
        "embodiment_id",
        "delay_action_state",
        "delay_action_history",
        "delay_action_frame_id",
        "delay_valid_mask",
        "delay_completion",
    ):
        if len(buffer[feature_name]) != episode_length + 1:
            raise ValueError(f"{feature_name} does not have T+1 entries")

    directory = output / f"task_{task_id:02d}" / outcome
    path = directory / f"trajectory_{trajectory_prefix}{outcome_index:06d}.npz"
    try:
        atomic_save_npz(path, arrays)

        record = {
            "schema_version": SCHEMA_VERSION,
            "path": str(path.relative_to(output)),
            "task_id": task_id,
            "task_name": language,
            "outcome": outcome,
            "episode_success": success,
            "source_stream": source_stream,
            "num_decisions": episode_length,
            "num_env_steps": int(valid_mask.sum()),
            "init_state_id": buffer["init_state_id"],
        }
        append_jsonl_locked(output / f"task_{task_id:02d}" / "manifest.jsonl", record)
        append_jsonl_locked(output / "manifest.jsonl", record)
    finally:
        if reservation_path is not None:
            reservation_path.unlink(missing_ok=True)
    return path


def apply_action_corruption(
    action: np.ndarray,
    valid_envs: np.ndarray,
    rngs: list[np.random.Generator],
    noise_std: float,
    dropout_prob: float,
) -> np.ndarray:
    """Apply optional recorded exploration corruption independently per env."""
    executed = action.copy()
    for env_id in range(len(executed)):
        if not valid_envs[env_id]:
            executed[env_id] = 0
            executed[env_id, -1] = -1
            continue
        rng = rngs[env_id]
        if noise_std > 0:
            executed[env_id, :6] += rng.normal(
                0.0, noise_std, size=executed[env_id, :6].shape
            )
        if dropout_prob > 0 and rng.random() < dropout_prob:
            executed[env_id, :6] = 0
    return executed


def append_feature_row(
    buffer: dict[str, Any], features: dict[str, np.ndarray], row: int
) -> None:
    """Append one packet and its causal metadata to an episode buffer."""
    for key in (
        "semantic_tokens",
        "semantic_attention_mask",
        "semantic_source_frame_id",
        "semantic_version",
        "semantic_episode_generation",
        "semantic_source_wallclock_s",
        "semantic_completed_wallclock_s",
        "action_frame_id",
        "action_wallclock_s",
        "packet_age_s",
        "action_state",
        "action_history",
        "embodiment_id",
    ):
        buffer[key].append(features[key][row])


def append_duplicate_feature_row(buffer: dict[str, Any]) -> None:
    """Pad a timeout trajectory without running a label-free terminal DiT."""
    for key in (
        "semantic_tokens",
        "semantic_attention_mask",
        "semantic_source_frame_id",
        "semantic_version",
        "semantic_episode_generation",
        "semantic_source_wallclock_s",
        "semantic_completed_wallclock_s",
        "action_frame_id",
        "action_wallclock_s",
        "packet_age_s",
        "action_state",
        "action_history",
        "embodiment_id",
    ):
        buffer[key].append(np.array(buffer[key][-1], copy=True))


def snapshot_model_action_history(model) -> dict[str, Any]:
    """Capture action history so semantic retries do not append fake actions."""
    history = getattr(model, "_action_history", None)
    return {
        "history": None if history is None else history.clone(),
        "by_env": {
            key: value.clone()
            for key, value in getattr(model, "_action_history_by_env", {}).items()
        },
        "keys": list(getattr(model, "_current_action_history_keys", [])),
    }


def restore_model_action_history(model, snapshot: dict[str, Any]) -> None:
    """Restore action history after a stale semantic prediction is discarded."""
    history = snapshot["history"]
    model._action_history = None if history is None else history.clone()
    model._action_history_by_env = {
        key: value.clone() for key, value in snapshot["by_env"].items()
    }
    model._current_action_history_keys = list(snapshot["keys"])


def append_normalized_action_history(
    history: np.ndarray, actions: np.ndarray, active: np.ndarray
) -> np.ndarray:
    """Mirror GR00T's rolling normalized-action history update."""
    if history.shape[1] == 0:
        return history
    padded = np.zeros((actions.shape[0], 1, history.shape[-1]), dtype=history.dtype)
    width = min(actions.shape[-1], history.shape[-1])
    padded[:, 0, :width] = actions[:, :width].astype(history.dtype, copy=False)
    updated = np.concatenate((history, padded), axis=1)[:, -history.shape[1] :]
    return np.where(active[:, None, None], updated, history)


@torch.no_grad()
def collect_task(args: argparse.Namespace, model, task_id: int) -> None:
    """Collect exact balanced quotas for one LIBERO-10 task."""
    from rlinf.envs.action_utils import prepare_actions_for_libero

    task_root = args.output / f"task_{task_id:02d}"
    target = {"success": args.successes_per_task, "failure": args.failures_per_task}
    complete, occupied, full = quota_snapshot(task_root, target)
    if full:
        print(f"[task {task_id}] quotas already complete: {complete}", flush=True)
        return

    env = make_env(
        task_id,
        args.seed + task_id * 100_003,
        args.num_envs,
        args.max_decisions * args.action_chunk,
    )
    obs, _ = env.reset()
    language = str(env.task_descriptions[0])
    episode_number = 0
    rngs = [
        np.random.default_rng(args.seed + task_id * 1_000_003 + env_id)
        for env_id in range(args.num_envs)
    ]
    init_ids = [
        int(env.trial_ids[env_id]) if hasattr(env, "trial_ids") else -1
        for env_id in range(args.num_envs)
    ]
    env_ids = [
        args.env_id_offset + task_id * args.env_id_task_stride + env_id
        for env_id in range(args.num_envs)
    ]
    frame_ids = np.zeros(args.num_envs, dtype=np.int64)
    episode_generations = np.zeros(args.num_envs, dtype=np.int64)
    buffers = [
        new_buffer(
            obs_at(obs, env_id), init_ids[env_id], int(rngs[env_id].integers(2**31))
        )
        for env_id in range(args.num_envs)
    ]

    writer_pool = ThreadPoolExecutor(max_workers=args.writer_threads)
    pending_writes: list[Future[Path]] = []

    print(
        f"[task {task_id}] {language!r}; complete={complete}, occupied={occupied}, "
        f"target={target}, source_stream={args.source_stream}, "
        f"parallel_envs={args.num_envs}, writers={args.writer_threads}",
        flush=True,
    )
    while True:
        _, _, full = quota_snapshot(task_root, target)
        if full:
            break
        model_obs = attach_rollout_metadata(
            copy_model_obs(obs),
            env_ids=env_ids,
            frame_ids=frame_ids.tolist(),
            episode_generations=episode_generations.tolist(),
            task_id=task_id,
            trial_ids=init_ids,
        )
        while pending_writes and pending_writes[0].done():
            pending_writes.pop(0).result()

        action_history_snapshot = snapshot_model_action_history(model)
        for boundary_attempt in range(args.boundary_packet_attempts):
            if boundary_attempt:
                restore_model_action_history(model, action_history_snapshot)
            policy_action_t, result = model.predict_action_batch(
                model_obs,
                mode=args.inference_mode,
                return_semantic_features=True,
            )
            policy_action = as_numpy(policy_action_t, np.float32)
            features = unpack_features(
                result,
                action_frame_ids=frame_ids.tolist(),
                episode_generations=episode_generations.tolist(),
            )
            if np.array_equal(features["semantic_source_frame_id"], frame_ids):
                break
        else:
            raise RuntimeError(
                "Semantic server did not produce source-aligned boundary packets"
            )

        executed_action = prepare_actions_for_libero(
            policy_action.copy(), model_type="gr00t_n1d7"
        )
        executed_action = as_numpy(executed_action, np.float32)
        executed_action = apply_action_corruption(
            executed_action,
            np.ones(args.num_envs, dtype=bool),
            rngs,
            args.action_noise_std,
            args.action_dropout_prob,
        )

        chunk_reward = np.zeros((args.num_envs, args.action_chunk), dtype=np.float32)
        chunk_terminated = np.zeros((args.num_envs, args.action_chunk), dtype=bool)
        chunk_truncated = np.zeros((args.num_envs, args.action_chunk), dtype=bool)
        valid_mask = np.zeros((args.num_envs, args.action_chunk), dtype=bool)
        done = np.zeros(args.num_envs, dtype=bool)
        boundary_obs: list[dict[str, np.ndarray] | None] = [None] * args.num_envs
        latest_obs = obs
        delay_count = args.delay_max_frames + 1
        delay_state = np.repeat(features["action_state"][:, None], delay_count, axis=1)
        delay_history = np.repeat(
            features["action_history"][:, None], delay_count, axis=1
        )
        delay_frames = np.repeat(frame_ids[:, None], delay_count, axis=1)
        delay_valid = np.zeros((args.num_envs, delay_count), dtype=bool)
        delay_completion = np.zeros((args.num_envs, delay_count), dtype=bool)
        delay_valid[:, 0] = True
        running_history = features["action_history"].copy()

        for chunk_index in range(args.action_chunk):
            step_action = executed_action[:, chunk_index].copy()
            step_action[done, :6] = 0
            step_action[done, 6] = -1
            next_obs, reward_t, terminated_t, truncated_t, _ = env.step(
                step_action, auto_reset=False
            )
            reward = as_numpy(reward_t, np.float32)
            terminated = as_numpy(terminated_t, bool)
            truncated = as_numpy(truncated_t, bool)
            active = ~done
            frame_ids[active] += 1
            running_history = append_normalized_action_history(
                running_history,
                features["normalized_executed_action"][:, chunk_index],
                active,
            )
            valid_mask[active, chunk_index] = True
            chunk_reward[active, chunk_index] = reward[active]
            chunk_terminated[active, chunk_index] = terminated[active]
            chunk_truncated[active, chunk_index] = truncated[active]
            newly_done = active & (terminated | truncated)
            for env_id in np.flatnonzero(newly_done):
                boundary_obs[env_id] = obs_at(next_obs, int(env_id))
            done |= newly_done
            latest_obs = next_obs
            delay_index = chunk_index + 1
            if delay_index <= args.delay_max_frames:
                current_condition = model.prepare_action_condition_batch(
                    copy_model_obs(next_obs)
                )
                current_state = as_numpy(current_condition["state"], np.float32)
                delay_state[active, delay_index] = current_state[active]
                delay_history[active, delay_index] = running_history[active]
                delay_frames[active, delay_index] = frame_ids[active]
                delay_valid[active, delay_index] = True

        successful_chunk = chunk_terminated.any(axis=1)
        delay_completion = delay_valid & successful_chunk[:, None]

        for env_id in np.flatnonzero(~done):
            boundary_obs[int(env_id)] = obs_at(latest_obs, int(env_id))

        for env_id in range(args.num_envs):
            buffer = buffers[env_id]
            append_feature_row(buffer, features, env_id)
            buffer["delay_action_state"].append(delay_state[env_id])
            buffer["delay_action_history"].append(delay_history[env_id])
            buffer["delay_action_frame_id"].append(delay_frames[env_id])
            buffer["delay_valid_mask"].append(delay_valid[env_id])
            buffer["delay_completion"].append(delay_completion[env_id])
            buffer["policy_action"].append(policy_action[env_id])
            buffer["executed_action"].append(executed_action[env_id])
            buffer["chunk_reward"].append(chunk_reward[env_id])
            buffer["chunk_terminated"].append(chunk_terminated[env_id])
            buffer["chunk_truncated"].append(chunk_truncated[env_id])
            buffer["action_valid_mask"].append(valid_mask[env_id])
            assert boundary_obs[env_id] is not None
            buffer["observations"].append(boundary_obs[env_id])

        done_ids = np.flatnonzero(done)
        if len(done_ids):
            done_env_ids = [env_ids[int(env_id)] for env_id in done_ids]
            done_frame_ids = [int(frame_ids[int(env_id)]) for env_id in done_ids]
            done_generations = [
                int(episode_generations[int(env_id)]) for env_id in done_ids
            ]
            done_trial_ids = [init_ids[int(env_id)] for env_id in done_ids]
            terminal_features = None
            if target["success"] > 0:
                terminal_batch = batch_from_records(
                    [boundary_obs[int(env_id)] for env_id in done_ids], language
                )
                terminal_batch = attach_rollout_metadata(
                    terminal_batch,
                    env_ids=done_env_ids,
                    frame_ids=done_frame_ids,
                    episode_generations=done_generations,
                    task_id=task_id,
                    trial_ids=done_trial_ids,
                )
                old_hard_max_age = getattr(
                    model, "_semantic_fetch_hard_max_age_frames", -1
                )
                if args.semantic_server:
                    model._semantic_fetch_hard_max_age_frames = 0
                try:
                    for _ in range(args.terminal_packet_attempts):
                        _, terminal_result = model.predict_action_batch(
                            terminal_batch,
                            mode="eval",
                            return_semantic_features=True,
                        )
                        terminal_features = unpack_features(
                            terminal_result,
                            action_frame_ids=done_frame_ids,
                            episode_generations=done_generations,
                        )
                        if np.array_equal(
                            terminal_features["semantic_source_frame_id"],
                            np.asarray(done_frame_ids, dtype=np.int64),
                        ):
                            break
                    else:
                        raise RuntimeError(
                            "Semantic server did not produce source-aligned terminal packets"
                        )
                finally:
                    model._semantic_fetch_hard_max_age_frames = old_hard_max_age

            for local_index, env_id_raw in enumerate(done_ids):
                env_id = int(env_id_raw)
                buffer = buffers[env_id]
                success = bool(chunk_terminated[env_id].any())
                if terminal_features is None:
                    append_duplicate_feature_row(buffer)
                    terminal_state = buffer["action_state"][-1]
                    terminal_history = buffer["action_history"][-1]
                    terminal_frame = buffer["action_frame_id"][-1]
                else:
                    append_feature_row(buffer, terminal_features, local_index)
                    terminal_state = terminal_features["action_state"][local_index]
                    terminal_history = terminal_features["action_history"][local_index]
                    terminal_frame = terminal_features["action_frame_id"][local_index]
                buffer["delay_action_state"].append(
                    np.repeat(terminal_state[None], delay_count, axis=0)
                )
                buffer["delay_action_history"].append(
                    np.repeat(terminal_history[None], delay_count, axis=0)
                )
                buffer["delay_action_frame_id"].append(
                    np.repeat(terminal_frame, delay_count)
                )
                terminal_valid = np.zeros(delay_count, dtype=bool)
                if terminal_features is not None:
                    terminal_valid[0] = True
                buffer["delay_valid_mask"].append(terminal_valid)
                terminal_completion = np.zeros(delay_count, dtype=bool)
                if terminal_features is not None:
                    terminal_completion[0] = success
                buffer["delay_completion"].append(terminal_completion)
                outcome = "success" if success else "failure"
                episode_number += 1
                reservation = reserve_outcome_slot(
                    task_root,
                    outcome,
                    target[outcome],
                    args.trajectory_prefix,
                )
                if reservation is not None:
                    outcome_index, reservation_path, reserved_count = reservation
                    future = writer_pool.submit(
                        save_episode,
                        output=args.output,
                        task_id=task_id,
                        language=language,
                        checkpoint=args.checkpoint,
                        inference_mode=args.inference_mode,
                        action_chunk=args.action_chunk,
                        gamma=args.gamma,
                        failure_terminal_reward=args.failure_terminal_reward,
                        action_noise_std=args.action_noise_std,
                        action_dropout_prob=args.action_dropout_prob,
                        buffer=buffer,
                        success=success,
                        outcome_index=outcome_index,
                        trajectory_prefix=args.trajectory_prefix,
                        source_stream=args.source_stream,
                        reservation_path=reservation_path,
                    )
                    pending_writes.append(future)
                    print(
                        f"[task {task_id}] reserved {outcome}: "
                        f"{reserved_count}/{target[outcome]} "
                        f"(source={args.source_stream}, attempts={episode_number})",
                        flush=True,
                    )

                    if len(pending_writes) >= args.max_pending_writes:
                        pending_writes.pop(0).result()
            reset_obs, _ = env.reset(env_idx=done_ids)
            obs = reset_obs
            for env_id_raw in done_ids:
                env_id = int(env_id_raw)
                episode_generations[env_id] += 1
                frame_ids[env_id] = 0
                init_state_id = (
                    int(env.trial_ids[env_id]) if hasattr(env, "trial_ids") else -1
                )
                init_ids[env_id] = init_state_id
                buffers[env_id] = new_buffer(
                    obs_at(obs, env_id),
                    init_state_id,
                    int(rngs[env_id].integers(2**31)),
                )
        else:
            obs = latest_obs

    if hasattr(env, "close"):
        env.close()
    for future in pending_writes:
        future.result()
    writer_pool.shutdown(wait=True)

    complete, _, _ = quota_snapshot(task_root, target)
    print(f"[task {task_id}] complete: {complete}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect balanced GR00T-N1.7 shared-semantic RM trajectories"
    )
    parser.add_argument("--checkpoint", required=True, help="GR00T N1.7 checkpoint")
    parser.add_argument("--backbone-model-path", default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", default="0-9")
    parser.add_argument("--successes-per-task", type=int, default=750)
    parser.add_argument("--failures-per-task", type=int, default=750)
    parser.add_argument("--trajectory-prefix", default="")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--action-chunk", type=int, default=16)
    parser.add_argument("--max-decisions", type=int, default=30)
    parser.add_argument("--inference-mode", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--source-stream", choices=("success", "failure", "mixed"), default="mixed"
    )
    parser.add_argument("--writer-threads", type=int, default=4)
    parser.add_argument("--max-pending-writes", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--failure-terminal-reward", type=float, default=-300.0)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--action-dropout-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-server", action="store_true")
    parser.add_argument("--semantic-server-host", default="127.0.0.1")
    parser.add_argument("--semantic-server-port", type=int, default=6677)
    parser.add_argument("--semantic-server-publish-port", type=int, default=None)
    parser.add_argument("--semantic-timeout-ms", type=float, default=30000.0)
    parser.add_argument("--semantic-target-age-frames", type=int, default=-1)
    parser.add_argument("--semantic-hard-max-age-frames", type=int, default=6)
    parser.add_argument("--semantic-tokens", type=int, default=160)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--delay-max-frames", type=int, default=6)
    parser.add_argument("--boundary-packet-attempts", type=int, default=10)
    parser.add_argument("--terminal-packet-attempts", type=int, default=5)
    parser.add_argument("--env-id-offset", type=int, default=0)
    parser.add_argument("--env-id-task-stride", type=int, default=10000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.successes_per_task < 0 or args.failures_per_task < 0:
        raise ValueError("Trajectory quotas must be non-negative")
    if not 0 <= args.action_dropout_prob <= 1:
        raise ValueError("--action-dropout-prob must be in [0, 1]")
    if args.action_chunk <= 0 or args.max_decisions <= 0:
        raise ValueError("Action chunk and max decisions must be positive")
    if args.semantic_tokens <= 0 or args.control_hz <= 0:
        raise ValueError("semantic-tokens and control-hz must be positive")
    if not 0 <= args.delay_max_frames < args.action_chunk:
        raise ValueError("delay-max-frames must be in [0, action-chunk)")
    if args.boundary_packet_attempts <= 0:
        raise ValueError("boundary-packet-attempts must be positive")
    if args.terminal_packet_attempts <= 0:
        raise ValueError("terminal-packet-attempts must be positive")
    if args.env_id_task_stride < args.num_envs:
        raise ValueError("env-id-task-stride must be at least num-envs")
    if args.writer_threads <= 0:
        raise ValueError("--writer-threads must be positive")
    if args.max_pending_writes < args.writer_threads:
        raise ValueError("--max-pending-writes must be >= --writer-threads")
    if not all(
        character.isalnum() or character in "_-" for character in args.trajectory_prefix
    ):
        raise ValueError(
            "--trajectory-prefix may contain only letters, digits, _ and -"
        )

    args.output = args.output.resolve()
    if args.semantic_server_publish_port is None:
        args.semantic_server_publish_port = args.semantic_server_port + 1
    args.output.mkdir(parents=True, exist_ok=True)
    write_schema(args.output)
    tasks = parse_tasks(args.tasks)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, _ = build_model(args)
    print(
        f"[collector] schema={SCHEMA_VERSION}; tasks={tasks}; output={args.output}; "
        f"semantic_server={args.semantic_server}",
        flush=True,
    )
    for task_id in tasks:
        collect_task(args, model, task_id)
    print("[collector] all requested tasks complete", flush=True)


if __name__ == "__main__":
    main()
