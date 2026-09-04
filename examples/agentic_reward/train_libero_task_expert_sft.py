# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train one CNN policy from LIBERO demonstrations, without policy replay."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from PIL import Image

from rlinf.models.embodiment.cnn_policy import get_model

MANAGER_STAGE_NAMES = tuple(
    f"{object_name}:{stage}"
    for object_name in ("alphabet_soup_1", "tomato_sauce_1")
    for stage in (
        "approach",
        "descend",
        "close",
        "lift",
        "transport",
        "lower",
        "release",
        "retreat",
    )
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root")
    source.add_argument(
        "--scripted-demo-dir",
        help=(
            "Current-reset demonstrations exported by "
            "collect_current_env_scripted_teacher.py."
        ),
    )
    source.add_argument(
        "--gpt-replay-dir",
        nargs="+",
        help=(
            "Actor-visible replay panels filtered by exact per-frame physical "
            "reward and sampled uniformly across reset scenes."
        ),
    )
    parser.add_argument(
        "--additional-scripted-demo-dir",
        action="append",
        default=[],
        help=(
            "Additional current-reset demo dataset to aggregate with either "
            "source; may be repeated to explicitly oversample it."
        ),
    )
    parser.add_argument("--resnet-dir", required=True)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Optional critic-free CNN checkpoint used only as a warmup initialization.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--dataset-task-index",
        type=int,
        help=(
            "Optional LeRobot task_index when its ordering differs from the "
            "LIBERO benchmark task id. The checkpoint lineage keeps --task-id."
        ),
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--encoder-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Keep pretrained visual encoders and normalization buffers frozen.",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help=(
            "Allow low-rate teacher-warmup updates to the pretrained ResNet "
            "backbone; the default keeps its historical frozen behavior."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--object-filter",
        choices=("alphabet_soup_1", "tomato_sauce_1"),
        help=(
            "Train one manager-routed object lesson. Routing labels select "
            "samples but are never passed to the Actor."
        ),
    )
    parser.add_argument(
        "--manager-stage-conditioning",
        action="store_true",
        help="Append a manager-provided 16-way stage one-hot to proprioception.",
    )
    parser.add_argument(
        "--manager-uniform-phase-conditioning",
        action="store_true",
        help=(
            "Derive the shared manager stage from normalized demonstration time. "
            "This keeps one shared Actor while disambiguating long sequential tasks."
        ),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument(
        "--random-shift-pad",
        type=int,
        default=0,
        help=(
            "Replicate-pad each camera image and apply an independent random "
            "spatial crop during teacher warmup; zero disables augmentation."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _decode_png(value: dict[str, object]) -> np.ndarray:
    data = value.get("bytes")
    if not isinstance(data, bytes):
        raise ValueError("LIBERO image row does not contain embedded PNG bytes.")
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _episode_task_ids(root: Path) -> dict[int, int]:
    task_names = {
        str(item["task"]): int(item["task_index"])
        for item in map(json.loads, (root / "meta" / "tasks.jsonl").open())
    }
    return {
        int(item["episode_index"]): task_names[str(item["tasks"][0])]
        for item in map(json.loads, (root / "meta" / "episodes.jsonl").open())
    }


def _load_task_dataset(
    root: Path,
    *,
    task_id: int,
    chunk_size: int,
    dataset_task_index: int | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, object]]:
    resolved_dataset_task_index = (
        task_id if dataset_task_index is None else dataset_task_index
    )
    episode_ids = tuple(
        episode_id
        for episode_id, episode_task in sorted(_episode_task_ids(root).items())
        if episode_task == resolved_dataset_task_index
    )
    if not episode_ids:
        raise ValueError(
            "No expert episodes found for "
            f"dataset_task_index={resolved_dataset_task_index}."
        )
    main_images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    source_hashes: dict[str, str] = {}
    episode_lengths: list[int] = []
    for episode_id in episode_ids:
        path = (
            root
            / "data"
            / f"chunk-{episode_id // 1000:03d}"
            / (f"episode_{episode_id:06d}.parquet")
        )
        table = pq.read_table(
            path,
            columns=["image", "wrist_image", "state", "actions", "task_index"],
        )
        values = table.to_pydict()
        observed_task_ids = {int(value) for value in values["task_index"]}
        if observed_task_ids != {resolved_dataset_task_index}:
            raise ValueError(f"Task identity mismatch in {path}: {observed_task_ids}")
        actions = np.asarray(values["actions"], dtype=np.float32)
        episode_lengths.append(len(actions))
        for start in range(0, len(actions) - chunk_size + 1):
            main_images.append(_decode_png(values["image"][start]))
            wrist_images.append(_decode_png(values["wrist_image"][start]))
            states.append(np.asarray(values["state"][start], dtype=np.float32))
            chunks.append(actions[start : start + chunk_size])
        source_hashes[str(path.resolve())] = _sha256(path)
    observations = {
        "main_images": torch.from_numpy(np.stack(main_images)),
        "extra_view_images": torch.from_numpy(np.stack(wrist_images)).unsqueeze(1),
        "states": torch.from_numpy(np.stack(states)),
    }
    actions = torch.from_numpy(np.stack(chunks)).clamp(-1.0, 1.0)
    manifest = {
        "dataset_root": str(root.resolve()),
        "task_id": task_id,
        "dataset_task_index": resolved_dataset_task_index,
        "episode_ids": episode_ids,
        "episode_lengths": episode_lengths,
        "source_file_sha256s": source_hashes,
        "sample_count": int(actions.shape[0]),
        "chunk_size": chunk_size,
        "actor_visible_fields": ["image", "wrist_image", "state", "actions"],
        "excluded_fields": ["return", "reward", "prompt"],
    }
    return observations, actions, manifest


def _load_scripted_dataset(
    root: Path,
    *,
    task_id: int,
    chunk_size: int,
    object_filter: str | None = None,
    manager_stage_conditioning: bool = False,
    manager_uniform_phase_conditioning: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, object]]:
    """Load current-reset scripted demos through their Actor-only export."""
    audit_path = root / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Scripted teacher audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_kind = audit.get("kind")
    allowed_kinds = {
        "current_reset_aligned_expert_replay_audit",
        "privileged_current_reset_scripted_teacher_audit",
        "privileged_current_reset_dagger_audit",
    }
    if audit_kind not in allowed_kinds:
        raise ValueError("Scripted demonstration audit kind is invalid.")
    is_dagger = audit_kind == "privileged_current_reset_dagger_audit"
    is_aligned_expert = audit_kind == "current_reset_aligned_expert_replay_audit"
    if int(audit.get("task_id", -1)) != task_id:
        raise ValueError("Scripted demonstration task does not match --task-id.")
    records = audit.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Scripted demonstration audit has no records.")
    main_parts: list[np.ndarray] = []
    wrist_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    stage_parts: list[np.ndarray] = []
    source_hashes: dict[str, str] = {str(audit_path.resolve()): _sha256(audit_path)}
    reset_ids: list[int] = []
    trajectory_lengths: list[int] = []
    allowed_arrays = {
        "main_images",
        "wrist_images",
        "states",
        "actions",
        "reset_state_id",
    }
    for record in records:
        if not record.get("exported"):
            continue
        if is_dagger:
            if not record.get("teacher_labels_valid"):
                continue
        elif not record.get("success_at_end"):
            continue
        payload_name = record.get("payload")
        if not isinstance(payload_name, str):
            raise ValueError("Successful scripted record has no payload.")
        path = (root / payload_name).resolve()
        expected_hash = str(record.get("payload_sha256", ""))
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"Scripted demonstration hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != allowed_arrays:
                raise ValueError(f"Scripted demonstration schema mismatch: {path}")
            main = np.array(payload["main_images"], copy=True)
            wrist = np.array(payload["wrist_images"], copy=True)
            states = np.asarray(payload["states"], dtype=np.float32)
            trajectory_actions = np.asarray(payload["actions"], dtype=np.float32)
            reset_id = int(payload["reset_state_id"])
        stage_counts = record.get("stage_step_counts")
        if manager_stage_conditioning or object_filter is not None:
            if not isinstance(stage_counts, dict):
                raise ValueError(f"Scripted stage routing metadata is missing: {path}")
        stage_ids = None
        if manager_stage_conditioning:
            if manager_uniform_phase_conditioning:
                stage_ids = np.minimum(
                    np.arange(len(trajectory_actions), dtype=np.int64)
                    * len(MANAGER_STAGE_NAMES)
                    // len(trajectory_actions),
                    len(MANAGER_STAGE_NAMES) - 1,
                )
            else:
                stage_ids = np.concatenate(
                    [
                        np.full(
                            int(stage_counts.get(stage, 0)), index, dtype=np.int64
                        )
                        for index, stage in enumerate(MANAGER_STAGE_NAMES)
                    ]
                )
                trailing = len(trajectory_actions) - len(stage_ids)
                if trailing < 0:
                    raise ValueError(
                        f"Scripted stage counts exceed payload length: {path}"
                    )
                if trailing:
                    stage_ids = np.concatenate(
                        [stage_ids, np.full(trailing, 15, dtype=np.int64)]
                    )
        if object_filter is not None:
            soup_length = sum(
                int(count)
                for stage, count in stage_counts.items()
                if str(stage).startswith("alphabet_soup_1:")
            )
            tomato_length = sum(
                int(count)
                for stage, count in stage_counts.items()
                if str(stage).startswith("tomato_sauce_1:")
            )
            start = 0 if object_filter == "alphabet_soup_1" else soup_length
            selected_length = (
                soup_length if object_filter == "alphabet_soup_1" else tomato_length
            )
            stop = start + selected_length
            if selected_length < chunk_size or stop > len(trajectory_actions):
                raise ValueError(f"Scripted stage routing metadata is invalid: {path}")
            main = main[start:stop]
            wrist = wrist[start:stop]
            states = states[start:stop]
            trajectory_actions = trajectory_actions[start:stop]
            if stage_ids is not None:
                stage_ids = stage_ids[start:stop]
        length = len(trajectory_actions)
        if (
            main.dtype != np.uint8
            or wrist.dtype != np.uint8
            or main.ndim != 4
            or wrist.shape != main.shape
            or states.ndim != 2
            or trajectory_actions.shape != (length, 7)
            or not (len(main) == len(wrist) == len(states) == length)
            or length < chunk_size
        ):
            raise ValueError(f"Scripted demonstration tensor shape is invalid: {path}")
        sample_count = length - chunk_size + 1
        main_parts.append(main[:sample_count])
        wrist_parts.append(wrist[:sample_count])
        state_parts.append(states[:sample_count])
        if stage_ids is not None:
            stage_parts.append(stage_ids[:sample_count])
        action_parts.append(
            np.stack(
                [
                    trajectory_actions[start : start + chunk_size]
                    for start in range(sample_count)
                ]
            )
        )
        source_hashes[str(path)] = observed_hash
        reset_ids.append(reset_id)
        trajectory_lengths.append(length)
    if not reset_ids:
        raise ValueError(
            "No final-state successful scripted demonstrations were exported."
        )
    if len(reset_ids) != len(set(reset_ids)):
        raise ValueError("Scripted demonstration reset IDs must be unique.")
    observations = {
        "main_images": torch.from_numpy(np.concatenate(main_parts)),
        "extra_view_images": torch.from_numpy(np.concatenate(wrist_parts)).unsqueeze(1),
        "states": torch.from_numpy(np.concatenate(state_parts)),
    }
    if manager_stage_conditioning:
        all_stage_ids = torch.from_numpy(np.concatenate(stage_parts))
        stage_one_hot = torch.nn.functional.one_hot(
            all_stage_ids, num_classes=len(MANAGER_STAGE_NAMES)
        ).to(observations["states"].dtype)
        observations["states"] = torch.cat(
            [observations["states"], stage_one_hot], dim=1
        )
    actions = torch.from_numpy(np.concatenate(action_parts)).clamp(-1.0, 1.0)
    manifest = {
        "dataset_root": str(root.resolve()),
        "dataset_kind": (
            "current_reset_dagger"
            if is_dagger
            else (
                "current_reset_aligned_expert"
                if is_aligned_expert
                else "current_reset_scripted_teacher"
            )
        ),
        "task_id": task_id,
        "reset_ids": sorted(reset_ids),
        "trajectory_lengths": trajectory_lengths,
        "source_file_sha256s": source_hashes,
        "sample_count": int(actions.shape[0]),
        "chunk_size": chunk_size,
        "manager_routed_object_filter": object_filter,
        "manager_stage_conditioning": manager_stage_conditioning,
        "manager_stage_names": MANAGER_STAGE_NAMES,
        "manager_stage_max_steps": (
            {
                stage: max(
                    1,
                    max(math.ceil(length / len(MANAGER_STAGE_NAMES)) for length in trajectory_lengths),
                )
                for stage in MANAGER_STAGE_NAMES
            }
            if manager_uniform_phase_conditioning
            else {
                stage: max(
                    int(record["stage_step_counts"].get(stage, 0))
                    for record in records
                    if isinstance(record.get("stage_step_counts"), dict)
                )
                for stage in MANAGER_STAGE_NAMES
            }
        ),
        "manager_uniform_phase_conditioning": manager_uniform_phase_conditioning,
        "actor_visible_fields": [
            "main_images",
            "wrist_images",
            "states",
            "actions",
        ],
        "privileged_teacher_fields_excluded_from_model_input": audit[
            "privileged_teacher_fields_excluded_from_export"
        ],
        "simulator_state_used_to_generate_teacher_actions": bool(
            audit.get("simulator_state_used_to_generate_teacher_actions", True)
        ),
        "simulator_success_used_only_as_teacher_export_gate": not is_dagger,
        "dagger_policy_induced_states": is_dagger,
        "reward_labels_used_by_sft": False,
    }
    return observations, actions, manifest


def _sample_weights(actions: torch.Tensor) -> torch.Tensor:
    """Upweight rare motion and gripper-transition chunks over static frames."""
    translation = actions[..., :3].norm(dim=-1).amax(dim=1)
    rotation = actions[..., 3:6].norm(dim=-1).amax(dim=1)
    gripper = actions[..., 6]
    gripper_transition = (gripper[:, 1:] != gripper[:, :-1]).any(dim=1).float()
    weights = 1.0 + 6.0 * (translation / 0.02).clamp(max=2.0)
    weights += 2.0 * (rotation / 0.05).clamp(max=2.0)
    weights += 8.0 * gripper_transition
    return weights / weights.mean()


def _load_physical_reward_replay_dataset(
    roots: list[Path],
    *,
    task_id: int,
    chunk_size: int,
    source_entity: str = "white_yellow_mug_1",
    target_entity: str = "microwave_1",
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    """Load Actor tensors from trajectories retaining exact physical success."""
    from rlinf.agents.agentic_reward.gpt_replay import load_gpt_observation_replay
    from rlinf.agents.agentic_reward.raw_physical_dense_reward import (
        raw_physical_reward_timelines,
    )
    observation_parts: dict[str, list[torch.Tensor]] = {}
    action_parts: list[torch.Tensor] = []
    scene_parts: list[torch.Tensor] = []
    selected: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    actor_keys: set[str] | None = None
    for root in roots:
        root = root.resolve()
        manifest_path = root / "manifest.json"
        for trajectory in load_gpt_observation_replay(root):
            if trajectory.provenance.task_id != task_id:
                raise ValueError("Physical replay task identity does not match.")
            if trajectory.chunk_size != chunk_size:
                raise ValueError("Physical replay action chunk size does not match.")
            physical = (
                trajectory.control_object_poses
                if trajectory.control_object_poses
                else trajectory.object_poses
            )
            rewards = raw_physical_reward_timelines(
                physical,
                source_entity=source_entity,
                target_entity=target_entity,
            )["libero_binary"]
            record = {
                "replay_manifest": str(manifest_path.resolve()),
                "replay_manifest_sha256": _sha256(manifest_path),
                "trajectory_id": trajectory.trajectory_id,
                "reset_id": int(trajectory.reset_id),
                "exact_libero_binary_max": float(rewards.max()),
                "exact_libero_binary_final": float(rewards[-1]),
            }
            if float(rewards.max()) != 1.0 or float(rewards[-1]) != 1.0:
                rejected.append(record)
                continue
            keys = set(trajectory.actor.observations)
            if actor_keys is None:
                actor_keys = keys
            elif keys != actor_keys:
                raise ValueError("Physical replay Actor observation schemas differ.")
            for name, value in trajectory.actor.observations.items():
                observation_parts.setdefault(name, []).append(value)
            action_parts.append(trajectory.actor.actions.float())
            scene_parts.append(
                torch.full(
                    (trajectory.num_chunks,),
                    trajectory.reset_id,
                    dtype=torch.long,
                )
            )
            record["sample_count"] = trajectory.num_chunks
            selected.append(record)
    if not selected:
        raise ValueError("No replay trajectory retained exact physical success.")
    scene_ids = torch.cat(scene_parts)
    covered_reset_ids = sorted(torch.unique(scene_ids).tolist())
    observations = {
        name: torch.cat(parts) for name, parts in observation_parts.items()
    }
    actions = torch.cat(action_parts).clamp(-1.0, 1.0)
    manifest = {
        "dataset_kind": "physical_reward_filtered_replay",
        "dataset_root": [str(root.resolve()) for root in roots],
        "task_id": task_id,
        "chunk_size": chunk_size,
        "sample_count": int(actions.shape[0]),
        "selected_trajectory_count": len(selected),
        "rejected_trajectory_count": len(rejected),
        "covered_reset_ids": covered_reset_ids,
        "selected_trajectories": selected,
        "rejected_trajectories": rejected,
        "selection_signal": "exact per-frame libero_binary max=final=1",
        "physical_reward_used_for_trajectory_selection": True,
        "physical_reward_used_as_actor_loss": False,
        "scene_balanced_sampling": True,
        "actor_visible_fields": sorted(observations),
        "excluded_actor_fields": [
            "object_poses",
            "physical_reward",
            "simulator_success",
            "environment_reward",
            "reset_id",
        ],
        "reward_labels_used_by_sft": False,
    }
    return observations, actions, scene_ids, manifest


def _balance_weights_by_scene(
    weights: torch.Tensor, scene_ids: torch.Tensor
) -> torch.Tensor:
    """Normalize motion weights so every physical reset has equal mass."""
    if weights.shape != scene_ids.shape:
        raise ValueError("Scene IDs must align with SFT sample weights.")
    balanced = weights.clone()
    for reset_id in torch.unique(scene_ids):
        group = scene_ids == reset_id
        balanced[group] /= balanced[group].sum().clamp_min(1.0e-8)
    return balanced / balanced.mean()


def _action_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a balanced physical-action loss for tanh-squashed chunks."""
    prediction = torch.tanh(prediction)
    element_loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.02)
    dimension_weights = target.new_tensor((4.0, 4.0, 4.0, 1.5, 1.5, 1.5, 2.0))
    regression = (element_loss * dimension_weights).mean()
    gripper_target = target[..., 6].gt(0.0).to(prediction.dtype)
    gripper_loss = F.binary_cross_entropy_with_logits(
        prediction[..., 6] * 4.0,
        gripper_target,
    )
    return regression + 0.25 * gripper_loss


def _random_shift_images(
    images: torch.Tensor,
    *,
    pad: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply deterministic per-image replicate-pad random crops on CPU."""
    if pad < 0:
        raise ValueError("Random-shift padding must be non-negative.")
    if pad == 0:
        return images
    if images.ndim < 4 or images.shape[-1] not in (1, 3, 4):
        raise ValueError("Images must be channel-last tensors with rank >= 4.")
    leading_shape = images.shape[:-3]
    height, width, channels = images.shape[-3:]
    flattened = images.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
    padded = F.pad(flattened, (pad, pad, pad, pad), mode="replicate")
    count = flattened.shape[0]
    offsets = torch.randint(
        0,
        2 * pad + 1,
        (count, 2),
        generator=generator,
        device="cpu",
    )
    rows = offsets[:, :1] + torch.arange(height).unsqueeze(0)
    columns = offsets[:, 1:] + torch.arange(width).unsqueeze(0)
    cropped_rows = torch.gather(
        padded,
        2,
        rows[:, None, :, None].expand(-1, channels, -1, padded.shape[3]),
    )
    cropped = torch.gather(
        cropped_rows,
        3,
        columns[:, None, None, :].expand(-1, channels, height, -1),
    )
    return cropped.permute(0, 2, 3, 1).reshape(*leading_shape, height, width, channels)


def _resize_actor_images(
    images: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Resize channel-last actor images while preserving their 0-255 range."""
    leading = images.shape[:-3]
    height, width, channels = images.shape[-3:]
    if (height, width) == size:
        return images
    flat = images.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
    resized = F.interpolate(
        flat.float(), size=size, mode="bilinear", align_corners=False, antialias=True
    )
    return resized.permute(0, 2, 3, 1).reshape(*leading, *size, channels)


def _save_checkpoint(
    output: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    manifest: dict[str, object],
    metrics: list[dict[str, float]],
) -> Path:
    destination = output / "checkpoints" / f"step_{step:06d}"
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    actor_state = {
        name: value for name, value in model.state_dict().items() if "q_head" not in name
    }
    torch.save(actor_state, temporary / "model.pt")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    metadata = {
        "schema_version": 1,
        "origin": {
            "current_reset_scripted_teacher": "current_env_scripted_sft_v1",
            "current_reset_dagger": "current_env_dagger_sft_v1",
            "current_reset_dagger_aggregate": "current_env_dagger_sft_v1",
            "current_reset_aligned_expert": "current_env_aligned_expert_sft_v1",
            "current_reset_aligned_expert_aggregate": (
                "current_env_aligned_expert_sft_v1"
            ),
            "offline_expert_plus_current_reset_demo": (
                "current_env_aligned_expert_sft_v1"
            ),
            "offline_expert_plus_current_reset_dagger": (
                "current_env_dagger_sft_v1"
            ),
        }.get(str(manifest.get("dataset_kind")), "expert_sft_v2"),
        "policy_start": "random_action_head_with_generic_resnet10_encoder",
        "policy_replay_used": False,
        "reward_labels_used": False,
        "simulator_state_used_by_actor": False,
        "simulator_state_used_to_generate_teacher_actions": bool(
            manifest.get("simulator_state_used_to_generate_teacher_actions", False)
        ),
        "task_id": args.task_id,
        "seed": args.seed,
        "step": step,
        "head_lr": args.head_lr,
        "encoder_lr": args.encoder_lr,
        "encoder_frozen": args.freeze_encoder,
        "backbone_frozen": not args.unfreeze_backbone,
        "weight_decay": args.weight_decay,
        "chunk_size": args.chunk_size,
        "manager_routed_object_filter": args.object_filter,
        "manager_stage_conditioning": args.manager_stage_conditioning,
        "random_shift_pad": args.random_shift_pad,
        "add_value_head": False,
        "q_head_exported": False,
        "training_manifest": manifest,
        "latest_metrics": metrics[-1],
    }
    if args.initial_checkpoint is not None:
        parent = args.initial_checkpoint.resolve()
        metadata["policy_start"] = "continued_teacher_warmup_from_checkpoint"
        metadata["initial_checkpoint"] = str(parent)
        metadata["initial_checkpoint_model_sha256"] = _sha256(parent / "model.pt")
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)
    return destination


def main() -> None:
    args = _parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("Training budgets must be positive.")
    if args.random_shift_pad < 0:
        raise ValueError("--random-shift-pad must be non-negative.")
    if (
        args.manager_uniform_phase_conditioning
        and not args.manager_stage_conditioning
    ):
        raise ValueError(
            "--manager-uniform-phase-conditioning requires "
            "--manager-stage-conditioning."
        )
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    replay_scene_ids = None
    if args.gpt_replay_dir is not None:
        if args.additional_scripted_demo_dir:
            raise ValueError(
                "--gpt-replay-dir cannot mix unfiltered scripted sources."
            )
        observations, actions, replay_scene_ids, manifest = (
            _load_physical_reward_replay_dataset(
                [Path(root) for root in args.gpt_replay_dir],
                task_id=args.task_id,
                chunk_size=args.chunk_size,
            )
        )
    elif args.scripted_demo_dir is not None:
        roots = [args.scripted_demo_dir, *args.additional_scripted_demo_dir]
        loaded = [
            _load_scripted_dataset(
                Path(root).resolve(),
                task_id=args.task_id,
                chunk_size=args.chunk_size,
                object_filter=args.object_filter,
                manager_stage_conditioning=args.manager_stage_conditioning,
                manager_uniform_phase_conditioning=(
                    args.manager_uniform_phase_conditioning
                ),
            )
            for root in roots
        ]
        observations = {
            key: torch.cat([item[0][key] for item in loaded]) for key in loaded[0][0]
        }
        actions = torch.cat([item[1] for item in loaded])
        manifest = loaded[0][2]
        if len(loaded) > 1:
            aligned_expert_only = all(
                item[2].get("dataset_kind") == "current_reset_aligned_expert"
                for item in loaded
            )
            manifest = {
                **manifest,
                "dataset_root": [item[2]["dataset_root"] for item in loaded],
                "dataset_kind": (
                    "current_reset_aligned_expert_aggregate"
                    if aligned_expert_only
                    else "current_reset_dagger_aggregate"
                ),
                "component_manifests": [item[2] for item in loaded],
                "sample_count": int(actions.shape[0]),
                "dagger_policy_induced_states": not aligned_expert_only,
            }
    else:
        if args.object_filter is not None or args.manager_stage_conditioning:
            raise ValueError(
                "Manager routing options are supported only for scripted demos."
            )
        base = _load_task_dataset(
            Path(args.dataset_root).resolve(),
            task_id=args.task_id,
            chunk_size=args.chunk_size,
            dataset_task_index=args.dataset_task_index,
        )
        if args.additional_scripted_demo_dir:
            loaded = [
                base,
                *[
                    _load_scripted_dataset(
                        Path(root).resolve(),
                        task_id=args.task_id,
                        chunk_size=args.chunk_size,
                    )
                    for root in args.additional_scripted_demo_dir
                ],
            ]
            observations = {
                key: torch.cat([item[0][key] for item in loaded])
                for key in loaded[0][0]
            }
            actions = torch.cat([item[1] for item in loaded])
            has_dagger = any(
                item[2].get("dataset_kind") == "current_reset_dagger"
                for item in loaded[1:]
            )
            manifest = {
                **base[2],
                "dataset_root": [item[2]["dataset_root"] for item in loaded],
                "dataset_kind": (
                    "offline_expert_plus_current_reset_dagger"
                    if has_dagger
                    else "offline_expert_plus_current_reset_demo"
                ),
                "component_manifests": [item[2] for item in loaded],
                "sample_count": int(actions.shape[0]),
                "current_reset_demo_source_count": len(loaded) - 1,
                "dagger_policy_induced_states": has_dagger,
                "reward_labels_used_by_sft": False,
            }
        else:
            observations, actions, manifest = base
    weights = _sample_weights(actions)
    if replay_scene_ids is not None:
        weights = _balance_weights_by_scene(weights, replay_scene_ids)

    repo_root = Path(__file__).resolve().parents[2]
    embodied_root = repo_root / "examples" / "embodiment"
    os.environ["EMBODIED_PATH"] = str(embodied_root)
    resnet_root = Path(args.resnet_dir).resolve()
    candidates = tuple(sorted(resnet_root.glob("*.pt")))
    if len(candidates) != 1:
        raise ValueError("--resnet-dir must contain exactly one .pt checkpoint.")
    os.environ["RLINF_RESNET10_DIR"] = str(resnet_root)
    with initialize_config_dir(
        config_dir=str(embodied_root / "config"),
        version_base="1.1",
    ):
        config = compose(config_name="libero_spatial_task9_enpire_sac")
    config.actor.model.model_path = str(resnet_root)
    with open_dict(config.actor.model.encoder_config):
        config.actor.model.encoder_config.use_pretrain = True
        config.actor.model.encoder_config.ckpt_name = candidates[0].name
        config.actor.model.encoder_config.freeze_backbone = not args.unfreeze_backbone
    config.actor.model.add_value_head = False
    config.actor.model.num_action_chunks = args.chunk_size
    if args.manager_stage_conditioning:
        config.actor.model.state_dim += len(MANAGER_STAGE_NAMES)
    model = get_model(config.actor.model, torch.float32).to(args.device)
    if args.initial_checkpoint is not None:
        parent = args.initial_checkpoint.resolve()
        parent_model = parent / "model.pt"
        parent_metadata = parent / "metadata.json"
        if not parent_model.is_file() or not parent_metadata.is_file():
            raise FileNotFoundError(
                "--initial-checkpoint must contain model.pt and metadata.json."
            )
        lineage = json.loads(parent_metadata.read_text(encoding="utf-8"))
        if bool(lineage.get("add_value_head", True)):
            raise ValueError("Warmup initialization must be critic-free.")
        if int(lineage.get("task_id", -1)) != args.task_id:
            raise ValueError("Warmup initialization task does not match --task-id.")
        parent_state = torch.load(parent_model, map_location="cpu", weights_only=True)
        if args.manager_stage_conditioning and not lineage.get(
            "manager_stage_conditioning", False
        ):
            expanded_state = model.state_dict()
            for name, value in parent_state.items():
                target = expanded_state[name]
                if target.shape == value.shape:
                    expanded_state[name] = value
                elif (
                    target.ndim == 2
                    and value.ndim == 2
                    and target.shape[0] == value.shape[0]
                    and target.shape[1]
                    == value.shape[1] + len(MANAGER_STAGE_NAMES)
                ):
                    expanded = target.new_zeros(target.shape)
                    expanded[:, : value.shape[1]] = value
                    expanded_state[name] = expanded
                else:
                    raise ValueError(
                        f"Cannot expand warmup parameter {name}: "
                        f"{tuple(value.shape)} -> {tuple(target.shape)}."
                    )
            model.load_state_dict(expanded_state, strict=True)
        else:
            model.load_state_dict(parent_state, strict=True)
    named_parameters = dict(model.named_parameters())
    encoder_parameters = [
        parameter
        for name, parameter in named_parameters.items()
        if name.startswith("encoders.")
    ]
    head_parameters = [
        parameter
        for name, parameter in named_parameters.items()
        if not name.startswith("encoders.") and "actor_logstd" not in name
    ]
    model.actor_logstd.requires_grad_(False)
    if args.freeze_encoder:
        for parameter in encoder_parameters:
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ),
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=min(args.encoder_lr, args.head_lr) * 0.05,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    augmentation_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    metric_history: list[dict[str, float]] = []
    start_time = time.monotonic()
    model.train()
    if args.freeze_encoder:
        model.encoders.eval()
    for step in range(1, args.steps + 1):
        indices = torch.multinomial(
            weights,
            args.batch_size,
            replacement=True,
            generator=generator,
        )
        batch_observations = {
            key: value[indices] for key, value in observations.items()
        }
        for image_key in ("main_images", "extra_view_images"):
            batch_observations[image_key] = _random_shift_images(
                batch_observations[image_key],
                pad=args.random_shift_pad,
                generator=augmentation_generator,
            )
            batch_observations[image_key] = _resize_actor_images(
                batch_observations[image_key],
                size=tuple(config.actor.model.image_size[-2:]),
            )
        batch_observations = {
            key: value.to(args.device, non_blocking=True)
            for key, value in batch_observations.items()
        }
        target = actions[indices].to(args.device, non_blocking=True)
        processed = model.preprocess_env_obs(batch_observations)
        _, _, action_mean, _ = model._actor_forward_from_processed_tensors(
            main_images=processed["main_images"],
            states=processed["states"],
            extra_view_images=processed["extra_view_images"],
        )
        prediction = action_mean.reshape(-1, args.chunk_size, 7)
        loss = _action_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            with torch.no_grad():
                action_mae = (torch.tanh(prediction) - target).abs().mean()
                gripper_accuracy = (
                    (torch.tanh(prediction[..., 6]).gt(0.0) == target[..., 6].gt(0.0))
                    .float()
                    .mean()
                )
            record = {
                "step": float(step),
                "loss": float(loss.item()),
                "action_mae": float(action_mae.item()),
                "gripper_accuracy": float(gripper_accuracy.item()),
                "grad_norm": float(grad_norm.item()),
                "encoder_lr": float(optimizer.param_groups[0]["lr"]),
                "head_lr": float(optimizer.param_groups[1]["lr"]),
                "elapsed_seconds": time.monotonic() - start_time,
            }
            metric_history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if step % args.checkpoint_interval == 0 or step == args.steps:
            checkpoint = _save_checkpoint(
                output,
                model=model,
                optimizer=optimizer,
                step=step,
                args=args,
                manifest=manifest,
                metrics=metric_history,
            )
            print(json.dumps({"checkpoint": str(checkpoint), "step": step}), flush=True)
    (output / "training_metrics.json").write_text(
        json.dumps(metric_history, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
