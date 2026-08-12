#!/usr/bin/env python

"""Collect one-pass GR00T N1.7 semantic packets for both expert SFT stages."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.training.delay_augmentation import DelayAugmentationConfig

from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    _resize_semantic_token_axis,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.semantic_server import (
    Gr00tN1d7SemanticBackbonePolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backbone-model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--samples-per-task", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--min-delay-frames", type=int, default=0)
    parser.add_argument("--max-delay-frames", type=int, default=8)
    parser.add_argument("--action-history-length", type=int, default=6)
    parser.add_argument("--terminal-fraction", type=float, default=0.15)
    parser.add_argument("--control-dt-ms", type=float, default=50.0)
    parser.add_argument("--semantic-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_dataset_metadata(
    dataset_path: Path,
) -> tuple[list[str], dict[str, list[dict]]]:
    meta_path = dataset_path / "meta"
    with (meta_path / "tasks.jsonl").open() as file:
        tasks = [json.loads(line)["task"] for line in file]
    episodes_by_task: dict[str, list[dict]] = defaultdict(list)
    with (meta_path / "episodes.jsonl").open() as file:
        for line in file:
            episode = json.loads(line)
            episodes_by_task[episode["tasks"][0]].append(episode)
    missing = [task for task in tasks if not episodes_by_task[task]]
    if missing:
        raise RuntimeError(f"Tasks without expert episodes: {missing}")
    return tasks, episodes_by_task


def _sample_specs(
    dataset_path: Path,
    samples_per_task: int,
    min_delay_frames: int,
    max_delay_frames: int,
    action_horizon: int,
    terminal_fraction: float,
    seed: int,
) -> list[dict]:
    tasks, episodes_by_task = _load_dataset_metadata(dataset_path)
    rng = np.random.default_rng(seed)
    delays = np.arange(min_delay_frames, max_delay_frames + 1, dtype=np.int64)
    specs = []
    terminal_count = max(1, round(samples_per_task * terminal_fraction))
    for task_id, task in enumerate(tasks):
        episodes = list(episodes_by_task[task])
        rng.shuffle(episodes)
        for sample_id in range(samples_per_task):
            episode = episodes[sample_id % len(episodes)]
            episode_length = int(episode["length"])
            effective_length = episode_length - action_horizon + 1
            if effective_length <= 0:
                raise RuntimeError(
                    f"Episode {episode['episode_index']} is shorter than action horizon"
                )
            if sample_id < terminal_count:
                step_index = effective_length - 1
            else:
                step_index = int(rng.integers(0, effective_length))
            delay_frames = int(delays[(sample_id + task_id) % len(delays)])
            specs.append(
                {
                    "task_id": task_id,
                    "episode_index": int(episode["episode_index"]),
                    "episode_length": episode_length,
                    "step_index": step_index,
                    "source_step_index": max(0, step_index - delay_frames),
                    "delay_frames": delay_frames,
                }
            )
    rng.shuffle(specs)
    return specs


def _delay_config(
    delay_frames: int,
    control_dt_ms: float,
    action_history_length: int,
) -> DelayAugmentationConfig:
    delay_ms = float(delay_frames) * control_dt_ms
    return DelayAugmentationConfig(
        probability_zero_delay=1.0 if delay_frames == 0 else 0.0,
        min_delay_ms=delay_ms,
        max_delay_ms=delay_ms,
        delay_distribution="uniform",
        action_history_length=action_history_length,
        enable_fresh_teacher_context=False,
    )


def _encode_batch(
    policy: Gr00tN1d7SemanticBackbonePolicy,
    datapoints: list[dict],
    semantic_tokens: int,
) -> dict[str, torch.Tensor]:
    collated = dict(policy.processor.collator(datapoints)["inputs"])
    collated = {
        key: value for key, value in collated.items() if not key.startswith("fresh_")
    }
    collated = policy._canonicalize_text_inputs(collated, semantic_tokens)
    backbone_inputs = policy._prepare_backbone_input(collated)
    with torch.inference_mode(), torch.autocast(
        device_type=policy.device.type,
        dtype=policy.torch_dtype,
        enabled=policy.device.type == "cuda",
    ):
        semantic_outputs = policy.model.backbone(backbone_inputs)
    semantic_outputs = _resize_semantic_token_axis(semantic_outputs, semantic_tokens)

    required = {"state", "action", "action_mask", "embodiment_id", "packet_age"}
    missing = required - set(collated)
    if missing:
        raise RuntimeError(f"GR00T processor output is missing {sorted(missing)}")
    result = {
        "state": collated["state"].to(torch.float32),
        "action": collated["action"].to(torch.float32),
        "action_mask": collated["action_mask"].to(torch.bool),
        "embodiment_id": collated["embodiment_id"].to(torch.int32),
        "packet_age_s": collated["packet_age"].to(torch.float32),
        "action_history": collated["action_history"].to(torch.float32),
    }
    result.update(
        {f"semantic_{key}": value for key, value in dict(semantic_outputs).items()}
    )
    return result


def _append_batch(
    storage: dict[str, list[torch.Tensor]], batch: dict[str, torch.Tensor]
) -> None:
    for key, value in batch.items():
        storage[key].append(value.detach().cpu().contiguous())


def main() -> None:
    args = parse_args()
    if args.samples_per_task <= 0 or args.batch_size <= 0:
        raise ValueError("samples-per-task and batch-size must be positive")
    if args.min_delay_frames < 0 or args.max_delay_frames < args.min_delay_frames:
        raise ValueError("invalid delay frame range")
    if args.action_history_length <= 0:
        raise ValueError("action-history-length must be positive")
    if not 0.0 <= args.terminal_fraction <= 1.0:
        raise ValueError("terminal-fraction must be in [0,1]")

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policy = Gr00tN1d7SemanticBackbonePolicy(
        args.model_path,
        device=args.device,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        load_bf16=True,
        backbone_model_path=args.backbone_model_path,
        enable_raw_preprocessing=False,
        text_padding_tokens=args.semantic_tokens,
    )
    policy.processor.eval()
    embodiment_tag = EmbodimentTag("libero_sim")
    modality_configs = policy.processor.modality_configs[embodiment_tag.value]
    dataset = ShardedSingleStepDataset(
        dataset_path=dataset_path,
        embodiment_tag=embodiment_tag,
        modality_configs=modality_configs,
        shard_size=2048,
        episode_sampling_rate=0.1,
        seed=args.seed,
        allow_padding=True,
        delay_augmentation_config=_delay_config(
            0, args.control_dt_ms, args.action_history_length
        ),
        control_dt_ms=args.control_dt_ms,
    )
    dataset.set_processor(policy.processor)
    action_horizon = len(modality_configs["action"].delta_indices)
    specs = _sample_specs(
        dataset_path,
        args.samples_per_task,
        args.min_delay_frames,
        args.max_delay_frames,
        action_horizon,
        args.terminal_fraction,
        args.seed,
    )

    storage: dict[str, list[torch.Tensor]] = defaultdict(list)
    for start in range(0, len(specs), args.batch_size):
        batch_specs = specs[start : start + args.batch_size]
        datapoints = []
        for spec in batch_specs:
            dataset.delay_augmentation_config = _delay_config(
                spec["delay_frames"],
                args.control_dt_ms,
                args.action_history_length,
            )
            episode = dataset.episode_loader[spec["episode_index"]]
            datapoints.append(dataset.get_datapoint(episode, spec["step_index"]))
        _append_batch(
            storage,
            _encode_batch(policy, datapoints, args.semantic_tokens),
        )
        print(f"encoded {min(start + len(batch_specs), len(specs))}/{len(specs)}")

    forward_inputs = {key: torch.cat(values) for key, values in storage.items()}
    metadata = {
        key: torch.tensor([spec[key] for spec in specs], dtype=torch.long)
        for key in (
            "task_id",
            "episode_index",
            "episode_length",
            "step_index",
            "source_step_index",
            "delay_frames",
        )
    }
    count = len(specs)
    if any(value.shape[0] != count for value in forward_inputs.values()):
        raise RuntimeError("Collected tensor fields have inconsistent sample counts")
    payload = {
        "format_version": 2,
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "dataset_path": str(dataset_path),
        "semantic_tokens": args.semantic_tokens,
        "control_dt_ms": args.control_dt_ms,
        "action_history_length": args.action_history_length,
        "action_horizon": action_horizon,
        "forward_inputs": forward_inputs,
        "metadata": metadata,
    }
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "samples": count,
                "task_counts": dict(Counter(metadata["task_id"].tolist())),
                "delay_counts": dict(Counter(metadata["delay_frames"].tolist())),
                "bytes": output_path.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
