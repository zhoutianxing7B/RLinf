#!/usr/bin/env python

"""Build group-safe manifests for N1.7 shared-semantic reward SFT."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

COLLECTOR_SCHEMA = "libero10-gr00t-n1.7-shared-semantic-rm-v2"
MANIFEST_SCHEMA = "rlinf-shared-semantic-rollout-v1"


def read_collector_manifest(data_root: Path) -> list[dict[str, Any]]:
    manifest_path = data_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Collector manifest does not exist: {manifest_path}")
    records: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != COLLECTOR_SCHEMA:
                raise ValueError(f"Unexpected schema at {manifest_path}:{line_number}")
            records[str(record["path"])] = record
    return list(records.values())


def validate_episode(data_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = data_root / record["path"]
    required = {
        "feature_semantic_tokens",
        "feature_semantic_attention_mask",
        "semantic_source_frame_id",
        "semantic_version",
        "semantic_episode_generation",
        "action_frame_id",
        "packet_age_s",
        "label_frame_success",
        "label_episode_success",
        "feature_action_history",
        "feature_embodiment_id",
        "delay_action_state",
        "delay_action_history",
        "delay_action_frame_id",
        "delay_valid_mask",
        "label_delay_completion",
    }
    with np.load(path, allow_pickle=False) as trajectory:
        missing = required - set(trajectory.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        frame_count = len(trajectory["semantic_source_frame_id"])
        for name in required - {"label_episode_success"}:
            if len(trajectory[name]) != frame_count:
                raise ValueError(f"{path}: {name} is not frame-aligned")
        episode_success = bool(trajectory["label_episode_success"])
        positive_count = int(trajectory["label_frame_success"].sum())
        if positive_count != int(episode_success):
            raise ValueError(
                f"{path}: success={episode_success}, positive_frames={positive_count}"
            )
        if episode_success:
            positive_index = int(np.flatnonzero(trajectory["label_frame_success"])[-1])
            if int(trajectory["semantic_source_frame_id"][positive_index]) != int(
                trajectory["action_frame_id"][positive_index]
            ):
                raise ValueError(f"{path}: positive packet is not source aligned")
        identities = np.stack(
            (
                trajectory["semantic_episode_generation"],
                trajectory["semantic_version"],
                trajectory["semantic_source_frame_id"],
            ),
            axis=1,
        )
        distinct_packets = 1 + int(
            np.any(identities[1:] != identities[:-1], axis=1).sum()
        )
        age_frames = np.maximum(
            trajectory["action_frame_id"] - trajectory["semantic_source_frame_id"],
            0,
        )
        delayed_age_frames = np.maximum(
            trajectory["delay_action_frame_id"]
            - trajectory["semantic_source_frame_id"][:, None],
            0,
        )
        valid_delay = trajectory["delay_valid_mask"]
        if np.any(delayed_age_frames[valid_delay] > 6):
            raise ValueError(f"{path}: delayed condition exceeds 6 frames")
        expected_delay = np.broadcast_to(
            np.arange(valid_delay.shape[1]), valid_delay.shape
        )
        if not np.array_equal(
            delayed_age_frames[valid_delay], expected_delay[valid_delay]
        ):
            raise ValueError(f"{path}: delayed state is not frame aligned")
    validated = dict(record)
    validated.update(
        {
            "episode_success": episode_success,
            "num_frames": frame_count,
            "num_distinct_packets": distinct_packets,
            "semantic_age_frames_max": int(age_frames.max(initial=0)),
        }
    )
    return validated


def group_key(episode: dict[str, Any]) -> tuple[int, str]:
    init_state_id = int(episode.get("init_state_id", -1))
    identity = str(init_state_id) if init_state_id >= 0 else str(episode["path"])
    return int(episode["task_id"]), identity


def assign_groups(
    episodes: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[tuple[int, str], str]:
    groups_by_task: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for episode in episodes:
        key = group_key(episode)
        if key not in groups_by_task[key[0]]:
            groups_by_task[key[0]].append(key)

    assignments = {}
    for task_id, groups in sorted(groups_by_task.items()):
        rng = np.random.default_rng(seed + task_id * 1009)
        groups = list(groups)
        rng.shuffle(groups)
        count = len(groups)
        if count == 1:
            train_end, val_end = 1, 1
        elif count == 2:
            train_end, val_end = 1, 2
        else:
            train_end = min(count - 2, max(1, round(count * train_ratio)))
            val_count = min(count - train_end - 1, max(1, round(count * val_ratio)))
            val_end = train_end + val_count
        for position, key in enumerate(groups):
            assignments[key] = (
                "train"
                if position < train_end
                else "val"
                if position < val_end
                else "test"
            )
    return assignments


def build_manifests(args: argparse.Namespace) -> None:
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    records = read_collector_manifest(data_root)
    episodes = []
    rejected = 0
    for record in records:
        try:
            episode = (
                validate_episode(data_root, record)
                if args.validate
                else dict(record)
            )
        except (FileNotFoundError, ValueError) as error:
            if not args.skip_invalid:
                raise
            rejected += 1
            print(f"rejected: path={record.get('path')} reason={error}")
            continue
        episodes.append(episode)
    if rejected:
        print(f"rejected_total={rejected}")
    assignments = assign_groups(episodes, args.seed, args.train_ratio, args.val_ratio)
    split_episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        split_episodes[assignments[group_key(episode)]].append(episode)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        selected = split_episodes[split]
        payload = {
            "schema_version": MANIFEST_SCHEMA,
            "collector_schema_version": COLLECTOR_SCHEMA,
            "split": split,
            "data_root": str(data_root),
            "episodes": selected,
        }
        output_path = output_dir / f"{split}.json"
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        outcomes = Counter(
            "success" if episode["episode_success"] else "failure"
            for episode in selected
        )
        groups = {group_key(episode) for episode in selected}
        print(
            f"{split}: episodes={len(selected)} groups={len(groups)} "
            f"outcomes={dict(outcomes)} path={output_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.set_defaults(validate=True)
    args = parser.parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train-ratio must be in (0,1)")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("val-ratio must be in [0,1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be less than 1")
    return args


if __name__ == "__main__":
    build_manifests(parse_args())
