#!/usr/bin/env python

"""Evaluate a delayed shared-semantic reward expert with per-age metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlinf.data.datasets.reward_model import SharedSemanticRolloutDataset
from rlinf.models.embodiment.reward.shared_semantic_reward_model import (
    SharedSemanticTemporalRewardModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--delay-max-frames", type=int, default=6)
    return parser.parse_args()


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    begin = 0
    while begin < len(scores):
        end = begin + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1) + 1.0
        begin = end
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = int(len(labels) - positive_count)
    return float(
        (ranks[positives].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    positive_positions = np.flatnonzero(sorted_labels)
    return float(
        np.mean(cumulative[positive_positions] / (positive_positions + 1))
    )


def metric_block(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    predictions = scores >= 0.5
    tn = int(((labels == 0) & ~predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    tp = int(((labels == 1) & predictions).sum())
    metrics: dict[str, float | int] = {
        "samples": int(labels.size),
        "positives": int(labels.sum()),
        "negatives": int(labels.size - labels.sum()),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(
            0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        ),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if np.unique(labels).size == 2:
        metrics["roc_auc"] = binary_auc(labels, scores)
        metrics["pr_auc"] = average_precision(labels, scores)
    return metrics


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if (checkpoint / "actor").is_dir():
        checkpoint = checkpoint / "actor"
    dataset = SharedSemanticRolloutDataset(
        args.manifest,
        history_size=args.history_size,
        samples_per_episode=14,
        positive_samples_per_episode=7,
        delay_min_frames=0,
        delay_max_frames=args.delay_max_frames,
        control_hz=20.0,
        state_dim=132,
        action_history_length=4,
        action_dim=132,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    cfg = OmegaConf.create(
        {
            "history_size": args.history_size,
            "history_buffer_name": "semantic_window",
            "token_dim": 2048,
            "adapter_dim": 256,
            "temporal_dim": 256,
            "task_dim": 64,
            "num_tasks": 10,
            "state_dim": 132,
            "action_history_length": 4,
            "action_dim": 132,
            "num_embodiments": 32,
            "require_action_condition": True,
            "age_hidden_dim": 64,
            "age_normalization_frames": 8,
            "age_normalization_s": 0.4,
            "head_hidden_dim": 128,
            "progress_weight": 0.0,
            "completion_weight": 1.0,
            "failure_weight": 0.0,
            "uncertainty_penalty": 0.0,
            "model_path": str(checkpoint),
            "require_checkpoint": True,
        }
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = SharedSemanticTemporalRewardModel(cfg).to(device=device, dtype=dtype).eval()

    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_ages: list[np.ndarray] = []
    with torch.inference_mode():
        for features, labels in loader:
            outputs = model(features)
            all_labels.append(labels["completion"].numpy().astype(np.int64))
            all_scores.append(outputs["completion"].float().cpu().numpy())
            all_ages.append(
                features["semantic_age_frames"][:, -1].numpy().astype(np.int64)
            )

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    ages = np.concatenate(all_ages)
    report: dict[str, object] = {
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(checkpoint),
        "overall": metric_block(labels, scores),
        "by_age": {},
    }
    for age in range(args.delay_max_frames + 1):
        mask = ages == age
        report["by_age"][str(age)] = metric_block(labels[mask], scores[mask])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
