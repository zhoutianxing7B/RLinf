# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train the shared ``ValueHead`` on frozen VLM potential/progress features.

Usage (from repo root):

    export FEAT_ROOT=/path/to/vlm_trend_potential_features
    export SCALAR_OUTPUT_ROOT=/path/to/vlm_trend_scalar_head
    python examples/reward/train_vlm_trend_scalar_head.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger

logger = get_logger()


def load_potential_shards(pattern: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and concatenate potential feature shards matching ``pattern``."""
    paths = (
        sorted(Path().glob(pattern))
        if not pattern.startswith("/")
        else sorted(Path(pattern).parent.glob(Path(pattern).name))
    )
    if not paths:
        raise ValueError(f"No feature shards match {pattern}")
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    return (
        torch.cat([payload["features"].float() for payload in payloads]),
        torch.cat([payload["targets"].float() for payload in payloads]),
    )


def load_progress_shards(pattern: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load progress pair-feature shards matching ``pattern``."""
    paths = sorted(Path(pattern).parent.glob(Path(pattern).name))
    if not paths:
        raise ValueError(f"No progress shards match {pattern}")
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    labels = [label for payload in payloads for label in payload["labels"]]
    return (
        torch.cat([payload["features"].float() for payload in payloads]),
        torch.cat([payload["teacher_deltas"].float() for payload in payloads]),
        labels,
    )


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Return Spearman correlation, or ``0.0`` when the statistic is non-finite."""
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


@torch.no_grad()
def predict(
    model: nn.Module, features: torch.Tensor, device: torch.device, batch_size: int
) -> torch.Tensor:
    """Run the head and return sigmoid probabilities on CPU."""
    model.eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        logits = model(features[start : start + batch_size].to(device)).squeeze(-1)
        outputs.append(torch.sigmoid(logits).cpu())
    return torch.cat(outputs)


def evaluate_head(
    model: nn.Module,
    potential_features: torch.Tensor,
    potential_targets: torch.Tensor,
    progress_features: torch.Tensor,
    progress_deltas: torch.Tensor,
    progress_labels: list[str],
    device: torch.device,
    batch_size: int,
    deadband: float,
) -> dict[str, Any]:
    """Evaluate potential fit and progress-direction accuracy."""
    values = predict(model, potential_features, device, batch_size)
    pair_values = predict(
        model,
        progress_features.reshape(-1, progress_features.shape[-1]),
        device,
        batch_size,
    ).reshape(-1, 2)
    predicted_deltas = pair_values[:, 1] - pair_values[:, 0]
    predicted_labels = [
        "up" if value > deadband else "down" if value < -deadband else "same"
        for value in predicted_deltas.tolist()
    ]
    return {
        "potential_mae": float(torch.abs(values - potential_targets).mean()),
        "potential_mse": float(torch.mean((values - potential_targets) ** 2)),
        "potential_spearman": safe_spearman(values.numpy(), potential_targets.numpy()),
        "delta_spearman": safe_spearman(
            predicted_deltas.numpy(), progress_deltas.numpy()
        ),
        "direction_accuracy": float(
            np.mean(
                [
                    prediction == target
                    for prediction, target in zip(predicted_labels, progress_labels)
                ]
            )
        ),
        "predicted_delta_mean": float(predicted_deltas.mean()),
        "predicted_delta_std": float(predicted_deltas.std()),
    }


def _pairwise_rank_loss(
    logits: torch.Tensor, targets: torch.Tensor, min_gap: float
) -> torch.Tensor:
    """Softplus pairwise ranking loss over a random permutation of the batch."""
    permutation = torch.randperm(len(targets), device=targets.device)
    target_difference = targets - targets[permutation]
    rank_mask = target_difference.abs() >= min_gap
    if not rank_mask.any():
        return logits.sum() * 0.0
    logit_difference = logits - logits[permutation]
    return nn.functional.softplus(
        -torch.sign(target_difference[rank_mask]) * logit_difference[rank_mask]
    ).mean()


def _train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    progress_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """Run one optimization epoch and return mean train losses."""
    model.train()
    losses: list[float] = []
    value_losses: list[float] = []
    delta_losses: list[float] = []
    local_rank_losses: list[float] = []
    progress_iterator = iter(progress_loader)
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features).squeeze(-1)
        value_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        rank_loss = _pairwise_rank_loss(logits, targets, cfg.rank_min_gap)
        try:
            pair_features, pair_targets = next(progress_iterator)
        except StopIteration:
            progress_iterator = iter(progress_loader)
            pair_features, pair_targets = next(progress_iterator)
        pair_features = pair_features.to(device)
        pair_targets = pair_targets.to(device)
        pair_logits = (
            model(pair_features.reshape(-1, pair_features.shape[-1]))
            .squeeze(-1)
            .reshape(-1, 2)
        )
        predicted_deltas = torch.sigmoid(pair_logits[:, 1]) - torch.sigmoid(
            pair_logits[:, 0]
        )
        delta_loss = nn.functional.smooth_l1_loss(
            predicted_deltas, pair_targets, beta=cfg.delta_beta
        )
        local_rank_mask = pair_targets.abs() >= cfg.local_rank_min_gap
        if local_rank_mask.any():
            local_logit_differences = pair_logits[:, 1] - pair_logits[:, 0]
            local_rank_loss = nn.functional.softplus(
                -torch.sign(pair_targets[local_rank_mask])
                * local_logit_differences[local_rank_mask]
            ).mean()
        else:
            local_rank_loss = pair_logits.sum() * 0.0
        loss = (
            value_loss
            + cfg.rank_weight * rank_loss
            + cfg.delta_weight * delta_loss
            + cfg.local_rank_weight * local_rank_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        value_losses.append(float(value_loss.detach().cpu()))
        delta_losses.append(float(delta_loss.detach().cpu()))
        local_rank_losses.append(float(local_rank_loss.detach().cpu()))
    return {
        "train_loss": float(np.mean(losses)),
        "train_value_loss": float(np.mean(value_losses)),
        "train_delta_loss": float(np.mean(delta_losses)),
        "train_local_rank_loss": float(np.mean(local_rank_losses)),
    }


def run(cfg: DictConfig) -> None:
    """Train ``ValueHead`` on feature shards and write ``best.pt`` / metrics."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_features, train_targets = load_potential_shards(cfg.train_pattern)
    eval_features, eval_targets = load_potential_shards(cfg.eval_pattern)
    progress_features, progress_deltas, progress_labels = load_progress_shards(
        cfg.progress_pattern
    )
    train_progress_features, train_progress_deltas, _ = load_progress_shards(
        cfg.train_progress_pattern
    )
    hidden_sizes = (int(cfg.hidden_dim),)
    model = ValueHead(
        int(train_features.shape[-1]),
        hidden_sizes=hidden_sizes,
        output_dim=1,
        activation=str(cfg.activation),
        bias_last=bool(cfg.bias_last),
        dropout=float(cfg.dropout),
        use_input_norm=bool(cfg.use_input_norm),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    progress_loader = DataLoader(
        TensorDataset(train_progress_features, train_progress_deltas),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_score = float("-inf")
    best_metrics: dict[str, Any] = {}

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, int(cfg.epochs) + 1):
            train_losses = _train_one_epoch(
                model, optimizer, loader, progress_loader, cfg, device
            )
            if epoch % int(cfg.eval_interval) != 0 and epoch != int(cfg.epochs):
                continue
            metrics = evaluate_head(
                model,
                eval_features,
                eval_targets,
                progress_features,
                progress_deltas,
                progress_labels,
                device,
                int(cfg.eval_batch_size),
                float(cfg.progress_deadband),
            )
            metrics.update({"epoch": epoch, **train_losses})
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            score = metrics["potential_spearman"] + metrics["delta_spearman"]
            if score > best_score:
                best_score = score
                best_metrics = metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "input_dim": int(train_features.shape[-1]),
                            "hidden_dim": int(cfg.hidden_dim),
                            "hidden_sizes": list(hidden_sizes),
                            "dropout": float(cfg.dropout),
                            "activation": str(cfg.activation),
                            "use_input_norm": bool(cfg.use_input_norm),
                            "bias_last": bool(cfg.bias_last),
                        },
                        "metrics": metrics,
                    },
                    output_dir / "best.pt",
                )
            logger.info("%s", json.dumps(metrics))
    (output_dir / "best_metrics.json").write_text(
        json.dumps(best_metrics, indent=2), encoding="utf-8"
    )


@hydra.main(
    version_base="1.1", config_path="config", config_name="vlm_trend_scalar_head"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
