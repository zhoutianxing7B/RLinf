"""Deterministic helpers shared by semantic-age scheduling and eval sampling."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def stable_text_ids(texts: Sequence[str]) -> torch.Tensor:
    """Map task descriptions to reproducible non-negative integer ids."""
    return torch.tensor([_stable_int("task", text) % (2**31 - 1) for text in texts])


def eval_noise_seeds(
    task_ids: Sequence[int] | torch.Tensor,
    trial_ids: Sequence[int] | torch.Tensor,
    frame_ids: Sequence[int] | torch.Tensor,
    base_seed: int | Sequence[int],
) -> list[int]:
    """Derive one reproducible diffusion-noise seed per evaluation sample."""
    task_values = torch.as_tensor(task_ids).reshape(-1).tolist()
    trial_values = torch.as_tensor(trial_ids).reshape(-1).tolist()
    frame_values = torch.as_tensor(frame_ids).reshape(-1).tolist()
    if not (len(task_values) == len(trial_values) == len(frame_values)):
        raise ValueError("task_ids, trial_ids, and frame_ids must have equal length")
    if isinstance(base_seed, int):
        base_values = [base_seed] * len(task_values)
    else:
        base_values = [int(value) for value in base_seed]
        if len(base_values) != len(task_values):
            raise ValueError("per-sample base_seed must match the batch length")
    return [
        _stable_int("noise", seed, task, trial, frame) % (2**63 - 1)
        for seed, task, trial, frame in zip(
            base_values, task_values, trial_values, frame_values, strict=True
        )
    ]


def _validate_age_range(min_frames: int, max_frames: int) -> tuple[int, int]:
    minimum = int(min_frames)
    maximum = int(max_frames)
    if minimum < 0 or maximum < minimum:
        raise ValueError(
            f"invalid semantic age range [{minimum}, {maximum}]"
        )
    return minimum, maximum


def eval_semantic_age_frames(
    current_frames: Sequence[int] | torch.Tensor,
    min_frames: int,
    max_frames: int,
    seed: int,
) -> list[int]:
    """Choose deterministic per-env ages for a reproducible evaluation rollout."""
    minimum, maximum = _validate_age_range(min_frames, max_frames)
    width = maximum - minimum + 1
    return [
        minimum + _stable_int("eval-age", seed, index, int(frame)) % width
        for index, frame in enumerate(torch.as_tensor(current_frames).reshape(-1).tolist())
    ]


def eval_semantic_age_frame(
    rollout_step: int,
    stream_id: int,
    min_frames: int,
    max_frames: int,
    seed: int,
) -> int:
    """Choose one deterministic age shared by an evaluation env chunk."""
    minimum, maximum = _validate_age_range(min_frames, max_frames)
    width = maximum - minimum + 1
    return minimum + _stable_int("eval-stream-age", seed, stream_id, rollout_step) % width


def train_semantic_age_frame(
    rollout_step: int,
    stream_id: int,
    min_frames: int,
    max_frames: int,
    seed: int,
) -> int:
    """Choose a deterministic age for one training stream and rollout step."""
    minimum, maximum = _validate_age_range(min_frames, max_frames)
    width = maximum - minimum + 1
    return minimum + _stable_int("train-age", seed, stream_id, rollout_step) % width
