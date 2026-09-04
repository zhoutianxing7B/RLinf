# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small, dependency-light utilities shared by SAC implementations."""

from __future__ import annotations

import torch


def discounted_chunk_rewards(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Aggregate per-step rewards for an action chunk with SAC discounting.

    Args:
        rewards: Tensor whose last dimension contains chronological rewards.
        gamma: Per-environment-step discount factor.

    Returns:
        A tensor with the last dimension reduced to size one.
    """
    if rewards.ndim == 0 or rewards.shape[-1] < 1:
        raise ValueError("rewards must contain at least one chunk step")
    discounts = torch.pow(
        torch.as_tensor(gamma, device=rewards.device, dtype=rewards.dtype),
        torch.arange(rewards.shape[-1], device=rewards.device, dtype=rewards.dtype),
    )
    return (rewards * discounts).sum(dim=-1, keepdim=True)
