# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small, dependency-light utilities shared by SAC implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def behavior_regularized_actor_loss(
    sac_actor_loss: torch.Tensor,
    policy_actions: torch.Tensor,
    behavior_actions: torch.Tensor,
    coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor an SAC actor to replay behavior while still optimizing Q.

    The replay action is an action label, never a reward. This conservative
    term prevents a high-dimensional actor from immediately exploiting Q
    extrapolation errors outside the data collected by the warm-start policy.
    """
    coefficient = float(coefficient)
    if not 0.0 <= coefficient < float("inf"):
        raise ValueError(
            "behavior regularization coefficient must be finite and non-negative"
        )
    if policy_actions.shape != behavior_actions.shape:
        raise ValueError(
            "policy and replay behavior actions must have identical shapes"
        )
    behavior_loss = F.mse_loss(policy_actions, behavior_actions.detach())
    return sac_actor_loss + coefficient * behavior_loss, behavior_loss
