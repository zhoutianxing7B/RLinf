# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small, dependency-light utilities shared by SAC implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.data.schema.embodied_types import Trajectory


def actor_only_warmup_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Discard reward-specific Q heads from a policy warm-start checkpoint."""
    return {name: tensor for name, tensor in state_dict.items() if "q_head" not in name}


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


def _select_trajectory_batch_columns(
    trajectory: Trajectory, indices: torch.Tensor
) -> Trajectory:
    """Select complete environment trajectories along the batch dimension."""
    if trajectory.rewards is None:
        raise ValueError("trajectory rewards are required for batch selection")
    batch_size = int(trajectory.rewards.shape[1])

    def select(value):
        if isinstance(value, torch.Tensor):
            if value.ndim >= 2 and value.shape[1] == batch_size:
                return value.index_select(1, indices).contiguous()
            return value.clone()
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        return value

    selected = Trajectory()
    for field_name in trajectory.__dataclass_fields__:
        setattr(selected, field_name, select(getattr(trajectory, field_name)))
    return selected


def extract_reward_elite_trajectory(
    trajectory: Trajectory, reward_threshold: float
) -> Trajectory | None:
    """Keep full environment trajectories crossing a physical-reward threshold.

    This selector reads only the executable agentic physical reward. It never
    reads simulator success, environment reward, or task predicates. Keeping
    the whole trajectory preserves rare grasp/place actions without turning
    those actions into rewards.
    """
    reward_threshold = float(reward_threshold)
    if not 0.0 < reward_threshold < 1.0:
        raise ValueError("reward_threshold must be in (0, 1)")
    if trajectory.rewards is None or trajectory.rewards.ndim < 2:
        return None
    reduce_dims = (0, *range(2, trajectory.rewards.ndim))
    max_reward = trajectory.rewards.amax(dim=reduce_dims)
    indices = torch.nonzero(max_reward > reward_threshold, as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    return _select_trajectory_batch_columns(trajectory, indices)
