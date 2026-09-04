from __future__ import annotations

import pytest
import torch

from rlinf.algorithms.sac_utils import (
    actor_only_warmup_state_dict,
    behavior_regularized_actor_loss,
    discounted_chunk_rewards,
    extract_reward_elite_trajectory,
)
from rlinf.data.schema.embodied_types import Trajectory


def test_discounted_chunk_rewards_matches_manual_sum():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    actual = discounted_chunk_rewards(rewards, gamma=0.9)
    expected = torch.tensor(
        [[1.0 + 0.9 * 2.0 + 0.9**2 * 3.0], [-1.0 + 0.9 * 0.5 + 0.9**2 * 2.0]]
    )
    torch.testing.assert_close(actual, expected)


def test_potential_shaping_telescopes_across_action_chunk():
    gamma = 0.99
    potentials = torch.tensor([[0.2, 0.4, 0.1, 0.8]])
    shaped = gamma * potentials[:, 1:] - potentials[:, :-1]
    actual = discounted_chunk_rewards(shaped, gamma=gamma)
    expected = gamma**3 * potentials[:, -1:] - potentials[:, :1]
    torch.testing.assert_close(actual, expected)


def test_behavior_regularization_anchors_policy_actions():
    policy_actions = torch.ones(2, requires_grad=True)
    behavior_actions = torch.zeros(2)
    sac_loss = torch.tensor(2.0)
    total, behavior = behavior_regularized_actor_loss(
        sac_loss, policy_actions, behavior_actions, coefficient=3.0
    )
    torch.testing.assert_close(behavior, torch.tensor(1.0))
    torch.testing.assert_close(total, torch.tensor(5.0))
    total.backward()


def test_reward_elite_replay_keeps_full_selected_environment_trajectory():
    rewards = torch.zeros(4, 3, 2)
    rewards[2, 1, 0] = 0.8
    trajectory = Trajectory(
        max_episode_length=4,
        model_weights_id="step-7",
        rewards=rewards,
        actions=torch.arange(4 * 3 * 2).reshape(4, 3, 2),
        terminations=torch.zeros(4, 3, dtype=torch.bool),
        curr_obs={"states": torch.arange(4 * 3 * 5).reshape(4, 3, 5)},
        next_obs={"states": torch.ones(4, 3, 5)},
    )

    elite = extract_reward_elite_trajectory(trajectory, reward_threshold=0.5)

    assert elite is not None
    assert elite.rewards.shape == (4, 1, 2)
    torch.testing.assert_close(elite.rewards[:, 0], trajectory.rewards[:, 1])
    torch.testing.assert_close(elite.actions[:, 0], trajectory.actions[:, 1])
    torch.testing.assert_close(
        elite.curr_obs["states"][:, 0], trajectory.curr_obs["states"][:, 1]
    )
    assert elite.max_episode_length == 4
    assert elite.model_weights_id == "step-7"


def test_reward_elite_replay_rejects_no_completion_and_invalid_threshold():
    trajectory = Trajectory(rewards=torch.full((3, 2, 1), 0.25))
    assert extract_reward_elite_trajectory(trajectory, reward_threshold=0.5) is None
    with pytest.raises(ValueError):
        extract_reward_elite_trajectory(trajectory, reward_threshold=1.0)


def test_actor_warmup_discards_reward_specific_q_heads():
    state = {
        "encoder.weight": torch.ones(1),
        "actor_mean.weight": torch.ones(1),
        "q_head.0.weight": torch.ones(1),
    }
    filtered = actor_only_warmup_state_dict(state)
    assert set(filtered) == {"encoder.weight", "actor_mean.weight"}
