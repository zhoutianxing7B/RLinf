from __future__ import annotations

import torch

from rlinf.algorithms.sac_utils import discounted_chunk_rewards


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
