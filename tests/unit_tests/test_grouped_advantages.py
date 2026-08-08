import pytest
import torch

from rlinf.algorithms.advantages import (
    compute_gae_advantages_and_returns,
    normalize_advantages_by_group,
)


def test_normalize_advantages_by_group_uses_independent_statistics():
    advantages = torch.tensor([[1.0, 10.0], [3.0, 14.0]])
    group_ids = torch.tensor([[0, 1], [0, 1]])

    normalized = normalize_advantages_by_group(advantages, group_ids)

    for group_id in (0, 1):
        values = normalized[group_ids == group_id]
        assert values.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert values.std().item() == pytest.approx(1.0, abs=2e-5)


def test_normalize_advantages_by_group_respects_mask_and_singleton_group():
    advantages = torch.tensor([[1.0, 10.0], [3.0, 99.0]])
    group_ids = torch.tensor([[0, 1], [0, 1]])
    loss_mask = torch.tensor([[True, True], [True, False]])

    normalized = normalize_advantages_by_group(advantages, group_ids, loss_mask)

    assert normalized[0, 1].item() == pytest.approx(0.0)
    assert normalized[1, 1].item() == pytest.approx(99.0)
    assert normalized[group_ids == 0].mean().item() == pytest.approx(0.0, abs=1e-6)


def test_gae_uses_grouped_advantage_normalization_when_enabled():
    rewards = torch.tensor([[1.0, 3.0, 10.0, 14.0], [0.0, 2.0, 0.0, 4.0]])
    values = torch.zeros(3, 4)
    dones = torch.zeros(3, 4, dtype=torch.bool)
    group_ids = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])

    advantages, _ = compute_gae_advantages_and_returns(
        rewards,
        gamma=0.0,
        gae_lambda=0.0,
        values=values,
        dones=dones,
        group_normalize_advantages=True,
        advantage_group_ids=group_ids,
    )

    for group_id in (0, 1):
        group_values = advantages[group_ids == group_id]
        assert group_values.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert group_values.std().item() == pytest.approx(1.0, abs=2e-5)
