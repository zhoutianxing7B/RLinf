
import pytest
import torch

from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    GR00T_N1_7_ForRLActionPrediction,
    _execution_action_prefix,
)


def test_execution_prefix_keeps_model_horizon_separate_from_env_horizon():
    predictions = torch.arange(2 * 16 * 7).reshape(2, 16, 7)

    executed = _execution_action_prefix(predictions, execution_horizon=4)

    assert executed.shape == (2, 4, 7)
    torch.testing.assert_close(executed, predictions[:, :4])


@pytest.mark.parametrize(
    ("cached_env_ids", "cached_generations", "has_cache", "expected"),
    [
        ((10, 11), (4, 4), True, False),
        ((10, 11), (3, 4), True, True),
        ((10, 12), (4, 4), True, True),
        ((10, 11), (4, 4), False, True),
    ],
)
def test_semantic_action_only_transform_requires_matching_cache_identity(
    cached_env_ids, cached_generations, has_cache, expected
):
    policy = GR00T_N1_7_ForRLActionPrediction.__new__(GR00T_N1_7_ForRLActionPrediction)
    policy._semantic_enabled = True
    policy._semantic_central_cache = True
    policy._semantic_cache = object() if has_cache else None
    policy._latest_semantic_metadata = {
        "env_ids": cached_env_ids,
        "episode_generations": cached_generations,
    }
    policy._rollout_semantic_metadata = {
        "env_ids": torch.tensor([10, 11]),
        "frame_ids": torch.tensor([8, 8]),
        "episode_generations": torch.tensor([4, 4]),
    }
    policy._semantic_last_episode_generations = {10: 4, 11: 4}
    policy._semantic_publish_interval_frames = 0
    policy._semantic_boundary_publish = False
    policy._semantic_env_bootstrap_publish = True

    assert policy._semantic_requires_publish_inputs() is expected


@pytest.mark.parametrize("execution_horizon", [0, 17])
def test_execution_prefix_rejects_invalid_horizon(execution_horizon):
    with pytest.raises(ValueError, match="execution_horizon"):
        _execution_action_prefix(torch.zeros(2, 16, 7), execution_horizon)
