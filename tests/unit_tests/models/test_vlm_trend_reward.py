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

from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.reward.vlm_reward_model import (
    BufferedVLMRewardModel,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    VLMTrendRewardInputBuilder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    VLMTrendBinaryDigitRewardParser,
)


class _HiddenModel:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden

    def __call__(self, **_kwargs):
        return SimpleNamespace(hidden_states=[self.hidden])


class _IdentityScalarHead:
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return features.squeeze(-1)


def test_compute_scalar_potential_uses_last_nonpadding_token() -> None:
    model = object.__new__(BufferedVLMRewardModel)
    hidden = torch.zeros(2, 4, 1)
    hidden[0, 1, 0] = -2.0
    hidden[1, 3, 0] = 2.0
    model._model = _HiddenModel(hidden)
    model.scalar_head = _IdentityScalarHead()

    potentials = model.compute_scalar_potential(
        {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.long
            ),
        }
    )

    torch.testing.assert_close(potentials, torch.sigmoid(torch.tensor([-2.0, 2.0])))


def test_setup_scalar_head_loads_shared_value_head(tmp_path) -> None:
    head = ValueHead(
        8,
        hidden_sizes=(4,),
        activation="silu",
        dropout=0.1,
        use_input_norm=True,
        bias_last=True,
    )
    head.eval()
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": head.state_dict(),
            "config": {
                "input_dim": 8,
                "hidden_dim": 4,
                "hidden_sizes": [4],
                "dropout": 0.1,
                "activation": "silu",
                "use_input_norm": True,
                "bias_last": True,
            },
        },
        path,
    )
    model = object.__new__(BufferedVLMRewardModel)
    torch.nn.Module.__init__(model)
    model.inference_mode = "scalar_head"
    model.scalar_head_path = str(path)
    model._model = SimpleNamespace(device="cpu")
    model.setup_scalar_head()
    features = torch.randn(3, 8)
    torch.testing.assert_close(model.scalar_head(features), head(features))


def test_setup_scalar_head_accepts_legacy_net_prefix(tmp_path) -> None:
    head = ValueHead(
        8,
        hidden_sizes=(4,),
        activation="silu",
        dropout=0.1,
        use_input_norm=True,
        bias_last=True,
    )
    head.eval()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {
                key.replace("mlp.", "net.", 1): value
                for key, value in head.state_dict().items()
            },
            "config": {
                "input_dim": 8,
                "hidden_dim": 4,
                "dropout": 0.1,
            },
        },
        path,
    )
    model = object.__new__(BufferedVLMRewardModel)
    torch.nn.Module.__init__(model)
    model.inference_mode = "scalar_head"
    model.scalar_head_path = str(path)
    model._model = SimpleNamespace(device="cpu")
    model.setup_scalar_head()
    features = torch.randn(3, 8)
    torch.testing.assert_close(model.scalar_head(features), head(features))


def _potential_model(
    scale: float = 1.0,
    gamma: float = 1.0,
    ema_alpha: float = 1.0,
    clip: float = 0.0,
):
    model = object.__new__(BufferedVLMRewardModel)
    model.potential_scale = scale
    model.potential_gamma = gamma
    model.potential_ema_alpha = ema_alpha
    model.potential_clip = clip
    model._previous_potentials = None
    return model


def test_potential_difference_is_zero_on_first_and_static_observation() -> None:
    model = _potential_model()
    valid = torch.tensor([True, True])

    first = model.potential_differences(torch.tensor([0.2, 0.8]), valid)
    static = model.potential_differences(torch.tensor([0.2, 0.8]), valid)

    torch.testing.assert_close(first, torch.zeros(2))
    torch.testing.assert_close(static, torch.zeros(2))


def test_potential_difference_is_signed_and_scaled() -> None:
    model = _potential_model(scale=0.5)
    valid = torch.tensor([True, True])
    model.potential_differences(torch.tensor([0.2, 0.8]), valid)

    rewards = model.potential_differences(torch.tensor([0.6, 0.5]), valid)

    torch.testing.assert_close(rewards, torch.tensor([0.2, -0.15]))


def test_done_resets_potential_history() -> None:
    model = _potential_model()
    valid = torch.tensor([True, True])
    model.potential_differences(torch.tensor([0.2, 0.8]), valid)
    terminal = model.potential_differences(
        torch.tensor([0.9, 0.7]), valid, dones=torch.tensor([True, False])
    )
    next_episode = model.potential_differences(torch.tensor([0.1, 0.6]), valid)

    torch.testing.assert_close(terminal, torch.tensor([0.7, -0.1]))
    torch.testing.assert_close(next_episode, torch.tensor([0.0, -0.1]))


def test_potential_difference_applies_ema_and_clip() -> None:
    model = _potential_model(scale=2.0, ema_alpha=0.5, clip=0.25)
    valid = torch.tensor([True, True])
    model.potential_differences(torch.tensor([0.2, 0.8]), valid)

    rewards = model.potential_differences(torch.tensor([0.8, 0.0]), valid)

    torch.testing.assert_close(rewards, torch.tensor([0.25, -0.25]))
    torch.testing.assert_close(model._previous_potentials, torch.tensor([0.5, 0.4]))


def test_model_success_bonus_is_one_shot_and_resets_on_done() -> None:
    model = _potential_model()
    model.success_threshold = 0.8
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model._success_fired = None
    model._success_streak = None
    valid = torch.tensor([True, True])

    first = model.apply_model_success_bonus(
        torch.zeros(2), torch.tensor([0.9, 0.7]), valid
    )
    repeated = model.apply_model_success_bonus(
        torch.zeros(2), torch.tensor([0.95, 0.9]), valid
    )
    terminal = model.apply_model_success_bonus(
        torch.zeros(2),
        torch.tensor([0.95, 0.9]),
        valid,
        dones=torch.tensor([True, False]),
    )
    next_episode = model.apply_model_success_bonus(
        torch.zeros(2), torch.tensor([0.9, 0.9]), valid
    )

    torch.testing.assert_close(first, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(repeated, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(terminal, torch.zeros(2))
    torch.testing.assert_close(next_episode, torch.tensor([1.0, 0.0]))


def test_model_success_bonus_requires_consecutive_confirmations() -> None:
    model = _potential_model()
    model.success_threshold = 0.5
    model.success_bonus = 1.0
    model.success_confirmation_windows = 2
    model._success_fired = None
    model._success_streak = None
    valid = torch.tensor([True])

    first = model.apply_model_success_bonus(torch.zeros(1), torch.tensor([0.9]), valid)
    interrupted = model.apply_model_success_bonus(
        torch.zeros(1), torch.tensor([0.1]), valid
    )
    restart = model.apply_model_success_bonus(
        torch.zeros(1), torch.tensor([0.9]), valid
    )
    confirmed = model.apply_model_success_bonus(
        torch.zeros(1), torch.tensor([0.9]), valid
    )

    torch.testing.assert_close(first, torch.zeros(1))
    torch.testing.assert_close(interrupted, torch.zeros(1))
    torch.testing.assert_close(restart, torch.zeros(1))
    torch.testing.assert_close(confirmed, torch.ones(1))


def test_binary_digit_parser_uses_sparse_rewards():
    parser = VLMTrendBinaryDigitRewardParser()

    rewards = parser.parse_rewards(["1", "0", "answer: 1", "unclear", "10"])

    torch.testing.assert_close(
        rewards,
        torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]),
    )


def test_binary_digit_parser_success_scores_ignore_reward_scaling():
    parser = VLMTrendBinaryDigitRewardParser(positive_reward=0.5, negative_reward=-1.0)

    scores = parser.parse_success_scores(["1", "0", "answer: 1", "unclear", "10"])
    rewards = parser.parse_rewards(["1", "0"])

    torch.testing.assert_close(scores[:3], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.isnan(scores[3]).item()
    torch.testing.assert_close(rewards, torch.tensor([0.5, -1.0]))


def test_done_shape_mismatch_raises_on_potential_and_success() -> None:
    model = _potential_model()
    model.success_threshold = 0.8
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model._success_fired = None
    model._success_streak = None
    valid = torch.tensor([True, True])

    with pytest.raises(ValueError, match="potential_differences"):
        model.potential_differences(
            torch.tensor([0.2, 0.8]), valid, dones=torch.tensor([True])
        )
    with pytest.raises(ValueError, match="apply_model_success_bonus"):
        model.apply_model_success_bonus(
            torch.zeros(2),
            torch.tensor([0.9, 0.9]),
            valid,
            dones=torch.tensor([True]),
        )


def test_invalid_success_score_does_not_trigger_bonus() -> None:
    model = _potential_model()
    model.success_threshold = 0.95
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model._success_fired = None
    model._success_streak = None
    valid = torch.tensor([True])

    scaled_reward_as_score = model.apply_model_success_bonus(
        torch.zeros(1), torch.tensor([0.5]), valid
    )
    nan_score = model.apply_model_success_bonus(
        torch.zeros(1), torch.tensor([float("nan")]), valid
    )
    label_score = model.apply_model_success_bonus(
        torch.zeros(1), torch.tensor([1.0]), valid
    )

    torch.testing.assert_close(scaled_reward_as_score, torch.zeros(1))
    torch.testing.assert_close(nan_score, torch.zeros(1))
    torch.testing.assert_close(label_score, torch.ones(1))


def test_success_adapter_restored_after_error() -> None:
    class _AdapterModel:
        def __init__(self) -> None:
            self.adapter = "default"

        def set_adapter(self, name: str) -> None:
            self.adapter = name

    model = object.__new__(BufferedVLMRewardModel)
    model._success_adapter_name = "success"
    model._model = _AdapterModel()

    def _boom() -> torch.Tensor:
        assert model._model.adapter == "success"
        raise RuntimeError("generate failed")

    with pytest.raises(RuntimeError, match="generate failed"):
        model._run_on_success_adapter(_boom)
    assert model._model.adapter == "default"


def test_terminal_success_builder_matches_sft_prompt(monkeypatch):
    builder = VLMTrendRewardInputBuilder(
        history_buffer_names=["history_window"],
        default_task_description="fallback task",
        include_task=True,
        prompt_template=(
            "Estimate task-conditioned success potential for this robot "
            "manipulation state.{task_text} The two synchronized videos show "
            "the same 5-frame history from two camera views."
        ),
        _processor=None,
    )
    videos = [[["main frames"], ["extra frames"]]]
    monkeypatch.setattr(builder, "extract_videos", lambda *_: videos)
    observations = {"task_descriptions": ["Pick up the cube."]}
    history_input = {"history_window": {}}

    prepared = builder.prepare_inputs(observations, history_input, [0])

    assert prepared["videos_list"] == videos
    assert prepared["prompt_texts_list"] == [
        [
            "Estimate task-conditioned success potential for this robot "
            "manipulation state. Task: Pick up the cube.. The two synchronized "
            "videos show the same 5-frame history from two camera views."
        ]
    ]


def test_buffered_vlm_returns_zero_before_first_window(monkeypatch):
    from rlinf.models.embodiment.reward.vlm_reward_model import BufferedVLMRewardModel

    model = object.__new__(BufferedVLMRewardModel)
    model.interval_reward = 0.0
    monkeypatch.setattr(model, "apply_gt_success_bonus", lambda rewards, _: rewards)

    rewards = model.compute_reward(
        {
            "dones": torch.zeros(3, dtype=torch.bool),
            "history_input": {"history_window": {}},
        }
    )

    torch.testing.assert_close(rewards, torch.zeros(3))
