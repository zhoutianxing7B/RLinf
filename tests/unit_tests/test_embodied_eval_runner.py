# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner


def _handle(result):
    handle = MagicMock()
    handle.wait.return_value = result
    return handle


def test_evaluate_aggregates_all_configured_repeats():
    runner = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)
    runner.cfg = SimpleNamespace(runner={"eval_repeats": 2})
    runner.env_channel = "env-channel"
    runner.rollout_channel = "rollout-channel"
    runner.env = MagicMock()
    runner.rollout = MagicMock()
    runner.env.evaluate.side_effect = [
        _handle([{"success": torch.tensor([1.0, 0.0])}]),
        _handle([{"success": torch.tensor([1.0, 1.0])}]),
    ]
    runner.rollout.evaluate.side_effect = [
        _handle([{"latency": torch.tensor([1.0])}]),
        _handle([{"latency": torch.tensor([3.0])}]),
    ]

    metrics = runner.evaluate()

    assert runner.env.evaluate.call_args_list == 2 * [
        call(input_channel="env-channel", rollout_channel="rollout-channel")
    ]
    assert runner.rollout.evaluate.call_count == 2
    assert float(metrics["success"]) == pytest.approx(0.75)
    assert float(metrics["latency"]) == pytest.approx(2.0)
    assert metrics["num_trajectories"] == 4


def test_evaluate_rejects_non_positive_repeat_count():
    runner = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)
    runner.cfg = SimpleNamespace(runner={"eval_repeats": 0})
