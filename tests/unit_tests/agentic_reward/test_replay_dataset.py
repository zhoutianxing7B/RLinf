from __future__ import annotations

import torch

from rlinf.data.storage.replay.dataset import ReplayBufferDataset


class _FakeBuffer:
    def __init__(self, total_samples: int, value: float):
        self.total_samples = total_samples
        self.value = value
        self.sample_calls: list[int] = []

    def is_ready(self, minimum: int) -> bool:
        return self.total_samples >= minimum

    def sample(self, count: int) -> dict[str, torch.Tensor]:
        self.sample_calls.append(count)
        return {"value": torch.full((count, 1), self.value)}


def test_optional_empty_elite_buffer_does_not_block_online_training():
    replay = _FakeBuffer(total_samples=10, value=1.0)
    elite = _FakeBuffer(total_samples=0, value=2.0)
    dataset = ReplayBufferDataset(
        replay_buffer=replay,
        demo_buffer=elite,
        batch_size=8,
        min_replay_buffer_size=1,
        min_demo_buffer_size=0,
    )

    batch = next(iter(dataset))

    assert batch["value"].shape == (8, 1)
    assert replay.sample_calls == [8]
    assert elite.sample_calls == []


def test_populated_elite_buffer_is_mixed_half_and_half():
    replay = _FakeBuffer(total_samples=10, value=1.0)
    elite = _FakeBuffer(total_samples=4, value=2.0)
    dataset = ReplayBufferDataset(
        replay_buffer=replay,
        demo_buffer=elite,
        batch_size=8,
        min_replay_buffer_size=1,
        min_demo_buffer_size=0,
    )

    batch = next(iter(dataset))

    assert replay.sample_calls == [4]
    assert elite.sample_calls == [4]
    assert sorted(batch["value"].flatten().tolist()) == [1.0] * 4 + [2.0] * 4
