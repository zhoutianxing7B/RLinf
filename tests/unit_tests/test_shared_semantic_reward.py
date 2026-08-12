import json
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.reward_model import (
    SharedSemanticRewardDataset,
    SharedSemanticRolloutDataset,
)
from rlinf.models.embodiment.reward.shared_semantic_reward_model import (
    SharedSemanticTemporalRewardModel,
)
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker


def _model() -> SharedSemanticTemporalRewardModel:
    return SharedSemanticTemporalRewardModel(
        OmegaConf.create(
            {
                "history_size": 2,
                "history_buffer_name": "semantic_window",
                "token_dim": 8,
                "adapter_dim": 6,
                "temporal_dim": 6,
                "task_dim": 3,
                "num_tasks": 2,
                "age_hidden_dim": 4,
                "head_hidden_dim": 5,
                "interval_reward": 0.0,
            }
        )
    )


def _history_input():
    tokens = [torch.randn(3, 8), torch.randn(3, 8)]
    duplicate = [torch.randn(3, 8), torch.randn(3, 8)]
    repeated_token = torch.randn(3, 8)
    repeated = [repeated_token, repeated_token]
    return {
        "task_ids": torch.tensor([0, 1, 1]),
        "history_input": {
            "semantic_window": {
                "semantic_tokens": [tokens, duplicate, repeated],
                "semantic_attention_mask": [
                    [torch.ones(3, dtype=torch.bool)] * 2,
                    [torch.ones(3, dtype=torch.bool)] * 2,
                    [torch.ones(3, dtype=torch.bool)] * 2,
                ],
                "semantic_source_frame_ids": [[0, 4], [2, 2], [6, 6]],
                "semantic_episode_generations": [[0, 0], [1, 1], [2, 2]],
                "semantic_versions": [[1, 2], [3, 3], [4, 4]],
                "action_frame_ids": [[1, 5], [3, 4], [7, 7]],
                "packet_age_s": [[0.05, 0.05], [0.05, 0.10], [0.05, 0.05]],
            }
        },
    }


def test_shared_reward_scores_new_state_with_reused_semantic():
    model = _model().eval()
    features = model._extract_online_features(_history_input())
    outputs = model(features)

    assert outputs["rewards"].shape == (3,)
    assert outputs["valid_transition"].tolist() == [True, True, False]
    assert outputs["rewards"][2].item() == 0.0


def test_shared_reward_has_no_visual_backbone():
    module_names = {name for name, _ in _model().named_modules()}

    assert not any("backbone" in name for name in module_names)
    assert "semantic_adapter" in module_names
    assert "temporal_expert" in module_names


def test_rollout_semantics_are_converted_without_images():
    entry = EnvWorker._shared_semantic_reward_entry(
        {
            "semantic_backbone_features": torch.randn(2, 3, 8),
            "semantic_backbone_attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "rollout_semantic_source_frame_ids": torch.tensor([3, 4]),
            "rollout_semantic_episode_generations": torch.tensor([0, 1]),
            "rollout_semantic_versions": torch.tensor([7, 8]),
            "action_frame_ids": torch.tensor([5, 6]),
            "packet_age_s": torch.tensor([0.1, 0.1]),
            "rollout_task_ids": torch.tensor([0, 1]),
            "state": torch.randn(2, 1, 132),
            "action_history": torch.randn(2, 4, 32),
            "embodiment_id": torch.tensor([3, 3]),
        }
    )

    assert entry["semantic_tokens"].shape == (2, 3, 8)
    assert "main_images" not in entry
    assert "wrist_images" not in entry
    assert entry["action_states"].shape == (2, 1, 132)
    assert entry["action_history"].shape == (2, 4, 32)


def test_endpoint_credit_assignment_uses_source_frame_interval():
    worker = EnvWorker.__new__(EnvWorker)
    worker.reward_weight = 1.0
    worker.semantic_reward_assign_mode = "endpoint"
    worker._semantic_reward_previous_packets = [[(0, 0, 1)]]
    rollout = SimpleNamespace(
        rewards=[torch.zeros(1, 2), torch.zeros(1, 2)],
        forward_inputs=[
            {
                "action_frame_ids": torch.tensor([1]),
                "rollout_semantic_episode_generations": torch.tensor([0]),
            },
            {
                "action_frame_ids": torch.tensor([5]),
                "rollout_semantic_episode_generations": torch.tensor([0]),
            },
        ],
    )
    worker.rollout_results = [rollout]

    worker.assign_semantic_interval_reward(
        0,
        torch.tensor([[2.0]]),
        {
            "rollout_semantic_source_frame_ids": torch.tensor([5]),
            "rollout_semantic_episode_generations": torch.tensor([0]),
            "rollout_semantic_versions": torch.tensor([2]),
        },
    )

    assert rollout.rewards[0].sum().item() == 0.0
    assert rollout.rewards[1][0, -1].item() == 2.0


def test_endpoint_credit_assignment_aligns_unrewarded_bootstrap_input():
    worker = EnvWorker.__new__(EnvWorker)
    worker.reward_weight = 1.0
    worker.semantic_reward_assign_mode = "endpoint"
    worker._semantic_reward_previous_packets = [[(0, 0, 1)]]
    rollout = SimpleNamespace(
        rewards=[torch.zeros(1, 2)],
        forward_inputs=[
            {
                "action_frame_ids": torch.tensor([1]),
                "rollout_semantic_episode_generations": torch.tensor([0]),
            },
            {
                "action_frame_ids": torch.tensor([5]),
                "rollout_semantic_episode_generations": torch.tensor([0]),
            },
        ],
    )
    worker.rollout_results = [rollout]

    worker.assign_semantic_interval_reward(
        0,
        torch.tensor([[2.0]]),
        {
            "rollout_semantic_source_frame_ids": torch.tensor([5]),
            "rollout_semantic_episode_generations": torch.tensor([0]),
            "rollout_semantic_versions": torch.tensor([2]),
        },
    )

    assert rollout.rewards[0][0, -1].item() == 2.0


def test_reward_worker_skips_model_for_terminal_marker():
    worker = EmbodiedRewardWorker.__new__(EmbodiedRewardWorker)

    rewards = EmbodiedRewardWorker.compute_reward.__wrapped__.__wrapped__(
        worker,
        {"skip_reward": torch.ones(3, 1, dtype=torch.bool)}
    )

    assert rewards.shape == (3, 1)
    assert rewards.sum().item() == 0.0


def test_shared_reward_terminal_marker_closes_worker_without_reward():
    worker = EnvWorker.__new__(EnvWorker)
    worker.use_shared_semantic_reward = True
    worker.train_num_envs_per_stage = 2
    worker.train_batch_size = 2
    worker.env_decoupled_mode = False
    worker.cfg = OmegaConf.create({"reward": {"group_name": "RewardGroup"}})
    sent = {}
    worker.send_to = lambda **kwargs: sent.update(kwargs)
    worker.recv_from = lambda **kwargs: torch.zeros(2, 1)

    output = EnvWorker.get_reward_model_output.__wrapped__.__wrapped__(
        worker,
        SimpleNamespace(),
        send_channel=object(),
        recv_channel=object(),
        stage_id=0,
        last_run=True,
        policy_forward_inputs={},
    )

    assert output is None
    assert sent["data"]["skip_reward"].all()
    assert sent["data"]["last_run"].all()


def test_shared_reward_empty_intermediate_batch_keeps_worker_open():
    worker = EnvWorker.__new__(EnvWorker)
    worker.use_shared_semantic_reward = True
    worker.train_num_envs_per_stage = 2
    worker.train_batch_size = 2
    worker.env_decoupled_mode = False
    worker.cfg = OmegaConf.create({"reward": {"group_name": "RewardGroup"}})
    sent = {}
    received = {}
    worker.send_to = lambda **kwargs: sent.update(kwargs)
    worker.recv_from = lambda **kwargs: (
        received.update(kwargs) or torch.zeros(2, 1)
    )

    output = EnvWorker.get_reward_model_output.__wrapped__.__wrapped__(
        worker,
        SimpleNamespace(),
        send_channel=object(),
        recv_channel=object(),
        stage_id=0,
        last_run=False,
        policy_forward_inputs={},
    )

    assert output is None
    assert sent["data"]["skip_reward"].all()
    assert not sent["data"]["last_run"].any()
    assert received["tag"] == "train_reward_obs"


def test_shared_semantic_dataset_and_training_forward(tmp_path):
    path = tmp_path / "packets.pt"
    torch.save(
        {
            "features": {
                "semantic_tokens": torch.randn(3, 2, 3, 8),
                "semantic_attention_mask": torch.ones(3, 2, 3, dtype=torch.bool),
                "semantic_age_frames": torch.ones(3, 2),
                "semantic_age_s": torch.full((3, 2), 0.05),
                "semantic_interval_frames": torch.tensor([[0, 2]] * 3),
                "semantic_versions": torch.tensor([[1, 2]] * 3),
                "semantic_episode_generations": torch.zeros(3, 2, dtype=torch.long),
                "history_valid_lengths": torch.full((3,), 2),
                "task_ids": torch.tensor([0, 1, 0]),
            },
            "labels": {
                "progress": torch.tensor([0.2, 0.5, 0.8]),
                "completion": torch.tensor([0.0, 0.0, 1.0]),
                "failure": torch.tensor([0.0, 1.0, 0.0]),
            },
        },
        path,
    )

    dataset = SharedSemanticRewardDataset(str(path))
    features, labels = dataset[2]
    batched_features = {key: value.unsqueeze(0) for key, value in features.items()}
    batched_labels = {key: value.unsqueeze(0) for key, value in labels.items()}
    outputs = _model()(batched_features, batched_labels)

    assert len(dataset) == 3
    assert outputs["loss"].item() > 0
    outputs["loss"].backward()


def test_shared_semantic_rollout_dataset_uses_distinct_causal_packets(tmp_path):
    trajectory_dir = tmp_path / "task_00" / "success"
    trajectory_dir.mkdir(parents=True)
    trajectory_path = trajectory_dir / "trajectory_000000.npz"
    np.savez_compressed(
        trajectory_path,
        feature_semantic_tokens=np.random.randn(4, 3, 8).astype(np.float16),
        feature_semantic_attention_mask=np.ones((4, 3), dtype=bool),
        semantic_source_frame_id=np.array([0, 0, 2, 4], dtype=np.int64),
        semantic_version=np.array([1, 1, 2, 3], dtype=np.int64),
        semantic_episode_generation=np.zeros(4, dtype=np.int64),
        action_frame_id=np.array([0, 1, 3, 4], dtype=np.int64),
        packet_age_s=np.array([0.0, 0.05, 0.05, 0.0], dtype=np.float32),
        label_frame_success=np.array([False, False, False, True]),
        label_episode_success=np.asarray(True),
    )
    manifest_path = tmp_path / "train.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "rlinf-shared-semantic-rollout-v1",
                "data_root": str(tmp_path),
                "episodes": [
                    {
                        "path": str(trajectory_path.relative_to(tmp_path)),
                        "task_id": 0,
                        "episode_success": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = SharedSemanticRolloutDataset(
        str(manifest_path), history_size=2, samples_per_episode=2
    )
    positive_features, positive_label = dataset[0]
    negative_features, negative_label = dataset[1]

    assert len(dataset) == 2
    assert positive_label["completion"].item() == 1.0
    assert negative_label["completion"].item() == 0.0
    assert positive_features["semantic_versions"].tolist() == [2, 3]
    assert positive_features["semantic_age_frames"].tolist() == [1.0, 0.0]
    assert positive_features["history_valid_lengths"].item() == 2
