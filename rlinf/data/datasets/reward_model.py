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

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class RewardDatasetPayload:
    """Canonical payload schema for processed reward dataset files."""

    images: list[torch.Tensor]
    labels: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.images) != len(self.labels):
            raise ValueError("Images and labels must have same length")
        self.labels = [int(v) for v in self.labels]

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str = "<memory>"):
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid processed dataset payload from {source}")
        return cls(
            images=payload.get("images", []),
            labels=payload.get("labels", []),
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_dict(payload, source=path)


class RewardBinaryDataset(Dataset):
    """Dataset for binary classification reward model training.

    Uses per-frame 'is_obj_placed' field from infos to determine success/fail labels.
    This is more accurate than using episode-level labels from filenames.
    """

    def __init__(
        self,
        data_path: str,
    ):
        """Initialize dataset from a preprocessed .pt file.

        Args:
            data_path: Path to preprocessed dataset .pt file.

        Required payload schema is defined by `RewardDatasetPayload`.
        """
        payload = RewardDatasetPayload.load(data_path)
        self.images = payload.images
        self.labels = payload.labels
        self.metadata = payload.metadata

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get (image, label) pair.

        Returns:
            Tuple of (image tensor (C, H, W), label (0 or 1))
        """
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


class SharedSemanticRewardDataset(Dataset):
    """Dataset of frozen semantic packet windows for reward-adapter SFT.

    Each ``.pt`` payload contains a tensor dictionary named ``features`` and a
    tensor (or tensor dictionary) named ``labels``. The leading dimension is
    the sample dimension; the remaining feature dimensions are already padded
    to ``[history, tokens, hidden]`` as needed.
    """

    REQUIRED_FEATURES = {
        "semantic_tokens",
        "semantic_age_frames",
        "semantic_age_s",
        "semantic_interval_frames",
        "semantic_versions",
        "semantic_episode_generations",
        "task_ids",
    }

    def __init__(self, data_path: str | list[str]):
        paths = [data_path] if isinstance(data_path, str) else list(data_path)
        if not paths:
            raise ValueError("Shared semantic reward dataset paths cannot be empty")

        feature_parts: dict[str, list[torch.Tensor]] = {}
        label_parts: dict[str, list[torch.Tensor]] = {}
        tensor_label_parts: list[torch.Tensor] = []
        labels_are_dict: bool | None = None
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("features"), dict
            ):
                raise ValueError(f"Invalid shared semantic reward payload: {path}")
            features = payload["features"]
            missing = self.REQUIRED_FEATURES - set(features)
            if missing:
                raise ValueError(f"Missing features in {path}: {sorted(missing)}")
            sample_count = int(features["semantic_tokens"].shape[0])
            for key, value in features.items():
                tensor = torch.as_tensor(value)
                if tensor.shape[0] != sample_count:
                    raise ValueError(
                        f"Feature {key!r} has inconsistent length in {path}"
                    )
                feature_parts.setdefault(key, []).append(tensor)

            labels = payload.get("labels")
            current_labels_are_dict = isinstance(labels, dict)
            if labels_are_dict is None:
                labels_are_dict = current_labels_are_dict
            elif labels_are_dict != current_labels_are_dict:
                raise ValueError(
                    "All semantic reward payloads must use one label schema"
                )
            if current_labels_are_dict:
                for key, value in labels.items():
                    tensor = torch.as_tensor(value)
                    if tensor.shape[0] != sample_count:
                        raise ValueError(
                            f"Label {key!r} has inconsistent length in {path}"
                        )
                    label_parts.setdefault(key, []).append(tensor)
            else:
                tensor = torch.as_tensor(labels)
                if tensor.shape[0] != sample_count:
                    raise ValueError(f"Labels have inconsistent length in {path}")
                tensor_label_parts.append(tensor)

        self.features = {
            key: torch.cat(parts, dim=0) for key, parts in feature_parts.items()
        }
        self.labels: torch.Tensor | dict[str, torch.Tensor]
        if labels_are_dict:
            self.labels = {
                key: torch.cat(parts, dim=0) for key, parts in label_parts.items()
            }
        else:
            self.labels = torch.cat(tensor_label_parts, dim=0)

    def __len__(self) -> int:
        return int(self.features["semantic_tokens"].shape[0])

    def __getitem__(self, idx: int):
        features = {key: value[idx] for key, value in self.features.items()}
        if isinstance(self.labels, dict):
            labels = {key: value[idx] for key, value in self.labels.items()}
        else:
            labels = self.labels[idx]
        return features, labels


class SharedSemanticRolloutDataset(Dataset):
    """Lazy temporal-success windows over N1.7 rollout trajectory files."""

    SCHEMA_VERSION = "rlinf-shared-semantic-rollout-v1"

    def __init__(
        self,
        manifest_path: str,
        history_size: int = 4,
        samples_per_episode: int = 8,
        positive_samples_per_episode: int = 1,
        hard_negative_tail: int = 8,
        delay_min_frames: int = 0,
        delay_max_frames: int = 0,
        control_hz: float = 20.0,
        state_dim: int = 132,
        action_history_length: int = 4,
        action_dim: int = 132,
    ) -> None:
        manifest_file = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"Invalid shared-semantic manifest: {manifest_file}")
        self.data_root = Path(manifest["data_root"])
        self.episodes = list(manifest.get("episodes", []))
        self.history_size = int(history_size)
        self.samples_per_episode = int(samples_per_episode)
        self.positive_samples_per_episode = int(positive_samples_per_episode)
        self.hard_negative_tail = int(hard_negative_tail)
        self.delay_min_frames = int(delay_min_frames)
        self.delay_max_frames = int(delay_max_frames)
        self.control_hz = float(control_hz)
        self.state_dim = int(state_dim)
        self.action_history_length = int(action_history_length)
        self.action_dim = int(action_dim)
        if self.history_size < 2:
            raise ValueError("history_size must be at least 2")
        if self.samples_per_episode < 1:
            raise ValueError("samples_per_episode must be positive")
        if not 1 <= self.positive_samples_per_episode <= self.samples_per_episode:
            raise ValueError("positive_samples_per_episode must be in [1, samples_per_episode]")
        if not 0 <= self.delay_min_frames <= self.delay_max_frames:
            raise ValueError("invalid delay frame range")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")

    def __len__(self) -> int:
        return len(self.episodes) * self.samples_per_episode

    @staticmethod
    def _distinct_packet_indices(trajectory: Any) -> list[int]:
        identities = zip(
            trajectory["semantic_episode_generation"].tolist(),
            trajectory["semantic_version"].tolist(),
            trajectory["semantic_source_frame_id"].tolist(),
            strict=True,
        )
        result = []
        previous = None
        for index, identity in enumerate(identities):
            if identity != previous:
                result.append(index)
                previous = identity
        return result

    def _negative_indices(
        self, packet_indices: list[int], positive_index: int | None
    ) -> list[int]:
        candidates = [index for index in packet_indices if index != positive_index]
        if not candidates:
            return [packet_indices[0]]
        hard = candidates[-self.hard_negative_tail :]
        uniform_count = max(1, self.samples_per_episode - len(hard))
        uniform_positions = np.linspace(
            0, len(candidates) - 1, uniform_count, dtype=np.int64
        )
        selected = [candidates[int(position)] for position in uniform_positions]
        selected.extend(hard)
        return list(dict.fromkeys(selected))

    def _target_index(
        self,
        trajectory: Any,
        packet_indices: list[int],
        sample_slot: int,
        episode_success: bool,
    ) -> int:
        success_indices = np.flatnonzero(trajectory["label_frame_success"]).tolist()
        positive_index = int(success_indices[-1]) if success_indices else None
        if episode_success:
            if positive_index is None:
                raise ValueError("Successful trajectory has no positive frame")
            if positive_index not in packet_indices:
                raise ValueError("Positive frame is not a distinct semantic packet")
            if sample_slot == 0:
                return positive_index
            sample_slot -= 1
        negatives = self._negative_indices(packet_indices, positive_index)
        return negatives[sample_slot % len(negatives)]

    def _select_delay_pair(
        self,
        pairs: list[tuple[int, int]],
        sample_slot: int,
        episode_index: int,
    ) -> tuple[int, int]:
        """Select pairs with deterministic, balanced coverage of every delay."""
        desired_delay = self.delay_min_frames + (
            sample_slot
            % (self.delay_max_frames - self.delay_min_frames + 1)
        )
        matching = [pair for pair in pairs if pair[1] == desired_delay]
        candidates = matching or pairs
        return candidates[episode_index % len(candidates)]

    def __getitem__(self, idx: int):
        episode_index = idx // self.samples_per_episode
        episode = self.episodes[episode_index]
        sample_slot = idx % self.samples_per_episode
        path = self.data_root / episode["path"]
        with np.load(path, allow_pickle=False) as trajectory:
            packet_indices = self._distinct_packet_indices(trajectory)
            use_delay_pairs = self.delay_max_frames > 0
            if use_delay_pairs:
                required = {
                    "delay_action_state",
                    "delay_action_history",
                    "delay_action_frame_id",
                    "delay_valid_mask",
                    "label_delay_completion",
                    "feature_embodiment_id",
                }
                missing = required - set(trajectory.files)
                if missing:
                    raise ValueError(
                        f"{path} cannot provide exact delayed conditions: "
                        f"missing {sorted(missing)}"
                    )
                valid_pairs = [
                    (packet_index, delay)
                    for packet_index in packet_indices
                    for delay in range(self.delay_min_frames, self.delay_max_frames + 1)
                    if delay < trajectory["delay_valid_mask"].shape[1]
                    and bool(trajectory["delay_valid_mask"][packet_index, delay])
                ]
                positive_pairs = [
                    pair
                    for pair in valid_pairs
                    if bool(trajectory["label_delay_completion"][pair])
                ]
                positive_slots = (
                    self.positive_samples_per_episode
                    if bool(episode["episode_success"])
                    else 0
                )
                if sample_slot < positive_slots:
                    if not positive_pairs:
                        raise ValueError(f"{path} has no delayed completion pair")
                    target_index, target_delay = self._select_delay_pair(
                        positive_pairs,
                        sample_slot=sample_slot,
                        episode_index=episode_index,
                    )
                else:
                    negative_pairs = [
                        pair
                        for pair in valid_pairs
                        if not bool(trajectory["label_delay_completion"][pair])
                    ]
                    if not negative_pairs:
                        raise ValueError(f"{path} has no delayed negative pair")
                    negative_slot = sample_slot - positive_slots
                    target_index, target_delay = self._select_delay_pair(
                        negative_pairs,
                        sample_slot=negative_slot,
                        episode_index=episode_index,
                    )
            else:
                target_index = self._target_index(
                    trajectory,
                    packet_indices,
                    sample_slot,
                    bool(episode["episode_success"]),
                )
                target_delay = 0
            target_position = packet_indices.index(target_index)
            history = packet_indices[
                max(0, target_position - self.history_size + 1) : target_position + 1
            ]
            valid_length = len(history)
            history = [history[0]] * (self.history_size - valid_length) + history

            source_frames = trajectory["semantic_source_frame_id"][history]
            action_frames = trajectory["action_frame_id"][history].copy()
            if "feature_action_state" in trajectory.files:
                action_states = trajectory["feature_action_state"][history].copy()
            else:
                action_states = np.zeros(
                    (len(history), 1, self.state_dim), dtype=np.float32
                )
            if "feature_action_history" in trajectory.files:
                action_history = trajectory["feature_action_history"][history].copy()
            else:
                action_history = np.zeros(
                    (len(history), self.action_history_length, self.action_dim),
                    dtype=np.float32,
                )
            if "feature_embodiment_id" in trajectory.files:
                embodiment_ids = trajectory["feature_embodiment_id"][history].copy()
            else:
                embodiment_ids = np.zeros(len(history), dtype=np.int64)
            if use_delay_pairs:
                action_frames[-1] = trajectory["delay_action_frame_id"][
                    target_index, target_delay
                ]
                action_states[-1] = trajectory["delay_action_state"][
                    target_index, target_delay
                ]
                action_history[-1] = trajectory["delay_action_history"][
                    target_index, target_delay
                ]
            interval_frames = np.concatenate(
                (
                    np.zeros(1, dtype=np.float32),
                    np.maximum(np.diff(source_frames), 0).astype(np.float32),
                )
            )
            features = {
                "semantic_tokens": torch.from_numpy(
                    trajectory["feature_semantic_tokens"][history].astype(np.float32)
                ),
                "semantic_attention_mask": torch.from_numpy(
                    trajectory["feature_semantic_attention_mask"][history].astype(bool)
                ),
                "semantic_age_frames": torch.from_numpy(
                    np.maximum(action_frames - source_frames, 0).astype(np.float32)
                ),
                "semantic_age_s": torch.from_numpy(
                    trajectory["packet_age_s"][history].astype(np.float32)
                ),
                "action_states": torch.from_numpy(action_states.astype(np.float32)),
                "action_history": torch.from_numpy(action_history.astype(np.float32)),
                "embodiment_ids": torch.from_numpy(embodiment_ids.astype(np.int64)),
                "semantic_interval_frames": torch.from_numpy(interval_frames),
                "semantic_versions": torch.from_numpy(
                    trajectory["semantic_version"][history].astype(np.int64)
                ),
                "semantic_episode_generations": torch.from_numpy(
                    trajectory["semantic_episode_generation"][history].astype(np.int64)
                ),
                "history_valid_lengths": torch.tensor(valid_length, dtype=torch.long),
                "task_ids": torch.tensor(episode["task_id"], dtype=torch.long),
            }
            features["semantic_age_s"][-1] = (
                features["semantic_age_frames"][-1] / self.control_hz
            )
            labels = {
                "completion": torch.tensor(
                    bool(
                        trajectory["label_delay_completion"][target_index, target_delay]
                        if use_delay_pairs
                        else trajectory["label_frame_success"][target_index]
                    ),
                    dtype=torch.float32,
                )
            }
        return features, labels
