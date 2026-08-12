# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Temporal reward expert over cached GR00T semantic packets.

The expensive VLM is intentionally absent from this module. Online inference
accepts the exact token tensors emitted by the action rollout and stored in the
semantic history buffer. A private adapter and a small temporal expert turn
those shared tokens into progress, completion, and failure rewards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class RewardSemanticAdapter(nn.Module):
    """Project shared VLM tokens into a reward-private representation."""

    def __init__(
        self,
        token_dim: int,
        adapter_dim: int,
        age_hidden_dim: int,
        age_normalization_frames: float,
        age_normalization_s: float,
    ) -> None:
        super().__init__()
        self.age_normalization_frames = max(float(age_normalization_frames), 1.0)
        self.age_normalization_s = max(float(age_normalization_s), 1e-6)
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim),
        )
        self.token_score = nn.Linear(adapter_dim, 1, bias=False)
        self.age_adapter = nn.Sequential(
            nn.Linear(3, age_hidden_dim),
            nn.SiLU(),
            nn.Linear(age_hidden_dim, adapter_dim),
        )
        self.output_norm = nn.LayerNorm(adapter_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
        age_frames: torch.Tensor,
        age_s: torch.Tensor,
        interval_frames: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("semantic tokens must have shape [B,K,T,D]")
        adapted = self.token_adapter(tokens)
        scores = self.token_score(adapted).squeeze(-1)
        if attention_mask is None:
            attention_mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        empty_rows = ~attention_mask.any(dim=-1)
        if empty_rows.any():
            scores = scores.clone()
            scores[empty_rows, 0] = 0
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(adapted * weights.unsqueeze(-1), dim=-2)

        age_features = torch.stack(
            (
                age_frames / self.age_normalization_frames,
                age_s / self.age_normalization_s,
                interval_frames / self.age_normalization_frames,
            ),
            dim=-1,
        ).to(device=pooled.device, dtype=pooled.dtype)
        return self.output_norm(pooled + self.age_adapter(age_features))


class SharedSemanticTemporalRewardModel(BaseRewardModel):
    """Reward expert sharing the frozen GR00T N1.7 semantic stream.

    The model never accepts raw images during online inference. Repeated reads
    of the same control frame produce zero reward. A reused semantic packet is
    still scored when the current state, action history, or packet age advances.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.history_size = int(cfg.get("history_size", 4))
        if self.history_size < 2:
            raise ValueError("history_size must be at least 2")
        self.history_buffer_name = str(
            cfg.get("history_buffer_name", "semantic_window")
        )
        self.token_dim = int(cfg.get("token_dim", 2048))
        self.adapter_dim = int(cfg.get("adapter_dim", 256))
        self.temporal_dim = int(cfg.get("temporal_dim", 256))
        self.task_dim = int(cfg.get("task_dim", 64))
        self.num_tasks = int(cfg.get("num_tasks", 10))
        self.state_dim = int(cfg.get("state_dim", 132))
        self.action_history_length = int(cfg.get("action_history_length", 4))
        self.action_dim = int(cfg.get("action_dim", 132))
        self.num_embodiments = int(cfg.get("num_embodiments", 32))
        self.require_action_condition = bool(
            cfg.get("require_action_condition", False)
        )
        self.interval_reward = float(cfg.get("interval_reward", 0.0))
        self.progress_weight = float(cfg.get("progress_weight", 1.0))
        self.completion_weight = float(cfg.get("completion_weight", 1.0))
        self.completion_pos_weight = float(cfg.get("completion_pos_weight", 1.0))
        self.failure_weight = float(cfg.get("failure_weight", 1.0))
        self.uncertainty_penalty = float(cfg.get("uncertainty_penalty", 0.0))
        self.uncertainty_loss_weight = float(
            cfg.get("uncertainty_loss_weight", 1.0)
        )
        self.completion_threshold = cfg.get("completion_threshold", None)

        self.semantic_adapter = RewardSemanticAdapter(
            token_dim=self.token_dim,
            adapter_dim=self.adapter_dim,
            age_hidden_dim=int(cfg.get("age_hidden_dim", 64)),
            age_normalization_frames=float(cfg.get("age_normalization_frames", 8.0)),
            age_normalization_s=float(cfg.get("age_normalization_s", 0.4)),
        )
        self.task_embedding = nn.Embedding(self.num_tasks, self.task_dim)
        self.state_adapter = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.adapter_dim),
            nn.GELU(),
            nn.Linear(self.adapter_dim, self.adapter_dim),
        )
        history_dim = self.action_history_length * self.action_dim
        self.action_history_adapter = nn.Sequential(
            nn.LayerNorm(history_dim),
            nn.Linear(history_dim, self.adapter_dim),
            nn.GELU(),
            nn.Linear(self.adapter_dim, self.adapter_dim),
        )
        self.embodiment_embedding = nn.Embedding(
            self.num_embodiments, self.adapter_dim
        )
        self.condition_norm = nn.LayerNorm(self.adapter_dim)
        self.frame_projector = nn.Sequential(
            nn.LayerNorm(self.adapter_dim + self.task_dim),
            nn.Linear(self.adapter_dim + self.task_dim, self.temporal_dim),
            nn.GELU(),
        )
        self.temporal_expert = nn.GRU(
            input_size=self.temporal_dim,
            hidden_size=self.temporal_dim,
            batch_first=True,
        )
        pair_dim = 4 * self.temporal_dim
        head_hidden = int(cfg.get("head_hidden_dim", 128))

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )

        self.progress_head = make_head()
        self.completion_head = make_head()
        self.failure_head = make_head()
        self.uncertainty_head = make_head()
        model_path = cfg.get("model_path", None)
        if bool(cfg.get("require_checkpoint", False)) and not model_path:
            raise ValueError("shared semantic reward requires reward.model.model_path")
        self._load_weights(model_path)

    @staticmethod
    def _resolve_checkpoint(path: Path) -> Path:
        if path.is_file():
            return path
        for candidate in (
            path / "actor" / "model_state_dict" / "full_weights.pt",
            path / "model_state_dict" / "full_weights.pt",
            path / "full_weights.pt",
            path / "best_model.pt",
        ):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No reward checkpoint found under {path}")

    def _load_weights(self, model_path: str | None) -> None:
        if not model_path:
            return
        state = torch.load(
            self._resolve_checkpoint(Path(model_path)),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(state, dict):
            state = state.get("model_state_dict", state.get("model", state))
        cleaned = {}
        for key, value in state.items():
            for prefix in ("module.", "_orig_mod.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            cleaned[key] = value
        self.load_state_dict(cleaned, strict=True)

    @staticmethod
    def _as_tensor(value: Any, *, device: torch.device, dtype=None) -> torch.Tensor:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        return tensor.to(device=device, dtype=dtype or tensor.dtype)

    @staticmethod
    def _pad_packet_sequence(
        sequence: list[Any],
        history_size: int,
        *,
        default: Any,
    ) -> tuple[list[Any], int]:
        selected = list(sequence[-history_size:])
        valid = len(selected)
        if not selected:
            selected = [default]
        while len(selected) < history_size:
            selected.insert(0, selected[0])
        return selected, valid

    def _stack_token_history(
        self,
        token_sequences: list[list[Any]],
        mask_sequences: list[list[Any]] | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_tokens: list[list[torch.Tensor]] = []
        selected_masks: list[list[torch.Tensor]] = []
        valid_lengths = []
        max_tokens = 1
        for env_id, sequence in enumerate(token_sequences):
            packets, valid = self._pad_packet_sequence(
                sequence,
                self.history_size,
                default=torch.zeros(1, self.token_dim),
            )
            masks = None if mask_sequences is None else mask_sequences[env_id]
            selected_mask_values, _ = self._pad_packet_sequence(
                masks or [],
                self.history_size,
                default=torch.ones(1, dtype=torch.bool),
            )
            packet_tensors = [torch.as_tensor(packet) for packet in packets]
            mask_tensors = [
                torch.as_tensor(mask).bool() for mask in selected_mask_values
            ]
            max_tokens = max(
                max_tokens, *(int(packet.shape[-2]) for packet in packet_tensors)
            )
            selected_tokens.append(packet_tensors)
            selected_masks.append(mask_tensors)
            valid_lengths.append(valid)

        token_batch = torch.zeros(
            len(selected_tokens),
            self.history_size,
            max_tokens,
            self.token_dim,
            device=device,
            dtype=next(self.parameters()).dtype,
        )
        mask_batch = torch.zeros(
            len(selected_tokens),
            self.history_size,
            max_tokens,
            device=device,
            dtype=torch.bool,
        )
        for env_id, (packets, masks) in enumerate(
            zip(selected_tokens, selected_masks, strict=True)
        ):
            for packet_id, (packet, mask) in enumerate(
                zip(packets, masks, strict=True)
            ):
                if packet.shape[-1] != self.token_dim:
                    raise ValueError(
                        f"semantic token dim {packet.shape[-1]} != {self.token_dim}"
                    )
                token_count = int(packet.shape[-2])
                token_batch[env_id, packet_id, :token_count] = packet.to(
                    device=device, dtype=token_batch.dtype
                )
                flat_mask = mask.reshape(-1)[:token_count].to(device=device)
                mask_batch[env_id, packet_id, : flat_mask.numel()] = flat_mask
        return token_batch, mask_batch, torch.tensor(valid_lengths, device=device)

    def _stack_scalar_history(
        self,
        sequences: list[list[Any]],
        device: torch.device,
        *,
        dtype: torch.dtype,
        default: float = 0.0,
    ) -> torch.Tensor:
        rows = []
        for sequence in sequences:
            values, _ = self._pad_packet_sequence(
                sequence, self.history_size, default=default
            )
            rows.append([torch.as_tensor(value).reshape(-1)[0] for value in values])
        return torch.stack([torch.stack(row) for row in rows]).to(
            device=device, dtype=dtype
        )

    def _stack_dense_history(
        self,
        sequences: list[list[Any]],
        device: torch.device,
        *,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        rows = []
        default = torch.zeros(shape)
        for sequence in sequences:
            values, _ = self._pad_packet_sequence(
                sequence, self.history_size, default=default
            )
            rows.append(torch.stack([torch.as_tensor(value) for value in values]))
        return torch.stack(rows).to(device=device, dtype=next(self.parameters()).dtype)

    def _extract_online_features(self, reward_input: dict[str, Any]) -> dict[str, Any]:
        history_input = reward_input.get("history_input")
        if not isinstance(history_input, dict):
            raise ValueError("shared semantic reward requires history_input")
        history = history_input.get(self.history_buffer_name)
        if not isinstance(history, dict):
            raise ValueError(
                f"missing semantic history buffer {self.history_buffer_name!r}"
            )
        token_sequences = history.get("semantic_tokens")
        if not isinstance(token_sequences, list):
            raise ValueError("semantic history is missing semantic_tokens")

        device = next(self.parameters()).device
        tokens, attention_mask, valid_lengths = self._stack_token_history(
            token_sequences,
            history.get("semantic_attention_mask"),
            device,
        )

        def scalars(key: str, dtype: torch.dtype, default: float = 0.0):
            values = history.get(key)
            if values is None:
                values = [[] for _ in token_sequences]
            return self._stack_scalar_history(
                values, device, dtype=dtype, default=default
            )

        source_frames = scalars("semantic_source_frame_ids", torch.float32)
        action_frames = scalars("action_frame_ids", torch.float32)
        packet_age_s = scalars("packet_age_s", torch.float32)
        versions = scalars("semantic_versions", torch.long, default=-1)
        generations = scalars("semantic_episode_generations", torch.long, default=-1)
        age_frames = (action_frames - source_frames).clamp_min(0.0)
        interval_frames = torch.cat(
            (
                torch.zeros_like(source_frames[:, :1]),
                (source_frames[:, 1:] - source_frames[:, :-1]).clamp_min(0.0),
            ),
            dim=1,
        )
        task_ids = reward_input.get("task_ids")
        if task_ids is None:
            task_ids = torch.zeros(len(token_sequences), dtype=torch.long)
        task_ids = self._as_tensor(task_ids, device=device, dtype=torch.long).reshape(
            -1
        )
        condition_keys = ("action_states", "action_history", "embodiment_ids")
        missing_condition = [key for key in condition_keys if key not in history]
        if self.require_action_condition and missing_condition:
            raise ValueError(
                "shared semantic reward is missing action-side history fields: "
                + ", ".join(missing_condition)
            )
        action_states = self._stack_dense_history(
            history.get("action_states", [[] for _ in token_sequences]),
            device,
            shape=(1, self.state_dim),
        )
        action_history = self._stack_dense_history(
            history.get("action_history", [[] for _ in token_sequences]),
            device,
            shape=(self.action_history_length, self.action_dim),
        )
        embodiment_ids = scalars("embodiment_ids", torch.long)
        return {
            "semantic_tokens": tokens,
            "semantic_attention_mask": attention_mask,
            "semantic_age_frames": age_frames,
            "semantic_age_s": packet_age_s,
            "semantic_interval_frames": interval_frames,
            "semantic_versions": versions,
            "semantic_episode_generations": generations,
            "history_valid_lengths": valid_lengths,
            "task_ids": task_ids,
            "action_states": action_states,
            "action_history": action_history,
            "embodiment_ids": embodiment_ids,
        }

    def forward(
        self,
        input_data: dict[str, Any],
        labels: Optional[torch.Tensor | dict[str, torch.Tensor]] = None,
    ) -> dict[str, Any]:
        tokens = input_data["semantic_tokens"]
        device = next(self.parameters()).device
        tokens = self._as_tensor(
            tokens, device=device, dtype=next(self.parameters()).dtype
        )
        mask = input_data.get("semantic_attention_mask")
        if mask is not None:
            mask = self._as_tensor(mask, device=device, dtype=torch.bool)
        age_frames = self._as_tensor(
            input_data["semantic_age_frames"], device=device, dtype=tokens.dtype
        )
        age_s = self._as_tensor(
            input_data["semantic_age_s"], device=device, dtype=tokens.dtype
        )
        interval_frames = self._as_tensor(
            input_data["semantic_interval_frames"], device=device, dtype=tokens.dtype
        )
        generations = self._as_tensor(
            input_data["semantic_episode_generations"], device=device, dtype=torch.long
        )
        valid_lengths = self._as_tensor(
            input_data.get(
                "history_valid_lengths",
                torch.full((tokens.shape[0],), tokens.shape[1]),
            ),
            device=device,
            dtype=torch.long,
        )
        task_ids = self._as_tensor(
            input_data["task_ids"], device=device, dtype=torch.long
        ).reshape(-1)

        condition_keys = ("action_states", "action_history", "embodiment_ids")
        missing_condition = [key for key in condition_keys if key not in input_data]
        if self.require_action_condition and missing_condition:
            raise ValueError(
                "shared semantic reward is missing action-side conditions: "
                + ", ".join(missing_condition)
            )
        action_states = input_data.get("action_states")
        if action_states is None:
            action_states = tokens.new_zeros(
                tokens.shape[0], tokens.shape[1], 1, self.state_dim
            )
        action_states = self._as_tensor(
            action_states, device=device, dtype=tokens.dtype
        ).reshape(tokens.shape[0], tokens.shape[1], -1)
        if action_states.shape[-1] != self.state_dim:
            raise ValueError(
                f"action state dim {action_states.shape[-1]} != {self.state_dim}"
            )
        action_history = input_data.get("action_history")
        if action_history is None:
            action_history = tokens.new_zeros(
                tokens.shape[0],
                tokens.shape[1],
                self.action_history_length,
                self.action_dim,
            )
        action_history = self._as_tensor(
            action_history, device=device, dtype=tokens.dtype
        ).reshape(tokens.shape[0], tokens.shape[1], -1)
        expected_history_dim = self.action_history_length * self.action_dim
        if action_history.shape[-1] != expected_history_dim:
            raise ValueError(
                f"action history dim {action_history.shape[-1]} "
                f"!= {expected_history_dim}"
            )
        embodiment_ids = input_data.get("embodiment_ids")
        if embodiment_ids is None:
            embodiment_ids = torch.zeros(
                tokens.shape[:2], device=device, dtype=torch.long
            )
        embodiment_ids = self._as_tensor(
            embodiment_ids, device=device, dtype=torch.long
        ).reshape(tokens.shape[0], tokens.shape[1])

        adapted = self.semantic_adapter(
            tokens, mask, age_frames, age_s, interval_frames
        )
        adapted = self.condition_norm(
            adapted
            + self.state_adapter(action_states)
            + self.action_history_adapter(action_history)
            + self.embodiment_embedding(embodiment_ids)
        )
        task_features = (
            self.task_embedding(task_ids).unsqueeze(1).expand(-1, adapted.shape[1], -1)
        )
        frames = self.frame_projector(torch.cat((adapted, task_features), dim=-1))
        temporal, _ = self.temporal_expert(frames)
        previous = frames[:, -2]
        current = frames[:, -1]
        delta = current - previous
        pair = torch.cat((previous, current, delta, temporal[:, -1]), dim=-1)

        progress = torch.tanh(self.progress_head(pair).squeeze(-1))
        completion_logits = self.completion_head(pair).squeeze(-1)
        failure_logits = self.failure_head(pair).squeeze(-1)
        completion = torch.sigmoid(completion_logits)
        failure = torch.sigmoid(failure_logits)
        uncertainty = F.softplus(self.uncertainty_head(pair).squeeze(-1))
        control_advanced = (interval_frames[:, -1] > 0) | (
            (age_frames[:, -1] - age_frames[:, -2]).abs() > 1e-6
        )
        valid_transition = (
            (valid_lengths >= 2)
            & (generations[:, -1] == generations[:, -2])
            & control_advanced
        )
        rewards = (
            self.progress_weight * progress
            + self.completion_weight * completion
            - self.failure_weight * failure
            - self.uncertainty_penalty * uncertainty
        )
        rewards = torch.where(
            valid_transition,
            rewards,
            rewards.new_full(rewards.shape, self.interval_reward),
        )
        if self.completion_threshold is not None:
            completion = (completion >= float(self.completion_threshold)).to(
                completion.dtype
            )

        loss = rewards.new_zeros(())
        accuracy = rewards.new_zeros(())
        if labels is not None:
            if isinstance(labels, dict):
                completion_target = labels.get("completion")
                progress_target = labels.get("progress")
                failure_target = labels.get("failure")
                uncertainty_target = labels.get("uncertainty")
            else:
                completion_target = labels
                progress_target = None
                failure_target = None
                uncertainty_target = None
            if completion_target is not None:
                completion_target = self._as_tensor(
                    completion_target, device=device, dtype=completion_logits.dtype
                ).reshape(-1)
                loss = loss + F.binary_cross_entropy_with_logits(
                    completion_logits,
                    completion_target,
                    pos_weight=completion_logits.new_tensor(self.completion_pos_weight),
                )
                accuracy = (
                    ((completion_logits >= 0) == completion_target.bool())
                    .float()
                    .mean()
                )
                if (
                    uncertainty_target is None
                    and self.uncertainty_loss_weight > 0
                ):
                    uncertainty_target = (
                        completion.detach() - completion_target
                    ).abs()
            if progress_target is not None:
                progress_target = self._as_tensor(
                    progress_target, device=device, dtype=progress.dtype
                ).reshape(-1)
                loss = loss + F.smooth_l1_loss(progress, progress_target)
            if failure_target is not None:
                failure_target = self._as_tensor(
                    failure_target, device=device, dtype=failure_logits.dtype
                ).reshape(-1)
                loss = loss + F.binary_cross_entropy_with_logits(
                    failure_logits, failure_target
                )
            if uncertainty_target is not None:
                uncertainty_target = self._as_tensor(
                    uncertainty_target, device=device, dtype=uncertainty.dtype
                ).reshape(-1)
                loss = loss + self.uncertainty_loss_weight * F.smooth_l1_loss(
                    uncertainty, uncertainty_target
                )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "rewards": rewards,
            "progress": progress,
            "completion": completion,
            "failure": failure,
            "uncertainty": uncertainty,
            "valid_transition": valid_transition,
        }

    @torch.no_grad()
    def compute_reward(self, observations: dict[str, Any]) -> torch.Tensor:
        outputs = self(self._extract_online_features(observations))
        return outputs["rewards"]
