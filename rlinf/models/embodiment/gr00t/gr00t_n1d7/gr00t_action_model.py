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

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Literal, Optional, Union
from unittest.mock import patch

import numpy as np
import torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from torch import nn
from torch.distributions import Normal
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.gr00t.gr00t_n1d7.eval_noise import (
    eval_noise_seeds,
    eval_semantic_age_frames,
    stable_text_ids,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.semantic_server import (
    Gr00tN1d7AsyncSemanticBackboneClient,
    Gr00tN1d7SemanticBackboneClient,
    Gr00tN1d7SemanticCacheClient,
)
from rlinf.models.embodiment.gr00t.simulation_io import (
    ACTION_CONVERSION_N1D7,
    OBS_CONVERSION,
)
from rlinf.models.embodiment.gr00t.utils import (
    squeeze_dict_values,
    unsqueeze_dict_values,
)
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger

logger = get_logger()


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:16]


def _execution_action_prefix(
    actions: torch.Tensor, execution_horizon: int
) -> torch.Tensor:
    """Return exactly the action prefix represented by PPO and executed by the env."""
    if actions.ndim < 2:
        raise ValueError(
            f"Expected batched action chunks, got shape={tuple(actions.shape)}"
        )
    if execution_horizon < 1 or execution_horizon > actions.shape[1]:
        raise ValueError(
            "execution_horizon must be within the predicted action horizon: "
            f"{execution_horizon=} predicted_horizon={actions.shape[1]}"
        )
    return actions[:, :execution_horizon]


@contextmanager
def redirect_qwen3_backbone_to_local(canonical_name: str, local_path: str | None):
    if local_path is None:
        yield
        return

    local_path = Path(local_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"Backbone model path does not exist: {local_path}")

    original_model_from_pretrained = Qwen3VLForConditionalGeneration.from_pretrained
    original_processor_from_pretrained = Qwen3VLProcessor.from_pretrained

    def _make_local_redirect(original):
        def from_pretrained_with_local_redirect(model_name, *args, **kwargs):
            if str(model_name) == canonical_name:
                model_name = str(local_path)
                kwargs["local_files_only"] = True
            return original(model_name, *args, **kwargs)

        return from_pretrained_with_local_redirect

    # Redirect both the backbone weights and its processor (image processor +
    # tokenizer): GR00T's Gr00tN1d7DataCollator builds the processor via the
    # canonical hub name, which would otherwise hit the Hub and fail offline.
    with (
        patch.object(
            Qwen3VLForConditionalGeneration,
            "from_pretrained",
            side_effect=_make_local_redirect(original_model_from_pretrained),
        ),
        patch.object(
            Qwen3VLProcessor,
            "from_pretrained",
            side_effect=_make_local_redirect(original_processor_from_pretrained),
        ),
    ):
        yield


def _find_processor_dir(model_path: Path) -> Path | None:
    """Find the local directory containing GR00T N1.7 processor files."""
    processor_required_files = (
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    )
    for candidate in (model_path / "processor", model_path):
        if candidate.is_dir() and all(
            (candidate / f).is_file() for f in processor_required_files
        ):
            return candidate
    return None


# Keys produced during rollout that must never be replayed as backbone inputs by
# the actor (they are RL bookkeeping rather than model inputs).
_FORWARD_INPUT_SKIP_KEYS = {
    "advantages",
    "returns",
    "values",
    "prev_values",
    "prev_logprobs",
    "old_values",
    "loss_mask",
    "loss_mask_sum",
    "chains",
    "denoise_inds",
}

_FORWARD_INPUT_MODEL_KEYS = {
    "state",
    "state_mask",
    "packet_age",
    "packet_age_s",
    "action_history",
    "semantic_backbone_features",
    "semantic_backbone_attention_mask",
    "semantic_image_mask",
    "action",
    "action_mask",
    "embodiment_id",
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "image_sizes",
}


def _resolve_env_action_dim(action_dim: int | None, valid_action_dim: int) -> int:
    """Resolve the environment-facing action dim, clamped by ``valid_action_dim``."""
    env_action_dim = valid_action_dim if action_dim is None else int(action_dim)
    assert env_action_dim <= int(valid_action_dim), (
        f"Configured action_dim ({env_action_dim}) exceeds valid_action_dim "
        f"({valid_action_dim})."
    )
    return env_action_dim


def _resize_semantic_token_axis(
    outputs: BatchFeature, target_tokens: int
) -> BatchFeature:
    """Keep cached semantic tensors on one stable token axis for PPO replay."""
    features = outputs.get("backbone_features")
    if target_tokens <= 0 or not torch.is_tensor(features) or features.ndim < 2:
        return outputs
    source_tokens = int(features.shape[1])
    if source_tokens == target_tokens:
        return outputs

    resized = {}
    for key, value in dict(outputs).items():
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and int(value.shape[1]) == source_tokens
        ):
            if source_tokens > target_tokens:
                value = value[:, :target_tokens].contiguous()
            else:
                pad_shape = list(value.shape)
                pad_shape[1] = target_tokens - source_tokens
                value = torch.cat((value, value.new_zeros(pad_shape)), dim=1)
        resized[key] = value
    return BatchFeature(data=resized)


def _dit_tail_trainable_prefixes(
    action_head: nn.Module, last_n_blocks: int
) -> tuple[str, ...]:
    """Return an allowlist for tuning only the tail of the original DiT."""
    dit = getattr(action_head, "model", None)
    blocks = getattr(dit, "transformer_blocks", None)
    if blocks is None:
        raise ValueError(
            "DiT tail training requires action_head.model.transformer_blocks"
        )
    block_count = len(blocks)
    if not 1 <= last_n_blocks <= block_count:
        raise ValueError(
            f"dit_train_last_n_blocks must be in [1, {block_count}], got {last_n_blocks}"
        )
    first_block = block_count - last_n_blocks
    return (
        "action_head.model.timestep_encoder",
        *(
            f"action_head.model.transformer_blocks.{idx}"
            for idx in range(first_block, block_count)
        ),
        "action_head.model.norm_out",
        "action_head.model.proj_out_1",
        "action_head.model.proj_out_2",
    )


def _dit_cross_attention_trainable_prefixes(
    action_head: nn.Module,
    *,
    include_query: bool = False,
    last_n_blocks: int = 0,
) -> tuple[str, ...]:
    """Tune only the original DiT projections that consume semantic tokens."""
    dit = getattr(action_head, "model", None)
    blocks = getattr(dit, "transformer_blocks", None)
    if blocks is None:
        raise ValueError(
            "DiT cross-attention training requires action_head.model.transformer_blocks"
        )

    cross_attention_indices = [
        idx
        for idx, block in enumerate(blocks)
        if getattr(block, "cross_attention_dim", None) is not None
    ]
    if not cross_attention_indices:
        raise ValueError("DiT contains no cross-attention transformer blocks")
    if last_n_blocks < 0 or last_n_blocks > len(cross_attention_indices):
        raise ValueError(
            "dit_train_cross_attention_last_n_blocks must be in "
            f"[0, {len(cross_attention_indices)}], got {last_n_blocks}"
        )
    if last_n_blocks > 0:
        cross_attention_indices = cross_attention_indices[-last_n_blocks:]

    projections = (
        ("to_q", "to_k", "to_v", "to_out")
        if include_query
        else (
            "to_k",
            "to_v",
            "to_out",
        )
    )
    return (
        *(
            f"action_head.model.transformer_blocks.{idx}.attn1.{projection}"
            for idx in cross_attention_indices
            for projection in projections
        ),
        "action_head.model.proj_out_2",
    )


def _apply_delay_adapter_overrides(config: Any, rl_head_config: dict[str, Any]) -> None:
    """Add delay inputs when bootstrapping from a checkpoint without them."""
    if bool(rl_head_config.get("initialize_packet_age_adapter", False)):
        config.use_packet_age_embedding = True
    history_length = int(rl_head_config.get("initialize_action_history_length", 0))
    if history_length < 0:
        raise ValueError("initialize_action_history_length must be non-negative")
    if history_length > 0:
        config.action_history_length = history_length


def _resolve_gr00t_execution_mode(rl_head_config: dict[str, Any]) -> str:
    """Resolve and validate the single GR00T execution mode switch.

    The legacy configuration remains supported: semantic-server settings imply
    decoupled execution, otherwise the model is coupled and runs its local VLM.
    """
    explicit_mode = rl_head_config.get("execution_mode")
    if explicit_mode is None:
        return (
            "decoupled"
            if bool(rl_head_config.get("semantic_server_enabled", False))
            or bool(rl_head_config.get("drop_local_backbone", False))
            else "coupled"
        )

    mode = str(explicit_mode).strip().lower()
    if mode not in {"coupled", "decoupled"}:
        raise ValueError(
            f"execution_mode must be 'coupled' or 'decoupled', got {explicit_mode!r}"
        )

    semantic_enabled = bool(rl_head_config.get("semantic_server_enabled", False))
    local_backbone_dropped = bool(rl_head_config.get("drop_local_backbone", False))
    if mode == "coupled" and (semantic_enabled or local_backbone_dropped):
        raise ValueError(
            "coupled execution requires semantic_server_enabled=False and "
            "drop_local_backbone=False"
        )
    if mode == "decoupled" and not semantic_enabled:
        raise ValueError(
            "decoupled execution requires semantic_server_enabled=True"
        )
    return mode

def _stale_age_gate(
    age_s: torch.Tensor, control_hz: float, threshold_frames: float
) -> torch.Tensor:
    """Return a normalized gate that is exactly zero below the stale threshold."""
    threshold = max(float(threshold_frames), 1.0)
    age_frames = age_s * max(float(control_hz), 1e-6)
    return torch.clamp(
        torch.relu(age_frames - threshold) / threshold,
        max=8.0,
    )


def _semantic_publish_due(
    last_published: dict[int, tuple[int, int]],
    env_ids: list[int],
    episode_generations: list[int],
    frame_ids: list[int],
    interval_frames: int,
) -> bool:
    """Return whether a semantic batch is new or has reached its frame interval."""
    for env_id, generation, frame_id in zip(
        env_ids, episode_generations, frame_ids, strict=True
    ):
        previous = last_published.get(int(env_id))
        if previous is None or previous[0] != int(generation):
            return True
        if interval_frames > 0 and int(frame_id) - previous[1] >= interval_frames:
            return True
    return False


def _prepare_action_only_observation(
    processor: Gr00tN1d7Processor,
    observation: dict[str, Any],
    embodiment_tag: EmbodimentTag,
) -> BatchFeature:
    """Normalize only the state/action-head fields used by cached-semantic control."""
    tag_value = embodiment_tag.value
    modality_config = processor.modality_configs[tag_value]
    state_config = modality_config["state"]
    state_keys = state_config.modality_keys
    state_data = {key: observation[f"state.{key}"] for key in state_keys}
    exclude_state = processor.exclude_state or getattr(
        state_config, "exclude_state", False
    )
    if exclude_state:
        normalized_states = torch.cat(
            [torch.from_numpy(np.zeros_like(state_data[key])) for key in state_keys],
            dim=-1,
        )
    else:
        normalized = processor.state_action_processor.apply_state(
            state=state_data, embodiment_tag=tag_value
        )
        normalized_states = torch.cat(
            [torch.from_numpy(normalized[key]) for key in state_keys], dim=-1
        )

    if normalized_states.shape[-1] > processor.max_state_dim:
        raise ValueError(
            f"State dimension {normalized_states.shape[-1]} exceeds "
            f"max_state_dim {processor.max_state_dim}"
        )
    padding_shape = (
        *normalized_states.shape[:-1],
        processor.max_state_dim - normalized_states.shape[-1],
    )
    normalized_states = torch.cat(
        (
            normalized_states,
            torch.zeros(padding_shape, dtype=normalized_states.dtype),
        ),
        dim=-1,
    )
    batch_size = normalized_states.shape[0]
    action_horizon = len(modality_config["action"].delta_indices)
    if action_horizon > processor.max_action_horizon:
        raise ValueError(
            f"Action horizon {action_horizon} exceeds "
            f"max_action_horizon {processor.max_action_horizon}"
        )
    action_mask = torch.zeros(
        (batch_size, processor.max_action_horizon), dtype=torch.float32
    )
    action_mask[:, :action_horizon] = 1.0
    embodiment_id = torch.full(
        (batch_size,),
        processor.embodiment_id_mapping[tag_value],
        dtype=torch.int32,
    )
    return BatchFeature(
        data={
            "state": normalized_states,
            "embodiment_id": embodiment_id,
            "action_mask": action_mask,
        }
    )


def _reshape_forward_tensor(key: str, value: Any) -> Any:
    """Normalize rollout-stashed tensors back to backbone-friendly shapes."""
    if not torch.is_tensor(value):
        return value

    if key == "pixel_values" and value.ndim > 4:
        return value.reshape(-1, *value.shape[-3:])
    if key in {"image_grid_thw", "image_sizes"} and value.ndim > 2:
        return value.reshape(-1, value.shape[-1])
    return value


def _canonicalize_gr00t_text_forward_inputs(
    forward_inputs: dict[str, Any],
    padding_value: int,
) -> dict[str, Any]:
    """Right-pad ``input_ids`` and ``attention_mask`` to ``padding_value``."""
    canonicalized = dict(forward_inputs)

    for key in ("input_ids", "attention_mask"):
        tensor = canonicalized.get(key)
        if tensor is None:
            continue
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Expected GR00T text field '{key}' to be a tensor, "
                f"got {type(tensor).__name__}."
            )
        if tensor.ndim < 2:
            raise ValueError(
                f"Expected GR00T text field '{key}' to be at least 2D, "
                f"got shape {tuple(tensor.shape)}."
            )
        if padding_value > 0 and tensor.shape[-1] > padding_value:
            raise ValueError(
                f"GR00T text field '{key}' length {tensor.shape[-1]} exceeds "
                f"padding_value={padding_value}."
            )
        if padding_value > 0 and tensor.shape[-1] < padding_value:
            tensor = torch.nn.functional.pad(
                tensor,
                pad=(0, padding_value - tensor.shape[-1]),
                mode="constant",
                value=0,
            )
        canonicalized[key] = tensor

    return canonicalized


def _normalize_gr00t_forward_inputs(forward_inputs: dict[str, Any]) -> dict[str, Any]:
    """Convert cached actor ``forward_inputs`` back into backbone inputs.

    Drops RL bookkeeping keys, restores flattened visual shapes, and synthesizes
    a default ``state_mask`` when missing.
    """
    normalized_input = {}
    for key, value in forward_inputs.items():
        if key in _FORWARD_INPUT_SKIP_KEYS:
            continue
        if key not in _FORWARD_INPUT_MODEL_KEYS:
            continue
        normalized_input[key] = _reshape_forward_tensor(key, value)

    state = normalized_input.get("state")
    if "state_mask" not in normalized_input and torch.is_tensor(state):
        normalized_input["state_mask"] = torch.ones(
            state.shape[:-1], dtype=torch.bool, device=state.device
        )

    return {key: value for key, value in normalized_input.items() if value is not None}


def _batchify_gr00t_forward_input(
    key: str,
    value: Any,
    batch_size: int,
) -> Any:
    """Store rollout forward inputs with an explicit batch dimension.

    Some GR00T processor outputs, especially visual fields such as
    ``pixel_values`` and ``image_grid_thw``, are emitted in flattened
    backbone-friendly shapes like ``[num_patches, hidden]`` or
    ``[num_images, 3]``. Those shapes are correct for immediate inference, but
    once cached into trajectory buffers they get treated as if dim-0 were the
    env batch dimension and are later sliced incorrectly. We therefore restore a
    leading batch axis before stashing them, and flatten them back inside
    :func:`_normalize_gr00t_forward_inputs` when the actor consumes them.
    """
    if not torch.is_tensor(value) or batch_size <= 0:
        return value

    if key == "pixel_values" and value.ndim >= 2 and value.shape[0] != batch_size:
        if value.shape[0] % batch_size != 0:
            raise ValueError(
                f"{key} leading dim {value.shape[0]} is not divisible by batch size {batch_size}"
            )
        return value.reshape(batch_size, value.shape[0] // batch_size, *value.shape[1:])

    if key in {"image_grid_thw", "image_sizes"} and value.ndim >= 2:
        if value.shape[0] != batch_size:
            if value.shape[0] % batch_size != 0:
                raise ValueError(
                    f"{key} leading dim {value.shape[0]} is not divisible by batch size {batch_size}"
                )
            return value.reshape(
                batch_size, value.shape[0] // batch_size, value.shape[-1]
            )

    return value


def _tensorize_forward_input(value: Any) -> Any:
    """Convert list-valued cached inputs into tensors."""
    if not isinstance(value, list):
        return value
    if len(value) == 0:
        return torch.tensor(value)
    if torch.is_tensor(value[0]):
        return torch.stack(value)
    return torch.tensor(value)


class FlowMatchingActionHeadForRLActionPrediction(Gr00tN1d7ActionHead):
    """Flow-matching action head with RL extensions for GR00T N1.7.

    Extends the upstream :class:`Gr00tN1d7ActionHead` with:

    * stochastic (flow-SDE) denoising for exploration,
    * per-denoising-step Gaussian log-probabilities, and
    * an optional value head for actor-critic style training.
    """

    def __init__(
        self,
        config: Any,  # Gr00tN1d7Config
        rl_head_config: dict[str, Any],
        output_action_chunks: int = 1,
    ):
        super().__init__(config)
        self.config = config
        self.rl_config = rl_head_config
        # Only set defaults if not already specified in config.
        if "noise_method" not in self.rl_config:
            self.rl_config["noise_method"] = "flow_sde"
        if "noise_level" not in self.rl_config:
            self.rl_config["noise_level"] = 0.5
        if "noise_anneal" not in self.rl_config:
            self.rl_config["noise_anneal"] = False
        self.padding_value = rl_head_config.get("padding_value", 0)
        self.output_action_chunks = output_action_chunks
        # Keep the upstream diffusion/action-head width separate from the
        # environment-facing action width inferred from modality metadata.
        self.model_action_dim = getattr(
            config, "max_action_dim", getattr(config, "action_dim", 7)
        )
        self.valid_action_dim = self.model_action_dim
        self.env_action_dim = _resolve_env_action_dim(
            getattr(config, "action_dim", self.valid_action_dim),
            self.valid_action_dim,
        )
        self.action_chunk = output_action_chunks
        self.hidden_size = getattr(
            config, "hidden_size", getattr(self, "hidden_size", 1024)
        )
        self.action_horizon = getattr(
            config, "action_horizon", getattr(self, "action_horizon", 16)
        )
        self.num_timestep_buckets = getattr(
            config, "num_timestep_buckets", getattr(self, "num_timestep_buckets", 1000)
        )
        self.num_inference_timesteps = getattr(
            config,
            "num_inference_timesteps",
            getattr(self, "num_inference_timesteps", 4),
        )

        vlm_width = getattr(config, "backbone_embedding_dim", 2048)
        state_width = getattr(config, "input_embedding_dim", 1536)
        if self.rl_config.get("use_vlm_value", False):
            proj_width = vlm_width
        else:
            proj_width = vlm_width + state_width
        self.value_include_packet_age = bool(
            self.rl_config.get("value_include_packet_age", False)
        )
        value_input_dim = proj_width + (1 if self.value_include_packet_age else 0)

        if self.rl_config.get("add_value_head", False):
            self.value_head = ValueHead(
                input_dim=value_input_dim,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        if self.rl_config.get("noise_method") == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=self.hidden_size,
                out_dim=getattr(config, "max_action_dim", 7),
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn",
            )

        adapter_hidden = int(getattr(config, "delay_adapter_hidden_dim", 128))
        state_width = int(getattr(config, "input_embedding_dim", 1536))
        self.packet_age_normalization_ms = float(
            getattr(config, "packet_age_normalization_ms", 400.0)
        )
        if bool(getattr(config, "use_packet_age_embedding", False)):
            self.packet_age_adapter = nn.Sequential(
                nn.Linear(1, adapter_hidden),
                nn.SiLU(),
                nn.Linear(adapter_hidden, state_width),
            )
            if bool(self.rl_config.get("zero_init_new_delay_adapters", False)):
                torch.nn.init.zeros_(self.packet_age_adapter[-1].weight)
                torch.nn.init.zeros_(self.packet_age_adapter[-1].bias)
        else:
            self.packet_age_adapter = None
        self.action_history_length = int(getattr(config, "action_history_length", 0))
        if self.action_history_length > 0:
            self.action_history_adapter = nn.Sequential(
                nn.Linear(
                    self.action_history_length * self.model_action_dim, adapter_hidden
                ),
                nn.SiLU(),
                nn.Linear(adapter_hidden, state_width),
            )
            if bool(self.rl_config.get("zero_init_new_delay_adapters", False)):
                torch.nn.init.zeros_(self.action_history_adapter[-1].weight)
                torch.nn.init.zeros_(self.action_history_adapter[-1].bias)
        else:
            self.action_history_adapter = None
        self.stale_semantic_control_hz = float(
            self.rl_config.get("semantic_control_hz", 20.0)
        )
        self.stale_semantic_threshold_frames = float(
            self.rl_config.get("semantic_stale_adapter_threshold_frames", 8.0)
        )
        self.stale_semantic_context_width = int(vlm_width)
        if bool(self.rl_config.get("semantic_stale_adapter_enabled", False)):
            stale_condition_dim = 1 + self.action_history_length * self.model_action_dim
            stale_input_dim = stale_condition_dim + self.stale_semantic_context_width
            self.stale_semantic_adapter = nn.Sequential(
                nn.Linear(stale_input_dim, adapter_hidden),
                nn.SiLU(),
                nn.Linear(adapter_hidden, state_width),
            )
            torch.nn.init.zeros_(self.stale_semantic_adapter[-1].weight)
            torch.nn.init.zeros_(self.stale_semantic_adapter[-1].bias)
            if self.stale_semantic_token_adapter is None:
                self.stale_semantic_token_adapter = nn.Sequential(
                    nn.Linear(stale_input_dim, adapter_hidden),
                    nn.SiLU(),
                    nn.Linear(adapter_hidden, self.stale_semantic_context_width),
                )
                torch.nn.init.zeros_(self.stale_semantic_token_adapter[-1].weight)
                torch.nn.init.zeros_(self.stale_semantic_token_adapter[-1].bias)
        else:
            self.stale_semantic_adapter = None

    def _get_component(self, name: str):
        """Return a named submodule of the head, or ``None`` if absent."""
        return getattr(self, name, None)

    def _process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """Apply the optional VL layer-norm and self-attention refinements."""
        if not hasattr(backbone_output, "backbone_features"):
            return backbone_output

        backbone_features = backbone_output.backbone_features
        vlln = self._get_component("vlln")
        if vlln is not None:
            backbone_features = vlln(backbone_features)

        vl_self_attention = self._get_component("vl_self_attention")
        if vl_self_attention is not None:
            backbone_features = vl_self_attention(backbone_features)

        backbone_output.backbone_features = backbone_features
        return backbone_output

    def _apply_stale_semantic_token_correction(
        self, semantic_features: torch.Tensor, action_input: BatchFeature
    ) -> torch.Tensor:
        if self.stale_semantic_token_adapter is None:
            return semantic_features
        if semantic_features.ndim != 3:
            raise ValueError("Semantic token features must have shape [B, T, D]")
        batch_size, token_count, width = semantic_features.shape
        if width != self.stale_semantic_context_width:
            raise ValueError(
                f"Semantic token width {width} != {self.stale_semantic_context_width}"
            )
        age = action_input.get("packet_age_s", action_input.get("packet_age"))
        if age is None:
            age = torch.zeros(batch_size, device=semantic_features.device)
        age = age.to(
            device=semantic_features.device, dtype=semantic_features.dtype
        ).reshape(batch_size, 1)
        native_token_adapter = bool(
            getattr(
                getattr(self, "config", None), "use_stale_semantic_token_adapter", False
            )
        )
        if native_token_adapter:
            norm_s = max(self.packet_age_normalization_ms / 1000.0, 1e-6)
            condition_age = (age / norm_s).clamp(min=0.0)
            stale_gate = condition_age.clamp(max=1.0)
        else:
            stale_gate = _stale_age_gate(
                age,
                self.stale_semantic_control_hz,
                self.stale_semantic_threshold_frames,
            )
            condition_age = stale_gate
        history = action_input.get("action_history")
        if history is None:
            history = torch.zeros(
                batch_size,
                self.action_history_length,
                self.model_action_dim,
                device=semantic_features.device,
                dtype=semantic_features.dtype,
            )
        history = history.to(
            device=semantic_features.device, dtype=semantic_features.dtype
        ).reshape(batch_size, -1)
        condition = torch.cat((condition_age, history), dim=-1)
        condition = condition.unsqueeze(1).expand(-1, token_count, -1)
        correction_input = torch.cat((semantic_features, condition), dim=-1)
        residual = self.stale_semantic_token_adapter(correction_input)
        return semantic_features + stale_gate.unsqueeze(1) * residual

    def _encode_state_features(
        self,
        action_input: BatchFeature,
        embodiment_id: int | torch.Tensor,
        semantic_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the proprioceptive state into the action-head feature space."""
        state = action_input.state
        # Match the documented GR00T behavior when the environment emits only
        # the current state but the checkpoint expects a fixed history.
        if state.ndim == 2:
            state = state[:, None, :]
        current_T = state.shape[1]
        expected_T = self.config.state_history_length
        if current_T == 1 and expected_T > 1:
            state = state.expand(-1, expected_T, -1)
            current_T = expected_T
        if current_T != expected_T:
            raise ValueError(
                f"State time dimension {current_T} != "
                f"config.state_history_length {expected_T}"
            )
        if state.ndim >= 3:
            state = state.reshape(state.shape[0], 1, -1)

        state_encoder = self._get_component("state_encoder")
        if state_encoder is None:
            return state
        state_features = state_encoder(state, embodiment_id)
        batch_size = state_features.shape[0]
        age = action_input.get("packet_age_s", action_input.get("packet_age"))
        if age is None:
            age = torch.zeros(batch_size, device=state_features.device)
        age = age.to(device=state_features.device, dtype=state_features.dtype).reshape(
            batch_size, 1
        )
        if self.packet_age_adapter is not None:
            norm_s = max(self.packet_age_normalization_ms / 1000.0, 1e-6)
            state_features = state_features + self.packet_age_adapter(
                age / norm_s
            ).unsqueeze(1)

        history = action_input.get("action_history")
        if history is None:
            history = torch.zeros(
                batch_size,
                self.action_history_length,
                self.model_action_dim,
                device=state_features.device,
                dtype=state_features.dtype,
            )
        history = history.to(device=state_features.device, dtype=state_features.dtype)
        if self.action_history_adapter is not None:
            state_features = state_features + self.action_history_adapter(
                history.reshape(batch_size, -1)
            ).unsqueeze(1)
        if self.stale_residual_adapter is not None:
            norm_s = max(self.packet_age_normalization_ms / 1000.0, 1e-6)
            normalized_age = (age / norm_s).clamp(min=0.0)
            stale_inputs = torch.cat(
                (normalized_age, history.reshape(batch_size, -1)), dim=-1
            )
            stale_gate = normalized_age.clamp(max=1.0)
            stale_residual = self.stale_residual_adapter(stale_inputs)
            state_features = state_features + stale_gate.unsqueeze(
                1
            ) * stale_residual.unsqueeze(1)
        if self.stale_semantic_adapter is not None:
            stale_gate = _stale_age_gate(
                age,
                self.stale_semantic_control_hz,
                self.stale_semantic_threshold_frames,
            )
            if semantic_features is None:
                semantic_context = torch.zeros(
                    batch_size,
                    self.stale_semantic_context_width,
                    device=state_features.device,
                    dtype=state_features.dtype,
                )
            else:
                semantic_context = semantic_features
                if semantic_context.ndim == 3:
                    semantic_context = semantic_context.mean(dim=1)
                if semantic_context.ndim != 2:
                    raise ValueError(
                        "Semantic features must have shape [B, D] or [B, T, D]"
                    )
                if semantic_context.shape != (
                    batch_size,
                    self.stale_semantic_context_width,
                ):
                    raise ValueError(
                        "Pooled semantic feature shape "
                        f"{tuple(semantic_context.shape)} != "
                        f"({batch_size}, {self.stale_semantic_context_width})"
                    )
                semantic_context = semantic_context.to(
                    device=state_features.device, dtype=state_features.dtype
                )
            stale_inputs = torch.cat(
                [
                    stale_gate,
                    history.reshape(batch_size, -1),
                    semantic_context,
                ],
                dim=-1,
            )
            stale_residual = self.stale_semantic_adapter(stale_inputs)
            state_features = state_features + stale_gate.unsqueeze(
                1
            ) * stale_residual.unsqueeze(1)
        sampler_dt_adapter = getattr(self, "sampler_dt_adapter", None)
        if sampler_dt_adapter is not None:
            reference_steps = int(getattr(self.config, "sampler_dt_reference_steps", 4))
            reference_dt = 1.0 / float(reference_steps)
            sampler_dt = 1.0 / float(self.num_inference_timesteps)
            relative_dt = torch.full(
                (batch_size, 1),
                (sampler_dt - reference_dt) / reference_dt,
                device=state_features.device,
                dtype=state_features.dtype,
            )
            sampler_residual = sampler_dt_adapter(relative_dt) * relative_dt.abs()
            state_features = state_features + sampler_residual.unsqueeze(1)
        return state_features

    def prepare_input(self, inputs: dict) -> BatchFeature:
        """Collect the action-head relevant fields into a ``BatchFeature``."""
        action_inputs = {}
        for k in [
            "state",
            "action",
            "action_mask",
            "embodiment_id",
            "packet_age_s",
            "packet_age",
            "action_history",
        ]:
            if k in inputs:
                action_inputs[k] = inputs[k]

        return BatchFeature(data=action_inputs)

    def get_logprob_norm(self, sample, mu, sigma):
        """Gaussian log-probability of ``sample`` under ``Normal(mu, sigma)``.

        Deterministic (``sigma == 0``) coordinates contribute zero log-prob.
        """
        if self.rl_config.get("safe_get_logprob", False):
            dist = Normal(loc=mu, scale=sigma)
            return dist.log_prob(sample)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
            return log_prob

    def sample_noise(self, shape, device, dtype=None):
        """Sample standard Gaussian exploration noise in bf16."""
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=dtype, device=device)

    def sample_mean_var_val(
        self,
        vl_embs: torch.Tensor,
        denoise_steps: int,
        x_t: torch.Tensor,
        embodiment_id: int,
        state_features: torch.Tensor,
        idx: Optional[int | torch.Tensor],
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
        backbone_output: Optional[BatchFeature] = None,
        use_compiled_denoising: bool = True,
    ):
        """Compute the mean and std of the posterior over the next denoising state.

        In ``eval`` mode the transition is deterministic (zero std). In ``train``
        mode with ``noise_method == "flow_sde"`` an SDE-consistent Gaussian
        perturbation is injected to enable exploration.
        """
        bsize = vl_embs.shape[0]
        device = vl_embs.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)

        if self.rl_config.get("noise_anneal"):
            noise_start, noise_end, anneal_steps = self.rl_config.get(
                "noise_params", (0.0, 0.0, 100)
            )
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(getattr(self, "global_step", 0), anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.rl_config.get("noise_level", 0.5)).to(
                device
            )

        t_cont = idx / float(denoise_steps)
        timesteps_tensor = (
            (t_cont * self.num_timestep_buckets).to(torch.int64).to(device)
        )
        action_encoder = self._get_component("action_encoder")
        action_features = (
            action_encoder(x_t, timesteps_tensor, embodiment_id)
            if action_encoder is not None
            else x_t
        )
        position_embedding = self._get_component("position_embedding")
        if (
            getattr(self.config, "add_pos_embed", False)
            and position_embedding is not None
        ):
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)
        sampler_dt = None
        reference_steps = int(getattr(self.config, "sampler_dt_reference_steps", 4))
        if denoise_steps != reference_steps:
            sampler_dt = torch.full(
                (bsize,),
                1.0 / float(denoise_steps),
                device=device,
                dtype=vl_embs.dtype,
            )

        denoising_model = self._get_component("model")
        denoising_forward = (
            getattr(self, "_compiled_denoising_forward", denoising_model)
            if use_compiled_denoising
            else denoising_model
        )
        if denoising_model is not None:
            if (
                getattr(self.config, "use_alternate_vl_dit", False)
                and backbone_output is not None
            ):
                model_output = denoising_forward(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    sampler_dt=sampler_dt,
                )
            else:
                model_output = denoising_forward(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                    sampler_dt=sampler_dt,
                )
        else:
            model_output = sa_embs
        model_output = model_output[:, -self.action_horizon :]

        action_decoder = self._get_component("action_decoder")
        v_t = (
            action_decoder(model_output, embodiment_id)
            if action_decoder is not None
            else torch.zeros_like(model_output)
        )

        timesteps = torch.linspace(
            0, 1, denoise_steps + 1, device=device, dtype=vl_embs.dtype
        )
        t_input = timesteps[idx]
        delta = timesteps[idx + 1] - timesteps[idx]
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)

        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input + delta)
            x1_weight = t_input + delta
            x_t_std = torch.zeros_like(t_input)
        else:
            if self.rl_config.get("noise_method") == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        (1 - timesteps)
                        / torch.where(timesteps == 0, timesteps[1], timesteps)
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = (
                    torch.ones_like(t_input)
                    - (t_input + delta)
                    - sigma_i**2 * delta / (2 * (1 - t_input))
                )
                x1_weight = t_input + delta
                x_t_std = torch.sqrt(delta) * sigma_i
            else:
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
                x_t_std = torch.zeros_like(t_input)

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def get_value(self, vl_embs, state_features, packet_age=None):
        """Estimate the state value from pooled VL and state features."""
        bsize = vl_embs.shape[0]
        mask_length = vl_embs.shape[1]
        if self.rl_config.get("value_vlm_mode") == "mean_token":
            prefix_mask = [True] * mask_length
        elif self.rl_config.get("value_vlm_mode") == "last_token":
            prefix_mask = [False] * (mask_length - 1) + [True] * 1
        elif self.rl_config.get("value_vlm_mode") == "first_token":
            prefix_mask = [True] * 1 + [False] * (mask_length - 1)
        vl_embs_value = vl_embs[:, prefix_mask, :].mean(dim=1, keepdim=False)
        state_features_value = state_features.reshape(bsize, -1)
        if self.rl_config.get("use_vlm_value", False):
            value_embs = vl_embs_value
        else:
            value_embs = torch.cat((vl_embs_value, state_features_value), dim=1)
        if self.value_include_packet_age:
            if packet_age is None:
                packet_age = torch.zeros(
                    bsize, device=value_embs.device, dtype=value_embs.dtype
                )
            packet_age = packet_age.to(
                device=value_embs.device, dtype=value_embs.dtype
            ).reshape(bsize, 1)
            age_norm_s = max(self.packet_age_normalization_ms / 1000.0, 1e-6)
            value_embs = torch.cat((value_embs, packet_age / age_norm_s), dim=1)
        if self.rl_config.get("detach_critic_input", False):
            value_embs = value_embs.detach()

        values_vlm = self.value_head(value_embs)[:, 0]
        return values_vlm

    def get_rl_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
    ) -> BatchFeature:
        """Sample an action chunk via stochastic denoising (rollout path).

        Returns the predicted action together with the full denoising ``chains``,
        per-step log-probabilities, value estimate and the sampled denoising
        indices, which the actor later replays in :meth:`forward`.
        """
        compute_values = compute_values and hasattr(self, "value_head")
        if hasattr(backbone_output, "backbone_features"):
            backbone_output = self._process_backbone_output(backbone_output)
        vl_embs = (
            backbone_output.backbone_features
            if hasattr(backbone_output, "backbone_features")
            else backbone_output
        )
        embodiment_id = (
            action_input.embodiment_id if hasattr(action_input, "embodiment_id") else 0
        )
        vl_embs = self._apply_stale_semantic_token_correction(vl_embs, action_input)
        state_features = self._encode_state_features(
            action_input, embodiment_id, vl_embs
        )
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.action_horizon, self.model_action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        chains = [x_t]
        log_probs = []

        if self.rl_config.get("joint_logprob"):
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            log_probs.append(initial_log_prob)

        num_steps = self.num_inference_timesteps
        if mode == "train":
            if self.rl_config.get("joint_logprob"):
                denoise_inds = torch.arange(num_steps, device=device)
            else:
                rand_idx = random.randint(0, num_steps - 1)
                denoise_inds = torch.full(
                    (num_steps,), rand_idx, dtype=torch.long, device=device
                )
        else:
            denoise_inds = torch.full((num_steps,), -1, dtype=torch.long, device=device)
        denoise_inds = denoise_inds.unsqueeze(0).repeat(batch_size, 1)

        for idx in range(num_steps):
            # Stochastic noise is injected only on the sampled denoising index;
            # all other steps follow the deterministic ("eval") transition.
            step_mode = "train" if idx == denoise_inds[0][idx] else "eval"
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=idx,
                x_t=x_t,
                embodiment_id=embodiment_id,
                state_features=state_features,
                mode=step_mode,
                denoise_steps=num_steps,
                compute_values=compute_values,
                backbone_output=backbone_output,
            )

            x_t = (
                x_t_mean
                + self.sample_noise(x_t.shape, device, dtype=x_t.dtype) * x_t_std
            )
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.action_chunk, : self.env_action_dim
        ]
        if compute_values:
            packet_age = action_input.get("packet_age_s", action_input.get("packet_age"))
            values = self.get_value(vl_embs, state_features, packet_age)
            values = values[:, None]
        else:
            values = torch.zeros((batch_size, 1), device=device, dtype=vl_embs.dtype)

        return BatchFeature(data={"action_pred": x_0}), {
            "actions": x_0,
            "action_pred": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def get_eval_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        initial_noise: torch.Tensor | None = None,
    ) -> BatchFeature:
        """Run deterministic denoising without allocating PPO rollout bookkeeping."""
        if hasattr(backbone_output, "backbone_features"):
            backbone_output = self._process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        vl_embs = self._apply_stale_semantic_token_correction(vl_embs, action_input)
        state_features = self._encode_state_features(
            action_input, embodiment_id, vl_embs
        )
        expected_noise_shape = (
            vl_embs.shape[0],
            self.action_horizon,
            self.model_action_dim,
        )
        if initial_noise is None:
            x_t = torch.randn(
                expected_noise_shape,
                dtype=vl_embs.dtype,
                device=vl_embs.device,
            )
        else:
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"initial_noise shape {tuple(initial_noise.shape)} does not match "
                    f"{expected_noise_shape}"
                )
            x_t = initial_noise.to(device=vl_embs.device, dtype=vl_embs.dtype)
        for idx in range(self.num_inference_timesteps):
            x_t, _ = self.sample_mean_var_val(
                vl_embs=vl_embs,
                denoise_steps=self.num_inference_timesteps,
                x_t=x_t,
                embodiment_id=embodiment_id,
                state_features=state_features,
                idx=idx,
                mode="eval",
                compute_values=False,
                backbone_output=backbone_output,
            )
        return BatchFeature(data={"action_pred": x_t})

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        chains,
        denoise_inds,
        compute_values=True,
    ):
        """Recompute log-probabilities and values for cached denoising chains."""
        compute_values = compute_values and hasattr(self, "value_head")
        if hasattr(backbone_output, "backbone_features"):
            backbone_output = self._process_backbone_output(backbone_output)
        vl_embs = (
            backbone_output.backbone_features
            if hasattr(backbone_output, "backbone_features")
            else backbone_output
        )
        embodiment_id = (
            action_input.embodiment_id if hasattr(action_input, "embodiment_id") else 0
        )
        vl_embs = self._apply_stale_semantic_token_correction(vl_embs, action_input)
        state_features = self._encode_state_features(
            action_input, embodiment_id, vl_embs
        )
        batch_size = vl_embs.shape[0]

        chains_log_probs = []
        if self.rl_config.get("joint_logprob"):
            num_steps = getattr(self.config, "num_steps", 1)
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1

        denoise_inds = denoise_inds.to(chains.device)
        batch_indices = torch.arange(batch_size, device=chains.device)
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[batch_indices, denoise_ind]
            chains_next = chains[batch_indices, denoise_ind + 1]
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=denoise_ind,
                x_t=chains_pre,
                embodiment_id=embodiment_id,
                state_features=state_features,
                mode="train",
                denoise_steps=self.num_inference_timesteps,
                compute_values=compute_values,
                backbone_output=backbone_output,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(log_probs)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        if compute_values:
            packet_age = action_input.get("packet_age_s", action_input.get("packet_age"))
            chains_values = self.get_value(vl_embs, state_features, packet_age)
            chains_values = chains_values[:, None]
        else:
            chains_values = torch.zeros(
                (batch_size, 1), device=chains_log_probs.device, dtype=vl_embs.dtype
            )
        return chains_log_probs, chains_values


class _InputOnlyBackbone(torch.nn.Module):
    """Backbone-shaped input adapter that never constructs the VLM."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Local VLM backbone is disabled; use the semantic server")


class GR00T_N1_7_ForRLActionPrediction(Gr00tN1d7, BasePolicy):
    """GR00T N1.7 model for reinforcement-learning action prediction."""

    _no_split_modules = [
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionBlock",
        "BasicTransformerBlock",
    ]

    def __init__(
        self,
        config: Gr00tN1d7Config,
        rl_head_config: dict[str, Any],
        embodiment_tag: Union[str, EmbodimentTag],
        local_model_path: str,
        modality_config: Optional[Any] = None,
        modality_transform: Optional[Any] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: Optional[int] = None,
        obs_converter_type: str = "libero",
        output_action_chunks: int = 1,
        **kwargs,
    ):
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )

        backbone_model_path = kwargs.pop("backbone_model_path", None)
        if backbone_model_path is not None:
            backbone_model_path = str(Path(backbone_model_path).expanduser().resolve())
            if not Path(backbone_model_path).is_dir():
                raise FileNotFoundError(
                    f"Backbone model path does not exist: {backbone_model_path}"
                )
            loading_kwargs["local_files_only"] = True

        original_model_name = str(config.model_name)

        if backbone_model_path is not None:
            logger.info(
                "Loading backbone locally from %s with canonical model_name=%s",
                backbone_model_path,
                original_model_name,
            )
        else:
            logger.info("Loading backbone from HuggingFace: %s", original_model_name)

        for key in list(kwargs.keys()):
            if hasattr(config, key):
                setattr(config, key, kwargs.pop(key))
        if kwargs:
            logger.warning("Ignoring unexpected kwargs: %s", sorted(kwargs))

        _apply_delay_adapter_overrides(config, rl_head_config)
        if (
            bool(rl_head_config.get("initialize_packet_age_adapter", False))
            or int(rl_head_config.get("initialize_action_history_length", 0)) > 0
        ):
            logger.info(
                "Delay adapters enabled for checkpoint bootstrap: packet_age=%s "
                "action_history_length=%d zero_init=%s",
                bool(getattr(config, "use_packet_age_embedding", False)),
                int(getattr(config, "action_history_length", 0)),
                bool(rl_head_config.get("zero_init_new_delay_adapters", False)),
            )

        execution_mode = _resolve_gr00t_execution_mode(rl_head_config)
        logger.info("GR00T execution mode: %s", execution_mode)
        drop_local_backbone = bool(rl_head_config.get("drop_local_backbone", False))
        from gr00t.model.gr00t_n1d7 import gr00t_n1d7 as upstream_gr00t_n1d7

        original_get_backbone_cls = upstream_gr00t_n1d7.get_backbone_cls
        if drop_local_backbone:
            upstream_gr00t_n1d7.get_backbone_cls = lambda _config: _InputOnlyBackbone
            logger.info("Skipping local VLM construction for DiT-only worker")
        try:
            with redirect_qwen3_backbone_to_local(
                original_model_name, backbone_model_path
            ):
                super().__init__(config, transformers_loading_kwargs=loading_kwargs)
                self._modality_config, self._modality_transform = (
                    self._load_modality_processor(
                        modality_config=modality_config,
                        modality_transform=modality_transform,
                        local_model_path=local_model_path,
                        backbone_model_path=backbone_model_path,
                    )
                )
        finally:
            upstream_gr00t_n1d7.get_backbone_cls = original_get_backbone_cls

        self.padding_value = rl_head_config.get("padding_value", 0)
        self.model_path = Path(local_model_path)
        self.compute_dtype = compute_dtype
        self.output_action_chunks = output_action_chunks
        self._profile_rollout_phases = os.environ.get(
            "PROFILE_ROLLOUT_PHASES", "false"
        ).lower() in {"1", "true", "yes", "on"}
        self._rollout_profile_ms: dict[str, float] = {}
        self._rollout_profile_last: float | None = None
        self._last_semantic_fetch_s = 0.0

        self.action_head = FlowMatchingActionHeadForRLActionPrediction(
            config, rl_head_config, output_action_chunks
        )

        if denoising_steps is not None and hasattr(
            self.action_head, "num_inference_timesteps"
        ):
            self.action_head.num_inference_timesteps = denoising_steps

        self.obs_converter_type = obs_converter_type
        self.obs_convert_fn = OBS_CONVERSION[obs_converter_type]
        self.action_convert_fn = ACTION_CONVERSION_N1D7[obs_converter_type]
        exp_cfg_path = self.model_path / "experiment_cfg"
        self._load_metadata(exp_cfg_path)
        self.action_dim = _resolve_env_action_dim(
            getattr(config, "action_dim", self.valid_action_dim),
            self.valid_action_dim,
        )
        self.action_head.env_action_dim = self.action_dim
        self.action_head.valid_action_dim = self.valid_action_dim
        self._execution_mode = execution_mode

        self._semantic_enabled = bool(
            rl_head_config.get("semantic_server_enabled", False)
        )
        self._semantic_feature_tokens = max(
            0, int(rl_head_config.get("semantic_feature_tokens", 0))
        )
        self._semantic_non_blocking = bool(
            rl_head_config.get("semantic_server_non_blocking", True)
        )
        self._semantic_central_cache = bool(
            rl_head_config.get("semantic_server_central_cache", True)
        )
        self._semantic_boundary_publish = bool(
            rl_head_config.get("semantic_server_boundary_publish", True)
        )
        self._semantic_boundary_publish_interval = max(
            1, int(rl_head_config.get("semantic_boundary_publish_interval", 1))
        )
        self._semantic_env_bootstrap_publish = bool(
            rl_head_config.get("semantic_env_bootstrap_publish", False)
        )
        self._semantic_control_only_transform = bool(
            rl_head_config.get("semantic_control_only_transform", True)
        )
        self._semantic_publish_interval_frames = max(
            0, int(rl_head_config.get("semantic_publish_interval_frames", 0))
        )
        self._semantic_client = None
        if self._semantic_enabled:
            semantic_port = rl_head_config.get("semantic_server_port", 6666)
            client_kwargs = {
                "host": str(rl_head_config.get("semantic_server_host", "127.0.0.1")),
                "port": semantic_port,
                "timeout_ms": int(
                    rl_head_config.get("semantic_server_timeout_ms", 120000)
                ),
            }
            if self._semantic_central_cache:
                client_cls = Gr00tN1d7SemanticCacheClient
                publish_port = rl_head_config.get("semantic_server_publish_port")
                if publish_port is None:
                    semantic_ports = [
                        int(value.strip())
                        for value in str(semantic_port).split(",")
                        if value.strip()
                    ]
                    publish_port = ",".join(str(value + 1) for value in semantic_ports)
                client_kwargs["publish_port"] = publish_port
                client_kwargs["fetch_target_age_frames"] = int(
                    rl_head_config.get("semantic_fetch_target_age_frames", 0)
                )
                client_kwargs["fetch_max_wait_ms"] = float(
                    rl_head_config.get("semantic_fetch_max_wait_ms", 0.0)
                )
            else:
                client_cls = (
                    Gr00tN1d7AsyncSemanticBackboneClient
                    if self._semantic_non_blocking
                    else Gr00tN1d7SemanticBackboneClient
                )
            self._semantic_client = client_cls(**client_kwargs)
        self._semantic_cache = None
        self._semantic_source_wallclock_s = None
        self._semantic_source_frame = None
        self._semantic_frame = 0
        self._semantic_episode_generations = None
        self._semantic_last_episode_generations: dict[int, int] = {}
        self._semantic_last_published_frames: dict[int, tuple[int, int]] = {}
        self._semantic_age_mode = str(
            rl_head_config.get("semantic_age_mode", "wallclock")
        )
        self._semantic_control_hz = float(
            rl_head_config.get("semantic_control_hz", 20.0)
        )
        self._semantic_fetch_hard_max_age_frames = int(
            rl_head_config.get(
                "semantic_fetch_hard_max_age_frames",
                rl_head_config.get("semantic_fetch_target_age_frames", 0),
            )
        )
        self._semantic_eval_fixed_age_frames = int(
            rl_head_config.get("semantic_eval_fixed_age_frames", -1)
        )
        self._semantic_eval_random_age_min_frames = int(
            rl_head_config.get("semantic_eval_random_age_min_frames", -1)
        )
        self._semantic_eval_random_age_max_frames = int(
            rl_head_config.get("semantic_eval_random_age_max_frames", -1)
        )
        self._semantic_eval_random_age_seed = int(
            rl_head_config.get("semantic_eval_random_age_seed", 2026)
        )
        self._semantic_eval_fixed_age_max_wait_ms = float(
            rl_head_config.get("semantic_eval_fixed_age_max_wait_ms", 30000.0)
        )
        self._semantic_train_random_age_min_frames = int(
            rl_head_config.get("semantic_train_random_age_min_frames", -1)
        )
        self._semantic_train_random_age_max_frames = int(
            rl_head_config.get("semantic_train_random_age_max_frames", -1)
        )
        self._semantic_train_fixed_age_max_wait_ms = float(
            rl_head_config.get("semantic_train_fixed_age_max_wait_ms", 30000.0)
        )
        self._semantic_fetch_delay_fraction = max(
            0.0, float(rl_head_config.get("semantic_fetch_delay_fraction", 0.45))
        )
        self._semantic_fetch_delay_initial_s = max(
            0.0,
            float(rl_head_config.get("semantic_fetch_delay_initial_ms", 800.0))
            / 1000.0,
        )
        self._semantic_fetch_delay_min_s = max(
            0.0,
            float(rl_head_config.get("semantic_fetch_delay_min_ms", 100.0)) / 1000.0,
        )
        self._semantic_fetch_delay_max_s = max(
            self._semantic_fetch_delay_min_s,
            float(rl_head_config.get("semantic_fetch_delay_max_ms", 1500.0)) / 1000.0,
        )
        self._semantic_fetch_delay_ema_alpha = min(
            1.0,
            max(
                0.0,
                float(rl_head_config.get("semantic_fetch_delay_ema_alpha", 0.25)),
            ),
        )
        self._semantic_observation_interval_ema_s: float | None = None
        self._semantic_last_observation_wallclock_s: float | None = None
        self._semantic_last_request_delay_s = self._semantic_fetch_delay_initial_s
        self._global_step = 0
        self._semantic_requests = 0
        self._semantic_publishes = 0
        self._action_history = None
        self._action_history_by_env: dict[tuple[int, int], torch.Tensor] = {}
        self._current_action_history_keys: list[tuple[int, int]] = []
        self._rollout_reset_mask = None
        self._rollout_semantic_metadata: dict[str, torch.Tensor] = {}
        self._latest_semantic_metadata: dict[str, Any] = {}
        self._local_backbone_dropped = bool(
            rl_head_config.get("drop_local_backbone", False)
        )
        self._require_packet_age_input = bool(
            rl_head_config.get("require_packet_age_input", False)
        )
        self._semantic_zero_packet_age = bool(
            rl_head_config.get("semantic_zero_packet_age", False)
        )
        self._semantic_zero_action_history = bool(
            rl_head_config.get("semantic_zero_action_history", False)
        )
        if (
            self._require_packet_age_input
            and self.action_head.packet_age_adapter is None
        ):
            raise RuntimeError(
                "require_packet_age_input=True requires an SFT checkpoint with "
                "use_packet_age_embedding=True"
            )
        if self._local_backbone_dropped:
            if not self._semantic_enabled:
                raise ValueError(
                    "drop_local_backbone requires semantic_server_enabled=True"
                )
            if not isinstance(self.backbone, _InputOnlyBackbone):
                raise RuntimeError("DiT-only worker unexpectedly constructed a VLM")
            logger.info("DiT-only worker contains no local VLM parameters")

        if bool(rl_head_config.get("dit_only_train", self._semantic_enabled)):
            dit_train_last_n_blocks = int(
                rl_head_config.get("dit_train_last_n_blocks", 0)
            )
            dit_train_cross_attention_mode = str(
                rl_head_config.get("dit_train_cross_attention_mode", "none")
            ).lower()
            dit_train_cross_attention_last_n_blocks = int(
                rl_head_config.get("dit_train_cross_attention_last_n_blocks", 0)
            )
            semantic_adapter_only_train = bool(
                rl_head_config.get("semantic_adapter_only_train", False)
            )
            if dit_train_cross_attention_mode not in {"none", "kv_out", "qkv_out"}:
                raise ValueError(
                    "dit_train_cross_attention_mode must be one of "
                    "none, kv_out, or qkv_out"
                )
            if (
                dit_train_cross_attention_mode == "none"
                and dit_train_cross_attention_last_n_blocks != 0
            ):
                raise ValueError(
                    "dit_train_cross_attention_last_n_blocks requires "
                    "dit_train_cross_attention_mode"
                )
            if dit_train_last_n_blocks > 0 and dit_train_cross_attention_mode != "none":
                raise ValueError(
                    "dit_train_last_n_blocks and dit_train_cross_attention_mode "
                    "are mutually exclusive"
                )
            if semantic_adapter_only_train and (
                dit_train_last_n_blocks > 0 or dit_train_cross_attention_mode != "none"
            ):
                raise ValueError(
                    "semantic_adapter_only_train cannot be combined with DiT tail or "
                    "cross-attention training"
                )
            if bool(rl_head_config.get("semantic_stale_adapter_only_train", False)):
                trainable_prefixes = ("action_head.stale_semantic_token_adapter",)
            elif semantic_adapter_only_train:
                trainable_prefixes = (
                    "action_head.packet_age_adapter",
                    "action_head.action_history_adapter",
                    "action_head.stale_residual_adapter",
                )
            elif bool(rl_head_config.get("dit_core_and_adapters_only_train", False)):
                # Keep the pretrained state/action projections fixed. PPO still
                # updates the complete DiT plus delay adapters and value head.
                trainable_prefixes = (
                    "action_head.model",
                    "action_head.packet_age_adapter",
                    "action_head.action_history_adapter",
                    "action_head.sampler_dt_adapter",
                    "action_head.stale_residual_adapter",
                    "action_head.stale_semantic_token_adapter",
                    "action_head.value_head",
                )
            elif dit_train_cross_attention_mode != "none":
                trainable_prefixes = (
                    *_dit_cross_attention_trainable_prefixes(
                        self.action_head,
                        include_query=dit_train_cross_attention_mode == "qkv_out",
                        last_n_blocks=dit_train_cross_attention_last_n_blocks,
                    ),
                    "action_head.packet_age_adapter",
                    "action_head.action_history_adapter",
                    "action_head.stale_residual_adapter",
                )
                logger.info(
                    "DiT semantic cross-attention training enabled: mode=%s blocks=%d",
                    dit_train_cross_attention_mode,
                    dit_train_cross_attention_last_n_blocks
                    or sum(
                        block.cross_attention_dim is not None
                        for block in self.action_head.model.transformer_blocks
                    ),
                )
            elif dit_train_last_n_blocks > 0:
                trainable_prefixes = (
                    *_dit_tail_trainable_prefixes(
                        self.action_head, dit_train_last_n_blocks
                    ),
                    "action_head.packet_age_adapter",
                    "action_head.action_history_adapter",
                )
                logger.info(
                    "DiT tail training enabled: last %d/%d transformer blocks",
                    dit_train_last_n_blocks,
                    len(self.action_head.model.transformer_blocks),
                )
            else:
                trainable_prefixes = tuple(
                    rl_head_config.get(
                        "trainable_modules",
                        [
                            "action_head.model",
                            "action_head.state_encoder",
                            "action_head.action_encoder",
                            "action_head.action_decoder",
                            "action_head.packet_age_adapter",
                            "action_head.action_history_adapter",
                            "action_head.sampler_dt_adapter",
                            "action_head.model.sampler_dt_adapters",
                            "action_head.stale_residual_adapter",
                            "action_head.stale_semantic_token_adapter",
                            "action_head.stale_residual_adapter",
                            "action_head.value_head",
                        ],
                    )
                )
            if bool(rl_head_config.get("train_value_head_with_dit_only", False)):
                if getattr(self.action_head, "value_head", None) is None:
                    raise ValueError(
                        "train_value_head_with_dit_only=True requires add_value_head=True"
                    )
                trainable_prefixes = (*trainable_prefixes, "action_head.value_head")
            for name, parameter in self.named_parameters():
                parameter.requires_grad = any(
                    name.startswith(prefix) for prefix in trainable_prefixes
                )
            unexpected_trainable = [
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
                and not any(name.startswith(prefix) for prefix in trainable_prefixes)
            ]
            if unexpected_trainable:
                raise RuntimeError(
                    "DiT-only worker has trainable parameters outside the allowlist: "
                    f"{unexpected_trainable}"
                )
            trainable_parameter_count = sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
            if trainable_parameter_count == 0:
                raise RuntimeError("DiT-only worker has no trainable parameters")
            logger.info("DiT-only trainable prefixes: %s", trainable_prefixes)
            logger.info(
                "DiT-only trainable parameter count: %d", trainable_parameter_count
            )

        self._no_split_modules = (
            ["BasicTransformerBlock"]
            if self._local_backbone_dropped
            else self.__class__._no_split_modules
        )
        if hasattr(self, "config"):
            self.config.no_split_modules = self._no_split_modules
            self.config._no_split_modules = self._no_split_modules
        logger.info(
            "Forced FSDP _no_split_modules into config: %s",
            self.config.no_split_modules,
        )

    def _load_modality_processor(
        self,
        modality_config: Optional[Any],
        modality_transform: Optional[Any],
        local_model_path: str,
        backbone_model_path: Optional[str],
    ) -> tuple[Any, Any]:
        """Resolve the modality config and transform (processor)."""
        if modality_config is not None and modality_transform is not None:
            return modality_config, modality_transform

        processor_dir = _find_processor_dir(Path(local_model_path))
        if processor_dir is not None:
            logger.info("Loading processor from local dir: %s", processor_dir)
            modality_transform, modality_config = self._load_processor_from_dir(
                processor_dir,
                backbone_model_path=backbone_model_path,
            )
        else:
            from transformers import AutoProcessor

            logger.info(
                "Loading processor via AutoProcessor from: %s", local_model_path
            )
            processor = AutoProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=Path(local_model_path).is_dir(),
            )
            modality_transform = processor
            modality_config = getattr(
                processor,
                "modality_configs",
                getattr(processor, "modality_config", None),
            )

        return modality_config, modality_transform

    @staticmethod
    def _load_processor_from_dir(
        processor_dir: Path,
        *,
        backbone_model_path: Optional[str],
    ) -> tuple[Gr00tN1d7Processor, Any]:
        """Load the official GR00T N1.7 processor from a local directory."""
        with open(processor_dir / "processor_config.json", "r") as f:
            processor_cfg = json.load(f)["processor_kwargs"]
        with open(processor_dir / "statistics.json", "r") as f:
            processor_cfg["statistics"] = json.load(f)
        with open(processor_dir / "embodiment_id.json", "r") as f:
            processor_cfg["embodiment_id_mapping"] = json.load(f)
        if backbone_model_path is not None:
            processor_cfg.setdefault("transformers_loading_kwargs", {})
            processor_cfg["transformers_loading_kwargs"]["local_files_only"] = True
        modality_transform = Gr00tN1d7Processor(**processor_cfg)
        modality_config = getattr(modality_transform, "modality_configs", None)
        return modality_transform, modality_config

    def eval(self):
        self._modality_transform.eval()
        super().eval()

    def enable_torch_compile(self, mode: str = "max-autotune-no-cudagraphs") -> None:
        denoising_model = self.action_head._get_component("model")
        if denoising_model is None:
            raise RuntimeError("GR00T action head has no denoising model to compile")
        self.action_head._compiled_denoising_forward = torch.compile(
            denoising_model.forward,
            mode=mode,
            fullgraph=False,
        )
        logger.info("Enabled torch.compile for GR00T DiT forward with mode=%s", mode)

    def _rollout_profile_mark(self, phase: str) -> None:
        if not self._profile_rollout_phases:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        if self._rollout_profile_last is not None:
            self._rollout_profile_ms[phase] = (
                now - self._rollout_profile_last
            ) * 1000.0
        self._rollout_profile_last = now

    @staticmethod
    def _check_state_is_batched(obs: dict[str, Any]) -> bool:
        """Return whether observation state tensors already carry a batch dim."""
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:
                return False
        return True

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Actor forward pass: recompute log-probs/values from cached rollouts."""
        normalized_input = _normalize_gr00t_forward_inputs(forward_inputs)
        semantic_keys = {
            key: key.removeprefix("semantic_")
            for key in normalized_input
            if key.startswith("semantic_")
        }
        if semantic_keys:
            backbone_outputs = BatchFeature(
                data={
                    target: normalized_input[source]
                    for source, target in semantic_keys.items()
                }
            )
            action_inputs = self.action_head.prepare_input(normalized_input)
        else:
            if self._local_backbone_dropped:
                raise RuntimeError(
                    "DiT-only actor replay requires semantic_* tensors in the PPO buffer"
                )
            normalized_input = _canonicalize_gr00t_text_forward_inputs(
                normalized_input, getattr(self, "padding_value", 0)
            )
            backbone_inputs, action_inputs = self.prepare_input(normalized_input)
            backbone_outputs = self.backbone(backbone_inputs)

        packet_age = action_inputs.get("packet_age_s", action_inputs.get("packet_age"))
        if self._require_packet_age_input:
            if packet_age is None:
                raise RuntimeError(
                    "DiT-only PPO replay is missing packet_age_s in forward_inputs"
                )
            if not torch.isfinite(packet_age).all():
                raise RuntimeError("DiT-only PPO replay contains non-finite packet age")

        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        log_probs, value_t = self.action_head(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            chains=chains,
            denoise_inds=denoise_inds,
            compute_values=compute_values,
        )

        log_probs = log_probs[
            :,
            :,
            : self.action_head.action_chunk,
            : self.valid_action_dim,
        ]
        if self.action_head.rl_config.get("joint_logprob"):
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = kwargs["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:, 0]
            prev_logprobs = kwargs["prev_logprobs"]
            prev_logprobs = prev_logprobs[
                torch.arange(bsize, device=prev_logprobs.device),
                denoise_inds[:, 0].to(device=prev_logprobs.device),
                : self.action_head.action_chunk,
                : self.valid_action_dim,
            ]
        value_t = value_t.mean(dim=-1, keepdim=False)

        env_action_dim = self.action_dim
        log_probs = log_probs[..., :env_action_dim]
        prev_logprobs = prev_logprobs[..., :env_action_dim]

        result = {
            "logprobs": log_probs.float(),
            "prev_logprobs": prev_logprobs.float(),
            "values": value_t,
            "entropy": None,
        }
        if packet_age is not None:
            result["packet_age_mean_s"] = packet_age.detach().float().mean()
        return result

    def set_global_step(self, global_step: int) -> None:
        """Keep semantic scheduling state aligned across FSDP ranks."""
        self._global_step = int(global_step)

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ):
        """Rollout entry point: produce env-ready actions and RL bookkeeping."""
        del kwargs
        self._rollout_profile_ms = {}
        self._rollout_profile_last = None
        self._rollout_profile_mark("start")
        env_obs = dict(env_obs)
        metadata_keys = {
            "env_ids": "__rlinf_semantic_env_ids",
            "frame_ids": "__rlinf_semantic_frame_ids",
            "episode_generations": "__rlinf_semantic_generations",
            "observation_wallclock_s": "__rlinf_semantic_observation_wallclock_s",
            "task_ids": "__rlinf_task_ids",
            "trial_ids": "__rlinf_trial_ids",
            "target_age_frames": "__rlinf_semantic_target_age_frames",
        }
        self._rollout_semantic_metadata = {
            name: torch.as_tensor(env_obs.pop(key)).reshape(-1).cpu()
            for name, key in metadata_keys.items()
            if key in env_obs
        }
        if "task_ids" not in self._rollout_semantic_metadata:
            task_descriptions = env_obs.get("task_descriptions")
            if task_descriptions is not None:
                self._rollout_semantic_metadata["task_ids"] = stable_text_ids(
                    list(task_descriptions)
                )
        if "trial_ids" not in self._rollout_semantic_metadata:
            env_ids = self._rollout_semantic_metadata.get("env_ids")
            if env_ids is not None:
                self._rollout_semantic_metadata["trial_ids"] = env_ids.clone()
        elapsed = env_obs.get("elapsed_steps")
        if elapsed is not None:
            elapsed = torch.as_tensor(elapsed).reshape(-1)
            self._rollout_reset_mask = elapsed <= 0
        else:
            batch_size = int(torch.as_tensor(env_obs["states"]).shape[0])
            self._rollout_reset_mask = torch.zeros(batch_size, dtype=torch.bool)
        self._rollout_profile_mark("metadata")
        observations, obs_copy, is_batch = self._prepare_rollout_observation(env_obs)
        self._rollout_profile_mark("observation_prepare")
        normalized_action, result = self._predict_normalized_action(obs_copy, mode)
        unnormalized_action = self._get_unnormalized_action(
            normalized_action,
            state=observations,
        )
        self._rollout_profile_mark("decode")

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        raw_action = self.action_convert_fn(
            unnormalized_action,
            chunk_size=self.output_action_chunks,
        )
        raw_action = self._apply_exploration_noise(raw_action, mode)
        self._rollout_profile_mark("action_convert")
        if self._profile_rollout_phases:
            logger.info("Rollout phase ms: %s", self._rollout_profile_ms)
        return raw_action, result

    @staticmethod
    def _coerce_observation_values_to_numpy(
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Ensure every observation value is a numpy array."""
        coerced = {}
        for key, value in observation.items():
            coerced[key] = value if isinstance(value, np.ndarray) else np.array(value)
        return coerced

    @staticmethod
    def _cast_float_tensors_to_compute_dtype(
        normalized_input: dict[str, Any],
        compute_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Cast float32 tensors to the model compute dtype, leaving others intact."""
        casted = {}
        for key, value in normalized_input.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                casted[key] = value.to(compute_dtype)
            else:
                casted[key] = value
        return casted

    def _prepare_rollout_observation(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Convert raw env observations into batched GR00T processor inputs."""
        env_obs = dict(env_obs)
        # Here we have a source causing tiny inference-training inconsistency,
        # force convert the state to bf16 then back to float32 to reproduce the info loss in training.
        env_obs["states"] = env_obs["states"].to(torch.bfloat16)
        env_obs["states"] = env_obs["states"].cpu().float()

        observations = self.obs_convert_fn(env_obs)
        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)
        obs_copy = self._coerce_observation_values_to_numpy(obs_copy)
        return observations, obs_copy, is_batch

    def _predict_normalized_action(
        self,
        obs_copy: dict[str, Any],
        mode: Literal["train", "eval"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the policy and return normalized actions plus RL bookkeeping."""
        if (
            self._semantic_control_only_transform
            and not self._semantic_requires_publish_inputs()
        ):
            normalized_input = _prepare_action_only_observation(
                self._modality_transform, obs_copy, self.embodiment_tag
            )
        else:
            normalized_input = self.apply_transforms(obs_copy)
        self._rollout_profile_mark("transform")
        normalized_input = self._cast_float_tensors_to_compute_dtype(
            normalized_input,
            self.compute_dtype,
        )
        normalized_input = _canonicalize_gr00t_text_forward_inputs(
            normalized_input,
            getattr(self, "padding_value", 0),
        )
        self._rollout_profile_mark("canonicalize")

        if mode == "eval":
            normalized_action = self._get_deterministic_eval_action(normalized_input)
            result = {
                "prev_logprobs": None,
                "prev_values": None,
                "forward_inputs": {},
            }
        else:
            normalized_action, result = self._get_rl_action(
                normalized_input,
                mode=mode,
            )
        return normalized_action, result

    def _semantic_requires_publish_inputs(self) -> bool:
        if not (self._semantic_enabled and self._semantic_central_cache):
            return True
        metadata = self._rollout_semantic_metadata
        required = ("env_ids", "frame_ids", "episode_generations")
        if any(key not in metadata for key in required):
            return True
        env_ids = metadata["env_ids"].tolist()
        generations = metadata["episode_generations"].tolist()
        frame_ids = metadata["frame_ids"].tolist()
        latest_metadata = getattr(self, "_latest_semantic_metadata", {})
        cached_env_ids = latest_metadata.get("env_ids", ())
        cached_generations = latest_metadata.get("episode_generations", ())
        cache_identity_matches = (
            getattr(self, "_semantic_cache", None) is not None
            and len(cached_env_ids) == len(env_ids)
            and all(
                int(cached_env_id) == int(env_id)
                for cached_env_id, env_id in zip(cached_env_ids, env_ids, strict=True)
            )
            and len(cached_generations) == len(generations)
            and all(
                int(cached_generation) == int(generation)
                for cached_generation, generation in zip(
                    cached_generations, generations, strict=True
                )
            )
        )
        if not cache_identity_matches:
            # Identity recovery can synchronously republish this transformed
            # observation. Keep the complete VLM inputs until the expected
            # env/generation batch has reached the local cache.
            return True
        generation_changed = any(
            self._semantic_last_episode_generations.get(env_id) != generation
            for env_id, generation in zip(env_ids, generations, strict=True)
        )
        interval_due = (
            self._semantic_publish_interval_frames > 0
            and all(
                int(env_id) in self._semantic_last_published_frames
                for env_id in env_ids
            )
            and _semantic_publish_due(
                self._semantic_last_published_frames,
                env_ids,
                generations,
                frame_ids,
                self._semantic_publish_interval_frames,
            )
        )
        return (
            self._semantic_boundary_publish_due()
            or (generation_changed and not self._semantic_env_bootstrap_publish)
            or interval_due
        )

    def _semantic_boundary_publish_due(self) -> bool:
        return (
            self._semantic_boundary_publish
            and self._global_step % self._semantic_boundary_publish_interval == 0
        )

    def _apply_exploration_noise(
        self,
        raw_action: np.ndarray | torch.Tensor,
        mode: Literal["train", "eval"],
    ) -> np.ndarray | torch.Tensor:
        """Optionally perturb actions with clipped Gaussian noise during training."""
        if mode != "train":
            return raw_action

        noise_scale = float(self.action_head.rl_config.get("action_noise_scale", 0.1))
        if noise_scale <= 0:
            return raw_action

        is_numpy = isinstance(raw_action, np.ndarray)
        raw_tensor = torch.from_numpy(raw_action) if is_numpy else raw_action
        noise = torch.randn_like(raw_tensor) * noise_scale
        raw_tensor = (raw_tensor + noise).clamp(-1.0, 1.0)
        return raw_tensor.numpy() if is_numpy else raw_tensor

    def apply_transforms(self, obs: dict) -> dict:
        """Tokenize/normalize a batched observation via the GR00T processor."""
        return self._modality_transform.process_observation(obs, self.embodiment_tag)

    def unapply_transforms(
        self,
        action: dict[str, Any],
        state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Decode (unnormalize, relative->absolute) a normalized action chunk."""
        raw_action_tensor = action["action"]

        if isinstance(raw_action_tensor, torch.Tensor):
            raw_action_tensor = raw_action_tensor.detach().cpu().numpy()

        decoded_state = None
        if state is not None:
            decoded_state = {}
            for key, value in state.items():
                if not key.startswith("state."):
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                decoded_state[key.split(".", 1)[1]] = value

        decoded = self._modality_transform.decode_action(
            action=raw_action_tensor,
            embodiment_tag=self.embodiment_tag,
            state=decoded_state,
        )
        return decoded

    def _update_semantic_fetch_request_delay(self, wallclock: Any) -> float:
        """Estimate the next mid-chunk publish time from observed control wallclock."""
        delay_initial_s = float(getattr(self, "_semantic_fetch_delay_initial_s", 0.8))
        delay_min_s = float(getattr(self, "_semantic_fetch_delay_min_s", 0.1))
        delay_max_s = float(getattr(self, "_semantic_fetch_delay_max_s", 1.5))
        delay_fraction = float(getattr(self, "_semantic_fetch_delay_fraction", 0.45))
        ema_alpha = float(getattr(self, "_semantic_fetch_delay_ema_alpha", 0.25))
        values = torch.as_tensor(wallclock, dtype=torch.float64).reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return float(
                getattr(self, "_semantic_last_request_delay_s", delay_initial_s)
            )
        current_wallclock_s = float(values.median().item())
        previous_wallclock_s = getattr(
            self, "_semantic_last_observation_wallclock_s", None
        )
        self._semantic_last_observation_wallclock_s = current_wallclock_s
        if previous_wallclock_s is not None:
            interval_s = current_wallclock_s - previous_wallclock_s
            max_valid_interval_s = max(5.0, delay_max_s * 4.0)
            if 0.0 < interval_s <= max_valid_interval_s:
                previous_ema_s = getattr(
                    self, "_semantic_observation_interval_ema_s", None
                )
                if previous_ema_s is None:
                    self._semantic_observation_interval_ema_s = interval_s
                else:
                    self._semantic_observation_interval_ema_s = (
                        1.0 - ema_alpha
                    ) * previous_ema_s + ema_alpha * interval_s
        interval_ema_s = getattr(self, "_semantic_observation_interval_ema_s", None)
        if interval_ema_s is None:
            delay_s = delay_initial_s
        else:
            delay_s = interval_ema_s * delay_fraction
        delay_s = min(delay_max_s, max(delay_min_s, delay_s))
        self._semantic_last_request_delay_s = delay_s
        return delay_s

    def _semantic_backbone(
        self, backbone_inputs: BatchFeature
    ) -> tuple[BatchFeature, torch.Tensor]:
        """Return the newest completed semantic packet without blocking after bootstrap."""
        if not self._semantic_enabled:
            outputs = self.backbone(backbone_inputs)
            age = torch.zeros(
                outputs["backbone_features"].shape[0],
                device=outputs["backbone_features"].device,
                dtype=outputs["backbone_features"].dtype,
            )
            return outputs, age

        if self._semantic_central_cache:
            metadata = self._rollout_semantic_metadata
            required = ("env_ids", "frame_ids", "episode_generations")
            missing = [key for key in required if key not in metadata]
            if missing:
                raise RuntimeError(
                    f"Central semantic cache requires env metadata: missing {missing}"
                )
            batch_size = int(metadata["env_ids"].numel())
            now = time.time()
            wallclock = metadata.get(
                "observation_wallclock_s",
                torch.full((batch_size,), now, dtype=torch.float64),
            )
            fetch_request_delay_s = self._update_semantic_fetch_request_delay(wallclock)
            publish_metadata = {
                "env_ids": metadata["env_ids"].tolist(),
                "frame_ids": metadata["frame_ids"].tolist(),
                "episode_generations": metadata["episode_generations"].tolist(),
                "observation_wallclock_s": wallclock.tolist(),
                "semantic_priority": 0,
            }
            env_ids = publish_metadata["env_ids"]
            current_generations = publish_metadata["episode_generations"]
            current_frames = publish_metadata["frame_ids"]
            generation_changed = any(
                self._semantic_last_episode_generations.get(env_id) != generation
                for env_id, generation in zip(env_ids, current_generations, strict=True)
            )
            publish_frames = {
                int(env_id): (int(generation), int(frame_id))
                for env_id, generation, frame_id in zip(
                    env_ids, current_generations, current_frames, strict=True
                )
            }
            if generation_changed and self._semantic_env_bootstrap_publish:
                self._semantic_last_published_frames.update(publish_frames)
            interval_due = (
                self._semantic_publish_interval_frames > 0
                and _semantic_publish_due(
                    self._semantic_last_published_frames,
                    env_ids,
                    current_generations,
                    current_frames,
                    self._semantic_publish_interval_frames,
                )
            )
            if (
                self._semantic_boundary_publish_due()
                or (generation_changed and not self._semantic_env_bootstrap_publish)
                or interval_due
            ):
                self._semantic_client.publish(backbone_inputs, publish_metadata)
                self._semantic_last_published_frames.update(publish_frames)
            self._semantic_last_episode_generations.update(
                zip(env_ids, current_generations, strict=True)
            )
            if (
                self._semantic_non_blocking
                and self._semantic_fetch_hard_max_age_frames < 0
            ):
                # Read the newest completed packet at the action boundary. The
                # zero server wait keeps this independent from VLM completion.
                outputs, response_metadata = self._semantic_client.fetch_latest(
                    env_ids=publish_metadata["env_ids"],
                    episode_generations=publish_metadata["episode_generations"],
                    current_frame_ids=publish_metadata["frame_ids"],
                    wait_for_initial=True,
                    device=self.device,
                    floating_dtype=self.compute_dtype,
                )
                self._semantic_cache = outputs
                self._latest_semantic_metadata = dict(response_metadata)
            elif self._semantic_non_blocking:
                completed = self._semantic_client.poll_latest(
                    device=self.device, floating_dtype=self.compute_dtype
                )
                if completed is not None:
                    self._semantic_cache, response_metadata = completed
                    self._latest_semantic_metadata = dict(response_metadata)

                def cache_identity_matches() -> bool:
                    cached_env_ids = self._latest_semantic_metadata.get("env_ids", ())
                    cached_generations = self._latest_semantic_metadata.get(
                        "episode_generations", ()
                    )
                    return (
                        self._semantic_cache is not None
                        and len(cached_env_ids) == len(env_ids)
                        and all(
                            int(cached_env_id) == int(env_id)
                            for cached_env_id, env_id in zip(
                                cached_env_ids, env_ids, strict=True
                            )
                        )
                        and len(cached_generations) == len(current_generations)
                        and all(
                            int(cached_generation) == int(current_generation)
                            for cached_generation, current_generation in zip(
                                cached_generations, current_generations, strict=True
                            )
                        )
                    )

                def cache_is_within_target_age() -> bool:
                    if not cache_identity_matches():
                        return False
                    hard_max_age = self._semantic_fetch_hard_max_age_frames
                    if hard_max_age < 0:
                        return True
                    source_frames = self._latest_semantic_metadata.get(
                        "source_frame_ids", ()
                    )
                    return len(source_frames) == len(current_frames) and all(
                        0 <= int(current_frame) - int(source_frame) <= hard_max_age
                        for current_frame, source_frame in zip(
                            current_frames, source_frames, strict=True
                        )
                    )

                freshness_attempts = 0
                while not cache_is_within_target_age() and freshness_attempts < 3:
                    self._semantic_client.submit_latest(
                        env_ids=publish_metadata["env_ids"],
                        episode_generations=publish_metadata["episode_generations"],
                        current_frame_ids=publish_metadata["frame_ids"],
                        floating_dtype=self.compute_dtype,
                        request_delay_s=fetch_request_delay_s,
                        max_wait_ms=1000.0,
                    )
                    completed = self._semantic_client.wait_latest(
                        device=self.device,
                        floating_dtype=self.compute_dtype,
                        timeout_ms=2000.0,
                    )
                    freshness_attempts += 1
                    if completed is not None:
                        self._semantic_cache, response_metadata = completed
                        self._latest_semantic_metadata = dict(response_metadata)

                identity_attempts = 0
                while not cache_identity_matches() and identity_attempts < 3:
                    # A latest-only raw publisher can replace the frame-0 packet
                    # during asynchronous auto-reset. Re-publish the current
                    # generation before retrying so one missing env cannot stall
                    # the entire rollout batch forever.
                    self._semantic_client.publish(backbone_inputs, publish_metadata)
                    self._semantic_client.submit_latest(
                        env_ids=publish_metadata["env_ids"],
                        episode_generations=publish_metadata["episode_generations"],
                        current_frame_ids=publish_metadata["frame_ids"],
                        floating_dtype=self.compute_dtype,
                        max_wait_ms=1000.0,
                    )
                    completed = self._semantic_client.wait_latest(
                        device=self.device,
                        floating_dtype=self.compute_dtype,
                        timeout_ms=2000.0,
                    )
                    identity_attempts += 1
                    if completed is not None:
                        self._semantic_cache, response_metadata = completed
                        self._latest_semantic_metadata = dict(response_metadata)

                if not cache_is_within_target_age():
                    source_frames = self._latest_semantic_metadata.get(
                        "source_frame_ids", ()
                    )
                    logger.warning(
                        "Central semantic freshness fallback: current=%s source=%s "
                        "target_max_age=%d",
                        current_frames,
                        source_frames,
                        self._semantic_fetch_hard_max_age_frames,
                    )
                anticipated_frames = [
                    int(frame_id) + int(self.output_action_chunks)
                    for frame_id in publish_metadata["frame_ids"]
                ]
                self._semantic_client.submit_latest(
                    env_ids=publish_metadata["env_ids"],
                    episode_generations=publish_metadata["episode_generations"],
                    current_frame_ids=anticipated_frames,
                    floating_dtype=self.compute_dtype,
                    request_delay_s=fetch_request_delay_s,
                    max_wait_ms=0.0,
                )
                outputs = self._semantic_cache
                response_metadata = self._latest_semantic_metadata
            else:
                outputs, response_metadata = self._semantic_client.fetch_latest(
                    env_ids=publish_metadata["env_ids"],
                    episode_generations=publish_metadata["episode_generations"],
                    current_frame_ids=publish_metadata["frame_ids"],
                    wait_for_initial=True,
                    device=self.device,
                    floating_dtype=self.compute_dtype,
                )
                self._latest_semantic_metadata = dict(response_metadata)
            source_frames = torch.as_tensor(
                response_metadata["source_frame_ids"],
                device=outputs["backbone_features"].device,
                dtype=torch.float32,
            )
            current_frames = metadata["frame_ids"].to(
                device=source_frames.device, dtype=torch.float32
            )
            if self._semantic_age_mode == "simulator":
                age = (current_frames - source_frames).clamp_min(0)
                age = age / max(self._semantic_control_hz, 1e-6)
                age_frames = current_frames - source_frames
                logger.info(
                    "Central semantic simulator age: mean=%.2f max=%.2f frames "
                    "current_mean=%.2f source_mean=%.2f fetch_delay_ms=%.1f",
                    age_frames.float().mean().item(),
                    age_frames.float().max().item(),
                    current_frames.float().mean().item(),
                    source_frames.float().mean().item(),
                    self._semantic_last_request_delay_s * 1000.0,
                )
            else:
                source_wallclock = torch.as_tensor(
                    response_metadata["source_wallclock_s"],
                    device=source_frames.device,
                    dtype=torch.float64,
                )
                age = (time.time() - source_wallclock).clamp_min(0).float()
            outputs = _resize_semantic_token_axis(
                outputs, getattr(self, "_semantic_feature_tokens", 0)
            )
            return outputs, age.to(dtype=outputs["backbone_features"].dtype)

        batch_size = (
            int(backbone_inputs["state"].shape[0])
            if "state" in backbone_inputs
            else int(next(iter(backbone_inputs.values())).shape[0])
        )
        now = time.time()
        current_frame = self._semantic_frame
        reset_mask = self._rollout_reset_mask
        if (
            self._semantic_episode_generations is None
            or self._semantic_episode_generations.numel() != batch_size
        ):
            self._semantic_episode_generations = torch.zeros(
                batch_size, dtype=torch.int64
            )
        if reset_mask is not None and reset_mask.numel() == batch_size:
            reset_cpu = reset_mask.detach().to(device="cpu", dtype=torch.bool)
            self._semantic_episode_generations[reset_cpu] += 1
        else:
            reset_cpu = torch.zeros(batch_size, dtype=torch.bool)
        metadata = {
            "env_slots": list(range(batch_size)),
            "frame_ids": [current_frame] * batch_size,
            "episode_generations": self._semantic_episode_generations.tolist(),
            "observation_wallclock_s": now,
        }
        self._semantic_frame += 1

        if self._semantic_non_blocking:
            completed = self._semantic_client.poll(
                device=self.device, floating_dtype=self.compute_dtype
            )
            if completed is not None:
                outputs, response_metadata, _ = completed
                self._semantic_cache = outputs
                self._semantic_source_wallclock_s = float(
                    response_metadata.get("source_wallclock_s", now)
                )
                source_frames = response_metadata.get(
                    "source_frame_ids", [current_frame]
                )
                self._semantic_source_frame = int(min(source_frames))
                self._semantic_publishes += 1
            if self._semantic_cache is None or bool(reset_cpu.any()):
                self._semantic_cache = self._semantic_client.encode_backbone_blocking(
                    backbone_inputs,
                    metadata=metadata,
                    device=self.device,
                    floating_dtype=self.compute_dtype,
                )
                self._semantic_source_wallclock_s = now
                self._semantic_source_frame = current_frame
                self._semantic_requests += 1
                self._semantic_publishes += 1
            else:
                self._semantic_client.submit(backbone_inputs, metadata=metadata)
                self._semantic_requests += 1
        else:
            self._semantic_cache = self._semantic_client.encode_backbone(
                backbone_inputs,
                metadata=metadata,
                device=self.device,
                floating_dtype=self.compute_dtype,
            )
            self._semantic_source_wallclock_s = now
            self._semantic_source_frame = current_frame
            self._semantic_requests += 1
            self._semantic_publishes += 1

        features = self._semantic_cache["backbone_features"]
        if self._semantic_age_mode == "simulator":
            source_frame = self._semantic_source_frame
            age_calls = max(
                0,
                current_frame
                - int(current_frame if source_frame is None else source_frame),
            )
            simulated_frames = age_calls * int(self.output_action_chunks)
            age_s = simulated_frames / max(self._semantic_control_hz, 1e-6)
        else:
            age_s = max(
                0.0, time.time() - float(self._semantic_source_wallclock_s or now)
            )
        age = torch.full(
            (batch_size,), age_s, device=features.device, dtype=features.dtype
        )
        outputs = _resize_semantic_token_axis(
            self._semantic_cache, getattr(self, "_semantic_feature_tokens", 0)
        )
        self._semantic_cache = outputs
        return outputs, age

    def _requested_eval_semantic_age_frames(
        self, current_frames: list[int]
    ) -> list[int] | None:
        random_max = getattr(self, "_semantic_eval_random_age_max_frames", -1)
        if random_max >= 0:
            return eval_semantic_age_frames(
                torch.as_tensor(current_frames, dtype=torch.int64),
                getattr(self, "_semantic_eval_random_age_min_frames", 0),
                random_max,
                getattr(self, "_semantic_eval_random_age_seed", 2026),
            )
        fixed_age = getattr(self, "_semantic_eval_fixed_age_frames", -1)
        if fixed_age < 0:
            return None
        return [fixed_age] * len(current_frames)

    def _requested_train_semantic_age_frames(
        self, current_frames: list[int]
    ) -> list[int] | None:
        random_max = getattr(self, "_semantic_train_random_age_max_frames", -1)
        if random_max < 0:
            return None
        metadata_ages = self._rollout_semantic_metadata.get("target_age_frames")
        if metadata_ages is None:
            raise RuntimeError(
                "Exact-age semantic train requires target_age_frames from the env worker"
            )
        requested_ages = [int(value) for value in metadata_ages.tolist()]
        if len(requested_ages) != len(current_frames):
            raise RuntimeError(
                "Exact-age semantic train received a mismatched target-age batch: "
                f"ages={len(requested_ages)} frames={len(current_frames)}"
            )
        random_min = getattr(self, "_semantic_train_random_age_min_frames", 0)
        invalid = [
            age for age in requested_ages if age < random_min or age > random_max
        ]
        if invalid:
            raise RuntimeError(
                "Exact-age semantic train received ages outside the configured range: "
                f"invalid={invalid} range=[{random_min}, {random_max}]"
            )
        return requested_ages

    def _fixed_age_train_semantic(
        self,
        fallback_outputs: BatchFeature,
        fallback_age: torch.Tensor,
    ) -> tuple[BatchFeature, torch.Tensor]:
        """Select the exact env-scheduled simulator-frame packets for PPO."""
        if getattr(self, "_semantic_train_random_age_max_frames", -1) < 0:
            return fallback_outputs, fallback_age
        if not (
            self._semantic_enabled
            and self._semantic_central_cache
            and isinstance(self._semantic_client, Gr00tN1d7SemanticCacheClient)
        ):
            raise RuntimeError(
                "Exact-age semantic train requires the central semantic cache"
            )
        if self._semantic_age_mode != "simulator":
            raise RuntimeError(
                "Exact-age semantic train requires semantic_age_mode=simulator"
            )

        metadata = self._rollout_semantic_metadata
        required = ("env_ids", "frame_ids", "episode_generations")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise RuntimeError(
                f"Exact-age semantic train requires env metadata: missing {missing}"
            )
        env_ids = [int(value) for value in metadata["env_ids"].tolist()]
        generations = [int(value) for value in metadata["episode_generations"].tolist()]
        current_frames = [int(value) for value in metadata["frame_ids"].tolist()]
        requested_ages = self._requested_train_semantic_age_frames(current_frames)
        assert requested_ages is not None
        source_frames = [
            max(0, current_frame - requested_age)
            for current_frame, requested_age in zip(
                current_frames, requested_ages, strict=True
            )
        ]
        fetched = self._semantic_client.fetch_exact(
            env_ids=env_ids,
            episode_generations=generations,
            source_frame_ids=source_frames,
            max_wait_ms=self._semantic_train_fixed_age_max_wait_ms,
            device=self.device,
            floating_dtype=self.compute_dtype,
        )
        if fetched is None:
            raise RuntimeError(
                "Exact-age semantic train could not fetch exact packets: "
                f"current={current_frames} source={source_frames} "
                f"requested_age={requested_ages}"
            )
        outputs, response_metadata = fetched
        returned_source_frames = [
            int(value) for value in response_metadata.get("source_frame_ids", ())
        ]
        if returned_source_frames != source_frames:
            raise RuntimeError(
                "Exact-age semantic train received mismatched packets: "
                f"requested={source_frames} returned={returned_source_frames}"
            )
        self._latest_semantic_metadata = dict(response_metadata)
        age_frames = torch.tensor(
            [
                current_frame - source_frame
                for current_frame, source_frame in zip(
                    current_frames, source_frames, strict=True
                )
            ],
            device=outputs["backbone_features"].device,
            dtype=outputs["backbone_features"].dtype,
        )
        outputs = _resize_semantic_token_axis(
            outputs, getattr(self, "_semantic_feature_tokens", 0)
        )
        logger.info(
            "Exact-age semantic train: requested_mean=%.2f actual_mean=%.2f "
            "current_mean=%.2f source_mean=%.2f",
            torch.tensor(requested_ages, dtype=torch.float32).mean().item(),
            age_frames.float().mean().item(),
            torch.tensor(current_frames, dtype=torch.float32).mean().item(),
            torch.tensor(source_frames, dtype=torch.float32).mean().item(),
        )
        age_s = age_frames / max(self._semantic_control_hz, 1e-6)
        return outputs, age_s

    def _fixed_age_eval_semantic(
        self,
        fallback_outputs: BatchFeature,
        fallback_age: torch.Tensor,
    ) -> tuple[BatchFeature, torch.Tensor]:
        """Select exact simulator-frame packets for reproducible policy eval."""
        fixed_age = getattr(self, "_semantic_eval_fixed_age_frames", -1)
        random_age_max = getattr(self, "_semantic_eval_random_age_max_frames", -1)
        if fixed_age < 0 and random_age_max < 0:
            return fallback_outputs, fallback_age
        if not (
            self._semantic_enabled
            and self._semantic_central_cache
            and isinstance(self._semantic_client, Gr00tN1d7SemanticCacheClient)
        ):
            raise RuntimeError(
                "Exact-age semantic eval requires the central semantic cache"
            )
        if self._semantic_age_mode != "simulator":
            raise RuntimeError(
                "Exact-age semantic eval requires semantic_age_mode=simulator"
            )

        metadata = self._rollout_semantic_metadata
        required = ("env_ids", "frame_ids", "episode_generations")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise RuntimeError(
                f"Exact-age semantic eval requires env metadata: missing {missing}"
            )
        env_ids = [int(value) for value in metadata["env_ids"].tolist()]
        generations = [int(value) for value in metadata["episode_generations"].tolist()]
        current_frames = [int(value) for value in metadata["frame_ids"].tolist()]
        requested_ages = self._requested_eval_semantic_age_frames(current_frames)
        if requested_ages is None:
            return fallback_outputs, fallback_age
        source_frames = [
            max(0, current_frame - requested_age)
            for current_frame, requested_age in zip(
                current_frames, requested_ages, strict=True
            )
        ]
        fetched = self._semantic_client.fetch_exact(
            env_ids=env_ids,
            episode_generations=generations,
            source_frame_ids=source_frames,
            max_wait_ms=self._semantic_eval_fixed_age_max_wait_ms,
            device=self.device,
            floating_dtype=self.compute_dtype,
        )
        if fetched is None:
            raise RuntimeError(
                "Exact-age semantic eval could not fetch exact packets: "
                f"current={current_frames} source={source_frames} "
                f"requested_age={requested_ages}"
            )
        outputs, response_metadata = fetched
        returned_source_frames = [
            int(value) for value in response_metadata.get("source_frame_ids", ())
        ]
        if returned_source_frames != source_frames:
            raise RuntimeError(
                "Exact-age semantic eval received mismatched packets: "
                f"requested={source_frames} returned={returned_source_frames}"
            )
        next_current_frames = [
            current_frame + int(self.output_action_chunks)
            for current_frame in current_frames
        ]
        next_requested_ages = self._requested_eval_semantic_age_frames(
            next_current_frames
        )
        assert next_requested_ages is not None
        next_source_frames = [
            max(0, current_frame - requested_age)
            for current_frame, requested_age in zip(
                next_current_frames, next_requested_ages, strict=True
            )
        ]
        if next_source_frames != source_frames and all(
            next_source <= current_frame
            for next_source, current_frame in zip(
                next_source_frames, current_frames, strict=True
            )
        ):
            prefetched = self._semantic_client.fetch_exact(
                env_ids=env_ids,
                episode_generations=generations,
                source_frame_ids=next_source_frames,
                max_wait_ms=self._semantic_eval_fixed_age_max_wait_ms,
                device=self.device,
                floating_dtype=self.compute_dtype,
            )
            if prefetched is None:
                raise RuntimeError(
                    "Exact-age semantic eval could not prefetch next packets: "
                    f"current={current_frames} source={next_source_frames} "
                    f"requested_age={next_requested_ages}"
                )
            returned_next_sources = [
                int(value) for value in prefetched[1].get("source_frame_ids", ())
            ]
            if returned_next_sources != next_source_frames:
                raise RuntimeError(
                    "Exact-age semantic eval prefetched mismatched packets: "
                    f"requested={next_source_frames} returned={returned_next_sources}"
                )
        age_frames = torch.tensor(
            [
                current_frame - source_frame
                for current_frame, source_frame in zip(
                    current_frames, source_frames, strict=True
                )
            ],
            dtype=outputs["backbone_features"].dtype,
        )
        outputs = _resize_semantic_token_axis(
            outputs, getattr(self, "_semantic_feature_tokens", 0)
        )
        logger.info(
            "Exact-age semantic eval: requested_mean=%.2f actual_mean=%.2f "
            "current_mean=%.2f source_mean=%.2f",
            torch.tensor(requested_ages, dtype=torch.float32).mean().item(),
            age_frames.float().mean().item(),
            torch.tensor(current_frames, dtype=torch.float32).mean().item(),
            torch.tensor(source_frames, dtype=torch.float32).mean().item(),
        )
        age_s = torch.nextafter(age_frames / max(self._semantic_control_hz, 1e-6), torch.zeros_like(age_frames))
        return outputs, age_s

    def _current_action_history(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        history_len = int(getattr(self.action_head, "action_history_length", 0))
        action_width = int(
            getattr(self.action_head, "model_action_dim", self.action_dim)
        )
        expected = (batch_size, history_len, action_width)
        if self._semantic_central_cache and self._rollout_semantic_metadata:
            env_ids = self._rollout_semantic_metadata["env_ids"].tolist()
            generations = self._rollout_semantic_metadata[
                "episode_generations"
            ].tolist()
            keys = [
                (int(env_id), int(generation))
                for env_id, generation in zip(env_ids, generations, strict=True)
            ]
            histories = []
            for key in keys:
                stale_keys = [
                    existing
                    for existing in self._action_history_by_env
                    if existing[0] == key[0] and existing != key
                ]
                for stale_key in stale_keys:
                    self._action_history_by_env.pop(stale_key, None)
                history = self._action_history_by_env.get(key)
                if history is None or tuple(history.shape) != expected[1:]:
                    history = torch.zeros(
                        expected[1:], device=device, dtype=self.compute_dtype
                    )
                    self._action_history_by_env[key] = history
                histories.append(
                    self._action_history_by_env[key].to(
                        device=device, dtype=self.compute_dtype
                    )
                )
            self._current_action_history_keys = keys
            self._action_history = torch.stack(histories, dim=0)
            return self._action_history.clone()
        if (
            self._action_history is None
            or tuple(self._action_history.shape) != expected
        ):
            self._action_history = torch.zeros(
                expected, device=device, dtype=self.compute_dtype
            )
        else:
            self._action_history = self._action_history.to(
                device=device, dtype=self.compute_dtype
            )
        reset = self._rollout_reset_mask
        if reset is not None and reset.numel() == batch_size:
            self._action_history[reset.to(device=device, dtype=torch.bool)] = 0
        return self._action_history.clone()

    def _append_action_history(self, actions: torch.Tensor) -> None:
        if self._action_history is None or self._action_history.shape[1] == 0:
            return
        padded = torch.zeros(
            actions.shape[0],
            actions.shape[1],
            self._action_history.shape[-1],
            device=self._action_history.device,
            dtype=self._action_history.dtype,
        )
        width = min(actions.shape[-1], padded.shape[-1])
        padded[..., :width] = actions[..., :width].to(padded)
        history_len = self._action_history.shape[1]
        self._action_history = torch.cat((self._action_history, padded), dim=1)[
            :, -history_len:
        ]
        if self._semantic_central_cache and self._current_action_history_keys:
            for row, key in enumerate(self._current_action_history_keys):
                self._action_history_by_env[key] = self._action_history[row].detach()

    def _append_executed_action_history(self, predicted_actions: torch.Tensor) -> None:
        """Append only the action prefix that will be sent to the environment."""
        executed_actions = _execution_action_prefix(
            predicted_actions, self.output_action_chunks
        )
        self._append_action_history(executed_actions)

    def _get_rl_action(
        self,
        normalized_input: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ):
        """Sample an action and assemble the ``forward_inputs`` cached for the actor."""
        normalized_input = _normalize_gr00t_forward_inputs(normalized_input)

        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        self._rollout_profile_mark("split_inputs")
        semantic_fetch_started = time.perf_counter()
        backbone_outputs, packet_age = self._semantic_backbone(backbone_inputs)
        if mode == "train":
            backbone_outputs, packet_age = self._fixed_age_train_semantic(
                backbone_outputs, packet_age
            )
        self._last_semantic_fetch_s = time.perf_counter() - semantic_fetch_started
        self._rollout_profile_mark("semantic_fetch")
        batch_size = int(backbone_outputs["backbone_features"].shape[0])
        action_history = self._current_action_history(
            batch_size, backbone_outputs["backbone_features"].device
        )
        if self._semantic_zero_packet_age:
            packet_age = torch.zeros_like(packet_age)
        if self._semantic_zero_action_history:
            action_history = torch.zeros_like(action_history)
        self._rollout_profile_mark("action_history")
        action_inputs["packet_age_s"] = packet_age
        action_inputs["action_history"] = action_history
        action_head_outputs, rlinf_outputs = self.action_head.get_rl_action(
            backbone_outputs,
            action_inputs,
            mode=mode,
        )
        self._rollout_profile_mark("action_head")
        actions = _execution_action_prefix(
            rlinf_outputs["actions"], self.output_action_chunks
        )
        self._append_executed_action_history(actions)
        if hasattr(self, "validate_data"):
            self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        actions = actions.float()

        stashed_forward_inputs = {
            key: _batchify_gr00t_forward_input(key, value, batch_size)
            for key, value in dict(action_inputs).items()
        }
        self._rollout_profile_mark("rl_postprocess")
        semantic_forward_inputs = {
            f"semantic_{key}": value.detach()
            for key, value in dict(backbone_outputs).items()
        }
        forward_inputs = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            **stashed_forward_inputs,
            **semantic_forward_inputs,
        }
        if self._semantic_central_cache and self._latest_semantic_metadata:
            semantic_meta = self._latest_semantic_metadata
            forward_inputs.update(
                {
                    "rollout_semantic_env_ids": self._rollout_semantic_metadata[
                        "env_ids"
                    ].to(actions.device),
                    "rollout_semantic_episode_generations": self._rollout_semantic_metadata[
                        "episode_generations"
                    ].to(actions.device),
                    "rollout_semantic_source_frame_ids": torch.as_tensor(
                        semantic_meta["source_frame_ids"], device=actions.device
                    ),
                    "rollout_semantic_versions": torch.as_tensor(
                        semantic_meta["semantic_versions"], device=actions.device
                    ),
                    "action_frame_ids": self._rollout_semantic_metadata["frame_ids"].to(
                        actions.device
                    ),
                    "action_wallclock_s": torch.full(
                        (batch_size,),
                        time.time(),
                        dtype=torch.float64,
                        device=actions.device,
                    ),
                }
            )
        if "task_ids" in self._rollout_semantic_metadata:
            forward_inputs["rollout_task_ids"] = self._rollout_semantic_metadata[
                "task_ids"
            ].to(actions.device)
        if "trial_ids" in self._rollout_semantic_metadata:
            forward_inputs["rollout_trial_ids"] = self._rollout_semantic_metadata[
                "trial_ids"
            ].to(actions.device)
        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "forward_inputs": self._finalize_rollout_forward_inputs(forward_inputs),
        }
        return actions, result

    def _build_deterministic_eval_noise(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        rl_config = self.action_head.rl_config
        if not bool(rl_config.get("deterministic_eval_noise", False)):
            return None
        required = ("task_ids", "trial_ids", "frame_ids")
        missing = [
            key for key in required if key not in self._rollout_semantic_metadata
        ]
        if missing:
            raise RuntimeError(
                "deterministic_eval_noise requires rollout metadata: "
                + ", ".join(missing)
            )
        task_ids = self._rollout_semantic_metadata["task_ids"]
        base_seed: int | list[int] = int(rl_config.get("eval_noise_seed", 1234))
        base_seed_by_task = rl_config.get("eval_noise_seed_by_task")
        if base_seed_by_task is not None:
            task_seed_values = [int(value) for value in base_seed_by_task]
            task_values = torch.as_tensor(task_ids).reshape(-1).tolist()
            invalid_tasks = [
                int(task_id)
                for task_id in task_values
                if int(task_id) < 0 or int(task_id) >= len(task_seed_values)
            ]
            if invalid_tasks:
                raise ValueError(
                    "eval_noise_seed_by_task has no entry for task ids "
                    f"{sorted(set(invalid_tasks))}"
                )
            base_seed = [task_seed_values[int(task_id)] for task_id in task_values]
        seeds = eval_noise_seeds(
            task_ids,
            self._rollout_semantic_metadata["trial_ids"],
            self._rollout_semantic_metadata["frame_ids"],
            base_seed,
        )
        if len(seeds) != batch_size:
            raise RuntimeError(
                f"eval seed count {len(seeds)} does not match batch size {batch_size}"
            )
        sample_shape = (
            self.action_head.action_horizon,
            self.action_head.model_action_dim,
        )
        rows = []
        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            rows.append(
                torch.randn(
                    sample_shape,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
        return torch.stack(rows, dim=0)

    def _get_deterministic_eval_action(
        self, normalized_input: dict[str, Any]
    ) -> torch.Tensor:
        normalized_input = _normalize_gr00t_forward_inputs(normalized_input)
        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        semantic_fetch_started = time.perf_counter()
        backbone_outputs, packet_age = self._semantic_backbone(backbone_inputs)
        backbone_outputs, packet_age = self._fixed_age_eval_semantic(
            backbone_outputs, packet_age
        )
        self._last_semantic_fetch_s = time.perf_counter() - semantic_fetch_started
        batch_size = int(backbone_outputs["backbone_features"].shape[0])
        if self._semantic_zero_packet_age:
            packet_age = torch.zeros_like(packet_age)
        action_inputs["packet_age_s"] = packet_age
        action_history = self._current_action_history(
            batch_size, backbone_outputs["backbone_features"].device
        )
        if self._semantic_zero_action_history:
            action_history = torch.zeros_like(action_history)
        action_inputs["action_history"] = action_history
        initial_noise = self._build_deterministic_eval_noise(
            batch_size=batch_size,
            device=backbone_outputs["backbone_features"].device,
            dtype=backbone_outputs["backbone_features"].dtype,
        )
        model_pred = self.action_head.get_eval_action(
            backbone_outputs,
            action_inputs,
            initial_noise=initial_noise,
        )
        actions = _execution_action_prefix(
            model_pred["action_pred"], self.output_action_chunks
        )
        if bool(self.action_head.rl_config.get("eval_repro_diagnostics", False)):
            env_ids = self._rollout_semantic_metadata.get("env_ids")
            frame_ids = self._rollout_semantic_metadata.get("frame_ids")
            generations = self._rollout_semantic_metadata.get("episode_generations")
            fingerprint_key = (
                tuple(env_ids.tolist()) if env_ids is not None else (),
                tuple(generations.tolist()) if generations is not None else (),
            )
            if fingerprint_key != getattr(self, "_eval_repro_fingerprint_key", None):
                self._eval_repro_fingerprint_key = fingerprint_key
                logger.info(
                    "Eval policy fingerprint semantic=%s action=%s "
                    "generations=%s frame_ids=%s",
                    _tensor_fingerprint(backbone_outputs["backbone_features"]),
                    _tensor_fingerprint(actions),
                    generations,
                    frame_ids,
                )
        self._append_executed_action_history(actions)
        if hasattr(self, "validate_data"):
            self.validate_data(model_pred, backbone_outputs, is_training=False)
        return actions.float()

    def _finalize_rollout_forward_inputs(
        self,
        forward_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Ensure cached rollout inputs are batch-splittable tensors."""
        finalized = {}
        batch_size = int(forward_inputs["chains"].shape[0])
        for key, value in forward_inputs.items():
            value = _tensorize_forward_input(value)
            if key in _FORWARD_INPUT_MODEL_KEYS:
                value = _batchify_gr00t_forward_input(key, value, batch_size)
            finalized[key] = value
        return finalized

    def _get_action_from_normalized_input(
        self, normalized_input: dict[str, Any]
    ) -> torch.Tensor:
        """Deterministic action prediction (eval path) without RL bookkeeping."""
        device_type = getattr(self.device, "type", "cpu")
        autocast_context = (
            torch.autocast(device_type=device_type, dtype=self.compute_dtype)
            if device_type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            model_pred = self.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(
        self,
        normalized_action: torch.Tensor,
        state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()}, state=state)

    def _load_metadata(self, exp_cfg_dir: Path):
        """Populate ``valid_action_dim`` and ``image_nums`` from checkpoint metadata.

        Falls back to inferring these from the modality config (and finally from
        the model config) when ``metadata.json`` is absent.
        """
        metadata_path = exp_cfg_dir / "metadata.json"
        if not metadata_path.exists():
            logger.info(
                "Metadata file not found at %s. Inferring from modality_config...",
                metadata_path,
            )

            tag_value = self.embodiment_tag.value
            if self._modality_config is None or tag_value not in self._modality_config:
                logger.info(
                    "Modality config is missing or does not contain tag %s. "
                    "Attempting to infer valid_action_dim and image_nums from "
                    "config attributes.",
                    self.embodiment_tag.value,
                )
                self.valid_action_dim = getattr(
                    self.config, "max_action_dim", getattr(self.config, "action_dim", 7)
                )
                self.image_nums = getattr(self.config, "image_nums", 1)
                logger.info(
                    "Inferred fallback: valid_action_dim=%s, image_nums=%s",
                    self.valid_action_dim,
                    self.image_nums,
                )
                return

            current_modality = self._modality_config[tag_value]

            valid_action_dim = 0
            if "action" in current_modality:
                action_modality_cfg = current_modality["action"]
                if (
                    hasattr(action_modality_cfg, "dim_map")
                    and action_modality_cfg.dim_map
                ):
                    for dim_val in action_modality_cfg.dim_map.values():
                        valid_action_dim += dim_val
                elif (
                    hasattr(action_modality_cfg, "modality_keys")
                    and action_modality_cfg.modality_keys
                ):
                    norm_params = getattr(
                        getattr(
                            self._modality_transform, "state_action_processor", None
                        ),
                        "norm_params",
                        {},
                    )
                    action_norm_params = (
                        norm_params.get(tag_value, {}).get("action", {})
                        if isinstance(norm_params, dict)
                        else {}
                    )
                    if action_norm_params:
                        valid_action_dim = sum(
                            int(action_norm_params[key]["dim"].item())
                            for key in action_modality_cfg.modality_keys
                            if key in action_norm_params
                        )
                    else:
                        valid_action_dim = getattr(self.config, "max_action_dim", 29)
                elif isinstance(action_modality_cfg, dict):
                    if action_modality_cfg.get("dim_map"):
                        for dim_val in action_modality_cfg["dim_map"].values():
                            valid_action_dim += dim_val
                    elif "dim" in action_modality_cfg:
                        valid_action_dim = action_modality_cfg["dim"]
                    else:
                        valid_action_dim = getattr(self.config, "max_action_dim", 29)
                else:
                    valid_action_dim = getattr(self.config, "max_action_dim", 29)

            self.valid_action_dim = valid_action_dim

            if "video" in current_modality:
                video_modality_cfg = current_modality["video"]
                if (
                    hasattr(video_modality_cfg, "modality_keys")
                    and video_modality_cfg.modality_keys
                ):
                    self.image_nums = len(video_modality_cfg.modality_keys)
                elif (
                    isinstance(video_modality_cfg, dict)
                    and "modality_keys" in video_modality_cfg
                ):
                    self.image_nums = len(video_modality_cfg["modality_keys"])
                else:
                    self.image_nums = 1
            else:
                self.image_nums = 1

            logger.info(
                "Inferred from modality_config: valid_action_dim=%s, image_nums=%s",
                self.valid_action_dim,
                self.image_nums,
            )
            return

        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}"
            )

        self.metadata = metadata_dict
        if hasattr(self._modality_transform, "set_metadata"):
            self._modality_transform.set_metadata(metadata_dict)

        valid_action_dim = 0
        action_mods = self.metadata.get("modalities", {}).get("action", {})
        for v in action_mods.values():
            shape = v.get("shape", [0]) if isinstance(v, dict) else [0]
            valid_action_dim += shape[0] if len(shape) > 0 else 0
        self.valid_action_dim = valid_action_dim

        video_mods = self.metadata.get("modalities", {}).get("video", {})
        self.image_nums = len(video_mods)
