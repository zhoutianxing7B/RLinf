from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn


def gr00t_delay_adapter_missing_prefixes(
    rl_head_config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Allow only delay adapters explicitly requested for checkpoint migration."""
    prefixes = ["action_head.value_head."]
    if bool(rl_head_config.get("initialize_packet_age_adapter", False)):
        prefixes.append("action_head.packet_age_adapter.")
    if int(rl_head_config.get("initialize_action_history_length", 0)) > 0:
        prefixes.append("action_head.action_history_adapter.")
    return tuple(prefixes)


def load_state_dict_with_allowed_missing(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    allowed_missing_prefixes: Sequence[str] = (),
) -> list[str]:
    """Load a checkpoint while allowing only explicitly new model modules."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint state dict is incompatible: "
            f"missing={invalid_missing[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    return list(incompatible.missing_keys)
