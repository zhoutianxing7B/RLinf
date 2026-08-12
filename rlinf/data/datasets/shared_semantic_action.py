# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Cached GR00T semantic packets for DiT-only action-expert SFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SharedSemanticActionDataset(Dataset):
    """Load one or more canonical semantic action caches.

    The cache contains ``forward_inputs`` with a leading sample dimension. It
    stores the exact backbone tensors consumed by PPO plus expert actions. The
    legacy rehearsal cache is accepted by treating ``chains[:, 0]`` as actions.
    """

    REQUIRED_KEYS = {
        "state",
        "embodiment_id",
        "packet_age_s",
        "semantic_backbone_features",
    }

    def __init__(self, data_paths: str | list[str]):
        paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        if not paths:
            raise ValueError("Shared semantic action dataset paths cannot be empty")

        parts: dict[str, list[torch.Tensor]] = {}
        expected_keys: set[str] | None = None
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("forward_inputs"), dict
            ):
                raise ValueError(f"Invalid semantic action cache: {path}")
            inputs = dict(payload["forward_inputs"])
            if "action" not in inputs:
                chains = inputs.get("chains")
                if not torch.is_tensor(chains) or chains.ndim != 4 or chains.shape[1] != 1:
                    raise ValueError(
                        f"{path} requires action or legacy chains with shape [N,1,H,D]"
                    )
                inputs["action"] = chains[:, 0]
            inputs.pop("chains", None)
            inputs.pop("denoise_inds", None)
            if "action_mask" not in inputs:
                inputs["action_mask"] = torch.ones_like(
                    inputs["action"], dtype=torch.bool
                )

            missing = self.REQUIRED_KEYS - set(inputs)
            if missing:
                raise ValueError(f"Missing fields in {path}: {sorted(missing)}")
            sample_count = int(inputs["action"].shape[0])
            tensor_inputs = {}
            for key, value in inputs.items():
                if not torch.is_tensor(value):
                    continue
                if value.shape[0] != sample_count:
                    raise ValueError(f"Field {key!r} has inconsistent length in {path}")
                tensor_inputs[key] = value
            current_keys = set(tensor_inputs)
            if expected_keys is None:
                expected_keys = current_keys
            elif current_keys != expected_keys:
                raise ValueError(
                    "All semantic action caches must have identical tensor fields; "
                    f"{path} differs by {sorted(current_keys ^ expected_keys)}"
                )
            for key, value in tensor_inputs.items():
                parts.setdefault(key, []).append(value)

        self.forward_inputs = {
            key: torch.cat(values, dim=0) for key, values in parts.items()
        }

    def __len__(self) -> int:
        return int(self.forward_inputs["action"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.forward_inputs.items()}


def build_shared_semantic_action_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: str | list[str],
    eval_dataset: bool = False,
) -> tuple[DataLoader, None]:
    dataset = SharedSemanticActionDataset(data_paths)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        seed=int(cfg.actor.get("seed", 0)),
    )
    batch_size = int(
        cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size)
        if eval_dataset
        else cfg.actor.micro_batch_size
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(cfg.data.get("num_workers", 4)),
        pin_memory=bool(cfg.data.get("pin_memory", True)),
        drop_last=not eval_dataset,
    )
    return loader, None
