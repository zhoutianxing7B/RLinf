# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deployable physical rewards for agent-driven robot RL experiments.

The language model authors a small JSON program outside the control loop.  The
program is evaluated locally from poses, joints, and velocities that have a
direct real-robot sensor counterpart.  It cannot read simulator success,
environment reward, task predicates, images, or policy actions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_FORBIDDEN_REFERENCE_PARTS = (
    "action",
    "done",
    "image",
    "predicate",
    "reward",
    "success",
    "termination",
)
_SUPPORTED_TERM_TYPES = frozenset(
    {"distance", "height_delta", "relative_scalar", "scalar"}
)
_SUPPORTED_CONDITION_TYPES = frozenset(
    {"delta_distance", "distance", "relative_scalar", "scalar"}
)
_SUPPORTED_OPERATORS = frozenset({"gt", "lt"})
_MAX_COMPONENTS = 8


class PhysicalRewardProgramError(ValueError):
    """Raised when a physical reward program is unsafe or malformed."""


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PhysicalRewardProgramError(f"{name} must be numeric.") from error
    if not math.isfinite(result):
        raise PhysicalRewardProgramError(f"{name} must be finite.")
    return result


def _validate_reference(reference: Any, available_keys: set[str] | None) -> str:
    if not isinstance(reference, str) or not reference:
        raise PhysicalRewardProgramError(
            "Physical references must be non-empty strings."
        )
    lowered = reference.lower()
    if any(part in lowered for part in _FORBIDDEN_REFERENCE_PARTS):
        raise PhysicalRewardProgramError(
            f"Physical reference {reference!r} contains a forbidden signal."
        )
    if available_keys is not None and reference not in available_keys:
        raise PhysicalRewardProgramError(
            f"Physical reference {reference!r} is absent from the scene schema."
        )
    return reference


def _require_index(raw: Mapping[str, Any], name: str) -> int:
    if "index" not in raw:
        raise PhysicalRewardProgramError(f"{name}.index is required.")
    try:
        index = int(raw["index"])
    except (TypeError, ValueError) as error:
        raise PhysicalRewardProgramError(
            f"{name}.index must be an integer in [0, 6]."
        ) from error
    if isinstance(raw["index"], float) and raw["index"] != index:
        raise PhysicalRewardProgramError(f"{name}.index must be an integer in [0, 6].")
    if not 0 <= index <= 6:
        raise PhysicalRewardProgramError(f"{name}.index must be an integer in [0, 6].")
    return index


def _validate_axes(raw_axes: Any) -> tuple[int, ...]:
    if not isinstance(raw_axes, Sequence) or isinstance(raw_axes, (str, bytes)):
        raise PhysicalRewardProgramError("axes must be a non-empty integer list.")
    axes = tuple(int(axis) for axis in raw_axes)
    if (
        not axes
        or len(set(axes)) != len(axes)
        or any(axis < 0 or axis > 6 for axis in axes)
    ):
        raise PhysicalRewardProgramError("axes must contain unique values in [0, 6].")
    return axes


def validate_physical_reward_program(
    program: Mapping[str, Any],
    *,
    available_keys: Sequence[str] | None = None,
    expected_gamma: float | None = None,
) -> dict[str, Any]:
    """Validate and normalize a bounded potential-based reward program.

    Args:
        program: Parsed JSON mapping authored by the reward manager.
        available_keys: Optional raw-observation key allowlist for the scene.
        expected_gamma: SAC discount.  Potential shaping must use exactly this
            value, otherwise it is not policy-invariant shaping.

    Returns:
        A JSON-serializable normalized program.
    """
    if not isinstance(program, Mapping) or int(program.get("schema_version", -1)) != 1:
        raise PhysicalRewardProgramError("schema_version must equal 1.")
    name = program.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PhysicalRewardProgramError("A reward program needs a name.")
    gamma = _finite_float(program.get("gamma"), "gamma")
    if not 0.0 < gamma <= 1.0:
        raise PhysicalRewardProgramError("gamma must be in (0, 1].")
    if expected_gamma is not None and not math.isclose(
        gamma, float(expected_gamma), rel_tol=0.0, abs_tol=1e-9
    ):
        raise PhysicalRewardProgramError(
            f"Program gamma {gamma} does not match SAC gamma {expected_gamma}."
        )

    completion_bonus = _finite_float(
        program.get("completion_bonus", 1.0), "completion_bonus"
    )
    if not math.isclose(completion_bonus, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise PhysicalRewardProgramError("completion_bonus must equal 1.0.")
    hold_steps = int(program.get("completion_hold_steps", 1))
    if not 1 <= hold_steps <= 32:
        raise PhysicalRewardProgramError("completion_hold_steps must be in [1, 32].")
    potential_scale = _finite_float(
        program.get("potential_scale", 0.1), "potential_scale"
    )
    if not 0.0 <= potential_scale <= 0.25 * completion_bonus:
        raise PhysicalRewardProgramError(
            "potential_scale must stay in [0, 0.25 * completion_bonus]."
        )

    key_set = None if available_keys is None else set(available_keys)
    conditions = program.get("completion_conditions")
    if not isinstance(conditions, list) or not conditions:
        raise PhysicalRewardProgramError("completion_conditions must be non-empty.")
    if len(conditions) > _MAX_COMPONENTS:
        raise PhysicalRewardProgramError(
            f"completion_conditions cannot exceed {_MAX_COMPONENTS}."
        )
    normalized_conditions: list[dict[str, Any]] = []
    for index, raw in enumerate(conditions):
        if (
            not isinstance(raw, Mapping)
            or raw.get("type") not in _SUPPORTED_CONDITION_TYPES
        ):
            raise PhysicalRewardProgramError(
                f"Unsupported completion condition {index}."
            )
        condition_type = str(raw["type"])
        operator = str(raw.get("op", "lt"))
        if operator not in _SUPPORTED_OPERATORS:
            raise PhysicalRewardProgramError(
                f"Unsupported operator in condition {index}."
            )
        threshold = _finite_float(raw.get("threshold"), f"condition[{index}].threshold")
        normalized: dict[str, Any] = {
            "type": condition_type,
            "op": operator,
            "threshold": threshold,
        }
        if condition_type == "distance":
            normalized.update(
                {
                    "left": _validate_reference(raw.get("left"), key_set),
                    "right": _validate_reference(raw.get("right"), key_set),
                    "axes": list(_validate_axes(raw.get("axes", [0, 1, 2]))),
                }
            )
        elif condition_type == "scalar":
            normalized.update(
                {
                    "key": _validate_reference(raw.get("key"), key_set),
                    "index": _require_index(raw, f"condition[{index}]"),
                }
            )
        elif condition_type == "relative_scalar":
            normalized.update(
                {
                    "left": _validate_reference(raw.get("left"), key_set),
                    "right": _validate_reference(raw.get("right"), key_set),
                    "index": _require_index(raw, f"condition[{index}]"),
                }
            )
        else:
            normalized.update(
                {
                    "key": _validate_reference(raw.get("key"), key_set),
                    "axes": list(_validate_axes(raw.get("axes", [0, 1, 2]))),
                }
            )
        normalized_conditions.append(normalized)

    terms = program.get("potential_terms")
    if not isinstance(terms, list) or not terms:
        raise PhysicalRewardProgramError("potential_terms must be non-empty.")
    if len(terms) > _MAX_COMPONENTS:
        raise PhysicalRewardProgramError(
            f"potential_terms cannot exceed {_MAX_COMPONENTS}."
        )
    normalized_terms: list[dict[str, Any]] = []
    total_weight = 0.0
    for index, raw in enumerate(terms):
        if not isinstance(raw, Mapping) or raw.get("type") not in _SUPPORTED_TERM_TYPES:
            raise PhysicalRewardProgramError(f"Unsupported potential term {index}.")
        term_type = str(raw["type"])
        weight = _finite_float(raw.get("weight"), f"term[{index}].weight")
        scale = _finite_float(raw.get("scale"), f"term[{index}].scale")
        if weight < 0.0 or scale <= 0.0:
            raise PhysicalRewardProgramError(
                "Potential weights and scales must be positive."
            )
        total_weight += weight
        normalized = {"type": term_type, "weight": weight, "scale": scale}
        if term_type == "distance":
            normalized.update(
                {
                    "left": _validate_reference(raw.get("left"), key_set),
                    "right": _validate_reference(raw.get("right"), key_set),
                    "axes": list(_validate_axes(raw.get("axes", [0, 1, 2]))),
                }
            )
        elif term_type in {"height_delta", "scalar"}:
            normalized.update(
                {
                    "key": _validate_reference(raw.get("key"), key_set),
                    "index": _require_index(raw, f"term[{index}]"),
                }
            )
            if term_type == "scalar":
                normalized["target"] = _finite_float(
                    raw.get("target"), f"term[{index}].target"
                )
        else:
            normalized.update(
                {
                    "left": _validate_reference(raw.get("left"), key_set),
                    "right": _validate_reference(raw.get("right"), key_set),
                    "index": _require_index(raw, f"term[{index}]"),
                    "target": _finite_float(raw.get("target"), f"term[{index}].target"),
                }
            )
        normalized_terms.append(normalized)
    if total_weight > 1.0 + 1e-9:
        raise PhysicalRewardProgramError(
            "Potential term weights must sum to at most 1."
        )

    task_ids_raw = program.get("task_ids")
    if not isinstance(task_ids_raw, list) or not task_ids_raw:
        raise PhysicalRewardProgramError("task_ids must be a non-empty integer list.")
    task_ids = sorted({int(task_id) for task_id in task_ids_raw})
    if any(task_id < 0 for task_id in task_ids):
        raise PhysicalRewardProgramError("task_ids cannot contain negative values.")

    return {
        "schema_version": 1,
        "name": name.strip(),
        "rationale": str(program.get("rationale", "")).strip(),
        "task_ids": task_ids,
        "gamma": gamma,
        "completion_bonus": completion_bonus,
        "completion_hold_steps": hold_steps,
        "completion_conditions": normalized_conditions,
        "potential_scale": potential_scale,
        "potential_terms": normalized_terms,
    }


def physical_program_digest(program: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 identity of a normalized reward program."""
    payload = json.dumps(program, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_physical_reward_program(
    path: str | Path, program: Mapping[str, Any]
) -> None:
    """Atomically publish a validated program without exposing partial JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(program, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


@dataclass
class PhysicalRewardStep:
    """Vectorized reward result and audit components for one simulator step."""

    rewards: np.ndarray
    completion: np.ndarray
    raw_completion: np.ndarray
    potential: np.ndarray
    potential_delta: np.ndarray
    condition_pass: np.ndarray
    term_scores: np.ndarray
    revision: int
    digest: str


class PhysicalPotentialRewardRuntime:
    """Hot-reloadable, vectorized evaluator for physical reward programs."""

    def __init__(
        self,
        program_path: str | Path,
        *,
        num_envs: int,
        expected_gamma: float,
        reload_interval_steps: int = 16,
    ) -> None:
        self.program_path = Path(program_path)
        self.num_envs = int(num_envs)
        self.expected_gamma = float(expected_gamma)
        self.reload_interval_steps = max(1, int(reload_interval_steps))
        self._calls = 0
        self._mtime_ns = -1
        self.revision = 0
        self.program: dict[str, Any] = {}
        self.digest = ""
        self._baseline: list[dict[str, np.ndarray]] = [{} for _ in range(num_envs)]
        self._previous_state: list[dict[str, np.ndarray]] = [
            {} for _ in range(num_envs)
        ]
        self._previous_potential = np.zeros(num_envs, dtype=np.float64)
        self._hold = np.zeros(num_envs, dtype=np.int32)
        self._reload(force=True)

    def _reload(self, *, force: bool = False) -> bool:
        stat = self.program_path.stat()
        if not force and stat.st_mtime_ns == self._mtime_ns:
            return False
        raw = json.loads(self.program_path.read_text())
        normalized = validate_physical_reward_program(
            raw, expected_gamma=self.expected_gamma
        )
        self.program = normalized
        self.digest = physical_program_digest(normalized)
        self._mtime_ns = stat.st_mtime_ns
        self.revision += 1
        return True

    @staticmethod
    def _array(observation: Mapping[str, Any], key: str) -> np.ndarray:
        if key not in observation:
            raise PhysicalRewardProgramError(f"Observation is missing {key!r}.")
        value = np.asarray(observation[key], dtype=np.float64).reshape(-1)
        if not np.isfinite(value).all():
            raise PhysicalRewardProgramError(f"Observation {key!r} is not finite.")
        return value

    def reset(
        self,
        observations: Sequence[Mapping[str, Any]],
        env_indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """Capture episode baselines and initialize the shaping potential."""
        self._reload()
        indices = (
            list(range(self.num_envs))
            if env_indices is None
            else [int(i) for i in env_indices]
        )
        if len(indices) != len(observations):
            raise PhysicalRewardProgramError("reset observations and indices disagree.")
        for local_index, env_index in enumerate(indices):
            observation = observations[local_index]
            keys = {
                str(term["key"])
                for term in self.program["potential_terms"]
                if term["type"] == "height_delta"
            }
            self._baseline[env_index] = {
                key: self._array(observation, key).copy() for key in keys
            }
            temporal_keys = {
                str(condition["key"])
                for condition in self.program["completion_conditions"]
                if condition["type"] == "delta_distance"
            }
            self._previous_state[env_index] = {
                key: self._array(observation, key).copy() for key in temporal_keys
            }
            self._hold[env_index] = 0
            self._previous_potential[env_index] = self._potential(
                observation, env_index
            )

    def _distance(
        self, observation: Mapping[str, Any], spec: Mapping[str, Any]
    ) -> float:
        left = self._array(observation, str(spec["left"]))
        right = self._array(observation, str(spec["right"]))
        axes = np.asarray(spec["axes"], dtype=np.int64)
        if axes.max(initial=0) >= min(left.size, right.size):
            raise PhysicalRewardProgramError(
                "A distance axis exceeds its physical vector."
            )
        return float(np.linalg.norm(left[axes] - right[axes]))

    def _potential_with_terms(
        self, observation: Mapping[str, Any], env_index: int
    ) -> tuple[float, np.ndarray]:
        scores: list[float] = []
        for term in self.program["potential_terms"]:
            term_type = term["type"]
            if term_type == "distance":
                score = math.exp(-self._distance(observation, term) / term["scale"])
            elif term_type == "height_delta":
                current = self._array(observation, term["key"])[term["index"]]
                if term["key"] not in self._baseline[env_index]:
                    self._baseline[env_index][term["key"]] = self._array(
                        observation, term["key"]
                    ).copy()
                baseline = self._baseline[env_index][term["key"]][term["index"]]
                score = float(np.clip((current - baseline) / term["scale"], 0.0, 1.0))
            elif term_type == "scalar":
                current = self._array(observation, term["key"])[term["index"]]
                score = math.exp(-abs(current - term["target"]) / term["scale"])
            else:
                left = self._array(observation, term["left"])[term["index"]]
                right = self._array(observation, term["right"])[term["index"]]
                score = math.exp(-abs((left - right) - term["target"]) / term["scale"])
            scores.append(float(score))
        value = sum(
            term["weight"] * score
            for term, score in zip(self.program["potential_terms"], scores, strict=True)
        )
        return float(np.clip(value, 0.0, 1.0)), np.asarray(scores, dtype=np.float64)

    def _potential(self, observation: Mapping[str, Any], env_index: int) -> float:
        return self._potential_with_terms(observation, env_index)[0]

    def _condition(
        self,
        observation: Mapping[str, Any],
        spec: Mapping[str, Any],
        env_index: int,
    ) -> bool:
        if spec["type"] == "distance":
            value = self._distance(observation, spec)
        elif spec["type"] == "scalar":
            value = float(self._array(observation, spec["key"])[spec["index"]])
        elif spec["type"] == "relative_scalar":
            left = self._array(observation, spec["left"])[spec["index"]]
            right = self._array(observation, spec["right"])[spec["index"]]
            value = float(left - right)
        else:
            current = self._array(observation, spec["key"])
            previous = self._previous_state[env_index].get(spec["key"], current)
            axes = np.asarray(spec["axes"], dtype=np.int64)
            if axes.max(initial=0) >= min(current.size, previous.size):
                raise PhysicalRewardProgramError(
                    "A delta_distance axis exceeds its physical vector."
                )
            value = float(np.linalg.norm(current[axes] - previous[axes]))
        return (
            value < spec["threshold"]
            if spec["op"] == "lt"
            else value > spec["threshold"]
        )

    def compute(
        self,
        observations: Sequence[Mapping[str, Any]],
        task_ids: Sequence[int] | np.ndarray,
        *,
        terminal: Sequence[bool] | np.ndarray | None = None,
    ) -> PhysicalRewardStep:
        """Evaluate ``C(s') + eta * (gamma * Phi(s') - Phi(s))``."""
        self._calls += 1
        reloaded = False
        if self._calls % self.reload_interval_steps == 0:
            reloaded = self._reload()
        if len(observations) != self.num_envs or len(task_ids) != self.num_envs:
            raise PhysicalRewardProgramError(
                "Physical reward batch has the wrong size."
            )
        terminal_mask = (
            np.zeros(self.num_envs, dtype=bool)
            if terminal is None
            else np.asarray(terminal, dtype=bool)
        )
        completion = np.zeros(self.num_envs, dtype=np.float64)
        raw_completion = np.zeros(self.num_envs, dtype=np.float64)
        potential = np.zeros(self.num_envs, dtype=np.float64)
        potential_delta = np.zeros(self.num_envs, dtype=np.float64)
        condition_pass = np.zeros(
            (self.num_envs, len(self.program["completion_conditions"])),
            dtype=np.float64,
        )
        term_scores = np.zeros(
            (self.num_envs, len(self.program["potential_terms"])), dtype=np.float64
        )
        task_allowlist = set(self.program["task_ids"])
        for env_index, (observation, task_id) in enumerate(
            zip(observations, task_ids, strict=True)
        ):
            if int(task_id) not in task_allowlist:
                self._hold[env_index] = 0
                continue
            current_potential, current_term_scores = self._potential_with_terms(
                observation, env_index
            )
            potential[env_index] = current_potential
            term_scores[env_index] = current_term_scores
            passed = np.asarray(
                [
                    self._condition(observation, condition, env_index)
                    for condition in self.program["completion_conditions"]
                ],
                dtype=np.float64,
            )
            condition_pass[env_index] = passed
            completed_now = bool(passed.all())
            raw_completion[env_index] = float(completed_now)
            self._hold[env_index] = self._hold[env_index] + 1 if completed_now else 0
            completion[env_index] = float(
                self._hold[env_index] >= self.program["completion_hold_steps"]
            )
            if not reloaded:
                next_potential = 0.0 if terminal_mask[env_index] else current_potential
                potential_delta[env_index] = (
                    self.program["gamma"] * next_potential
                    - self._previous_potential[env_index]
                )
            self._previous_potential[env_index] = current_potential
            for condition in self.program["completion_conditions"]:
                if condition["type"] == "delta_distance":
                    self._previous_state[env_index][condition["key"]] = self._array(
                        observation, condition["key"]
                    ).copy()
        rewards = (
            self.program["completion_bonus"] * completion
            + self.program["potential_scale"] * potential_delta
        )
        return PhysicalRewardStep(
            rewards=rewards.astype(np.float32),
            completion=completion.astype(np.float32),
            raw_completion=raw_completion.astype(np.float32),
            potential=potential.astype(np.float32),
            potential_delta=potential_delta.astype(np.float32),
            condition_pass=condition_pass.astype(np.float32),
            term_scores=term_scores.astype(np.float32),
            revision=self.revision,
            digest=self.digest,
        )
