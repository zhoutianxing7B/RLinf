# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""A compact, secret-safe Luna manager for physical reward evolution."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rlinf.agents.enpire_reward.physical_potential import (
    PhysicalRewardProgramError,
    physical_program_digest,
    validate_physical_reward_program,
)

_MANAGER_METRIC_PREFIXES = (
    "env/physical_",
    "train/actor/",
    "train/critic/",
    "train/sac/",
    "train/demo_buffer/",
)


def _compact_experiment(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Keep causal reward evidence without resending full trainer telemetry."""
    compact = {
        key: experiment[key]
        for key in (
            "kind",
            "step",
            "action",
            "score",
            "score_panel",
            "reward_digest",
            "candidate_evaluation",
            "candidate_burn_in_completed",
        )
        if key in experiment
    }
    metrics = experiment.get("metrics", {})
    if isinstance(metrics, Mapping):
        compact["metrics"] = {
            str(key): value
            for key, value in metrics.items()
            if str(key).startswith(_MANAGER_METRIC_PREFIXES)
        }
    return compact


class LunaRewardManagerError(RuntimeError):
    """A redacted manager transport or response failure."""


@dataclass(frozen=True)
class LunaRewardManagerConfig:
    """Connection and retry settings without the credential value."""

    base_url: str = "https://maimai.it.com"
    model: str = "gpt-5.6-luna"
    api_key_env: str = "AGENTIC_MODEL_API_KEY"
    timeout_seconds: float = 180.0
    max_retries: int = 3


@dataclass(frozen=True)
class LunaProposal:
    """Validated proposal and non-secret request audit."""

    program: Mapping[str, Any]
    response_id: str | None
    usage: Mapping[str, int]
    elapsed_seconds: float
    attempt: int


def _extract_json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    fence = chr(96) * 3
    if stripped.startswith(fence) and stripped.endswith(fence):
        lines = stripped.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    else:
        candidate = stripped
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise LunaRewardManagerError("Luna returned no JSON object.") from None
        try:
            decoded = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as error:
            raise LunaRewardManagerError("Luna returned invalid JSON.") from error
    if not isinstance(decoded, Mapping):
        raise LunaRewardManagerError("Luna proposal must be a JSON object.")
    return decoded


class LunaRewardManager:
    """Propose executable rewards from task context and experiment history.

    Luna is deliberately outside the per-frame path. A request contains a
    compact scene schema, the current reward program, and aggregated experiment
    evidence. The returned program is locally validated before it can reach an
    environment worker.
    """

    def __init__(self, config: LunaRewardManagerConfig) -> None:
        self.config = config
        self.request_count = 0
        self.prompt_bytes = 0
        self.completion_bytes = 0
        self.total_seconds = 0.0

    def _credential(self) -> str:
        value = os.environ.get(self.config.api_key_env)
        if not value:
            raise LunaRewardManagerError(
                f"Missing credential environment variable {self.config.api_key_env}."
            )
        return value

    def _request(
        self, messages: Sequence[Mapping[str, str]]
    ) -> tuple[Mapping[str, Any], int, float]:
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.prompt_bytes += len(body)
        last_error = "unknown transport failure"
        for attempt in range(1, self.config.max_retries + 1):
            started = time.monotonic()
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._credential()}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    response_body = response.read()
                elapsed = time.monotonic() - started
                decoded = json.loads(response_body)
                if not isinstance(decoded, Mapping):
                    raise LunaRewardManagerError("Luna endpoint returned a non-object.")
                self.request_count += 1
                self.completion_bytes += len(response_body)
                self.total_seconds += elapsed
                return decoded, attempt, elapsed
            except urllib.error.HTTPError as error:
                last_error = f"HTTP {error.code}"
                retryable = error.code in {408, 409, 425, 429} or error.code >= 500
                if not retryable or attempt == self.config.max_retries:
                    break
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as error:
                last_error = type(error).__name__
                if attempt == self.config.max_retries:
                    break
            time.sleep(min(2 ** (attempt - 1), 8))
        raise LunaRewardManagerError(
            f"Luna request failed after {self.config.max_retries} attempts: "
            f"{last_error}."
        )

    def propose(
        self,
        *,
        scene_context: Mapping[str, Any],
        current_program: Mapping[str, Any],
        experiment_history: Sequence[Mapping[str, Any]],
        expected_gamma: float,
        reward_history: Sequence[Mapping[str, Any]] = (),
    ) -> LunaProposal:
        """Ask Luna for one locally validated physical reward revision."""
        available_keys = [str(key) for key in scene_context["available_physical_keys"]]
        contract = {
            "schema_version": 1,
            "required_fields": {
                "name": "short identifier",
                "rationale": "one concise causal hypothesis",
                "task_ids": scene_context["task_ids"],
                "gamma": expected_gamma,
                "completion_bonus": "fixed number 1.0",
                "completion_hold_steps": "integer in [1, 32]",
                "completion_reward_mode": (
                    "occupancy emits the bonus on every verified frame; "
                    "first_onset emits it only once per episode; capped_occupancy "
                    "emits on verified frames up to completion_reward_cap_steps"
                ),
                "completion_reward_cap_steps": (
                    "required integer in [2, 32] only for capped_occupancy"
                ),
                "completion_conditions": [
                    {
                        "type": (
                            "distance, scalar, relative_scalar, or delta_distance"
                        ),
                        "references": "available physical keys only",
                        "axes": "coordinate indices for distance/delta_distance",
                        "index": "coordinate for scalar/relative_scalar",
                        "op": "lt or gt",
                        "threshold": "physical-unit threshold",
                    }
                ],
                "potential_scale": ("number in [0, 0.25 * completion_bonus]"),
                "potential_terms": [
                    {
                        "type": ("distance, height_delta, scalar, or relative_scalar"),
                        "references": "available physical keys only",
                        "axes": "required coordinate indices for distance",
                        "index": (
                            "required coordinate for height_delta/scalar/relative_scalar; "
                            "for xyz positions, x=0, y=1, z=2 and height uses z=2"
                        ),
                        "target": "required for scalar/relative_scalar",
                        "scale": "positive physical-unit scale",
                        "weight": "nonnegative; all weights sum <= 1",
                    }
                ],
            },
            "type_specific_fields": {
                "completion.distance": "left, right, axes, op, threshold",
                "completion.scalar": "key, index, op, threshold",
                "completion.relative_scalar": "left, right, index, op, threshold",
                "completion.delta_distance": "key, axes, op, threshold",
                "potential.distance": "left, right, axes, scale, weight",
                "potential.height_delta": "key, index, scale, weight",
                "potential.scalar": "key, index, target, scale, weight",
                "potential.relative_scalar": (
                    "left, right, index, target, scale, weight"
                ),
            },
            "executed_formula": (
                "completion_reward(C(s_next), mode) + potential_scale * "
                "(gamma*Phi(s_next)-Phi(s))"
            ),
            "physical_semantics": {
                "relative_scalar": "left[index] - right[index]",
                "delta_distance": (
                    "norm of one tracked key's displacement since the prior frame; "
                    "usable as a real-sensor stability/contact proxy"
                ),
                "distance": "Euclidean separation over selected axes",
                "height_delta": "positive displacement from the episode reset pose",
                "coordinate_convention": (
                    "position arrays are xyz in metres: x=0, y=1, z=2; z is up/height"
                ),
            },
            "audit_semantics": {
                "physical_completion_tp_once": (
                    "physical verifier and independent task verifier both fired"
                ),
                "physical_completion_fp_once": (
                    "physical verifier fired but independent task verifier never did"
                ),
                "physical_completion_fn_once": (
                    "independent task verifier fired but physical verifier never did"
                ),
                "success_regressions": (
                    "task became false after first becoming true because rollout "
                    "continued; do not mistake this for a completion false positive"
                ),
                "condition_i_occupancy": (
                    "per-condition pass count; use it to identify the weak gate"
                ),
                "physical_completion_reward_return": (
                    "completion bonuses actually emitted to SAC; first_onset caps "
                    "this at one per episode, capped_occupancy caps it at the chosen "
                    "K, while occupancy can grow with horizon"
                ),
            },
            "forbidden_inputs": [
                "environment reward",
                "simulator success or task predicates",
                "termination or done flags",
                "teacher or demo actions",
                "policy actions",
                "images",
            ],
        }
        evidence = {
            "scene": scene_context,
            "current_program": current_program,
            "recent_experiments": [
                _compact_experiment(experiment)
                for experiment in experiment_history[-8:]
            ],
            "past_reward_trials": list(reward_history[-8:]),
            "contract": contract,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the reward-evolution module in an ENPIRE-style "
                    "robot RL loop, using a Reward-as-An-Agent verification-first "
                    "design. Return exactly one JSON reward program. First infer "
                    "the task stages and physical completion semantics from the "
                    "task and scene; then diagnose verifier precision/recall, "
                    "stability, and SAC learnability from recent experiments. "
                    "Use TP/FP/FN and per-condition evidence for verifier changes. "
                    "Treat past_reward_trials as causal memory: never return an "
                    "exact reward that was rolled back, and after a rollback change "
                    "a structural hypothesis instead of merely cycling old thresholds. "
                    "Do not infer false positives from completion occupancy alone: "
                    "rollouts continue after success, so occupancy and "
                    "success_at_end are not directly comparable. success_regressions "
                    "describe policy behavior, not verifier error, so never tighten a "
                    "completion gate solely to reduce that metric. For an object "
                    "placed on a support, prefer signed relative height plus XY "
                    "alignment and low inter-frame displacement over absolute "
                    "height distance alone. When repeated occupancy-style candidates "
                    "show rising Q scale or critic loss followed by policy regression, "
                    "make a structural temporal change: consider first_onset completion "
                    "and zero or very small potential shaping instead of another small "
                    "threshold or weight adjustment. If first_onset then keeps Q and "
                    "critic loss stable but its Q contribution is dwarfed by the actor "
                    "objective and fixed-panel success does not improve, consider "
                    "capped_occupancy with a small K rather than returning to unbounded "
                    "occupancy. Keep potential terms bounded and "
                    "stage-aligned. Independent simulator success may be used only "
                    "as audit evidence here and must never appear in the executable "
                    "program. Do not invent unavailable signals, code, hard "
                    "curricula, images, or policy actions."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(evidence, sort_keys=True, separators=(",", ":")),
            },
        ]
        total_elapsed = 0.0
        total_usage: dict[str, int] = {}
        last_error = "unknown local validation failure"
        for schema_attempt in range(1, self.config.max_retries + 1):
            decoded, _, elapsed = self._request(messages)
            total_elapsed += elapsed
            usage_raw = decoded.get("usage", {})
            if isinstance(usage_raw, Mapping):
                for key, value in usage_raw.items():
                    if isinstance(value, int):
                        total_usage[str(key)] = total_usage.get(str(key), 0) + value

            choices = decoded.get("choices")
            message = (
                choices[0].get("message", {})
                if isinstance(choices, list) and choices
                else {}
            )
            response_text = (
                message.get("content") if isinstance(message, Mapping) else None
            )
            try:
                if not isinstance(response_text, str):
                    raise LunaRewardManagerError("Luna response has no text content.")
                raw_program = _extract_json_object(response_text)
                program = validate_physical_reward_program(
                    raw_program,
                    available_keys=available_keys,
                    expected_gamma=expected_gamma,
                )
                rejected_digests = {
                    str(trial.get("digest"))
                    for trial in reward_history
                    if str(trial.get("terminal_action", "")).startswith("rollback")
                }
                if physical_program_digest(program) in rejected_digests:
                    raise PhysicalRewardProgramError(
                        "This exact reward program was already rolled back; propose a "
                        "structurally different hypothesis."
                    )
            except (LunaRewardManagerError, PhysicalRewardProgramError) as error:
                last_error = str(error)
                if schema_attempt == self.config.max_retries:
                    break
                if isinstance(response_text, str):
                    messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The proposed JSON failed local validation: "
                            f"{last_error} Return the complete corrected reward program, "
                            "not a patch. Follow every type_specific_fields entry and "
                            "use only available_physical_keys."
                        ),
                    }
                )
                continue
            return LunaProposal(
                program=program,
                response_id=(
                    str(decoded["id"]) if decoded.get("id") is not None else None
                ),
                usage=total_usage,
                elapsed_seconds=total_elapsed,
                attempt=schema_attempt,
            )
        raise LunaRewardManagerError(
            "Luna failed local reward validation after "
            f"{self.config.max_retries} attempts: {last_error}"
        )

    def audit(self) -> dict[str, Any]:
        """Return cumulative, secret-free transport accounting."""
        return {
            "base_url": self.config.base_url,
            "model": self.config.model,
            "request_count": self.request_count,
            "prompt_bytes": self.prompt_bytes,
            "completion_bytes": self.completion_bytes,
            "total_seconds": self.total_seconds,
        }


def proposal_audit(proposal: LunaProposal) -> dict[str, Any]:
    """Serialize proposal provenance without reasoning or credentials."""
    return {
        "response_id": proposal.response_id,
        "usage": dict(proposal.usage),
        "elapsed_seconds": proposal.elapsed_seconds,
        "attempt": proposal.attempt,
        "program_digest": physical_program_digest(proposal.program),
    }
