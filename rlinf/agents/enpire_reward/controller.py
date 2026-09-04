# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Measured propose-run-evaluate-accept/rollback reward evolution."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from rlinf.agents.enpire_reward.manager import (
    LunaRewardManager,
    LunaRewardManagerConfig,
    LunaRewardManagerError,
    proposal_audit,
)
from rlinf.agents.enpire_reward.physical_potential import (
    atomic_write_physical_reward_program,
    physical_program_digest,
    validate_physical_reward_program,
)


@dataclass(frozen=True)
class EvolutionDecision:
    """Result of one evaluation boundary."""

    action: str
    score: float
    champion_score: float
    rollback_checkpoint: str | None
    candidate_digest: str | None
    manager_called: bool


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return str(value)


class RewardEvolutionController:
    """Persist and advance an ENPIRE-style reward experiment campaign."""

    def __init__(
        self,
        *,
        program_path: str | Path,
        scene_context_path: str | Path,
        audit_dir: str | Path,
        expected_gamma: float,
        score_keys: Sequence[str],
        score_panels: Mapping[str, Mapping[str, str]] | None,
        target_score: float,
        min_improvement: float,
        regression_tolerance: float,
        manager_config: LunaRewardManagerConfig,
        bootstrap_reward_history_path: str | Path | None = None,
        baseline_warmup_evaluations: int = 1,
        candidate_burn_in_evaluations: int = 0,
        candidate_min_evaluations: int = 1,
        candidate_patience_evaluations: int = 1,
        champion_patience_evaluations: int = 2,
    ) -> None:
        self.program_path = Path(program_path)
        self.scene_context_path = Path(scene_context_path)
        self.audit_dir = Path(audit_dir)
        self.state_path = self.audit_dir / "state.json"
        self.event_path = self.audit_dir / "events.jsonl"
        self.report_path = self.audit_dir / "report.md"
        self.expected_gamma = float(expected_gamma)
        self.score_keys = tuple(str(key) for key in score_keys)
        self.score_panels = {
            str(name): {str(key): str(value) for key, value in spec.items()}
            for name, spec in (score_panels or {}).items()
        }
        self.target_score = float(target_score)
        self.min_improvement = float(min_improvement)
        self.regression_tolerance = float(regression_tolerance)
        self.baseline_warmup_evaluations = int(baseline_warmup_evaluations)
        self.candidate_burn_in_evaluations = int(candidate_burn_in_evaluations)
        self.candidate_min_evaluations = int(candidate_min_evaluations)
        self.candidate_patience_evaluations = int(candidate_patience_evaluations)
        self.champion_patience_evaluations = int(champion_patience_evaluations)
        if self.baseline_warmup_evaluations < 1:
            raise ValueError("baseline_warmup_evaluations must be positive.")
        if self.candidate_patience_evaluations < 1:
            raise ValueError("candidate_patience_evaluations must be positive.")
        if self.candidate_burn_in_evaluations < 0:
            raise ValueError("candidate_burn_in_evaluations must be non-negative.")
        if (
            not 1
            <= self.candidate_min_evaluations
            <= self.candidate_patience_evaluations
        ):
            raise ValueError(
                "candidate_min_evaluations must be in [1, candidate_patience_evaluations]."
            )
        if self.champion_patience_evaluations < 1:
            raise ValueError("champion_patience_evaluations must be positive.")
        if not self.score_keys and not self.score_panels:
            raise ValueError("Agentic reward requires score keys or score panels.")
        self.scene_context = json.loads(self.scene_context_path.read_text())
        self.manager = LunaRewardManager(manager_config)
        bootstrap_reward_history = self._load_bootstrap_reward_history(
            bootstrap_reward_history_path
        )
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            seed_program = validate_physical_reward_program(
                json.loads(self.program_path.read_text()),
                available_keys=self.scene_context["available_physical_keys"],
                expected_gamma=self.expected_gamma,
            )
            atomic_write_physical_reward_program(self.program_path, seed_program)
            self.state = {
                "schema_version": 1,
                "champion_program": seed_program,
                "champion_score": None,
                "champion_checkpoint": None,
                "candidate_pending": False,
                "candidate_digest": None,
                "candidate_evaluations": 0,
                "candidate_burn_in_completed": 0,
                "candidate_recent_scores": [],
                "baseline_evaluations": 0,
                "champion_stale_evaluations": 0,
                "champion_improvement_scores": [],
                "manager_cycle_started": False,
                "experiments": [],
                "reward_trials": bootstrap_reward_history,
                "manager": self.manager.audit(),
            }
            self._save_state()
        self.state.setdefault("candidate_evaluations", 0)
        self.state.setdefault("candidate_burn_in_completed", 0)
        self.state.setdefault("candidate_recent_scores", [])
        self.state.setdefault("baseline_evaluations", 0)
        self.state.setdefault("champion_stale_evaluations", 0)
        self.state.setdefault("champion_improvement_scores", [])
        self.state.setdefault("manager_cycle_started", False)
        self.state.setdefault("reward_trials", [])
        self._write_report()

    def _load_bootstrap_reward_history(
        self, path: str | Path | None
    ) -> list[dict[str, Any]]:
        """Load validated, secret-free reward trials from an earlier campaign."""
        if path is None or not str(path):
            return []
        decoded = json.loads(Path(path).read_text())
        if not isinstance(decoded, list):
            raise ValueError("bootstrap_reward_history_path must contain a JSON list.")
        trials = []
        for index, raw_trial in enumerate(decoded):
            if not isinstance(raw_trial, Mapping):
                raise ValueError(f"Bootstrap reward trial {index} must be an object.")
            program = validate_physical_reward_program(
                raw_trial.get("program", {}),
                available_keys=self.scene_context["available_physical_keys"],
                expected_gamma=self.expected_gamma,
            )
            digest = physical_program_digest(program)
            recorded_digest = raw_trial.get("digest")
            if recorded_digest is not None and str(recorded_digest) != digest:
                raise ValueError(f"Bootstrap reward trial {index} digest is invalid.")
            evaluations = raw_trial.get("evaluations", [])
            if not isinstance(evaluations, list):
                raise ValueError(
                    f"Bootstrap reward trial {index} evaluations must be a list."
                )
            terminal_action = raw_trial.get("terminal_action")
            if terminal_action is not None and not isinstance(terminal_action, str):
                raise ValueError(
                    f"Bootstrap reward trial {index} terminal_action must be text."
                )
            trials.append(
                {
                    "digest": digest,
                    "program": program,
                    "evaluations": evaluations,
                    "terminal_action": terminal_action,
                    "source": str(path),
                }
            )
        return trials

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RewardEvolutionController":
        manager = config.get("manager", {})
        return cls(
            program_path=config["program_path"],
            scene_context_path=config["scene_context_path"],
            audit_dir=config["audit_dir"],
            expected_gamma=config["gamma"],
            score_keys=config.get(
                "score_keys", ["eval/success_once", "eval/success_at_end"]
            ),
            score_panels=config.get("score_panels"),
            target_score=config.get("target_score", 0.7),
            min_improvement=config.get("min_improvement", 0.02),
            regression_tolerance=config.get("regression_tolerance", 0.02),
            baseline_warmup_evaluations=config.get("baseline_warmup_evaluations", 1),
            candidate_patience_evaluations=config.get(
                "candidate_patience_evaluations", 1
            ),
            candidate_burn_in_evaluations=config.get(
                "candidate_burn_in_evaluations", 0
            ),
            candidate_min_evaluations=config.get("candidate_min_evaluations", 1),
            champion_patience_evaluations=config.get(
                "champion_patience_evaluations", 2
            ),
            bootstrap_reward_history_path=config.get("bootstrap_reward_history_path"),
            manager_config=LunaRewardManagerConfig(
                base_url=manager.get("base_url", "https://maimai.it.com"),
                model=manager.get("model", "gpt-5.6-luna"),
                api_key_env=manager.get("api_key_env", "AGENTIC_MODEL_API_KEY"),
                timeout_seconds=manager.get("timeout_seconds", 180.0),
                max_retries=manager.get("max_retries", 3),
            ),
        )

    def _save_state(self) -> None:
        temporary = self.state_path.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.state_path)

    def _append_event(self, event: Mapping[str, Any]) -> None:
        record = dict(event)
        record.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        with self.event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.event_path.exists():
            return []
        events = []
        for line in self.event_path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def _write_report(self) -> None:
        """Atomically refresh a secret-free human-readable campaign audit."""
        events = self._read_events()
        evaluations = [event for event in events if event.get("kind") == "evaluation"]
        proposals = [event for event in events if event.get("kind") == "proposal"]
        failures = [event for event in events if event.get("kind") == "manager_failure"]
        rollbacks = [
            event
            for event in evaluations
            if str(event.get("action", "")).startswith("rollback")
        ]
        token_total = sum(
            int(event.get("proposal", {}).get("usage", {}).get("total_tokens", 0))
            for event in proposals
        )
        current = json.loads(self.program_path.read_text())
        manager = self.state.get("manager", {})
        lines = [
            "# Agentic Reward Audit",
            "",
            f"Updated: {datetime.now(timezone.utc).isoformat()}",
            "",
            f"- Current reward: `{current.get('name', 'unknown')}`",
            f"- Current rationale: {current.get('rationale', '')}",
            f"- Conservative champion score: {self.state.get('champion_score')}",
            f"- Luna model: `{manager.get('model', self.manager.config.model)}`",
            f"- Luna requests: {manager.get('request_count', 0)}",
            f"- Luna tokens recorded: {token_total}",
            f"- Luna wall time: {float(manager.get('total_seconds', 0.0)):.3f}s",
            f"- Rollbacks: {len(rollbacks)}",
            f"- Manager failures: {len(failures)}",
            "",
            "## Evaluation History",
            "",
            "| Step | Action | Min score | Panels | Physical TP/FP/FN | Elite samples |",
            "| ---: | --- | ---: | --- | --- | ---: |",
        ]
        for event in evaluations:
            metrics = event.get("metrics", {})
            panel = ", ".join(
                f"{key}={float(value):.3f}"
                for key, value in event.get("score_panel", {}).items()
            )
            physical = "/".join(
                str(metrics.get(f"env/physical_completion_{kind}_once", "-"))
                for kind in ("tp", "fp", "fn")
            )
            elite_samples = metrics.get("train/demo_buffer/total_samples", "-")
            lines.append(
                f"| {event.get('step', '-')} | {event.get('action', '-')} | "
                f"{float(event.get('score', 0.0)):.3f} | {panel} | {physical} | "
                f"{elite_samples} |"
            )
        lines.extend(
            [
                "",
                "## Guardrails",
                "",
                "The executable reward uses only scene physical observations. "
                "Simulator reward, success predicates, actions, images, and done flags "
                "are forbidden reward inputs. Independent success is retained only for "
                "evaluation and verifier precision/recall auditing.",
                "",
            ]
        )
        temporary = self.report_path.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text("\n".join(lines))
        os.replace(temporary, self.report_path)

    def _score(self, metrics: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
        panel: dict[str, float] = {}
        for name, spec in self.score_panels.items():
            numerator_key = spec.get("numerator")
            denominator_key = spec.get("denominator")
            if numerator_key not in metrics or denominator_key not in metrics:
                raise ValueError(
                    f"Evaluation did not produce panel metrics for {name!r}."
                )
            numerator = float(metrics[numerator_key])
            denominator = float(metrics[denominator_key])
            if (
                not math.isfinite(numerator)
                or not math.isfinite(denominator)
                or denominator <= 0.0
            ):
                raise ValueError(f"Evaluation panel {name!r} is invalid.")
            panel[name] = numerator / denominator
        for key in self.score_keys:
            if key not in metrics:
                raise ValueError(f"Evaluation did not produce required metric {key!r}.")
            value = float(metrics[key])
            if not math.isfinite(value):
                raise ValueError(f"Evaluation metric {key!r} is not finite.")
            panel[key] = value
        return min(panel.values()), panel

    def _propose_next(self) -> tuple[str | None, dict[str, Any] | None]:
        current_program = json.loads(self.program_path.read_text())
        try:
            proposal = self.manager.propose(
                scene_context=self.scene_context,
                current_program=current_program,
                experiment_history=self.state["experiments"],
                expected_gamma=self.expected_gamma,
                reward_history=self.state["reward_trials"],
            )
        except LunaRewardManagerError as error:
            event = {
                "kind": "manager_failure",
                "error": str(error),
                "manager": self.manager.audit(),
            }
            self._append_event(event)
            return None, event
        atomic_write_physical_reward_program(self.program_path, proposal.program)
        digest = physical_program_digest(proposal.program)
        self.state["candidate_pending"] = True
        self.state["candidate_digest"] = digest
        self.state["candidate_evaluations"] = 0
        self.state["candidate_burn_in_completed"] = 0
        self.state["candidate_recent_scores"] = []
        self.state["champion_stale_evaluations"] = 0
        self.state["manager"] = self.manager.audit()
        self.state["reward_trials"].append(
            {
                "digest": digest,
                "program": dict(proposal.program),
                "evaluations": [],
                "terminal_action": None,
            }
        )
        event = {
            "kind": "proposal",
            "proposal": proposal_audit(proposal),
            "program": dict(proposal.program),
            "manager": self.manager.audit(),
        }
        self._append_event(event)
        return digest, event

    def process_evaluation(
        self,
        *,
        step: int,
        metrics: Mapping[str, Any],
        checkpoint_path: str,
    ) -> EvolutionDecision:
        """Advance warmup or evaluate a reward candidate at a fixed boundary."""
        score, panel = self._score(metrics)
        evaluated_program = json.loads(self.program_path.read_text())
        evaluated_reward_digest = physical_program_digest(evaluated_program)
        checkpoint_path = str(Path(checkpoint_path).resolve())
        rollback_checkpoint = None
        action = "baseline"
        candidate_evaluation = 0
        cycle_started = bool(self.state["manager_cycle_started"])

        if not cycle_started:
            baseline_evaluations = int(self.state["baseline_evaluations"]) + 1
            self.state["baseline_evaluations"] = baseline_evaluations
            self.state["champion_score"] = score
            self.state["champion_checkpoint"] = checkpoint_path
            self.state["champion_program"] = json.loads(self.program_path.read_text())
            if baseline_evaluations < self.baseline_warmup_evaluations:
                action = "baseline_warmup"
            else:
                self.state["manager_cycle_started"] = True
                cycle_started = True
        elif self.state["candidate_pending"]:
            burn_in_completed = int(self.state["candidate_burn_in_completed"])
            if burn_in_completed < self.candidate_burn_in_evaluations:
                burn_in_completed += 1
                self.state["candidate_burn_in_completed"] = burn_in_completed
                action = "candidate_burn_in"
            else:
                candidate_evaluations = int(self.state["candidate_evaluations"]) + 1
                self.state["candidate_evaluations"] = candidate_evaluations
                candidate_evaluation = candidate_evaluations
                recent_scores = list(self.state["candidate_recent_scores"])
                recent_scores.append(score)
                recent_scores = recent_scores[-self.candidate_min_evaluations :]
                self.state["candidate_recent_scores"] = recent_scores
                champion_score = float(self.state["champion_score"])
                enough_evidence = (
                    candidate_evaluations >= self.candidate_min_evaluations
                )
                stable_improvement = enough_evidence and all(
                    value - champion_score >= self.min_improvement
                    for value in recent_scores
                )
                stable_regression = enough_evidence and all(
                    value - champion_score < -self.regression_tolerance
                    for value in recent_scores
                )
                if stable_improvement:
                    action = "accept"
                    self.state["champion_score"] = min(recent_scores)
                    self.state["champion_checkpoint"] = checkpoint_path
                    self.state["champion_program"] = json.loads(
                        self.program_path.read_text()
                    )
                    self.state["champion_stale_evaluations"] = 0
                    self.state["candidate_pending"] = False
                elif stable_regression:
                    action = "rollback_regression"
                    rollback_checkpoint = self.state["champion_checkpoint"]
                    atomic_write_physical_reward_program(
                        self.program_path, self.state["champion_program"]
                    )
                    self.state["candidate_pending"] = False
                elif candidate_evaluations < self.candidate_patience_evaluations:
                    action = "continue_candidate"
                else:
                    action = "rollback_no_gain"
                    rollback_checkpoint = self.state["champion_checkpoint"]
                    atomic_write_physical_reward_program(
                        self.program_path, self.state["champion_program"]
                    )
                    self.state["candidate_pending"] = False

            if not self.state["candidate_pending"]:
                self.state["candidate_digest"] = None
                self.state["candidate_evaluations"] = 0
                self.state["candidate_burn_in_completed"] = 0
                self.state["candidate_recent_scores"] = []
        else:
            action = "champion_continue"
            champion_score = float(self.state["champion_score"])
            if score >= champion_score + self.min_improvement:
                improvement_scores = list(self.state["champion_improvement_scores"])
                improvement_scores.append(score)
                improvement_scores = improvement_scores[
                    -self.candidate_min_evaluations :
                ]
                self.state["champion_improvement_scores"] = improvement_scores
                if len(improvement_scores) >= self.candidate_min_evaluations:
                    self.state["champion_score"] = min(improvement_scores)
                    self.state["champion_checkpoint"] = checkpoint_path
                    self.state["champion_stale_evaluations"] = 0
                    self.state["champion_improvement_scores"] = []
                else:
                    action = "champion_improvement_pending"
            else:
                self.state["champion_improvement_scores"] = []
                stale_evaluations = int(self.state["champion_stale_evaluations"]) + 1
                self.state["champion_stale_evaluations"] = stale_evaluations
                if stale_evaluations >= self.champion_patience_evaluations:
                    if score < champion_score - self.regression_tolerance:
                        action = "rollback_champion_regression"
                        rollback_checkpoint = self.state["champion_checkpoint"]
                    else:
                        action = "champion_plateau"

        experiment = {
            "kind": "evaluation",
            "step": int(step),
            "action": action,
            "score": score,
            "score_panel": panel,
            "checkpoint_path": checkpoint_path,
            "rollback_checkpoint": rollback_checkpoint,
            "candidate_evaluation": candidate_evaluation,
            "candidate_burn_in_completed": int(
                self.state["candidate_burn_in_completed"]
            ),
            "reward_digest": evaluated_reward_digest,
            "metrics": {
                str(key): _json_scalar(value) for key, value in metrics.items()
            },
        }
        self.state["experiments"].append(experiment)
        for trial in reversed(self.state["reward_trials"]):
            if trial.get("digest") != evaluated_reward_digest:
                continue
            trial["evaluations"].append(
                {
                    "step": int(step),
                    "action": action,
                    "score": score,
                    "score_panel": panel,
                    "physical_tp_once": _json_scalar(
                        metrics.get("env/physical_completion_tp_once")
                    ),
                    "physical_fp_once": _json_scalar(
                        metrics.get("env/physical_completion_fp_once")
                    ),
                    "physical_fn_once": _json_scalar(
                        metrics.get("env/physical_completion_fn_once")
                    ),
                    "q_data": _json_scalar(metrics.get("train/critic/q_data")),
                }
            )
            if action.startswith("rollback") or action == "accept":
                trial["terminal_action"] = action
            break
        self._append_event(experiment)

        champion_score = float(self.state["champion_score"])
        candidate_digest = None
        manager_called = False
        propose_after = {
            "baseline",
            "rollback_regression",
            "rollback_no_gain",
            "rollback_champion_regression",
            "champion_plateau",
        }
        if (
            cycle_started
            and not self.state["candidate_pending"]
            and champion_score < self.target_score
            and action in propose_after
        ):
            candidate_digest, _ = self._propose_next()
            manager_called = candidate_digest is not None
        self._save_state()
        self._write_report()
        return EvolutionDecision(
            action=action,
            score=score,
            champion_score=champion_score,
            rollback_checkpoint=rollback_checkpoint,
            candidate_digest=candidate_digest,
            manager_called=manager_called,
        )
