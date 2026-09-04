from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rlinf.agents.enpire_reward.controller import RewardEvolutionController
from rlinf.agents.enpire_reward.manager import (
    LunaProposal,
    LunaRewardManager,
    LunaRewardManagerConfig,
)
from rlinf.agents.enpire_reward.physical_potential import (
    PhysicalPotentialRewardRuntime,
    PhysicalRewardProgramError,
    atomic_write_physical_reward_program,
    physical_program_digest,
    validate_physical_reward_program,
)


def _program(**overrides):
    program = {
        "schema_version": 1,
        "name": "test_reward",
        "rationale": "test",
        "task_ids": [9],
        "gamma": 0.99,
        "completion_bonus": 1.0,
        "completion_hold_steps": 2,
        "completion_reward_mode": "occupancy",
        "completion_conditions": [
            {
                "type": "distance",
                "left": "object_pos",
                "right": "target_pos",
                "axes": [0, 1, 2],
                "op": "lt",
                "threshold": 0.1,
            }
        ],
        "potential_scale": 0.2,
        "potential_terms": [
            {
                "type": "distance",
                "left": "object_pos",
                "right": "target_pos",
                "axes": [0, 1, 2],
                "scale": 0.5,
                "weight": 1.0,
            }
        ],
    }
    program.update(overrides)
    return program


def _obs(x):
    return {
        "object_pos": np.asarray([x, 0.0, 0.0]),
        "target_pos": np.asarray([1.0, 0.0, 0.0]),
    }


def test_validator_rejects_leakage_and_gamma_mismatch():
    leaked = _program(
        completion_conditions=[
            {
                "type": "scalar",
                "key": "sim_success",
                "index": 0,
                "op": "gt",
                "threshold": 0.5,
            }
        ]
    )
    with pytest.raises(PhysicalRewardProgramError, match="forbidden signal"):
        validate_physical_reward_program(leaked)
    with pytest.raises(PhysicalRewardProgramError, match="does not match"):
        validate_physical_reward_program(_program(), expected_gamma=0.999)


def test_validator_rejects_missing_coordinate_index():
    missing_index = _program(
        potential_terms=[
            {
                "type": "height_delta",
                "key": "object_pos",
                "scale": 0.5,
                "weight": 1.0,
            }
        ]
    )
    with pytest.raises(
        PhysicalRewardProgramError, match=r"term\[0\]\.index is required"
    ):
        validate_physical_reward_program(missing_index)


def test_manager_retries_invalid_program_with_validator_feedback(monkeypatch):
    invalid = _program(
        potential_terms=[
            {
                "type": "height_delta",
                "key": "object_pos",
                "scale": 0.5,
                "weight": 1.0,
            }
        ]
    )
    valid = _program()
    responses = [invalid, valid]
    observed_messages = []
    manager = LunaRewardManager(LunaRewardManagerConfig(max_retries=2))

    def fake_request(messages):
        observed_messages.append(list(messages))
        program = responses.pop(0)
        return (
            {
                "id": f"response-{len(observed_messages)}",
                "choices": [{"message": {"content": json.dumps(program)}}],
                "usage": {"total_tokens": 10},
            },
            1,
            0.1,
        )

    monkeypatch.setattr(manager, "_request", fake_request)
    proposal = manager.propose(
        scene_context={
            "task_ids": [9],
            "available_physical_keys": ["object_pos", "target_pos"],
        },
        current_program=valid,
        experiment_history=[],
        expected_gamma=0.99,
    )

    assert proposal.attempt == 2
    assert proposal.usage["total_tokens"] == 20
    assert proposal.program == validate_physical_reward_program(valid)
    assert "term[0].index is required" in observed_messages[1][-1]["content"]


def test_potential_progress_and_completion_hold(tmp_path):
    path = tmp_path / "reward.json"
    atomic_write_physical_reward_program(path, _program())
    runtime = PhysicalPotentialRewardRuntime(
        path, num_envs=1, expected_gamma=0.99, reload_interval_steps=16
    )
    runtime.reset([_obs(0.0)])

    progress = runtime.compute([_obs(0.5)], [9])
    assert progress.completion.tolist() == [0.0]
    assert progress.potential_delta[0] > 0.0
    assert progress.rewards[0] > 0.0

    first_complete = runtime.compute([_obs(0.95)], [9])
    second_complete = runtime.compute([_obs(0.95)], [9])
    assert first_complete.completion.tolist() == [0.0]
    assert second_complete.completion.tolist() == [1.0]
    assert second_complete.rewards[0] > 0.9


def test_first_onset_completion_reward_is_capped_per_episode(tmp_path):
    path = tmp_path / "reward.json"
    atomic_write_physical_reward_program(
        path, _program(completion_reward_mode="first_onset")
    )
    runtime = PhysicalPotentialRewardRuntime(
        path, num_envs=1, expected_gamma=0.99, reload_interval_steps=16
    )
    incomplete = _obs(0.0)
    complete = _obs(0.95)
    runtime.reset([incomplete])

    runtime.compute([complete], [9])
    onset = runtime.compute([complete], [9])
    sustained = runtime.compute([complete], [9])
    runtime.compute([incomplete], [9])
    reacquired_1 = runtime.compute([complete], [9])
    reacquired_2 = runtime.compute([complete], [9])

    assert onset.completion.tolist() == [1.0]
    assert onset.completion_reward.tolist() == [1.0]
    assert sustained.completion.tolist() == [1.0]
    assert sustained.completion_reward.tolist() == [0.0]
    assert reacquired_1.completion.tolist() == [0.0]
    assert reacquired_2.completion.tolist() == [1.0]
    assert reacquired_2.completion_reward.tolist() == [0.0]

    runtime.reset([incomplete])
    runtime.compute([complete], [9])
    next_episode_onset = runtime.compute([complete], [9])
    assert next_episode_onset.completion_reward.tolist() == [1.0]


def test_capped_occupancy_limits_reward_but_not_verifier(tmp_path):
    path = tmp_path / "reward.json"
    atomic_write_physical_reward_program(
        path,
        _program(
            completion_reward_mode="capped_occupancy",
            completion_reward_cap_steps=3,
        ),
    )
    runtime = PhysicalPotentialRewardRuntime(
        path, num_envs=1, expected_gamma=0.99, reload_interval_steps=16
    )
    incomplete = _obs(0.0)
    complete = _obs(0.95)
    runtime.reset([incomplete])

    steps = [runtime.compute([complete], [9]) for _ in range(6)]

    assert [step.completion_reward.item() for step in steps] == [0, 1, 1, 1, 0, 0]
    assert [step.completion.item() for step in steps] == [0, 1, 1, 1, 1, 1]

    runtime.reset([incomplete])
    next_episode = [runtime.compute([complete], [9]) for _ in range(2)]
    assert next_episode[-1].completion_reward.tolist() == [1.0]


def test_hot_reload_suppresses_artificial_potential_impulse(tmp_path):
    path = tmp_path / "reward.json"
    atomic_write_physical_reward_program(path, _program())
    runtime = PhysicalPotentialRewardRuntime(
        path, num_envs=1, expected_gamma=0.99, reload_interval_steps=1
    )
    runtime.reset([_obs(0.0)])
    runtime.compute([_obs(0.2)], [9])

    changed = _program(
        name="changed",
        potential_terms=[
            {
                "type": "distance",
                "left": "object_pos",
                "right": "target_pos",
                "axes": [0, 1, 2],
                "scale": 0.1,
                "weight": 1.0,
            }
        ],
    )
    atomic_write_physical_reward_program(path, changed)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    result = runtime.compute([_obs(0.2)], [9])
    assert result.potential_delta.tolist() == [0.0]
    assert result.rewards.tolist() == [0.0]


def test_controller_scores_disjoint_panel_ratios(tmp_path):
    program_path = tmp_path / "reward.json"
    scene_path = tmp_path / "scene.json"
    atomic_write_physical_reward_program(program_path, _program())
    scene_path.write_text(
        json.dumps(
            {
                "task_ids": [9],
                "available_physical_keys": ["object_pos", "target_pos"],
            }
        )
    )
    controller = RewardEvolutionController(
        program_path=program_path,
        scene_context_path=scene_path,
        audit_dir=tmp_path / "audit",
        expected_gamma=0.99,
        score_keys=[],
        score_panels={
            "a": {"numerator": "a_success", "denominator": "a_mass"},
            "b": {"numerator": "b_success", "denominator": "b_mass"},
        },
        target_score=0.7,
        min_improvement=0.02,
        regression_tolerance=0.02,
        manager_config=LunaRewardManagerConfig(),
    )
    score, panels = controller._score(
        {"a_success": 0.4, "a_mass": 0.5, "b_success": 0.35, "b_mass": 0.5}
    )
    assert panels == pytest.approx({"a": 0.8, "b": 0.7})
    assert score == pytest.approx(0.7)
    report = (tmp_path / "audit" / "report.md").read_text()
    assert "Agentic Reward Audit" in report
    assert "Simulator reward" in report


def test_controller_warmup_and_candidate_patience(tmp_path, monkeypatch):
    program_path = tmp_path / "reward.json"
    scene_path = tmp_path / "scene.json"
    atomic_write_physical_reward_program(program_path, _program())
    scene_path.write_text(
        json.dumps(
            {
                "task_ids": [9],
                "available_physical_keys": ["object_pos", "target_pos"],
            }
        )
    )
    controller = RewardEvolutionController(
        program_path=program_path,
        scene_context_path=scene_path,
        audit_dir=tmp_path / "audit",
        expected_gamma=0.99,
        score_keys=[],
        score_panels={
            "a": {"numerator": "a_success", "denominator": "a_mass"},
            "b": {"numerator": "b_success", "denominator": "b_mass"},
        },
        target_score=0.7,
        min_improvement=0.02,
        regression_tolerance=0.02,
        manager_config=LunaRewardManagerConfig(),
        baseline_warmup_evaluations=2,
        candidate_burn_in_evaluations=1,
        candidate_min_evaluations=2,
        candidate_patience_evaluations=3,
    )
    proposal_count = 0

    def fake_proposal():
        nonlocal proposal_count
        proposal_count += 1
        digest = f"candidate-{proposal_count}"
        controller.state["candidate_pending"] = True
        controller.state["candidate_digest"] = digest
        controller.state["candidate_evaluations"] = 0
        return digest, {}

    monkeypatch.setattr(controller, "_propose_next", fake_proposal)
    metrics = {
        "a_success": 0.0,
        "a_mass": 0.5,
        "b_success": 0.0,
        "b_mass": 0.5,
    }

    first = controller.process_evaluation(
        step=5, metrics=metrics, checkpoint_path=tmp_path / "step5"
    )
    assert first.action == "baseline_warmup"
    assert not first.manager_called

    second = controller.process_evaluation(
        step=10, metrics=metrics, checkpoint_path=tmp_path / "step10"
    )
    assert second.action == "baseline"
    assert second.manager_called

    third = controller.process_evaluation(
        step=15, metrics=metrics, checkpoint_path=tmp_path / "step15"
    )
    assert third.action == "candidate_burn_in"
    assert third.rollback_checkpoint is None
    assert not third.manager_called

    fourth = controller.process_evaluation(
        step=20, metrics=metrics, checkpoint_path=tmp_path / "step20"
    )
    assert fourth.action == "continue_candidate"
    assert fourth.rollback_checkpoint is None
    assert not fourth.manager_called

    fifth = controller.process_evaluation(
        step=25, metrics=metrics, checkpoint_path=tmp_path / "step25"
    )
    assert fifth.action == "continue_candidate"
    assert fifth.rollback_checkpoint is None
    assert not fifth.manager_called

    sixth = controller.process_evaluation(
        step=30, metrics=metrics, checkpoint_path=tmp_path / "step30"
    )
    assert sixth.action == "rollback_no_gain"
    assert sixth.rollback_checkpoint.endswith("step10")
    assert sixth.manager_called


def test_controller_accepts_stable_gain_then_consolidates(tmp_path, monkeypatch):
    program_path = tmp_path / "reward.json"
    scene_path = tmp_path / "scene.json"
    atomic_write_physical_reward_program(program_path, _program())
    scene_path.write_text(
        json.dumps(
            {
                "task_ids": [9],
                "available_physical_keys": ["object_pos", "target_pos"],
            }
        )
    )
    controller = RewardEvolutionController(
        program_path=program_path,
        scene_context_path=scene_path,
        audit_dir=tmp_path / "audit",
        expected_gamma=0.99,
        score_keys=[],
        score_panels={
            "a": {"numerator": "a_success", "denominator": "a_mass"},
            "b": {"numerator": "b_success", "denominator": "b_mass"},
        },
        target_score=0.7,
        min_improvement=0.02,
        regression_tolerance=0.02,
        manager_config=LunaRewardManagerConfig(),
        candidate_min_evaluations=2,
        candidate_patience_evaluations=3,
        champion_patience_evaluations=2,
    )
    proposal_count = 0

    def fake_proposal():
        nonlocal proposal_count
        proposal_count += 1
        digest = f"candidate-{proposal_count}"
        controller.state["candidate_pending"] = True
        controller.state["candidate_digest"] = digest
        controller.state["candidate_evaluations"] = 0
        return digest, {}

    monkeypatch.setattr(controller, "_propose_next", fake_proposal)
    baseline = {
        "a_success": 0.05,
        "a_mass": 0.5,
        "b_success": 0.05,
        "b_mass": 0.5,
    }
    improved = {
        "a_success": 0.15,
        "a_mass": 0.5,
        "b_success": 0.15,
        "b_mass": 0.5,
    }

    first = controller.process_evaluation(
        step=5, metrics=baseline, checkpoint_path=tmp_path / "step5"
    )
    assert first.manager_called

    second = controller.process_evaluation(
        step=10, metrics=improved, checkpoint_path=tmp_path / "step10"
    )
    assert second.action == "continue_candidate"
    assert not second.manager_called

    third = controller.process_evaluation(
        step=15, metrics=improved, checkpoint_path=tmp_path / "step15"
    )
    assert third.action == "accept"
    assert not third.manager_called

    fourth = controller.process_evaluation(
        step=20, metrics=improved, checkpoint_path=tmp_path / "step20"
    )
    assert fourth.action == "champion_continue"
    assert not fourth.manager_called

    fifth = controller.process_evaluation(
        step=25, metrics=improved, checkpoint_path=tmp_path / "step25"
    )
    assert fifth.action == "champion_plateau"


def test_relative_height_and_motion_stability_conditions(tmp_path):
    program = _program(
        completion_hold_steps=2,
        completion_conditions=[
            {
                "type": "distance",
                "left": "object_pos",
                "right": "target_pos",
                "axes": [0, 1],
                "op": "lt",
                "threshold": 0.1,
            },
            {
                "type": "relative_scalar",
                "left": "object_pos",
                "right": "target_pos",
                "index": 2,
                "op": "gt",
                "threshold": 0.0,
            },
            {
                "type": "relative_scalar",
                "left": "object_pos",
                "right": "target_pos",
                "index": 2,
                "op": "lt",
                "threshold": 0.1,
            },
            {
                "type": "delta_distance",
                "key": "object_pos",
                "axes": [0, 1, 2],
                "op": "lt",
                "threshold": 0.01,
            },
        ],
        potential_terms=[
            {
                "type": "relative_scalar",
                "left": "object_pos",
                "right": "target_pos",
                "index": 2,
                "target": 0.05,
                "scale": 0.02,
                "weight": 1.0,
            }
        ],
    )
    path = tmp_path / "reward.json"
    atomic_write_physical_reward_program(path, program)
    runtime = PhysicalPotentialRewardRuntime(
        path, num_envs=1, expected_gamma=0.99, reload_interval_steps=16
    )
    target = np.asarray([0.0, 0.0, 1.0])
    stable = {
        "object_pos": np.asarray([0.0, 0.0, 1.05]),
        "target_pos": target,
    }
    runtime.reset([stable])

    first = runtime.compute([stable], [9])
    second = runtime.compute([stable], [9])
    assert first.raw_completion.tolist() == [1.0]
    assert first.completion.tolist() == [0.0]
    assert second.completion.tolist() == [1.0]
    assert second.condition_pass.tolist() == [[1.0, 1.0, 1.0, 1.0]]
    assert second.term_scores[0, 0] == pytest.approx(1.0)

    moved = {
        "object_pos": np.asarray([0.02, 0.0, 1.05]),
        "target_pos": target,
    }
    unstable = runtime.compute([moved], [9])
    assert unstable.raw_completion.tolist() == [0.0]
    assert unstable.condition_pass[0, 3] == 0.0
    assert unstable.completion.tolist() == [0.0]


def test_validator_bounds_audit_component_count():
    conditions = _program()["completion_conditions"] * 9
    with pytest.raises(PhysicalRewardProgramError, match="cannot exceed"):
        validate_physical_reward_program(_program(completion_conditions=conditions))


def test_completion_bonus_is_fixed_for_comparable_elite_threshold():
    with pytest.raises(PhysicalRewardProgramError, match="must equal 1.0"):
        validate_physical_reward_program(_program(completion_bonus=2.0))


def test_validator_rejects_unknown_completion_reward_mode():
    with pytest.raises(PhysicalRewardProgramError, match="completion_reward_mode"):
        validate_physical_reward_program(
            _program(completion_reward_mode="repeat_onset")
        )


@pytest.mark.parametrize("cap_steps", [None, 1, 33, 2.5])
def test_validator_bounds_capped_occupancy_steps(cap_steps):
    with pytest.raises(PhysicalRewardProgramError, match="cap_steps"):
        validate_physical_reward_program(
            _program(
                completion_reward_mode="capped_occupancy",
                completion_reward_cap_steps=cap_steps,
            )
        )


def test_manager_compacts_evidence_and_rejects_rolled_back_duplicate(monkeypatch):
    rolled_back = _program(name="rolled_back")
    revised = _program(name="revised", completion_hold_steps=3)
    responses = [rolled_back, revised]
    observed_messages = []
    manager = LunaRewardManager(LunaRewardManagerConfig(max_retries=2))

    def fake_request(messages):
        observed_messages.append(list(messages))
        program = responses.pop(0)
        return (
            {
                "id": f"response-{len(observed_messages)}",
                "choices": [{"message": {"content": json.dumps(program)}}],
                "usage": {"total_tokens": 10},
            },
            1,
            0.1,
        )

    monkeypatch.setattr(manager, "_request", fake_request)
    proposal = manager.propose(
        scene_context={
            "task_ids": [9],
            "available_physical_keys": ["object_pos", "target_pos"],
        },
        current_program=rolled_back,
        experiment_history=[
            {
                "step": 10,
                "action": "rollback_regression",
                "score": 0.1,
                "score_panel": {"a": 0.1},
                "metrics": {
                    "env/physical_completion_fp_once": 0.2,
                    "time/irrelevant": 999.0,
                },
            }
        ],
        expected_gamma=0.99,
        reward_history=[
            {
                "digest": physical_program_digest(rolled_back),
                "program": rolled_back,
                "evaluations": [{"action": "rollback_regression"}],
                "terminal_action": "rollback_regression",
            }
        ],
    )

    assert proposal.attempt == 2
    assert proposal.program["name"] == "revised"
    first_evidence = json.loads(observed_messages[0][-1]["content"])
    assert first_evidence["past_reward_trials"][0]["terminal_action"] == (
        "rollback_regression"
    )
    compact_metrics = first_evidence["recent_experiments"][0]["metrics"]
    assert "env/physical_completion_fp_once" in compact_metrics
    assert "time/irrelevant" not in compact_metrics
    assert "already rolled back" in observed_messages[1][-1]["content"]


def test_controller_requires_consecutive_champion_improvement(tmp_path):
    program_path = tmp_path / "reward.json"
    scene_path = tmp_path / "scene.json"
    atomic_write_physical_reward_program(program_path, _program())
    scene_path.write_text(
        json.dumps(
            {
                "task_ids": [9],
                "available_physical_keys": ["object_pos", "target_pos"],
            }
        )
    )
    controller = RewardEvolutionController(
        program_path=program_path,
        scene_context_path=scene_path,
        audit_dir=tmp_path / "audit",
        expected_gamma=0.99,
        score_keys=["score"],
        score_panels={},
        target_score=0.7,
        min_improvement=0.02,
        regression_tolerance=0.02,
        manager_config=LunaRewardManagerConfig(),
        candidate_min_evaluations=2,
        candidate_patience_evaluations=3,
    )
    controller.state.update(
        {
            "manager_cycle_started": True,
            "champion_score": 0.2,
            "champion_checkpoint": str(tmp_path / "old"),
            "candidate_pending": False,
        }
    )

    spike = controller.process_evaluation(
        step=10, metrics={"score": 0.5}, checkpoint_path=tmp_path / "step10"
    )
    assert spike.action == "champion_improvement_pending"
    assert spike.champion_score == pytest.approx(0.2)
    assert controller.state["champion_checkpoint"].endswith("old")

    confirmed = controller.process_evaluation(
        step=15, metrics={"score": 0.4}, checkpoint_path=tmp_path / "step15"
    )
    assert confirmed.action == "champion_continue"
    assert confirmed.champion_score == pytest.approx(0.4)
    assert controller.state["champion_checkpoint"].endswith("step15")


def test_controller_records_reward_trial_outcome(tmp_path, monkeypatch):
    program_path = tmp_path / "reward.json"
    scene_path = tmp_path / "scene.json"
    seed = _program(name="seed")
    candidate = _program(name="candidate", completion_hold_steps=3)
    atomic_write_physical_reward_program(program_path, seed)
    scene_path.write_text(
        json.dumps(
            {
                "task_ids": [9],
                "available_physical_keys": ["object_pos", "target_pos"],
            }
        )
    )
    controller = RewardEvolutionController(
        program_path=program_path,
        scene_context_path=scene_path,
        audit_dir=tmp_path / "audit",
        expected_gamma=0.99,
        score_keys=["score"],
        score_panels={},
        target_score=0.7,
        min_improvement=0.02,
        regression_tolerance=0.02,
        manager_config=LunaRewardManagerConfig(),
        candidate_min_evaluations=1,
        candidate_patience_evaluations=1,
    )
    controller.state.update(
        {
            "manager_cycle_started": True,
            "champion_score": 0.5,
            "champion_checkpoint": str(tmp_path / "champion"),
        }
    )
    monkeypatch.setattr(
        controller.manager,
        "propose",
        lambda **_: LunaProposal(candidate, "response", {}, 0.1, 1),
    )
    digest, _ = controller._propose_next()
    assert digest == physical_program_digest(candidate)
    monkeypatch.setattr(controller, "_propose_next", lambda: (None, None))

    decision = controller.process_evaluation(
        step=10,
        metrics={
            "score": 0.1,
            "env/physical_completion_tp_once": 0.2,
            "env/physical_completion_fp_once": 0.3,
            "env/physical_completion_fn_once": 0.1,
            "train/critic/q_data": 4.0,
        },
        checkpoint_path=tmp_path / "candidate_step",
    )

    assert decision.action == "rollback_regression"
    trial = controller.state["reward_trials"][0]
    assert trial["terminal_action"] == "rollback_regression"
    assert trial["evaluations"][0]["physical_fp_once"] == pytest.approx(0.3)
    assert trial["evaluations"][0]["q_data"] == pytest.approx(4.0)
