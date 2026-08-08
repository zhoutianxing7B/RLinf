from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.libero.libero_env import LiberoEnv


def _bare_env(*, preserve: bool = True) -> LiberoEnv:
    env = object.__new__(LiberoEnv)
    env.is_eval = False
    env.cfg = OmegaConf.create(
        {
            "balance_train_task_assignment": False,
            "preserve_train_task_assignment": preserve,
            "use_ordered_reset_state_ids": False,
            "libero_variant": "standard",
        }
    )
    env.group_size = 2
    env.num_group = 2
    env.task_ids = np.asarray([0, 0, 2, 2], dtype=np.int64)
    env.trial_ids = np.zeros(4, dtype=np.int64)
    env.cumsum_trial_id_bins = np.asarray([3, 7, 12], dtype=np.int64)
    env._generator = np.random.default_rng(7)
    return env


def _bare_balanced_env(*, seed_offset: int) -> LiberoEnv:
    env = object.__new__(LiberoEnv)
    env.seed_offset = seed_offset
    env.is_eval = False
    env.cfg = OmegaConf.create(
        {
            "balance_train_task_assignment": True,
            "preserve_train_task_assignment": True,
            "use_ordered_reset_state_ids": False,
        }
    )
    env.group_size = 1
    env.num_group = 5
    env.num_envs = 5
    env.task_id_filter = None
    env.trial_id_bins = [3] * 10
    env.cumsum_trial_id_bins = np.arange(1, 11, dtype=np.int64) * 3
    env._generator = np.random.default_rng(7 + seed_offset)
    return env


def test_balanced_train_assignment_covers_tasks_across_workers():
    first = _bare_balanced_env(seed_offset=0)
    second = _bare_balanced_env(seed_offset=1)

    first.update_reset_state_ids()
    second.update_reset_state_ids()

    first_tasks, _ = first._get_task_and_trial_ids_from_reset_state_ids(
        first.reset_state_ids
    )
    second_tasks, _ = second._get_task_and_trial_ids_from_reset_state_ids(
        second.reset_state_ids
    )
    assert np.array_equal(np.concatenate((first_tasks, second_tasks)), np.arange(10))


def test_balanced_train_reset_keeps_assigned_task():
    env = _bare_balanced_env(seed_offset=1)
    env.update_reset_state_ids()
    env._init_task_and_trial_ids()

    reset_state_ids = env._get_assigned_train_reset_state_ids(np.asarray([1, 3]))
    task_ids, _ = env._get_task_and_trial_ids_from_reset_state_ids(reset_state_ids)

    assert np.array_equal(task_ids, np.asarray([6, 8]))


def test_train_reset_preserves_task_assignment_and_varies_trials():
    env = _bare_env()

    env.update_reset_state_ids()

    assert len(env.reset_state_ids) == 4
    assert np.array_equal(env.reset_state_ids[::2], env.reset_state_ids[1::2])
    task_ids, _ = env._get_task_and_trial_ids_from_reset_state_ids(env.reset_state_ids)
    assert np.array_equal(task_ids, env.task_ids)


def test_same_task_reset_does_not_reconfigure_simulator():
    env = _bare_env()
    env.group_size = 1
    env.num_group = 1
    env.task_ids = np.asarray([1], dtype=np.int64)
    env.trial_ids = np.asarray([0], dtype=np.int64)
    env.seed = 0
    env.env = SimpleNamespace(
        reconfigure_env_fns=Mock(),
        seed=Mock(),
        reset=Mock(),
        set_init_state=Mock(),
    )
    env._get_reset_states = Mock(return_value=np.zeros((1, 4), dtype=np.float32))

    env._reconfigure(np.asarray([4]), np.asarray([0]))

    env.env.reconfigure_env_fns.assert_not_called()
    env.env.reset.assert_called_once()
    assert env.task_ids[0] == 1
    assert env.trial_ids[0] == 1


def test_randomized_train_sim_seed_is_reproducible_and_changes():
    first = _bare_env()
    second = _bare_env()
    for env in (first, second):
        env.seed = 11
        env.cfg.randomize_train_sim_seed_on_reset = True
        env.cfg.train_sim_seed_low = 100
        env.cfg.train_sim_seed_high = 1000

    first_seeds = [first._next_simulator_reset_seed(3) for _ in range(3)]
    second_seeds = [second._next_simulator_reset_seed(3) for _ in range(3)]

    assert first_seeds == second_seeds
    assert len(set(first_seeds)) == 3
    assert all(100 <= seed < 1000 for seed in first_seeds)


def test_eval_sim_seed_remains_fixed_when_train_randomization_is_enabled():
    env = _bare_env()
    env.is_eval = True
    env.seed = 17
    env.cfg.randomize_train_sim_seed_on_reset = True

    assert env._next_simulator_reset_seed(5) == 85


def test_eval_auto_reset_restarts_exhausted_pool_without_recounting():
    env = object.__new__(LiberoEnv)
    env.num_envs = 2
    env.task_ids = np.asarray([0, 0], dtype=np.int64)
    env.trial_ids = np.asarray([0, 1], dtype=np.int64)
    env.reset_state_ids = np.asarray([0, 1], dtype=np.int64)
    env._eval_seen_trials = set()
    env._task_success_stats = {}
    env._get_ordered_reset_state_ids = Mock(
        return_value=np.asarray([-1, 2], dtype=np.int64)
    )
    env.reset = Mock(return_value=({"obs": "reset"}, {"fresh": True}))
    final_obs = {"obs": "final"}
    final_info = {"episode": {"success_once": np.asarray([True, False], dtype=bool)}}

    obs, infos, count_mask = env._handle_eval_auto_reset(
        np.asarray([True, True]), final_obs, final_info
    )

    reset_call = env.reset.call_args.kwargs
    np.testing.assert_array_equal(reset_call["env_idx"], np.asarray([0, 1]))
    np.testing.assert_array_equal(reset_call["reset_state_ids"], np.asarray([0, 2]))
    np.testing.assert_array_equal(env.reset_state_ids, np.asarray([0, 2]))
    np.testing.assert_array_equal(count_mask, np.asarray([True, True]))
    assert obs == {"obs": "reset"}
    assert infos["fresh"] is True
    assert infos["final_observation"] == final_obs
    np.testing.assert_array_equal(
        infos["final_info"]["episode"]["success_once"],
        final_info["episode"]["success_once"],
    )


def test_eval_initial_reset_replays_pool_for_excess_simulators():
    env = object.__new__(LiberoEnv)
    env.is_eval = True
    env.num_group = 5
    env.group_size = 1
    env.specific_reset_id = None
    env._eval_reset_pool = np.asarray([10, 11, 12], dtype=np.int64)
    env.start_idx = 0

    env.update_reset_state_ids()

    np.testing.assert_array_equal(env.reset_state_ids, np.asarray([10, 11, 12, 10, 11]))
    assert env.start_idx == 3


def test_legacy_train_reset_still_reconfigures_same_task():
    env = _bare_env(preserve=False)
    env.group_size = 1
    env.num_group = 1
    env.task_ids = np.asarray([1], dtype=np.int64)


def test_balanced_train_assignment_uses_unique_trials_across_workers():
    envs = [_bare_balanced_env(seed_offset=rank) for rank in range(3)]
    for env in envs:
        env.cfg.unique_train_trial_assignment = True
        env.cfg.seed = 123
        env.num_group = 80
        env.num_envs = 80
        env.total_num_processes = 3
        env.trial_id_bins = [50] * 10
        env.cumsum_trial_id_bins = np.arange(1, 11, dtype=np.int64) * 50

    def collect_task_trials():
        pairs = []
        for env in envs:
            env.update_reset_state_ids()
            task_ids, trial_ids = env._get_task_and_trial_ids_from_reset_state_ids(
                env.reset_state_ids
            )
            pairs.append(np.stack((task_ids, trial_ids), axis=1))
        return np.concatenate(pairs)

    first_round = collect_task_trials()
    second_round = collect_task_trials()

    assert first_round.shape == (240, 2)
    assert np.unique(first_round, axis=0).shape[0] == 240
    assert np.unique(second_round, axis=0).shape[0] == 240
    assert np.array_equal(
        np.bincount(first_round[:, 0], minlength=10),
        np.full(10, 24),
    )
    assert not np.array_equal(first_round, second_round)
