# Copyright 2025 The RLinf Authors.
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

import copy
import glob
import importlib
import os
import re
import sys
from typing import Optional, Union

import gym
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.libero.utils import (
    build_interleaved_eval_reset_state_ids,
    distribute_reset_state_ids_round_robin,
    get_benchmark_overridden,
    get_libero_image,
    get_libero_type,
    get_libero_wrist_image,
    quat2axisangle,
    record_completed_episode_task_stats,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger


def _repoint_libero_config(libero_module) -> None:
    """Point LIBERO's cached config at the package that is actually installed.

    LIBERO writes absolute paths into a config file under ``$HOME`` the first
    time it is imported and afterwards only reads it back, so a config left by
    another venv on the same machine silently redirects asset and init-state
    lookups to directories this venv does not have.
    """
    installed_root = os.path.dirname(os.path.abspath(libero_module.__file__))
    try:
        configured_root = libero_module.get_libero_path("benchmark_root")
    except Exception:
        configured_root = None
    if configured_root != installed_root:
        libero_module.set_libero_default_path(installed_root)


logger = get_logger()
_MAX_AGENTIC_REWARD_COMPONENTS = 8


def _read_bddl_language_and_goal(bddl_path: str):
    """Parse (:language ...) and a compact (:goal ...) summary from a BDDL file."""
    try:
        with open(bddl_path, "r", encoding="utf-8") as f:
            bddl_text = f.read()
    except OSError:
        return None, None
    lang_m = re.search(r"\(:language\s+([^)]+)\)", bddl_text)
    language = lang_m.group(1).strip() if lang_m else None
    goal_m = re.search(r"\(:goal\s*\n?\s*\(And\s*\(([^)]+)\)\)", bddl_text)
    if goal_m is None:
        goal_m = re.search(r"\(:goal[\s\S]*?\(And\s*\(([^)]+)\)\)", bddl_text)
    goal = goal_m.group(1).strip() if goal_m else None
    return language, goal


libero_type = get_libero_type()

if libero_type in ["pro", "plus"]:
    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
    LIBERO_PKG_NAME = f"libero{libero_type}"
    LIBERO_MAIN_MODULE_PATH = f"{LIBERO_PKG_NAME}.{LIBERO_PKG_NAME}"
    try:
        real_libero_pkg = importlib.import_module(LIBERO_PKG_NAME)
        real_libero_core = importlib.import_module(LIBERO_MAIN_MODULE_PATH)

        try:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.benchmark"
            )
        except ImportError:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_PKG_NAME}.benchmark"
            )

        try:
            real_libero_envs = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.envs"
            )
        except ImportError:
            real_libero_envs = importlib.import_module(f"{LIBERO_PKG_NAME}.envs")

        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs
    except ImportError as e:
        print(
            f"[Main Process Routing Error] Failed to import '{LIBERO_MAIN_MODULE_PATH}'. Error: {e}"
        )

if libero_type == "pro":
    import liberopro.liberopro as _libero_core
    from liberopro.liberopro.benchmark import Benchmark
elif libero_type == "plus":
    import liberoplus.liberoplus as _libero_core
    from liberoplus.liberoplus.benchmark import Benchmark
else:
    import libero.libero as _libero_core
    from libero.libero.benchmark import Benchmark

# Must run before any benchmark lookup: get_task_init_states() reads the cached
# config, so repointing it later (e.g. when resolving bddl_files) is too late.
_repoint_libero_config(_libero_core)


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info

        if seed_offset == 0:
            self._log_evaluation_mode()
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_id_filter = cfg.get("task_id_filter", None)
        if self.task_id_filter is not None:
            self.task_id_filter = list(self.task_id_filter)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self.is_eval = cfg.get("is_eval", False)

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        if self.is_eval:
            pool = self.reset_state_ids_all[self.seed_offset]
            self._eval_reset_pool = pool[pool >= 0].copy()
        else:
            self._eval_reset_pool = np.array([], dtype=np.int64)
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self.use_step_penalty = getattr(cfg, "use_step_penalty", False)

        self.agentic_reward = None
        agentic_reward_cfg = cfg.get("agentic_reward", None)
        if agentic_reward_cfg is not None and agentic_reward_cfg.get("enabled", False):
            from rlinf.agents.enpire_reward.physical_potential import (
                PhysicalPotentialRewardRuntime,
            )

            self.agentic_reward = PhysicalPotentialRewardRuntime(
                agentic_reward_cfg.program_path,
                num_envs=self.num_envs,
                expected_gamma=agentic_reward_cfg.gamma,
                reload_interval_steps=agentic_reward_cfg.get(
                    "reload_interval_steps", 16
                ),
            )

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None
        self.skip_intermediate_renders = bool(
            OmegaConf.select(cfg, "skip_intermediate_renders", default=False)
        )

    def _log_evaluation_mode(self):
        """Log the LIBERO evaluation mode banner (rank 0 env worker only)."""
        libero_type = get_libero_type()
        if libero_type == "pro":
            perturbation = os.environ.get("LIBERO_PERTURBATION", "all")
            logger.info(f"Evaluation Mode: LIBERO-PRO | Perturbation: {perturbation}")
        elif libero_type == "plus":
            suffix = os.environ.get("LIBERO_SUFFIX", "all")
            logger.info(f"Evaluation Mode: LIBERO-PLUS | Suffix: {suffix}")
        else:
            logger.info("Evaluation Mode: Standard LIBERO")

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        current_type_val = get_libero_type()

        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param, _type_val=current_type_val):
                os.environ["LIBERO_TYPE"] = _type_val
                seed = param.pop("seed")

                if _type_val in ["pro", "plus"]:
                    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]

                    pkg_name = f"libero{_type_val}"
                    core_name = f"{pkg_name}.{pkg_name}"

                    try:
                        real_pkg = importlib.import_module(pkg_name)
                        real_core = importlib.import_module(core_name)
                        real_bench = importlib.import_module(f"{core_name}.benchmark")
                        real_envs = importlib.import_module(f"{core_name}.envs")

                        sys.modules["libero"] = real_pkg
                        sys.modules["libero.libero"] = real_core
                        sys.modules["libero.libero.benchmark"] = real_bench
                        sys.modules["libero.libero.envs"] = real_envs

                        loaded_path = os.path.dirname(real_core.__file__)
                        os.environ["LIBERO_ASSET_ROOT"] = os.path.join(
                            loaded_path, "assets"
                        )
                        os.environ["LIBERO_BDDL_PATH"] = os.path.join(
                            loaded_path, "bddl_files"
                        )
                        os.environ["LIBERO_INIT_STATES_PATH"] = os.path.join(
                            loaded_path, "init_files"
                        )

                        WorkerEnv = real_envs.OffScreenRenderEnv

                    except ImportError as e:
                        print(f"[Worker Env Error] {e}")
                        raise e
                else:
                    from libero.libero.envs import OffScreenRenderEnv as WorkerEnv

                env = WorkerEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        raw_suffix = os.environ.get(
            "LIBERO_SUFFIX",
            os.environ.get(
                "LIBERO_PERTURBATION",
                self.cfg.get("perturbation_suffix", None)
                if hasattr(self.cfg, "get")
                else None,
            ),
        )
        if variant == "pro":
            import liberopro.liberopro as l_pro

            _repoint_libero_config(l_pro)
            bddl_root = l_pro.get_libero_path("bddl_files")
        elif variant == "plus":
            import liberoplus.liberoplus as l_plus

            _repoint_libero_config(l_plus)
            bddl_root = l_plus.get_libero_path("bddl_files")
        else:
            import libero.libero as l_base

            _repoint_libero_config(l_base)
            bddl_root = l_base.get_libero_path("bddl_files")

        suite_name = self.cfg.task_suite_name.lower()
        suite_keyword = suite_name.replace("libero_", "").strip()

        task_descriptions = []
        pert_init_folders = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(
                    self.task_descriptions[env_id]
                    if hasattr(self, "task_descriptions")
                    else ""
                )
                pert_init_folders.append(
                    self._pert_init_folders[env_id]
                    if hasattr(self, "_pert_init_folders")
                    else ""
                )
                continue

            task = self.task_suite.get_task(self.task_ids[env_id])
            folder_name = task.problem_folder
            file_name = task.bddl_file
            original_path = os.path.join(bddl_root, folder_name, file_name)

            final_path = original_path

            if variant == "pro":
                pro_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = ["_lan", "_object", "_swap", "_task"]
                if pro_suffix == "all":
                    filter_perts = valid_perts
                elif pro_suffix is not None:
                    # Map bare name (e.g. "task") to directory suffix (e.g. "_task")
                    normalized = (
                        f"_{pro_suffix}"
                        if not pro_suffix.startswith("_")
                        else pro_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    all_sub_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                        and suite_keyword in d
                        and any(d.endswith(pert) for pert in filter_perts)
                    ]

                    core_task_name = file_name.replace(".bddl", "")
                    all_candidates = []

                    for sub_dir in all_sub_dirs:
                        target_dir_path = os.path.join(bddl_root, sub_dir)
                        matches = [
                            os.path.join(target_dir_path, f)
                            for f in os.listdir(target_dir_path)
                            if core_task_name in f and f.endswith(".bddl")
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            elif variant == "plus":
                plus_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = [
                    "_light",
                    "_language",
                    "_table",
                    "_add",
                    "_tb",
                    "_sample",
                    "_level",
                ]
                if plus_suffix == "all":
                    filter_perts = valid_perts
                elif plus_suffix is not None:
                    normalized = (
                        f"_{plus_suffix}"
                        if not plus_suffix.startswith("_")
                        else plus_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    clean_name = file_name.replace(".bddl", "")
                    for marker in valid_perts:
                        if marker in clean_name:
                            clean_name = clean_name.split(marker)[0]
                            break

                    suite_pattern = folder_name.replace("_", "").lower()
                    all_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                    ]
                    search_dirs = [
                        os.path.join(bddl_root, d)
                        for d in all_dirs
                        if suite_pattern in d.lower().replace("_", "")
                    ]

                    if not search_dirs:
                        search_dirs = [os.path.join(bddl_root, folder_name)]

                    all_candidates = []
                    for target_dir in search_dirs:
                        matches = [
                            f
                            for f in glob.glob(os.path.join(target_dir, "*.bddl"))
                            if clean_name in os.path.basename(f)
                            and any(
                                pert in os.path.basename(f) for pert in filter_perts
                            )
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": final_path,
                    "seed": self.seed,
                }
            )
            # LIBERO-PRO: use selected BDDL language (not original suite task.language)
            # and remember the perturbation folder for pruned_init loading.
            pert_folder = os.path.basename(os.path.dirname(os.path.abspath(final_path)))
            pert_init_folders.append(pert_folder)
            if variant == "pro":
                bddl_lang, bddl_goal = _read_bddl_language_and_goal(final_path)
                desc = bddl_lang if bddl_lang else task.language
                task_descriptions.append(desc)
                if self.is_eval:
                    logger.info(
                        "[LIBERO-PRO lang] env=%s pert_folder=%s "
                        "prompt=%r suite_orig=%r bddl_lang=%r goal=%r bddl=%s",
                        env_id,
                        pert_folder,
                        desc,
                        task.language,
                        bddl_lang,
                        bddl_goal,
                        final_path,
                    )
            else:
                task_descriptions.append(task.language)

        self.task_descriptions = task_descriptions
        self._pert_init_folders = pert_init_folders
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        if self.task_id_filter is not None:
            num_tasks = len(self.trial_id_bins)
            validated_tids = []
            for tid in self.task_id_filter:
                if not isinstance(tid, (int, np.integer)):
                    raise ValueError(
                        f"task_id_filter must contain ints, got "
                        f"{type(tid).__name__}: {tid}"
                    )
                tid_int = int(tid)
                if tid_int < 0 or tid_int >= num_tasks:
                    raise ValueError(
                        f"task_id {tid_int} in task_id_filter is out of range "
                        f"[0, {num_tasks - 1}]"
                    )
                validated_tids.append(tid_int)
            validated_tids = sorted(set(validated_tids))

            self._valid_reset_state_ids = []
            for tid in validated_tids:
                start = self.cumsum_trial_id_bins[tid - 1] if tid > 0 else 0
                end = self.cumsum_trial_id_bins[tid]
                self._valid_reset_state_ids.extend(range(start, end))
            self._valid_reset_state_ids = np.array(self._valid_reset_state_ids)
        else:
            self._valid_reset_state_ids = None

    def update_reset_state_ids(self):
        if self.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        elif self._valid_reset_state_ids is not None:
            indices = self._generator.integers(
                low=0, high=len(self._valid_reset_state_ids), size=(num_reset_states,)
            )
            reset_state_ids = self._valid_reset_state_ids[indices]
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        if self.is_eval:
            if self._valid_reset_state_ids is not None:
                reset_state_ids = self._valid_reset_state_ids.copy()
            else:
                reset_state_ids = build_interleaved_eval_reset_state_ids(
                    self.trial_id_bins, self.cumsum_trial_id_bins
                )
            return distribute_reset_state_ids_round_robin(
                reset_state_ids, self.total_num_processes
            )

        if self._valid_reset_state_ids is not None:
            reset_state_ids = self._valid_reset_state_ids.copy()
        else:
            reset_state_ids = np.arange(self.total_num_group_envs)

        self._generator_ordered.shuffle(reset_state_ids)

        # Ensure we have enough IDs for all processes by tiling if needed
        if len(reset_state_ids) < self.total_num_processes:
            repeats = (self.total_num_processes // len(reset_state_ids)) + 1
            reset_state_ids = np.tile(reset_state_ids, repeats)

        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            return self.specific_reset_id * np.ones((num_reset_states,), dtype=int)

        if self.is_eval:
            pool = self._eval_reset_pool
            if self.start_idx >= len(pool):
                return np.full((num_reset_states,), -1, dtype=np.int64)
            end = min(self.start_idx + num_reset_states, len(pool))
            n_valid = end - self.start_idx
            result = np.full((num_reset_states,), -1, dtype=np.int64)
            if n_valid > 0:
                result[:n_valid] = pool[self.start_idx : end]
            self.start_idx = end
            return result

        if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
        reset_state_ids = self.reset_state_ids_all[self.seed_offset][
            self.start_idx : self.start_idx + num_reset_states
        ]
        self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        # LIBERO-PRO: load pruned_init from the selected perturbation folder
        # (e.g. libero_object_task/), not the original suite folder.
        if variant == "pro" and getattr(self, "_pert_init_folders", None):
            import liberopro.liberopro as l_pro

            init_root = l_pro.get_libero_path("init_states")
            init_state = []
            for env_id in env_idx:
                task = self.task_suite.get_task(self.task_ids[env_id])
                folder = self._pert_init_folders[env_id] or task.problem_folder
                pert_init_path = os.path.join(init_root, folder, task.init_states_file)
                states = None
                init_path = None
                used_folder = folder
                if os.path.exists(pert_init_path):
                    loaded = torch.load(pert_init_path, weights_only=False)
                    n = len(loaded) if hasattr(loaded, "__len__") else 0
                    if n == 0:
                        if self.is_eval:
                            logger.warning(
                                "[LIBERO-PRO init] empty pruned_init, skip: %s",
                                pert_init_path,
                            )
                    else:
                        states = loaded
                        init_path = pert_init_path
                if states is None:
                    msg = (
                        "[LIBERO-PRO init] perturbation init missing or empty; "
                        "suite fallback is invalid for eval "
                        f"env={env_id} wanted_folder={folder} "
                        f"file={task.init_states_file} path={pert_init_path}"
                    )
                    if self.is_eval:
                        logger.error(msg)
                        raise RuntimeError(msg)
                    states = self.task_suite.get_task_init_states(self.task_ids[env_id])
                    init_path = f"<suite:{task.problem_folder}/{task.init_states_file}>"
                    used_folder = task.problem_folder
                    logger.warning(
                        "%s; falling back to suite init for training: %s n=%s",
                        msg,
                        init_path,
                        len(states),
                    )
                trial = int(self.trial_ids[env_id])
                if trial >= len(states):
                    trial = trial % len(states)
                init_state.append(states[trial])
                if self.is_eval and env_id == env_idx[0]:
                    logger.info(
                        "[LIBERO-PRO init] env=%s folder=%s trial=%s path=%s n=%s",
                        env_id,
                        used_folder,
                        trial,
                        init_path,
                        len(states),
                    )
            return init_state

        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)
        self.agentic_completion_return = np.zeros(self.num_envs)
        self.agentic_completion_reward_return = np.zeros(self.num_envs)
        self.agentic_raw_completion_return = np.zeros(self.num_envs)
        self.agentic_completion_once = np.zeros(self.num_envs, dtype=bool)
        self.agentic_completion_previous = np.zeros(self.num_envs, dtype=bool)
        self.agentic_completion_regressions = np.zeros(self.num_envs, dtype=np.int32)
        self.success_previous = np.zeros(self.num_envs, dtype=bool)
        self.success_regressions = np.zeros(self.num_envs, dtype=np.int32)
        component_shape = (self.num_envs, _MAX_AGENTIC_REWARD_COMPONENTS)
        self.agentic_condition_return = np.zeros(component_shape)
        self.agentic_condition_final = np.zeros(component_shape)
        self.agentic_term_max = np.zeros(component_shape)
        self.agentic_term_final = np.zeros(component_shape)
        self.agentic_potential_delta_return = np.zeros(self.num_envs)
        self.agentic_potential = np.zeros(self.num_envs)
        self._task_success_stats: dict[int, dict[str, int]] = {}
        self._eval_seen_trials: set[tuple[int, int]] = set()

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.success_episode_len[mask] = 0
            self.agentic_completion_return[mask] = 0.0
            self.agentic_completion_reward_return[mask] = 0.0
            self.agentic_raw_completion_return[mask] = 0.0
            self.agentic_completion_once[mask] = False
            self.agentic_completion_previous[mask] = False
            self.agentic_completion_regressions[mask] = 0
            self.success_previous[mask] = False
            self.success_regressions[mask] = 0
            self.agentic_condition_return[mask] = 0.0
            self.agentic_condition_final[mask] = 0.0
            self.agentic_term_max[mask] = 0.0
            self.agentic_term_final[mask] = 0.0
            self.agentic_potential_delta_return[mask] = 0.0
            self.agentic_potential[mask] = 0.0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self.agentic_completion_return[:] = 0.0
            self.agentic_completion_reward_return[:] = 0.0
            self.agentic_raw_completion_return[:] = 0.0
            self.agentic_completion_once[:] = False
            self.agentic_completion_previous[:] = False
            self.agentic_completion_regressions[:] = 0
            self.success_previous[:] = False
            self.success_regressions[:] = 0
            self.agentic_condition_return[:] = 0.0
            self.agentic_condition_final[:] = 0.0
            self.agentic_term_max[:] = 0.0
            self.agentic_term_final[:] = 0.0
            self.agentic_potential_delta_return[:] = 0.0
            self.agentic_potential[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos, agentic_step=None):
        episode_info = {}
        # Simulator success is retained only as an independent audit verifier.
        # It never enters step_reward when agentic_reward is enabled.
        self.success_regressions += self.success_previous & ~terminations
        self.success_previous = terminations.copy()

        # Only accumulate returns while not yet succeeded
        self.returns += step_reward * (~self.success_once)
        # Record episode_len at first success
        new_success_mask = terminations & ~self.success_once
        if new_success_mask.any():
            self.success_episode_len[new_success_mask] = self.elapsed_steps[
                new_success_mask
            ]

        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        if self.is_eval:
            panel_a = self.trial_ids % 2 == 0
            panel_b = ~panel_a
            episode_info["panel_a_success_mass"] = self.success_once * panel_a
            episode_info["panel_a_mass"] = panel_a.astype(np.float32)
            episode_info["panel_b_success_mass"] = self.success_once * panel_b
            episode_info["panel_b_mass"] = panel_b.astype(np.float32)

        # Use success episode_len for reward if already succeeded, else current elapsed
        episode_len_for_reward = np.where(
            self.success_once, self.success_episode_len, self.elapsed_steps
        )
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_len_for_reward, 1
        )
        if agentic_step is not None:
            completion_now = agentic_step.completion.astype(bool)
            self.agentic_completion_return += agentic_step.completion
            self.agentic_completion_reward_return += agentic_step.completion_reward
            self.agentic_raw_completion_return += agentic_step.raw_completion
            self.agentic_completion_regressions += (
                self.agentic_completion_previous & ~completion_now
            )
            self.agentic_completion_previous = completion_now
            self.agentic_completion_once |= completion_now
            self.agentic_potential_delta_return += agentic_step.potential_delta
            self.agentic_potential = agentic_step.potential.astype(np.float64)

            condition_count = agentic_step.condition_pass.shape[1]
            term_count = agentic_step.term_scores.shape[1]
            self.agentic_condition_return[:, :condition_count] += (
                agentic_step.condition_pass
            )
            self.agentic_condition_final[:, :condition_count] = (
                agentic_step.condition_pass
            )
            self.agentic_term_max[:, :term_count] = np.maximum(
                self.agentic_term_max[:, :term_count], agentic_step.term_scores
            )
            self.agentic_term_final[:, :term_count] = agentic_step.term_scores

            success_once = self.success_once
            success_at_end = terminations.astype(bool)
            completion_once = self.agentic_completion_once
            episode_info["physical_completion_occupancy"] = (
                self.agentic_completion_return.copy()
            )
            episode_info["physical_completion_reward_return"] = (
                self.agentic_completion_reward_return.copy()
            )
            episode_info["physical_raw_completion_occupancy"] = (
                self.agentic_raw_completion_return.copy()
            )
            episode_info["physical_completion_once"] = completion_once.copy()
            episode_info["physical_completion_at_end"] = completion_now.copy()
            episode_info["physical_completion_regressions"] = (
                self.agentic_completion_regressions.copy()
            )
            episode_info["success_regressions"] = self.success_regressions.copy()
            episode_info["physical_completion_tp_once"] = completion_once & success_once
            episode_info["physical_completion_fp_once"] = (
                completion_once & ~success_once
            )
            episode_info["physical_completion_fn_once"] = (
                ~completion_once & success_once
            )
            episode_info["physical_completion_tp_end"] = completion_now & success_at_end
            episode_info["physical_completion_fp_end"] = (
                completion_now & ~success_at_end
            )
            episode_info["physical_completion_fn_end"] = (
                ~completion_now & success_at_end
            )
            for index in range(condition_count):
                episode_info[f"physical_condition_{index}_occupancy"] = (
                    self.agentic_condition_return[:, index].copy()
                )
                episode_info[f"physical_condition_{index}_at_end"] = (
                    self.agentic_condition_final[:, index].copy()
                )
            for index in range(term_count):
                episode_info[f"physical_term_{index}_max"] = self.agentic_term_max[
                    :, index
                ].copy()
                episode_info[f"physical_term_{index}_final"] = self.agentic_term_final[
                    :, index
                ].copy()
            episode_info["physical_potential_delta"] = (
                self.agentic_potential_delta_return.copy()
            )
            episode_info["physical_potential"] = self.agentic_potential.copy()
            episode_info["physical_reward_revision"] = np.full(
                self.num_envs, agentic_step.revision, dtype=np.int32
            )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )

        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if task_changed or not self.is_eval:
                reconfig_env_idx.append(env_id)
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        if variant != "plus":
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            if self.is_eval:
                self._task_success_stats = {}
                self._eval_seen_trials = set()
                self.start_idx = 0
                pool = self.reset_state_ids_all[self.seed_offset]
                self._eval_reset_pool = pool[pool >= 0].copy()
                self.update_reset_state_ids()
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        if self.agentic_reward is not None:
            self.agentic_reward.reset(raw_obs, env_indices=env_idx)

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def get_camera_meta(
        self, camera_name: str = "agentview", height: int = 256, width: int = 256
    ) -> dict:
        """Fetch camera intrinsics/extrinsics and depth planes.

        Returns camera calibration from worker 0's robosuite sim: intrinsic
        matrix, cam-to-world transform, and depth near/far.  The agentview
        camera is fixed in the world, so this is constant per episode.
        """
        return self.env.workers[0].get_camera_meta(
            camera_name=camera_name, height=height, width=width
        )

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Render an arbitrary camera at the requested resolution.

        Returns:
            The rendered RGB image, or a ``(rgb, depth)`` tuple when
            *depth* is True.
        """
        return self.env.workers[0].render_camera(
            camera_name=camera_name,
            height=height,
            width=width,
            depth=depth,
        )

    def step(self, actions=None, auto_reset=True, _skip_obs_wrap=False):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = None if _skip_obs_wrap else self._wrap_obs(raw_obs)

        agentic_step = None
        if self.agentic_reward is None:
            step_reward = self._calc_step_reward(terminations)
        else:
            agentic_step = self.agentic_reward.compute(
                raw_obs,
                self.task_ids,
                # SAC bootstraps through time-limit truncation. Keeping Phi(s')
                # here makes the shaped reward consistent with that target.
                terminal=None,
            )
            step_reward = agentic_step.rewards
            infos["agentic_reward"] = to_tensor(
                {
                    "completion": agentic_step.completion,
                    "raw_completion": agentic_step.raw_completion,
                    "condition_pass": agentic_step.condition_pass,
                    "term_scores": agentic_step.term_scores,
                    "potential": agentic_step.potential,
                    "potential_delta": agentic_step.potential_delta,
                    "revision": np.full(
                        self.num_envs, agentic_step.revision, dtype=np.int32
                    ),
                }
            )

        infos = self._record_metrics(
            step_reward, terminations, infos, agentic_step=agentic_step
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos, _ = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        rendering_disabled = False
        try:
            for i in range(chunk_size):
                should_render = (not self.skip_intermediate_renders) or (
                    i == chunk_size - 1
                )
                if self.skip_intermediate_renders and i == 0 and chunk_size > 1:
                    self.env.set_camera_rendering(False)
                    rendering_disabled = True
                elif (
                    self.skip_intermediate_renders
                    and i == chunk_size - 1
                    and rendering_disabled
                ):
                    self.env.set_camera_rendering(True)
                    rendering_disabled = False

                actions = chunk_actions[:, i]
                extracted_obs, step_reward, terminations, truncations, infos = (
                    self.step(
                        actions, auto_reset=False, _skip_obs_wrap=not should_render
                    )
                )
                obs_list.append(extracted_obs)
                infos_list.append(infos)

                chunk_rewards.append(step_reward)
                raw_chunk_terminations.append(terminations)
                raw_chunk_truncations.append(truncations)
        finally:
            if rendering_disabled:
                self.env.set_camera_rendering(True)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1], eval_count_mask = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations

            if eval_count_mask is not None:
                eval_count_mask = torch.tensor(
                    eval_count_mask,
                    dtype=torch.bool,
                    device=past_terminations.device,
                )
                chunk_terminations[:, -1] &= eval_count_mask
                chunk_truncations[:, -1] &= eval_count_mask
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        if self.is_eval:
            return self._handle_eval_auto_reset(dones, _final_obs, infos)
        obs, infos = self._handle_train_auto_reset(dones, _final_obs, infos)
        return obs, infos, None

    def _handle_eval_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        count_mask = record_completed_episode_task_stats(
            env_idx,
            final_info,
            self.task_ids,
            self.trial_ids,
            self.num_envs,
            self._eval_seen_trials,
            self._task_success_stats,
        )

        new_reset_state_ids = self._get_ordered_reset_state_ids(len(env_idx))
        valid_mask = new_reset_state_ids >= 0
        env_to_reset = env_idx[valid_mask]
        if len(env_to_reset) > 0:
            self.reset_state_ids[env_to_reset] = new_reset_state_ids[valid_mask]
            obs, infos = self.reset(
                env_idx=env_to_reset,
                reset_state_ids=self.reset_state_ids[env_to_reset],
            )
        else:
            obs = _final_obs
            infos = {}

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool) & count_mask
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos, count_mask

    def _handle_train_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        if self.use_fixed_reset_state_ids:
            self.update_reset_state_ids()
            obs, infos = self.reset(
                env_idx=env_idx,
                reset_state_ids=self.reset_state_ids[env_idx],
            )
        else:
            obs, infos = self.reset(env_idx=env_idx, reset_state_ids=None)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool)
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        step_penalty = -1 if self.use_step_penalty else 0
        termination_bonus = self.cfg.reward_coef * terminations
        reward = step_penalty + termination_bonus

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward
