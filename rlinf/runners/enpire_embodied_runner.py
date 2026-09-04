# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Synchronous embodied runner with measured reward evolution boundaries."""

from __future__ import annotations

import os

from omegaconf import OmegaConf

from rlinf.agents.enpire_reward.controller import RewardEvolutionController
from rlinf.runners.embodied_runner import EmbodiedRunner


class ENPIREEmbodiedRunner(EmbodiedRunner):
    """Run SAC and evolve its physical reward only after fresh evaluation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        controller_cfg = OmegaConf.to_container(
            self.cfg.agentic_reward.controller, resolve=True
        )
        self.reward_evolution = RewardEvolutionController.from_config(controller_cfg)
        self.reward_evolution_interval = int(
            self.cfg.agentic_reward.get(
                "evolution_interval", self.cfg.runner.val_check_interval
            )
        )

    def _checkpoint_path(self) -> str:
        return os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )

    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        eval_metrics = super()._maybe_eval_and_checkpoint(step)
        if not eval_metrics or self.global_step % self.reward_evolution_interval != 0:
            return eval_metrics

        checkpoint_path = self._checkpoint_path()
        if not os.path.isdir(os.path.join(checkpoint_path, "actor")):
            self._save_checkpoint()
        manager_metrics = dict(eval_metrics)
        manager_metrics.update(getattr(self, "latest_training_context", {}))
        decision = self.reward_evolution.process_evaluation(
            step=self.global_step,
            metrics=manager_metrics,
            checkpoint_path=checkpoint_path,
        )
        self.logger.info(
            "ENPIRE reward cycle: action=%s score=%.4f champion=%.4f "
            "manager_called=%s candidate=%s",
            decision.action,
            decision.score,
            decision.champion_score,
            decision.manager_called,
            decision.candidate_digest,
        )
        if decision.manager_called:
            actor_checkpoint = decision.rollback_checkpoint or checkpoint_path
            actor_path = os.path.join(actor_checkpoint, "actor")
            self.logger.info(
                "Reward changed: restoring actor only from %s and resetting "
                "Q, target Q, alpha, optimizers, and replay.",
                actor_path,
            )
            self.actor.restore_actor_for_reward_revision(actor_path).wait()
            self.actor.clear_replay_buffer().wait()
            self.update_rollout_weights()
        elif decision.rollback_checkpoint is not None:
            actor_path = os.path.join(decision.rollback_checkpoint, "actor")
            self.logger.info(
                "Reward unchanged after Manager failure: restoring the matching "
                "actor, critics, optimizers, and replay from %s.",
                actor_path,
            )
            self.actor.load_checkpoint(actor_path).wait()
            self.update_rollout_weights()
        return eval_metrics
