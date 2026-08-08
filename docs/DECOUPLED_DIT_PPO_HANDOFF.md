# GR00T N1.7 Decoupled DiT-only PPO: Handoff

## 1. Scope

This profile is the single supported entry point for the LIBERO-Spatial decoupled experiment.
The frozen VLM backbone runs in a semantic server on GPU 0. Rollout and PPO run on GPUs 1-3.
The control side reads the newest completed semantic packet for each environment, together with simulator-frame age and state, then runs the original GR00T N1.7 DiT.

This is synchronous on-policy actor-critic PPO. It is not GRPO, replay, SAC, or an acceptance-gated trainer.

## 2. Canonical files

- Config: `examples/embodiment/config/libero_spatial_ditonly_ppo_semantic_server_clean.yaml`
- Launcher: `examples/embodiment/run_spatial_ditonly_semantic_ppo_clean.sh`
- Shared semantic launcher: `examples/embodiment/run_gr00t_semantic_cache_sync_ppo.sh`
- Semantic server: `rlinf/models/embodiment/gr00t/gr00t_n1d7/semantic_server.py`
- GR00T action path: `rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py`
- PPO loss: `rlinf/algorithms/losses.py`

Run from `/vepfs-mlp2/c20250301/240403026/async_vla/RLinf`:

```bash
bash examples/embodiment/run_spatial_ditonly_semantic_ppo_clean.sh
```

Functional smoke without a pre-evaluation:

```bash
PPO_MAX_STEPS=2 PPO_VAL_INTERVAL=2 PPO_EVAL_BEFORE_TRAINING=false \
  bash examples/embodiment/run_spatial_ditonly_semantic_ppo_clean.sh
```

## 3. Trainable boundary

- Frozen: VLM/backbone, state encoder, action encoder, action decoder, and unrelated adapters.
- Trainable: original DiT action model, packet-age adapter, and PPO value head.
- The packet-age adapter is zero-initialized when bootstrapping from an older checkpoint, so enabling it does not change the initial policy.
- `ACTION_CHUNK_SIZE=16` remains the control contract. Semantic refresh is asynchronous; this profile does not impose a fixed three-step delay.

## 4. Data and timing contract

1. Each environment has a stable `(env_id, episode_generation)` key.
2. The semantic server stores the newest completed feature packet per key.
3. A control boundary fetches the newest packet without waiting for an old request queue.
4. Age is computed from simulator frame ids: `(current_frame - source_frame) / control_hz`.
5. The PPO buffer stores semantic tensors and packet age inside `forward_inputs`; actor recomputation never calls the VLM.

## 5. PPO contract

- `algorithm.loss_type=actor_critic`
- `algorithm.adv_type=gae`
- `algorithm.group_size=1`
- `filter_rewards=false`, `kl_beta=0`, no replay buffer, no reward filtering, no acceptance/rejection, no rollback.
- One rollout is followed by `update_epoch` native PPO passes.
- `critic_warmup_steps` is zero by default. If enabled, it uses RLinf native optimizer-step warmup, not a custom rollout counter.
- Current and old log-prob tensors must have identical shape after chunk aggregation. The loss rejects impossible shapes and aligns masks explicitly.

## 6. Required invariants

Check the first training update before any long run:

- `actor/ratio_exp_logratio_gap` is near zero.
- `actor/logratio_mean` agrees with the `actor/approx_kl` sign convention.
- `critic/nonfinite_target_rate` is zero.
- `actor/lr` and `critic/lr` match the active config.
- `semantic_server_cache_entries` grows and packet age is finite.

If any invariant fails, stop the run. Do not tune success rate around a broken ratio, stale packet identity, or non-finite critic target.

## 7. Checkpoint and evaluation

Use a fresh model checkpoint for the first clean smoke. Do not mix an optimizer state from an old experimental recipe with this profile. For continuation, resume the full RLinf checkpoint directory and keep the same config; `PPO_CKPT_PATH` is only for the rollout model override and must point to `full_weights.pt`.

Run validation every 20 PPO updates for throughput. Use at least 400 fixed trials for a final claim; a 60-trial smoke is only a wiring check.

## 8. Removed from the canonical path

The old `grpo` launcher/config, custom rollout-unit critic warmup, update acceptance and checkpoint rollback, reward filtering experiments, reference-policy KL path, fixed/random age sweep controls, eval-noise helpers, and task-resampling experiments are not part of this handoff.

Legacy logs and unrelated RLinf examples may still exist, but they must not be used to start this experiment.
