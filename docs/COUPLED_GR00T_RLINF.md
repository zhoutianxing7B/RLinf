# Coupled GR00T N1.7 PPO in RLinf

This profile runs the native coupled path: every rollout observation is sent to the local GR00T N1.7 VLM backbone and the resulting features are consumed by the original DiT/action head in the same RLinf worker. The semantic server and cached-feature transport are disabled.

## Run

From `RLinf/`:

```bash
bash examples/embodiment/run_gr00t_coupled_ppo.sh
```

The launcher defaults to the local LIBERO checkpoint and Cosmos Reason2 backbone. Override `GR00T_MODEL_PATH`, `GR00T_BACKBONE_PATH`, and the GPU placement variables when needed. A short smoke run can be selected with Hydra overrides, for example:

```bash
bash examples/embodiment/run_gr00t_coupled_ppo.sh   runner.max_steps=2 env.train.total_num_envs=2 env.train.max_steps_per_rollout_epoch=32
```

## Contract

The coupled profile sets `execution_mode=coupled`, `semantic_server_enabled=false`, `drop_local_backbone=false`, and `dit_only_train=false`. The model constructor validates these flags and logs `GR00T execution mode: coupled`; contradictory settings fail fast. With no explicit mode, legacy configurations retain the previous inference rule: semantic-server settings select decoupled mode, otherwise coupled mode.

This profile is intentionally separate from the existing semantic-server/DiT-only experiments so either path can be reproduced without changing the other configuration.
