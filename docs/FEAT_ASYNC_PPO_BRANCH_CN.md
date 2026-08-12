# `feat/async-ppo` 分支说明与交接

更新时间：2026-08-12

## 1. 仓库与分支

- 工作仓库：`/vepfs-mlp2/c20250301/240403026/async_vla/RLinf`
- 当前分支：`feat/async-ppo`
- 远端：`https://github.com/zhoutianxing7B/RLinf.git`
- 基线：`origin/main`，当前 HEAD 为 `573c9394`
- 当前状态：功能修改仍在工作树中，尚未 commit，也未 push
- 注意：本实现不在 `/vepfs-mlp2/c20250301/240403026/async_ppo` 工作树中

当前分支的已提交历史与 `origin/main` 一致。所有下文描述的功能均来自当前未提交修改及新增文件，交接、提交或迁移时必须同时包含 tracked 和 untracked 文件。

## 2. 要解决的问题

将 GR00T N1.7 接入 RLinf，形成真正解耦的低频语义系统和高频动作系统：

1. VLM semantic server 独立运行，按环境维护最新语义 packet 及历史 packet。
2. DiT 不加载本地 VLM backbone，只读取已完成的语义 token。
3. DiT 同时输入最新 state、语义 token、语义 age 和可选 action history。
4. 训练时从缓存历史中采样 simulator frame age 0-6，学习抵抗语义延迟。
5. Action expert 与 reward expert 使用一致的语义、age、state/action 时间边界。
6. RL 阶段保持 RLinf 原生 actor-critic PPO，只改变模型输入和组件放置，不把算法伪装成 GRPO。

目标端到端流程为：

```text
LIBERO rollout 收集
  -> 基于绝对 simulator frame 后处理 age 0-6 样本
  -> cached-semantic action expert SFT
  -> shared-semantic reward expert SFT 与独立 eval
  -> semantic server + DiT-only PPO 闭环
  -> 固定 trial eval 与 best-only checkpoint
```

## 3. 当前运行架构

当前四卡验证采用以下放置：

```text
GPU 0: frozen GR00T VLM semantic server
GPU 1-3: RLinf DiT actor + rollout + LIBERO env + reward worker
```

语义链路：

```text
env observation + task text
  -> semantic publish queue
  -> VLM server forward
  -> per-env packet history {tokens, source_frame, completion_wallclock}
  -> 取最新完成 packet，或按训练目标取 age 0-6 历史 packet
  -> DiT(tokens, packet_age, current_state, action_history)
  -> action chunk
```

这里的 age 使用 simulator frame，而不是宿主机秒数。场景变化引起的错位由 `current_frame - source_frame` 表示；wallclock 只用于测量服务吞吐、排队和完成时间。训练和最终部署均不重新 forward 旧视觉输入。

Reward expert 与 action expert 共享同一份 frozen semantic token 和 age 定义，但参数完全独立。在线 PPO 的优化奖励来自 Reward Worker；环境 success 继续用于观测和独立评测，不直接替代 learned reward。

## 4. 核心入口

### PPO

- 主配置：`examples/embodiment/config/libero_10_ditonly_ppo_semantic_server_clean.yaml`
- 资源调度与启动：`examples/embodiment/run_gr00t_semantic_cache_sync_ppo.sh`
- RLinf 入口：`examples/embodiment/train_embodied_agent.py`

该配置明确设置：

- `loss_type: actor_critic`
- `adv_type: gae`
- `group_size: 1`
- `dit_only_train: true`
- `drop_local_backbone: true`
- action chunk 为 16，默认 4 个 denoising steps
- train/eval semantic age 均可配置为 0-6 simulator frames
- 仅训练 `action_head.model`、`packet_age_adapter` 和 `value_head`

### Action expert SFT

- 数据集：`rlinf/data/datasets/shared_semantic_action.py`
- 配置：`examples/sft/config/libero_gr00t_n1d7_cached_semantic_action_sft.yaml`
- Worker 接入：`rlinf/workers/sft/fsdp_vla_sft_worker.py`

### Reward expert

- 模型：`rlinf/models/embodiment/reward/shared_semantic_reward_model.py`
- 训练数据接入：`rlinf/data/datasets/reward_model.py`
- 训练配置：`examples/reward/config/shared_semantic_reward_training.yaml`
- 独立评测：`examples/reward/eval_shared_semantic_reward.py`
- 在线 Worker：`rlinf/workers/reward/reward_worker.py`

### 数据收集与后处理

- 原始 reward rollout：`examples/embodiment/collect_libero10_rm_data_n1d7.py`
- shared semantic 收集：`examples/embodiment/collect_libero_shared_semantic_dataset.py`
- 人为延迟后处理：`examples/reward/preprocess_shared_semantic_rollouts.py`
- 收集启动脚本：`examples/embodiment/run_collect_libero10_rm_n1d7.sh`

## 5. 本分支修改范围

核心运行时修改：

- `gr00t_action_model.py`：semantic packet 输入、age/action-history adapter、DiT-only 装载和训练边界。
- `semantic_server.py`：独立 VLM 服务与 packet history 参数。
- `env_worker.py`：按 simulator frame 发布/取回语义、固定评测流重置、trial 记录和 reward 闭环。
- `reward_worker.py`：shared-semantic reward batch 与空中间 batch 处理。
- `eval_noise.py`：可重复评测噪声。
- `rlinf/config.py`：新增数据和模型配置注册。

新增业务模块：

- shared-semantic action SFT 数据集，共 122 行。
- shared-semantic reward model，共 615 行。
- 数据收集、后处理、reward eval 和相应 YAML/脚本。

测试：

- `tests/unit_tests/test_shared_semantic_action_sft.py`
- `tests/unit_tests/test_shared_semantic_reward.py`
- `tests/unit_tests/test_semantic_batch_scheduler.py`

完整逐文件差异和历史实验说明见 `docs/DECOUPLED_DIT_PPO_HANDOFF_CN.md`。不要把日志、TensorBoard event、Ray 临时目录或 checkpoint 加入源码提交。

## 6. 当前长跑配置与结果

当前运行目录：

```text
logs/20260812-10:10:48-libero_10_ditonly_ppo_semantic_server_clean/
```

实验名：

```text
task0_reward_v2_age0to6_fresh0_long50_env240eq_v3
```

关键配置：

- Task 0 单任务验证
- 60 个并发 train env
- 每个 PPO update 执行 4 个 rollout epoch，即 240 env-rollout equivalents
- 每个 rollout epoch 256 simulator frames
- global batch 384
- 50 个 PPO update
- 每 10 步固定评测 48 trials
- action chunk 16，denoising steps 4
- semantic age 0-6，额外 fetch 人为等待为 0 ms
- actor LR `1e-7`，value LR `2e-5`
- 只保存历史最佳 checkpoint

截至第 10 个 PPO update：

| 指标 | Step 0 | Step 10 | 变化 |
|---|---:|---:|---:|
| 固定 eval success | 33/48，68.75% | 39/48，81.25% | +12.5pp |
| fail -> success | - | 9 | - |
| success -> fail | - | 3 | - |
| exact McNemar p | - | 0.146 | 未显著 |

该点是严格重置 eval age RNG 后的新结果，不再沿用旧版未重置 age stream 的配对统计。它说明训练出现积极信号，但 48 trials 仍不足以证明统计显著提升，需要后续 step 20/30/40/50 以及至少 400 个固定 trial 复核。

训练健康度：

- actor `approx_kl` 约为 `-0.0006` 到 `0.0024`，未爆炸。
- actor `clip_fraction` 约为 `0.3%` 到 `0.7%`，更新较保守。
- critic value loss 从约 `0.50` 降至 `0.19`。
- critic explained variance 仍为负，value 预测尚不可靠，是当前主要风险。
- 每个 PPO update 约 5.5 分钟；带固定 eval 的第 10 步约 6.6 分钟。

TensorBoard：

```text
http://localhost:6010/#timeseries
```

## 7. 已完成验证

最近一次清理后完成：

- 102 个相关单元测试通过。
- 修改文件通过 `ruff check`。
- Python 入口通过 `py_compile`。
- Shell 启动脚本通过语法检查。
- Hydra clean PPO 配置能够完整解析。
- 固定 eval 每轮重置语义 age 随机流，并输出独立 `eval_trials_step_*.jsonl`。
- launcher 中的 GPU、batch、denoising、weight sync、fetch delay 参数已真正接入 YAML，不再只是无效环境变量。

这些检查证明当前实现与设计边界一致，但不能等价为“没有任何潜在 bug”。最终结论仍应以完整 50 步长跑、fresh process 大样本评测和一次从零复现为准。

## 8. 交接与提交注意事项

1. 当前分支尚未提交。提交前先检查 `git status --short`，尤其不能遗漏新增文件。
2. 不要把旧的 `codex/*` 备份分支或外部 `async_ppo` 工作树混入本分支。
3. 不要恢复已经删除的 GRPO、replay、固定三步刷新等实验性路径。
4. PPO 算法保持 RLinf 原生 actor-critic；解耦只发生在 VLM 服务、语义缓存和 DiT 输入边界。
5. checkpoint 只保留最佳结果，避免再次占满磁盘。
6. 对外报告成功率时，必须区分 train rollout success 与固定 eval success。
7. 统计检验只能配对同一 task/trial/age 条件；旧的未重置 age stream 结果只能作为 smoke 参考。

## 9. 建议的下一步

1. 不中断当前 50 步任务，记录 step 20/30/40/50 固定评测。
2. 选择 best checkpoint，用 fresh process 跑至少 400 个固定 trial。
3. 同时评测 age 0、随机 age 0-6 和固定 age 6，避免均值掩盖延迟边界。
4. 确认统计提升后，再扩展至 LIBERO-Spatial 全任务和 LIBERO-10 全任务。
5. 最后运行完整相关测试并由维护者决定 commit/push；本次交接不自动提交。
