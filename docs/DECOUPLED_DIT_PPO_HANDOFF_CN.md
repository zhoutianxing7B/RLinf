# GR00T N1.7 解耦 DiT-only PPO 交接文档

## 1. 目标和范围

本分支的主线是把已经验证过的 **semantic server + DiT-only PPO** 管线完整放进 RLinf。

系统分成三个计算部分：

- **慢系统**：VLM semantic backbone 独立运行在 semantic server，只负责生成语义向量。
- **快系统**：RLinf rollout/actor 只运行 GR00T N1.7 原版 DiT，读取每个环境最新的语义缓存，并使用 state 和语义 age 产生 action。
- **奖励系统（可选）**：读取同一份冻结语义 packet 的时序窗口，通过私有 reward adapter 输出 progress、completion、failure 和 uncertainty，不加载第二份 VLM。

PPO 阶段冻结 VLM、state/action encoder、decoder 和无关 adapter，只训练 DiT action model、action-private adapter、packet-age adapter 和 value head。reward adapter 独立 SFT，默认在 PPO 中冻结。

### 双 adapter 边界

- `A_pi`：`packet_age_adapter`、`action_history_adapter`、`stale_semantic_adapter` 和 `stale_semantic_token_adapter`，服务于动作策略。
- `A_r`：`RewardSemanticAdapter`，对共享 token 做私有投影、mask attention pooling 和 age/interval 条件化，服务于奖励专家。
- 两者只共享 semantic server 产生的冻结 token packet，不共享 adapter 参数、optimizer 或梯度。
- reward 不能读取 actor adapter 的输出；actor 也不能读取 reward adapter 的输出。

## 2. 关键入口

在 RLinf 仓库根目录执行：

- 配置：`examples/embodiment/config/libero_spatial_ditonly_ppo_semantic_server_clean.yaml`
- 主启动脚本：`examples/embodiment/run_spatial_ditonly_semantic_ppo_clean.sh`
- server 管理脚本：`examples/embodiment/run_gr00t_semantic_cache_sync_ppo.sh`
- semantic server：`rlinf/models/embodiment/gr00t/gr00t_n1d7/semantic_server.py`
- GR00T action model：`rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py`
- 语义预处理：`rlinf/models/embodiment/gr00t/gr00t_n1d7/semantic_preprocess_proxy.py`
- PPO loss/advantage：`rlinf/algorithms/losses.py`、`rlinf/algorithms/advantages.py`
- 环境解耦通路：`rlinf/workers/env/env_worker.py`
- rollout 解耦通路：`rlinf/workers/rollout/hf/huggingface_worker.py`
- DiT-only actor：`rlinf/workers/actor/fsdp_actor_worker.py`
- reward adapter：`rlinf/models/embodiment/reward/shared_semantic_reward_model.py`
- reward packet 数据集：`rlinf/data/datasets/reward_model.py`
- reward SFT 配置：`examples/reward/config/shared_semantic_reward_training.yaml`
- PPO reward 配置组：`examples/embodiment/config/shared_semantic_reward/enabled.yaml`

## 3. 启动方式

```bash
cd /vepfs-mlp2/c20250301/240403026/async_vla/RLinf
bash examples/embodiment/run_spatial_ditonly_semantic_ppo_clean.sh
```

启用共享语义 reward expert 时，主配置不变，追加 Hydra 配置组：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-name libero_spatial_ditonly_ppo_semantic_server_clean \
  +shared_semantic_reward=enabled
```

必须设置 `SHARED_REWARD_MODEL_PATH`。默认拒绝随机初始化的 reward head 进入 PPO。

历史结果的对齐配置如下：

```text
GPU 0: semantic server
GPU 1-3: DiT rollout 和 FSDP actor
train envs: 240
eval envs: 120
train/eval rollout steps: 256
action chunk: 16
global batch: 960
micro batch: 2
PPO update epochs: 1
actor lr: 1e-7
value lr: 2e-5
critic warmup: 0
filter rewards: false
KL beta: 0
```

需要临时覆盖参数时，例如：

```bash
ALLOWED_GPU_IDS=0,1,2,3 \
SEMANTIC_GPU_IDS=0 \
DIT_GPU_IDS=1,2,3 \
ACTOR_GPU_IDS=1,2,3 \
PPO_MAX_STEPS=100 \
PPO_VAL_INTERVAL=20 \
bash examples/embodiment/run_spatial_ditonly_semantic_ppo_clean.sh
```

## 4. 语义缓存和延迟定义

每个环境使用 `(env_id, episode_generation)` 作为唯一 key。semantic server 对每个 key 维护最新的已完成 packet，packet 至少包含语义 tensor、source frame、完成时间和 episode generation。

控制端在 action boundary 读取最新 packet，不等待旧请求，也不重新执行 VLM。延迟用 simulator frame 定义：

```text
age_frames = current_frame - source_frame
```

semantic tensor、source frame 和 age 会一起写入 PPO buffer。actor update 重新计算 logprob 时直接使用 buffer 中的语义数据，因此不会再次调用 VLM。

reward 读取相同的 buffer tensor，并维护每个环境的 K-packet 历史。只有同一 episode generation 内、version 变化且 source frame 严格增加的 packet 才产生 reward；重复读取同一 packet 的 reward 为 0。新 packet 的 reward 按 proposal 的因果区间

```text
(previous_source_frame, current_source_frame]
```

回填到动作 transition，默认使用区间 endpoint，也可设置 `SEMANTIC_REWARD_ASSIGN=uniform` 均匀分配。source/completed/action wallclock 同时保存用于延迟审计，但 simulator frame 仍是控制误差和 credit assignment 的主时间轴。

`ACTION_CHUNK_SIZE=16` 是当前控制协议。semantic server 可以异步刷新，但不依赖固定的三步刷新间隔。

## 5. Reward rollout 收集与 adapter 训练

reward 的主数据源是 policy 在 LIBERO-10 中产生的真实成功/失败 rollout，不是从专家演示的时间位置伪造标签。四卡默认用 GPU0 运行共享 semantic server，GPU1-3 运行 N1.7 DiT 与环境：

```bash
GR00T_MODEL_PATH=/path/to/n1d7_checkpoint \
GR00T_BACKBONE_PATH=/path/to/cosmos_backbone \
REWARD_DATA_ROOT=/path/to/libero10_n1d7_reward \
SEMANTIC_GPU_ID=0 \
COLLECTOR_GPU_IDS=1,2,3 \
TARGET_PER_OUTCOME=1500 \
bash examples/embodiment/run_collect_libero10_rm_n1d7.sh
```

collector 为每个任务分别启动 clean success stream 和带 action noise/dropout 的 failure/exploration stream，并用全局 quota 精确收集等量成功和失败轨迹。每条 NPZ 保存：

- 原始主视角、腕部图像和机器人 state；
- DiT 实际消费的 token-level semantic packet 与 attention mask；
- source frame、semantic version、episode generation 和 packet age；
- source/completed/action wallclock；
- policy action、实际执行 action、chunk mask、reward 和终止状态；
- 真实 terminal-success 标签。

普通控制帧允许使用 age 0-6 的旧 packet。成功终止帧必须取得 source frame 等于 terminal frame 的新 packet，否则拒绝写盘，避免把旧场景语义错误标成成功。reward 不执行第二次 VLM；保存的特征就是 action rollout 写入 PPO buffer 的同源 semantic tensor。

收集完成后按 `(task_id, init_state_id)` 分组切分，避免相同初始状态泄漏：

```bash
python examples/reward/preprocess_shared_semantic_rollouts.py \
  --data-root /path/to/libero10_n1d7_reward \
  --output-dir /path/to/libero10_n1d7_reward/manifests
```

训练直接惰性读取每轨迹 NPZ，不生成巨型 token `.pt`：

```bash
SHARED_REWARD_TRAIN_MANIFEST=/path/to/manifests/train.json \
SHARED_REWARD_VAL_MANIFEST=/path/to/manifests/val.json \
SEMANTIC_REWARD_HISTORY=4 \
python examples/reward/train_reward_model.py \
  --config-name shared_semantic_reward_training
```

当前移植严格采用 hzp 的 factual temporal-success 监督：成功轨迹的精确终止 packet 是正样本，成功前和全部失败 packet 是负样本。progress/failure/uncertainty 头暂不参与 reward，避免未监督随机输出污染 RL。后续若增加人工阶段标注、轨迹偏好或 counterfactual relabeling，再单独开放对应权重。

### Task 0 正式训练与独立评测（2026-08-12）

本次正式数据位于：

```text
/vepfs-mlp2/c20250301/240403026/async_vla/reward_data/task0_delay0to6_chunk_success_300
```

共 300 条成功和 300 条失败轨迹。按 init_state_id 分组后得到 486/48/66 条 train/val/test
轨迹，三个集合无初始状态交叉。每条轨迹采 14 个监督点；reward 输入与 action expert 一致：
旧 semantic(t-d)、当前 state(t)、当前四帧 action history、embodiment 和真实 age=d，
其中 d 覆盖 0-6 simulator frame。

最佳 reward checkpoint：

```text
/vepfs-mlp2/c20250301/240403026/async_vla/reward_runs/task0_delay0to6_exact/task0_delay0to6_reward_v2/checkpoints/best_model
```

模型 1,934,892 参数，训练 300 step；最佳 validation accuracy 为 96.28%。独立 test 有
924 个样本（189 positive、735 negative），总体 accuracy、balanced accuracy、ROC-AUC 和
PR-AUC 均为 1.0；age 0、1、2、3、4、5、6 的分桶指标也全部为 1.0，FP/FN 均为 0。
这个结果异常容易，说明 Task 0 的 terminal action/state-history 很可分；它只能证明离线分类器
拟合成功，不能单独证明在线 reward 能改善策略。

闭环使用 GPU0 的冻结 semantic server，GPU1-3 的 DiT/env/reward，24 个 train env、48 个
fixed eval trial、action chunk 16、随机真实 age 0-6。action boundary 发布必须关闭；env 在
每个 chunk 内选择一个可复现的 age，同一 worker batch 共享该 age，跨 worker/chunk 覆盖
0-6。env 发布帧、action exact fetch 和 reward age 都读取同一 metadata。日志已验证
requested age 与 actual age 一致；episode 开头因 source frame 不能小于 0，实际 age 会因果截断。

10-step PPO run：

```text
/vepfs-mlp2/c20250301/240403026/async_vla/RLinf/logs/20260812-06:32:15-libero_10_ditonly_ppo_semantic_server_clean
```

固定 trial success 从 step 0 的 33/48（68.75%），到 step 5 的 33/48（68.75%），再到
step 10 的 39/48（81.25%），净提升 12.5pp。逐 trial 有 10 个 failure-to-success 和
4 个 success-to-failure；exact McNemar 双侧 p=0.180，因此是正向 online 信号，但 48 trials
不足以宣称统计显著。actor ratio 维持约 1，KL 和 gradient norm 全程有限，证明 PPO 确实更新
了 DiT，而不是冻结或只评测。step-5 checkpoint 已删除，只保留：

审计修正（2026-08-12）：旧实现没有在每次 validation 开始时重置 deterministic age
stream，因此相同 trial id 在不同 step 使用的 age 序列并不完全相同。上述成功率仍可作为
48-trial aggregate smoke，但逐 trial 转移数和 McNemar p 值不再视为严格配对统计。当前
代码已在每次 `evaluate()` 开始时重置 age stream，后续统计结论必须基于修复后的 fresh run。

同次审计还发现 launcher 导出的 `SEMANTIC_FETCH_DELAY_*`、`PPO_GROUP_SIZE`、
`DENOISING_STEPS` 和 `WEIGHT_SYNC_INTERVAL` 曾未接入 YAML。旧 run 实际使用模型内部的
1500ms fetch delay，而不是脚本声明的 0ms。当前配置已显式消费这些参数；修复版日志必须出现
`fetch_delay_ms=0.0`，否则视为启动失败。

```text
/vepfs-mlp2/c20250301/240403026/async_vla/RLinf/logs/20260812-06:32:15-libero_10_ditonly_ppo_semantic_server_clean/task0_reward_v2_age0to6_closed_loop_long10_v3/checkpoints/global_step_10
```

普通 PPO update 的平均 wallclock 为 74.6s：rollout 60.5s、actor training 12.8s、
weight sync 1.56s。每轮有 24x256=6144 simulator frame，对应约 82.4 aggregate
simulator frame/s（包含 PPO update）或 rollout 阶段约 101.5 frame/s。reward output 的
环境侧取回与同步平均 3.91s，其中 reward 模型计算仅 0.30s；当前瓶颈是 rollout 和 semantic
fetch，不是 1.9M reward adapter。TensorBoard 中 semantic_fetch 是三个并发 rollout rank
的累计嵌套 timer，不能直接当作单进程 wallclock 与 60.5s 相除。

## 6. 训练边界和启动检查

启动日志必须出现：

```text
GR00T execution mode: decoupled
Skipping local VLM construction for DiT-only worker
```

默认可训练参数前缀为：

```text
action_head.model
action_head.packet_age_adapter
action_head.action_history_adapter
action_head.stale_semantic_adapter
action_head.stale_semantic_token_adapter
action_head.value_head
```

第一轮 PPO update 需要检查：

- `actor/ratio_exp_logratio_gap` 接近 0
- `actor/logratio_mean` 与 `actor/approx_kl` 符号一致
- `critic/nonfinite_target_rate` 为 0
- actor/value learning rate 与配置一致
- semantic cache entry 数量增长
- packet age 是有限值且符合 simulator frame 变化

如果出现 logprob shape 不一致、packet key 串环境、critic target 非有限或本地 VLM 被构造，应先停止并修复数据契约，不要直接调学习率。

## 7. PPO 和评测口径

这是 RLinf 原生同步 on-policy actor-critic PPO：一次 rollout 后执行配置中的 PPO update。默认不使用 replay、reward filter、KL 惩罚、接受/拒绝门控和训练状态回滚。

第 0 步 fixed pre-eval 只用于确认初始策略，不代表已经完成 PPO update。至少看到 `Global Step: 1` 以及 actor/critic 指标后，才算训练链路真正跑通。

验证建议：

- 训练过程中每 20 个 PPO update 做一次 validation。
- 最终比较至少使用 400 个固定 trials。
- 少于 60 个 trials 只能作为 wiring smoke，不能作为成功率结论。
- 不要把训练 rollout 的 `success_once` 直接当成固定评测成功率。

## 8. 日志、TensorBoard 和资源

每次运行的日志在 `RLinf/logs/<run>/`。TensorBoard 使用该 run 的 `tensorboard` 子目录：

```bash
tensorboard \
  --logdir RLinf/logs/<run>/tensorboard \
  --port 6009 \
  --bind_all
```

运行时应看到一个 semantic server 使用 GPU0，Ray/RLinf rollout 和 actor 使用 GPU1-3。启动新 run 前先检查：

```bash
nvidia-smi
tmux list-sessions
```

不要删除当前运行使用的 `/tmp/rlinf-ray` session。旧 run 完成后再清理旧 Ray 临时目录、TensorBoard 日志和无用 checkpoint。

## 9. 当前工作状态

- 目标仓库：RLinf，远端为 `https://github.com/zhoutianxing7B/RLinf.git`
- 当前分支：`feat/async-ppo`
- 当前修改：解耦 semantic server、DiT-only PPO、延迟 packet、RLinf worker/runner 和对应测试
- 当前状态：修改未提交、未推送；后续由维护者检查后再决定 commit/push
- Task 0 reward expert：已完成收集、训练、独立 age 0-6 分桶评测和在线闭环
- Task 0 首段闭环：10 个 PPO update 已完成；旧评测的 48-trial aggregate success 为 68.75% -> 81.25%，但旧 age stream 未逐轮重置，不能把 p=0.180 当作严格配对统计
- Task 0 恢复长跑：从 global_step_10 继续到 global_step_50；历史 aggregate 评测依次为 75.00%、64.58%、79.17%、77.08%、79.17%，修复前曲线仅作 smoke 参考
- 当前最佳：原 global_step_10 与恢复长跑 global_step_30 均保留；恢复长跑目录内仅保留 global_step_30
- 后续统计验证：用 fresh process 对 baseline/global_step_10 运行相同 age seed 的至少 400 trials

## 10. 单测和已知检查

语法检查：

```bash
python -m py_compile \
  rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py \
  rlinf/models/embodiment/gr00t/gr00t_n1d7/semantic_server.py \
  rlinf/workers/actor/fsdp_actor_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py
```

解耦相关单测：

```bash
PYTHONPATH=/vepfs-mlp2/c20250301/240403026/async_vla:/vepfs-mlp2/c20250301/240403026/async_vla/RLinf \
python -m pytest \
  tests/unit_tests/test_decoupled_ppo_loss_shapes.py \
  tests/unit_tests/test_gr00t_action_execution.py \
  tests/unit_tests/test_grouped_advantages.py \
  tests/unit_tests/test_libero_reset_assignment.py \
  tests/unit_tests/test_semantic_batch_scheduler.py -q
```

双 adapter 与 reward credit assignment：

```bash
python -m pytest tests/unit_tests/test_shared_semantic_reward.py -q
```

## 11. 2026-08-12 恢复长跑结果

运行目录：

```text
RLinf/logs/20260812-07:52:18-libero_10_ditonly_ppo_semantic_server_clean/
```

训练从此前的 `global_step_10` 恢复，继续执行 40 个真实 PPO update，总 wallclock 为 58 分 24 秒。GPU0 只运行 semantic server，GPU1-3 运行 DiT/env/reward worker；PPO 优化奖励完全来自 Reward Worker，`env_reward_weight=0`、`reward_weight=1`。训练和评测均使用真实 simulator frame 的随机 age 0-6 语义缓存。

| Step | 固定成功数 | 固定成功率 | 相对 step 10 的 fail->success / success->fail | Exact McNemar p |
|---:|---:|---:|---:|---:|
| 10 | 36/48 | 75.00% | 0 / 0 | 1.000 |
| 20 | 31/48 | 64.58% | 8 / 13 | 0.383 |
| 30 | 38/48 | 79.17% | 6 / 4 | 0.754 |
| 40 | 37/48 | 77.08% | 10 / 9 | 1.000 |
| 50 | 38/48 | 79.17% | 8 / 6 | 0.791 |

结论：闭环可以连续训练且未发生单向崩溃，step 30/50 均略高于恢复起点，但 48 trials 下没有统计显著提升，也没有形成单调上升趋势。actor 更新稳定且偏小（40 步平均 `approx_kl=0.00110`、`clip_fraction=0.00462`）；critic 仍是主要薄弱点（平均 `explained_variance=-0.376`，全程为负）。因此该结果证明工程闭环和真实更新成立，但不能据此声称 PPO 已稳定提高成功率。

本次 best-only 保存：

```text
RLinf/logs/20260812-07:52:18-libero_10_ditonly_ppo_semantic_server_clean/task0_reward_v2_age0to6_resume10_long50/checkpoints/global_step_30
```

该 checkpoint 约 11GB。step 40/50 未严格超过 step 30，因此没有重复保存。
