# GR00T N1.7 解耦 DiT-only PPO 交接文档

## 1. 目标和范围

本分支的主线是把已经验证过的 **semantic server + DiT-only PPO** 管线完整放进 RLinf。

系统分成两个部分：

- **慢系统**：VLM semantic backbone 独立运行在 semantic server，只负责生成语义向量。
- **快系统**：RLinf rollout/actor 只运行 GR00T N1.7 原版 DiT，读取每个环境最新的语义缓存，并使用 state 和语义 age 产生 action。

PPO 阶段冻结 VLM、state/action encoder、decoder 和无关 adapter，只训练 DiT action model、packet-age adapter 和 value head。

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

## 3. 启动方式

```bash
cd /vepfs-mlp2/c20250301/240403026/async_vla/RLinf
bash examples/embodiment/run_spatial_ditonly_semantic_ppo_clean.sh
```

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

`ACTION_CHUNK_SIZE=16` 是当前控制协议。semantic server 可以异步刷新，但不依赖固定的三步刷新间隔。

## 5. 训练边界和启动检查

启动日志必须出现：

```text
GR00T execution mode: decoupled
Skipping local VLM construction for DiT-only worker
```

默认可训练参数前缀为：

```text
action_head.model
action_head.packet_age_adapter
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

## 6. PPO 和评测口径

这是 RLinf 原生同步 on-policy actor-critic PPO：一次 rollout 后执行配置中的 PPO update。默认不使用 replay、reward filter、KL 惩罚、接受/拒绝门控和训练状态回滚。

第 0 步 fixed pre-eval 只用于确认初始策略，不代表已经完成 PPO update。至少看到 `Global Step: 1` 以及 actor/critic 指标后，才算训练链路真正跑通。

验证建议：

- 训练过程中每 20 个 PPO update 做一次 validation。
- 最终比较至少使用 400 个固定 trials。
- 少于 60 个 trials 只能作为 wiring smoke，不能作为成功率结论。
- 不要把训练 rollout 的 `success_once` 直接当成固定评测成功率。

## 7. 日志、TensorBoard 和资源

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

## 8. 当前工作状态

- 目标仓库：RLinf，远端为 `https://github.com/zhoutianxing7B/RLinf.git`
- 当前分支：`feat/async-ppo`
- 当前修改：解耦 semantic server、DiT-only PPO、延迟 packet、RLinf worker/runner 和对应测试
- 当前状态：修改未提交、未推送；后续由维护者检查后再决定 commit/push
- 当前长训：使用 semantic server + DiT-only PPO，不要切换到 coupled 入口

## 9. 单测和已知检查

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