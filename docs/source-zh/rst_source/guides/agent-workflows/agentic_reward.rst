在 LIBERO 上运行 Agentic Reward SAC
====================================

这套流程在 LIBERO Spatial task 9 上训练 SAC，并让 Luna 在评估边界修改物理
reward program。Luna 不会逐帧调用。每个候选 reward 都会在两个固定 reset
panel 上评估，再根据审计结果接受或回滚。

Luna 可以在 1 到 10 之间校准有界 completion bonus。重复 occupancy 的 bonus
固定为 1，capped completion 的总奖励最多为 20，避免奖励缩放重新产生无界
occupancy return。

准备环境
--------

在 RLinf 仓库根目录运行命令。如果尚未安装 LIBERO 环境，先执行：

.. code-block:: bash

   bash requirements/install.sh embodied --env libero

指定预训练 CNN policy，并创建本次运行的私有目录：

.. code-block:: bash

   export RLINF_RESNET_MODEL_PATH=/absolute/path/to/RLinf-ResNet10-pretrained
   # 可选的纯 actor SFT warmup；它不会作为 reward。
   export ENPIRE_WARMUP_CHECKPOINT=/absolute/path/to/model.pt
   export ENPIRE_RUN_ROOT=/absolute/path/to/agentic-reward-run
   export ENPIRE_PROGRAM_PATH="$ENPIRE_RUN_ROOT/reward_program.json"
   export ENPIRE_AUDIT_DIR="$ENPIRE_RUN_ROOT/agentic_audit"
   mkdir -p "$ENPIRE_RUN_ROOT" "$ENPIRE_AUDIT_DIR"
   cp examples/agentic_reward/programs/libero_spatial_task9_seed.json \
      "$ENPIRE_PROGRAM_PATH"

把 Luna 密钥保存在仓库外。只在本机替换占位符，不要提交生成的文件：

.. code-block:: bash

   umask 077
   printf '%s\n' 'AGENTIC_MODEL_API_KEY=<your-private-key>' \
      > /tmp/agentic_maimai.env
   chmod 600 /tmp/agentic_maimai.env

启动训练
--------

把密钥加载到当前进程环境，然后启动已有配置：

.. code-block:: bash

   set -a
   . /tmp/agentic_maimai.env
   set +a
   export EMBODIED_PATH="$PWD/examples/embodiment"
   export CUDA_VISIBLE_DEVICES=0,1
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl

   python examples/embodiment/train_embodied_agent.py \
      --config-name libero_spatial_task9_enpire_sac

默认长跑使用 40 个训练环境、50 个评估环境、两轮重复评估，并调用
``https://maimai.it.com`` 上的 ``gpt-5.6-luna``。如果只想检查环境和 API
能否连通，在启动命令后追加以下 Hydra 参数：

.. code-block:: bash

   runner.max_steps=5 runner.max_epochs=5 algorithm.update_epoch=1 \
      agentic_reward.controller.baseline_warmup_evaluations=1

查看结果
--------

阅读 ``$ENPIRE_AUDIT_DIR/report.md`` 查看完整决策记录。``state.json`` 保存
controller 状态，``events.jsonl`` 保存每次 proposal、接受和回滚事件。
TensorBoard 指标位于 ``$ENPIRE_RUN_ROOT/tensorboard/``，checkpoint 位于
``$ENPIRE_RUN_ROOT/libero_spatial_task9_enpire_sac/checkpoints/``。

只有 ``reset_panel_a`` 和 ``reset_panel_b`` 都超过 0.70，才算达到配置中的
目标。Simulator success 只用于评估和候选选择，不会作为 SAC reward 输入。
