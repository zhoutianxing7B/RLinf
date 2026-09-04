Run Agentic Reward SAC on LIBERO
================================

This workflow trains LIBERO Spatial task 9 with SAC while Luna revises a
physical reward program at evaluation boundaries. Luna does not run once per
frame. Each candidate is evaluated on two fixed reset panels and is accepted or
rolled back from the audit result.

Prepare the environment
-----------------------

Run commands from the RLinf repository root. Install the LIBERO environment if
you have not already done so:

.. code-block:: bash

   bash requirements/install.sh embodied --env libero

Point RLinf at the pretrained CNN policy and create a private run directory:

.. code-block:: bash

   export RLINF_RESNET_MODEL_PATH=/absolute/path/to/RLinf-ResNet10-pretrained
   # Optional actor-only SFT warmup; this is never used as reward.
   export ENPIRE_WARMUP_CHECKPOINT=/absolute/path/to/model.pt
   export ENPIRE_RUN_ROOT=/absolute/path/to/agentic-reward-run
   export ENPIRE_PROGRAM_PATH="$ENPIRE_RUN_ROOT/reward_program.json"
   export ENPIRE_AUDIT_DIR="$ENPIRE_RUN_ROOT/agentic_audit"
   mkdir -p "$ENPIRE_RUN_ROOT" "$ENPIRE_AUDIT_DIR"
   cp examples/agentic_reward/programs/libero_spatial_task9_seed.json \
      "$ENPIRE_PROGRAM_PATH"

Store the Luna key outside the repository. Replace the placeholder locally; do
not commit the resulting file:

.. code-block:: bash

   umask 077
   printf '%s\n' 'AGENTIC_MODEL_API_KEY=<your-private-key>' \
      > /tmp/agentic_maimai.env
   chmod 600 /tmp/agentic_maimai.env

Start training
--------------

Load the key into the process environment and launch the existing configuration:

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

The default run uses 40 training environments, 50 evaluation environments, two
evaluation repeats, and ``gpt-5.6-luna`` at ``https://maimai.it.com``. To run a
short connectivity check, append these Hydra overrides:

.. code-block:: bash

   runner.max_steps=5 runner.max_epochs=5 algorithm.update_epoch=1 \
      agentic_reward.controller.baseline_warmup_evaluations=1

Read the result
---------------

Open ``$ENPIRE_AUDIT_DIR/report.md`` for the readable decision history.
``state.json`` stores controller state, and ``events.jsonl`` stores individual
proposal, acceptance, and rollback events. TensorBoard metrics are under
``$ENPIRE_RUN_ROOT/tensorboard/``. Checkpoints are under
``$ENPIRE_RUN_ROOT/libero_spatial_task9_enpire_sac/checkpoints/``.

A reward has met the configured target only when both ``reset_panel_a`` and
``reset_panel_b`` exceed 0.70. Simulator success is used for evaluation and
selection only; it is not exposed as the SAC reward.
