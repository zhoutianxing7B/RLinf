#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EMBODIED_PATH="${ROOT}/examples/embodiment"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rlinf-ray}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${RAY_TMPDIR}"
export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${ROOT}/../checkpoints/GR00T-N1.7-LIBERO/libero_spatial}"
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${ROOT}/../.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
export ACTOR_GPU_PLACEMENT="${ACTOR_GPU_PLACEMENT:-1-3}"
export ROLLOUT_GPU_PLACEMENT="${ROLLOUT_GPU_PLACEMENT:-1-3}"
export ENV_GPU_PLACEMENT="${ENV_GPU_PLACEMENT:-1-3}"

for path in "${GR00T_MODEL_PATH}" "${GR00T_BACKBONE_PATH}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing model directory: ${path}" >&2
    exit 2
  fi
done

cd "${ROOT}"
exec python examples/embodiment/train_embodied_agent.py   --config-path config   --config-name libero_spatial_ppo_gr00t_n1d7_coupled   "$@"
