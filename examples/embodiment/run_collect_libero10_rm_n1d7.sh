#!/usr/bin/env bash

set -euo pipefail

REPO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GR00T_REPO_PATH=$(dirname "${REPO_PATH}")
MODEL_PATH=${GR00T_MODEL_PATH:?Set GR00T_MODEL_PATH to the N1.7 checkpoint}
BACKBONE_PATH=${GR00T_BACKBONE_PATH:?Set GR00T_BACKBONE_PATH}
OUTPUT_ROOT=${REWARD_DATA_ROOT:-${REPO_PATH}/data/libero10_n1d7_reward}
SEMANTIC_GPU_ID=${SEMANTIC_GPU_ID:-0}
COLLECTOR_GPU_IDS=${COLLECTOR_GPU_IDS:-1,2,3}
SEMANTIC_SERVER_PORT=${SEMANTIC_SERVER_PORT:-6677}
SEMANTIC_SERVER_PUBLISH_PORT=${SEMANTIC_SERVER_PUBLISH_PORT:-6678}
TARGET_PER_OUTCOME=${TARGET_PER_OUTCOME:-1500}
NUM_ENVS=${NUM_ENVS:-16}
ACTION_CHUNK=${ACTION_CHUNK:-16}
MAX_DECISIONS=${MAX_DECISIONS:-30}
SEMANTIC_TOKENS=${SEMANTIC_TOKENS:-160}
SEMANTIC_HARD_MAX_AGE_FRAMES=${SEMANTIC_HARD_MAX_AGE_FRAMES:-0}
DELAY_MAX_FRAMES=${DELAY_MAX_FRAMES:-6}
TASK_START=${TASK_START:-0}
TASK_END=${TASK_END:-9}

export PYTHONPATH="${GR00T_REPO_PATH}:${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p "${OUTPUT_ROOT}/logs"

SERVER_LOG="${OUTPUT_ROOT}/logs/semantic_server.log"
SERVER_CMD=(
    python -m rlinf.models.embodiment.gr00t.gr00t_n1d7.semantic_server
    --model-path "${MODEL_PATH}"
    --backbone-model-path "${BACKBONE_PATH}"
    --device cuda:0
    --port "${SEMANTIC_SERVER_PORT}"
    --publish-port "${SEMANTIC_SERVER_PUBLISH_PORT}"
    --dtype bf16
    --local-files-only
    --load-bf16
    --batch-max-requests 8
    --batch-target-envs "${NUM_ENVS}"
    --batch-wait-ms 2
    --rpc-batch-wait-ms 2
    --raw-preprocess-workers 12
    --text-padding-tokens "${SEMANTIC_TOKENS}"
    --cache-history-size 64
)
CUDA_VISIBLE_DEVICES="${SEMANTIC_GPU_ID}" "${SERVER_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
WORKER_PIDS=()

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 240); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        tail -n 100 "${SERVER_LOG}" >&2
        exit 1
    fi
    if grep -q "Semantic server listening" "${SERVER_LOG}"; then
        break
    fi
    sleep 1
done
if ! grep -q "Semantic server listening" "${SERVER_LOG}"; then
    tail -n 100 "${SERVER_LOG}" >&2
    exit 1
fi

IFS=, read -r -a COLLECTOR_GPUS <<<"${COLLECTOR_GPU_IDS}"
if (( ${#COLLECTOR_GPUS[@]} == 0 )); then
    echo "COLLECTOR_GPU_IDS must contain at least one GPU" >&2
    exit 1
fi

for worker_index in "${!COLLECTOR_GPUS[@]}"; do
    gpu_id=${COLLECTOR_GPUS[${worker_index}]}
    (
        job_index=0
        for task_id in $(seq "${TASK_START}" "${TASK_END}"); do
            for stream in success failure; do
                if (( job_index % ${#COLLECTOR_GPUS[@]} != worker_index )); then
                    job_index=$((job_index + 1))
                    continue
                fi
                if [[ "${stream}" == "success" ]]; then
                    inference_mode=eval
                    noise_std=0.0
                    dropout_prob=0.0
                    success_target=${TARGET_PER_OUTCOME}
                    failure_target=0
                else
                    inference_mode=train
                    noise_std=0.02
                    dropout_prob=0.05
                    success_target=0
                    failure_target=${TARGET_PER_OUTCOME}
                fi
                log_path="${OUTPUT_ROOT}/logs/task_${task_id}_${stream}.log"
                COLLECTOR_CMD=(
                    python
                    "${REPO_PATH}/examples/embodiment/collect_libero10_rm_data_n1d7.py"
                    --checkpoint "${MODEL_PATH}"
                    --backbone-model-path "${BACKBONE_PATH}"
                    --output "${OUTPUT_ROOT}"
                    --tasks "${task_id}"
                    --successes-per-task "${success_target}"
                    --failures-per-task "${failure_target}"
                    --trajectory-prefix "${stream}_w${worker_index}_"
                    --source-stream "${stream}"
                    --num-envs "${NUM_ENVS}"
                    --action-chunk "${ACTION_CHUNK}"
                    --max-decisions "${MAX_DECISIONS}"
                    --inference-mode "${inference_mode}"
                    --action-noise-std "${noise_std}"
                    --action-dropout-prob "${dropout_prob}"
                    --semantic-server
                    --semantic-server-port "${SEMANTIC_SERVER_PORT}"
                    --semantic-server-publish-port "${SEMANTIC_SERVER_PUBLISH_PORT}"
                    --semantic-hard-max-age-frames "${SEMANTIC_HARD_MAX_AGE_FRAMES}"
                    --delay-max-frames "${DELAY_MAX_FRAMES}"
                    --semantic-tokens "${SEMANTIC_TOKENS}"
                    # Collector processes restart episode generations at zero.
                    # Give each task/stream a distinct server cache namespace.
                    --env-id-offset "$(((worker_index + 1) * 10000000 + job_index * 100000))"
                    --device cuda:0
                )
                CUDA_VISIBLE_DEVICES="${gpu_id}" "${COLLECTOR_CMD[@]}" >"${log_path}" 2>&1
                job_index=$((job_index + 1))
            done
        done
    ) &
    WORKER_PIDS+=("$!")
done

status=0
for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" || status=$?
done
exit "${status}"
