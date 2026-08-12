#!/usr/bin/env bash

set -euo pipefail

CONFIG_NAME=${1:-libero_10_sync_ppo_gr00t_n1d7_central_cache}
MODEL_PATH=${GR00T_MODEL_PATH:-/vepfs-mlp2/c20250301/240403026/async_vla/async_libero_runs/libero10_decoupled_sft_0to8/S_libero10_decoupled_0to8_scalar_age_s4_a4/checkpoint-5000}
export GR00T_MODEL_PATH="${MODEL_PATH}"
BACKBONE_PATH=${GR00T_BACKBONE_PATH:-/vepfs-mlp2/c20250301/240403026/async_vla/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}
BASE_PORT=${SEMANTIC_SERVER_PORT:-6677}
MAX_USED_MIB=${MAX_USED_MIB:-1024}
ENV_WORKERS_PER_GPU=${ENV_WORKERS_PER_GPU:-1}
TARGET_TRAIN_ENVS=${TARGET_TRAIN_ENVS:-128}
TARGET_EVAL_ENVS=${TARGET_EVAL_ENVS:-${TARGET_TRAIN_ENVS}}
MAX_DIT_GPUS=${MAX_DIT_GPUS:-0}
ALLOWED_GPU_IDS=${ALLOWED_GPU_IDS:-0,1,2,3,4,5,6,7}
NUM_SEMANTIC_GPUS=${NUM_SEMANTIC_GPUS:-1}
SEMANTIC_GPU_IDS=${SEMANTIC_GPU_IDS:-0}
DIT_GPU_IDS=${DIT_GPU_IDS:-1,2,3,4,5,6,7}
# Actor training can use a wider GPU set than rollout inference. This lets
# external semantic servers colocate with idle actor ranks while rollout
# workers remain on dedicated GPUs.
ACTOR_GPU_IDS=${ACTOR_GPU_IDS:-${DIT_GPU_IDS}}
SEMANTIC_BATCH_WAIT_MS=${SEMANTIC_BATCH_WAIT_MS:-0}
SEMANTIC_RPC_BATCH_WAIT_MS=${SEMANTIC_RPC_BATCH_WAIT_MS:-2}
export ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-16}
export TRAIN_ROLLOUT_STEPS=${TRAIN_ROLLOUT_STEPS:-480}
export DENOISING_STEPS=${DENOISING_STEPS:-4}
export BALANCE_TRAIN_TASK_ASSIGNMENT=${BALANCE_TRAIN_TASK_ASSIGNMENT:-true}
export SEMANTIC_BOUNDARY_PUBLISH=${SEMANTIC_BOUNDARY_PUBLISH:-false}
export SEMANTIC_ENV_BOUNDARY_PUBLISH=${SEMANTIC_ENV_BOUNDARY_PUBLISH:-false}
export SEMANTIC_PUBLISH_INTERVAL_FRAMES=${SEMANTIC_PUBLISH_INTERVAL_FRAMES:-0}
export SEMANTIC_MID_CHUNK_PUBLISH=${SEMANTIC_MID_CHUNK_PUBLISH:-true}
export SEMANTIC_MID_CHUNK_FRAME=${SEMANTIC_MID_CHUNK_FRAME:-10}
export SEMANTIC_MID_CHUNK_MIN_FRAME=${SEMANTIC_MID_CHUNK_MIN_FRAME:-10}
export SEMANTIC_MID_CHUNK_STAGGER_BY_RANK=${SEMANTIC_MID_CHUNK_STAGGER_BY_RANK:-false}
export SEMANTIC_FETCH_MAX_WAIT_MS=${SEMANTIC_FETCH_MAX_WAIT_MS:-30000}
export SEMANTIC_FETCH_TARGET_AGE_FRAMES=${SEMANTIC_FETCH_TARGET_AGE_FRAMES:-12}
export SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES=${SEMANTIC_FETCH_HARD_MAX_AGE_FRAMES:-12}
export SEMANTIC_FETCH_DELAY_FRACTION=${SEMANTIC_FETCH_DELAY_FRACTION:-0.0}
export SEMANTIC_FETCH_DELAY_INITIAL_MS=${SEMANTIC_FETCH_DELAY_INITIAL_MS:-0}
export SEMANTIC_FETCH_DELAY_MIN_MS=${SEMANTIC_FETCH_DELAY_MIN_MS:-0}
export SEMANTIC_FETCH_DELAY_MAX_MS=${SEMANTIC_FETCH_DELAY_MAX_MS:-0}

SEMANTIC_PREPROCESS_WORKERS=${SEMANTIC_PREPROCESS_WORKERS:-12}
SEMANTIC_OMP_NUM_THREADS=${SEMANTIC_OMP_NUM_THREADS:-1}
SEMANTIC_CPUSET=${SEMANTIC_CPUSET:-0-1,32-33,64-65,96-97}
WORKLOAD_CPUSET=${WORKLOAD_CPUSET:-14-31,46-63,78-95,110-127}
SEMANTIC_PREPROCESS_PROXY=${SEMANTIC_PREPROCESS_PROXY:-true}
SEMANTIC_PROXY_CPUSET=${SEMANTIC_PROXY_CPUSET:-2-13,34-45,66-77,98-109}
SEMANTIC_PROXY_PORT_OFFSET=${SEMANTIC_PROXY_PORT_OFFSET:-1000}
SEMANTIC_TRANSPORT_QUANTIZATION=${SEMANTIC_TRANSPORT_QUANTIZATION:-none}
export SEMANTIC_TEXT_PADDING_TOKENS=${SEMANTIC_TEXT_PADDING_TOKENS:-160}
SEMANTIC_CACHE_HISTORY_SIZE=${SEMANTIC_CACHE_HISTORY_SIZE:-32}
COLOCATED_SEMANTIC_FETCH_PAUSE_MS=${COLOCATED_SEMANTIC_FETCH_PAUSE_MS:-0}
export PPO_MAX_EPOCHS=${PPO_MAX_EPOCHS:-${PPO_MAX_STEPS:-2000}}
export TRAIN_ROLLOUT_EPOCH=${TRAIN_ROLLOUT_EPOCH:-1}
export RLINF_FORCE_SYNC_CHANNEL_TRANSPORT=${RLINF_FORCE_SYNC_CHANNEL_TRANSPORT:-1}
ONLY_EVAL=${ONLY_EVAL:-false}

if (( ENV_WORKERS_PER_GPU < 1 || NUM_SEMANTIC_GPUS < 1 )); then
    echo "ENV_WORKERS_PER_GPU and NUM_SEMANTIC_GPUS must be at least 1." >&2
    exit 1
fi

mapfile -t FREE_GPUS < <(
    nvidia-smi --id="${ALLOWED_GPU_IDS}" --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, -v max_used="${MAX_USED_MIB}" '{gsub(/ /, "", $2); if (($2 + 0) <= max_used) print $1}'
)
IFS=, read -r -a ALLOWED_GPUS <<<"${ALLOWED_GPU_IDS}"
if [[ -n "${SEMANTIC_GPU_IDS}" || -n "${DIT_GPU_IDS}" ]]; then
    if [[ -z "${SEMANTIC_GPU_IDS}" || -z "${DIT_GPU_IDS}" ]]; then
        echo "SEMANTIC_GPU_IDS and DIT_GPU_IDS must be set together." >&2
        exit 1
    fi
    IFS=, read -r -a SEMANTIC_GPUS <<<"${SEMANTIC_GPU_IDS}"
    IFS=, read -r -a DIT_GPUS <<<"${DIT_GPU_IDS}"
    IFS=, read -r -a ACTOR_GPUS <<<"${ACTOR_GPU_IDS}"
    NUM_SEMANTIC_GPUS=${#SEMANTIC_GPUS[@]}
    declare -A SEMANTIC_GPU_SEEN=()
    declare -A DIT_GPU_SEEN=()
    declare -A ACTOR_GPU_SEEN=()
    for selected_gpu in "${SEMANTIC_GPUS[@]}"; do
        if [[ -n "${SEMANTIC_GPU_SEEN[$selected_gpu]:-}" ]]; then
            echo "Duplicate semantic GPU ${selected_gpu}." >&2
            exit 1
        fi
        SEMANTIC_GPU_SEEN[$selected_gpu]=1
    done
    for selected_gpu in "${DIT_GPUS[@]}"; do
        if [[ -n "${DIT_GPU_SEEN[$selected_gpu]:-}" ]]; then
            echo "Duplicate DiT GPU ${selected_gpu}." >&2
            exit 1
        fi
        DIT_GPU_SEEN[$selected_gpu]=1
    done
    for selected_gpu in "${ACTOR_GPUS[@]}"; do
        if [[ -n "${ACTOR_GPU_SEEN[$selected_gpu]:-}" ]]; then
            echo "Duplicate actor GPU ${selected_gpu}." >&2
            exit 1
        fi
        ACTOR_GPU_SEEN[$selected_gpu]=1
    done
    for selected_gpu in "${SEMANTIC_GPUS[@]}" "${DIT_GPUS[@]}" "${ACTOR_GPUS[@]}"; do
        if [[ ! " ${ALLOWED_GPUS[*]} " =~ " ${selected_gpu} " ]]; then
            echo "Requested GPU ${selected_gpu} is outside ALLOWED_GPU_IDS=${ALLOWED_GPU_IDS}." >&2
            exit 1
        fi
        if [[ ! " ${FREE_GPUS[*]} " =~ " ${selected_gpu} " ]]; then
            echo "Requested GPU ${selected_gpu} is not free." >&2
            exit 1
        fi
    done
else
    if (( ${#FREE_GPUS[@]} < NUM_SEMANTIC_GPUS + 1 )); then
        echo "Need NUM_SEMANTIC_GPUS plus at least one free DiT GPU." >&2
        exit 1
    fi
    SEMANTIC_GPUS=( "${FREE_GPUS[@]:0:NUM_SEMANTIC_GPUS}" )
    DIT_GPUS=( "${FREE_GPUS[@]:NUM_SEMANTIC_GPUS}" )
    ACTOR_GPUS=( "${DIT_GPUS[@]}" )
fi
if (( MAX_DIT_GPUS > 0 && MAX_DIT_GPUS < ${#DIT_GPUS[@]} )); then
    DIT_GPUS=( "${DIT_GPUS[@]:0:MAX_DIT_GPUS}" )
fi
DIT_GPU_CSV=$(IFS=,; echo "${DIT_GPUS[*]}")
if (( ${#ACTOR_GPUS[@]} < 1 )); then
    echo "ACTOR_GPU_IDS must contain at least one GPU." >&2
    exit 1
fi

# RLinf requires resource ranks in each placement string to be ascending.
# Keep physical GPU ranks identical to RLinf resource ranks, including semantic-only GPUs.
mapfile -t WORKLOAD_GPUS < <(
    printf '%s\n' "${SEMANTIC_GPUS[@]}" "${ACTOR_GPUS[@]}" "${DIT_GPUS[@]}" | sort -n -u
)
WORKLOAD_GPU_CSV=$(IFS=,; echo "${WORKLOAD_GPUS[*]}")
declare -A WORKLOAD_LOCAL_RANK=()
for gpu_index in "${!WORKLOAD_GPUS[@]}"; do
    WORKLOAD_LOCAL_RANK[${WORKLOAD_GPUS[$gpu_index]}]=${gpu_index}
done

ACTOR_GPU_PLACEMENT=""
for selected_gpu in "${ACTOR_GPUS[@]}"; do
    if [[ -n "${ACTOR_GPU_PLACEMENT}" ]]; then
        ACTOR_GPU_PLACEMENT+=","
    fi
    ACTOR_GPU_PLACEMENT+="${WORKLOAD_LOCAL_RANK[$selected_gpu]}"
done
export ACTOR_GPU_PLACEMENT
DIT_REPLICAS_PER_GPU=${DIT_REPLICAS_PER_GPU:-1}
if (( DIT_REPLICAS_PER_GPU < 1 )); then
    echo "DIT_REPLICAS_PER_GPU must be at least 1." >&2
    exit 1
fi
ROLLOUT_WORLD_SIZE=$(( ${#DIT_GPUS[@]} * DIT_REPLICAS_PER_GPU ))
ROLLOUT_GPU_PLACEMENT=""
for gpu_index in "${!DIT_GPUS[@]}"; do
    first_rank=$((gpu_index * DIT_REPLICAS_PER_GPU))
    last_rank=$((first_rank + DIT_REPLICAS_PER_GPU - 1))
    rank_spec="${first_rank}"
    if (( last_rank != first_rank )); then
        rank_spec="${first_rank}-${last_rank}"
    fi
    if [[ -n "${ROLLOUT_GPU_PLACEMENT}" ]]; then
        ROLLOUT_GPU_PLACEMENT+=","
    fi
    ROLLOUT_GPU_PLACEMENT+="${WORKLOAD_LOCAL_RANK[${DIT_GPUS[$gpu_index]}]}:${rank_spec}"
done
export ROLLOUT_GPU_PLACEMENT
ENV_WORKERS_PER_PHYSICAL_GPU=$((ENV_WORKERS_PER_GPU * DIT_REPLICAS_PER_GPU))
ENV_WORLD_SIZE=$(( ${#DIT_GPUS[@]} * ENV_WORKERS_PER_PHYSICAL_GPU ))
# Process one rollout-rank packet at a time. Merging all ranks into one large
# VLM forward improves nominal batching but makes every environment wait for
# the slowest packet and materially increases semantic age.
SEMANTIC_BATCH_MAX_REQUESTS=${SEMANTIC_BATCH_MAX_REQUESTS:-1}
SEMANTIC_BATCH_TARGET_REQUESTS=${SEMANTIC_BATCH_TARGET_REQUESTS:-0}
ENV_GPU_PLACEMENT=""
for gpu_index in "${!DIT_GPUS[@]}"; do
    first_rank=$((gpu_index * ENV_WORKERS_PER_PHYSICAL_GPU))
    last_rank=$((first_rank + ENV_WORKERS_PER_PHYSICAL_GPU - 1))
    rank_spec="${first_rank}"
    if (( last_rank != first_rank )); then
        rank_spec="${first_rank}-${last_rank}"
    fi
    if [[ -n "${ENV_GPU_PLACEMENT}" ]]; then
        ENV_GPU_PLACEMENT+=","
    fi
    ENV_GPU_PLACEMENT+="${WORKLOAD_LOCAL_RANK[${DIT_GPUS[$gpu_index]}]}:${rank_spec}"
done
export ENV_GPU_PLACEMENT
export TRAIN_NUM_ENVS=$((TARGET_TRAIN_ENVS / ENV_WORLD_SIZE * ENV_WORLD_SIZE))
export EVAL_NUM_ENVS=$((TARGET_EVAL_ENVS / ENV_WORLD_SIZE * ENV_WORLD_SIZE))
PPO_GROUP_SIZE=${PPO_GROUP_SIZE:-1}
export PPO_GROUP_SIZE
if (( TRAIN_NUM_ENVS % PPO_GROUP_SIZE != 0 )); then
    echo "TRAIN_NUM_ENVS must be divisible by PPO_GROUP_SIZE: ${TRAIN_NUM_ENVS} vs ${PPO_GROUP_SIZE}" >&2
    exit 2
fi
if [[ "${BALANCE_TRAIN_TASK_ASSIGNMENT,,}" == "true" ]] && (( TRAIN_NUM_ENVS % (10 * PPO_GROUP_SIZE) != 0 )); then
    echo "Balanced LIBERO-10 PPO requires TRAIN_NUM_ENVS divisible by 10 * PPO_GROUP_SIZE: ${TRAIN_NUM_ENVS} vs ${PPO_GROUP_SIZE}" >&2
    exit 2
fi
if (( TRAIN_ROLLOUT_STEPS % ACTION_CHUNK_SIZE != 0 )); then
    echo "TRAIN_ROLLOUT_STEPS must be divisible by ACTION_CHUNK_SIZE: ${TRAIN_ROLLOUT_STEPS} vs ${ACTION_CHUNK_SIZE}" >&2
    exit 2
fi
# A threshold of one starts the newest complete packet immediately. The packet
# itself remains vectorized over all environments owned by its rollout rank.
SEMANTIC_BATCH_TARGET_ENVS=${SEMANTIC_BATCH_TARGET_ENVS:-1}
SEMANTIC_BOOTSTRAP_TARGET_ENVS=${SEMANTIC_BOOTSTRAP_TARGET_ENVS:-0}
SEMANTIC_BOOTSTRAP_WAIT_MS=${SEMANTIC_BOOTSTRAP_WAIT_MS:-30000}
FULL_ROLLOUT_SAMPLES=$((TRAIN_NUM_ENVS * TRAIN_ROLLOUT_STEPS * TRAIN_ROLLOUT_EPOCH / ACTION_CHUNK_SIZE))
PPO_GLOBAL_BATCH_SIZE=${PPO_GLOBAL_BATCH_SIZE:-1008}
export PPO_GLOBAL_BATCH_SIZE
if (( FULL_ROLLOUT_SAMPLES % PPO_GLOBAL_BATCH_SIZE != 0 )); then
    echo "Rollout samples must be divisible by PPO_GLOBAL_BATCH_SIZE: ${FULL_ROLLOUT_SAMPLES} vs ${PPO_GLOBAL_BATCH_SIZE}" >&2
    exit 2
fi
if (( PPO_GLOBAL_BATCH_SIZE % ${#ACTOR_GPUS[@]} != 0 )); then
    echo "PPO_GLOBAL_BATCH_SIZE must be divisible by actor GPU count: ${PPO_GLOBAL_BATCH_SIZE} vs ${#ACTOR_GPUS[@]}" >&2
    exit 2
fi
export WEIGHT_SYNC_INTERVAL=${WEIGHT_SYNC_INTERVAL:-1}

if [[ -n "${SEMANTIC_SERVER_PORTS:-}" ]]; then
    IFS=, read -r -a FETCH_PORTS <<<"${SEMANTIC_SERVER_PORTS}"
else
    FETCH_PORTS=()
    for ((index = 0; index < NUM_SEMANTIC_GPUS; index++)); do
        FETCH_PORTS+=( "$((BASE_PORT + index * 2))" )
    done
fi
if [[ -n "${SEMANTIC_SERVER_PUBLISH_PORTS:-}" ]]; then
    IFS=, read -r -a PUBLISH_PORTS <<<"${SEMANTIC_SERVER_PUBLISH_PORTS}"
else
    PUBLISH_PORTS=()
    for fetch_port in "${FETCH_PORTS[@]}"; do
        PUBLISH_PORTS+=( "$((fetch_port + 1))" )
    done
fi
if (( ${#FETCH_PORTS[@]} != NUM_SEMANTIC_GPUS || ${#PUBLISH_PORTS[@]} != NUM_SEMANTIC_GPUS )); then
    echo "Semantic fetch/publish port counts must match NUM_SEMANTIC_GPUS." >&2
    exit 1
fi
FETCH_PORT_CSV=$(IFS=,; echo "${FETCH_PORTS[*]}")
PUBLISH_PORT_CSV=$(IFS=,; echo "${PUBLISH_PORTS[*]}")
INTERNAL_PUBLISH_PORTS=("${PUBLISH_PORTS[@]}")
if [[ "${SEMANTIC_PREPROCESS_PROXY}" == "true" ]]; then
    INTERNAL_PUBLISH_PORTS=()
    for publish_port in "${PUBLISH_PORTS[@]}"; do
        INTERNAL_PUBLISH_PORTS+=( "$((publish_port + SEMANTIC_PROXY_PORT_OFFSET))" )
    done
fi
INTERNAL_PUBLISH_PORT_CSV=$(IFS=,; echo "${INTERNAL_PUBLISH_PORTS[*]}")

REPO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GR00T_REPO_PATH=$(dirname "${REPO_PATH}")
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
export PYTHONPATH="${GR00T_REPO_PATH}:${REPO_PATH}:${PYTHONPATH:-}"
export TMPDIR="${RLINF_TMPDIR:-/tmp/rlinf}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rlinf-ray}"
export RLINF_RAY_TEMP_DIR="${RLINF_RAY_TEMP_DIR:-${RAY_TMPDIR}}"
export RLINF_FORCE_LOCAL_RAY=1
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"

echo "semantic_gpus=$(IFS=,; echo "${SEMANTIC_GPUS[*]}") semantic_ports=${FETCH_PORT_CSV} semantic_publish_ports=${PUBLISH_PORT_CSV} semantic_internal_publish_ports=${INTERNAL_PUBLISH_PORT_CSV} semantic_preprocess_proxy=${SEMANTIC_PREPROCESS_PROXY} semantic_proxy_cpuset=${SEMANTIC_PROXY_CPUSET:-none} semantic_batch_max_requests=${SEMANTIC_BATCH_MAX_REQUESTS} semantic_batch_target_requests=${SEMANTIC_BATCH_TARGET_REQUESTS} semantic_batch_target_envs=${SEMANTIC_BATCH_TARGET_ENVS} semantic_batch_wait_ms=${SEMANTIC_BATCH_WAIT_MS} semantic_bootstrap_target_envs=${SEMANTIC_BOOTSTRAP_TARGET_ENVS} semantic_bootstrap_wait_ms=${SEMANTIC_BOOTSTRAP_WAIT_MS} semantic_preprocess_workers=${SEMANTIC_PREPROCESS_WORKERS} semantic_omp_threads=${SEMANTIC_OMP_NUM_THREADS} semantic_cpuset=${SEMANTIC_CPUSET:-none} workload_cpuset=${WORKLOAD_CPUSET:-none} semantic_rpc_batch_wait_ms=${SEMANTIC_RPC_BATCH_WAIT_MS} semantic_text_padding_tokens=${SEMANTIC_TEXT_PADDING_TOKENS} rollout_gpus=${DIT_GPU_CSV} actor_gpus=$(IFS=,; echo "${ACTOR_GPUS[*]}") workload_gpus=${WORKLOAD_GPU_CSV} dit_replicas_per_gpu=${DIT_REPLICAS_PER_GPU} actor_placement=${ACTOR_GPU_PLACEMENT} rollout_placement=${ROLLOUT_GPU_PLACEMENT} env_placement=${ENV_GPU_PLACEMENT} rollout_world_size=${ROLLOUT_WORLD_SIZE} env_world_size=${ENV_WORLD_SIZE} train_envs=${TRAIN_NUM_ENVS} global_batch=${PPO_GLOBAL_BATCH_SIZE} semantic_transport_quantization=${SEMANTIC_TRANSPORT_QUANTIZATION}" | tee "${LOG_DIR}/placement.log"

SERVER_PIDS=()
SERVER_LOGS=()
READY_PATTERNS=()
cleanup() {
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for index in "${!SEMANTIC_GPUS[@]}"; do
    server_log="${LOG_DIR}/semantic_server_${FETCH_PORTS[$index]}.log"
    SERVER_LOGS+=( "${server_log}" )
    READY_PATTERNS+=( "Semantic server listening" )
    SERVER_CMD=(
        python -m rlinf.models.embodiment.gr00t.gr00t_n1d7.semantic_server
        --model-path "${MODEL_PATH}"
        --backbone-model-path "${BACKBONE_PATH}"
        --device cuda:0
        --port "${FETCH_PORTS[$index]}"
        --publish-port "${INTERNAL_PUBLISH_PORTS[$index]}"
        --dtype bf16
        --local-files-only
        --load-bf16
        --batch-max-requests "${SEMANTIC_BATCH_MAX_REQUESTS}"
        --batch-target-requests "${SEMANTIC_BATCH_TARGET_REQUESTS}"
        --batch-target-envs "${SEMANTIC_BATCH_TARGET_ENVS}"
        --batch-wait-ms "${SEMANTIC_BATCH_WAIT_MS}"
        --bootstrap-target-envs "${SEMANTIC_BOOTSTRAP_TARGET_ENVS}"
        --bootstrap-wait-ms "${SEMANTIC_BOOTSTRAP_WAIT_MS}"
        --rpc-batch-wait-ms "${SEMANTIC_RPC_BATCH_WAIT_MS}"
        --raw-preprocess-workers "${SEMANTIC_PREPROCESS_WORKERS}"
        --text-padding-tokens "${SEMANTIC_TEXT_PADDING_TOKENS}"
        --cache-history-size "${SEMANTIC_CACHE_HISTORY_SIZE}"
        --transport-quantization "${SEMANTIC_TRANSPORT_QUANTIZATION}"
    )
    if [[ "${SEMANTIC_PREPROCESS_PROXY}" == "true" ]]; then
        SERVER_CMD+=( --disable-raw-preprocessing )
    fi
    for dit_gpu in "${DIT_GPUS[@]}"; do
        if [[ "${SEMANTIC_GPUS[$index]}" == "${dit_gpu}" ]]; then
            SERVER_CMD+=(
                --fetch-pause-ms "${COLOCATED_SEMANTIC_FETCH_PAUSE_MS}"
            )
            break
        fi
    done
    if [[ -n "${SEMANTIC_CPUSET}" ]]; then
        CUDA_VISIBLE_DEVICES="${SEMANTIC_GPUS[$index]}" OMP_NUM_THREADS="${SEMANTIC_OMP_NUM_THREADS}" MKL_NUM_THREADS="${SEMANTIC_OMP_NUM_THREADS}" taskset -c "${SEMANTIC_CPUSET}" "${SERVER_CMD[@]}" >"${server_log}" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="${SEMANTIC_GPUS[$index]}" OMP_NUM_THREADS="${SEMANTIC_OMP_NUM_THREADS}" MKL_NUM_THREADS="${SEMANTIC_OMP_NUM_THREADS}" "${SERVER_CMD[@]}" >"${server_log}" 2>&1 &
    fi
    SERVER_PIDS+=( "$!" )
    if [[ "${SEMANTIC_PREPROCESS_PROXY}" == "true" ]]; then
        proxy_log="${LOG_DIR}/semantic_preprocess_proxy_${PUBLISH_PORTS[$index]}.log"
        PROXY_CMD=(
            python -m rlinf.models.embodiment.gr00t.gr00t_n1d7.semantic_preprocess_proxy
            --model-path "${MODEL_PATH}"
            --port "${PUBLISH_PORTS[$index]}"
            --target-port "${INTERNAL_PUBLISH_PORTS[$index]}"
            --workers "${SEMANTIC_PREPROCESS_WORKERS}"
            --batch-max-requests "${SEMANTIC_BATCH_MAX_REQUESTS}"
            --batch-target-envs "${SEMANTIC_BATCH_TARGET_ENVS}"
            --batch-wait-ms "${SEMANTIC_BATCH_WAIT_MS}"
            --local-files-only
            --text-padding-tokens "${SEMANTIC_TEXT_PADDING_TOKENS}"
        )
        SERVER_LOGS+=( "${proxy_log}" )
        READY_PATTERNS+=( "Semantic preprocess proxy listening" )
        if [[ -n "${SEMANTIC_PROXY_CPUSET}" ]]; then
            CUDA_VISIBLE_DEVICES="" taskset -c "${SEMANTIC_PROXY_CPUSET}" "${PROXY_CMD[@]}" >"${proxy_log}" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES="" "${PROXY_CMD[@]}" >"${proxy_log}" 2>&1 &
        fi
        SERVER_PIDS+=( "$!" )
    fi
done

for index in "${!SERVER_PIDS[@]}"; do
    ready=false
    for _ in $(seq 1 240); do
        if ! kill -0 "${SERVER_PIDS[$index]}" 2>/dev/null; then
            tail -n 100 "${SERVER_LOGS[$index]}" >&2
            exit 1
        fi
        if grep -q "${READY_PATTERNS[$index]}" "${SERVER_LOGS[$index]}"; then
            ready=true
            break
        fi
        sleep 1
    done
    if [[ "${ready}" != true ]]; then
        tail -n 100 "${SERVER_LOGS[$index]}" >&2
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES="${WORKLOAD_GPU_CSV}"
export SEMANTIC_SERVER_PORT="${FETCH_PORT_CSV}"
export SEMANTIC_SERVER_PUBLISH_PORT="${PUBLISH_PORT_CSV}"
WORKLOAD_PREFIX=()
if [[ -n "${WORKLOAD_CPUSET}" ]]; then
    WORKLOAD_PREFIX=(taskset -c "${WORKLOAD_CPUSET}")
fi
if [[ "${ONLY_EVAL}" == "true" ]]; then
    "${WORKLOAD_PREFIX[@]}" python "${REPO_PATH}/evaluations/eval_embodied_agent.py" \
        --config-path "${REPO_PATH}/examples/embodiment/config" \
        --config-name "${CONFIG_NAME}" \
        "runner.logger.log_path=${LOG_DIR}"
else
    "${WORKLOAD_PREFIX[@]}" bash "${REPO_PATH}/examples/embodiment/run_embodiment.sh" "${CONFIG_NAME}" LIBERO
fi
