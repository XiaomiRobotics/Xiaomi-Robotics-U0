#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
CONFIG=${CONFIG:-configs/train/u0_4b.yaml}

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-3}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_CUDA_SUPPORT="${NCCL_IB_CUDA_SUPPORT:-1}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-13}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WM_FSDP_DIST_TIMEOUT_SECONDS="${WM_FSDP_DIST_TIMEOUT_SECONDS:-1800}"

NNODES="${NNODES:-${GROUP_WORLD_SIZE:-${SLURM_NNODES:-${PET_NNODES:-${WORLD_SIZE:-}}}}}"
NODE_RANK="${NODE_RANK:-${GROUP_RANK:-${SLURM_NODEID:-${PET_NODE_RANK:-${RANK:-}}}}}"
if [[ -n "${PET_NPROC_PER_NODE:-}" && "${PET_NPROC_PER_NODE}" != "auto" ]]; then
  DEFAULT_NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
else
  DEFAULT_NPROC_PER_NODE=8
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC_PER_NODE}}"
MASTER_ADDR="${MASTER_ADDR:-${GROUP_MASTER_ADDR:-${MASTER_NODE_ADDR:-${MASTER_HOST:-${HEAD_NODE_ADDR:-${PET_MASTER_ADDR:-}}}}}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-}}"

require_positive_integer() {
  local name=$1
  local value=$2
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] ${name} must be a positive integer, got ${value:-<empty>}" >&2
    exit 2
  }
}

require_positive_integer NNODES "${NNODES}"
require_positive_integer NPROC_PER_NODE "${NPROC_PER_NODE}"
require_positive_integer MASTER_PORT "${MASTER_PORT}"
if (( NPROC_PER_NODE != 8 )); then
  echo "[ERROR] this launcher requires NPROC_PER_NODE=8, got ${NPROC_PER_NODE}" >&2
  exit 2
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  echo "[ERROR] NODE_RANK must be an integer in [0, ${NNODES}), got ${NODE_RANK:-<empty>}" >&2
  exit 2
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "[ERROR] MASTER_ADDR must be provided by the platform" >&2
  exit 2
fi
if (( MASTER_PORT > 65535 )); then
  echo "[ERROR] MASTER_PORT must be in [1, 65535], got ${MASTER_PORT}" >&2
  exit 2
fi

CMD=(
  "${TORCHRUN_BIN}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

TORCHRUN_LOG_DIR=${TORCHRUN_LOG_DIR:-}
if [[ -z "${TORCHRUN_LOG_DIR}" && -n "${RESULTS_DIR:-}" ]]; then
  TORCHRUN_LOG_DIR="${RESULTS_DIR}/torchrun/node${NODE_RANK}"
fi
if [[ -n "${TORCHRUN_LOG_DIR}" ]]; then
  mkdir -p "${TORCHRUN_LOG_DIR}"
  CMD+=(--log-dir "${TORCHRUN_LOG_DIR}")
  CMD+=(--redirects "${TORCHRUN_REDIRECTS:-3}")
  CMD+=(--tee "${TORCHRUN_TEE:-0:3}")
fi
CMD+=(-m wm_fsdp.train.bootstrap --config "${CONFIG}")

append_override() {
  local env_name=$1
  local flag=$2
  if [[ -n "${!env_name:-}" ]]; then
    CMD+=("${flag}" "${!env_name}")
  fi
}

MODEL_CHECKPOINT_OVERRIDE=${WM_FSDP_MODEL_CHECKPOINT:-${HF_CHECKPOINT:-}}
TOKENIZER_PATH_OVERRIDE=${WM_FSDP_TOKENIZER_PATH:-${TOKENIZER_PATH:-}}
SPECIAL_TOKENS_OVERRIDE=${WM_FSDP_SPECIAL_TOKENS_FILE:-${SPECIAL_TOKENS_FILE:-}}
ULYSSES_SIZE_OVERRIDE=${WM_FSDP_ULYSSES_SIZE:-${ULYSSES_SIZE:-}}
SHARDING_STRATEGY_OVERRIDE=${WM_FSDP_SHARDING_STRATEGY:-${FSDP_STRATEGY:-}}
NUM_REPLICATE_OVERRIDE=${WM_FSDP_NUM_REPLICATE:-${FSDP_NUM_REPLICATE:-}}
NUM_SHARD_OVERRIDE=${WM_FSDP_NUM_SHARD:-${FSDP_NUM_SHARD:-}}
SYNC_MICRO_BATCH_OVERRIDE=${WM_FSDP_SYNC_EACH_MICRO_BATCH:-${FSDP_SYNC_EACH_MICRO_BATCH:-}}
GRAD_ACCUMULATION_OVERRIDE=${WM_FSDP_GRAD_ACCUMULATION_STEPS:-${GRAD_ACCUMULATION_STEPS:-}}
GLOBAL_BATCH_SIZE_OVERRIDE=${WM_FSDP_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TARGET_GLOBAL_BATCH_SIZE:-}}}
WARMUP_STEPS_OVERRIDE=${WM_FSDP_WARMUP_STEPS:-${WARMUP_STEPS:-}}
LR_OVERRIDE=${WM_FSDP_LR:-${LR:-}}
MIN_LR_OVERRIDE=${WM_FSDP_MIN_LR:-${MIN_LR:-}}
WEIGHT_DECAY_OVERRIDE=${WM_FSDP_WEIGHT_DECAY:-${WEIGHT_DECAY:-}}
MAX_GRAD_NORM_OVERRIDE=${WM_FSDP_MAX_GRAD_NORM:-${MAX_GRAD_NORM:-}}

append_override DATASET_CONFIG --dataset-config
if [[ -n "${MODEL_CHECKPOINT_OVERRIDE}" ]]; then CMD+=(--model-checkpoint "${MODEL_CHECKPOINT_OVERRIDE}"); fi
if [[ -n "${TOKENIZER_PATH_OVERRIDE}" ]]; then CMD+=(--tokenizer-path "${TOKENIZER_PATH_OVERRIDE}"); fi
if [[ -n "${SPECIAL_TOKENS_OVERRIDE}" ]]; then CMD+=(--special-tokens-file "${SPECIAL_TOKENS_OVERRIDE}"); fi
append_override RESULTS_DIR --results-dir
append_override BATCH_SIZE --batch-size
append_override NUM_WORKERS --num-workers
append_override WM_FSDP_PREFETCH_FACTOR --prefetch-factor
if [[ -n "${ULYSSES_SIZE_OVERRIDE}" ]]; then CMD+=(--ulysses-size "${ULYSSES_SIZE_OVERRIDE}"); fi
if [[ -n "${SHARDING_STRATEGY_OVERRIDE}" ]]; then CMD+=(--sharding-strategy "${SHARDING_STRATEGY_OVERRIDE}"); fi
if [[ -n "${NUM_REPLICATE_OVERRIDE}" ]]; then CMD+=(--num-replicate "${NUM_REPLICATE_OVERRIDE}"); fi
if [[ -n "${NUM_SHARD_OVERRIDE}" ]]; then CMD+=(--num-shard "${NUM_SHARD_OVERRIDE}"); fi
if [[ -n "${SYNC_MICRO_BATCH_OVERRIDE}" ]]; then CMD+=(--sync-each-micro-batch "${SYNC_MICRO_BATCH_OVERRIDE}"); fi
append_override WM_FSDP_ACTIVATION_CHECKPOINT --activation-checkpoint
append_override WM_FSDP_ACTIVATION_CHECKPOINT_EVERY_N_LAYERS --activation-checkpoint-every-n-layers
append_override TOTAL_STEPS --total-steps
if [[ -n "${GLOBAL_BATCH_SIZE_OVERRIDE}" ]]; then CMD+=(--global-batch-size "${GLOBAL_BATCH_SIZE_OVERRIDE}"); fi
if [[ -n "${GRAD_ACCUMULATION_OVERRIDE}" ]]; then CMD+=(--grad-accumulation-steps "${GRAD_ACCUMULATION_OVERRIDE}"); fi
if [[ -n "${WARMUP_STEPS_OVERRIDE}" ]]; then CMD+=(--warmup-steps "${WARMUP_STEPS_OVERRIDE}"); fi
if [[ -n "${LR_OVERRIDE}" ]]; then CMD+=(--lr "${LR_OVERRIDE}"); fi
if [[ -n "${MIN_LR_OVERRIDE}" ]]; then CMD+=(--min-lr "${MIN_LR_OVERRIDE}"); fi
if [[ -n "${WEIGHT_DECAY_OVERRIDE}" ]]; then CMD+=(--weight-decay "${WEIGHT_DECAY_OVERRIDE}"); fi
if [[ -n "${MAX_GRAD_NORM_OVERRIDE}" ]]; then CMD+=(--max-grad-norm "${MAX_GRAD_NORM_OVERRIDE}"); fi
append_override SAVE_EVERY --save-every
append_override LOG_EVERY --log-every
append_override RESUME_FROM --resume-from

if [[ -n "${SAVE_FINAL:-}" ]]; then
  case "${SAVE_FINAL}" in
    1) CMD+=(--save-final) ;;
    0) CMD+=(--no-save-final) ;;
    *) echo "[ERROR] SAVE_FINAL must be 0 or 1" >&2; exit 2 ;;
  esac
fi
if [[ -n "${NO_FSDP:-}" ]]; then
  case "${NO_FSDP}" in
    1) CMD+=(--no-fsdp) ;;
    0) ;;
    *) echo "[ERROR] NO_FSDP must be 0 or 1" >&2; exit 2 ;;
  esac
fi

case "${DRY_RUN:-0}" in
  0) ;;
  1)
    printf '[DRY_RUN] topology: NNODES=%q NODE_RANK=%q NPROC_PER_NODE=%q MASTER_ADDR=%q MASTER_PORT=%q CONFIG=%q\n'       "${NNODES}" "${NODE_RANK}" "${NPROC_PER_NODE}" "${MASTER_ADDR}" "${MASTER_PORT}" "${CONFIG}"
    printf '[DRY_RUN] command:'
    printf ' %q' "${CMD[@]}" "$@"
    printf '\n'
    exit 0
    ;;
  *) echo "[ERROR] DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

exec "${CMD[@]}" "$@"
