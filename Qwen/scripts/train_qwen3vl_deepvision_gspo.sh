#!/bin/bash
# Single-node-first VERL GSPO launcher for DeepVision-103K with vLLM rollout and rule-based reward.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/../../envs/ocrflow/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}"
    return
  fi
  "$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
}

TIMESTAMP="${QWEN3VL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUNTIME_ENV_STAMP="${QWEN3VL_RUNTIME_ENV_STAMP:-$TIMESTAMP}"
VISIBLE_GPUS="$(count_visible_gpus)"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "$VISIBLE_GPUS" =~ ^[0-9]+$ ]] && [[ "$VISIBLE_GPUS" -gt 0 ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((VISIBLE_GPUS - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

PROJECT_BASE_CONFIG="${PROJECT_BASE_CONFIG:-$REPO_ROOT/Qwen/configs/rl/deepvision_gspo.yaml}"
PROJECT_CONFIG="${PROJECT_CONFIG:-}"
MODEL_PATH="${MODEL_PATH:-/share/project/xiyan/sources/DeepSeek-OCR/Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/Qwen/data/deepvision_103k_verl}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/Qwen/checkpoints/qwen3vl-2b/verl/deepvision_gspo/run_${TIMESTAMP}}"

NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$VISIBLE_GPUS}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
SP_SIZE="${SP_SIZE:-1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-8}"
ROLLOUT_LOGPROB_MB="${ROLLOUT_LOGPROB_MB:-16}"
REF_LOGPROB_MB="${REF_LOGPROB_MB:-16}"
FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}"
ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))}"
INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))}"

LR="${LR:-1e-6}"
KL_COEF="${KL_COEF:-0.001}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.0003}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.0004}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:--1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE="${ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE:-false}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
ENTROPY_CHECKPOINTING="${ENTROPY_CHECKPOINTING:-true}"

ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-true}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-2000}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-0.1}"
OVERLONG_LOG="${OVERLONG_LOG:-false}"

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
PROJECT_NAME="${PROJECT_NAME:-qwen3vl-deepvision-gspo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-single_node_${TIMESTAMP}}"
INIT_LORA_PATH="${INIT_LORA_PATH:-${LORA_ADAPTER_PATH:-}}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
  cat >&2 <<EOF2
DeepVision VERL parquet not found.
Build it first with:
  $PYTHON_BIN Qwen/scripts/build_deepvision_verl_dataset.py
EOF2
  exit 1
fi

if [[ -n "$INIT_LORA_PATH" ]] && [ ! -d "$INIT_LORA_PATH" ]; then
  echo "Initial LoRA adapter path not found: $INIT_LORA_PATH" >&2
  exit 1
fi

if [[ "$RESUME_MODE" == "resume_path" ]]; then
  if [[ -z "$RESUME_FROM_PATH" ]]; then
    echo "RESUME_FROM_PATH must be set when RESUME_MODE=resume_path" >&2
    exit 1
  fi
  if [[ ! -d "$RESUME_FROM_PATH" ]]; then
    echo "VERL resume checkpoint not found: $RESUME_FROM_PATH" >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/training.log}"

sanitize_log_stream() {
  stdbuf -oL -eL perl -MIO::Handle -ne 'BEGIN { STDOUT->autoflush(1) } s/\e\[[0-9;]*[[:alpha:]]//g; s/\r/\n/g; print'
}

exec > >(sanitize_log_stream | tee -a "$LOG_FILE") 2>&1

export DEEPSEEK_OCR_ROOT="$REPO_ROOT"
export MODEL_PATH
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$MODEL_PATH}"
export TRAIN_FILE
export VAL_FILE
export OUTPUT_DIR
export NNODES
export GPUS_PER_NODE
export ROLLOUT_TP_SIZE
export SP_SIZE
export GEN_BATCH_SIZE
export TRAIN_BATCH_SIZE
export VAL_BATCH_SIZE
export PPO_MINI_BATCH_SIZE
export ACTOR_MICRO_BATCH_SIZE
export ROLLOUT_LOGPROB_MB
export REF_LOGPROB_MB
export FILTER_OVERLONG_PROMPTS_WORKERS
export MAX_PROMPT_LENGTH
export MAX_RESPONSE_LENGTH
export ROLLOUT_N
export ROLLOUT_MAX_MODEL_LEN
export ROLLOUT_MAX_BATCHED_TOKENS
export ACTOR_PPO_MAX_TOKEN_LEN
export INFER_PPO_MAX_TOKEN_LEN
export LR
export KL_COEF
export CLIP_RATIO_LOW
export CLIP_RATIO_HIGH
export TEMPERATURE
export TOP_P
export TOP_K
export GPU_MEMORY_UTILIZATION
export ROLLOUT_ENFORCE_EAGER
export ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE
export TOTAL_EPOCHS
export SAVE_FREQ
export TEST_FREQ
export USE_DYNAMIC_BSZ
export ENTROPY_CHECKPOINTING
export ENABLE_OVERLONG_BUFFER
export OVERLONG_BUFFER_LEN
export OVERLONG_PENALTY_FACTOR
export OVERLONG_LOG
export LORA_RANK
export LORA_ALPHA
export PROJECT_NAME
export EXPERIMENT_NAME
export RESUME_MODE
export QWEN3VL_RUNTIME_ENV_STAMP="$RUNTIME_ENV_STAMP"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/vllm_thinking_plugin:${PYTHONPATH:-}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-vllm_thinking}"
export VLLM_THINKING="${VLLM_THINKING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
if [[ -n "$INIT_LORA_PATH" ]]; then
  export VLLM_LORA_CHECKPOINT_PATH="${VLLM_LORA_CHECKPOINT_PATH:-$INIT_LORA_PATH}"
fi

CMD=(
  "$PYTHON_BIN" -u "$REPO_ROOT/Qwen/scripts/run_verl_ppo.py"
  --project-config "$PROJECT_BASE_CONFIG"
)

if [[ ! -f "$PROJECT_BASE_CONFIG" ]]; then
  echo "Project base config not found: $PROJECT_BASE_CONFIG" >&2
  exit 1
fi

if [[ -n "$PROJECT_CONFIG" ]]; then
  if [[ ! -f "$PROJECT_CONFIG" ]]; then
    echo "Project config not found: $PROJECT_CONFIG" >&2
    exit 1
  fi
  CMD+=(--project-config "$PROJECT_CONFIG")
fi

if [[ -n "$INIT_LORA_PATH" ]]; then
  CMD+=(actor_rollout_ref.model.lora_adapter_path="$INIT_LORA_PATH")
fi

if [[ "$RESUME_MODE" == "resume_path" ]]; then
  CMD+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi

if [[ -n "${RAY_ADDRESS:-}" ]]; then
  CMD+=(ray_kwargs.ray_init.address="$RAY_ADDRESS")
fi

CMD+=("$@")

printf 'Launching GSPO with output_dir=%s\n' "$OUTPUT_DIR"
printf 'Training log=%s\n' "$LOG_FILE"
printf 'Visible GPUs=%s cuda_visible_devices=%s nnodes=%s tp=%s train=%s val=%s\n' "$VISIBLE_GPUS" "${CUDA_VISIBLE_DEVICES:-<unset>}" "$NNODES" "$ROLLOUT_TP_SIZE" "$TRAIN_FILE" "$VAL_FILE"
printf 'Gen batch=%s train batch=%s rollout_n=%s loss_mode=%s reward_manager=%s\n' "$GEN_BATCH_SIZE" "$TRAIN_BATCH_SIZE" "$ROLLOUT_N" 'gspo' 'dapo_batch'
printf 'Rollout eager=%s gpu_mem_util=%s mm_preproc_cache_disabled=%s\n' "$ROLLOUT_ENFORCE_EAGER" "$GPU_MEMORY_UTILIZATION" "$ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE"
printf 'Filter workers=%s rollout_max_model_len=%s rollout_max_batched_tokens=%s\n' "$FILTER_OVERLONG_PROMPTS_WORKERS" "$ROLLOUT_MAX_MODEL_LEN" "$ROLLOUT_MAX_BATCHED_TOKENS"
printf 'Overlong buffer enabled=%s len=%s penalty_factor=%s log=%s\n' "$ENABLE_OVERLONG_BUFFER" "$OVERLONG_BUFFER_LEN" "$OVERLONG_PENALTY_FACTOR" "$OVERLONG_LOG"
printf 'Runtime env stamp=%s\n' "$RUNTIME_ENV_STAMP"
printf 'Ray no-set cuda visible devices=%s\n' "$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
printf 'Init LoRA=%s resume_mode=%s resume_from=%s\n' "${INIT_LORA_PATH:-<none>}" "$RESUME_MODE" "${RESUME_FROM_PATH:-<none>}"
printf 'VAE source path=%s\n' "${VLLM_LORA_CHECKPOINT_PATH:-<none>}"
printf 'VLLM model path=%s\n' "${VLLM_MODEL_PATH:-<none>}"
printf 'Project config=%s overlay=%s\n' "$PROJECT_BASE_CONFIG" "${PROJECT_CONFIG:-<none>}"

exec "${CMD[@]}"
