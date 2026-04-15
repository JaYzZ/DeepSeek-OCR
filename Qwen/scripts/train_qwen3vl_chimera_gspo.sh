#!/bin/bash
# Single-node-first VERL GSPO launcher for Chimera with vLLM rollout and rule-based reward.
#
# This script trains Qwen3VL-2B-Thinking on Chimera dataset using:
# - GSPO (Group Supervised Policy Optimization)
# - Rule-based reward function (answer correctness checking)
# - vLLM rollout for efficient generation
# - No verifier required - uses ground truth answers directly
#
# Usage:
#   bash Qwen/scripts/train_qwen3vl_chimera_gspo.sh
#   bash Qwen/scripts/train_qwen3vl_chimera_gspo.sh [overlay_config.yaml]
#   INIT_LORA_PATH=/path/to/sft/checkpoint bash Qwen/scripts/train_qwen3vl_chimera_gspo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="${ROOT_DIR:-/share/project/xiyan}"
source "$SCRIPT_DIR/qwen3vl_common.sh"
PYTHON_BIN="$(qwen3vl_require_python_bin "$REPO_ROOT")"

TIMESTAMP="${QWEN3VL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUNTIME_ENV_STAMP="${QWEN3VL_RUNTIME_ENV_STAMP:-$TIMESTAMP}"
VISIBLE_GPUS="$(qwen3vl_count_visible_gpus "$PYTHON_BIN")"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "$VISIBLE_GPUS" =~ ^[0-9]+$ ]] && [[ "$VISIBLE_GPUS" -gt 0 ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((VISIBLE_GPUS - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

PROJECT_BASE_CONFIG="${PROJECT_BASE_CONFIG:-$REPO_ROOT/Qwen/configs/rl/chimera_gspo.yaml}"
PROJECT_CONFIG="${PROJECT_CONFIG:-}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/huggingface/Qwen/Qwen3-VL-2B-Thinking}"
CHIMERA_DATA_DIR="${CHIMERA_DATA_DIR:-$PROJECT_ROOT/huggingface/TianHongZXY/CHIMERA/Qwen3.5-397B}"
CHIMERA_IMAGES_DIR="${CHIMERA_IMAGES_DIR:-$REPO_ROOT/Qwen/data/chimera_images}"
TEXT_ONLY="${TEXT_ONLY:-false}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/Qwen/data/chimera_verl}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/Qwen/checkpoints/qwen3vl-2b/verl/chimera_gspo/run_${TIMESTAMP}}"

NNODES="${NNODES:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-}"
SP_SIZE="${SP_SIZE:-}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-}"
ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-}"
ROLLOUT_LOGPROB_MB="${ROLLOUT_LOGPROB_MB:-}"
REF_LOGPROB_MB="${REF_LOGPROB_MB:-}"
FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-}"
ROLLOUT_N="${ROLLOUT_N:-}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-}"
ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-}"
INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-}"

# Auto-detect hardware facts only; YAML remains the source of truth for configurable settings.
if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$VISIBLE_GPUS"
fi

LR="${LR:-}"
KL_COEF="${KL_COEF:-}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-}"
ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE="${ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
TEST_FREQ="${TEST_FREQ:-}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-}"
ENTROPY_CHECKPOINTING="${ENTROPY_CHECKPOINTING:-}"

ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-false}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-2000}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-0.1}"
OVERLONG_LOG="${OVERLONG_LOG:-false}"

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
PROJECT_NAME="${PROJECT_NAME:-qwen3vl-chimera-gspo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-single_node_${TIMESTAMP}}"
INIT_LORA_PATH="${INIT_LORA_PATH:-${LORA_ADAPTER_PATH:-}}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
  cat >&2 <<EOF2
Chimera VERL dataset not found.
Build it first with:
  $PYTHON_BIN Qwen/data/build_chimera_verl_dataset.py --data-dir "$CHIMERA_DATA_DIR"

This creates proper train/val splits (default: 99%/1%) from the Chimera parquet files.
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

if [[ ! -f "$PROJECT_BASE_CONFIG" ]]; then
  echo "Project base config not found: $PROJECT_BASE_CONFIG" >&2
  exit 1
fi

if [[ -n "$PROJECT_CONFIG" ]] && [[ ! -f "$PROJECT_CONFIG" ]]; then
  echo "Project config not found: $PROJECT_CONFIG" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/training.log}"
RESOLVED_CONFIG_FILE="$OUTPUT_DIR/resolved_runtime_config.yaml"

qwen3vl_copy_if_present "$PROJECT_BASE_CONFIG" "$OUTPUT_DIR"
qwen3vl_copy_if_present "$PROJECT_CONFIG" "$OUTPUT_DIR"

exec > >(qwen3vl_sanitize_log_stream | tee -a "$LOG_FILE") 2>&1

export ROOT_DIR="$PROJECT_ROOT"
export MODEL_PATH
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$MODEL_PATH}"
export CHIMERA_IMAGES_DIR
export TEXT_ONLY
export TRAIN_FILE
export VAL_FILE
export OUTPUT_DIR

# Only export if set (non-empty) to avoid OmegaConf parse errors
[[ -n "$NNODES" ]] && export NNODES
[[ -n "$GPUS_PER_NODE" ]] && export GPUS_PER_NODE
[[ -n "$AGENT_NUM_WORKERS" ]] && export AGENT_NUM_WORKERS
[[ -n "$ROLLOUT_TP_SIZE" ]] && export ROLLOUT_TP_SIZE
[[ -n "$SP_SIZE" ]] && export SP_SIZE
[[ -n "$GEN_BATCH_SIZE" ]] && export GEN_BATCH_SIZE
[[ -n "$TRAIN_BATCH_SIZE" ]] && export TRAIN_BATCH_SIZE
[[ -n "$VAL_BATCH_SIZE" ]] && export VAL_BATCH_SIZE
[[ -n "$PPO_MINI_BATCH_SIZE" ]] && export PPO_MINI_BATCH_SIZE
[[ -n "$ACTOR_MICRO_BATCH_SIZE" ]] && export ACTOR_MICRO_BATCH_SIZE
[[ -n "$ROLLOUT_LOGPROB_MB" ]] && export ROLLOUT_LOGPROB_MB
[[ -n "$REF_LOGPROB_MB" ]] && export REF_LOGPROB_MB
[[ -n "$FILTER_OVERLONG_PROMPTS_WORKERS" ]] && export FILTER_OVERLONG_PROMPTS_WORKERS
[[ -n "$MAX_PROMPT_LENGTH" ]] && export MAX_PROMPT_LENGTH
[[ -n "$MAX_RESPONSE_LENGTH" ]] && export MAX_RESPONSE_LENGTH
[[ -n "$ROLLOUT_N" ]] && export ROLLOUT_N
[[ -n "$ROLLOUT_MAX_MODEL_LEN" ]] && export ROLLOUT_MAX_MODEL_LEN
[[ -n "$ROLLOUT_MAX_BATCHED_TOKENS" ]] && export ROLLOUT_MAX_BATCHED_TOKENS
[[ -n "$ACTOR_PPO_MAX_TOKEN_LEN" ]] && export ACTOR_PPO_MAX_TOKEN_LEN
[[ -n "$INFER_PPO_MAX_TOKEN_LEN" ]] && export INFER_PPO_MAX_TOKEN_LEN
[[ -n "$LR" ]] && export LR
[[ -n "$KL_COEF" ]] && export KL_COEF
[[ -n "$CLIP_RATIO_LOW" ]] && export CLIP_RATIO_LOW
[[ -n "$CLIP_RATIO_HIGH" ]] && export CLIP_RATIO_HIGH
[[ -n "$TEMPERATURE" ]] && export TEMPERATURE
[[ -n "$TOP_P" ]] && export TOP_P
[[ -n "$TOP_K" ]] && export TOP_K
[[ -n "$GPU_MEMORY_UTILIZATION" ]] && export GPU_MEMORY_UTILIZATION
[[ -n "$ROLLOUT_ENFORCE_EAGER" ]] && export ROLLOUT_ENFORCE_EAGER
[[ -n "$ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE" ]] && export ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE
[[ -n "$TOTAL_EPOCHS" ]] && export TOTAL_EPOCHS
[[ -n "$SAVE_FREQ" ]] && export SAVE_FREQ
[[ -n "$TEST_FREQ" ]] && export TEST_FREQ
[[ -n "$USE_DYNAMIC_BSZ" ]] && export USE_DYNAMIC_BSZ
[[ -n "$ENTROPY_CHECKPOINTING" ]] && export ENTROPY_CHECKPOINTING
[[ -n "$ENABLE_OVERLONG_BUFFER" ]] && export ENABLE_OVERLONG_BUFFER
[[ -n "$OVERLONG_BUFFER_LEN" ]] && export OVERLONG_BUFFER_LEN
[[ -n "$OVERLONG_PENALTY_FACTOR" ]] && export OVERLONG_PENALTY_FACTOR
[[ -n "$OVERLONG_LOG" ]] && export OVERLONG_LOG
[[ -n "$LORA_RANK" ]] && export LORA_RANK
[[ -n "$LORA_ALPHA" ]] && export LORA_ALPHA
[[ -n "$PROJECT_NAME" ]] && export PROJECT_NAME
[[ -n "$EXPERIMENT_NAME" ]] && export EXPERIMENT_NAME
[[ -n "$RESUME_MODE" ]] && export RESUME_MODE

export QWEN3VL_RUNTIME_ENV_STAMP="$RUNTIME_ENV_STAMP"
export VLLM_PLUGINS="${VLLM_PLUGINS:-vllm_thinking}"
export VLLM_THINKING="${VLLM_THINKING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
if [[ -n "$INIT_LORA_PATH" ]]; then
  export VLLM_LORA_CHECKPOINT_PATH="${VLLM_LORA_CHECKPOINT_PATH:-$INIT_LORA_PATH}"
fi

EFFECTIVE_CONFIG_LINES="$("$PYTHON_BIN" - "$PROJECT_BASE_CONFIG" "${PROJECT_CONFIG:-}" "$RESOLVED_CONFIG_FILE" <<'PY'
import sys
from omegaconf import OmegaConf

base_config = sys.argv[1]
overlay_config = sys.argv[2]
resolved_config_path = sys.argv[3]

if not OmegaConf.has_resolver("gpu_adapt"):
    OmegaConf.register_new_resolver(
        "gpu_adapt",
        lambda gpus, one, two, four, eight, fallback=None: (
            one
            if int(gpus) == 1
            else two
            if int(gpus) == 2
            else four
            if int(gpus) == 4
            else eight
            if int(gpus) == 8
            else (fallback if fallback is not None else eight)
        ),
    )

config = OmegaConf.load(base_config)
if overlay_config:
    config = OmegaConf.merge(config, OmegaConf.load(overlay_config))
OmegaConf.resolve(config)

with open(resolved_config_path, "w", encoding="utf-8") as fh:
    fh.write(OmegaConf.to_yaml(config, resolve=True))

keys = {
    "NNODES": "trainer.nnodes",
    "GPUS_PER_NODE": "trainer.n_gpus_per_node",
    "ROLLOUT_TP_SIZE": "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "GEN_BATCH_SIZE": "data.gen_batch_size",
    "TRAIN_BATCH_SIZE": "data.train_batch_size",
    "ROLLOUT_N": "actor_rollout_ref.rollout.n",
    "ROLLOUT_ENFORCE_EAGER": "actor_rollout_ref.rollout.enforce_eager",
    "GPU_MEMORY_UTILIZATION": "actor_rollout_ref.rollout.gpu_memory_utilization",
    "FILTER_OVERLONG_PROMPTS_WORKERS": "data.filter_overlong_prompts_workers",
    "ROLLOUT_MAX_MODEL_LEN": "actor_rollout_ref.rollout.max_model_len",
    "ROLLOUT_MAX_BATCHED_TOKENS": "actor_rollout_ref.rollout.max_num_batched_tokens",
    "ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE": "actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache",
    "ENABLE_CHUNKED_PREFILL": "actor_rollout_ref.rollout.enable_chunked_prefill",
}
for key, path in keys.items():
    print(f"{key}={OmegaConf.select(config, path)}")
PY
)"

while IFS='=' read -r key value; do
  case "$key" in
    NNODES) EFFECTIVE_NNODES="$value" ;;
    GPUS_PER_NODE) EFFECTIVE_GPUS_PER_NODE="$value" ;;
    ROLLOUT_TP_SIZE) EFFECTIVE_ROLLOUT_TP_SIZE="$value" ;;
    GEN_BATCH_SIZE) EFFECTIVE_GEN_BATCH_SIZE="$value" ;;
    TRAIN_BATCH_SIZE) EFFECTIVE_TRAIN_BATCH_SIZE="$value" ;;
    ROLLOUT_N) EFFECTIVE_ROLLOUT_N="$value" ;;
    ROLLOUT_ENFORCE_EAGER) EFFECTIVE_ROLLOUT_ENFORCE_EAGER="$value" ;;
    GPU_MEMORY_UTILIZATION) EFFECTIVE_GPU_MEMORY_UTILIZATION="$value" ;;
    FILTER_OVERLONG_PROMPTS_WORKERS) EFFECTIVE_FILTER_WORKERS="$value" ;;
    ROLLOUT_MAX_MODEL_LEN) EFFECTIVE_ROLLOUT_MAX_MODEL_LEN="$value" ;;
    ROLLOUT_MAX_BATCHED_TOKENS) EFFECTIVE_ROLLOUT_MAX_BATCHED_TOKENS="$value" ;;
    ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE) EFFECTIVE_ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE="$value" ;;
    ENABLE_CHUNKED_PREFILL) EFFECTIVE_ENABLE_CHUNKED_PREFILL="$value" ;;
  esac
done <<< "$EFFECTIVE_CONFIG_LINES"

if [[ "$EFFECTIVE_ENABLE_CHUNKED_PREFILL" == "True" || "$EFFECTIVE_ENABLE_CHUNKED_PREFILL" == "true" ]]; then
  if (( EFFECTIVE_ROLLOUT_MAX_BATCHED_TOKENS < EFFECTIVE_ROLLOUT_MAX_MODEL_LEN )); then
    echo "Invalid rollout config: enable_chunked_prefill=true requires max_num_batched_tokens >= max_model_len" >&2
    echo "Resolved max_num_batched_tokens=$EFFECTIVE_ROLLOUT_MAX_BATCHED_TOKENS max_model_len=$EFFECTIVE_ROLLOUT_MAX_MODEL_LEN" >&2
    echo "Fix the YAML or environment override. Resolved config snapshot: $RESOLVED_CONFIG_FILE" >&2
    exit 1
  fi
fi

{
  printf 'PROJECT_BASE_CONFIG=%s\n' "$PROJECT_BASE_CONFIG"
  printf 'PROJECT_CONFIG=%s\n' "${PROJECT_CONFIG:-<none>}"
  printf 'NNODES=%s\n' "$EFFECTIVE_NNODES"
  printf 'GPUS_PER_NODE=%s\n' "$EFFECTIVE_GPUS_PER_NODE"
  printf 'ROLLOUT_TP_SIZE=%s\n' "$EFFECTIVE_ROLLOUT_TP_SIZE"
  printf 'GEN_BATCH_SIZE=%s\n' "$EFFECTIVE_GEN_BATCH_SIZE"
  printf 'TRAIN_BATCH_SIZE=%s\n' "$EFFECTIVE_TRAIN_BATCH_SIZE"
  printf 'ROLLOUT_N=%s\n' "$EFFECTIVE_ROLLOUT_N"
  printf 'ROLLOUT_ENFORCE_EAGER=%s\n' "$EFFECTIVE_ROLLOUT_ENFORCE_EAGER"
  printf 'GPU_MEMORY_UTILIZATION=%s\n' "$EFFECTIVE_GPU_MEMORY_UTILIZATION"
  printf 'FILTER_OVERLONG_PROMPTS_WORKERS=%s\n' "$EFFECTIVE_FILTER_WORKERS"
  printf 'ROLLOUT_MAX_MODEL_LEN=%s\n' "$EFFECTIVE_ROLLOUT_MAX_MODEL_LEN"
  printf 'ROLLOUT_MAX_BATCHED_TOKENS=%s\n' "$EFFECTIVE_ROLLOUT_MAX_BATCHED_TOKENS"
  printf 'ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE=%s\n' "$EFFECTIVE_ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE"
  printf 'ENABLE_CHUNKED_PREFILL=%s\n' "$EFFECTIVE_ENABLE_CHUNKED_PREFILL"
} > "$OUTPUT_DIR/launcher_effective_env.snapshot.txt"

CMD=(
  "$PYTHON_BIN" -u "$REPO_ROOT/Qwen/verl/run_verl_ppo.py"
  --project-config "$PROJECT_BASE_CONFIG"
)

if [[ -n "$PROJECT_CONFIG" ]]; then
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

printf 'Launching Chimera GSPO with output_dir=%s\n' "$OUTPUT_DIR"
printf 'Training log=%s\n' "$LOG_FILE"
printf 'Visible GPUs=%s cuda_visible_devices=%s nnodes=%s tp=%s train=%s val=%s\n' "$VISIBLE_GPUS" "${CUDA_VISIBLE_DEVICES:-<unset>}" "$EFFECTIVE_NNODES" "$EFFECTIVE_ROLLOUT_TP_SIZE" "$TRAIN_FILE" "$VAL_FILE"
printf 'Gen batch=%s train batch=%s rollout_n=%s loss_mode=%s reward_manager=%s\n' "$EFFECTIVE_GEN_BATCH_SIZE" "$EFFECTIVE_TRAIN_BATCH_SIZE" "$EFFECTIVE_ROLLOUT_N" 'gspo' 'dapo_batch'
printf 'Rollout eager=%s gpu_mem_util=%s mm_preproc_cache_disabled=%s\n' "$EFFECTIVE_ROLLOUT_ENFORCE_EAGER" "$EFFECTIVE_GPU_MEMORY_UTILIZATION" "$EFFECTIVE_ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE"
printf 'Filter workers=%s rollout_max_model_len=%s rollout_max_batched_tokens=%s\n' "$EFFECTIVE_FILTER_WORKERS" "$EFFECTIVE_ROLLOUT_MAX_MODEL_LEN" "$EFFECTIVE_ROLLOUT_MAX_BATCHED_TOKENS"
printf 'Overlong buffer enabled=%s len=%s penalty_factor=%s log=%s\n' "$ENABLE_OVERLONG_BUFFER" "$OVERLONG_BUFFER_LEN" "$OVERLONG_PENALTY_FACTOR" "$OVERLONG_LOG"
printf 'Runtime env stamp=%s\n' "$RUNTIME_ENV_STAMP"
printf 'Ray no-set cuda visible devices=%s\n' "$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
printf 'Init LoRA=%s resume_mode=%s resume_from=%s\n' "${INIT_LORA_PATH:-<none>}" "$RESUME_MODE" "${RESUME_FROM_PATH:-<none>}"
printf 'VAE source path=%s\n' "${VLLM_LORA_CHECKPOINT_PATH:-<none>}"
printf 'VLLM model path=%s\n' "${VLLM_MODEL_PATH:-<none>}"
printf 'Project config=%s overlay=%s\n' "$PROJECT_BASE_CONFIG" "${PROJECT_CONFIG:-<none>}"
printf 'Saved config snapshots: %s %s %s %s\n' \
  "$OUTPUT_DIR/$(basename "$PROJECT_BASE_CONFIG")" \
  "${PROJECT_CONFIG:+$OUTPUT_DIR/$(basename "$PROJECT_CONFIG")}" \
  "$RESOLVED_CONFIG_FILE" \
  "$OUTPUT_DIR/launcher_effective_env.snapshot.txt"
printf '\nChimera Data: Using custom dataset class (no pre-conversion needed)\n'

exec "${CMD[@]}"
