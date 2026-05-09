#!/bin/bash
# Privileged OPSD launcher for Qwen3-VL backed by ms-swift GKD.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/qwen3vl_common.sh"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

PYTHON_BIN="$(qwen3vl_require_python_bin "$REPO_ROOT")"
TIMESTAMP="${QWEN3VL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
VISIBLE_GPUS="$(qwen3vl_count_visible_gpus "$PYTHON_BIN")"
RUNTIME_ENV_CONFIG="${QWEN3VL_RUNTIME_ENV_CONFIG:-$REPO_ROOT/Qwen/configs/qwen3vl_runtime_env.yaml}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "$VISIBLE_GPUS" =~ ^[0-9]+$ ]] && [[ "$VISIBLE_GPUS" -gt 0 ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((VISIBLE_GPUS - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

CONFIG_PATH="${1:-$REPO_ROOT/Qwen/configs/distillation/qwen3vl_opsd_swift.yaml}"
if [[ "${1:-}" != "" ]]; then
  shift || true
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$RUNTIME_ENV_CONFIG" ]]; then
  echo "Runtime env config not found: $RUNTIME_ENV_CONFIG" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "model.model_name_or_path" "$ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking")}"
if [[ ! "$MODEL_PATH" = /* ]]; then
  MODEL_PATH="$(qwen3vl_resolve_path "$REPO_ROOT" "$MODEL_PATH")"
fi
STUDENT_ADAPTER_PATH="${STUDENT_ADAPTER_PATH:-${INIT_LORA_PATH:-}}"
if [[ -n "$STUDENT_ADAPTER_PATH" && ! "$STUDENT_ADAPTER_PATH" = /* ]]; then
  STUDENT_ADAPTER_PATH="$(qwen3vl_resolve_path "$REPO_ROOT" "$STUDENT_ADAPTER_PATH")"
fi

DATASET_NAMES="${DATASET_NAMES:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.dataset_names" "r1ov_thinking,deepvision_thinking")}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/Qwen/data/opsd_manifest}"
TRAIN_FILE="${TRAIN_FILE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.opsd_manifest" "$MANIFEST_DIR/r1ov_deepvision_opsd.jsonl")}"
if [[ ! "$TRAIN_FILE" = /* ]]; then
  TRAIN_FILE="$REPO_ROOT/$TRAIN_FILE"
fi

OVERWRITE_OPSD_MANIFEST="${OVERWRITE_OPSD_MANIFEST:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.overwrite_manifest" "0")}"
if [[ ! -f "$TRAIN_FILE" || "$OVERWRITE_OPSD_MANIFEST" == "1" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/Qwen/data/build_qwen3vl_opsd_dataset.py" \
    --datasets "$DATASET_NAMES" \
    --output-jsonl "$TRAIN_FILE"
fi

BASE_OUTPUT_DIR="$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.output_dir" "$REPO_ROOT/Qwen/checkpoints/qwen3vl-2b/swift/opsd")"
if [[ ! "$BASE_OUTPUT_DIR" = /* ]]; then
  BASE_OUTPUT_DIR="$REPO_ROOT/$BASE_OUTPUT_DIR"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_OUTPUT_DIR/run_${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/training.log}"

qwen3vl_copy_if_present "$CONFIG_PATH" "$OUTPUT_DIR"
qwen3vl_copy_if_present "$RUNTIME_ENV_CONFIG" "$OUTPUT_DIR"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(qwen3vl_resolve_master_port "$PYTHON_BIN")}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$VISIBLE_GPUS}"
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  NPROC_PER_NODE=1
fi
RANK="${RANK:-0}"
LOCAL_RANK="${LOCAL_RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR
export MASTER_PORT
export NPROC_PER_NODE
export RANK
export LOCAL_RANK
export WORLD_SIZE

MODEL_TYPE="${MODEL_TYPE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.model_type" "qwen3_vl")}"
TEMPLATE="${TEMPLATE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.template" "qwen3_vl")}"
RLHF_TYPE="${RLHF_TYPE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.rlhf_type" "gkd")}"
USE_VLLM="${USE_VLLM:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.use_vllm" "1")}"
VLLM_MODE="${VLLM_MODE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.vllm_mode" "colocate")}"
LMBDA="${LMBDA:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.lmbda" "1.0")}"
SFT_ALPHA="${SFT_ALPHA:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.sft_alpha" "0.0")}"
BETA="${BETA:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.beta" "0.5")}"
TORCH_DTYPE="${TORCH_DTYPE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.torch_dtype" "bfloat16")}"
OPSD_IMAGE_MAX_PIXELS="${OPSD_IMAGE_MAX_PIXELS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.image_max_pixels" "262144")}"
MAX_PIXELS="${MAX_PIXELS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.max_pixels" "$OPSD_IMAGE_MAX_PIXELS")}"
MAX_LENGTH="${MAX_LENGTH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.max_length" "8192")}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "generation.max_new_tokens" "8192")}"
VIT_GRADIENT_CHECKPOINTING="${VIT_GRADIENT_CHECKPOINTING:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.vit_gradient_checkpointing" "0")}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.deepspeed" "")}"
NUM_GENERATIONS="${NUM_GENERATIONS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.num_generations" "1")}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.generation_batch_size" "")}"
LEARNING_RATE="${LEARNING_RATE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.learning_rate" "1e-6")}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.num_train_epochs" "1")}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.per_device_train_batch_size" "1")}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.gradient_accumulation_steps" "4")}"
LOGGING_STEPS="${LOGGING_STEPS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.logging_steps" "1")}"
SAVE_STEPS="${SAVE_STEPS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.save_steps" "10")}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.save_total_limit" "2")}"
WARMUP_RATIO="${WARMUP_RATIO:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.warmup_ratio" "0.0")}"
TEMPERATURE="${TEMPERATURE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "generation.temperature" "1.0")}"
TOP_P="${TOP_P:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "generation.top_p" "1.0")}"
TOP_K="${TOP_K:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "generation.top_k" "20")}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "rollout.gpu_memory_utilization" "0.55")}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "rollout.max_model_len" "20480")}"
LORA_RANK="${LORA_RANK:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "lora.rank" "8")}"
LORA_ALPHA="${LORA_ALPHA:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "lora.alpha" "16")}"
TARGET_MODULES="${TARGET_MODULES:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "lora.target_modules" "all-linear")}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "model.gradient_checkpointing" "1")}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.dataset_num_proc" "1")}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.dataloader_num_workers" "0")}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "performance.dataloader_persistent_workers" "0")}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "performance.dataloader_prefetch_factor" "2")}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.load_from_cache_file" "1")}"
OPSD_SYSTEM_PROMPT="${OPSD_SYSTEM_PROMPT:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "system_prompt" "")}"
OPSD_STUDENT_TEMPLATE="${OPSD_STUDENT_TEMPLATE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "prompts.student_user" "Answer the question.\n\nQuestion:\n{question_text}")}"
OPSD_TEACHER_TEMPLATE="${OPSD_TEACHER_TEMPLATE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "prompts.teacher_user" "")}"
OPSD_IMAGE_MIN_PIXELS="${OPSD_IMAGE_MIN_PIXELS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.image_min_pixels" "1024")}"
OPSD_IMAGE_MAX_PIXELS="${OPSD_IMAGE_MAX_PIXELS:-$OPSD_IMAGE_MAX_PIXELS}"
OPSD_TEACHER_MAX_PROMPT_LENGTH="${OPSD_TEACHER_MAX_PROMPT_LENGTH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "prompts.teacher_max_prompt_length" "4096")}"
OPSD_LOSS_TEMPERATURE="${OPSD_LOSS_TEMPERATURE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.temperature" "1.0")}"
OPSD_LOGPROB_CHUNK_SIZE="${OPSD_LOGPROB_CHUNK_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.sampled_logprob_chunk_size" "1024")}"
OPSD_TOKEN_LOSS_TYPE="${OPSD_TOKEN_LOSS_TYPE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.token_type" "jsd")}"
OPSD_LOSS_BETA="${OPSD_LOSS_BETA:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.beta" "$BETA")}"
OPSD_TOKEN_CLIP="${OPSD_TOKEN_CLIP:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.token_clip" "0.05")}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "teacher.model_name_or_path" "")}"
if [[ -n "$TEACHER_MODEL_PATH" && ! "$TEACHER_MODEL_PATH" = /* ]]; then
  TEACHER_MODEL_PATH="$(qwen3vl_resolve_path "$REPO_ROOT" "$TEACHER_MODEL_PATH")"
fi
TEACHER_ADAPTER_PATH="${TEACHER_ADAPTER_PATH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "teacher.adapter_name_or_path" "")}"
if [[ -n "$TEACHER_ADAPTER_PATH" && ! "$TEACHER_ADAPTER_PATH" = /* ]]; then
  TEACHER_ADAPTER_PATH="$(qwen3vl_resolve_path "$REPO_ROOT" "$TEACHER_ADAPTER_PATH")"
fi
OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "teacher.offload_model" "0")}"
OPSD_OFF_POLICY_MODE="${OPSD_OFF_POLICY_MODE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "mode.off_policy" "0")}"

case "${USE_VLLM,,}" in
  1|true|yes|on) USE_VLLM=true ;;
  *) USE_VLLM=false ;;
esac

case "${VIT_GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|on) VIT_GRADIENT_CHECKPOINTING=true ;;
  *) VIT_GRADIENT_CHECKPOINTING=false ;;
esac

case "${OFFLOAD_TEACHER_MODEL,,}" in
  1|true|yes|on) OFFLOAD_TEACHER_MODEL=true ;;
  *) OFFLOAD_TEACHER_MODEL=false ;;
esac

case "${OPSD_OFF_POLICY_MODE,,}" in
  1|true|yes|on) OPSD_OFF_POLICY_MODE=true ;;
  *) OPSD_OFF_POLICY_MODE=false ;;
esac

if [[ "$OPSD_OFF_POLICY_MODE" == true ]]; then
  if [[ -z "$TEACHER_MODEL_PATH" ]]; then
    echo "teacher.model_name_or_path must be set when mode.off_policy=true" >&2
    exit 1
  fi
fi

if ! [[ "$DATASET_NUM_PROC" =~ ^[0-9]+$ ]] || [[ "$DATASET_NUM_PROC" -lt 1 ]]; then
  DATASET_NUM_PROC=1
fi

if ! [[ "$DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$DATALOADER_NUM_WORKERS" -lt 0 ]]; then
  DATALOADER_NUM_WORKERS=0
fi

case "${GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|on) GRADIENT_CHECKPOINTING=true ;;
  *) GRADIENT_CHECKPOINTING=false ;;
esac

case "${DATALOADER_PERSISTENT_WORKERS,,}" in
  1|true|yes|on) DATALOADER_PERSISTENT_WORKERS=true ;;
  *) DATALOADER_PERSISTENT_WORKERS=false ;;
esac

case "${LOAD_FROM_CACHE_FILE,,}" in
  1|true|yes|on) LOAD_FROM_CACHE_FILE=true ;;
  *) LOAD_FROM_CACHE_FILE=false ;;
esac

GENERATION_BATCH_SIZE="$(
  qwen3vl_resolve_generation_batch_size \
    "$NUM_GENERATIONS" \
    "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    "$NPROC_PER_NODE" \
    "$GENERATION_BATCH_SIZE"
)"

exec > >(qwen3vl_sanitize_log_stream | tee -a "$LOG_FILE") 2>&1

echo "CONFIG_PATH=$CONFIG_PATH"
echo "MODEL_PATH=$MODEL_PATH"
echo "STUDENT_ADAPTER_PATH=${STUDENT_ADAPTER_PATH:-<none>}"
echo "TRAIN_FILE=$TRAIN_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "RLHF_TYPE=$RLHF_TYPE"
echo "USE_VLLM=$USE_VLLM"
echo "VLLM_MODE=$VLLM_MODE"
echo "OPSD_OFF_POLICY_MODE=$OPSD_OFF_POLICY_MODE"
echo "OPSD_TOKEN_LOSS_TYPE=$OPSD_TOKEN_LOSS_TYPE"
echo "OPSD_LOSS_BETA=$OPSD_LOSS_BETA"
echo "OPSD_TOKEN_CLIP=$OPSD_TOKEN_CLIP"
echo "TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-<none>}"
echo "TEACHER_ADAPTER_PATH=${TEACHER_ADAPTER_PATH:-<none>}"
echo "MAX_LENGTH=$MAX_LENGTH"
echo "MAX_COMPLETION_LENGTH=$MAX_COMPLETION_LENGTH"
echo "NUM_GENERATIONS=$NUM_GENERATIONS"
echo "GENERATION_BATCH_SIZE=$GENERATION_BATCH_SIZE"
echo "VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN"
echo "VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
echo "DATASET_NUM_PROC=$DATASET_NUM_PROC"
echo "DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS"
echo "LOAD_FROM_CACHE_FILE=$LOAD_FROM_CACHE_FILE"
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  echo "DEEPSPEED_CONFIG=$DEEPSPEED_CONFIG"
fi

export OPSD_SWIFT_DATASET_PATH="$TRAIN_FILE"
export OPSD_STUDENT_ADAPTER_PATH="$STUDENT_ADAPTER_PATH"
export OPSD_SYSTEM_PROMPT
export OPSD_STUDENT_TEMPLATE
export OPSD_TEACHER_TEMPLATE
export OPSD_IMAGE_MIN_PIXELS
export OPSD_IMAGE_MAX_PIXELS
export OPSD_TEACHER_MAX_PROMPT_LENGTH
export OPSD_LOSS_TEMPERATURE
export OPSD_LOGPROB_CHUNK_SIZE
export OPSD_TOKEN_LOSS_TYPE
export OPSD_LOSS_BETA
export OPSD_TOKEN_CLIP
export OPSD_OFF_POLICY_MODE
export QWEN3VL_RUNTIME_ENV_CONFIG="$RUNTIME_ENV_CONFIG"
qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_LATENT_SUPERVISION" "latent_supervision" "1"
qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_LATENT_TOKEN_ID" "latent_token_id" "151669"
qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_THINKING_START_ID" "thinking_start_id" "151667"
qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_THINKING_END_ID" "thinking_end_id" "151668"
qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_THINKING_SEP_ID" "thinking_sep_id" "151670"
qwen3vl_export_env_from_main_then_runtime "$PYTHON_BIN" "$CONFIG_PATH" "generation.max_new_tokens" "$RUNTIME_ENV_CONFIG" "max_new_tokens" "QWEN3VL_MAX_NEW_TOKENS" "$MAX_COMPLETION_LENGTH"
qwen3vl_export_env_from_main_then_runtime "$PYTHON_BIN" "$CONFIG_PATH" "generation.min_continuous_steps" "$RUNTIME_ENV_CONFIG" "min_continuous_steps" "MIN_CONTINUOUS_STEPS" "0"
qwen3vl_export_max_continuous_steps_from_yaml "$PYTHON_BIN" "$CONFIG_PATH" "$RUNTIME_ENV_CONFIG"
qwen3vl_export_env_from_main_then_runtime "$PYTHON_BIN" "$CONFIG_PATH" "vae.intermediate_size" "$RUNTIME_ENV_CONFIG" "vae_intermediate_size" "QWEN3VL_VAE_INTERMEDIATE_SIZE" "512"
export QWEN3VL_LOSS_TYPE="${QWEN3VL_LOSS_TYPE:-none}"
export QWEN3VL_VAE_TRAINABLE="${QWEN3VL_VAE_TRAINABLE:-1}"
export QWEN3VL_HIDDEN_STATES_HOOK="${QWEN3VL_HIDDEN_STATES_HOOK:-0}"
if [[ -n "$STUDENT_ADAPTER_PATH" && -f "$STUDENT_ADAPTER_PATH/vae.safetensors" ]]; then
  export QWEN3VL_VAE_ENABLED="${QWEN3VL_VAE_ENABLED:-1}"
  export QWEN3VL_VAE_CHECKPOINT_PATH="${QWEN3VL_VAE_CHECKPOINT_PATH:-$STUDENT_ADAPTER_PATH/vae.safetensors}"
  export VLLM_LORA_CHECKPOINT_PATH="${VLLM_LORA_CHECKPOINT_PATH:-$STUDENT_ADAPTER_PATH}"
else
  export QWEN3VL_VAE_ENABLED="${QWEN3VL_VAE_ENABLED:-0}"
fi
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$MODEL_PATH}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-vllm_thinking}"
export VLLM_THINKING="${VLLM_THINKING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

SWIFT_CMD=(
  "$PYTHON_BIN" -m swift.cli.main rlhf
  --model "$MODEL_PATH"
  --model_type "$MODEL_TYPE"
  --template "$TEMPLATE"
  --rlhf_type "$RLHF_TYPE"
  --dataset qwen3vl_opsd_swift
  --custom_register_path "$REPO_ROOT/Qwen/swift/opsd_dataset.py"
  --output_dir "$OUTPUT_DIR"
  --do_train true
  --split_dataset_ratio 0
  --dataset_num_proc "$DATASET_NUM_PROC"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --dataloader_persistent_workers "$DATALOADER_PERSISTENT_WORKERS"
  --remove_unused_columns false
  --strict false
  --torch_dtype "$TORCH_DTYPE"
  --max_pixels "$MAX_PIXELS"
  --max_length "$MAX_LENGTH"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --num_generations "$NUM_GENERATIONS"
  --generation_batch_size "$GENERATION_BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --logging_steps "$LOGGING_STEPS"
  --save_steps "$SAVE_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --warmup_ratio "$WARMUP_RATIO"
  --check_model false
  --ddp_find_unused_parameters false
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
  --vit_gradient_checkpointing "$VIT_GRADIENT_CHECKPOINTING"
  --freeze_vit true
  --tuner_type lora
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --target_modules "$TARGET_MODULES"
  --beta "$BETA"
  --lmbda "$LMBDA"
  --sft_alpha "$SFT_ALPHA"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --top_k "$TOP_K"
  --use_vllm "$USE_VLLM"
  --load_from_cache_file "$LOAD_FROM_CACHE_FILE"
  --dataset_shuffle true
  --report_to none
  --model_kwargs "{\"TOKENIZERS_PARALLELISM\":\"true\"}"
)

if [[ "$DATALOADER_NUM_WORKERS" -gt 0 ]]; then
  SWIFT_CMD+=(--dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR")
fi

if [[ -n "$TEACHER_MODEL_PATH" ]]; then
  SWIFT_CMD+=(--teacher_model "$TEACHER_MODEL_PATH")
fi

if [[ -n "$TEACHER_ADAPTER_PATH" ]]; then
  SWIFT_CMD+=(--teacher_adapters "$TEACHER_ADAPTER_PATH")
fi

if [[ -n "$STUDENT_ADAPTER_PATH" ]]; then
  SWIFT_CMD+=(--adapters "$STUDENT_ADAPTER_PATH")
fi

if [[ "$USE_VLLM" == true ]]; then
  SWIFT_CMD+=(
    --vllm_mode "$VLLM_MODE"
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
  )
fi

if [[ "$OFFLOAD_TEACHER_MODEL" == true ]]; then
  SWIFT_CMD+=(--offload_teacher_model true)
fi

if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  SWIFT_CMD+=(--deepspeed "$DEEPSPEED_CONFIG")
fi

exec "${SWIFT_CMD[@]}" \
  "$@"
