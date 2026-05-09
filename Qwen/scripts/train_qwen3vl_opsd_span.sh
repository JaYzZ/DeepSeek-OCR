#!/bin/bash
# Unified ms-swift OPSD-span launcher for r1ov cold-start warmup and main OPSD training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/qwen3vl_common.sh"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

PYTHON_BIN="$(qwen3vl_require_python_bin "$REPO_ROOT")"
TIMESTAMP="${QWEN3VL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
VISIBLE_GPUS="$(qwen3vl_count_visible_gpus "$PYTHON_BIN")"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(qwen3vl_resolve_master_port "$PYTHON_BIN")}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$VISIBLE_GPUS}"
RANK="${RANK:-0}"
LOCAL_RANK="${LOCAL_RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR
export MASTER_PORT
export NPROC_PER_NODE
export RANK
export LOCAL_RANK
export WORLD_SIZE

CONFIG_PATH="$REPO_ROOT/Qwen/configs/distillation/qwen3vl_opsd_span.yaml"
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  CONFIG_PATH="$1"
  shift || true
fi

FORWARD_ARGS=()
CLI_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      CLI_STAGE="${2:-}"
      shift 2
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

yaml_value() {
  local key="$1"
  local default_value="$2"
  qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "$key" "$default_value"
}

STAGE="${CLI_STAGE:-${OPSD_SPAN_STAGE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "mode.stage" "all")}}"
MODEL_PATH="${MODEL_PATH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "model.model_name_or_path" "$ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking")}"
if [[ ! "$MODEL_PATH" = /* ]]; then
  MODEL_PATH="$(qwen3vl_resolve_path "$REPO_ROOT" "$MODEL_PATH")"
fi
TOKENIZER_SOURCE_PATH="$MODEL_PATH"

MANIFEST_PATH="${OPSD_SPAN_DATASET_PATH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.opsd_span_manifest" "Qwen/data/opsd_manifest/r1ov_opsd_span.jsonl")}"
if [[ ! "$MANIFEST_PATH" = /* ]]; then
  MANIFEST_PATH="$REPO_ROOT/$MANIFEST_PATH"
fi

DATASET_NAME="${DATASET_NAME:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.dataset_name" "r1ov_thinking")}"
OVERWRITE_MANIFEST="${OVERWRITE_OPSD_SPAN_MANIFEST:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.overwrite_manifest" "0")}"
if [[ ! -f "$MANIFEST_PATH" || "$OVERWRITE_MANIFEST" == "1" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/Qwen/data/build_qwen3vl_opsd_span_dataset.py" \
    --dataset "$DATASET_NAME" \
    --output-jsonl "$MANIFEST_PATH"
fi

BASE_OUTPUT_DIR="$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.output_dir" "$REPO_ROOT/Qwen/checkpoints/qwen3vl-2b/swift/opsd_span")"
if [[ ! "$BASE_OUTPUT_DIR" = /* ]]; then
  BASE_OUTPUT_DIR="$REPO_ROOT/$BASE_OUTPUT_DIR"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_OUTPUT_DIR/run_${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/training.log}"
TOKENIZER_DIR="$OUTPUT_DIR/tokenizer"
TOKENIZER_DIR="$(qwen3vl_prepare_thinking_tokenizer_dir "$PYTHON_BIN" "$TOKENIZER_SOURCE_PATH" "$TOKENIZER_DIR")"
qwen3vl_copy_if_present "$CONFIG_PATH" "$OUTPUT_DIR"

MODEL_TYPE="${MODEL_TYPE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "swift.model_type" "qwen3_vl")}"
TEMPLATE="$(yaml_value "swift.template" "qwen3_vl")"
RLHF_TYPE="$(yaml_value "swift.rlhf_type" "gkd")"
USE_VLLM="$(yaml_value "swift.use_vllm" "1")"
VLLM_MODE="$(yaml_value "swift.vllm_mode" "colocate")"
TORCH_DTYPE="$(yaml_value "swift.torch_dtype" "bfloat16")"
if [[ "$TORCH_DTYPE" == "bfloat16" ]]; then
  TORCH_DTYPE="${QWEN3VL_SMOKE_TORCH_DTYPE:-$TORCH_DTYPE}"
fi
MAX_PIXELS="$(yaml_value "swift.max_pixels" "262144")"
MAX_LENGTH="$(yaml_value "swift.max_length" "8192")"
MAX_COMPLETION_LENGTH="$(yaml_value "generation.max_new_tokens" "8192")"
VIT_GRADIENT_CHECKPOINTING="$(yaml_value "swift.vit_gradient_checkpointing" "0")"
NUM_GENERATIONS="$(yaml_value "swift.num_generations" "1")"
GENERATION_BATCH_SIZE="$(yaml_value "swift.generation_batch_size" "")"
LEARNING_RATE="$(yaml_value "training.learning_rate" "1e-6")"
NUM_TRAIN_EPOCHS="$(yaml_value "training.num_train_epochs" "1")"
PER_DEVICE_TRAIN_BATCH_SIZE="$(yaml_value "training.per_device_train_batch_size" "1")"
GRADIENT_ACCUMULATION_STEPS="$(yaml_value "training.gradient_accumulation_steps" "4")"
LOGGING_STEPS="$(yaml_value "training.logging_steps" "1")"
SAVE_STEPS="$(yaml_value "training.save_steps" "10")"
SAVE_TOTAL_LIMIT="$(yaml_value "training.save_total_limit" "2")"
WARMUP_RATIO="$(yaml_value "training.warmup_ratio" "0.0")"
TEMPERATURE="$(yaml_value "generation.temperature" "1.0")"
TOP_P="$(yaml_value "generation.top_p" "1.0")"
TOP_K="$(yaml_value "generation.top_k" "20")"
OPSD_DELTA_MEMORY_ENABLED="$(yaml_value "loss.delta_memory_enabled" "1")"
OPSD_DELTA_MEMORY_GAMMA="$(yaml_value "loss.delta_memory_gamma" "0.5")"
OPSD_DELTA_MEMORY_TARGET_WEIGHT="$(yaml_value "loss.delta_memory_target_weight" "0.25")"
OPSD_SPAN_REPLAY_MODE="$(yaml_value "replay.mode" "normal")"
OPSD_SPAN_REPLAY_DELTA_MODE="$(yaml_value "replay.delta_mode" "follow_global")"
OPSD_SPAN_REPLAY_LEGIT_LATENT_COUNT_MAX="$(yaml_value "replay.legit_latent_count_max" "12")"
VLLM_GPU_MEMORY_UTILIZATION="$(yaml_value "rollout.gpu_memory_utilization" "0.55")"
VLLM_MAX_MODEL_LEN="$(yaml_value "rollout.max_model_len" "20480")"
LORA_RANK="$(yaml_value "lora.rank" "8")"
LORA_ALPHA="$(yaml_value "lora.alpha" "16")"
TARGET_MODULES="$(yaml_value "lora.target_modules" "all-linear")"
GRADIENT_CHECKPOINTING="$(yaml_value "model.gradient_checkpointing" "1")"
DATASET_NUM_PROC="$(yaml_value "data.dataset_num_proc" "1")"
DATALOADER_NUM_WORKERS="$(yaml_value "data.dataloader_num_workers" "0")"
DATALOADER_PERSISTENT_WORKERS="$(yaml_value "performance.dataloader_persistent_workers" "0")"
DATALOADER_PREFETCH_FACTOR="$(yaml_value "performance.dataloader_prefetch_factor" "2")"
PYTORCH_ALLOC_CONF_VALUE="$(yaml_value "performance.pytorch_alloc_conf" "")"
LOAD_FROM_CACHE_FILE="$(yaml_value "data.load_from_cache_file" "1")"
OPSD_SYSTEM_PROMPT="$(yaml_value "system_prompt" "")"
OPSD_STUDENT_TEMPLATE="$(yaml_value "prompts.student_user" $'Answer the question.\n\nQuestion:\n{question_text}')"
OPSD_TEACHER_TEMPLATE="$(yaml_value "prompts.teacher_user" "")"
TEACHER_MODEL_PATH="$(yaml_value "teacher.model_name_or_path" "")"
if [[ -n "$TEACHER_MODEL_PATH" && ! "$TEACHER_MODEL_PATH" = /* ]]; then
  TEACHER_MODEL_PATH="$(qwen3vl_resolve_path "$REPO_ROOT" "$TEACHER_MODEL_PATH")"
fi
INITIAL_STUDENT_ADAPTER_PATH="${STUDENT_ADAPTER_PATH:-}"
INITIAL_OPSD_RLSD_ENABLED="${OPSD_RLSD_ENABLED:-$(yaml_value "mode.rlsd_enabled" "0")}"

if ! [[ "$DATASET_NUM_PROC" =~ ^[0-9]+$ ]] || [[ "$DATASET_NUM_PROC" -lt 1 ]]; then
  DATASET_NUM_PROC=1
fi

if ! [[ "$DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$DATALOADER_NUM_WORKERS" -lt 0 ]]; then
  DATALOADER_NUM_WORKERS=0
fi

case "${USE_VLLM,,}" in
  1|true|yes|on) USE_VLLM=true ;;
  *) USE_VLLM=false ;;
esac

case "${GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|on) GRADIENT_CHECKPOINTING=true ;;
  *) GRADIENT_CHECKPOINTING=false ;;
esac

case "${VIT_GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|on) VIT_GRADIENT_CHECKPOINTING=true ;;
  *) VIT_GRADIENT_CHECKPOINTING=false ;;
esac

case "${DATALOADER_PERSISTENT_WORKERS,,}" in
  1|true|yes|on) DATALOADER_PERSISTENT_WORKERS=true ;;
  *) DATALOADER_PERSISTENT_WORKERS=false ;;
esac

case "${LOAD_FROM_CACHE_FILE,,}" in
  1|true|yes|on) LOAD_FROM_CACHE_FILE=true ;;
  *) LOAD_FROM_CACHE_FILE=false ;;
esac

export OPSD_SPAN_DATASET_PATH="$MANIFEST_PATH"
export OPSD_SYSTEM_PROMPT
export OPSD_STUDENT_TEMPLATE
export OPSD_TEACHER_TEMPLATE
export OPSD_SPAN_TOKENIZER_PATH="$TOKENIZER_DIR"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export OPSD_DELTA_MEMORY_ENABLED
export OPSD_DELTA_MEMORY_GAMMA
export OPSD_DELTA_MEMORY_TARGET_WEIGHT
export OPSD_SPAN_REPLAY_MODE
export OPSD_SPAN_REPLAY_DELTA_MODE
export OPSD_SPAN_REPLAY_LEGIT_LATENT_COUNT_MAX
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$MODEL_PATH}"
export VLLM_TOKENIZER_PATH="${VLLM_TOKENIZER_PATH:-$TOKENIZER_DIR}"
export VLLM_THINKING=0
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-vllm_thinking}"
if [[ -n "$PYTORCH_ALLOC_CONF_VALUE" ]]; then
  export PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF_VALUE"
fi

exec > >(qwen3vl_sanitize_log_stream | tee -a "$LOG_FILE") 2>&1

echo "CONFIG_PATH=$CONFIG_PATH"
echo "MODEL_PATH=$MODEL_PATH"
echo "TOKENIZER_DIR=$TOKENIZER_DIR"
echo "MANIFEST_PATH=$MANIFEST_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "OPSD_SPAN_STAGE=$STAGE"
echo "ROLLOUT_MODE=discrete"
echo "VLLM_THINKING=$VLLM_THINKING"
echo "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
echo "DATASET_NUM_PROC=$DATASET_NUM_PROC"
echo "DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS"
echo "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-}"
echo "LOAD_FROM_CACHE_FILE=$LOAD_FROM_CACHE_FILE"
echo "OPSD_DELTA_MEMORY_ENABLED=$OPSD_DELTA_MEMORY_ENABLED"
echo "OPSD_DELTA_MEMORY_GAMMA=$OPSD_DELTA_MEMORY_GAMMA"
echo "OPSD_DELTA_MEMORY_TARGET_WEIGHT=$OPSD_DELTA_MEMORY_TARGET_WEIGHT"
echo "OPSD_SPAN_REPLAY_MODE=$OPSD_SPAN_REPLAY_MODE"
echo "OPSD_SPAN_REPLAY_DELTA_MODE=$OPSD_SPAN_REPLAY_DELTA_MODE"
echo "OPSD_SPAN_REPLAY_LEGIT_LATENT_COUNT_MAX=$OPSD_SPAN_REPLAY_LEGIT_LATENT_COUNT_MAX"

resolve_swift_adapter_path() {
  local candidate_path="${1:-}"
  if [[ -z "$candidate_path" ]]; then
    return 1
  fi

  if [[ -f "$candidate_path/adapter_config.json" ]]; then
    printf '%s\n' "$candidate_path"
    return 0
  fi

  local latest_checkpoint_rel=""
  latest_checkpoint_rel="$(qwen3vl_list_checkpoint_dirs "$PYTHON_BIN" "$candidate_path" | tail -n 1 || true)"
  if [[ -n "$latest_checkpoint_rel" && -f "$candidate_path/$latest_checkpoint_rel/adapter_config.json" ]]; then
    printf '%s\n' "$candidate_path/$latest_checkpoint_rel"
    return 0
  fi

  if [[ -d "$candidate_path" ]]; then
    while IFS= read -r subdir; do
      [[ -z "$subdir" ]] && continue
      latest_checkpoint_rel="$(qwen3vl_list_checkpoint_dirs "$PYTHON_BIN" "$subdir" | tail -n 1 || true)"
      if [[ -n "$latest_checkpoint_rel" && -f "$subdir/$latest_checkpoint_rel/adapter_config.json" ]]; then
        printf '%s\n' "$subdir/$latest_checkpoint_rel"
        return 0
      fi
    done < <(find "$candidate_path" -mindepth 1 -maxdepth 1 -type d | sort)
  fi

  return 1
}

resolve_stage_seed_adapter_path() {
  local explicit_path="${1:-}"
  if [[ -n "$explicit_path" ]]; then
    resolve_swift_adapter_path "$explicit_path"
    return $?
  fi

  local candidate=""
  for candidate in "$OUTPUT_DIR/gspo" "$OUTPUT_DIR/warmup"; do
    if [[ -d "$candidate" ]]; then
      resolve_swift_adapter_path "$candidate"
      return $?
    fi
  done

  return 1
}

run_warmup() {
  local warmup_dir="$OUTPUT_DIR/warmup"
  mkdir -p "$warmup_dir"
  local warmup_lr="$(yaml_value "warmup.learning_rate" "5e-6")"
  local warmup_epochs="$(yaml_value "warmup.num_train_epochs" "1")"
  local warmup_max_steps="$(yaml_value "warmup.max_steps" "0")"
  local warmup_save_steps="$(yaml_value "warmup.save_steps" "50")"
  local warmup_per_device_train_batch_size="$(yaml_value "warmup.per_device_train_batch_size" "$PER_DEVICE_TRAIN_BATCH_SIZE")"
  local warmup_gradient_accumulation_steps="$(yaml_value "warmup.gradient_accumulation_steps" "$GRADIENT_ACCUMULATION_STEPS")"

  export OPSD_SPAN_STAGE=warmup
  local cmd=(
    "$PYTHON_BIN" -m swift.cli.main sft
    --model "$MODEL_PATH"
    --model_type "$MODEL_TYPE"
    --template "$TEMPLATE"
    --new_special_tokens "<latent>" "<think_sep>"
    --dataset qwen3vl_opsd_span
    --custom_register_path "$REPO_ROOT/Qwen/swift/opsd_span.py"
    --output_dir "$warmup_dir"
    --do_train true
    --split_dataset_ratio 0
    --dataset_num_proc "$DATASET_NUM_PROC"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --dataloader_persistent_workers "$DATALOADER_PERSISTENT_WORKERS"
    --lazy_tokenize false
    --remove_unused_columns false
    --strict false
    --torch_dtype "$TORCH_DTYPE"
    --max_pixels "$MAX_PIXELS"
    --max_length "$MAX_LENGTH"
    --learning_rate "$warmup_lr"
    --num_train_epochs "$warmup_epochs"
    --per_device_train_batch_size "$warmup_per_device_train_batch_size"
    --gradient_accumulation_steps "$warmup_gradient_accumulation_steps"
    --logging_steps "$LOGGING_STEPS"
    --save_steps "$warmup_save_steps"
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
    --load_from_cache_file "$LOAD_FROM_CACHE_FILE"
    --dataset_shuffle true
    --report_to none
    --model_kwargs "{\"TOKENIZERS_PARALLELISM\":\"true\"}"
  )
  if [[ "$DATALOADER_NUM_WORKERS" -gt 0 ]]; then
    cmd+=(--dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR")
  fi
  if [[ "$warmup_max_steps" =~ ^[0-9]+$ ]] && [[ "$warmup_max_steps" -gt 0 ]]; then
    cmd+=(--max_steps "$warmup_max_steps")
  fi
  "${cmd[@]}" "${FORWARD_ARGS[@]}"
}

run_rl_stage() {
  local stage_name="$1"
  local stage_dir="$OUTPUT_DIR/$stage_name"
  mkdir -p "$stage_dir"

  local stage_rlhf_type="$RLHF_TYPE"
  local stage_learning_rate="$LEARNING_RATE"
  local stage_num_train_epochs="$NUM_TRAIN_EPOCHS"
  local stage_per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE"
  local stage_gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"
  local stage_save_steps="$SAVE_STEPS"
  local stage_max_steps="0"
  local stage_num_generations="$NUM_GENERATIONS"
  local stage_generation_batch_size="$GENERATION_BATCH_SIZE"
  local stage_generation_batch_size_min=""
  local stage_beta="0.0"
  local stage_lmbda="1.0"
  local stage_sft_alpha="0.0"
  local stage_loss_type=""
  local stage_advantage_estimator=""
  local stage_importance_sampling_level=""
  local stage_enable_reward_funcs=false
  local stage_enable_rlsd=false
  local stage_adapter_seed=""
  local stage_teacher_model_path=""

  if [[ "$stage_name" == "gspo" ]]; then
    stage_rlhf_type="$(yaml_value "gspo.rlhf_type" "grpo")"
    stage_learning_rate="$(yaml_value "gspo.learning_rate" "$LEARNING_RATE")"
    stage_num_train_epochs="$(yaml_value "gspo.num_train_epochs" "1")"
    stage_max_steps="$(yaml_value "gspo.max_steps" "0")"
    stage_per_device_train_batch_size="$(yaml_value "gspo.per_device_train_batch_size" "$PER_DEVICE_TRAIN_BATCH_SIZE")"
    stage_gradient_accumulation_steps="$(yaml_value "gspo.gradient_accumulation_steps" "$GRADIENT_ACCUMULATION_STEPS")"
    stage_save_steps="$(yaml_value "gspo.save_steps" "$SAVE_STEPS")"
    stage_num_generations="$(yaml_value "gspo.num_generations" "1")"
    stage_generation_batch_size="$(yaml_value "gspo.generation_batch_size" "$GENERATION_BATCH_SIZE")"
    stage_generation_batch_size_min="$stage_generation_batch_size"
    stage_beta="$(yaml_value "gspo.beta" "0.0")"
    stage_lmbda="$(yaml_value "gspo.lmbda" "1.0")"
    stage_sft_alpha="$(yaml_value "gspo.sft_alpha" "0.0")"
    stage_loss_type="$(yaml_value "gspo.loss_type" "grpo")"
    stage_advantage_estimator="$(yaml_value "gspo.advantage_estimator" "grpo")"
    stage_importance_sampling_level="$(yaml_value "gspo.importance_sampling_level" "sequence_token")"
    stage_enable_reward_funcs=true
    stage_enable_rlsd=false
    stage_adapter_seed="$INITIAL_STUDENT_ADAPTER_PATH"
    if [[ -z "$stage_adapter_seed" && -d "$OUTPUT_DIR/warmup" ]]; then
      stage_adapter_seed="$OUTPUT_DIR/warmup"
    fi
  else
    local off_policy_mode="$(yaml_value "mode.off_policy" "0")"
    case "${off_policy_mode,,}" in
      1|true|yes|on) off_policy_mode=true ;;
      *) off_policy_mode=false ;;
    esac
    if [[ "$RLHF_TYPE" == "grpo" ]]; then
      stage_enable_reward_funcs=true
      stage_loss_type="grpo"
      stage_advantage_estimator="grpo"
      if [[ "$INITIAL_OPSD_RLSD_ENABLED" =~ ^(1|true|yes|on)$ ]]; then
        stage_enable_rlsd=true
      fi
    fi
    if [[ "$off_policy_mode" == true && -n "$TEACHER_MODEL_PATH" ]]; then
      stage_teacher_model_path="$TEACHER_MODEL_PATH"
    fi
    if [[ -d "$OUTPUT_DIR/gspo" ]]; then
      stage_adapter_seed="$OUTPUT_DIR/gspo"
    elif [[ -d "$OUTPUT_DIR/warmup" ]]; then
      stage_adapter_seed="$OUTPUT_DIR/warmup"
    else
      stage_adapter_seed="$INITIAL_STUDENT_ADAPTER_PATH"
    fi
  fi

  if [[ -z "$stage_generation_batch_size_min" ]]; then
    stage_generation_batch_size_min="$stage_generation_batch_size"
  fi
  stage_generation_batch_size="$(
    qwen3vl_resolve_generation_batch_size \
      "$stage_num_generations" \
      "$stage_per_device_train_batch_size" \
      "$NPROC_PER_NODE" \
      "$stage_generation_batch_size_min"
  )"

  local student_adapter_path=""
  if [[ -n "$stage_adapter_seed" || -d "$OUTPUT_DIR/gspo" || -d "$OUTPUT_DIR/warmup" ]]; then
    student_adapter_path="$(resolve_stage_seed_adapter_path "$stage_adapter_seed" || true)"
  fi
  if [[ -n "$stage_adapter_seed" && -z "$student_adapter_path" ]]; then
    echo "Failed to resolve a valid Swift adapter checkpoint from: $stage_adapter_seed" >&2
    exit 1
  fi

  export OPSD_SPAN_STAGE="$stage_name"
  export OPSD_RLSD_ENABLED="$([[ "$stage_enable_rlsd" == true ]] && echo 1 || echo 0)"
  echo "STAGE_NAME=$stage_name"
  echo "STAGE_NUM_GENERATIONS=$stage_num_generations"
  echo "STAGE_PER_DEVICE_TRAIN_BATCH_SIZE=$stage_per_device_train_batch_size"
  echo "STAGE_GRADIENT_ACCUMULATION_STEPS=$stage_gradient_accumulation_steps"
  echo "STAGE_GENERATION_BATCH_SIZE=$stage_generation_batch_size"
  echo "STAGE_NUM_PROCESSES=$NPROC_PER_NODE"
  local cmd=(
    "$PYTHON_BIN" -m swift.cli.main rlhf
    --model "$MODEL_PATH"
    --model_type "$MODEL_TYPE"
    --template "$TEMPLATE"
    --new_special_tokens "<latent>" "<think_sep>"
    --rlhf_type "$stage_rlhf_type"
    --dataset qwen3vl_opsd_span
    --custom_register_path "$REPO_ROOT/Qwen/swift/opsd_span.py"
    --output_dir "$stage_dir"
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
    --num_generations "$stage_num_generations"
    --generation_batch_size "$stage_generation_batch_size"
    --learning_rate "$stage_learning_rate"
    --num_train_epochs "$stage_num_train_epochs"
    --per_device_train_batch_size "$stage_per_device_train_batch_size"
    --gradient_accumulation_steps "$stage_gradient_accumulation_steps"
    --logging_steps "$LOGGING_STEPS"
    --save_steps "$stage_save_steps"
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
    --beta "$stage_beta"
    --lmbda "$stage_lmbda"
    --sft_alpha "$stage_sft_alpha"
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
    cmd+=(--dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR")
  fi
  if [[ "$stage_max_steps" =~ ^-?[0-9]+$ ]] && [[ "$stage_max_steps" -gt 0 ]]; then
    cmd+=(--max_steps "$stage_max_steps")
  fi
  if [[ "$stage_name" == "main" && -n "$stage_teacher_model_path" ]]; then
    cmd+=(--teacher_model "$stage_teacher_model_path")
  fi
  if [[ "$stage_enable_reward_funcs" == true ]]; then
    cmd+=(
      --reward_funcs opsd_span_answer opsd_span_latent_format
      --reward_weights 1.0 1.0
      --loss_type "$stage_loss_type"
      --advantage_estimator "$stage_advantage_estimator"
    )
    if [[ -n "$stage_importance_sampling_level" ]]; then
      cmd+=(--importance_sampling_level "$stage_importance_sampling_level")
    fi
  fi
  if [[ -n "$student_adapter_path" ]]; then
    echo "STUDENT_ADAPTER_PATH=$student_adapter_path"
    cmd+=(--adapters "$student_adapter_path")
    if [[ "$stage_enable_reward_funcs" == true ]]; then
      cmd+=(--ref_adapters "$student_adapter_path")
    fi
  elif [[ "$stage_enable_rlsd" == true ]]; then
    echo "GRPO + OPSD_RLSD_ENABLED requires a student adapter path so ref_adapters can provide a static self-teacher." >&2
    exit 1
  fi
  if [[ "$USE_VLLM" == true ]]; then
    cmd+=(
      --vllm_mode "$VLLM_MODE"
      --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
      --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
    )
  fi
  "${cmd[@]}" "${FORWARD_ARGS[@]}"
}

run_gspo() {
  run_rl_stage gspo
}

run_main() {
  run_rl_stage main
}

case "$STAGE" in
  warmup)
    run_warmup
    ;;
  gspo)
    run_gspo
    ;;
  main)
    run_main
    ;;
  all)
    run_warmup
    run_gspo
    run_main
    ;;
  *)
    echo "Unsupported OPSD_SPAN_STAGE: $STAGE" >&2
    exit 1
    ;;
esac
