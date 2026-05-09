#!/bin/bash
# Privileged OPSD launcher for Qwen3-VL backed by VERL async rollout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/qwen3vl_common.sh"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

PYTHON_BIN="$(qwen3vl_require_python_bin "$REPO_ROOT")"
TIMESTAMP="${QWEN3VL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUNTIME_ENV_STAMP="${QWEN3VL_RUNTIME_ENV_STAMP:-$TIMESTAMP}"
VISIBLE_GPUS="$(qwen3vl_count_visible_gpus "$PYTHON_BIN")"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "$VISIBLE_GPUS" =~ ^[0-9]+$ ]] && [[ "$VISIBLE_GPUS" -gt 0 ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((VISIBLE_GPUS - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

PROJECT_BASE_CONFIG="$REPO_ROOT/Qwen/configs/rl/qwen3vl_opsd.yaml"
CONFIG_PATH="${1:-$REPO_ROOT/Qwen/configs/distillation/qwen3vl_opsd.yaml}"
if [[ "${1:-}" != "" ]]; then
  shift || true
fi
PROJECT_CONFIG="${PROJECT_CONFIG:-}"
RUNTIME_ENV_CONFIG="${QWEN3VL_RUNTIME_ENV_CONFIG:-$REPO_ROOT/Qwen/configs/qwen3vl_runtime_env.yaml}"
MODEL_PATH="${MODEL_PATH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "model.model_name_or_path" "$ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking")}"
if [[ ! "$MODEL_PATH" = /* ]]; then
  MODEL_PATH="$(qwen3vl_resolve_path "$REPO_ROOT" "$MODEL_PATH")"
fi
DATASET_NAMES="${DATASET_NAMES:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.dataset_names" "r1ov_thinking,deepvision_thinking")}"
MANIFEST_DIR="${MANIFEST_DIR:-$REPO_ROOT/Qwen/data/opsd_manifest}"
TRAIN_FILE="${TRAIN_FILE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.opsd_manifest" "$MANIFEST_DIR/r1ov_deepvision_opsd.jsonl")}"
if [[ ! "$TRAIN_FILE" = /* ]]; then
  TRAIN_FILE="$REPO_ROOT/$TRAIN_FILE"
fi
VAL_FILE="${VAL_FILE:-$TRAIN_FILE}"
if [[ ! "$VAL_FILE" = /* ]]; then
  VAL_FILE="$REPO_ROOT/$VAL_FILE"
fi
BASE_OUTPUT_DIR="$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.output_dir" "$REPO_ROOT/Qwen/checkpoints/qwen3vl-2b/verl/opsd")"
if [[ ! "$BASE_OUTPUT_DIR" = /* ]]; then
  BASE_OUTPUT_DIR="$REPO_ROOT/$BASE_OUTPUT_DIR"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_OUTPUT_DIR/run_${TIMESTAMP}}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$PROJECT_BASE_CONFIG" ]]; then
  echo "Project base config not found: $PROJECT_BASE_CONFIG" >&2
  exit 1
fi
if [[ -n "$PROJECT_CONFIG" && ! -f "$PROJECT_CONFIG" ]]; then
  echo "Project config not found: $PROJECT_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$RUNTIME_ENV_CONFIG" ]]; then
  echo "Runtime env config not found: $RUNTIME_ENV_CONFIG" >&2
  exit 1
fi

OVERWRITE_OPSD_MANIFEST="${OVERWRITE_OPSD_MANIFEST:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.overwrite_manifest" "0")}"
if [[ ! -f "$TRAIN_FILE" || "$OVERWRITE_OPSD_MANIFEST" == "1" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/Qwen/data/build_qwen3vl_opsd_dataset.py" \
    --datasets "$DATASET_NAMES" \
    --output-jsonl "$TRAIN_FILE"
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-$VISIBLE_GPUS}"
if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$GPUS_PER_NODE" -lt 1 ]]; then
  GPUS_PER_NODE=1
fi

qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_LATENT_SUPERVISION" "latent_supervision" "1"
qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_LATENT_TOKEN_ID" "latent_token_id" "151669"
qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_THINKING_START_ID" "thinking_start_id" "151667"
qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_THINKING_END_ID" "thinking_end_id" "151668"
qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$RUNTIME_ENV_CONFIG" "QWEN3VL_THINKING_SEP_ID" "thinking_sep_id" "151670"
qwen3vl_export_env_from_main_then_runtime "$PYTHON_BIN" "$CONFIG_PATH" "generation.max_new_tokens" "$RUNTIME_ENV_CONFIG" "max_new_tokens" "QWEN3VL_MAX_NEW_TOKENS" "8192"
qwen3vl_export_env_from_main_then_runtime "$PYTHON_BIN" "$CONFIG_PATH" "generation.min_continuous_steps" "$RUNTIME_ENV_CONFIG" "min_continuous_steps" "MIN_CONTINUOUS_STEPS" "0"
qwen3vl_export_max_continuous_steps_from_yaml "$PYTHON_BIN" "$CONFIG_PATH" "$RUNTIME_ENV_CONFIG"
export QWEN3VL_LOSS_TYPE="${QWEN3VL_LOSS_TYPE:-none}"
export QWEN3VL_LATENT_CE_ACTIVE="${QWEN3VL_LATENT_CE_ACTIVE:-0}"
export QWEN3VL_LATENT_CE_TOKEN="${QWEN3VL_LATENT_CE_TOKEN:-0}"
export QWEN3VL_HIDDEN_STATES_HOOK="${QWEN3VL_HIDDEN_STATES_HOOK:-0}"
qwen3vl_export_env_from_main_then_runtime "$PYTHON_BIN" "$CONFIG_PATH" "vae.intermediate_size" "$RUNTIME_ENV_CONFIG" "vae_intermediate_size" "QWEN3VL_VAE_INTERMEDIATE_SIZE" "512"
export OPSD_ENABLED="${OPSD_ENABLED:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.opsd_enabled" "1")}"
export OPSD_WEIGHT="${OPSD_WEIGHT:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.opsd_weight" "1.0")}"
export OPSD_TEMPERATURE="${OPSD_TEMPERATURE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.temperature" "1.0")}"
export OPSD_LOGPROB_CHUNK_SIZE="${OPSD_LOGPROB_CHUNK_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.sampled_logprob_chunk_size" "1024")}"
export OPSD_OT_REPLAY="${OPSD_OT_REPLAY:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.ot_replay_enabled" "0")}"
export OPSD_OT_WEIGHT="${OPSD_OT_WEIGHT:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "loss.ot_replay_weight" "0.0")}"
export OPSD_SYSTEM_PROMPT="${OPSD_SYSTEM_PROMPT:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "system_prompt" "")}"
export OPSD_STUDENT_TEMPLATE="${OPSD_STUDENT_TEMPLATE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "prompts.student_user" "Answer the question.\n\nQuestion:\n{question_text}")}"
export OPSD_TEACHER_TEMPLATE="${OPSD_TEACHER_TEMPLATE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "prompts.teacher_user" "Problem:\n{question_text}\n\nHere is a reference solution to this problem:\n=== Reference Solution Begin ===\n{reference_solution}\n=== Reference Solution End ===")}"
export OPSD_TEACHER_MAX_PROMPT_LENGTH="${OPSD_TEACHER_MAX_PROMPT_LENGTH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "prompts.teacher_max_prompt_length" "4096")}"
export OPSD_TEACHER_MODEL_PATH="${OPSD_TEACHER_MODEL_PATH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "teacher.model_name_or_path" "")}"
if [[ -n "$OPSD_TEACHER_MODEL_PATH" && ! "$OPSD_TEACHER_MODEL_PATH" = /* ]]; then
  OPSD_TEACHER_MODEL_PATH="$(qwen3vl_resolve_path "$REPO_ROOT" "$OPSD_TEACHER_MODEL_PATH")"
fi
export OPSD_TEACHER_ADAPTER_PATH="${OPSD_TEACHER_ADAPTER_PATH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "teacher.adapter_name_or_path" "")}"
if [[ -n "$OPSD_TEACHER_ADAPTER_PATH" && ! "$OPSD_TEACHER_ADAPTER_PATH" = /* ]]; then
  OPSD_TEACHER_ADAPTER_PATH="$(qwen3vl_resolve_path "$REPO_ROOT" "$OPSD_TEACHER_ADAPTER_PATH")"
fi
export OPSD_OFFLOAD_TEACHER_MODEL="${OPSD_OFFLOAD_TEACHER_MODEL:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "teacher.offload_model" "0")}"
export OPSD_OFF_POLICY_MODE="${OPSD_OFF_POLICY_MODE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "mode.off_policy" "0")}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.max_prompt_length" "8192")}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.max_response_length" "$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "generation.max_new_tokens" "8192")")}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.gen_batch_size" "4")}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.train_batch_size" "4")}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.val_batch_size" "4")}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.dataloader_num_workers" "0")}"
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.filter_overlong_prompts" "0")}"
export FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "data.filter_overlong_prompts_workers" "1")}"
export LORA_RANK="${LORA_RANK:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "lora.rank" "8")}"
export LORA_ALPHA="${LORA_ALPHA:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "lora.alpha" "16")}"
export LR="${LR:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.learning_rate" "1e-6")}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.num_train_epochs" "1")}"
export SAVE_FREQ="${SAVE_FREQ:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.save_steps" "10")}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.save_total_limit" "2")}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.ppo_mini_batch_size" "4")}"
export ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.actor_micro_batch_size" "1")}"
export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.actor_max_token_len" "20480")}"
export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.infer_max_token_len" "20480")}"
export USE_KL_LOSS="${USE_KL_LOSS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.use_kl_loss" "1")}"
export KL_COEF="${KL_COEF:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "training.kl_coef" "0.001")}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "rollout.tensor_parallel_size" "1")}"
export ROLLOUT_N="${ROLLOUT_N:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "rollout.n" "1")}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "rollout.gpu_memory_utilization" "0.55")}"
export ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "rollout.max_num_batched_tokens" "20480")}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "rollout.max_model_len" "20480")}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "rollout.enforce_eager" "0")}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "rollout.agent_num_workers" "1")}"
export REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "reward.num_workers" "1")}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "resources.ray_num_cpus" "8")}"
export TEMPERATURE="${TEMPERATURE:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "generation.temperature" "1.0")}"
export TOP_P="${TOP_P:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "generation.top_p" "1.0")}"
export TOP_K="${TOP_K:-$(qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "generation.top_k" "20")}"

case "${FILTER_OVERLONG_PROMPTS,,}" in
  1|true|yes|on) FILTER_OVERLONG_PROMPTS=true ;;
  *) FILTER_OVERLONG_PROMPTS=false ;;
esac
case "${USE_KL_LOSS,,}" in
  1|true|yes|on) USE_KL_LOSS=true ;;
  *) USE_KL_LOSS=false ;;
esac
case "${ROLLOUT_ENFORCE_EAGER,,}" in
  1|true|yes|on) ROLLOUT_ENFORCE_EAGER=true ;;
  *) ROLLOUT_ENFORCE_EAGER=false ;;
esac
case "${OPSD_OFF_POLICY_MODE,,}" in
  1|true|yes|on) OPSD_OFF_POLICY_MODE=true ;;
  *) OPSD_OFF_POLICY_MODE=false ;;
esac
if [[ "$OPSD_OFF_POLICY_MODE" == true && -z "$OPSD_TEACHER_MODEL_PATH" ]]; then
  echo "teacher.model_name_or_path must be set when mode.off_policy=true" >&2
  exit 1
fi
export FILTER_OVERLONG_PROMPTS
export USE_KL_LOSS
export ROLLOUT_ENFORCE_EAGER
export OPSD_OFF_POLICY_MODE

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/training.log}"
RESOLVED_CONFIG_FILE="$OUTPUT_DIR/resolved_runtime_config.yaml"

qwen3vl_copy_if_present "$CONFIG_PATH" "$OUTPUT_DIR"
qwen3vl_copy_if_present "$PROJECT_BASE_CONFIG" "$OUTPUT_DIR"
qwen3vl_copy_if_present "$PROJECT_CONFIG" "$OUTPUT_DIR"
qwen3vl_copy_if_present "$RUNTIME_ENV_CONFIG" "$OUTPUT_DIR"

exec > >(qwen3vl_sanitize_log_stream | tee -a "$LOG_FILE") 2>&1

export MODEL_PATH
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$MODEL_PATH}"
export TRAIN_FILE
export VAL_FILE
export OUTPUT_DIR
export ROOT_DIR
export GPUS_PER_NODE
export QWEN3VL_RUNTIME_ENV_STAMP="$RUNTIME_ENV_STAMP"
export VLLM_PLUGINS="${VLLM_PLUGINS:-vllm_thinking}"
export VLLM_THINKING="${VLLM_THINKING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
# vLLM's CuMem memory pool rejects expandable segments. Keep any other user-provided
# allocator options, but never forward expandable_segments:True into Ray/vLLM workers.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  PYTORCH_CUDA_ALLOC_CONF="$(printf '%s' "$PYTORCH_CUDA_ALLOC_CONF" | sed -E 's/(^|,)expandable_segments:True(,|$)/\1/g; s/,+/,/g; s/^,//; s/,$//')"
fi
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi
qwen3vl_configure_ray_noset_cuda_visible_devices

INIT_LORA_PATH="${INIT_LORA_PATH:-${LORA_ADAPTER_PATH:-}}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
if [[ -z "$INIT_LORA_PATH" && "$RESUME_MODE" == "resume_path" && -n "$RESUME_FROM_PATH" ]]; then
  INIT_LORA_PATH="$RESUME_FROM_PATH"
fi
TRAINER_RESUME_MODE="$RESUME_MODE"
TRAINER_RESUME_FROM_PATH="$RESUME_FROM_PATH"
if [[ "$TRAINER_RESUME_MODE" == "resume_path" && -n "$TRAINER_RESUME_FROM_PATH" && "$TRAINER_RESUME_FROM_PATH" != *global_step_* ]]; then
  TRAINER_RESUME_MODE="disable"
  TRAINER_RESUME_FROM_PATH=""
fi
export RESUME_MODE="$TRAINER_RESUME_MODE"
export RESUME_FROM_PATH="$TRAINER_RESUME_FROM_PATH"
if [[ -n "$INIT_LORA_PATH" ]]; then
  if [[ ! -d "$INIT_LORA_PATH" ]]; then
    echo "Initial LoRA adapter path not found: $INIT_LORA_PATH" >&2
    exit 1
  fi
  export VLLM_LORA_CHECKPOINT_PATH="${VLLM_LORA_CHECKPOINT_PATH:-$INIT_LORA_PATH}"
fi

EFFECTIVE_CONFIG_LINES="$(env PYTHONPATH= PYTHONSAFEPATH=1 "$PYTHON_BIN" - "$PROJECT_BASE_CONFIG" "${PROJECT_CONFIG:-}" "$RESOLVED_CONFIG_FILE" <<'PY'
import sys
from omegaconf import OmegaConf

base_config, overlay_config, resolved_config_path = sys.argv[1:4]
if not OmegaConf.has_resolver("gpu_adapt"):
    OmegaConf.register_new_resolver(
        "gpu_adapt",
        lambda gpus, one, two, four, eight, fallback=None: (
            one if int(gpus) == 1 else two if int(gpus) == 2 else four if int(gpus) == 4 else eight if int(gpus) == 8 else (fallback if fallback is not None else eight)
        ),
    )
config = OmegaConf.load(base_config)
if overlay_config:
    config = OmegaConf.merge(config, OmegaConf.load(overlay_config))
OmegaConf.resolve(config)
with open(resolved_config_path, "w", encoding="utf-8") as fh:
    fh.write(OmegaConf.to_yaml(config, resolve=True))
for key, path in {
    "NNODES": "trainer.nnodes",
    "GPUS_PER_NODE": "trainer.n_gpus_per_node",
    "GEN_BATCH_SIZE": "data.gen_batch_size",
    "TRAIN_BATCH_SIZE": "data.train_batch_size",
    "MAX_RESPONSE_LENGTH": "data.max_response_length",
    "ROLLOUT_N": "actor_rollout_ref.rollout.n",
    "SAVE_FREQ": "trainer.save_freq",
}.items():
    print(f"{key}={OmegaConf.select(config, path)}")
PY
)"

{
  printf 'CONFIG_PATH=%s\n' "$CONFIG_PATH"
  printf 'PROJECT_BASE_CONFIG=%s\n' "$PROJECT_BASE_CONFIG"
  printf 'PROJECT_CONFIG=%s\n' "${PROJECT_CONFIG:-<none>}"
  printf 'MODEL_PATH=%s\n' "$MODEL_PATH"
  printf 'TRAIN_FILE=%s\n' "$TRAIN_FILE"
  printf 'VAL_FILE=%s\n' "$VAL_FILE"
  printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
  printf 'ROOT_DIR=%s\n' "$ROOT_DIR"
  printf 'MAX_CONTINUOUS_STEPS=%s\n' "${MAX_CONTINUOUS_STEPS:-<unset>}"
  printf 'OPSD_ENABLED=%s\n' "$OPSD_ENABLED"
  printf 'OPSD_WEIGHT=%s\n' "$OPSD_WEIGHT"
  printf 'OPSD_OT_REPLAY=%s\n' "$OPSD_OT_REPLAY"
  printf 'OPSD_OFF_POLICY_MODE=%s\n' "$OPSD_OFF_POLICY_MODE"
  printf 'OPSD_TEACHER_MODEL_PATH=%s\n' "${OPSD_TEACHER_MODEL_PATH:-<none>}"
  printf 'OPSD_TEACHER_ADAPTER_PATH=%s\n' "${OPSD_TEACHER_ADAPTER_PATH:-<none>}"
  printf 'DATALOADER_NUM_WORKERS=%s\n' "$DATALOADER_NUM_WORKERS"
  printf 'FILTER_OVERLONG_PROMPTS=%s\n' "$FILTER_OVERLONG_PROMPTS"
  printf 'REWARD_NUM_WORKERS=%s\n' "$REWARD_NUM_WORKERS"
  printf 'RAY_NUM_CPUS=%s\n' "$RAY_NUM_CPUS"
  printf '%s\n' "$EFFECTIVE_CONFIG_LINES"
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
if [[ "$TRAINER_RESUME_MODE" == "resume_path" ]]; then
  CMD+=(trainer.resume_from_path="$TRAINER_RESUME_FROM_PATH")
fi
CMD+=("$@")

printf 'Launching Qwen3-VL OPSD with VERL backend output_dir=%s\n' "$OUTPUT_DIR"
printf 'Training log=%s\n' "$LOG_FILE"
printf 'Visible GPUs=%s cuda_visible_devices=%s train=%s\n' "$VISIBLE_GPUS" "${CUDA_VISIBLE_DEVICES:-<unset>}" "$TRAIN_FILE"
printf 'Runtime env stamp=%s max_continuous_steps=%s\n' "$RUNTIME_ENV_STAMP" "${MAX_CONTINUOUS_STEPS:-<unset>}"
printf 'Init LoRA=%s resume_mode=%s resume_from=%s\n' "${INIT_LORA_PATH:-<none>}" "$RESUME_MODE" "${RESUME_FROM_PATH:-<none>}"
printf 'Trainer resume_mode=%s trainer_resume_from=%s\n' "$TRAINER_RESUME_MODE" "${TRAINER_RESUME_FROM_PATH:-<none>}"
printf 'Saved config snapshots: %s %s %s\n' \
  "$OUTPUT_DIR/$(basename "$PROJECT_BASE_CONFIG")" \
  "$RESOLVED_CONFIG_FILE" \
  "$OUTPUT_DIR/launcher_effective_env.snapshot.txt"

exec "${CMD[@]}"
