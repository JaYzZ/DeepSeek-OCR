#!/bin/bash
# OPD-VCR launcher for Qwen3-VL continuous replay.
#
# Usage:
#   bash Qwen/scripts/train_qwen3vl_opd_vcr.sh
#   bash Qwen/scripts/train_qwen3vl_opd_vcr.sh Qwen/configs/distillation/qwen3vl_opd_vcr_continuous.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/qwen3vl_common.sh"
cd "$REPO_ROOT"

PYTHON_BIN="$(qwen3vl_require_python_bin "$REPO_ROOT")"

DEFAULT_CONFIG="$REPO_ROOT/Qwen/configs/distillation/qwen3vl_opd_vcr_continuous.yaml"
DEFAULT_RUNTIME_ENV_CONFIG="$REPO_ROOT/Qwen/configs/qwen3vl_runtime_env.yaml"
DEFAULT_ACCELERATE_CONFIG="$REPO_ROOT/Qwen/configs/distillation/accelerate_zero2.yaml"

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
if [ "${1:-}" != "" ]; then
  shift || true
fi

QWEN3VL_RUNTIME_ENV_CONFIG="${QWEN3VL_RUNTIME_ENV_CONFIG:-$DEFAULT_RUNTIME_ENV_CONFIG}"
OPD_ACCELERATE_CONFIG="${OPD_ACCELERATE_CONFIG:-$DEFAULT_ACCELERATE_CONFIG}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi
if [ ! -f "$QWEN3VL_RUNTIME_ENV_CONFIG" ]; then
  echo "Runtime env config not found: $QWEN3VL_RUNTIME_ENV_CONFIG" >&2
  exit 1
fi
if [ ! -f "$OPD_ACCELERATE_CONFIG" ]; then
  echo "Accelerate config not found: $OPD_ACCELERATE_CONFIG" >&2
  exit 1
fi

_get_main_config() {
  qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "$1" "${2:-}"
}

_set_env_from_runtime() {
  qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$QWEN3VL_RUNTIME_ENV_CONFIG" "$1" "$2" "$3"
}

_set_env_from_runtime "QWEN3VL_LATENT_SUPERVISION" "latent_supervision" "1"
_set_env_from_runtime "QWEN3VL_LATENT_TOKEN_ID" "latent_token_id" "151669"
_set_env_from_runtime "QWEN3VL_THINKING_START_ID" "thinking_start_id" "151667"
_set_env_from_runtime "QWEN3VL_THINKING_END_ID" "thinking_end_id" "151668"
_set_env_from_runtime "QWEN3VL_THINKING_SEP_ID" "thinking_sep_id" "151670"
_set_env_from_runtime "QWEN3VL_MAX_NEW_TOKENS" "max_new_tokens" "40960"
_set_env_from_runtime "MIN_CONTINUOUS_STEPS" "min_continuous_steps" "0"

unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

TIMESTAMP="${QWEN3VL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
BASE_OUTPUT_DIR="$(_get_main_config "training.output_dir" "$REPO_ROOT/Qwen/checkpoints/qwen3vl-2b/lora/opd_vcr_continuous")"
GRAD_ACCUM_STEPS="$(_get_main_config "training.gradient_accumulation_steps" "1")"
if [[ ! "$BASE_OUTPUT_DIR" = /* ]]; then
  BASE_OUTPUT_DIR="$REPO_ROOT/$BASE_OUTPUT_DIR"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_OUTPUT_DIR/run_${TIMESTAMP}}"
RUN_DIR="$OUTPUT_DIR"

mkdir -p "$RUN_DIR"
qwen3vl_snapshot_run_configs "$CONFIG_PATH" "$QWEN3VL_RUNTIME_ENV_CONFIG" "$RUN_DIR"
CONFIG_PATH="$RUN_DIR/$(basename "$CONFIG_PATH")"
QWEN3VL_RUNTIME_ENV_CONFIG="$RUN_DIR/$(basename "$QWEN3VL_RUNTIME_ENV_CONFIG")"
cp -f "$OPD_ACCELERATE_CONFIG" "$RUN_DIR/$(basename "$OPD_ACCELERATE_CONFIG")"
OPD_ACCELERATE_CONFIG="$RUN_DIR/$(basename "$OPD_ACCELERATE_CONFIG")"

LOG_FILE="$RUN_DIR/training.log"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(qwen3vl_count_visible_gpus "$PYTHON_BIN")}"
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [ "$NPROC_PER_NODE" -lt 1 ]; then
  NPROC_PER_NODE=1
fi

TRAIN_SCRIPT="Qwen/scripts/train_qwen3vl_opd_vcr.py"
TRAIN_ARGS=(
  "$TRAIN_SCRIPT"
  "$CONFIG_PATH"
  "--output-dir" "$RUN_DIR"
)

for arg in "$@"; do
  TRAIN_ARGS+=("$arg")
done

echo "========================================================================" | tee -a "$LOG_FILE"
echo "Qwen3-VL OPD-VCR continuous replay run" | tee -a "$LOG_FILE"
echo "Config: $CONFIG_PATH" | tee -a "$LOG_FILE"
echo "Runtime env: $QWEN3VL_RUNTIME_ENV_CONFIG" | tee -a "$LOG_FILE"
echo "Accelerate config: $OPD_ACCELERATE_CONFIG" | tee -a "$LOG_FILE"
echo "Run dir: $RUN_DIR" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}" | tee -a "$LOG_FILE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"

set +e
if [ "$NPROC_PER_NODE" -gt 1 ]; then
  MASTER_PORT="$(qwen3vl_resolve_master_port "$PYTHON_BIN")"
  export MASTER_PORT
  "$PYTHON_BIN" -m accelerate.commands.launch \
    --config_file "$OPD_ACCELERATE_CONFIG" \
    --num_processes "$NPROC_PER_NODE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --main_process_port "$MASTER_PORT" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  exit_code=${PIPESTATUS[0]}
else
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  exit_code=${PIPESTATUS[0]}
fi
set -e

exit "$exit_code"
