#!/bin/bash
# OCRVL LlamaFactory Training - Native LlamaFactory with OCR Model
#
# This script uses standard llamafactory-cli with:
# - Minimal `sitecustomize.py` integration (OCR model + OCRVL template registration)
# - LoRA + `additional_target` for connector training
#
# Usage:
#   bash OCRVL/scripts/train_llava_llamafactory.sh [config.yaml]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Required python interpreter (OCRFlow env).
PYTHON_BIN="$REPO_ROOT/../../envs/ocrflow/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ Python not found or not executable: $PYTHON_BIN" >&2
  echo "   Please ensure OCRFlow env exists at: $REPO_ROOT/../../envs/ocrflow" >&2
  exit 1
fi

# Default config (alignment mode)
DEFAULT_CONFIG="$REPO_ROOT/OCRVL/examples/llamafactory/qwen3vl_dpskocr_lora_alignment.yaml"

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
if [ "${1:-}" != "" ]; then
  shift || true
fi

# Generate timestamp for unique output directory (can be overridden by command line)
TIMESTAMP="${OCRVL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
DEFAULT_OUTDIR="$REPO_ROOT/OCRVL/checkpoints/llamafactory/qwen3vl-2b/lora/run_${TIMESTAMP}"

# Check if user specified output_dir in command line
HAS_OUTDIR=false
for arg in "$@"; do
  if [[ "$arg" == output_dir=* ]]; then
    HAS_OUTDIR=true
    break
  fi
done

# If no output_dir specified, use timestamped directory
if [ "$HAS_OUTDIR" = false ]; then
  set -- "$@" "output_dir=$DEFAULT_OUTDIR"
fi

# Determine final output directory (may be overridden by command line)
OUTPUT_DIR="$DEFAULT_OUTDIR"
for arg in "$@"; do
  if [[ "$arg" == output_dir=* ]]; then
    OUTPUT_DIR="${arg#output_dir=}"
    break
  fi
done

# Setup logging
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training.log"

echo "Logging to: $LOG_FILE"
echo ""

# Check if dataset exists, if not build it
DATASET_JSONL="$REPO_ROOT/OCRVL/llamafactory/data/ocrvl_alignment_prerendered.jsonl"
if [ ! -f "$DATASET_JSONL" ]; then
    echo "========================================================================"
    echo "Dataset not found: $DATASET_JSONL"
    echo "Building alignment dataset with pre-rendered instructions..."
    echo "========================================================================"
    echo ""

    "$PYTHON_BIN" "$REPO_ROOT/OCRVL/scripts/build_alignment_dataset_with_prerendered_instructions.py"

    if [ $? -ne 0 ]; then
        echo "❌ Failed to build dataset"
        exit 1
    fi

    echo ""
    echo "✓ Dataset built successfully"
    echo ""
fi

# Set Python path for `sitecustomize.py` integration.
# NOTE: `sitecustomize.py` is at repo root (no __init__.py) so Python auto-imports it on startup.
export PYTHONPATH="$REPO_ROOT/../LlamaFactory/src:$REPO_ROOT:$REPO_ROOT/OCRVL/llamafactory:${PYTHONPATH:-}"

# Default to 8 GPUs unless user explicitly sets CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
# Keep deepstack enabled unless explicitly disabled (matches alignment connector targets).
export OCRVL_DPSK_DEEPSTACK="${OCRVL_DPSK_DEEPSTACK:-1}"

# Prefer local mirrors for offline training
# Local DeepSeek-OCR copy ensures FSDP treats it as part of the model (not external dependency)
export DPSK_MODEL_PATH="${DPSK_MODEL_PATH:-$REPO_ROOT/OCRVL/checkpoints/OCR-Qwen3-VL-2B}"
export DPSK_DTYPE="${DPSK_DTYPE:-bf16}"

# Enable checkpoint evaluation
export OCRVL_ENABLE_TRANSPARENT_EVAL="${OCRVL_ENABLE_TRANSPARENT_EVAL:-1}"
export OCRVL_REPO_ROOT="$REPO_ROOT"
export OCRVL_TRANSPARENT_EVAL_SAMPLES="${OCRVL_TRANSPARENT_EVAL_SAMPLES:-$REPO_ROOT/OCRVL/llamafactory/transparent_eval_samples.json}"
export OCRVL_TRANSPARENT_EVAL_MAX_NEW_TOKENS="${OCRVL_TRANSPARENT_EVAL_MAX_NEW_TOKENS:-128}"
export OCRVL_TRANSPARENT_EVAL_TEMPERATURE="${OCRVL_TRANSPARENT_EVAL_TEMPERATURE:-0.0}"
export OCRVL_TRANSPARENT_EVAL_RUN_AT_START="${OCRVL_TRANSPARENT_EVAL_RUN_AT_START:-0}"
export OCRVL_TRANSPARENT_EVAL_LIMIT="${OCRVL_TRANSPARENT_EVAL_LIMIT:-}"

# Enable rendering (RENDER=1 for on-the-fly rendering during instruction tuning)
# For alignment stage, instructions are pre-rendered in the dataset (RENDER has no effect)
export RENDER="${RENDER:-1}"

# Determine rendering status for display
if [ "$RENDER" = "1" ]; then
    RENDER_STATUS="ENABLED (VQA questions as images)"
else
    RENDER_STATUS="DISABLED"
fi

echo "========================================================================"
echo "OCRVL LlamaFactory Training (Native llamafactory-cli)"
echo "========================================================================"
echo "Config: $CONFIG_PATH"
echo "DPSK_MODEL_PATH: $DPSK_MODEL_PATH"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Using standard LlamaFactory with:"
echo "  - OCR model registration (sitecustomize.py)"
echo "  - LoRA + additional_target for connectors"
echo "  - On-the-fly rendering (RENDER): $RENDER_STATUS"
echo "  - Alignment: Pre-rendered instructions in dataset"
echo "  - Transparent eval: $OCRVL_ENABLE_TRANSPARENT_EVAL (samples: $OCRVL_TRANSPARENT_EVAL_SAMPLES)"
echo "    - Run at start: $OCRVL_TRANSPARENT_EVAL_RUN_AT_START (writes checkpoint-0/eval_results)"
echo "    - Limit: ${OCRVL_TRANSPARENT_EVAL_LIMIT:-all} (set OCRVL_TRANSPARENT_EVAL_LIMIT=1 for quick smoke test)"
echo "========================================================================"
echo ""

# Run standard LlamaFactory CLI with tee for logging
# Use PIPESTATUS to preserve exit code from llamafactory-cli
NPROC_PER_NODE_DEFAULT="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$NPROC_PER_NODE_DEFAULT}"

"$PYTHON_BIN" -m llamafactory.cli train "$CONFIG_PATH" "$@" 2>&1 | tee -a "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
exit ${exit_code:-0}
