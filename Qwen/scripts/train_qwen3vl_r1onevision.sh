#!/bin/bash
# Qwen3VL R1-OneVision SFT Training with Latent Supervision
#
# This script trains Qwen3VL-2B-Thinking on R1-OneVision dataset using:
# - Pre-encoded vision features (no encoding during training)
# - Latent injection at <|latent_step|> positions
# - Thinking loss (REPA/OT/NCE/MSE) on latent predictions
#
# Usage:
#   bash Qwen/scripts/train_qwen3vl_r1onevision.sh [config.yaml]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Required python interpreter (OCRFlow env)
PYTHON_BIN="$REPO_ROOT/../../envs/ocrflow/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ Python not found or not executable: $PYTHON_BIN" >&2
  echo "   Please ensure OCRFlow env exists at: $REPO_ROOT/../../envs/ocrflow" >&2
  exit 1
fi

# Default config
DEFAULT_CONFIG="$REPO_ROOT/Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml"

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
if [ "${1:-}" != "" ]; then
  shift || true
fi

# Derive model path from config (used for HF_MODULES_CACHE if present)
MODEL_PATH="$(grep -E '^model_name_or_path:' "$CONFIG_PATH" | head -n 1 | awk '{print $2}')"

# Check dataset exists
DATASET_JSONL="$REPO_ROOT/Qwen/data/r1_onevision_thinking.jsonl"
if [ ! -f "$DATASET_JSONL" ]; then
    echo "❌ Dataset not found: $DATASET_JSONL" >&2
    echo "" >&2
    echo "Please build the dataset first:" >&2
    echo "  bash Qwen/scripts/build_r1_onevision_thinking.sh --encode-only --all" >&2
    exit 1
fi

echo "✓ Dataset found: $DATASET_JSONL"
echo "  Samples: $(wc -l < "$DATASET_JSONL")"
echo ""

# Generate timestamp for unique output directory
TIMESTAMP="${QWEN3VL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
DEFAULT_OUTDIR="$REPO_ROOT/Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_${TIMESTAMP}"

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

# Determine final output directory
OUTPUT_DIR="$DEFAULT_OUTDIR"
for arg in "$@"; do
  if [[ "$arg" == output_dir=* ]]; then
    OUTPUT_DIR="${arg#output_dir=}"
    break
  fi
done

# Setup logging
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/training.log"

echo "Logging to: $LOG_FILE"
echo ""

# Set Python path for llamafactory and sitecustomize.py integration
# sitecustomize.py at repo root enables latent supervision patches
export PYTHONPATH="$REPO_ROOT/../LlamaFactory/src:$REPO_ROOT:${PYTHONPATH:-}"

# If linear checkpoint provides a preferred HF modules cache, use it unless overridden
if [ -z "${HF_MODULES_CACHE:-}" ] && [ -n "${MODEL_PATH:-}" ] && [ -f "$MODEL_PATH/hf_modules_cache.path" ]; then
  export HF_MODULES_CACHE="$(cat "$MODEL_PATH/hf_modules_cache.path")"
fi

# Default to 8 GPUs unless user explicitly sets CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Enable latent supervision patches
export QWEN3VL_LATENT_SUPERVISION="${QWEN3VL_LATENT_SUPERVISION:-1}"

# Set max_new_tokens for generation (prevents max_length crash when input > cutoff_len)
export QWEN3VL_MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-2048}"

# Enable vision-only compilation (experimental, compiles after FSDP wrapping)
# Now handled by TransparentEvalCallback.on_train_begin to avoid FSDP conflicts
export QWEN3VL_COMPILE_VISION_ONLY="${QWEN3VL_COMPILE_VISION_ONLY:-1}"

# Reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Detect distributed backend from config (DeepSpeed vs FSDP) for display purposes
USE_DEEPSPEED=false
USE_FSDP=false
if grep -q "^deepspeed:" "$CONFIG_PATH" 2>/dev/null; then
    USE_DEEPSPEED=true
elif grep -q "^fsdp:" "$CONFIG_PATH" 2>/dev/null; then
    USE_FSDP=true
fi

# DeepSpeed-specific NCCL configuration for single-node multi-GPU
if [ "$USE_DEEPSPEED" = true ]; then
    # Debugging (turn off once stable)
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

    # Avoid IB issues on single machine
    export NCCL_IB_DISABLE=1

    # Use shared memory + PCIe/NVLink
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0

    # Interface selection (important!)
    export NCCL_SOCKET_IFNAME=lo
fi

# Ensure pack-after-injection uses the intended cutoff length
export QWEN3VL_CUTOFF_LEN="${QWEN3VL_CUTOFF_LEN:-8192}"

# Latent supervision configuration
# sitecustomize.py will add tokens and override IDs if needed.
export QWEN3VL_LATENT_TOKEN_ID="${QWEN3VL_LATENT_TOKEN_ID:-151669}"  # <|latent_step|>
export QWEN3VL_THINKING_START_ID="${QWEN3VL_THINKING_START_ID:-151667}"  # <think>
export QWEN3VL_THINKING_END_ID="${QWEN3VL_THINKING_END_ID:-151668}"    # </think>

# Loss type configuration (default: OT loss)
# Options: ot, mse, repa, nce, or combinations like "ot+mse", "repa:0.7+mse:0.3"
export QWEN3VL_LOSS_TYPE="${QWEN3VL_LOSS_TYPE:-ot}"

# Sequence matching strategy for loss computation (truncate, repeat, interpolate)
export QWEN3VL_MATCH_STRATEGY="${QWEN3VL_MATCH_STRATEGY:-truncate}"

# Thinking loss weight (relative to CE loss, default: 1.0 = equal weight)
export QWEN3VL_THINKING_LOSS_WEIGHT="${QWEN3VL_THINKING_LOSS_WEIGHT:-1.0}"

# Transparent evaluation (default: enabled)
# Runs qualitative inference on eval samples at each eval step
export OCRVL_ENABLE_TRANSPARENT_EVAL="${OCRVL_ENABLE_TRANSPARENT_EVAL:-1}"
export OCRVL_TRANSPARENT_EVAL_MAX_NEW_TOKENS="${OCRVL_TRANSPARENT_EVAL_MAX_NEW_TOKENS:-512}"

# Display configuration
TRAINING_TYPE="R1-OneVision SFT Training (Latent Supervision)"
LOSS_TYPE_DISPLAY="$QWEN3VL_LOSS_TYPE (weight: $QWEN3VL_THINKING_LOSS_WEIGHT)"

echo "========================================================================"
echo "Qwen3VL R1-OneVision SFT Training"
echo "========================================================================"
echo "Training Type: $TRAINING_TYPE"
echo "Distributed Backend: $([ "$USE_DEEPSPEED" = true ] && echo "DeepSpeed" || ([ "$USE_FSDP" = true ] && echo "FSDP" || echo "Unknown"))"
echo "Config: $CONFIG_PATH"
echo "Dataset: $DATASET_JSONL"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Latent Supervision:"
echo "  - Enabled: $QWEN3VL_LATENT_SUPERVISION"
echo "  - Loss type: $LOSS_TYPE_DISPLAY"
echo "  - Match strategy: $QWEN3VL_MATCH_STRATEGY"
echo ""
echo "Transparent Evaluation:"
echo "  - Enabled: $OCRVL_ENABLE_TRANSPARENT_EVAL"
echo "  - Max new tokens: $OCRVL_TRANSPARENT_EVAL_MAX_NEW_TOKENS"
echo ""
echo "Special Tokens:"
echo "  - <|latent_step|>: $QWEN3VL_LATENT_TOKEN_ID"
echo "  - Thinking start: $QWEN3VL_THINKING_START_ID"
echo "  - Thinking end: $QWEN3VL_THINKING_END_ID"
echo "========================================================================"
echo ""

# Run llamafactory CLI with tee for logging
# Use PIPESTATUS to preserve exit code from llamafactory-cli
NPROC_PER_NODE_DEFAULT="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$NPROC_PER_NODE_DEFAULT}"

# Run training
# FSDP: Direct Python call (LlamaFactory handles torchrun internally)
# DeepSpeed: Use torchrun explicitly
if [ "$USE_DEEPSPEED" = true ]; then
    torchrun \
      --standalone \
      --nproc_per_node="$NPROC_PER_NODE" \
      -m llamafactory.cli train "$CONFIG_PATH" "$@" 2>&1 | tee -a "$LOG_FILE"
else
    "$PYTHON_BIN" -m llamafactory.cli train "$CONFIG_PATH" "$@" 2>&1 | tee -a "$LOG_FILE"
fi
exit_code=${PIPESTATUS[0]}
exit ${exit_code:-0}
