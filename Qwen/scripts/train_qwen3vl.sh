#!/bin/bash
# Training script for Qwen3VL native training
# Usage: bash Qwen/scripts/train_qwen3vl.sh <config_file> [loss_type]
#
# Environment variables:
#   LOSS_TYPE         - Loss combination (default: ot)
#                       Syntax: loss1[:weight1]+loss2[:weight2]+...
#                       Available: repa, nce, ot, mse
#   MATCH_STRATEGY    - Sequence length matching for REPA/MSE (default: truncate)
#                       Options: truncate, repeat, interpolate
#   NCE_TEMP          - Temperature for NCE loss (default: 0.07)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_PATH="$1"
LOSS_TYPE="${2:-${LOSS_TYPE:-ot}}"

if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 <config_file> [loss_type]"
    echo ""
    echo "Loss type syntax: loss1[:weight1]+loss2[:weight2]+..."
    echo "Available losses: repa, nce, ot, mse"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/configs/qwen3vl_native_unified_sft.yaml"
    echo "  $0 Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml ot"
    echo "  $0 Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml ot+mse"
    echo "  $0 Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml ot:0.7+mse:0.3"
    echo "  $0 Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml repa:0.5+nce:0.3+ot:0.2"
    echo ""
    echo "Environment variables:"
    echo "  LOSS_TYPE         - Loss combination (default: ot)"
    echo "  MATCH_STRATEGY    - Sequence length matching for REPA/MSE (default: truncate)"
    echo "  NCE_TEMP          - Temperature for NCE loss (default: 0.07)"
    exit 1
fi

# Resolve relative paths
if [[ ! "$CONFIG_PATH" = /* ]]; then
    CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Extract config name for directory
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$REPO_ROOT/checkpoints/$CONFIG_NAME/run_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Qwen3VL Native Training"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Get number of GPUs
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
echo "Using $NUM_GPUS GPUs"
echo ""

# Activate conda environment if specified
if [ -n "$CONDA_ENV" ]; then
    echo "Activating conda environment: $CONDA_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# Set environment variables
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Enable latent supervision if config contains "thinking" in name
if [[ "$CONFIG_NAME" == *"thinking"* ]]; then
    echo "Enabling Qwen3VL latent supervision..."
    export QWEN3VL_LATENT_SUPERVISION=1
    export QWEN3VL_LATENT_DIM=${QWEN3VL_LATENT_DIM:-2048}  # LLM hidden dim (Qwen3VL-2B)
    export QWEN3VL_LATENT_TOKEN_ID=${QWEN3VL_LATENT_TOKEN_ID:-151670}  # <|latent_step|>
    export QWEN3VL_THINKING_START_ID=${QWEN3VL_THINKING_START_ID:-151667}  # <｜text▁begin▁of▁thinking｜>
    export QWEN3VL_THINKING_END_ID=${QWEN3VL_THINKING_END_ID:-151668}  # <｜text▁end▁of▁thinking｜>
    export QWEN3VL_THINKING_LOSS_WEIGHT=${QWEN3VL_THINKING_LOSS_WEIGHT:-1.0}

    # Loss type configuration
    export QWEN3VL_LOSS_TYPE=${LOSS_TYPE:-ot}
    export QWEN3VL_MATCH_STRATEGY=${MATCH_STRATEGY:-truncate}
    export QWEN3VL_NCE_TEMP=${NCE_TEMP:-0.07}

    echo "  QWEN3VL_LATENT_SUPERVISION=1"
    echo "  QWEN3VL_LATENT_DIM=$QWEN3VL_LATENT_DIM (LLM hidden dimension)"
    echo "  QWEN3VL_THINKING_LOSS_WEIGHT=$QWEN3VL_THINKING_LOSS_WEIGHT"
    echo "  QWEN3VL_LOSS_TYPE=$QWEN3VL_LOSS_TYPE"

    # Show additional config based on loss types used
    if [[ "$QWEN3VL_LOSS_TYPE" == *"repa"* ]] || [[ "$QWEN3VL_LOSS_TYPE" == *"mse"* ]]; then
        echo "  QWEN3VL_MATCH_STRATEGY=$QWEN3VL_MATCH_STRATEGY"
    fi
    if [[ "$QWEN3VL_LOSS_TYPE" == *"nce"* ]]; then
        echo "  QWEN3VL_NCE_TEMP=$QWEN3VL_NCE_TEMP"
    fi
    echo ""
fi

# Add output_dir override to command
EXTRA_ARGS="${EXTRA_ARGS} output_dir=$OUTPUT_DIR"

# Run llamafactory training
echo "Starting training..."
echo ""

cd "$REPO_ROOT"

llamafactory-cli train "$CONFIG_PATH" ${EXTRA_ARGS} \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Logs: $OUTPUT_DIR/training.log"
echo "Checkpoints: $OUTPUT_DIR"
echo "=========================================="
