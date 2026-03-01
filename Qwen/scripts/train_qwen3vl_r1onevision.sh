#!/bin/bash
# Qwen3VL R1-OneVision SFT Training with Latent Supervision
#
# This script trains Qwen3VL-2B-Thinking on R1-OneVision dataset using:
# - Pre-encoded vision features (no encoding during training)
# - Latent injection at <latent> positions
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

# ============================================================================
# Export config values as environment variables for Python callbacks
# ============================================================================
# Helper to read config values
_get_config() {
    grep -E "^${1}:" "$CONFIG_PATH" 2>/dev/null | head -n 1 | awk '{print $2}'
}

# Export env vars that Python code needs
export QWEN3VL_LATENT_SUPERVISION=$(_get_config "qwen3vl_latent_supervision" || echo "1")
export QWEN3VL_LATENT_TOKEN_ID=$(_get_config "qwen3vl_latent_token_id" || echo "151669")
export QWEN3VL_THINKING_START_ID=$(_get_config "qwen3vl_thinking_start_id" || echo "151667")
export QWEN3VL_THINKING_END_ID=$(_get_config "qwen3vl_thinking_end_id" || echo "151668")
export QWEN3VL_LOSS_TYPE=$(_get_config "qwen3vl_loss_type" || echo "vae+ot+pre_think_mse")
export QWEN3VL_MATCH_STRATEGY=$(_get_config "qwen3vl_match_strategy" || echo "truncate")
export QWEN3VL_THINKING_LOSS_WEIGHT=$(_get_config "qwen3vl_thinking_loss_weight" || echo "1.0")
export QWEN3VL_MAX_NEW_TOKENS=$(_get_config "qwen3vl_max_new_tokens" || echo "2048")
export QWEN3VL_VAE_INTERMEDIATE_SIZE=$(_get_config "qwen3vl_vae_intermediate_size" || echo "512")
export QWEN3VL_CURRICULUM_ENABLE=$(_get_config "qwen3vl_curriculum_enable" || echo "1")
export QWEN3VL_CURRICULUM_EPOCHS=$(_get_config "qwen3vl_curriculum_epochs" || echo "0,1,2")
export QWEN3VL_CURRICULUM_LOSS_TYPES=$(_get_config "qwen3vl_curriculum_loss_types" || echo "vae+pre_think_mse,vae+pre_think_mse,vae+ot+pre_think_mse")
export QWEN3VL_CURRICULUM_WEIGHTS=$(_get_config "qwen3vl_curriculum_weights" || echo "1.0,1.0,1.0")
export QWEN3VL_CURRICULUM_LATENT_STEP_CE=$(_get_config "qwen3vl_curriculum_latent_step_ce" || echo "1,1,1")
export DATALOADER_NUM_WORKERS=$(_get_config "dataloader_num_workers" || echo "4")

# Debug settings
export QWEN3VL_DEBUG_FORWARD="${QWEN3VL_DEBUG_FORWARD:-0}"

export TRANSPARENT_EVAL_MAX_NEW_TOKENS=$(_get_config "eval_max_new_tokens" || echo "512")
export RUN_BACKFILL=1

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

# Set max_new_tokens for generation (prevents max_length crash when input > cutoff_len)

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


# Display configuration
TRAINING_TYPE="R1-OneVision SFT Training (Latent Supervision)"

LOSS_TYPE=$(_get_config "qwen3vl_loss_type" || echo "vae+ot+pre_think_mse")
LOSS_WEIGHT=$(_get_config "qwen3vl_thinking_loss_weight" || echo "1.0")
LOSS_TYPE_DISPLAY="$LOSS_TYPE (weight: $LOSS_WEIGHT)"

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
echo "  - Backfill after training: $RUN_BACKFILL"
echo "  - Backfill max new tokens: $TRANSPARENT_EVAL_MAX_NEW_TOKENS"
echo ""
echo "Curriculum Learning:"
echo "  - Enabled: $QWEN3VL_CURRICULUM_ENABLE"
echo "  - Epochs: $QWEN3VL_CURRICULUM_EPOCHS"
echo "  - Loss types: $QWEN3VL_CURRICULUM_LOSS_TYPES"
echo "  - Weights: $QWEN3VL_CURRICULUM_WEIGHTS"
echo "  - Latent step CE: $QWEN3VL_CURRICULUM_LATENT_STEP_CE"
echo ""
echo "Special Tokens:"
echo "  - <latent>: $QWEN3VL_LATENT_TOKEN_ID"
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

# ============================================================================
# Post-training: Run backfill transparent eval on all checkpoints
# Uses all training GPUs in parallel (round-robin, M jobs at a time)
# ============================================================================
if [ "$exit_code" -eq 0 ] && [ "$RUN_BACKFILL" = "1" ]; then
    echo ""
    echo "========================================================================"
    echo "Training completed! Running backfill transparent eval on all checkpoints..."
    echo "========================================================================"

    # Get list of GPUs from training
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPU_ARRAY[@]}
    echo "Using $NUM_GPUS GPUs for backfill: ${GPU_ARRAY[*]}"

    # Find all checkpoints that need backfill
    CHECKPOINT_LIST=()
    for CHECKPOINT in $(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V); do
        CHECKPOINT_NAME=$(basename "$CHECKPOINT")

        # Skip if already has results
        if [ -d "$CHECKPOINT/eval_results" ] && [ "$(ls "$CHECKPOINT/eval_results"/backfill_*.json 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  Skipping $CHECKPOINT_NAME - already has backfill results"
            continue
        fi

        CHECKPOINT_LIST+=("$CHECKPOINT_NAME")
    done

    NUM_CHECKPOINTS=${#CHECKPOINT_LIST[@]}

    if [ "$NUM_CHECKPOINTS" -eq 0 ]; then
        echo "No checkpoints need backfill"
    else
        echo "Found $NUM_CHECKPOINTS checkpoints to backfill (running $NUM_GPUS at a time)"

        # Run in batches of NUM_GPUS
        for ((i=0; i<NUM_CHECKPOINTS; i+=NUM_GPUS)); do
            # Launch jobs for this batch (one per GPU)
            for ((j=0; j<NUM_GPUS && i+j<NUM_CHECKPOINTS; j++)); do
                idx=$((i+j))
                checkpoint_name="${CHECKPOINT_LIST[$idx]}"
                gpu_id="${GPU_ARRAY[$j]}"

                echo "  Starting $checkpoint_name on GPU $gpu_id..."

                # Run backfill in background with specific GPU
                (
                    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" Qwen/scripts/backfill_transparent_eval.py \
                        --checkpoint_dir "$OUTPUT_DIR" \
                        --checkpoint "$checkpoint_name" \
                        --gpu_memory_utilization 0.9 \
                        2>&1 >> "$OUTPUT_DIR/backfill_all_checkpoints.log"

                    # Check result
                    if ls "$OUTPUT_DIR/${checkpoint_name}/eval_results"/backfill_*.json >/dev/null 2>&1; then
                        echo "  $checkpoint_name: Completed!" >> "$OUTPUT_DIR/backfill_all_checkpoints.log"
                    else
                        echo "  $checkpoint_name: Failed!" >> "$OUTPUT_DIR/backfill_all_checkpoints.log"
                    fi
                ) &
            done

            # Wait for this batch to finish before starting next batch
            wait
        done

        echo ""
        echo "Backfill complete! Results saved to $OUTPUT_DIR/eval_results/"
    fi
fi

exit ${exit_code:-0}
