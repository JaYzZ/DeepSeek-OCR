#!/bin/bash
# Phase 3: Thinking Training with LLaVA-CoT
#
# Trains thinking projection MLP to align hidden states with OCR-encoded reasoning.
# Uses Chain-of-Thought data from LLaVA-CoT-100K dataset.
#
# Key Features:
#   - Dual loss: MSE for thinking tokens + CE for answer tokens
#   - Thinking projection: hidden_dim (4096) → latent_dim (1280)
#   - Loads from Phase 2 checkpoint (instruction-tuned model)
#
# Usage:
#   RESUME_CHECKPOINT=path/to/phase2/checkpoint OUTPUT_DIR=path/to/output bash train_thinking.sh
#
# Environment Variables:
#   RESUME_CHECKPOINT  - Path to Phase 2 checkpoint (required)
#   OUTPUT_DIR         - Output directory for checkpoints
#   NUM_GPUS           - Number of GPUs (default: 4)
#   GPU_IDS            - GPU IDs to use (default: 0,1,2,3)
#   LORA               - Use LoRA for LLM (default: 1)
#   LR                 - Learning rate (default: 1e-4)
#   NUM_EPOCHS         - Number of epochs (default: 1)
#   THINKING_LOSS_WEIGHT - Weight for thinking loss (default: 1.0)
#   MAX_SAMPLES        - Max samples to load (default: 100000)

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Required parameters
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-OCRVL/checkpoints/thinking_$(date '+%Y%m%d_%H%M%S')}"

# GPU configuration
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

# Training configuration
LORA="${LORA:-1}"                           # Use LoRA on LLM (recommended)
LR="${LR:-1e-4}"                            # Learning rate for thinking projection
NUM_EPOCHS="${NUM_EPOCHS:-1}"               # 1 epoch for 100K samples
THINKING_LOSS_WEIGHT="${THINKING_LOSS_WEIGHT:-1.0}"  # Weight for thinking alignment loss
MAX_SAMPLES="${MAX_SAMPLES:-100000}"        # Use all samples (set lower for testing)

# Batch configuration
BATCH_SIZE="${BATCH_SIZE:-4}"               # Per-GPU batch size (4 for 24GB GPU)
GRAD_ACCUM="${GRAD_ACCUM:-4}"               # Gradient accumulation steps
# Effective batch size: 4 GPUs × 4 batch × 4 accum = 64

# Learning rate schedule
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"   # Warmup over 500 steps
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"      # Cosine annealing
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"  # Enable to save ~30-40% memory (~20% slower)

# Checkpoint configuration
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"      # Save every 1000 steps
LOG_INTERVAL="${LOG_INTERVAL:-100}"         # Log every 100 steps

# Dataset configuration
THINKING_JSONL="${THINKING_JSONL:-/share/project/xiyan/huggingface/Xkev/LLaVA-CoT-100k/train.jsonl}"
THINKING_IMAGE_DIR="${THINKING_IMAGE_DIR:-/share/project/xiyan/huggingface}"

# Model paths
DPSK_MODEL_PATH="${DPSK_MODEL_PATH:-deepseek-ai/DeepSeek-OCR}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"

# ============================================================================
# Validation
# ============================================================================

if [ -z "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: RESUME_CHECKPOINT is required!"
    echo "Usage: RESUME_CHECKPOINT=path/to/checkpoint bash $0"
    exit 1
fi

if [ ! -d "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $RESUME_CHECKPOINT"
    exit 1
fi

if [ ! -f "$THINKING_JSONL" ]; then
    echo "ERROR: LLaVA-CoT dataset not found: $THINKING_JSONL"
    exit 1
fi

# ============================================================================
# Info
# ============================================================================

echo "========================================================================"
echo "Phase 3: Thinking Training"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Resume from:       $RESUME_CHECKPOINT"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Dataset:           $THINKING_JSONL"
echo "  Max samples:       $MAX_SAMPLES"
echo ""
echo "Model:"
echo "  DPSK OCR:          $DPSK_MODEL_PATH"
echo "  Qwen3-VL:          $QWEN_MODEL_PATH"
echo "  LoRA:              $LORA"
echo ""
echo "Training:"
echo "  GPUs:              $NUM_GPUS ($GPU_IDS)"
echo "  Batch size:        $BATCH_SIZE (per GPU)"
echo "  Gradient accum:    $GRAD_ACCUM steps"
echo "  Effective batch:   $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))"
echo "  Learning rate:     $LR"
echo "  LR warmup:         $LR_WARMUP_STEPS steps"
echo "  LR scheduler:      $LR_SCHEDULER"
echo "  Epochs:            $NUM_EPOCHS"
echo "  Thinking loss wt:  $THINKING_LOSS_WEIGHT"
echo ""
echo "Checkpoints:"
echo "  Save interval:     $SAVE_INTERVAL steps"
echo "  Log interval:      $LOG_INTERVAL steps"
echo "========================================================================"
echo ""

# ============================================================================
# Launch Training
# ============================================================================

# Build LoRA args
LORA_ARGS=""
if [ "$LORA" -eq 1 ]; then
    LORA_ARGS="--use_lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.05"
    echo "LoRA enabled: r=64, alpha=128, dropout=0.05"
else
    echo "LoRA disabled: training full model"
fi

# Build gradient checkpointing args
GRAD_CKPT_ARGS=""
if [ "$USE_GRADIENT_CHECKPOINTING" = "true" ]; then
    GRAD_CKPT_ARGS="--use_gradient_checkpointing"
    echo "Gradient checkpointing enabled: saves ~30-40% memory, ~20% slower"
fi

# Estimate total steps
SAMPLES_PER_EPOCH=$MAX_SAMPLES
EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))
STEPS_PER_EPOCH=$((SAMPLES_PER_EPOCH / EFFECTIVE_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))

echo ""
echo "Training estimates:"
echo "  Samples/epoch:     $SAMPLES_PER_EPOCH"
echo "  Steps/epoch:       $STEPS_PER_EPOCH"
echo "  Total steps:       $TOTAL_STEPS"
echo "  Training time:     ~$((TOTAL_STEPS * 3 / 60)) minutes (est. 3s/step)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Setup torchrun logging
TORCHRUN_LOG="${OUTPUT_DIR}/torchrun.log"

# Launch distributed training
echo "Starting training..."
echo "Torchrun output (including exceptions) will be logged to: $TORCHRUN_LOG"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    OCRVL/train.py \
    --dataset-type thinking \
    --thinking_jsonl_path "$THINKING_JSONL" \
    --thinking_image_dir "$THINKING_IMAGE_DIR" \
    --thinking_loss_weight "$THINKING_LOSS_WEIGHT" \
    --thinking_max_samples "$MAX_SAMPLES" \
    --dpsk_model_path "$DPSK_MODEL_PATH" \
    --qwen_model_path "$QWEN_MODEL_PATH" \
    --resume_checkpoint "$RESUME_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lr "$LR" \
    --lr_warmup_steps "$LR_WARMUP_STEPS" \
    --lr_scheduler "$LR_SCHEDULER" \
    --save_interval "$SAVE_INTERVAL" \
    --log_interval "$LOG_INTERVAL" \
    --max_grad_norm 1.0 \
    --use_amp \
    --seed 42 \
    $LORA_ARGS \
    $GRAD_CKPT_ARGS 2>&1 | tee "$TORCHRUN_LOG"

echo ""
echo "========================================================================"
echo "✓ Thinking Training Completed!"
echo "========================================================================"
echo ""
echo "Final checkpoint: $OUTPUT_DIR/step_latest"
echo ""
echo "To evaluate:"
echo "  bash OCRVL/evaluation/run_benchmarks.sh $OUTPUT_DIR/step_latest"
echo ""
echo "To continue to next stage:"
echo "  RESUME_CHECKPOINT=$OUTPUT_DIR/step_latest bash OCRVL/scripts/train_next_stage.sh"
echo "========================================================================"
