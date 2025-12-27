#!/bin/bash
# OCRVL Alignment Training - Unified Script
#
# Supports:
# - Fresh training or resume from checkpoint
# - LoRA + Connectors (default) or Connectors only
# - Flexible dataset and GPU configuration
#
# Usage Examples:
#   # Fresh training with LoRA + Connectors (default)
#   bash OCRVL/scripts/train_alignment.sh
#
#   # Fresh training with connectors only (no LoRA)
#   LORA=0 bash OCRVL/scripts/train_alignment.sh
#
#   # Load weights from checkpoint, start fresh from step 0 (default)
#   RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 bash OCRVL/scripts/train_alignment.sh
#
#   # Fully resume training (weights + optimizer + step counter)
#   RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 RESUME_TRAINING_STATE=true bash OCRVL/scripts/train_alignment.sh
#
#   # Resume without LoRA (even if checkpoint has LoRA)
#   LORA=0 RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 bash OCRVL/scripts/train_alignment.sh

set -e  # Exit on error

# ============================================================================
# Configuration - Override with environment variables
# ============================================================================

# Resume configuration
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Path to checkpoint directory (e.g., step_786)
RESUME_TRAINING_STATE="${RESUME_TRAINING_STATE:-false}"  # Resume optimizer/scaler/step (default: false, only load weights)

# LoRA configuration (enabled by default - use LORA=0 to disable)
LORA="${LORA:-1}"  # 1 = enabled (default), 0 = disabled
USE_LORA=$([ "$LORA" = "1" ] && echo "true" || echo "false")
LORA_R="${LORA_R:-8}"           # LoRA rank
LORA_ALPHA="${LORA_ALPHA:-16}"  # LoRA alpha (typically 2x rank)
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Data configuration
BLIP3O_DATASET="${BLIP3O_DATASET:-long}"  # Dataset variant: short, long, 60k, mixed
DATASET_PCT="${DATASET_PCT:-0.05}"        # Dataset percentage (0.05 = 5%)
NUM_EPOCHS="${NUM_EPOCHS:-1}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"      # Per-GPU batch size
GRAD_ACCUM="${GRAD_ACCUM:-32}"     # Gradient accumulation steps
LR="${LR:-4e-4}"                   # Learning rate (use 2e-4 for LoRA or continued training)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

# System configuration
NUM_GPUS="${NUM_GPUS:-4}"          # Number of GPUs to use
GPU_IDS="${GPU_IDS:-}"             # GPU device IDs (auto-generated if not set)
MASTER_PORT="${MASTER_PORT:-29500}"

# Logging
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
USE_SWANLAB="${USE_SWANLAB:-false}"

# ============================================================================
# Derived Configuration
# ============================================================================

# Auto-generate GPU_IDS if not set
if [ -z "$GPU_IDS" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

# Auto-adjust LR for LoRA (default is LoRA enabled with 2e-4)
if [ "$USE_LORA" = "true" ] && [ "$LR" = "4e-4" ]; then
    echo "ℹ️  Using LoRA: Automatically adjusting LR to 2e-4 for stability"
    LR="2e-4"
fi

# ============================================================================
# Validation
# ============================================================================

# Check if resuming from checkpoint
if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ ! -d "$RESUME_CHECKPOINT" ]; then
        echo "❌ Error: Resume checkpoint directory not found at $RESUME_CHECKPOINT"
        exit 1
    fi

    # Check if connectors.pt exists
    if [ ! -f "$RESUME_CHECKPOINT/connectors.pt" ]; then
        echo "❌ Error: connectors.pt not found in $RESUME_CHECKPOINT"
        exit 1
    fi

    # Check if LoRA adapters exist when USE_LORA=true
    if [ "$USE_LORA" = "true" ] && [ ! -d "$RESUME_CHECKPOINT/lora_adapters" ]; then
        echo "ℹ️  LoRA adapters not found in checkpoint, will initialize new LoRA layers"
    fi

    echo "✓ Resume checkpoint verified: $RESUME_CHECKPOINT"
fi

# Check GPU availability
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "⚠️  Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    echo "    Adjusting to use $AVAILABLE_GPUS GPUs"
    NUM_GPUS=$AVAILABLE_GPUS
    EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

    # Auto-adjust GPU_IDS
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

# ============================================================================
# Display Configuration
# ============================================================================

echo "========================================================================"
echo "OCRVL Alignment Training"
echo "========================================================================"
echo ""
echo "Mode:"
if [ "$USE_LORA" = "true" ]; then
    echo "  Training method: LoRA + Connectors (Default)"
    echo "  LoRA rank (r): $LORA_R"
    echo "  LoRA alpha: $LORA_ALPHA"
    echo "  LoRA dropout: $LORA_DROPOUT"
    echo "  Trainable params: ~16M (3M LoRA on LLM + 13M connectors)"
else
    echo "  Training method: Connectors Only (LoRA disabled)"
    echo "  Trainable params: ~13M (connectors only)"
fi
echo "  Status: $([ -n "$RESUME_CHECKPOINT" ] && echo "Resume from checkpoint" || echo "Fresh training")"
echo ""
echo "Dataset:"
echo "  BLIP3o variant: $BLIP3O_DATASET"
echo "  Sample percentage: $(echo "$DATASET_PCT * 100" | bc)%"
echo "  Epochs: $NUM_EPOCHS"
echo ""
echo "Hyperparameters:"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Number of GPUs: $NUM_GPUS"
echo "  GPU devices: $GPU_IDS"
echo "  Effective batch size: $EFFECTIVE_BATCH"
echo "  Learning rate: $LR"
echo "  Weight decay: $WEIGHT_DECAY"
echo ""
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume Configuration:"
    echo "  Checkpoint: $RESUME_CHECKPOINT"
    if [ "$RESUME_TRAINING_STATE" = "true" ]; then
        echo "  Mode: Full resume (weights + optimizer + step counter)"
    else
        echo "  Mode: Weights only (connectors/LoRA, starting from step 0)"
    fi
    echo ""
fi
echo "Expected Performance:"
echo "  ~90-130 samples/s (depends on hardware and batch size)"
echo "  ~4-5 hours per epoch (for 5% of BLIP3o long dataset)"
echo "========================================================================"
echo ""

# ============================================================================
# Build Command
# ============================================================================

TRAIN_ARGS=(
    --stage alignment
    --dataset-type blip3o
    --blip3o_dataset "$BLIP3O_DATASET"
    --blip3o_sample_percentage "$DATASET_PCT"
    --num_epochs "$NUM_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --num_workers 0
    --log_interval "$LOG_INTERVAL"
    --save_interval "$SAVE_INTERVAL"
)

# Add checkpoint resume if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--load_checkpoint "$RESUME_CHECKPOINT")
    # Add resume_training_state flag if enabled
    if [ "$RESUME_TRAINING_STATE" = "true" ]; then
        TRAIN_ARGS+=(--resume_training_state)
    fi
fi

# Add LoRA arguments if enabled
if [ "$USE_LORA" = "true" ]; then
    TRAIN_ARGS+=(
        --use_lora
        --lora_r "$LORA_R"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
    )
fi

# Add SwanLab if enabled
if [ "$USE_SWANLAB" = "true" ]; then
    TRAIN_ARGS+=(--use_swanlab)
fi

# ============================================================================
# Start Training
# ============================================================================

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    OCRVL/train.py \
    "${TRAIN_ARGS[@]}"

# ============================================================================
# Post-Training
# ============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Training completed successfully!"
    echo "========================================================================"

    # Find latest checkpoint
    LATEST_CHECKPOINT=$(ls -td OCRVL/checkpoints/alignment_* 2>/dev/null | head -1)

    if [ -n "$LATEST_CHECKPOINT" ]; then
        FINAL_STEP=$(ls -d "$LATEST_CHECKPOINT"/step_* 2>/dev/null | sort -V | tail -1)
        echo ""
        echo "Latest checkpoint: $LATEST_CHECKPOINT"
        if [ -n "$FINAL_STEP" ]; then
            echo "Final step: $FINAL_STEP"
            echo ""
            echo "To continue training from this checkpoint:"
            if [ "$LORA" = "1" ]; then
                echo "  RESUME_CHECKPOINT=$FINAL_STEP RESUME_TRAINING_STATE=true bash $0"
            else
                echo "  LORA=0 RESUME_CHECKPOINT=$FINAL_STEP RESUME_TRAINING_STATE=true bash $0"
            fi
            echo ""
            echo "To toggle LoRA mode:"
            if [ "$LORA" = "1" ]; then
                echo "  LORA=0 RESUME_CHECKPOINT=$FINAL_STEP bash $0  # Disable LoRA"
            else
                echo "  LORA=1 RESUME_CHECKPOINT=$FINAL_STEP bash $0  # Enable LoRA"
            fi
            echo ""
            echo "To evaluate on benchmarks:"
            echo "  bash OCRVL/evaluation/run_benchmarks.sh $FINAL_STEP"
        fi
    fi
else
    echo ""
    echo "========================================================================"
    echo "❌ Training failed. Check logs above for errors."
    echo "========================================================================"
    exit 1
fi
