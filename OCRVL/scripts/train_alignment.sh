#!/bin/bash
# OCRVL Alignment Training - Unified Script
#
# Supports:
# - Fresh training or resume from checkpoint
# - LoRA + Connectors (default) or Connectors only
# - Flexible dataset and GPU configuration
# - Checkpoint evaluation (enabled by default for training transparency)
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
#
# Checkpoint Evaluation:
#   - Automatically runs VQA inference on 10 fixed samples during each checkpoint save
#   - Results saved in: {checkpoint_dir}/eval_results/
#   - Provides training transparency without full benchmark overhead

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
DATASET_TYPE="${DATASET_TYPE:-blip3o}"      # Dataset type: blip3o or doclaynet
BLIP3O_DATASET="${BLIP3O_DATASET:-long}"    # Dataset variant: short, long, 60k, mixed
DATASET_PCT="${DATASET_PCT:-0.05}"          # Dataset percentage (0.05 = 5%)
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MIX_DOCLAYNET="${MIX_DOCLAYNET:-true}"      # Mix DocLayNet with BLIP3o (default: true for alignment)

# DocLayNet configuration
DOCLAYNET_SPLIT="${DOCLAYNET_SPLIT:-train}"  # DocLayNet split: train, val, test
DOCLAYNET_DATA_DIR="${DOCLAYNET_DATA_DIR:-/share/project/xiyan/huggingface/docling-project/DocLayNet}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"      # Per-GPU batch size
GRAD_ACCUM="${GRAD_ACCUM:-4}"      # Gradient accumulation steps
LR="${LR:-1e-3}"                   # Learning rate (1e-3 for alignment, matches LLaVA)
LR_MIN="${LR_MIN:-1e-4}"           # Minimum LR for cosine annealing (10% of peak, prevents decay to zero)
USE_LR_SCHEDULER="${USE_LR_SCHEDULER:-false}"  # Constant LR is standard for alignment (set to 'true' for cosine decay)
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"  # Enable to save ~30-40% memory (~20% slower)

# System configuration
NUM_GPUS="${NUM_GPUS:-8}"          # Number of GPUs to use
GPU_IDS="${GPU_IDS:-}"             # GPU device IDs (auto-generated if not set)
MASTER_PORT="${MASTER_PORT:-29500}"

# Output directory (optional override)
OUTPUT_DIR="${OUTPUT_DIR:-}"       # If set, overrides auto-generated output directory

# ============================================================================
# Derived Configuration
# ============================================================================

# Auto-generate OUTPUT_DIR if not set (so we know where to write torchrun.log)
if [ -z "$OUTPUT_DIR" ]; then
    from datetime import datetime
    timestamp=$(date '+%Y%m%d_%H%M%S')
    if [ "$DATASET_TYPE" = "llava" ]; then
        dataset_name="llava"
    elif [ "$DATASET_TYPE" = "blip3o" ]; then
        dataset_name="$BLIP3O_DATASET"
    elif [ "$DATASET_TYPE" = "thinking" ]; then
        dataset_name="thinking"
    elif [ "$DATASET_TYPE" = "doclaynet" ]; then
        dataset_name="doclaynet"
    else
        dataset_name="custom"
    fi
    OUTPUT_DIR="OCRVL/checkpoints/alignment_${dataset_name}_${timestamp}"
fi

# Create output directory upfront
mkdir -p "$OUTPUT_DIR"

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

# Auto-adjust LR for LoRA (only triggers if LR set to 4e-4)
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
if [ "$DATASET_TYPE" = "doclaynet" ]; then
    echo "  Type: DocLayNet only"
    echo "  Split: $DOCLAYNET_SPLIT"
    echo "  Data dir: $DOCLAYNET_DATA_DIR"
    echo "  Epochs: $NUM_EPOCHS"

    # Calculate dataset size
    if [ "$DOCLAYNET_SPLIT" = "train" ]; then
        DATASET_SIZE=69375
    elif [ "$DOCLAYNET_SPLIT" = "val" ]; then
        DATASET_SIZE=6489
    else
        DATASET_SIZE=4999
    fi
    echo "  Total samples: $DATASET_SIZE"
elif [ "$DATASET_TYPE" = "blip3o" ] && [ "$MIX_DOCLAYNET" = "true" ]; then
    echo "  Type: Mixed (BLIP3o + DocLayNet)"
    echo "  BLIP3o variant: $BLIP3O_DATASET"
    echo "  BLIP3o percentage: $(echo "$DATASET_PCT * 100" | bc)%"
    echo "  DocLayNet split: $DOCLAYNET_SPLIT"
    echo "  Epochs: $NUM_EPOCHS (iterates through both datasets)"
else
    echo "  Type: BLIP3o only"
    echo "  BLIP3o variant: $BLIP3O_DATASET"
    echo "  Sample percentage: $(echo "$DATASET_PCT * 100" | bc)%"
    echo "  Epochs: $NUM_EPOCHS"
fi
echo ""
echo "Hyperparameters:"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Number of GPUs: $NUM_GPUS"
echo "  GPU devices: $GPU_IDS"
echo "  Effective batch size: $EFFECTIVE_BATCH"
if [ "$USE_LR_SCHEDULER" = "true" ]; then
    echo "  Learning rate: $LR → $LR_MIN (cosine decay)"
else
    echo "  Learning rate: $LR (constant)"
fi
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
echo ""
echo "Checkpoint Evaluation:"
echo "  Enabled by default - runs VQA inference on 10 fixed samples per checkpoint"
echo "  Results saved in: {checkpoint_dir}/eval_results/"
echo "========================================================================"
echo ""

# ============================================================================
# Build Command
# ============================================================================

TRAIN_ARGS=(
    --stage alignment
    --dataset-type "$DATASET_TYPE"
    --num_epochs "$NUM_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --num_workers 4
    --log_interval "$LOG_INTERVAL"
    --save_interval "$SAVE_INTERVAL"
)

# Add dataset-specific arguments
if [ "$DATASET_TYPE" = "doclaynet" ]; then
    TRAIN_ARGS+=(
        --doclaynet_split "$DOCLAYNET_SPLIT"
        --doclaynet_data_dir "$DOCLAYNET_DATA_DIR"
    )
elif [ "$DATASET_TYPE" = "blip3o" ]; then
    TRAIN_ARGS+=(
        --blip3o_dataset "$BLIP3O_DATASET"
        --blip3o_sample_percentage "$DATASET_PCT"
    )
    # Add DocLayNet mixing if enabled
    if [ "$MIX_DOCLAYNET" = "true" ]; then
        TRAIN_ARGS+=(
            --blip3o_mix_doclaynet
            --doclaynet_split "$DOCLAYNET_SPLIT"
            --doclaynet_data_dir "$DOCLAYNET_DATA_DIR"
        )
    fi
fi

# Add LR scheduler if enabled (disabled by default for alignment)
if [ "$USE_LR_SCHEDULER" = "true" ]; then
    TRAIN_ARGS+=(
        --use_lr_scheduler
        --lr_scheduler_min_lr "$LR_MIN"
        --warmup_ratio "$WARMUP_RATIO"
    )
fi

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

# Add gradient checkpointing if enabled
if [ "$USE_GRADIENT_CHECKPOINTING" = "true" ]; then
    TRAIN_ARGS+=(--use_gradient_checkpointing)
fi

# Add SwanLab if enabled
if [ "$USE_SWANLAB" = "true" ]; then
    TRAIN_ARGS+=(--use_swanlab)
fi

# Add output directory if specified
if [ -n "$OUTPUT_DIR" ]; then
    TRAIN_ARGS+=(--output_dir "$OUTPUT_DIR")
fi

# Enable checkpoint evaluation by default for training transparency
TRAIN_ARGS+=(--enable_checkpoint_eval)

# ============================================================================
# Start Training
# ============================================================================

echo "Starting training... (torchrun output will be captured)"
echo "Torchrun log will be saved to: $OUTPUT_DIR/torchrun.log"
echo ""

# Run training with stderr redirected to capture all exceptions
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    OCRVL/train.py \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/torchrun.log"

# Store exit code before any other commands
TRAINING_EXIT_CODE=$?

# ============================================================================
# Post-Training
# ============================================================================

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Training completed successfully!"
    echo "========================================================================"

    # Find latest checkpoint
    LATEST_CHECKPOINT=$(ls -td OCRVL/checkpoints/alignment_* 2>/dev/null | head -1)

    if [ -n "$LATEST_CHECKPOINT" ]; then
        FINAL_STEP="$LATEST_CHECKPOINT/step_latest"
        echo ""
        echo "Latest checkpoint: $LATEST_CHECKPOINT"
        if [ -d "$FINAL_STEP" ]; then
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
