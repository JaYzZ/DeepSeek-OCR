#!/bin/bash
# OCRVL Instruction Tuning - LLaVA-Instruct-150K
#
# Continues training from alignment checkpoint on LLaVA-Instruct-150K for VQA capability.
#
# Input format (randomized):
#   - 50%: <Real Image> + <Rendered Question> -> Answer
#   - 50%: <Rendered Question> + <Real Image> -> Answer
#
# Checkpoint Evaluation:
#   - Enabled by default: runs VQA inference on 10 fixed samples per checkpoint
#   - Results saved in: {checkpoint_dir}/eval_results/
#   - Provides training transparency without full benchmark overhead
#
# Usage Examples:
#   # Continue from alignment checkpoint with LoRA
#   RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 bash OCRVL/scripts/train_instruction.sh
#
#   # Start fresh with LoRA (not recommended)
#   bash OCRVL/scripts/train_instruction.sh
#
#   # Continue without LoRA (full LLM training - requires more memory)
#   LORA=0 RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 bash OCRVL/scripts/train_instruction.sh

set -e  # Exit on error
PROJECT_ROOT="${ROOT_DIR:-/share/project/xiyan}"

# ============================================================================
# Configuration - Override with environment variables
# ============================================================================

# Resume configuration
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Path to checkpoint directory (e.g., step_786)
RESUME_TRAINING_STATE="${RESUME_TRAINING_STATE:-false}"  # Resume optimizer/scaler/step (default: false, only load weights)

# LoRA configuration (enabled by default for instruction tuning)
LORA="${LORA:-1}"  # 1 = enabled (default), 0 = disabled
USE_LORA=$([ "$LORA" = "1" ] && echo "true" || echo "false")
LORA_R="${LORA_R:-8}"           # LoRA rank
LORA_ALPHA="${LORA_ALPHA:-16}"  # LoRA alpha (typically 2x rank)
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# LLaVA-Instruct dataset paths (default to 665K mix with proper images/ directory)
LLAVA_JSON_PATH="${LLAVA_JSON_PATH:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json}"
LLAVA_IMAGE_DIR="${LLAVA_IMAGE_DIR:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/images}"
LLAVA_IMAGE_INDEX="${LLAVA_IMAGE_INDEX:-}"  # Optional: path to prebuilt image index JSON

# Dataset configuration
NUM_EPOCHS="${NUM_EPOCHS:-3}"       # Default 3 epochs for instruction tuning
RENDER="${RENDER:-1}"  # 1 = render questions as images (default), 0 = text prompts

# Training hyperparameters (optimized for LoRA instruction tuning)
BATCH_SIZE="${BATCH_SIZE:-12}"      # Per-GPU batch size
GRAD_ACCUM="${GRAD_ACCUM:-4}"       # Gradient accumulation steps (effective batch: 384 on 8 GPUs)
LR="${LR:-2e-4}"                    # Learning rate (2e-4 for LoRA instruction tuning)
LR_MIN="${LR_MIN:-2e-5}"            # Minimum LR for cosine annealing (10% of peak, prevents decay to zero)
USE_LR_SCHEDULER="${USE_LR_SCHEDULER:-false}"  # Constant LR is standard for LoRA (set to 'true' for cosine decay)
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"  # Enable to save ~30-40% memory (~20% slower)

# System configuration
NUM_GPUS="${NUM_GPUS:-8}"           # Number of GPUs to use (scaled from 4 to 8 for faster training)
GPU_IDS="${GPU_IDS:-}"              # GPU device IDs (auto-generated if not set)
MASTER_PORT="${MASTER_PORT:-29501}"

# Output directory (optional override)
OUTPUT_DIR="${OUTPUT_DIR:-}"        # If set, overrides auto-generated output directory

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

# Auto-generate OUTPUT_DIR if not set (so we know where to write torchrun.log)
if [ -z "$OUTPUT_DIR" ]; then
    timestamp=$(date '+%Y%m%d_%H%M%S')
    OUTPUT_DIR="OCRVL/checkpoints/llava_${timestamp}"
fi

# Create output directory upfront
mkdir -p "$OUTPUT_DIR"

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
else
    echo "⚠️  Warning: No checkpoint specified. Starting fresh is not recommended for instruction tuning."
    echo "   Consider using RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786"
fi

# Check LLaVA dataset files
if [ ! -f "$LLAVA_JSON_PATH" ]; then
    echo "❌ Error: LLaVA JSON not found at $LLAVA_JSON_PATH"
    exit 1
fi

if [ ! -d "$LLAVA_IMAGE_DIR" ]; then
    echo "❌ Error: LLaVA image directory not found at $LLAVA_IMAGE_DIR"
    exit 1
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
echo "OCRVL Instruction Tuning - LLaVA-Instruct-150K"
echo "========================================================================"
echo ""
echo "Mode:"
if [ "$USE_LORA" = "true" ]; then
    echo "  Training method: LoRA + Connectors"
    echo "  LoRA rank (r): $LORA_R"
    echo "  LoRA alpha: $LORA_ALPHA"
    echo "  LoRA dropout: $LORA_DROPOUT"
    echo "  Trainable params: ~22M (9M LoRA on LLM + 13M connectors)"
else
    echo "  Training method: Full LLM + Connectors"
    echo "  Trainable params: ~2B (full LLM + connectors)"
fi
echo "  Status: $([ -n "$RESUME_CHECKPOINT" ] && echo "Continue from checkpoint" || echo "Fresh training")"
echo ""
echo "Dataset:"
echo "  LLaVA v1.5 Mix-665K (full dataset, ~665K conversations)"
echo "  JSON: $LLAVA_JSON_PATH"
echo "  Images: $LLAVA_IMAGE_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Render mode: $([ "$RENDER" = "1" ] && echo "RENDER=1 (questions as images)" || echo "RENDER=0 (text prompts)")"
echo ""
if [ "$RENDER" = "1" ]; then
    echo "Input Format (randomized per sample):"
    echo "  50%: <Real Image> + <Rendered Question> -> Answer"
    echo "  50%: <Rendered Question> + <Real Image> -> Answer"
else
    echo "Input Format:"
    echo "  <Real Image> + Text Question -> Answer"
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
echo "  ~30-50 samples/s (depends on hardware and batch size)"
echo "  ~2-3 hours per epoch (for full LLaVA-Instruct-150K)"
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
    --dataset-type llava
    --llava_json_path "$LLAVA_JSON_PATH"
    --llava_image_dir "$LLAVA_IMAGE_DIR"
    --num_epochs "$NUM_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --num_workers 4
    --log_interval "$LOG_INTERVAL"
    --save_interval "$SAVE_INTERVAL"
)

# Add LR scheduler if enabled (disabled by default for LoRA)
if [ "$USE_LR_SCHEDULER" = "true" ]; then
    TRAIN_ARGS+=(
        --use_lr_scheduler
        --lr_scheduler_min_lr "$LR_MIN"
        --warmup_ratio "$WARMUP_RATIO"
    )
fi

# Add render mode (default is RENDER=1, which uses default --llava_render_questions)
if [ "$RENDER" = "0" ]; then
    TRAIN_ARGS+=(--no-llava_render_questions)
fi

# Add image index if provided
if [ -n "$LLAVA_IMAGE_INDEX" ]; then
    TRAIN_ARGS+=(--llava_image_index "$LLAVA_IMAGE_INDEX")
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
    echo "✓ Instruction tuning completed successfully!"
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
            echo "To evaluate on benchmarks:"
            echo "  bash OCRVL/evaluation/run_benchmarks.sh -b realworldqa,mmmu,mathvision $FINAL_STEP"
        fi
    fi
else
    echo ""
    echo "========================================================================"
    echo "❌ Training failed. Check logs above for errors."
    echo "========================================================================"
    exit 1
fi
