#!/bin/bash
# Launch DPSK-Qwen connector alignment training with BLIP3o dataset
#
# Multi-GPU distributed training using torchrun (8 GPUs).
# This script uses Lumina-DiMOO's cached WebDataset infrastructure for efficient
# sampling from the 27M BLIP3o dataset.
#
# Cache location: /share/project/xiyan/huggingface/cache/datasets
# First run will cache the dataset (takes ~5-10 min), subsequent runs are instant.
#
# Usage:
#   ./OCRVL/scripts/train_blip3o_alignment.sh

set -e

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading environment from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
STAGE="alignment"
OUTPUT_DIR=""  # Will be auto-generated as OCRVL/checkpoints/alignment_60k_{timestamp}
NUM_GPUS=8

# BLIP3o settings (using Lumina-DiMOO infrastructure)
USE_BLIP3O=true
BLIP3O_DATASET="long"          # Options: "short", "long", "60k", "mixed"
                               # - short: Concise captions (~10-20 tokens, 4.8M images)
                               # - long: Detailed captions (~120 tokens, 29.4M images)
                               # - 60k: Curated high-quality dataset (60K images)
                               # - mixed: Combination of short + long (uses MIX_RATIO)
BLIP3O_SAMPLE_PCT="0.5"        # 50% of long caption dataset (~14.7M samples)
                               # For short/long: 0.001=0.1%, 0.01=1%, 0.1=10%, 0.5=50%
BLIP3O_MIX_RATIO=0.5           # For "mixed" only: 50% short, 50% long captions
BLIP3O_USE_IMAGES=true         # Use real images
BLIP3O_IMAGE_CAPTION_RATIO=0.5 # 50% real images, 50% rendered text

# Training settings (optimized for 8× H100 80GB)
NUM_EPOCHS=1                   # Train for 1 epoch
MAX_STEPS=100000               # Max steps limit (safety limit)
BATCH_SIZE=32                  # Per GPU (reduced from 128 to avoid OOM)
GRAD_ACCUM=8                   # Gradient accumulation (increased to maintain effective batch)
LR=4e-4
WEIGHT_DECAY=0.01
LOG_INTERVAL=50
SAVE_INTERVAL=1000
SEED=42

# Model paths (using local mirrors)
QWEN_MODEL="/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Instruct"
DPSK_MODEL="deepseek-ai/DeepSeek-OCR"

echo "========================================================================"
echo "DPSK-Qwen Connector Alignment Training with BLIP3o (Multi-GPU)"
echo "========================================================================"
echo "Stage: $STAGE"
echo "Output: Auto-generated (OCRVL/checkpoints/alignment_{dataset}_{timestamp})"
echo "GPUs: $NUM_GPUS (distributed training with torchrun)"
echo ""
echo "BLIP3o Configuration (Lumina-DiMOO infrastructure):"
echo "  Dataset type: $BLIP3O_DATASET"
if [ -z "$BLIP3O_SAMPLE_PCT" ]; then
    echo "  Sample percentage: 100% (full dataset, using Lumina-DiMOO cache)"
    if [ "$BLIP3O_DATASET" = "short" ]; then
        echo "    (~4,808,616 samples)"
    elif [ "$BLIP3O_DATASET" = "long" ]; then
        echo "    (~29,353,470 samples)"
    elif [ "$BLIP3O_DATASET" = "60k" ]; then
        echo "    (~58,859 samples)"
    fi
else
    echo "  Sample percentage: ${BLIP3O_SAMPLE_PCT}"
    if [ "$BLIP3O_DATASET" = "60k" ]; then
        echo "    (~$(($(echo "$BLIP3O_SAMPLE_PCT * 60000" | bc | cut -d. -f1))) samples from 60k)"
    elif [ "$BLIP3O_DATASET" = "short" ]; then
        echo "    (~$(($(echo "$BLIP3O_SAMPLE_PCT * 4808616" | bc | cut -d. -f1))) samples from 4.8M)"
    else
        echo "    (~$(($(echo "$BLIP3O_SAMPLE_PCT * 29353470" | bc | cut -d. -f1))) samples from 29.4M)"
    fi
fi
if [ "$BLIP3O_DATASET" = "mixed" ]; then
    echo "  Short/Long mix: $BLIP3O_MIX_RATIO (0=all long, 1=all short)"
fi
echo "  Use images: $BLIP3O_USE_IMAGES"
echo "  Image/Caption ratio: $BLIP3O_IMAGE_CAPTION_RATIO (0=all rendered, 1=all real)"
echo "  Cache: /share/project/xiyan/huggingface/cache/datasets"
echo "  Random seed: $SEED (deterministic sampling)"
echo ""
echo "Training Configuration:"
echo "  Num epochs: $NUM_EPOCHS"
echo "  Max steps: $MAX_STEPS"
if [ "$NUM_EPOCHS" != "5" ]; then
  echo "  → Custom epochs: training by epochs (max_steps ignored)"
else
  echo "  → Default config: both limits apply"
fi
if [ "$BLIP3O_DATASET" = "short" ] && [ -z "$BLIP3O_SAMPLE_PCT" ]; then
    echo "  Steps per epoch: ~2,348 (4.8M samples / 2048 effective_batch)"
    echo "  Expected time: ~10.4 hours for 1 epoch (8 GPUs)"
elif [ "$BLIP3O_DATASET" = "long" ] && [ -z "$BLIP3O_SAMPLE_PCT" ]; then
    echo "  Steps per epoch: ~14,329 (29.4M samples / 2048 effective_batch)"
    echo "  Expected time: ~63.7 hours for 1 epoch (8 GPUs)"
else
    echo "  Steps per epoch: ~29 (60K samples / 2048 effective_batch)"
    echo "  Expected time: ~8 min for 1 epoch (8 GPUs)"
fi
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "  Learning rate: $LR"
echo ""
echo "Logging:"
echo "  SwanLab: Disabled (swanboard not installed)"
echo "  Logs: OCRVL/checkpoints/alignment_{dataset}_{timestamp}/training.log"
echo ""
echo "Trainable params: ~6.8M (connectors only, encoder+LLM frozen)"
echo "========================================================================"
echo ""

# Run training with torchrun
SAMPLE_PCT_ARG=""
if [ -n "$BLIP3O_SAMPLE_PCT" ]; then
    SAMPLE_PCT_ARG="--blip3o_sample_percentage $BLIP3O_SAMPLE_PCT"
fi

torchrun --standalone --nproc_per_node=$NUM_GPUS OCRVL/train.py \
    --stage $STAGE \
    --use_blip3o \
    --blip3o_dataset "$BLIP3O_DATASET" \
    $SAMPLE_PCT_ARG \
    --blip3o_mix_ratio $BLIP3O_MIX_RATIO \
    --blip3o_use_images \
    --blip3o_image_caption_ratio $BLIP3O_IMAGE_CAPTION_RATIO \
    --seed $SEED \
    --qwen_model_path "$QWEN_MODEL" \
    --dpsk_model_path "$DPSK_MODEL" \
    --num_epochs $NUM_EPOCHS \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --num_workers 0 2>&1

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Connectors saved to: $OUTPUT_DIR"
echo ""
echo "Architecture: LLaVA-1.5/1.6 standard (mlp2x_gelu)"
echo "  - Final connector: 1280 → 2048 → 2048 (6.8M params)"
echo "  - Deepstack connectors: 1024 → 2048 → 2048 × 3 levels (18.9M params)"
echo "  - Total trainable: 25.7M params"
echo ""
echo "Next steps:"
echo "1. Test inference with aligned connectors"
echo "2. Run Stage 2 (VIT) for task-specific finetuning (optional)"
echo "========================================================================"
