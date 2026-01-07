#!/bin/bash
# Three-Phase LLaVA-alike Training Pipeline
#
# Phase 1: Alignment (Connector only, no LoRA)
#   - Tasks: Image caption + OCR rendered text
#   - Data: BLIP3O 5% for 1 epoch
#   - LR: 1e-3 (constant, matches LLaVA)
#   - Config: RENDER=1, LORA=0
#
# Phase 2: Instruction Tuning (Connector + LoRA on Qwen3VL)
#   - Tasks: Standard VQA
#   - Data: LLaVA-Instruct-150K for 3 epochs
#   - LR: 2e-4 (constant, standard for LoRA)
#   - Config: RENDER=1, LORA=1
#   - Loads checkpoint from Phase 1
#
# Phase 3: Thinking Training (Thinking Projection + LoRA)
#   - Tasks: Chain-of-Thought reasoning with latent tokens
#   - Data: LLaVA-CoT-100K for 1-2 epochs
#   - LR: 1e-4 (thinking projection), 2e-5 (LoRA)
#   - Config: Thinking mode, LORA=1
#   - Loads checkpoint from Phase 2
#   - Note: Disabled by default (set ENABLE_PHASE3=true to enable)
#
# Checkpoint Evaluation:
#   - Enabled by default for all phases
#   - Runs VQA inference on 10 fixed samples per checkpoint
#   - Results saved in: {checkpoint_dir}/eval_results/
#   - Provides qualitative monitoring without full benchmark overhead
#
# Usage:
#   # Run Phase 1 + 2 only (default, Phase 3 disabled)
#   bash OCRVL/scripts/train_llava.sh
#
#   # Run all three phases (enable Phase 3)
#   ENABLE_PHASE3=true bash OCRVL/scripts/train_llava.sh
#
# To run in tmux:
#   This script will automatically create a tmux session named 'ocrvl_llava'

set -e  # Exit on error

# Ensure we're using the correct Python environment
# If not in ocrflow env, try to activate it
OCRFLOW_PYTHON="/share/project/xiyan/envs/ocrflow/bin/python"
if [ -f "$OCRFLOW_PYTHON" ]; then
    # Add ocrflow bin to PATH to ensure we use correct python/pip
    export PATH="/share/project/xiyan/envs/ocrflow/bin:$PATH"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
# Configuration
# ============================================================================

# Enable Phase 3 (Thinking Training) - disabled by default
ENABLE_PHASE3="${ENABLE_PHASE3:-false}"

# Phase 1 configuration
PHASE1_LORA=0           # Connector only
PHASE1_LR=1e-3          # 1e-3 learning rate (constant)
PHASE1_RENDER=1         # Render mode enabled
PHASE1_NUM_EPOCHS=1     # 1 epoch
PHASE1_DATASET_PCT=0.05 # 5% of BLIP3O
PHASE1_BLIP3O_DATASET=long  # BLIP3O dataset variant (long, 60k, mixed, short)

# Phase 2 configuration
PHASE2_LORA=1           # Connector + LoRA
PHASE2_LR=2e-4          # 2e-4 learning rate (constant)
PHASE2_RENDER=1         # Render mode enabled
PHASE2_NUM_EPOCHS=3     # 3 epochs

# Phase 3 configuration (Thinking Training) - only used if ENABLE_PHASE3=true
PHASE3_LORA=1                   # Thinking projection + LoRA
PHASE3_LR=1e-4                  # 1e-4 for thinking projection
PHASE3_NUM_EPOCHS=1             # 1 epoch for LLaVA-CoT-100K
PHASE3_THINKING_LOSS_WEIGHT=1.0 # Weight for thinking alignment loss
PHASE3_MAX_SAMPLES=100000       # Use all 100K samples (set to smaller number for testing)

# System configuration
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TMUX_SESSION_NAME="ocrvl_llava"

# Generate global timestamp for both phases
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Root directory for this training run
ROOT_DIR="OCRVL/checkpoints/llava_${TIMESTAMP}"

# Output directories for each phase
PHASE1_OUTPUT_DIR="${ROOT_DIR}/alignment"
PHASE2_OUTPUT_DIR="${ROOT_DIR}/instruction"
PHASE3_OUTPUT_DIR="${ROOT_DIR}/thinking"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# ============================================================================
# Main Training Pipeline
# ============================================================================

main() {
    cd "$REPO_ROOT"

    log_info "========================================================================"
    log_info "Three-Phase LLaVA-alike Training Pipeline"
    log_info "========================================================================"
    log_info ""
    log_info "Training run: llava_${TIMESTAMP}"
    log_info "Root directory: $ROOT_DIR"
    log_info ""

    # Copy validation images once at the start
    log_info "Copying validation images to ${ROOT_DIR}/validation_images..."
    mkdir -p "${ROOT_DIR}/validation_images"

    LLAVA_IMAGE_BASE="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/images"
    DOCLAYNET_BASE="/share/project/xiyan/huggingface/docling-project/DocLayNet/PNG"

    # COCO images (10 images)
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000009.jpg" "${ROOT_DIR}/validation_images/counting_001.jpg"
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000025.jpg" "${ROOT_DIR}/validation_images/spatial_001.jpg"
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000030.jpg" "${ROOT_DIR}/validation_images/action_001.jpg"
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000034.jpg" "${ROOT_DIR}/validation_images/reasoning_001.jpg"
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000036.jpg" "${ROOT_DIR}/validation_images/object_detail_001.jpg"
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000042.jpg" "${ROOT_DIR}/validation_images/scene_001.jpg"
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000049.jpg" "${ROOT_DIR}/validation_images/attribute_001.jpg"
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000061.jpg" "${ROOT_DIR}/validation_images/multi_object_001.jpg"
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000064.jpg" "${ROOT_DIR}/validation_images/comparison_001.jpg"
    cp "$LLAVA_IMAGE_BASE/coco/train2017/000000000071.jpg" "${ROOT_DIR}/validation_images/temporal_001.jpg"

    # DocLayNet images (2 images)
    cp "$DOCLAYNET_BASE/c6effb847ae7e4a80431696984fa90c98bb08c266481b9a03842422459c43bdd.png" "${ROOT_DIR}/validation_images/ocr_doclaynet_000.png"
    cp "$DOCLAYNET_BASE/f446422ed85e300319d3aff929762b8527cbc9ec26f55703b36df3e7447cc2d5.png" "${ROOT_DIR}/validation_images/ocr_doclaynet_001.png"

    log_info "✓ Validation images copied (12 images total)"
    log_info ""

    log_info "Phase 1: Alignment (Connector only)"
    log_info "  - LORA=$PHASE1_LORA (connector only)"
    log_info "  - LR=$PHASE1_LR (constant)"
    log_info "  - RENDER=$PHASE1_RENDER"
    log_info "  - Dataset: BLIP3O $PHASE1_BLIP3O_DATASET ${PHASE1_DATASET_PCT}% x ${PHASE1_NUM_EPOCHS} epoch"
    log_info "  - Output: ${ROOT_DIR}/alignment/"
    log_info ""
    log_info "Phase 2: Instruction Tuning (Connector + LoRA)"
    log_info "  - LORA=$PHASE2_LORA (connector + LoRA on Qwen3VL)"
    log_info "  - LR=$PHASE2_LR (constant)"
    log_info "  - RENDER=$PHASE2_RENDER"
    log_info "  - Dataset: LLaVA-150K x ${PHASE2_NUM_EPOCHS} epochs"
    log_info "  - Output: ${ROOT_DIR}/instruction/"
    log_info ""
    log_info "Phase 3: Thinking Training (Thinking Projection + LoRA)"
    log_info "  - LORA=$PHASE3_LORA (thinking projection + LoRA)"
    log_info "  - LR=$PHASE3_LR"
    log_info "  - Thinking loss weight: $PHASE3_THINKING_LOSS_WEIGHT"
    log_info "  - Dataset: LLaVA-CoT-100K x ${PHASE3_NUM_EPOCHS} epoch"
    log_info "  - Output: ${ROOT_DIR}/thinking/"
    if [ "$ENABLE_PHASE3" = "true" ]; then
        log_info "  - Status: ENABLED"
    else
        log_info "  - Status: DISABLED (set ENABLE_PHASE3=true to enable)"
    fi
    log_info ""
    log_info "System:"
    log_info "  - GPUs: $NUM_GPUS ($GPU_IDS)"
    log_info "========================================================================"
    log_info ""

    # ========================================================================
    # Phase 1: Alignment Training
    # ========================================================================

    log_info "=========================================="
    log_info "PHASE 1: Starting Alignment Training"
    log_info "=========================================="

    LORA=$PHASE1_LORA \
    LR=$PHASE1_LR \
    RENDER=$PHASE1_RENDER \
    NUM_EPOCHS=$PHASE1_NUM_EPOCHS \
    DATASET_PCT=$PHASE1_DATASET_PCT \
    BLIP3O_DATASET=$PHASE1_BLIP3O_DATASET \
    NUM_GPUS=$NUM_GPUS \
    GPU_IDS=$GPU_IDS \
    OUTPUT_DIR="$PHASE1_OUTPUT_DIR" \
    bash OCRVL/scripts/train_alignment.sh

    if [ $? -ne 0 ]; then
        log_error "Phase 1 training failed!"
        exit 1
    fi

    log_info "Phase 1 completed successfully!"

    # Phase 1 checkpoint is at known location
    PHASE1_CHECKPOINT="$PHASE1_OUTPUT_DIR/step_latest"

    if [ ! -d "$PHASE1_CHECKPOINT" ]; then
        log_error "Failed to find Phase 1 checkpoint at: $PHASE1_CHECKPOINT"
        exit 1
    fi

    log_info "Phase 1 checkpoint: $PHASE1_CHECKPOINT"

    # ========================================================================
    # Phase 2: Instruction Tuning
    # ========================================================================

    log_info ""
    log_info "=========================================="
    log_info "PHASE 2: Starting Instruction Tuning"
    log_info "=========================================="
    log_info "Loading from Phase 1: $PHASE1_CHECKPOINT"
    log_info ""

    LORA=$PHASE2_LORA \
    LR=$PHASE2_LR \
    RENDER=$PHASE2_RENDER \
    NUM_EPOCHS=$PHASE2_NUM_EPOCHS \
    NUM_GPUS=$NUM_GPUS \
    GPU_IDS=$GPU_IDS \
    RESUME_CHECKPOINT=$PHASE1_CHECKPOINT \
    OUTPUT_DIR="$PHASE2_OUTPUT_DIR" \
    bash OCRVL/scripts/train_instruction.sh

    if [ $? -ne 0 ]; then
        log_error "Phase 2 training failed!"
        exit 1
    fi

    log_info "Phase 2 completed successfully!"

    # Phase 2 checkpoint is at known location
    PHASE2_CHECKPOINT="$PHASE2_OUTPUT_DIR/step_latest"

    if [ ! -d "$PHASE2_CHECKPOINT" ]; then
        log_error "Failed to find Phase 2 checkpoint at: $PHASE2_CHECKPOINT"
        exit 1
    fi

    log_info "Phase 2 checkpoint: $PHASE2_CHECKPOINT"

    # ========================================================================
    # Phase 3: Thinking Training (conditional)
    # ========================================================================

    if [ "$ENABLE_PHASE3" = "true" ]; then
        log_info ""
        log_info "=========================================="
        log_info "PHASE 3: Starting Thinking Training"
        log_info "=========================================="
        log_info "Loading from Phase 2: $PHASE2_CHECKPOINT"
        log_info ""

        LORA=$PHASE3_LORA \
        LR=$PHASE3_LR \
        NUM_EPOCHS=$PHASE3_NUM_EPOCHS \
        THINKING_LOSS_WEIGHT=$PHASE3_THINKING_LOSS_WEIGHT \
        MAX_SAMPLES=$PHASE3_MAX_SAMPLES \
        NUM_GPUS=$NUM_GPUS \
        GPU_IDS=$GPU_IDS \
        RESUME_CHECKPOINT=$PHASE2_CHECKPOINT \
        OUTPUT_DIR="$PHASE3_OUTPUT_DIR" \
        bash OCRVL/scripts/train_thinking.sh

        if [ $? -ne 0 ]; then
            log_error "Phase 3 training failed!"
            exit 1
        fi

        log_info "Phase 3 completed successfully!"

        # Phase 3 checkpoint is at known location
        PHASE3_CHECKPOINT="$PHASE3_OUTPUT_DIR/step_latest"

        if [ ! -d "$PHASE3_CHECKPOINT" ]; then
            log_error "Failed to find Phase 3 checkpoint at: $PHASE3_CHECKPOINT"
            exit 1
        fi

        log_info "Phase 3 checkpoint: $PHASE3_CHECKPOINT"
        FINAL_CHECKPOINT=$PHASE3_CHECKPOINT
    else
        log_info ""
        log_info "=========================================="
        log_info "PHASE 3: Skipped (DISABLED)"
        log_info "=========================================="
        log_info "To enable Phase 3, set ENABLE_PHASE3=true"
        log_info ""
        FINAL_CHECKPOINT=$PHASE2_CHECKPOINT
    fi

    # ========================================================================
    # Training Complete
    # ========================================================================

    log_info ""
    log_info "========================================================================"
    log_info "✓ Three-Phase Training Pipeline Completed!"
    log_info "========================================================================"
    log_info ""
    log_info "Training run: llava_${TIMESTAMP}"
    log_info "Root directory: $ROOT_DIR"
    log_info ""
    log_info "Phase 1 checkpoint: ${ROOT_DIR}/alignment/step_latest"
    log_info "Phase 2 checkpoint: ${ROOT_DIR}/instruction/step_latest"
    if [ "$ENABLE_PHASE3" = "true" ]; then
        log_info "Phase 3 checkpoint: ${ROOT_DIR}/thinking/step_latest"
        log_info ""
        log_info "Final model: $FINAL_CHECKPOINT (Phase 3)"
    else
        log_info "Phase 3 checkpoint: Not created (Phase 3 disabled)"
        log_info ""
        log_info "Final model: $FINAL_CHECKPOINT (Phase 2)"
    fi
    log_info ""
    log_info "To evaluate the final model:"
    log_info "  bash OCRVL/evaluation/run_benchmarks.sh $FINAL_CHECKPOINT"
    log_info "========================================================================"
}

# ============================================================================
# Tmux Wrapper
# ============================================================================

# Check if already inside tmux
if [ -n "$TMUX" ]; then
    log_info "Already inside tmux session, running training directly..."
    main
else
    # Check if tmux session already exists
    if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
        log_error "Tmux session '$TMUX_SESSION_NAME' already exists!"
        log_info "Options:"
        log_info "  1. Attach to existing session: tmux attach -t $TMUX_SESSION_NAME"
        log_info "  2. Kill existing session: tmux kill-session -t $TMUX_SESSION_NAME"
        exit 1
    fi

    log_info "Creating tmux session: $TMUX_SESSION_NAME"
    log_info "To attach: tmux attach -t $TMUX_SESSION_NAME"
    log_info "To detach: Ctrl+B, then D"
    log_info ""

    # Create new tmux session and run training
    SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$REPO_ROOT"
    tmux send-keys -t "$TMUX_SESSION_NAME" "bash '$SCRIPT_PATH'" C-m

    log_info "Tmux session '$TMUX_SESSION_NAME' started in background"
    log_info "Attach with: tmux attach -t $TMUX_SESSION_NAME"
fi
