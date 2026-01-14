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
# Configuration (LlamaFactory-style YAML)
# ============================================================================

# Default config mirrors LlamaFactory's example YAML style, but drives OCRVL pipeline.
DEFAULT_CONFIG="OCRVL/examples/llamafactory/train_llava_pipeline.yaml"

CONFIG_PATH="${CONFIG_PATH:-}"
OVERRIDES=()
if [ $# -gt 0 ] && [[ "$1" == *.yml || "$1" == *.yaml ]]; then
    CONFIG_PATH="$1"
    shift
fi
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH="$DEFAULT_CONFIG"
fi
OVERRIDES=("$@")

# Resolve settings (supports key=value overrides like LlamaFactory).
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $CONFIG_PATH" >&2
    exit 1
fi

RESOLVED_ENV="$(python OCRVL/scripts/resolve_llava_config.py "$CONFIG_PATH" "${OVERRIDES[@]}")" || {
    echo "ERROR: Failed to resolve config: $CONFIG_PATH" >&2
    exit 1
}
eval "$RESOLVED_ENV"

# Backwards-compatible defaults (in case resolve script didn't emit something)
ENABLE_PHASE3="${ENABLE_PHASE3:-false}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-ocrvl_llava}"

TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
ROOT_DIR="${ROOT_DIR:-OCRVL/checkpoints/llava_${TIMESTAMP}}"
PHASE1_OUTPUT_DIR="${PHASE1_OUTPUT_DIR:-${ROOT_DIR}/alignment}"
PHASE2_OUTPUT_DIR="${PHASE2_OUTPUT_DIR:-${ROOT_DIR}/instruction}"
PHASE3_OUTPUT_DIR="${PHASE3_OUTPUT_DIR:-${ROOT_DIR}/thinking}"

VALIDATION_IMAGES="${VALIDATION_IMAGES:-true}"
LLAVA_IMAGE_BASE="${LLAVA_IMAGE_BASE:-/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/images}"
DOCLAYNET_BASE="${DOCLAYNET_BASE:-/share/project/xiyan/huggingface/docling-project/DocLayNet/PNG}"

# Phase defaults (if not provided by config/env)
PHASE1_LORA="${PHASE1_LORA:-0}"
PHASE1_LR="${PHASE1_LR:-1e-3}"
PHASE1_NUM_EPOCHS="${PHASE1_NUM_EPOCHS:-1}"
PHASE1_DATASET_PCT="${PHASE1_DATASET_PCT:-0.05}"
PHASE1_BLIP3O_DATASET="${PHASE1_BLIP3O_DATASET:-long}"
PHASE1_DATASET_TYPE="${PHASE1_DATASET_TYPE:-blip3o}"
PHASE1_BATCH_SIZE="${PHASE1_PER_DEVICE_TRAIN_BATCH_SIZE:-${PHASE1_BATCH_SIZE:-8}}"
PHASE1_GRAD_ACCUM="${PHASE1_GRADIENT_ACCUMULATION_STEPS:-${PHASE1_GRAD_ACCUM:-4}}"
PHASE1_LORA_R="${PHASE1_LORA_RANK:-${PHASE1_LORA_R:-8}}"
PHASE1_LORA_ALPHA="${PHASE1_LORA_ALPHA:-16}"
PHASE1_LORA_DROPOUT="${PHASE1_LORA_DROPOUT:-0.05}"

PHASE2_LORA="${PHASE2_LORA:-1}"
PHASE2_LR="${PHASE2_LR:-2e-4}"
PHASE2_RENDER="${PHASE2_RENDER:-1}"
PHASE2_NUM_EPOCHS="${PHASE2_NUM_EPOCHS:-3}"
PHASE2_BATCH_SIZE="${PHASE2_PER_DEVICE_TRAIN_BATCH_SIZE:-${PHASE2_BATCH_SIZE:-12}}"
PHASE2_GRAD_ACCUM="${PHASE2_GRADIENT_ACCUMULATION_STEPS:-${PHASE2_GRAD_ACCUM:-4}}"
PHASE2_LORA_R="${PHASE2_LORA_RANK:-${PHASE2_LORA_R:-8}}"
PHASE2_LORA_ALPHA="${PHASE2_LORA_ALPHA:-16}"
PHASE2_LORA_DROPOUT="${PHASE2_LORA_DROPOUT:-0.05}"
PHASE2_LLAVA_JSON_PATH="${PHASE2_LLAVA_JSON_PATH:-/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json}"
PHASE2_LLAVA_IMAGE_DIR="${PHASE2_LLAVA_IMAGE_DIR:-/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/images}"

PHASE3_LORA="${PHASE3_LORA:-1}"
PHASE3_LR="${PHASE3_LR:-1e-4}"
PHASE3_NUM_EPOCHS="${PHASE3_NUM_EPOCHS:-1}"
PHASE3_THINKING_LOSS_WEIGHT="${PHASE3_THINKING_LOSS_WEIGHT:-1.0}"
PHASE3_MAX_SAMPLES="${PHASE3_MAX_SAMPLES:-100000}"
PHASE3_BATCH_SIZE="${PHASE3_PER_DEVICE_TRAIN_BATCH_SIZE:-${PHASE3_BATCH_SIZE:-4}}"
PHASE3_GRAD_ACCUM="${PHASE3_GRADIENT_ACCUMULATION_STEPS:-${PHASE3_GRAD_ACCUM:-4}}"
PHASE3_THINKING_JSONL="${PHASE3_THINKING_JSONL:-/share/project/xiyan/huggingface/Xkev/LLaVA-CoT-100k/train.jsonl}"
PHASE3_THINKING_IMAGE_DIR="${PHASE3_THINKING_IMAGE_DIR:-/share/project/xiyan/huggingface}"

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
    log_info "Config: $CONFIG_PATH"
    log_info ""

    # Copy validation images once at the start
    if [ "$VALIDATION_IMAGES" = "true" ]; then
        log_info "Copying validation images to ${ROOT_DIR}/validation_images..."
        mkdir -p "${ROOT_DIR}/validation_images"

        # COCO images (10 images)
        declare -a VALIDATION_COPIES=(
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000009.jpg|counting_001.jpg"
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000025.jpg|spatial_001.jpg"
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000030.jpg|action_001.jpg"
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000034.jpg|reasoning_001.jpg"
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000036.jpg|object_detail_001.jpg"
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000042.jpg|scene_001.jpg"
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000049.jpg|attribute_001.jpg"
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000061.jpg|multi_object_001.jpg"
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000064.jpg|comparison_001.jpg"
            "$LLAVA_IMAGE_BASE/coco/train2017/000000000071.jpg|temporal_001.jpg"
            # DocLayNet images (2 images)
            "$DOCLAYNET_BASE/c6effb847ae7e4a80431696984fa90c98bb08c266481b9a03842422459c43bdd.png|ocr_doclaynet_000.png"
            "$DOCLAYNET_BASE/f446422ed85e300319d3aff929762b8527cbc9ec26f55703b36df3e7447cc2d5.png|ocr_doclaynet_001.png"
        )

        COPIED=0
        for entry in "${VALIDATION_COPIES[@]}"; do
            SRC="${entry%%|*}"
            DST="${entry##*|}"
            if [ -f "$SRC" ]; then
                cp "$SRC" "${ROOT_DIR}/validation_images/$DST"
                COPIED=$((COPIED + 1))
            else
                log_info "⚠️  Missing validation image, skipping: $SRC"
            fi
        done

        log_info "✓ Validation images prepared (${COPIED} files)"
        log_info ""
    else
        log_info "Validation image copying disabled (VALIDATION_IMAGES=false)"
        log_info ""
    fi

    log_info "Phase 1: Alignment (Connector only)"
    log_info "  - LORA=$PHASE1_LORA (connector only)"
    log_info "  - LR=$PHASE1_LR (constant)"
    log_info "  - Dataset: $PHASE1_DATASET_TYPE (BLIP3o=$PHASE1_BLIP3O_DATASET, pct=$PHASE1_DATASET_PCT)"
    log_info "  - Epochs: ${PHASE1_NUM_EPOCHS}"
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
    LORA_R=$PHASE1_LORA_R \
    LORA_ALPHA=$PHASE1_LORA_ALPHA \
    LORA_DROPOUT=$PHASE1_LORA_DROPOUT \
    LR=$PHASE1_LR \
    NUM_EPOCHS=$PHASE1_NUM_EPOCHS \
    BATCH_SIZE=$PHASE1_BATCH_SIZE \
    GRAD_ACCUM=$PHASE1_GRAD_ACCUM \
    DATASET_TYPE=$PHASE1_DATASET_TYPE \
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
    LORA_R=$PHASE2_LORA_R \
    LORA_ALPHA=$PHASE2_LORA_ALPHA \
    LORA_DROPOUT=$PHASE2_LORA_DROPOUT \
    LR=$PHASE2_LR \
    RENDER=$PHASE2_RENDER \
    NUM_EPOCHS=$PHASE2_NUM_EPOCHS \
    BATCH_SIZE=$PHASE2_BATCH_SIZE \
    GRAD_ACCUM=$PHASE2_GRAD_ACCUM \
    NUM_GPUS=$NUM_GPUS \
    GPU_IDS=$GPU_IDS \
    LLAVA_JSON_PATH=$PHASE2_LLAVA_JSON_PATH \
    LLAVA_IMAGE_DIR=$PHASE2_LLAVA_IMAGE_DIR \
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
        BATCH_SIZE=$PHASE3_BATCH_SIZE \
        GRAD_ACCUM=$PHASE3_GRAD_ACCUM \
        THINKING_JSONL=$PHASE3_THINKING_JSONL \
        THINKING_IMAGE_DIR=$PHASE3_THINKING_IMAGE_DIR \
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
