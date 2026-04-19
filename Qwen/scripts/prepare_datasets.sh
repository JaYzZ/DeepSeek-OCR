#!/bin/bash
# Build 3 thinking datasets FIRST (render -> encode), then the rest
#
# Usage:
#   ./prepare_datasets.sh [gpu_id]
#
# Examples:
#   ./prepare_datasets.sh         # Uses CUDA:0 (default)
#   ./prepare_datasets.sh 0       # Uses CUDA:0
#   ./prepare_datasets.sh 1       # Uses CUDA:1
#   ./prepare_datasets.sh 2       # Uses CUDA:2

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Parse arguments
GPU_ID=${1:-0}  # Default to CUDA:0 if not specified
DEVICE="cuda:$GPU_ID"

# Base directory
REPO_ROOT="/home/jianzhan/sources/DeepSeek-OCR"
cd "$REPO_ROOT"

# Use explicit Python path to avoid environment issues
PYTHON="/home/jianzhan/envs/ocrflow/bin/python"

# Create logs directory
LOG_DIR="$REPO_ROOT/logs/build_scripts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

log_info "=========================================="
log_info "Dataset Preparation Script"
log_info "=========================================="
log_info "Log directory: $LOG_DIR"
log_info "Using device: $DEVICE"
log_info "GPU ID: $GPU_ID"
echo ""

# ==============================================================================
# STEP 1: RENDER the 3 thinking datasets (CPU-bound)
# ==============================================================================
log_info "=========================================="
log_info "STEP 1: RENDER 3 THINKING DATASETS"
log_info "=========================================="
echo ""

# Render Chimera
log_info "Rendering: chimera_thinking"
$PYTHON Qwen/data/build_chimera_thinking.py \
    --render-only \
    --device $DEVICE \
    --batch-size 8 \
    2>&1 | tee "$LOG_DIR/chimera_thinking_render.log"
log_success "✓ chimera_thinking render complete"
echo ""

# Render DeepVision
log_info "Rendering: deepvision_thinking"
$PYTHON Qwen/data/build_deepvision_thinking.py \
    --render-only \
    --device $DEVICE \
    --batch-size 8 \
    --thinking-source auto \
    2>&1 | tee "$LOG_DIR/deepvision_thinking_render.log"
log_success "✓ deepvision_thinking render complete"
echo ""

# Render R1OneVision
log_info "Rendering: r1onevision_thinking"
$PYTHON Qwen/data/build_r1onevision_thinking.py \
    --render-only \
    --device $DEVICE \
    --batch-size 8 \
    --all \
    2>&1 | tee "$LOG_DIR/r1onevision_thinking_render.log"
log_success "✓ r1onevision_thinking render complete"
echo ""

log_success "=========================================="
log_success "STEP 1 (RENDER) COMPLETE"
log_success "=========================================="
echo ""
echo ""

# ==============================================================================
# STEP 2: ENCODE the 3 thinking datasets (GPU-bound on $DEVICE)
# ==============================================================================
log_info "=========================================="
log_info "STEP 2: ENCODE 3 THINKING DATASETS"
log_info "=========================================="
echo ""

# Encode Chimera
log_info "Encoding: chimera_thinking on $DEVICE"
$PYTHON Qwen/data/build_chimera_thinking.py \
    --encode-only \
    --device $DEVICE \
    --batch-size 8 \
    2>&1 | tee "$LOG_DIR/chimera_thinking_encode.log"
log_success "✓ chimera_thinking encode complete"
echo ""

# Encode DeepVision
log_info "Encoding: deepvision_thinking on $DEVICE"
$PYTHON Qwen/data/build_deepvision_thinking.py \
    --encode-only \
    --device $DEVICE \
    --batch-size 8 \
    --thinking-source auto \
    2>&1 | tee "$LOG_DIR/deepvision_thinking_encode.log"
log_success "✓ deepvision_thinking encode complete"
echo ""

# Encode R1OneVision
log_info "Encoding: r1onevision_thinking on $DEVICE"
$PYTHON Qwen/data/build_r1onevision_thinking.py \
    --encode-only \
    --device $DEVICE \
    --batch-size 8 \
    --all \
    2>&1 | tee "$LOG_DIR/r1onevision_thinking_encode.log"
log_success "✓ r1onevision_thinking encode complete"
echo ""

log_success "=========================================="
log_success "STEP 2 (ENCODE) COMPLETE"
log_success "=========================================="
echo ""
echo ""

# ==============================================================================
# STEP 3: Build VERL datasets (sequential)
# ==============================================================================
log_info "=========================================="
log_info "STEP 3: BUILD VERL DATASETS (sequential)"
log_info "=========================================="
echo ""

# Chimera VERL
log_info "Building: chimera_verl"
$PYTHON Qwen/data/build_chimera_verl_dataset.py \
    2>&1 | tee "$LOG_DIR/chimera_verl.log"
log_success "✓ chimera_verl complete"
echo ""

# DeepVision VERL
log_info "Building: deepvision_verl"
$PYTHON Qwen/data/build_deepvision_verl_dataset.py \
    2>&1 | tee "$LOG_DIR/deepvision_verl.log"
log_success "✓ deepvision_verl complete"
echo ""

log_success "=========================================="
log_success "STEP 3 (VERL) COMPLETE"
log_success "=========================================="
echo ""
echo ""

# ==============================================================================
# STEP 4: Build OPD/OPSD manifests (sequential)
# ==============================================================================
log_info "=========================================="
log_info "STEP 4: BUILD OPD/OPSD MANIFESTS (sequential)"
log_info "=========================================="
echo ""

# OPD Text
log_info "Building: qwen3vl_opd_text_manifest"
$PYTHON Qwen/data/build_qwen3vl_opd_text_dataset.py \
    --datasets chimera_qwen35_thinking_image_input,deepvision_thinking,r1ov_thinking \
    --output-jsonl Qwen/data/sft/opd_text_manifest.jsonl \
    2>&1 | tee "$LOG_DIR/opd_text.log"
log_success "✓ qwen3vl_opd_text_manifest complete"
echo ""

# OPSD
log_info "Building: qwen3vl_opsd_manifest"
$PYTHON Qwen/data/build_qwen3vl_opsd_dataset.py \
    --datasets chimera_qwen35_thinking_image_input,deepvision_thinking,r1ov_thinking \
    --output-jsonl Qwen/data/sft/opsd_manifest.jsonl \
    2>&1 | tee "$LOG_DIR/opsd.log"
log_success "✓ qwen3vl_opsd_manifest complete"
echo ""

log_success "=========================================="
log_success "STEP 4 (OPD/OPSD) COMPLETE"
log_success "=========================================="
echo ""
echo ""

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
log_success "=========================================="
log_success "ALL BUILD SCRIPTS COMPLETE"
log_success "=========================================="
log_info "Logs saved to: $LOG_DIR"
echo ""
log_info "Generated datasets:"
echo "  Thinking datasets:"
echo "    - chimera_thinking (image + text input)"
echo "    - deepvision_thinking"
echo "    - r1onevision_thinking"
echo "  VERL datasets:"
echo "    - chimera_verl"
echo "    - deepvision_verl"
echo "  Manifests:"
echo "    - qwen3vl_opd_text_manifest.jsonl"
echo "    - qwen3vl_opsd_manifest.jsonl"
echo ""
log_success "Done!"
