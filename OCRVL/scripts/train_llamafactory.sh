#!/bin/bash
# OCRVL LlamaFactory Training - Native LlamaFactory with OCR Model
#
# This script uses standard llamafactory-cli with:
# - Minimal `sitecustomize.py` integration (OCR model + OCRVL template registration)
# - LoRA + `additional_target` for connector training
#
# Usage:
#   bash OCRVL/scripts/train_llamafactory.sh [config.yaml]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="${ROOT_DIR:-/share/project/xiyan}"
cd "$REPO_ROOT"

# Required python interpreter (OCRFlow env).
PYTHON_BIN="$PROJECT_ROOT/envs/ocrflow/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ Python not found or not executable: $PYTHON_BIN" >&2
  echo "   Please ensure OCRFlow env exists at: $PROJECT_ROOT/envs/ocrflow" >&2
  exit 1
fi

# Default config (alignment mode)
DEFAULT_CONFIG="$REPO_ROOT/OCRVL/configs/alignment/qwen3vl_dpskocr_lora_alignment.yaml"

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
if [ "${1:-}" != "" ]; then
  shift || true
fi

# Generate timestamp for unique output directory (can be overridden by command line)
TIMESTAMP="${OCRVL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
DEFAULT_OUTDIR="$REPO_ROOT/OCRVL/checkpoints/llamafactory/qwen3vl-2b/lora/run_${TIMESTAMP}"

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

# Determine final output directory (may be overridden by command line)
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

# Determine which datasets to build based on config
CONFIG_BASENAME=$(basename "$CONFIG_PATH")

if [[ "$CONFIG_BASENAME" == *"unified_sft"* ]] || [[ "$CONFIG_BASENAME" == *"sft"* ]]; then
    # Unified SFT: Build alignment, VQA (rendered), VQA (text), and unified datasets
    ALIGNMENT_DATASET="$REPO_ROOT/OCRVL/data/ocrvl_alignment_text_prompts.jsonl"
    VQA_DATASET="$REPO_ROOT/OCRVL/data/ocrvl_llava_mix665k.jsonl"
    STANDARD_VQA_DATASET="$REPO_ROOT/OCRVL/data/ocrvl_llava_standard_vqa.jsonl"
    UNIFIED_DATASET="$REPO_ROOT/OCRVL/data/ocrvl_unified_sft.jsonl"

    # Build alignment dataset if needed
    if [ ! -f "$ALIGNMENT_DATASET" ]; then
        echo "========================================================================"
        echo "Building alignment dataset (text prompts + DocLayNet)..."
        echo "========================================================================"
        echo ""
        "$PYTHON_BIN" "$REPO_ROOT/OCRVL/scripts/build_alignment_dataset_text_prompts.py" \
            --doclaynet-json "${DOCLAYNET_JSON_PATH:-${PROJECT_ROOT}/huggingface/docling-project/DocLayNet/DocLayNet_core_train.json}" \
            --doclaynet-images "${DOCLAYNET_IMAGE_DIR:-${PROJECT_ROOT}/huggingface/docling-project/DocLayNet/PNG}"
        if [ $? -ne 0 ]; then
            echo "❌ Failed to build alignment dataset"
            exit 1
        fi
        echo "✓ Alignment dataset built (LLaVA-Pretrain + DocLayNet)"
        echo ""
    fi

    # Build VQA datasets if needed
    if [ ! -f "$VQA_DATASET" ] || [ ! -f "$STANDARD_VQA_DATASET" ]; then
        echo "========================================================================"
        echo "Building VQA datasets (standard + rendered)..."
        echo "========================================================================"
        echo ""

        # Build standard VQA (no rendering)
        if [ ! -f "$STANDARD_VQA_DATASET" ]; then
            "$PYTHON_BIN" "$REPO_ROOT/OCRVL/scripts/build_vqa.py" \
                --mode standard \
                --llava-json "${LLAVA_JSON:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json}" \
                --llava-images "${LLAVA_IMAGES:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/images}" \
                --output "$STANDARD_VQA_DATASET"
            if [ $? -ne 0 ]; then
                echo "❌ Failed to build standard VQA dataset"
                exit 1
            fi
        fi

        # Build rendered VQA (conversation history)
        if [ ! -f "$VQA_DATASET" ]; then
            "$PYTHON_BIN" "$REPO_ROOT/OCRVL/scripts/build_vqa.py" \
                --mode rendered \
                --llava-json "${LLAVA_JSON:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json}" \
                --llava-images "${LLAVA_IMAGES:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/images}" \
                --output "$VQA_DATASET" \
                --rendered-images-dir "$REPO_ROOT/OCRVL/data/ocrvl_rendered_conversations"
            if [ $? -ne 0 ]; then
                echo "❌ Failed to build rendered VQA dataset"
                exit 1
            fi
        fi

        echo "✓ VQA datasets built (standard + rendered)"
        echo ""
    fi

    # Build unified dataset
    if [ ! -f "$UNIFIED_DATASET" ]; then
        echo "========================================================================"
        echo "Building unified SFT dataset (alignment + VQA rendered + VQA text)..."
        echo "========================================================================"
        echo ""
        "$PYTHON_BIN" "$REPO_ROOT/OCRVL/scripts/build_unified_sft_dataset.py" \
            --alignment-jsonl "$ALIGNMENT_DATASET" \
            --vqa-jsonl "$VQA_DATASET" \
            --standard-vqa-jsonl "$STANDARD_VQA_DATASET" \
            --output "$UNIFIED_DATASET" \
            ${OCRVL_UNIFIED_MAX_SAMPLES:+--max-samples "$OCRVL_UNIFIED_MAX_SAMPLES"} \
            ${OCRVL_UNIFIED_ALIGNMENT_RATIO:+--alignment-ratio "$OCRVL_UNIFIED_ALIGNMENT_RATIO"} \
            ${OCRVL_UNIFIED_VQA_RATIO:+--vqa-ratio "$OCRVL_UNIFIED_VQA_RATIO"} \
            ${OCRVL_UNIFIED_STANDARD_VQA_RATIO:+--standard-vqa-ratio "$OCRVL_UNIFIED_STANDARD_VQA_RATIO"}
        if [ $? -ne 0 ]; then
            echo "❌ Failed to build unified SFT dataset"
            exit 1
        fi
        echo "✓ Unified SFT dataset built"
        echo ""
    fi

    # Check that alignment checkpoint exists (unified SFT loads from it)
    ALIGNMENT_CKPT="${ADAPTER_CHECKPOINT_PATH:-$REPO_ROOT/OCRVL/checkpoints/llamafactory/qwen3vl-2b/lora/alignment/checkpoint-311}"
    if [ ! -d "$ALIGNMENT_CKPT" ]; then
        echo "========================================================================"
        echo "⚠️  Alignment checkpoint not found: $ALIGNMENT_CKPT"
        echo ""
        echo "Unified SFT requires alignment checkpoint to load connector weights."
        echo "Please run alignment training first:"
        echo "  bash OCRVL/scripts/train_llamafactory.sh OCRVL/configs/alignment/qwen3vl_dpskocr_lora_alignment.yaml"
        echo ""
        echo "Or override with custom path:"
        echo "  ADAPTER_CHECKPOINT_PATH=/path/to/alignment/checkpoint \\"
        echo "  bash OCRVL/scripts/train_llamafactory.sh OCRVL/configs/sft/qwen3vl_dpskocr_lora_sft.yaml"
        echo "========================================================================"
        exit 1
    fi
    echo "✓ Alignment checkpoint found: $ALIGNMENT_CKPT"
    echo ""

elif [[ "$CONFIG_BASENAME" == *"alignment"* ]]; then
    # Alignment stage: Build alignment dataset
    ALIGNMENT_DATASET="$REPO_ROOT/OCRVL/data/ocrvl_alignment_text_prompts.jsonl"
    if [ ! -f "$ALIGNMENT_DATASET" ]; then
        echo "========================================================================"
        echo "Building alignment dataset (text prompts + DocLayNet)..."
        echo "========================================================================"
        echo ""
        "$PYTHON_BIN" "$REPO_ROOT/OCRVL/scripts/build_alignment_dataset_text_prompts.py" \
            --doclaynet-json "${DOCLAYNET_JSON_PATH:-${PROJECT_ROOT}/huggingface/docling-project/DocLayNet/DocLayNet_core_train.json}" \
            --doclaynet-images "${DOCLAYNET_IMAGE_DIR:-${PROJECT_ROOT}/huggingface/docling-project/DocLayNet/PNG}"
        if [ $? -ne 0 ]; then
            echo "❌ Failed to build alignment dataset"
            exit 1
        fi
        echo "✓ Alignment dataset built (LLaVA-Pretrain + DocLayNet)"
        echo ""
    fi

elif [[ "$CONFIG_BASENAME == *"llava"* ]] || [[ "$CONFIG_BASENAME == *"vqa"* ]]; then
    # VQA stage: Build VQA dataset (both modes)
    VQA_DATASET="$REPO_ROOT/OCRVL/data/ocrvl_llava_mix665k.jsonl"
    STANDARD_VQA_DATASET="$REPO_ROOT/OCRVL/data/ocrvl_llava_standard_vqa.jsonl"

    if [ ! -f "$VQA_DATASET" ] || [ ! -f "$STANDARD_VQA_DATASET" ]; then
        echo "========================================================================"
        echo "Building VQA datasets (standard + rendered)..."
        echo "========================================================================"
        echo ""

        # Build standard VQA
        if [ ! -f "$STANDARD_VQA_DATASET" ]; then
            "$PYTHON_BIN" "$REPO_ROOT/OCRVL/scripts/build_vqa.py" \
                --mode standard \
                --llava-json "${LLAVA_JSON:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json}" \
                --llava-images "${LLAVA_IMAGES:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/images}" \
                --output "$STANDARD_VQA_DATASET"
            if [ $? -ne 0 ]; then
                echo "❌ Failed to build standard VQA dataset"
                exit 1
            fi
        fi

        # Build rendered VQA
        if [ ! -f "$VQA_DATASET" ]; then
            "$PYTHON_BIN" "$REPO_ROOT/OCRVL/scripts/build_vqa.py" \
                --mode rendered \
                --llava-json "${LLAVA_JSON:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json}" \
                --llava-images "${LLAVA_IMAGES:-${PROJECT_ROOT}/huggingface/liuhaotian/LLaVA-Instruct-150K/images}" \
                --output "$VQA_DATASET" \
                --rendered-images-dir "$REPO_ROOT/OCRVL/data/ocrvl_rendered_conversations"
            if [ $? -ne 0 ]; then
                echo "❌ Failed to build rendered VQA dataset"
                exit 1
            fi
        fi

        echo "✓ VQA datasets built (standard + rendered)"
        echo ""
    fi

fi

# Set Python path for `sitecustomize.py` integration.
# NOTE: `sitecustomize.py` is at repo root (no __init__.py) so Python auto-imports it on startup.
export PYTHONPATH="$REPO_ROOT/../LlamaFactory/src:$REPO_ROOT:$REPO_ROOT/OCRVL/llamafactory:${PYTHONPATH:-}"

# Default to 8 GPUs unless user explicitly sets CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
# Keep deepstack enabled unless explicitly disabled (matches alignment connector targets).
export OCRVL_DPSK_DEEPSTACK="${OCRVL_DPSK_DEEPSTACK:-1}"

# Prefer local mirrors for offline training
# Local DeepSeek-OCR copy ensures FSDP treats it as part of the model (not external dependency)
export DPSK_MODEL_PATH="${DPSK_MODEL_PATH:-$REPO_ROOT/OCRVL/checkpoints/OCR-Qwen3-VL-2B}"
export DPSK_DTYPE="${DPSK_DTYPE:-bf16}"

# Enable checkpoint evaluation
export OCRVL_ENABLE_TRANSPARENT_EVAL="${OCRVL_ENABLE_TRANSPARENT_EVAL:-1}"
export OCRVL_TRANSPARENT_EVAL_SAMPLES="${OCRVL_TRANSPARENT_EVAL_SAMPLES:-$REPO_ROOT/OCRVL/llamafactory/transparent_eval_samples.json}"
export OCRVL_TRANSPARENT_EVAL_MAX_NEW_TOKENS="${OCRVL_TRANSPARENT_EVAL_MAX_NEW_TOKENS:-128}"
export OCRVL_TRANSPARENT_EVAL_TEMPERATURE="${OCRVL_TRANSPARENT_EVAL_TEMPERATURE:-0.0}"
export OCRVL_TRANSPARENT_EVAL_RUN_AT_START="${OCRVL_TRANSPARENT_EVAL_RUN_AT_START:-0}"
export OCRVL_TRANSPARENT_EVAL_LIMIT="${OCRVL_TRANSPARENT_EVAL_LIMIT:-}"

# Enable rendering (RENDER=1 for on-the-fly rendering during instruction tuning)
# For alignment stage, instructions are pre-rendered in the dataset (RENDER has no effect)
export RENDER="${RENDER:-1}"

# Determine rendering status for display
if [ "$RENDER" = "1" ]; then
    RENDER_STATUS="ENABLED (VQA questions as images)"
else
    RENDER_STATUS="DISABLED"
fi

# Determine training type for display
TRAINING_TYPE="Standard Training"
if [[ "$CONFIG_BASENAME" == *"unified_sft"* ]] || [[ "$CONFIG_BASENAME" == *"sft"* ]]; then
    TRAINING_TYPE="Unified SFT Training (Alignment + VQA)"
elif [[ "$CONFIG_BASENAME" == *"alignment"* ]]; then
    TRAINING_TYPE="Alignment Training"
elif [[ "$CONFIG_BASENAME" == *"llava"* ]] || [[ "$CONFIG_BASENAME" == *"vqa"* ]]; then
    TRAINING_TYPE="VQA Training"
fi

echo "========================================================================"
echo "OCRVL LlamaFactory Training"
echo "========================================================================"
echo "Training Type: $TRAINING_TYPE"
echo "Config: $CONFIG_PATH"
echo "DPSK_MODEL_PATH: $DPSK_MODEL_PATH"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Using standard LlamaFactory with:"
echo "  - OCR model registration (sitecustomize.py)"
echo "  - LoRA + additional_target for connectors"
echo "  - On-the-fly rendering (RENDER): $RENDER_STATUS"
echo "  - Transparent eval: $OCRVL_ENABLE_TRANSPARENT_EVAL (samples: $OCRVL_TRANSPARENT_EVAL_SAMPLES)"
echo "    - Run at start: $OCRVL_TRANSPARENT_EVAL_RUN_AT_START (writes checkpoint-0/eval_results)"
echo "    - Limit: ${OCRVL_TRANSPARENT_EVAL_LIMIT:-all} (set OCRVL_TRANSPARENT_EVAL_LIMIT=1 for quick smoke test)"
echo "========================================================================"
echo ""

# Run standard LlamaFactory CLI with tee for logging
# Use PIPESTATUS to preserve exit code from llamafactory-cli
NPROC_PER_NODE_DEFAULT="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$NPROC_PER_NODE_DEFAULT}"

"$PYTHON_BIN" -m llamafactory.cli train "$CONFIG_PATH" "$@" 2>&1 | tee -a "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
exit ${exit_code:-0}
