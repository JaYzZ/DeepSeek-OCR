#!/bin/bash
set -euo pipefail

# OCRQwen3VL Benchmark Runner
# Uses the unified evaluation script with vLLM multimodal support
#
# Usage:
#   bash run_ocrqwen3vl_benchmarks.sh [options]
#
# Examples:
#   # Run OCRQwen3VL merged checkpoint
#   bash run_ocrqwen3vl_benchmarks.sh -c OCRVL/checkpoints/OCRQwen3VL-2B-merged-lora
#
#   # Run OCRQwen3VL with question rendering (training pattern: vision + rendered_q + instruction → answer)
#   bash run_ocrqwen3vl_benchmarks.sh -c OCRVL/checkpoints/OCRQwen3VL-2B-merged-lora --render-questions
#
#   # Run official Qwen3VL baseline
#   bash run_ocrqwen3vl_benchmarks.sh -T qwen3vl -c ../../huggingface/Qwen/Qwen3-VL-2B-Instruct
#
#   # Run specific benchmark
#   bash run_ocrqwen3vl_benchmarks.sh -b scienceqa
#
#   # Quick test (10 samples)
#   bash run_ocrqwen3vl_benchmarks.sh -b scienceqa -m 10

# ============================================================================
# Configuration
# ============================================================================

# Default paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_CHECKPOINT="${REPO_ROOT}/OCRVL/checkpoints/OCR-Qwen3-VL-2B"
DEFAULT_OUTPUT="${REPO_ROOT}/OCRVL/evaluation/results/ocrqwen3vl_$(date +%Y%m%d_%H%M%S)"

# Default settings
BENCHMARKS="scienceqa,m3cot"
CHECKPOINT="${DEFAULT_CHECKPOINT}"
MODEL_TYPE="ocrqwen3vl"
OUTPUT_DIR="${DEFAULT_OUTPUT}"
RENDER_QUESTIONS=false
MAX_SAMPLES=""
MAX_TOKENS=512
GPU_MEMORY=0.85
BATCH_SIZE=4

# ============================================================================
# Parse arguments
# ============================================================================

print_help() {
    cat <<HELP
OCRQwen3VL Benchmark Runner

Usage: $0 [options]

Options:
  -b, --benchs         Comma-separated benchmarks (scienceqa, m3cot, mathvision, realworldqa, mmmu, odinw)
  -c, --checkpoint     Path to model checkpoint
  -T, --model-type     Model type: ocrqwen3vl (default) or qwen3vl (baseline)
  -r, --render-questions Render questions as images (OCRQwen3VL only)
  -o, --output         Output directory
  -m, --max-samples    Limit to N samples for testing
  -t, --max-tokens     Max tokens to generate (default: 512)
  -g, --gpu-memory     GPU memory utilization 0-1 (default: 0.85)
  -s, --batch-size     Batch size for inference (default: 4)
  -h, --help           Show this help

Examples:
  # Run OCRQwen3VL with merged checkpoint
  $0 -c OCRVL/checkpoints/OCRQwen3VL-2B-merged-lora

  # Run OCRQwen3VL with question rendering (training pattern)
  $0 -c OCRVL/checkpoints/OCRQwen3VL-2B-merged-lora --render-questions

  # Run official Qwen3VL baseline
  $0 -T qwen3vl -c ../../huggingface/Qwen/Qwen3-VL-2B-Instruct

  # Run specific benchmark
  $0 -b scienceqa

  # Run multiple benchmarks
  $0 -b m3cot,scienceqa,mathvision

  # Quick test (10 samples)
  $0 -b scienceqa -m 10

Environment:
  OCRQWEN3VL_CHECKPOINT   Default checkpoint path
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--benchs)
            BENCHMARKS="$2"; shift 2;;
        -c|--checkpoint)
            CHECKPOINT="$2"; shift 2;;
        -T|--model-type)
            MODEL_TYPE="$2"; shift 2;;
        -r|--render-questions)
            RENDER_QUESTIONS=true; shift;;
        -o|--output)
            OUTPUT_DIR="$2"; shift 2;;
        -m|--max-samples)
            MAX_SAMPLES="--max-samples $2"; shift 2;;
        -t|--max-tokens)
            MAX_TOKENS="$2"; shift 2;;
        -g|--gpu-memory)
            GPU_MEMORY="$2"; shift 2;;
        -s|--batch-size)
            BATCH_SIZE="$2"; shift 2;;
        -h|--help)
            print_help; exit 0;;
        *)
            echo "Unknown option: $1"
            print_help; exit 1;;
    esac
done

# Use environment variables as fallback
CHECKPOINT="${OCRQWEN3VL_CHECKPOINT:-${CHECKPOINT}}"

# Validate: render_questions only supported for ocrqwen3vl
if [[ "${RENDER_QUESTIONS}" == "true" ]] && [[ "${MODEL_TYPE}" != "ocrqwen3vl" ]]; then
    echo "⚠️  Warning: --render-questions is only supported for ocrqwen3vl model"
    echo "   Ignoring --render-questions option"
    RENDER_QUESTIONS=false
fi

# ============================================================================
# Validate paths
# ============================================================================

echo "================================================================================"
echo "OCRQwen3VL Benchmark Runner"
echo "================================================================================"
echo "Repository:  ${REPO_ROOT}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Model Type:  ${MODEL_TYPE}"
echo "Render Questions: ${RENDER_QUESTIONS}"
echo "Benchmarks:  ${BENCHMARKS}"
echo "Output:      ${OUTPUT_DIR}"
echo "Max Tokens:  ${MAX_TOKENS}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "================================================================================"
echo ""

# Check checkpoint exists
if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "❌ Error: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Run benchmarks
# ============================================================================

EVAL_SCRIPT="${REPO_ROOT}/OCRVL/evaluation/eval_ocrqwen3vl.py"

# Check eval script exists
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "❌ Error: Evaluation script not found: ${EVAL_SCRIPT}"
    exit 1
fi

# Convert comma-separated benchmarks to array
IFS=',' read -ra BENCH_ARRAY <<< "${BENCHMARKS}"

# Run each benchmark
for benchmark in "${BENCH_ARRAY[@]}"; do
    benchmark=$(echo "$benchmark" | tr '[:upper:]' '[:lower:]' | xargs)

    echo ""
    echo "================================================================================"
    echo "Running: ${benchmark^^}"
    echo "================================================================================"

    # Build command
    CMD="python ${EVAL_SCRIPT}"
    CMD+=" --benchmark ${benchmark}"
    CMD+=" --checkpoint ${CHECKPOINT}"
    CMD+=" --model-type ${MODEL_TYPE}"
    CMD+=" --output ${OUTPUT_DIR}/${benchmark}_predictions.jsonl"
    CMD+=" --max-tokens ${MAX_TOKENS}"
    CMD+=" --gpu-memory ${GPU_MEMORY}"
    CMD+=" --batch-size ${BATCH_SIZE}"

    if [[ "${RENDER_QUESTIONS}" == "true" ]]; then
        CMD+=" --render-questions"
    fi

    if [[ -n "${MAX_SAMPLES}" ]]; then
        CMD+=" ${MAX_SAMPLES}"
    fi

    # Run benchmark
    echo "Command: ${CMD}"
    echo ""

    if eval "${CMD}"; then
        echo ""
        echo "✓ ${benchmark^^} complete!"
    else
        echo ""
        echo "❌ ${benchmark^^} failed!"
    fi
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "================================================================================"
echo "Benchmark Results Summary"
echo "================================================================================"
echo ""

for benchmark in "${BENCH_ARRAY[@]}"; do
    benchmark=$(echo "$benchmark" | tr '[:upper:]' '[:lower:]' | xargs)
    pred_file="${OUTPUT_DIR}/${benchmark}_predictions.jsonl"
    acc_file="${OUTPUT_DIR}/${benchmark}_accuracy.json"

    echo "${benchmark^^}:"
    if [[ -f "${acc_file}" ]]; then
        python3 -c "
import json
with open('${acc_file}') as f:
    data = json.load(f)
acc = data.get('accuracy', 0.0)
correct = data.get('correct', '?')
total = data.get('total', '?')
print(f'  Accuracy: {acc*100:.2f}% ({correct}/{total})')
"
    elif [[ -f "${pred_file}" ]]; then
        # Count predictions
        count=$(wc -l < "${pred_file}")
        echo "  Predictions: ${count} samples (no accuracy computed)"
    else
        echo "  (no results found)"
    fi
    echo ""
done

echo "================================================================================"
echo "Full results: ${OUTPUT_DIR}"
echo "================================================================================"
echo ""
echo "✓ All benchmarks complete!"
