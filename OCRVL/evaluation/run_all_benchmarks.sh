#!/bin/bash
set -euo pipefail

# OCRVL Benchmark Runner
#
# Wrapper around Qwen3-VL benchmark suite that:
# - Loads OCRVL-trained connectors on top of base Qwen3-VL model
# - Outputs results to OCRVL/results/* (same naming as Qwen3-VL)
# - Maintains identical interface and functionality
#
# Required:
#   CHECKPOINT_PATH: path to OCRVL checkpoint directory (e.g., OCRVL/checkpoints/alignment_60k_*/step_1000)
#
# Optional (same as Qwen3-VL):
#   MODEL_PATH:  base Qwen3-VL model path (default: auto-detect from checkpoint config.json)
#   DATA_DIR:    shared TSV+decoded-images dir for benchmarks (default: ../Qwen3-VL/data/VLMEval)
#   ODINW_DIR:   ODinW-13 root dir (default: ../Qwen3-VL/data/odinw)
#   RUN_TAG:     run identifier (default: checkpoint dirname + timestamp)
#   NUM_GPUS:    number of GPUs (default: auto-detect)
#   GPUS:        comma-separated GPU IDs (default: auto-detect)
#
# Text-as-images variant:
#   RENDER_TEXT=1 (and optional RENDER_TEXT_* vars)
#
# Example usage:
#   # After training reaches step 1000
#   export CHECKPOINT_PATH=OCRVL/checkpoints/alignment_60k_20251221_023626/step_1000
#   ./OCRVL/evaluation/run_all_benchmarks.sh
#
# Output:
#   OCRVL/results/<checkpoint>_<timestamp>/
#   ├── mmmu/
#   ├── mathvision/
#   ├── realworldqa/
#   ├── odinw13/
#   ├── summary.json
#   ├── summary.csv
#   └── run_all_benchmarks.log

OCRVL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${OCRVL_ROOT}/.." && pwd)"
QWEN_EVAL_ROOT="$(cd "${REPO_ROOT}/../Qwen3-VL/evaluation" && pwd)"

PYTHON="${PYTHON:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_DIR="${DATA_DIR:-${QWEN_EVAL_ROOT}/../data/VLMEval}"
ODINW_DIR="${ODINW_DIR:-${QWEN_EVAL_ROOT}/../data/odinw}"

# Validate checkpoint path
if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "ERROR: set CHECKPOINT_PATH to OCRVL checkpoint directory" >&2
  echo "Example: export CHECKPOINT_PATH=OCRVL/checkpoints/alignment_60k_20251221_023626/step_1000" >&2
  echo "" >&2
  echo "Available checkpoints:" >&2
  find "${OCRVL_ROOT}/checkpoints" -type d -name "step_*" 2>/dev/null || echo "  (none saved yet - checkpoints saved every 1000 steps)" >&2
  exit 1
fi

CHECKPOINT_PATH="$(cd "${CHECKPOINT_PATH}" && pwd)"

if [ ! -d "${CHECKPOINT_PATH}" ]; then
  echo "ERROR: CHECKPOINT_PATH is not a directory: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}/connectors.pt" ]; then
  echo "ERROR: connectors.pt not found in ${CHECKPOINT_PATH}" >&2
  exit 1
fi

# Auto-detect base model path from checkpoint config if not specified
if [ -z "${MODEL_PATH}" ]; then
  CONFIG_JSON="${CHECKPOINT_PATH}/config.json"
  if [ ! -f "${CONFIG_JSON}" ]; then
    # Try parent directory
    CONFIG_JSON="$(dirname "${CHECKPOINT_PATH}")/config.json"
  fi

  if [ -f "${CONFIG_JSON}" ]; then
    MODEL_PATH=$(${PYTHON} -c "import json; print(json.load(open('${CONFIG_JSON}'))['qwen_model_path'])" 2>/dev/null || echo "")
  fi

  if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: Could not auto-detect MODEL_PATH from config. Please set it manually." >&2
    echo "Example: export MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct" >&2
    exit 1
  fi
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: Base model not found: ${MODEL_PATH}" >&2
  exit 1
fi

# Generate run tag from checkpoint name
CHECKPOINT_NAME="$(basename $(dirname "${CHECKPOINT_PATH}"))_$(basename "${CHECKPOINT_PATH}")"
RUN_TAG="${RUN_TAG:-${CHECKPOINT_NAME}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${OCRVL_ROOT}/results/${RUN_TAG}}"

mkdir -p "${OUT_ROOT}"

# Set up logging
GLOBAL_LOG="${OUT_ROOT}/run_all_benchmarks.log"
exec > >(tee -a "${GLOBAL_LOG}") 2>&1

echo "================================================================================"
echo "OCRVL Benchmark Suite Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "Global log: ${GLOBAL_LOG}"
echo "================================================================================"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "MODEL_PATH=${MODEL_PATH} (base Qwen3-VL model)"
echo "DATA_DIR=${DATA_DIR}"
echo "ODINW_DIR=${ODINW_DIR}"
echo "RUN_TAG=${RUN_TAG}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "PYTHON=${PYTHON}"
echo "RENDER_TEXT=${RENDER_TEXT:-0}"
echo ""
echo "OCRVL Connector Configuration:"
echo "  Connectors: ${CHECKPOINT_PATH}/connectors.pt"
echo "  Training config: $(dirname ${CHECKPOINT_PATH})/config.json"
echo ""

# Export OCRVL-specific environment variables for inference scripts
export OCRVL_CHECKPOINT_PATH="${CHECKPOINT_PATH}"
export OCRVL_CONNECTORS_PATH="${CHECKPOINT_PATH}/connectors.pt"
export OCRVL_MODE=1

# Call Qwen3-VL benchmark suite with modified paths
echo "Delegating to Qwen3-VL benchmark suite..."
echo "================================================================================"
cd "${QWEN_EVAL_ROOT}"

# Run Qwen3-VL benchmarks with our model + connectors
PYTHON="${PYTHON}" \
MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
ODINW_DIR="${ODINW_DIR}" \
RUN_TAG="${RUN_TAG}" \
OUT_ROOT="${OUT_ROOT}" \
RENDER_TEXT="${RENDER_TEXT:-0}" \
RENDER_TEXT_INSTRUCTION="${RENDER_TEXT_INSTRUCTION:-}" \
RENDER_TEXT_CHUNK_TOKENS="${RENDER_TEXT_CHUNK_TOKENS:-}" \
RENDER_TEXT_SIZE="${RENDER_TEXT_SIZE:-}" \
RENDER_TEXT_BACKEND="${RENDER_TEXT_BACKEND:-}" \
RENDER_TEXT_CACHE_DIR="${RENDER_TEXT_CACHE_DIR:-}" \
NUM_GPUS="${NUM_GPUS:-}" \
GPUS="${GPUS:-}" \
bash run_all_benchmarks.sh

echo ""
echo "================================================================================"
echo "OCRVL Benchmark Suite Completed: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "================================================================================"
echo "Results saved to: ${OUT_ROOT}"
echo ""
echo "NOTE: This uses the standard Qwen3-VL inference scripts."
echo "To enable OCRVL connector loading, you need to modify the Qwen3-VL model"
echo "loading code in the inference scripts, or use OCRVL's custom inference wrapper."
echo ""
echo "Next steps:"
echo "1. Create OCRVL-aware inference wrapper that loads connectors"
echo "2. Modify Qwen3-VL/evaluation/*/run_*.py to use OCRQwen3VL models"
echo "3. Re-run this script to test with OCRVL connectors"
