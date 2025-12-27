#!/bin/bash
set -euo pipefail

# OCRVL Centralized Evaluation Suite
# Unified interface for evaluating OCRVL checkpoints on vision-language benchmarks
#
# Usage:
#   bash OCRVL/evaluation/run_benchmarks.sh [options] <checkpoint_dir>
#   
#   Examples:
#     # RealWorldQA only (fast vLLM path)
#     bash OCRVL/evaluation/run_benchmarks.sh -b realworldqa OCRVL/checkpoints/.../step_786
#     
#     # Run any of the Qwen3-VL benchmarks (using HF backend via exported model)
#     bash OCRVL/evaluation/run_benchmarks.sh -b mmmu,mathvision,odinw13 --backend hf OCRVL/checkpoints/.../step_786
#
# Options (flags override env vars):
#   -b, --bench    Comma-separated benchmarks to run. Choices: realworldqa, mmmu, mathvision, odinw13
#                  Aliases accepted: odinw -> odinw13
#   -d, --data     Data directory (default: /share/project/xiyan/data)
#   -o, --out      Output root directory (default: OCRVL/evaluation/results/<RUN_TAG>)
#   -k, --backend  vllm|hf (default: vllm). Non-RealWorldQA benchmarks require hf.
#   -t, --tag      Run tag to use in output folder naming (default: timestamp)
#   -h, --help     Show help
#
# Environment Variables (alternative to flags):
#   BENCHMARKS    Same as --bench (default: "realworldqa,mmmu,mathvision,odinw13")
#   DATA_DIR      Same as --data (default: /share/project/xiyan/data)
#   NUM_GPUS      Number of GPUs for data parallelism (currently informational)
#   BACKEND       Same as --backend (default: vllm)
#   EVAL_MODEL    Default judge model id for MMMU/MathVision eval (e.g., gpt-4o, gpt-4o-mini)
#   API_TYPE      Default API type for judge (dash|mit); used when EVAL_MODEL present
#   EVAL_MODEL_MMMU / API_TYPE_MMMU           Override for MMMU only
#   EVAL_MODEL_MATHVISION / API_TYPE_MATHVISION Override for MathVision only
#
# Commands
# tmux new -d -s ocrqwen_bench "EVAL_MODEL=qwen3-next API_TYPE=mit CUDA_VISIBLE_DEVICES=0 bash OCRVL/evaluation/run_benchmarks.sh -b realworldqa,mmmu,mathvision,odinw13 -d /share/project/xiyan/data OCRVL/checkpoints/alignment_long_20251223_043933/step_786"

# ============================================================================
# Configuration
# ============================================================================

# Defaults
CHECKPOINT_DIR=""
USER_BENCH_ARG="${BENCHMARKS:-}"   # env fallback if no flag
DATA_DIR="${DATA_DIR:-/share/project/xiyan/data}"
NUM_GPUS="${NUM_GPUS:-1}"
BACKEND="${BACKEND:-vllm}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

REPO_ROOT="/share/project/xiyan/sources/DeepSeek-OCR"
QWEN_EVAL_ROOT="/share/project/xiyan/sources/Qwen3-VL/evaluation"
OUT_ROOT=""

# Resolve default ODINW data directory
# Priority: explicit $ODINW_DIR -> $DATA_DIR/odinw -> $QWEN_EVAL_ROOT/../data/odinw -> $DATA_DIR/odinw
if [ -z "${ODINW_DIR:-}" ]; then
  if [ -d "$DATA_DIR/odinw" ]; then
    ODINW_DIR_DEFAULT="$DATA_DIR/odinw"
  elif [ -d "$(readlink -f "$QWEN_EVAL_ROOT/../data/odinw")" ]; then
    ODINW_DIR_DEFAULT="$(readlink -f "$QWEN_EVAL_ROOT/../data/odinw")"
  else
    ODINW_DIR_DEFAULT="$DATA_DIR/odinw"
  fi
else
  ODINW_DIR_DEFAULT="$ODINW_DIR"
fi
export ODINW_DIR="$ODINW_DIR_DEFAULT"

# Default judge model for MMMU/MathVision when using local OpenAI-compatible vLLM
DEFAULT_JUDGE_MODEL="qwen3-next"

# Concurrency for evaluation phases
# Judge eval (MMMU/MathVision): configurable high defaults
# RealWorldQA eval is CPU-bound; allow tuning as well
EVAL_NPROC_MMMU="${EVAL_NPROC_MMMU:-128}"
EVAL_NPROC_MATHVISION="${EVAL_NPROC_MATHVISION:-128}"
EVAL_NPROC_REALWORLD="${EVAL_NPROC_REALWORLD:-32}"

# ----------------------------------------------------------------------------
# Parse flags
# ----------------------------------------------------------------------------
print_help() {
  sed -n '1,80p' "$0" | sed -n '1,80p' | sed -n '1,40p' | sed 's/^# \{0,1\}//' | sed 's/^$//' | sed '1,/Options/d' >/dev/null
  cat <<USAGE
OCRVL Centralized Evaluation Suite

Usage: $0 [options] <checkpoint_dir>

Options:
  -b, --bench    Comma-separated list of benchmarks to run (realworldqa, mmmu, mathvision, odinw13)
  -d, --data     Data directory (default: /share/project/xiyan/data)
  -o, --out      Output root directory (default: OCRVL/evaluation/results/<RUN_TAG>)
  -k, --backend  Backend for inference: vllm|hf (default: vllm)
  -t, --tag      Run tag to use in output folder naming (default: timestamp)
  -h, --help     Show this help and exit

Environment:
  BENCHMARKS, DATA_DIR, NUM_GPUS, BACKEND, RUN_TAG
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--bench)
      USER_BENCH_ARG="$2"; shift 2;;
    -d|--data)
      DATA_DIR="$2"; shift 2;;
    -o|--out)
      OUT_ROOT="$2"; shift 2;;
    -k|--backend)
      BACKEND="$2"; shift 2;;
    -t|--tag)
      RUN_TAG="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    --)
      shift; break;;
    -*)
      echo "Unknown option: $1"; print_help; exit 1;;
    *)
      # First non-flag is checkpoint dir
      if [[ -z "$CHECKPOINT_DIR" ]]; then
        CHECKPOINT_DIR="$1"; shift
      else
        echo "Unexpected positional argument: $1"; print_help; exit 1
      fi
      ;;
  esac
done

if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="$REPO_ROOT/OCRVL/evaluation/results/${RUN_TAG}"
fi

# Normalize benchmarks (flags > env > default)
if [[ -z "$USER_BENCH_ARG" ]]; then
  USER_BENCH_ARG="realworldqa,mmmu,mathvision,odinw13"
fi
BENCHMARKS="$USER_BENCH_ARG"

# ---------------------------------------------------------------------------
# Judge defaults for MMMU/MathVision (local vLLM @ http://localhost:8000)
# RealWorldQA and ODinW do not use a judge.
# These defaults apply only if the user didn't provide EVAL_MODEL* / API_TYPE*.
# ---------------------------------------------------------------------------
if [ -z "${EVAL_MODEL_MMMU:-}" ] && [ -z "${EVAL_MODEL:-}" ]; then
  EVAL_MODEL_MMMU="$DEFAULT_JUDGE_MODEL"
fi
if [ -z "${API_TYPE_MMMU:-}" ] && [ -z "${API_TYPE:-}" ]; then
  API_TYPE_MMMU="mit"
fi
if [ -z "${EVAL_MODEL_MATHVISION:-}" ] && [ -z "${EVAL_MODEL:-}" ]; then
  EVAL_MODEL_MATHVISION="$DEFAULT_JUDGE_MODEL"
fi
if [ -z "${API_TYPE_MATHVISION:-}" ] && [ -z "${API_TYPE:-}" ]; then
  API_TYPE_MATHVISION="mit"
fi

# If either benchmark uses mit (OpenAI-compatible), set local defaults
if [ "${API_TYPE_MMMU:-}" = "mit" ] || [ "${API_TYPE_MATHVISION:-}" = "mit" ] || [ "${API_TYPE:-}" = "mit" ]; then
  export MIT_SPIDER_URL="${MIT_SPIDER_URL:-http://localhost:8000/v1/chat/completions}"
  export MIT_SPIDER_TOKEN="${MIT_SPIDER_TOKEN:-sk-local}"
fi

# ============================================================================
# Validation
# ============================================================================

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: missing <checkpoint_dir>"
    print_help
    exit 1
fi

# Convert to absolute path
if [[ ! "$CHECKPOINT_DIR" = /* ]]; then
    CHECKPOINT_DIR="$REPO_ROOT/$CHECKPOINT_DIR"
fi

if [ ! -f "$CHECKPOINT_DIR/connectors.pt" ]; then
    echo "❌ Error: connectors.pt not found in $CHECKPOINT_DIR"
    exit 1
fi

# Validate backend
if [ "$BACKEND" != "vllm" ] && [ "$BACKEND" != "hf" ]; then
    echo "❌ Error: Invalid backend '$BACKEND'. Must be 'vllm' or 'hf'"
    exit 1
fi

mkdir -p "$OUT_ROOT"

# ----------------------------------------------------------------------------
# Track invocation and environment for reproducibility
# ----------------------------------------------------------------------------
INVOCATION="$(ps -o args= -p $$ 2>/dev/null || true)"
HOSTNAME="$(hostname 2>/dev/null || echo unknown)"
DATE_ISO="$(date -Iseconds)"

DEEPSEEK_GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
DEEPSEEK_GIT_DIRTY="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l | awk '{print $1}')"
QWEN_ROOT="$(dirname "$QWEN_EVAL_ROOT")"
QWEN_GIT_COMMIT="$(git -C "$QWEN_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
QWEN_GIT_DIRTY="$(git -C "$QWEN_ROOT" status --porcelain 2>/dev/null | wc -l | awk '{print $1}')"

cat > "$OUT_ROOT/reproduce.sh" <<REPRO
#!/usr/bin/env bash
set -euo pipefail
export BENCHMARKS="${BENCHMARKS}"
export DATA_DIR="${DATA_DIR}"
export BACKEND="${BACKEND}"
export NUM_GPUS="${NUM_GPUS}"
export RUN_TAG="${RUN_TAG}"
exec bash ${REPO_ROOT}/OCRVL/evaluation/run_benchmarks.sh -b "${BENCHMARKS}" -d "${DATA_DIR}" -o "${OUT_ROOT}" -k "${BACKEND}" -t "${RUN_TAG}" "${CHECKPOINT_DIR}"
REPRO
chmod +x "$OUT_ROOT/reproduce.sh"

cat > "$OUT_ROOT/run_info.json" <<JSON
{
  "timestamp": "${DATE_ISO}",
  "host": "${HOSTNAME}",
  "invocation": "${INVOCATION}",
  "checkpoint_dir": "${CHECKPOINT_DIR}",
  "benchmarks": "${BENCHMARKS}",
  "backend": "${BACKEND}",
  "data_dir": "${DATA_DIR}",
  "out_root": "${OUT_ROOT}",
  "num_gpus": "${NUM_GPUS}",
  "run_tag": "${RUN_TAG}",
  "repos": {
    "DeepSeek-OCR": {"path": "${REPO_ROOT}", "commit": "${DEEPSEEK_GIT_COMMIT}", "dirty_files": ${DEEPSEEK_GIT_DIRTY} },
    "Qwen3-VL": {"path": "${QWEN_ROOT}", "commit": "${QWEN_GIT_COMMIT}", "dirty_files": ${QWEN_GIT_DIRTY} }
  }
}
JSON

# Copy checkpoint training config (if present) for provenance
if [ -f "$(dirname "$CHECKPOINT_DIR")/config.json" ]; then
  cp -f "$(dirname "$CHECKPOINT_DIR")/config.json" "$OUT_ROOT/checkpoint_config.json" || true
fi

# Save exact command and environment snapshot
{
  echo "Command: $0 $*"
  echo "Date: ${DATE_ISO}"
  echo "Host: ${HOSTNAME}"
  echo "CWD: $(pwd)"
  echo
  echo "Environment (selected):"
  env | rg -n '^(BENCHMARKS|DATA_DIR|BACKEND|NUM_GPUS|RUN_TAG|CUDA|NVIDIA|PYTHON|CONDA|HF_|VLLM_|LMUData|ODINW_DIR)=' || true
} > "$OUT_ROOT/cmd.txt"

# ============================================================================
# Display Configuration
# ============================================================================

echo "================================================================================"
echo "OCRVL Evaluation Suite"
echo "================================================================================"
echo "Checkpoint:  $CHECKPOINT_DIR"
echo "Benchmarks:  $BENCHMARKS"
echo "Backend:     $BACKEND"
echo "Data Dir:    $DATA_DIR"
echo "Output Dir:  $OUT_ROOT"
echo "GPUs:        $NUM_GPUS"
echo "Run Tag:     $RUN_TAG"
echo "================================================================================"
echo ""

# Set up global logging
GLOBAL_LOG="$OUT_ROOT/evaluation.log"
exec > >(tee -a "$GLOBAL_LOG") 2>&1

# ============================================================================
# Step 1: Prepare Model (export HF when required)
# ============================================================================

MODEL_EXPORT_DIR="$OUT_ROOT/model_export"

# We only need an exported HF model if the user explicitly requests
# HF backend for RealWorldQA inference. All other benchmarks now use
# unified OCRVL vLLM generation and do not need export.
NEED_EXPORT="0"
if [[ "$BACKEND" == "hf" ]]; then
  IFS=',' read -ra __TMP_BENCH_ARR <<< "$BENCHMARKS"
  for b in "${__TMP_BENCH_ARR[@]}"; do
    b_norm=$(echo "$b" | tr '[:upper:]' '[:lower:]' | xargs)
    [[ "$b_norm" == "odinw" ]] && b_norm="odinw13"
    if [[ "$b_norm" == "realworldqa" ]]; then
      NEED_EXPORT="1"
    fi
  done
fi

if [ "$NEED_EXPORT" = "1" ]; then
    echo "Step 1/3: Exporting HuggingFace checkpoint (required for selected benchmarks)..."
    echo "------------------------------------------------------------"
    if [ -d "$MODEL_EXPORT_DIR" ]; then
        echo "⚠️  Model already exported at $MODEL_EXPORT_DIR"
        echo "   Skipping export step"
    else
        PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python "$REPO_ROOT/OCRVL/evaluation/export_hf_checkpoint.py" \
            --checkpoint "$CHECKPOINT_DIR" \
            --output "$MODEL_EXPORT_DIR"
        if [ $? -ne 0 ]; then
            echo "❌ Export failed"
            exit 1
        fi
        echo "✓ Model exported to: $MODEL_EXPORT_DIR"
    fi
    echo ""
else
    echo "Step 1/3: Using vLLM direct path (no HF export needed for RealWorldQA)"
    echo "------------------------------------------------------------"
    echo ""
fi

# ============================================================================
# Step 2: Run Benchmarks (with optional data parallelism across GPUs)
# ============================================================================

echo "Step 2/3: Running benchmarks..."
echo "------------------------------------------------------------"

IFS=',' read -ra BENCHMARK_ARRAY <<< "$BENCHMARKS"

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
    benchmark=$(echo "$benchmark" | tr '[:upper:]' '[:lower:]' | xargs)
    # alias
    if [[ "$benchmark" == "odinw" ]]; then benchmark="odinw13"; fi

    echo ""
    echo "Running: $benchmark"
    echo "============================================================"

    case "$benchmark" in
        realworldqa)
            # Ensure per-benchmark dir
            RWQ_DIR="$OUT_ROOT/realworldqa"
            mkdir -p "$RWQ_DIR"

            # vLLM path with OCRVL connectors; supports sharded data-parallel like Qwen's run_all_benchmarks.sh
            # Detect available GPUs from CUDA_VISIBLE_DEVICES or nvidia-smi
            GPULIST="${GPUS:-}"
            if [ -z "$GPULIST" ]; then
              if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
                GPULIST="$CUDA_VISIBLE_DEVICES"
              else
                GPULIST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 5000 {print $1}' | paste -sd, -)
                [ -z "$GPULIST" ] && GPULIST=0
              fi
            fi
            IFS=',' read -ra GPU_ARRAY <<< "$GPULIST"
            NUM_SHARDS=${#GPU_ARRAY[@]}

            if [ "$NUM_SHARDS" -le 1 ]; then
              PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]:-0}" \
                python "$REPO_ROOT/OCRVL/evaluation/eval_realworldqa_vllm.py" \
                  --checkpoint "$CHECKPOINT_DIR" \
                  --data-dir "$DATA_DIR" \
                  --output "$RWQ_DIR/RealWorldQA_results.jsonl" \
                  --batch-size 8 \
                  --max-tokens 512
            else
              echo "Launching data-parallel shards for RealWorldQA on GPUs: ${GPU_ARRAY[*]}"
              pids=()
              for i in "${!GPU_ARRAY[@]}"; do
                gid="${GPU_ARRAY[$i]}"
                shard_out="$RWQ_DIR/shard_${i}_RealWorldQA_results.jsonl"
                (
                  PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
                  CUDA_VISIBLE_DEVICES="$gid" \
                  SHARD_OUTPUT="$shard_out" \
                  python "$REPO_ROOT/OCRVL/evaluation/eval_realworldqa_vllm.py" \
                    --checkpoint "$CHECKPOINT_DIR" \
                    --data-dir "$DATA_DIR" \
                    --output "$shard_out" \
                    --batch-size 8 \
                    --max-tokens 512 \
                    --shard-id "$i" \
                    --num-shards "$NUM_SHARDS" 
                ) &
                pids+=($!)
              done
              # Wait for shards and reap any stragglers
              for p in "${pids[@]}"; do wait "$p" || true; done
              # Defensive: if any shard Python still lingers (vLLM workers), terminate
              pkill -f "eval_realworldqa_vllm.py" || true
              cat "$RWQ_DIR"/shard_*_RealWorldQA_results.jsonl > "$RWQ_DIR/RealWorldQA_results.jsonl"
            fi

            # Evaluate
            (
              cd "$QWEN_EVAL_ROOT" && \
              python RealWorldQA/run_realworldqa.py eval \
                --data-dir "$DATA_DIR" \
                --input-file "$RWQ_DIR/RealWorldQA_results.jsonl" \
                --output-file "$RWQ_DIR/RealWorldQA_evaluation.csv" \
                --dataset RealWorldQA \
                ${EVAL_MODEL_REALWORLD:+--eval-model "$EVAL_MODEL_REALWORLD"} \
                ${API_TYPE_REALWORLD:+--api-type "$API_TYPE_REALWORLD"} \
                ${EVAL_MODEL:+--eval-model "$EVAL_MODEL"} \
                ${API_TYPE:+--api-type "$API_TYPE"} \
                --nproc ${EVAL_NPROC_REALWORLD}
            ) || { echo "❌ RealWorldQA evaluation failed"; continue; }

            echo "✓ RealWorldQA complete"
            ;;

        mmmu)
            MMMU_DIR="$OUT_ROOT/mmmu"
            mkdir -p "$MMMU_DIR"

            # Use unified OCRVL vLLM generator then Qwen evaluator
            # Data-parallel MMMU
            GPULIST="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
            [ -z "$GPULIST" ] && GPULIST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 5000 {print $1}' | paste -sd, -)
            [ -z "$GPULIST" ] && GPULIST=0
            IFS=',' read -ra GPU_ARRAY <<< "$GPULIST"
            NUM_SHARDS=${#GPU_ARRAY[@]}
            if [ "$NUM_SHARDS" -le 1 ]; then
              PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]:-0}" \
                python "$REPO_ROOT/OCRVL/evaluation/eval_mmmu_vllm.py" \
                  --checkpoint "$CHECKPOINT_DIR" \
                  --data-dir "$DATA_DIR" \
                  --output "$MMMU_DIR/mmmu_dev_val_predictions.jsonl" \
                  --batch-size 8 \
                  --max-tokens 512 || { echo "❌ MMMU generation failed"; continue; }
            else
              echo "Launching data-parallel shards for MMMU on GPUs: ${GPU_ARRAY[*]}"
              pids=()
              for i in "${!GPU_ARRAY[@]}"; do
                gid="${GPU_ARRAY[$i]}"
                shard_out="$MMMU_DIR/shard_${i}_mmmu_dev_val_predictions.jsonl"
                (
                  PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
                  CUDA_VISIBLE_DEVICES="$gid" \
                  SHARD_OUTPUT="$shard_out" \
                  python "$REPO_ROOT/OCRVL/evaluation/eval_mmmu_vllm.py" \
                    --checkpoint "$CHECKPOINT_DIR" \
                    --data-dir "$DATA_DIR" \
                    --output "$shard_out" \
                    --batch-size 8 \
                    --max-tokens 512 \
                    --shard-id "$i" \
                    --num-shards "$NUM_SHARDS"
                ) &
                pids+=($!)
              done
              for p in "${pids[@]}"; do wait "$p" || true; done
              pkill -f "eval_mmmu_vllm.py" || true
              cat "$MMMU_DIR"/shard_*_mmmu_dev_val_predictions.jsonl > "$MMMU_DIR/mmmu_dev_val_predictions.jsonl"
            fi

            (
              cd "$QWEN_EVAL_ROOT" && \
              python mmmu/run_mmmu.py eval \
                --data-dir "$DATA_DIR" \
                --input-file "$MMMU_DIR/mmmu_dev_val_predictions.jsonl" \
                --output-file "$MMMU_DIR/mmmu_dev_val_eval_results.csv" \
                --dataset MMMU_DEV_VAL \
                ${EVAL_MODEL_MMMU:+--eval-model "$EVAL_MODEL_MMMU"} \
                ${API_TYPE_MMMU:+--api-type "$API_TYPE_MMMU"} \
                ${EVAL_MODEL:+--eval-model "$EVAL_MODEL"} \
                ${API_TYPE:+--api-type "$API_TYPE"} \
                --nproc ${EVAL_NPROC_MMMU}
            ) || { echo "❌ MMMU evaluation failed"; continue; }

            echo "✓ MMMU complete"
            ;;

        mathvision)
            MV_DIR="$OUT_ROOT/mathvision"
            mkdir -p "$MV_DIR"

            # Data-parallel MathVision
            GPULIST="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
            [ -z "$GPULIST" ] && GPULIST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 5000 {print $1}' | paste -sd, -)
            [ -z "$GPULIST" ] && GPULIST=0
            IFS=',' read -ra GPU_ARRAY <<< "$GPULIST"
            NUM_SHARDS=${#GPU_ARRAY[@]}
            if [ "$NUM_SHARDS" -le 1 ]; then
              PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]:-0}" \
                python "$REPO_ROOT/OCRVL/evaluation/eval_mathvision_vllm.py" \
                  --checkpoint "$CHECKPOINT_DIR" \
                  --data-dir "$DATA_DIR" \
                  --dataset MathVision \
                  --output "$MV_DIR/mathvision_predictions.jsonl" \
                  --batch-size 8 \
                  --max-tokens 512 || { echo "❌ MathVision generation failed"; continue; }
            else
              echo "Launching data-parallel shards for MathVision on GPUs: ${GPU_ARRAY[*]}"
              pids=()
              for i in "${!GPU_ARRAY[@]}"; do
                gid="${GPU_ARRAY[$i]}"
                shard_out="$MV_DIR/shard_${i}_mathvision_predictions.jsonl"
                (
                  PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
                  CUDA_VISIBLE_DEVICES="$gid" \
                  SHARD_OUTPUT="$shard_out" \
                  python "$REPO_ROOT/OCRVL/evaluation/eval_mathvision_vllm.py" \
                    --checkpoint "$CHECKPOINT_DIR" \
                    --data-dir "$DATA_DIR" \
                    --dataset MathVision \
                    --output "$shard_out" \
                    --batch-size 8 \
                    --max-tokens 512 \
                    --shard-id "$i" \
                    --num-shards "$NUM_SHARDS"
                ) &
                pids+=($!)
              done
              for p in "${pids[@]}"; do wait "$p" || true; done
              pkill -f "eval_mathvision_vllm.py" || true
              cat "$MV_DIR"/shard_*_mathvision_predictions.jsonl > "$MV_DIR/mathvision_predictions.jsonl"
            fi

            (
              cd "$QWEN_EVAL_ROOT" && \
              python MathVision/run_mathv.py eval \
                --data-dir "$DATA_DIR" \
                --input-file "$MV_DIR/mathvision_predictions.jsonl" \
                --output-file "$MV_DIR/mathvision_eval_results.csv" \
                --dataset MathVision \
                ${EVAL_MODEL_MATHVISION:+--eval-model "$EVAL_MODEL_MATHVISION"} \
                ${API_TYPE_MATHVISION:+--api-type "$API_TYPE_MATHVISION"} \
                ${EVAL_MODEL:+--eval-model "$EVAL_MODEL"} \
                ${API_TYPE:+--api-type "$API_TYPE"} \
                --nproc ${EVAL_NPROC_MATHVISION}
            ) || { echo "❌ MathVision evaluation failed"; continue; }

            echo "✓ MathVision complete"
            ;;

        odinw13)
            OD_DIR="$OUT_ROOT/odinw13"
            mkdir -p "$OD_DIR"

            # Data-parallel ODinW-13
            # Skip gracefully if ODinW data directory is missing
            if [ ! -d "$ODINW_DIR" ]; then
              echo "⚠️  ODinW data dir not found at $ODINW_DIR; skipping ODinW-13"
              continue
            fi
            GPULIST="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
            [ -z "$GPULIST" ] && GPULIST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 5000 {print $1}' | paste -sd, -)
            [ -z "$GPULIST" ] && GPULIST=0
            IFS=',' read -ra GPU_ARRAY <<< "$GPULIST"
            NUM_SHARDS=${#GPU_ARRAY[@]}
            if [ "$NUM_SHARDS" -le 1 ]; then
              PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]:-0}" \
                python "$REPO_ROOT/OCRVL/evaluation/eval_odinw_vllm.py" \
                  --checkpoint "$CHECKPOINT_DIR" \
                  --data-dir "$DATA_DIR" \
                  --output "$OD_DIR/odinw_predictions.jsonl" \
                  --batch-size 4 \
                  --max-tokens 256 || { echo "❌ ODinW-13 generation failed"; continue; }
            else
              echo "Launching data-parallel shards for ODinW-13 on GPUs: ${GPU_ARRAY[*]}"
              pids=()
              for i in "${!GPU_ARRAY[@]}"; do
                gid="${GPU_ARRAY[$i]}"
                shard_out="$OD_DIR/shard_${i}_odinw_predictions.jsonl"
                (
                  PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
                  CUDA_VISIBLE_DEVICES="$gid" \
                  SHARD_OUTPUT="$shard_out" \
                  python "$REPO_ROOT/OCRVL/evaluation/eval_odinw_vllm.py" \
                    --checkpoint "$CHECKPOINT_DIR" \
                    --data-dir "$DATA_DIR" \
                    --output "$shard_out" \
                    --batch-size 4 \
                    --max-tokens 256 \
                    --shard-id "$i" \
                    --num-shards "$NUM_SHARDS"
                ) &
                pids+=($!)
              done
              for p in "${pids[@]}"; do wait "$p" || true; done
              pkill -f "eval_odinw_vllm.py" || true
              # Merge shard outputs if any; allow missing shards without aborting the whole run
              cat "$OD_DIR"/shard_*_odinw_predictions.jsonl > "$OD_DIR/odinw_predictions.jsonl" 2>/dev/null || true
            fi

            # Only run ODinW evaluation if predictions exist and are non-empty
            if [ -s "$OD_DIR/odinw_predictions.jsonl" ]; then
              (
                cd "$QWEN_EVAL_ROOT" && \
                python ODinW-13/run_odinw.py eval \
                  --data-dir "$ODINW_DIR" \
                  --input-file "$OD_DIR/odinw_predictions.jsonl" \
                  --output-file "$OD_DIR/odinw_eval_results.json"
              ) || { echo "❌ ODinW-13 evaluation failed"; continue; }
            else
              echo "⚠️  No ODinW predictions produced; skipping ODinW-13 evaluation"
            fi

            echo "✓ ODinW-13 complete"
            ;;

        *)
            echo "❌ Unknown benchmark: $benchmark"
            echo "   Available: realworldqa, mmmu, mathvision, odinw13"
            ;;
    esac
done

echo ""

# ============================================================================
# Step 3: Aggregate Results
# ============================================================================

echo "Step 3/3: Aggregating results..."
echo "------------------------------------------------------------"

RESULTS_SUMMARY="$OUT_ROOT/results_summary.txt"

{
    echo "================================================================================"
    echo "OCRVL Evaluation Results"
    echo "================================================================================"
    echo "Checkpoint: $CHECKPOINT_DIR"
    echo "Run Tag:    $RUN_TAG"
    echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    echo ""

    for benchmark in "${BENCHMARK_ARRAY[@]}"; do
        benchmark=$(echo "$benchmark" | tr '[:upper:]' '[:lower:]' | xargs)
        if [[ "$benchmark" == "odinw" ]]; then benchmark="odinw13"; fi

        case "$benchmark" in
            realworldqa)
                echo "RealWorldQA:"
                ACC_JSON="$OUT_ROOT/realworldqa/RealWorldQA_evaluation_acc.json"
                if [ -f "$ACC_JSON" ]; then
                    python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); acc=d.get('overall_accuracy',0.0); c=d.get('correct','?'); t=d.get('total','?'); print(f'  Accuracy: {acc*100:.2f}% ({c}/{t} correct)')" "$ACC_JSON"
                else
                    # Fallback to OCRVL vLLM evaluator output
                    VLLM_ACC_JSON="$OUT_ROOT/realworldqa/accuracy.json"
                    if [ -f "$VLLM_ACC_JSON" ]; then
                        python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); acc=d.get('accuracy',0.0); c=d.get('correct','?'); t=d.get('total','?'); print(f'  Accuracy: {acc*100:.2f}% ({c}/{t} correct)')" "$VLLM_ACC_JSON"
                    else
                        echo "  (no metrics found)"
                    fi
                fi
                echo ""
                ;;
            mmmu)
                ACC_JSON="$OUT_ROOT/mmmu/mmmu_dev_val_eval_results_acc.json"
                if [ -f "$ACC_JSON" ]; then
                    echo "MMMU:"
                    python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); acc=d.get('overall_accuracy',0.0); dev=d.get('accuracy_by_split',{}).get('dev',0.0); val=d.get('accuracy_by_split',{}).get('validation',0.0); print(f'  Accuracy: {acc*100:.2f}% (dev={dev*100:.2f}%, val={val*100:.2f}%)')" "$ACC_JSON"
                    echo ""
                fi
                ;;
            mathvision)
                SCORE_CSV="$OUT_ROOT/mathvision/mathvision_eval_results_eval_score.csv"
                if [ -f "$SCORE_CSV" ]; then
                    echo "MathVision:"
                    python3 -c "import csv,sys; p=sys.argv[1]; r=csv.DictReader(open(p)); row=next(r); acc=float(row.get('acc',0.0)); tot=row.get('tot','?'); print(f'  Accuracy: {acc:.2f}% (total={tot})')" "$SCORE_CSV"
                    echo ""
                fi
                ;;
            odinw13)
                RES_JSON="$OUT_ROOT/odinw13/odinw_eval_results.json"
                if [ -f "$RES_JSON" ]; then
                    echo "ODinW-13:"
                    python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); avg=d.get('Average',0.0); print(f'  mAP: {avg*100:.2f}%')" "$RES_JSON"
                    echo ""
                fi
                ;;
        esac
    done

    echo "================================================================================"
    echo "Full results saved to: $OUT_ROOT"
    echo "================================================================================"
} | tee "$RESULTS_SUMMARY"

echo ""
echo "✓ Evaluation complete!"
echo ""
echo "Results summary: $RESULTS_SUMMARY"
echo "Full logs:       $GLOBAL_LOG"
