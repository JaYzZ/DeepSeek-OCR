#!/bin/bash
# Qwen3VL CHIMERA SFT Training with Latent Supervision
#
# This script trains Qwen3VL-2B-Thinking on R1-OneVision dataset using:
# - Pre-encoded vision features (no encoding during training)
# - Latent injection at <latent> positions
# - Thinking loss (REPA/OT/NCE/MSE) on latent predictions
#
# Usage:
#   bash Qwen/scripts/train_qwen3vl_chimera.sh [config.yaml]
#   tmux new-session -d -s chi_sft 'bash Qwen/scripts/train_qwen3vl_chimera.sh Qwen/configs/qwen3vl_native_chimera_thinking.yaml'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Required python interpreter (OCRFlow env)
PYTHON_BIN="$REPO_ROOT/../../envs/ocrflow/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ Python not found or not executable: $PYTHON_BIN" >&2
  echo "   Please ensure OCRFlow env exists at: $REPO_ROOT/../../envs/ocrflow" >&2
  exit 1
fi

# Default config
DEFAULT_CONFIG="$REPO_ROOT/Qwen/configs/qwen3vl_native_chimera_thinking.yaml"
DEFAULT_RUNTIME_ENV_CONFIG="$REPO_ROOT/Qwen/configs/qwen3vl_runtime_env.yaml"

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
if [ "${1:-}" != "" ]; then
  shift || true
fi

QWEN3VL_RUNTIME_ENV_CONFIG="${QWEN3VL_RUNTIME_ENV_CONFIG:-$DEFAULT_RUNTIME_ENV_CONFIG}"
if [ ! -f "$QWEN3VL_RUNTIME_ENV_CONFIG" ]; then
  echo "❌ Runtime env config not found: $QWEN3VL_RUNTIME_ENV_CONFIG" >&2
  exit 1
fi

# Derive model path from config (used for HF_MODULES_CACHE if present)
MODEL_PATH="$(grep -E '^model_name_or_path:' "$CONFIG_PATH" | head -n 1 | awk '{print $2}')"

# ============================================================================
# Export config values as environment variables for Python callbacks
# ============================================================================
# Helper to read a yaml key (supports dotted paths, e.g. "runtime.backfill_enable")
_get_yaml_value() {
    local file_path="$1"
    local key="$2"
    local default_value="${3:-}"
    "$PYTHON_BIN" - "$file_path" "$key" "$default_value" <<'PY'
import sys
from pathlib import Path
import yaml

file_path, key, default_value = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    data = yaml.safe_load(Path(file_path).read_text()) or {}
except Exception:
    print(default_value)
    raise SystemExit(0)

value = data
for part in key.split("."):
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        value = default_value
        break

if value is None:
    value = default_value
if isinstance(value, bool):
    print("1" if value else "0")
else:
    print(str(value))
PY
}

_get_main_config() {
    _get_yaml_value "$CONFIG_PATH" "$1" "${2:-}"
}

_get_runtime_config() {
    _get_yaml_value "$QWEN3VL_RUNTIME_ENV_CONFIG" "$1" "${2:-}"
}

_set_env_from_runtime() {
    local env_name="$1"
    local runtime_key="$2"
    local default_value="$3"
    local current_value="${!env_name:-}"
    if [[ -n "$current_value" ]]; then
        export "$env_name=$current_value"
    else
        export "$env_name=$(_get_runtime_config "$runtime_key" "$default_value")"
    fi
}

_set_env_from_main() {
    local env_name="$1"
    local main_key="$2"
    local default_value="$3"
    local current_value="${!env_name:-}"
    if [[ -n "$current_value" ]]; then
        export "$env_name=$current_value"
    else
        export "$env_name=$(_get_main_config "$main_key" "$default_value")"
    fi
}

_resolve_master_port() {
    "$PYTHON_BIN" - "${MASTER_PORT:-}" <<'PY'
import socket
import sys

preferred = sys.argv[1].strip()

def is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()

if not preferred:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    print(sock.getsockname()[1])
    sock.close()
    raise SystemExit(0)

port = max(1, min(int(preferred), 65535))
while port <= 65535:
    if is_free(port):
        print(port)
        raise SystemExit(0)
    port += 1

raise SystemExit("No available TCP port found.")
PY
}

_snapshot_run_configs() {
    local dest_dir="$1"
    local config_dest="$dest_dir/$(basename "$CONFIG_PATH")"
    local runtime_dest="$dest_dir/$(basename "$QWEN3VL_RUNTIME_ENV_CONFIG")"

    cp -f "$CONFIG_PATH" "$config_dest"
    cp -f "$QWEN3VL_RUNTIME_ENV_CONFIG" "$runtime_dest"
}

# Export env vars that Python code needs (canonicalized in qwen3vl_runtime_env.yaml)
_set_env_from_runtime "QWEN3VL_LATENT_SUPERVISION" "latent_supervision" "1"
_set_env_from_runtime "QWEN3VL_LATENT_TOKEN_ID" "latent_token_id" "151669"
_set_env_from_runtime "QWEN3VL_THINKING_START_ID" "thinking_start_id" "151667"
_set_env_from_runtime "QWEN3VL_THINKING_END_ID" "thinking_end_id" "151668"
_set_env_from_runtime "QWEN3VL_THINKING_SEP_ID" "thinking_sep_id" "151670"
_set_env_from_runtime "QWEN3VL_LOSS_TYPE" "loss_type" "vae+ot+mse"
_set_env_from_runtime "QWEN3VL_MATCH_STRATEGY" "match_strategy" "truncate"
_set_env_from_runtime "QWEN3VL_MAX_NEW_TOKENS" "max_new_tokens" "40960"
_set_env_from_runtime "QWEN3VL_VAE_INTERMEDIATE_SIZE" "vae_intermediate_size" "512"
_set_env_from_runtime "QWEN3VL_CURRICULUM_ENABLE" "curriculum_enable" "1"
_set_env_from_runtime "QWEN3VL_CURRICULUM_EPOCHS" "curriculum_epochs" "0,1,2"
_set_env_from_runtime "QWEN3VL_CURRICULUM_LOSS_TYPES" "curriculum_loss_types" "vae+mse,vae+mse,vae+ot+mse"
_set_env_from_runtime "QWEN3VL_CURRICULUM_LATENT_STEP_CE" "curriculum_latent_step_ce" "1,1,1"
_set_env_from_runtime "QWEN3VL_LATENT_STEP_CE_TOKEN" "latent_step_ce_token" "0"
_set_env_from_runtime "QWEN3VL_HIDDEN_STATES_HOOK" "hidden_states_hook" "1"
_set_env_from_main "DATALOADER_NUM_WORKERS" "dataloader_num_workers" "4"

# Runtime behavior
_set_env_from_runtime "QWEN3VL_TRANSPARENT_EVAL_MAX_NEW_TOKENS" "eval_max_new_tokens" "8192"
_set_env_from_runtime "QWEN3VL_COMPILE_VISION_ONLY" "compile_vision_only" "1"
_set_env_from_runtime "PYTORCH_CUDA_ALLOC_CONF" "cuda_alloc_conf" "expandable_segments:True"

# vLLM plugin runtime flags (used by backfill / benchmark subprocesses)
_set_env_from_runtime "VLLM_THINKING" "vllm_thinking" "1"
_set_env_from_runtime "VLLM_ENFORCE_EAGER" "vllm_enforce_eager" "0"
_set_env_from_runtime "VLLM_FORCE_THINK" "vllm_force_think" "0"
_set_env_from_runtime "MIN_CONTINUOUS_STEPS" "min_continuous_steps" "0"

RUN_BACKFILL="$(_get_runtime_config "backfill_enable" "1")"
RUN_BENCHMARK="$(_get_runtime_config "benchmark_enable" "0")"
BENCHMARK_LIST="$(_get_runtime_config "benchmark_list" "MathVision,RealWorldQA")"
BENCHMARK_NUM_SAMPLES="$(_get_runtime_config "benchmark_num_samples" "100")"

# Resolve requested dataset(s) from config + CLI overrides.
DATASET_SPEC="$(_get_main_config "dataset" || echo "")"
for arg in "$@"; do
  if [[ "$arg" == dataset=* ]]; then
    DATASET_SPEC="${arg#dataset=}"
  fi
done

if [ -z "$DATASET_SPEC" ]; then
  echo "❌ No dataset specified in config or CLI override (dataset=...)" >&2
  exit 1
fi

# Check only the dataset files that are actually requested.
declare -a DATASET_PATHS=()
declare -a MISSING_DATASETS=()
IFS=',' read -ra DATASET_NAMES <<< "$DATASET_SPEC"
for dataset_name in "${DATASET_NAMES[@]}"; do
  dataset_name="$(echo "$dataset_name" | xargs)"
  case "$dataset_name" in
    qwen3vl_chimera_thinking_image_input)
      DATASET_PATHS+=("$REPO_ROOT/Qwen/data/chimera_qwen35_thinking_image_input.jsonl")
      ;;
    qwen3vl_chimera_thinking_text_input)
      DATASET_PATHS+=("$REPO_ROOT/Qwen/data/chimera_qwen35_thinking_text_input.jsonl")
      ;;
    *)
      echo "❌ Unsupported CHIMERA dataset in dataset=...: $dataset_name" >&2
      echo "   Supported: qwen3vl_chimera_thinking_text_input, qwen3vl_chimera_thinking_image_input" >&2
      exit 1
      ;;
  esac
done

for dataset_path in "${DATASET_PATHS[@]}"; do
  if [ ! -f "$dataset_path" ]; then
    MISSING_DATASETS+=("$dataset_path")
  fi
done

if [ "${#MISSING_DATASETS[@]}" -gt 0 ]; then
  echo "❌ CHIMERA dataset file(s) not found for dataset=$DATASET_SPEC" >&2
  for missing in "${MISSING_DATASETS[@]}"; do
    echo "   - $missing" >&2
  done
  echo "" >&2
  echo "Please build the dataset first:" >&2
  echo "  python Qwen/scripts/build_chimera_thinking.py --encode-only --num-gpus 8" >&2
  exit 1
fi

echo "✓ CHIMERA dataset(s) found for dataset=$DATASET_SPEC:"
for dataset_path in "${DATASET_PATHS[@]}"; do
  echo "  - $dataset_path (Samples: $(wc -l < "$dataset_path"))"
done
echo ""

# Generate timestamp for unique output directory
TIMESTAMP="${QWEN3VL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
DEFAULT_OUTDIR="$REPO_ROOT/Qwen/checkpoints/qwen3vl-2b/lora/chimera_thinking/run_${TIMESTAMP}"

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

# Determine final output directory
OUTPUT_DIR="$DEFAULT_OUTDIR"
for arg in "$@"; do
  if [[ "$arg" == output_dir=* ]]; then
    OUTPUT_DIR="${arg#output_dir=}"
    break
  fi
done

USE_SWANLAB_CONFIG="$(_get_main_config "use_swanlab" || echo "false")"
SWANLAB_RUN_NAME=""
if [[ "$USE_SWANLAB_CONFIG" == "true" || "$USE_SWANLAB_CONFIG" == "True" || "$USE_SWANLAB_CONFIG" == "1" ]]; then
  export USE_SWANLAB=1
  SWANLAB_RUN_NAME="$(_get_main_config "swanlab_run_name" || echo "")"
  HAS_SWANLAB_RUN_NAME=false
  if [[ -n "$SWANLAB_RUN_NAME" && "$SWANLAB_RUN_NAME" != "null" ]]; then
    HAS_SWANLAB_RUN_NAME=true
  fi
  for arg in "$@"; do
    if [[ "$arg" == swanlab_run_name=* ]]; then
      HAS_SWANLAB_RUN_NAME=true
      SWANLAB_RUN_NAME="${arg#swanlab_run_name=}"
      break
    fi
  done
  if [ "$HAS_SWANLAB_RUN_NAME" = false ]; then
    SWANLAB_RUN_NAME="$(basename "$OUTPUT_DIR")"
    set -- "$@" "swanlab_run_name=$SWANLAB_RUN_NAME"
  fi
else
  export USE_SWANLAB=0
fi

# Setup logging
mkdir -p "$OUTPUT_DIR"
_snapshot_run_configs "$OUTPUT_DIR"
CONFIG_PATH="$OUTPUT_DIR/$(basename "$CONFIG_PATH")"
QWEN3VL_RUNTIME_ENV_CONFIG="$OUTPUT_DIR/$(basename "$QWEN3VL_RUNTIME_ENV_CONFIG")"
RERUN_EVALS_SH="$OUTPUT_DIR/rerun_evals.sh"
cat > "$RERUN_EVALS_SH" <<EOF
#!/bin/bash
set -euo pipefail
QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" CUDA_VISIBLE_DEVICES=\${BACKFILL_CUDA_VISIBLE_DEVICES:-0} $PYTHON_BIN Qwen/scripts/backfill_transparent_eval.py --checkpoint_dir "$OUTPUT_DIR" --checkpoint checkpoint_latest --gpu_memory_utilization \${GPU_MEMORY_UTILIZATION:-0.9} 2>&1 | tee \${BACKFILL_LOG:-/tmp/backfill_thinking_debug.log}
QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" VLLM_FORCE_THINK=$VLLM_FORCE_THINK CUDA_VISIBLE_DEVICES=\${BENCH_CUDA_VISIBLE_DEVICES:-\${CUDA_VISIBLE_DEVICES:-0}} $PYTHON_BIN Qwen/evaluation/run_all_benchmarks.py --start-server --benchmarks \${BENCHMARKS:-MathVision,RealWorldQA} --lora-path "$OUTPUT_DIR/checkpoint_latest"
EOF
chmod +x "$RERUN_EVALS_SH"
LOG_FILE="$OUTPUT_DIR/training.log"

echo "Logging to: $LOG_FILE"
echo ""

_log_wrapper_status() {
  local message="$1"
  printf '[%s] [wrapper] %s\n' "$(date '+%F %T')" "$message" | tee -a "$LOG_FILE"
}

# Load repo-local .env so SWANLAB_API_KEY and similar secrets work by default.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

# Set Python path for llamafactory and sitecustomize.py integration
# sitecustomize.py at repo root enables latent supervision patches
export PYTHONPATH="$REPO_ROOT/../LlamaFactory/src:$REPO_ROOT:${PYTHONPATH:-}"

# If linear checkpoint provides a preferred HF modules cache, use it unless overridden
if [ -z "${HF_MODULES_CACHE:-}" ] && [ -n "${MODEL_PATH:-}" ] && [ -f "$MODEL_PATH/hf_modules_cache.path" ]; then
  export HF_MODULES_CACHE="$(cat "$MODEL_PATH/hf_modules_cache.path")"
fi

# Default to 8 GPUs unless user explicitly sets CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Use local /tmp instead of parallel filesystem for PyTorch temp files
# This avoids "No space left on device" errors when /dev/shm is limited
export TMPDIR="${TMPDIR:-/tmp}"

if [ "${NNODES:-1}" = "1" ]; then
    export MASTER_PORT="$(_resolve_master_port)"
fi

# Detect distributed backend from config (DeepSpeed vs FSDP) for display purposes
USE_DEEPSPEED=false
USE_FSDP=false
if grep -q "^deepspeed:" "$CONFIG_PATH" 2>/dev/null; then
    USE_DEEPSPEED=true
elif grep -q "^fsdp:" "$CONFIG_PATH" 2>/dev/null; then
    USE_FSDP=true
fi

# DeepSpeed-specific NCCL configuration for single-node multi-GPU
if [ "$USE_DEEPSPEED" = true ]; then
    # Debugging (turn off once stable)
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

    # Avoid IB issues on single machine
    export NCCL_IB_DISABLE=1

    # Use shared memory + PCIe/NVLink
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0

    # Interface selection (important!)
    export NCCL_SOCKET_IFNAME=lo
fi

# Ensure pack-after-injection uses YAML cutoff_len by default.
# Priority:
# 1) pre-exported QWEN3VL_CUTOFF_LEN (explicit user override)
# 2) main training YAML cutoff_len
MAIN_CUTOFF_LEN="$(_get_main_config "cutoff_len" "8192")"
export QWEN3VL_CUTOFF_LEN="${QWEN3VL_CUTOFF_LEN:-$MAIN_CUTOFF_LEN}"


# Display configuration
TRAINING_TYPE="CHIMERA SFT Training (Latent Supervision)"

LOSS_TYPE="$QWEN3VL_LOSS_TYPE"
LOSS_TYPE_DISPLAY="$LOSS_TYPE"

echo "========================================================================"
echo "Qwen3VL CHIMERA SFT Training"
echo "========================================================================"
echo "Training Type: $TRAINING_TYPE"
echo "Distributed Backend: $([ "$USE_DEEPSPEED" = true ] && echo "DeepSpeed" || ([ "$USE_FSDP" = true ] && echo "FSDP" || echo "Unknown"))"
echo "Config: $CONFIG_PATH"
echo "Dataset(s): $DATASET_SPEC"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Latent Supervision:"
echo "  - Enabled: $QWEN3VL_LATENT_SUPERVISION"
echo "  - Loss type: $LOSS_TYPE_DISPLAY"
echo "  - Match strategy: $QWEN3VL_MATCH_STRATEGY"
echo ""
echo "Transparent Evaluation:"
echo "  - Backfill after training: $RUN_BACKFILL"
echo "  - Backfill max new tokens: $QWEN3VL_TRANSPARENT_EVAL_MAX_NEW_TOKENS"
echo ""
echo "Benchmark:"
echo "  - Run after backfill: $RUN_BENCHMARK"
echo "  - Benchmarks: $BENCHMARK_LIST"
echo "  - Num samples: $BENCHMARK_NUM_SAMPLES"
echo ""
echo "Curriculum Learning:"
echo "  - Enabled: $QWEN3VL_CURRICULUM_ENABLE"
echo "  - Epochs: $QWEN3VL_CURRICULUM_EPOCHS"
echo "  - Loss types: $QWEN3VL_CURRICULUM_LOSS_TYPES"
echo "  - Latent step CE: $QWEN3VL_CURRICULUM_LATENT_STEP_CE"
echo ""
echo "Special Tokens:"
echo "  - <latent>: $QWEN3VL_LATENT_TOKEN_ID"
echo "  - Thinking start: $QWEN3VL_THINKING_START_ID"
echo "  - Thinking end: $QWEN3VL_THINKING_END_ID"
echo "  - <think_sep>: $QWEN3VL_THINKING_SEP_ID"
if [[ "$USE_SWANLAB_CONFIG" == "true" || "$USE_SWANLAB_CONFIG" == "True" || "$USE_SWANLAB_CONFIG" == "1" ]]; then
  echo ""
  echo "SwanLab:"
  echo "  - Enabled: true"
  echo "  - Project: $(_get_main_config "swanlab_project" || echo "llamafactory")"
  echo "  - Run name: $SWANLAB_RUN_NAME"
  echo "  - Auth: SWANLAB_API_KEY (loaded from .env if present)"
fi
echo "========================================================================"
echo ""

_log_wrapper_status "wrapper_start config=$CONFIG_PATH output_dir=$OUTPUT_DIR cuda_visible_devices=$CUDA_VISIBLE_DEVICES run_backfill=$RUN_BACKFILL run_benchmark=$RUN_BENCHMARK"

# Run llamafactory CLI with tee for logging
# Use PIPESTATUS to preserve exit code from llamafactory-cli
NPROC_PER_NODE_DEFAULT="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$NPROC_PER_NODE_DEFAULT}"

# Run training
# FSDP: Direct Python call (LlamaFactory handles torchrun internally)
# DeepSpeed: Use torchrun explicitly
if [ "$USE_DEEPSPEED" = true ]; then
    torchrun \
      --standalone \
      --nproc_per_node="$NPROC_PER_NODE" \
      -m llamafactory.cli train "$CONFIG_PATH" "$@" 2>&1 | tee -a "$LOG_FILE"
else
    "$PYTHON_BIN" -m llamafactory.cli train "$CONFIG_PATH" "$@" 2>&1 | tee -a "$LOG_FILE"
fi
exit_code=${PIPESTATUS[0]}
_log_wrapper_status "training_finished exit_code=$exit_code"

# ============================================================================
# Post-training: Run backfill transparent eval on all checkpoints
# Uses all training GPUs in parallel (round-robin, M jobs at a time)
# ============================================================================
if [ "$exit_code" -eq 0 ] && [ "$RUN_BACKFILL" = "1" ]; then
    _log_wrapper_status "backfill_gate entered exit_code=$exit_code run_backfill=$RUN_BACKFILL"
    echo ""
    echo "========================================================================"
    echo "Training completed! Running backfill transparent eval on all checkpoints..."
    echo "========================================================================"

    # Get list of GPUs from training
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPU_ARRAY[@]}
    echo "Using $NUM_GPUS GPUs for backfill: ${GPU_ARRAY[*]}"

    # Find all checkpoints that need backfill
    CHECKPOINT_LIST=()
    for CHECKPOINT in $(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V); do
        CHECKPOINT_NAME=$(basename "$CHECKPOINT")

        # Skip if already has results
        if [ -d "$CHECKPOINT/eval_results" ] && [ "$(ls "$CHECKPOINT/eval_results"/backfill_*.json 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  Skipping $CHECKPOINT_NAME - already has backfill results"
            continue
        fi

        CHECKPOINT_LIST+=("$CHECKPOINT_NAME")
    done

    NUM_CHECKPOINTS=${#CHECKPOINT_LIST[@]}
    _log_wrapper_status "backfill_discovery num_gpus=$NUM_GPUS num_checkpoints=$NUM_CHECKPOINTS checkpoints=${CHECKPOINT_LIST[*]:-none}"

    if [ "$NUM_CHECKPOINTS" -eq 0 ]; then
        _log_wrapper_status "backfill_skip reason=no_checkpoints"
        echo "No checkpoints need backfill"
    else
        _log_wrapper_status "backfill_launch starting"
        echo "Found $NUM_CHECKPOINTS checkpoints to backfill (running $NUM_GPUS at a time)"

        # Run in batches of NUM_GPUS
        for ((i=0; i<NUM_CHECKPOINTS; i+=NUM_GPUS)); do
            # Launch jobs for this batch (one per GPU)
            for ((j=0; j<NUM_GPUS && i+j<NUM_CHECKPOINTS; j++)); do
                idx=$((i+j))
                checkpoint_name="${CHECKPOINT_LIST[$idx]}"
                gpu_id="${GPU_ARRAY[$j]}"

                echo "  Starting $checkpoint_name on GPU $gpu_id..."
                _log_wrapper_status "backfill_checkpoint_start checkpoint=$checkpoint_name gpu=$gpu_id"

                # Run backfill in background with specific GPU
                (
                    echo "[$(date '+%F %T')] Start $checkpoint_name on GPU $gpu_id" >> "$OUTPUT_DIR/backfill_all_checkpoints.log"
                    if QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" Qwen/scripts/backfill_transparent_eval.py \
                        --checkpoint_dir "$OUTPUT_DIR" \
                        --checkpoint "$checkpoint_name" \
                        --gpu_memory_utilization 0.9 \
                        >> "$OUTPUT_DIR/backfill_all_checkpoints.log" 2>&1; then
                        if ls "$OUTPUT_DIR/${checkpoint_name}/eval_results"/backfill_*.json >/dev/null 2>&1; then
                            echo "[$(date '+%F %T')] $checkpoint_name: Completed!" >> "$OUTPUT_DIR/backfill_all_checkpoints.log"
                        else
                            echo "[$(date '+%F %T')] $checkpoint_name: Python succeeded but no backfill_*.json generated" >> "$OUTPUT_DIR/backfill_all_checkpoints.log"
                        fi
                    else
                        code=$?
                        echo "[$(date '+%F %T')] $checkpoint_name: Failed with exit code $code" >> "$OUTPUT_DIR/backfill_all_checkpoints.log"
                    fi
                ) &
            done

            # Wait for this batch to finish before starting next batch
            wait || true
        done

        _log_wrapper_status "backfill_launch complete"
        echo ""
        echo "Backfill complete! Results saved to $OUTPUT_DIR/eval_results/"
    fi
else
    _log_wrapper_status "backfill_gate skipped exit_code=$exit_code run_backfill=$RUN_BACKFILL"
fi

cleanup_vllm_benchmark_processes() {
    # Ensure leaked vLLM worker/core processes do not affect later jobs.
    pkill -9 -f "VLLM::EngineCore" || true
}

if [ "$exit_code" -eq 0 ] && [ "$RUN_BENCHMARK" = "1" ]; then
    _log_wrapper_status "benchmark_gate entered exit_code=$exit_code run_benchmark=$RUN_BENCHMARK"
    LATEST_CHECKPOINT="$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
    if [ -z "${LATEST_CHECKPOINT:-}" ] || [ ! -d "$LATEST_CHECKPOINT" ]; then
        _log_wrapper_status "benchmark_skip reason=no_checkpoint"
        echo "⚠️  Benchmark skipped: no checkpoint-* found in $OUTPUT_DIR"
    else
        BENCH_DIR="$OUTPUT_DIR/bench"
        mkdir -p "$BENCH_DIR"
        BENCH_GPUS="$CUDA_VISIBLE_DEVICES"
        if [ -n "${BENCHMARK_GPUS:-}" ] && [ "$BENCHMARK_GPUS" != "$CUDA_VISIBLE_DEVICES" ]; then
            echo "⚠️  Ignoring BENCHMARK_GPUS=$BENCHMARK_GPUS to keep TP aligned with training GPUs ($CUDA_VISIBLE_DEVICES)"
        fi

        echo ""
        echo "========================================================================"
        echo "Running benchmarks on latest checkpoint..."
        echo "  - Checkpoint: $LATEST_CHECKPOINT"
        echo "  - GPUs: $BENCH_GPUS"
        echo "  - Num samples: $BENCHMARK_NUM_SAMPLES"
        echo "  - Output dir: $BENCH_DIR"
        echo "========================================================================"

        QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" "$PYTHON_BIN" -u Qwen/evaluation/run_all_benchmarks.py \
            --start-server \
            --benchmarks "$BENCHMARK_LIST" \
            --lora-path "$LATEST_CHECKPOINT" \
            --num-samples "$BENCHMARK_NUM_SAMPLES" \
            --gpus "$BENCH_GPUS" \
            --run-dir "$BENCH_DIR"
        bench_exit_code=$?
        _log_wrapper_status "benchmark_finished exit_code=$bench_exit_code latest_checkpoint=$LATEST_CHECKPOINT"
        if [ "$bench_exit_code" -ne 0 ]; then
            echo "⚠️  Benchmark failed with exit code $bench_exit_code"
            exit_code=$bench_exit_code
        fi
    fi
else
    _log_wrapper_status "benchmark_gate skipped exit_code=$exit_code run_benchmark=$RUN_BENCHMARK"
fi

cleanup_vllm_benchmark_processes

_log_wrapper_status "wrapper_exit exit_code=${exit_code:-0}"
exit ${exit_code:-0}
