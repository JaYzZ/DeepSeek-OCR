#!/bin/bash
# Qwen3VL R1-OneVision SFT Training with Latent Supervision
#
# This script trains Qwen3VL-2B-Thinking on R1-OneVision dataset using:
# - Pre-encoded vision features (no encoding during training)
# - Latent injection at <latent> positions
# - Thinking loss (REPA/OT/NCE/MSE) on latent predictions
#
# Usage:
#   bash Qwen/scripts/train_qwen3vl_r1onevision.sh [config.yaml]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/train_qwen3vl_dataset_mix.sh"

# Required python interpreter (OCRFlow env)
PYTHON_BIN="$REPO_ROOT/../../envs/ocrflow/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ Python not found or not executable: $PYTHON_BIN" >&2
  echo "   Please ensure OCRFlow env exists at: $REPO_ROOT/../../envs/ocrflow" >&2
  exit 1
fi

# Default config
DEFAULT_CONFIG="$REPO_ROOT/Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml"
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
_set_env_from_runtime "QWEN3VL_LATENT_AUX_LOSS_SOURCE" "latent_aux_loss_source" "vae_sample"
_set_env_from_runtime "QWEN3VL_MATCH_STRATEGY" "match_strategy" "truncate"
_set_env_from_runtime "QWEN3VL_MAX_NEW_TOKENS" "max_new_tokens" "40960"
_set_env_from_runtime "QWEN3VL_VAE_INTERMEDIATE_SIZE" "vae_intermediate_size" "512"
_set_env_from_runtime "QWEN3VL_CURRICULUM_ENABLE" "curriculum_enable" "1"
_set_env_from_runtime "QWEN3VL_CURRICULUM_EPOCHS" "curriculum_epochs" "0,1,2"
_set_env_from_runtime "QWEN3VL_CURRICULUM_LOSS_TYPES" "curriculum_loss_types" "vae+mse,vae:0.5+mse:0.5,vae:0.5+ot:0.25+mse:0.25"
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
BENCHMARK_LIST="$(_get_runtime_config "benchmark_list" "MathVision,MMMU,RealWorldQA")"
BENCHMARK_NUM_SAMPLES="$(_get_runtime_config "benchmark_num_samples" "100")"

# Generate timestamp for unique output directory
TIMESTAMP="${QWEN3VL_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
DEFAULT_OUTDIR="$REPO_ROOT/Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_${TIMESTAMP}"

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

# Determine final run directory
OUTPUT_DIR="$DEFAULT_OUTDIR"
for arg in "$@"; do
  if [[ "$arg" == output_dir=* ]]; then
    OUTPUT_DIR="${arg#output_dir=}"
    break
  fi
done
RUN_DIR="$OUTPUT_DIR"
CKPT_DIR="$RUN_DIR"

# Resolve requested dataset(s) from config + CLI overrides.
DATASET_SPEC="$(_get_main_config "dataset" || echo "")"
DATASET_DIR="$(_get_main_config "dataset_dir" "$REPO_ROOT/Qwen/data")"
DATASET_STREAMING="$(_get_main_config "streaming" "false")"
DATASET_MIX_STRATEGY="$(_get_main_config "mix_strategy" "concat")"
for arg in "$@"; do
  if [[ "$arg" == dataset=* ]]; then
    DATASET_SPEC="${arg#dataset=}"
  elif [[ "$arg" == dataset_dir=* ]]; then
    DATASET_DIR="${arg#dataset_dir=}"
  elif [[ "$arg" == streaming=* ]]; then
    DATASET_STREAMING="${arg#streaming=}"
  elif [[ "$arg" == mix_strategy=* ]]; then
    DATASET_MIX_STRATEGY="${arg#mix_strategy=}"
  fi
done

if [ -z "$DATASET_SPEC" ]; then
  echo "❌ No dataset specified in config or CLI override (dataset=...)" >&2
  exit 1
fi

if [[ ! "$DATASET_DIR" = /* ]]; then
  DATASET_DIR="$REPO_ROOT/$DATASET_DIR"
fi

mkdir -p "$RUN_DIR"
eval "$(qwen3vl_materialize_dataset_mix "$PYTHON_BIN" "$DATASET_DIR" "$DATASET_SPEC" "$RUN_DIR")"
DATASET_SPEC="$QWEN3VL_EFFECTIVE_DATASET_SPEC"
DATASET_DIR="$QWEN3VL_EFFECTIVE_DATASET_DIR"
DATASET_RATIO_APPLIED="${QWEN3VL_DATASET_RATIO_APPLIED:-0}"
DATASET_MIX_SUMMARY_PATH="${QWEN3VL_DATASET_MIX_SUMMARY_PATH:-}"
DATASET_COUNT="${QWEN3VL_DATASET_COUNT:-1}"

if [ "$DATASET_COUNT" -gt 1 ]; then
  case "${DATASET_STREAMING,,}" in
    1|true|yes)
      echo "❌ Multi-dataset SFT requires streaming=false so the merged dataset can be jointly shuffled." >&2
      exit 1
      ;;
  esac
  if [[ -z "$DATASET_MIX_STRATEGY" || "$DATASET_MIX_STRATEGY" == "null" ]]; then
    DATASET_MIX_STRATEGY="concat"
  fi
fi

set -- "$@" "dataset=$DATASET_SPEC" "dataset_dir=$DATASET_DIR"
if [ "$DATASET_COUNT" -gt 1 ]; then
  set -- "$@" "mix_strategy=$DATASET_MIX_STRATEGY"
fi

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
    SWANLAB_RUN_NAME="$(basename "$RUN_DIR")"
    set -- "$@" "swanlab_run_name=$SWANLAB_RUN_NAME"
  fi
else
  export USE_SWANLAB=0
fi

# Setup logging
mkdir -p "$RUN_DIR" "$CKPT_DIR"
_snapshot_run_configs "$RUN_DIR"
CONFIG_PATH="$RUN_DIR/$(basename "$CONFIG_PATH")"
QWEN3VL_RUNTIME_ENV_CONFIG="$RUN_DIR/$(basename "$QWEN3VL_RUNTIME_ENV_CONFIG")"
RERUN_EVALS_SH="$RUN_DIR/rerun_evals.sh"
cat > "$RERUN_EVALS_SH" <<EOF
#!/bin/bash
set -euo pipefail

GPUS=""
BACKFILL_GPUS=""
BENCH_GPUS=""
while [[ \$# -gt 0 ]]; do
  case "\$1" in
    --gpus)
      GPUS="\$2"
      shift 2
      ;;
    --backfill-gpus)
      BACKFILL_GPUS="\$2"
      shift 2
      ;;
    --bench-gpus)
      BENCH_GPUS="\$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: \$1" >&2
      echo "Usage: \$0 [--gpus 0,1,2,3] [--backfill-gpus 0,1] [--bench-gpus 0,1,2,3]" >&2
      exit 2
      ;;
  esac
done

if [[ -n "\$GPUS" ]]; then
  BACKFILL_GPUS="\$GPUS"
  BENCH_GPUS="\$GPUS"
fi

BACKFILL_GPUS="\${BACKFILL_GPUS:-\${BACKFILL_CUDA_VISIBLE_DEVICES:-\${CUDA_VISIBLE_DEVICES:-0}}}"
BENCH_GPUS="\${BENCH_GPUS:-\${BENCH_CUDA_VISIBLE_DEVICES:-\${CUDA_VISIBLE_DEVICES:-\$BACKFILL_GPUS}}}"

QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" CUDA_VISIBLE_DEVICES="\$BACKFILL_GPUS" $PYTHON_BIN Qwen/scripts/backfill_transparent_eval.py --checkpoint_dir "$CKPT_DIR" --checkpoint checkpoint_latest --gpu_memory_utilization \${GPU_MEMORY_UTILIZATION:-0.9} 2>&1 | tee \${BACKFILL_LOG:-/tmp/backfill_thinking_debug.log}
BENCH_DIR="$RUN_DIR/bench"
mkdir -p "\$BENCH_DIR"
QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" VLLM_FORCE_THINK=$VLLM_FORCE_THINK CUDA_VISIBLE_DEVICES="\$BENCH_GPUS" $PYTHON_BIN Qwen/evaluation/run_all_benchmarks.py --start-server --gpus "\$BENCH_GPUS" --benchmarks \${BENCHMARKS:-MathVision,MMMU,RealWorldQA} --num-samples \${BENCHMARK_NUM_SAMPLES:-$BENCHMARK_NUM_SAMPLES} --lora-path "$CKPT_DIR/checkpoint_latest" --run-dir "\$BENCH_DIR"
EOF
chmod +x "$RERUN_EVALS_SH"
LOG_FILE="$RUN_DIR/training.log"

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
TRAINING_TYPE="R1-OneVision SFT Training (Latent Supervision)"

LOSS_TYPE="$QWEN3VL_LOSS_TYPE"
LOSS_TYPE_DISPLAY="$LOSS_TYPE"

echo "========================================================================"
echo "Qwen3VL R1-OneVision SFT Training"
echo "========================================================================"
echo "Training Type: $TRAINING_TYPE"
echo "Distributed Backend: $([ "$USE_DEEPSPEED" = true ] && echo "DeepSpeed" || ([ "$USE_FSDP" = true ] && echo "FSDP" || echo "Unknown"))"
echo "Config: $CONFIG_PATH"
echo "Run dir: $RUN_DIR"
echo "Checkpoint dir: $CKPT_DIR"
echo "Dataset(s): $DATASET_SPEC"
echo "Dataset dir: $DATASET_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Dataset Mixing:"
echo "  - Entries: $DATASET_COUNT"
echo "  - Ratio parsing: $([ "$DATASET_RATIO_APPLIED" = "1" ] && echo "enabled" || echo "disabled")"
echo "  - Mix strategy: ${DATASET_MIX_STRATEGY:-concat}"
if [ "${DATASET_MIX_STRATEGY:-concat}" = "concat" ]; then
  echo "  - Joint shuffle: enabled via concat + Trainer random sampler"
else
  echo "  - Joint shuffle: Trainer random sampler enabled; mix ordering follows ${DATASET_MIX_STRATEGY}"
fi
if [ -n "$DATASET_MIX_SUMMARY_PATH" ]; then
  echo "  - Summary: $DATASET_MIX_SUMMARY_PATH"
fi
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

_log_wrapper_status "wrapper_start config=$CONFIG_PATH run_dir=$RUN_DIR ckpt_dir=$CKPT_DIR cuda_visible_devices=$CUDA_VISIBLE_DEVICES run_backfill=$RUN_BACKFILL run_benchmark=$RUN_BENCHMARK"

# Run llamafactory CLI with tee for logging
# Use PIPESTATUS to preserve exit code from llamafactory-cli
NPROC_PER_NODE_DEFAULT="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$NPROC_PER_NODE_DEFAULT}"

TRAIN_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == output_dir=* ]]; then
        continue
    fi
    TRAIN_ARGS+=("$arg")
done
TRAIN_ARGS+=("output_dir=$CKPT_DIR")

# Run training
# FSDP: Direct Python call (LlamaFactory handles torchrun internally)
# DeepSpeed: Use torchrun explicitly
if [ "$USE_DEEPSPEED" = true ]; then
    set +e
    torchrun \
      --standalone \
      --nproc_per_node="$NPROC_PER_NODE" \
      -m llamafactory.cli train "$CONFIG_PATH" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    set -e
else
    set +e
    "$PYTHON_BIN" -m llamafactory.cli train "$CONFIG_PATH" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    set -e
fi
_log_wrapper_status "training_finished exit_code=$exit_code"
train_exit_code=$exit_code

# ============================================================================
# Post-training: Run backfill transparent eval on all checkpoints
# Uses all training GPUs in parallel (round-robin, M jobs at a time)
# ============================================================================
if [ "$train_exit_code" -eq 0 ] && [ "$RUN_BACKFILL" = "1" ]; then
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
    for CHECKPOINT in $(ls -td "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V); do
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
        backfill_failed=0

        # Run in batches of NUM_GPUS
        for ((i=0; i<NUM_CHECKPOINTS; i+=NUM_GPUS)); do
            BATCH_PIDS=()
            BATCH_CHECKPOINTS=()
            # Launch jobs for this batch (one per GPU)
            for ((j=0; j<NUM_GPUS && i+j<NUM_CHECKPOINTS; j++)); do
                idx=$((i+j))
                checkpoint_name="${CHECKPOINT_LIST[$idx]}"
                gpu_id="${GPU_ARRAY[$j]}"

                echo "  Starting $checkpoint_name on GPU $gpu_id..."
                _log_wrapper_status "backfill_checkpoint_start checkpoint=$checkpoint_name gpu=$gpu_id"

                # Run backfill in background with specific GPU
                (
                    echo "[$(date '+%F %T')] Start $checkpoint_name on GPU $gpu_id" >> "$RUN_DIR/backfill_all_checkpoints.log"
                    if QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" Qwen/scripts/backfill_transparent_eval.py \
                        --checkpoint_dir "$CKPT_DIR" \
                        --checkpoint "$checkpoint_name" \
                        --gpu_memory_utilization 0.9 \
                        >> "$RUN_DIR/backfill_all_checkpoints.log" 2>&1; then
                        if ls "$CKPT_DIR/${checkpoint_name}/eval_results"/backfill_*.json >/dev/null 2>&1; then
                            echo "[$(date '+%F %T')] $checkpoint_name: Completed!" >> "$RUN_DIR/backfill_all_checkpoints.log"
                            exit 0
                        else
                            echo "[$(date '+%F %T')] $checkpoint_name: Python succeeded but no backfill_*.json generated" >> "$RUN_DIR/backfill_all_checkpoints.log"
                            exit 1
                        fi
                    else
                        code=$?
                        echo "[$(date '+%F %T')] $checkpoint_name: Failed with exit code $code" >> "$RUN_DIR/backfill_all_checkpoints.log"
                        exit "$code"
                    fi
                ) &
                BATCH_PIDS+=($!)
                BATCH_CHECKPOINTS+=("$checkpoint_name")
            done

            # Wait for this batch to finish before starting next batch
            for batch_idx in "${!BATCH_PIDS[@]}"; do
                pid="${BATCH_PIDS[$batch_idx]}"
                checkpoint_name="${BATCH_CHECKPOINTS[$batch_idx]}"
                if ! wait "$pid"; then
                    backfill_failed=1
                    _log_wrapper_status "backfill_checkpoint_failed checkpoint=$checkpoint_name"
                fi
            done
        done

        _log_wrapper_status "backfill_launch complete"
        echo ""
        echo "Backfill complete! Results saved under $CKPT_DIR/checkpoint-*/eval_results/"
        if [ "$backfill_failed" -ne 0 ]; then
            _log_wrapper_status "backfill_finished status=failed"
            echo "⚠️  One or more backfill jobs failed. See $RUN_DIR/backfill_all_checkpoints.log"
            exit_code=1
        else
            _log_wrapper_status "backfill_finished status=ok"
        fi
    fi
else
    _log_wrapper_status "backfill_gate skipped exit_code=$train_exit_code run_backfill=$RUN_BACKFILL"
fi

cleanup_vllm_benchmark_processes() {
    # Ensure leaked vLLM worker/core processes do not affect later jobs.
    "$PYTHON_BIN" - <<'PY'
from Qwen.scripts.vllm_utils import cleanup_vllm_engine_processes
cleanup_vllm_engine_processes()
PY
}

if [ "$train_exit_code" -eq 0 ] && [ "$RUN_BENCHMARK" = "1" ]; then
    _log_wrapper_status "benchmark_gate entered exit_code=$train_exit_code run_benchmark=$RUN_BENCHMARK"
    cleanup_vllm_benchmark_processes
    LATEST_CHECKPOINT="$(ls -td "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
    if [ -z "${LATEST_CHECKPOINT:-}" ] || [ ! -d "$LATEST_CHECKPOINT" ]; then
        _log_wrapper_status "benchmark_skip reason=no_checkpoint"
        echo "⚠️  Benchmark skipped: no checkpoint-* found in $CKPT_DIR"
    else
        BENCH_DIR="$RUN_DIR/bench"
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

        set +e
        QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" "$PYTHON_BIN" -u Qwen/evaluation/run_all_benchmarks.py \
            --start-server \
            --benchmarks "$BENCHMARK_LIST" \
            --lora-path "$LATEST_CHECKPOINT" \
            --num-samples "$BENCHMARK_NUM_SAMPLES" \
            --gpus "$BENCH_GPUS" \
            --run-dir "$BENCH_DIR"
        bench_exit_code=$?
        set -e
        _log_wrapper_status "benchmark_finished exit_code=$bench_exit_code latest_checkpoint=$LATEST_CHECKPOINT"
        if [ "$bench_exit_code" -ne 0 ]; then
            echo "⚠️  Benchmark failed with exit code $bench_exit_code"
            exit_code=$bench_exit_code
        fi
    fi
else
    _log_wrapper_status "benchmark_gate skipped exit_code=$train_exit_code run_benchmark=$RUN_BENCHMARK"
fi

cleanup_vllm_benchmark_processes

_log_wrapper_status "wrapper_exit exit_code=${exit_code:-0}"
exit ${exit_code:-0}
