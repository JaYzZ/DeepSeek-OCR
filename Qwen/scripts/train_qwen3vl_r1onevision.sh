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
source "$SCRIPT_DIR/qwen3vl_common.sh"
source "$SCRIPT_DIR/train_qwen3vl_dataset_mix.sh"

PYTHON_BIN="$(qwen3vl_require_python_bin "$REPO_ROOT")"

# Default config
DEFAULT_CONFIG="$REPO_ROOT/Qwen/configs/qwen3vl_r1onevision_thinking.yaml"
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

_get_main_config() {
    qwen3vl_get_yaml_value "$PYTHON_BIN" "$CONFIG_PATH" "$1" "${2:-}"
}

_get_runtime_config() {
    qwen3vl_get_yaml_value "$PYTHON_BIN" "$QWEN3VL_RUNTIME_ENV_CONFIG" "$1" "${2:-}"
}

_set_env_from_runtime() {
    qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$QWEN3VL_RUNTIME_ENV_CONFIG" "$1" "$2" "$3"
}

_set_env_from_main() {
    qwen3vl_export_env_from_yaml "$PYTHON_BIN" "$CONFIG_PATH" "$1" "$2" "$3"
}

_build_curriculum_stage_plan() {
    local epochs_str="$1"
    local loss_types_str="$2"
    local vae_trainable_str="$3"
    local lora_trainable_str="$4"
    local aux_source_str="$5"
    local latent_ce_str="$6"
    local total_epochs="$7"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$PYTHON_BIN" - "$epochs_str" "$loss_types_str" "$vae_trainable_str" "$lora_trainable_str" "$aux_source_str" "$latent_ce_str" "$total_epochs" <<'PY'
import sys

epochs_str, loss_types_str, vae_trainable_str, lora_trainable_str, aux_source_str, latent_ce_str, total_epochs_str = sys.argv[1:]
total_epochs = float(total_epochs_str)

if "," in epochs_str:
    boundaries = [float(x.strip()) for x in epochs_str.split(",") if x.strip()]
else:
    num_stages = int(epochs_str)
    boundaries = [float(i) for i in range(num_stages)]

if not boundaries:
    raise SystemExit("Curriculum split requires at least one stage boundary.")
if boundaries[0] != 0.0:
    raise SystemExit(f"Curriculum split requires boundaries to start at 0, got {boundaries[0]}.")
if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
    raise SystemExit(f"Curriculum boundaries must be strictly increasing: {boundaries}")
if boundaries[-1] >= total_epochs:
    raise SystemExit(
        f"Last curriculum boundary {boundaries[-1]} must be less than total num_train_epochs {total_epochs}."
    )

loss_types = [x.strip() for x in loss_types_str.split(",") if x.strip()]

def parse_stage_values(raw: str):
    if "," in raw:
        return [x.strip() for x in raw.split(",")]
    return [raw.strip()] * len(boundaries)

vae_trainable = parse_stage_values(vae_trainable_str)
lora_trainable = parse_stage_values(lora_trainable_str)
aux_source = parse_stage_values(aux_source_str)
latent_ce = parse_stage_values(latent_ce_str)

num_stages = len(boundaries)
if not all(len(seq) == num_stages for seq in [loss_types, vae_trainable, lora_trainable, aux_source, latent_ce]):
    raise SystemExit(
        "Curriculum config length mismatch: "
        f"boundaries={num_stages}, loss_types={len(loss_types)}, vae_trainable={len(vae_trainable)}, "
        f"lora_trainable={len(lora_trainable)}, aux_source={len(aux_source)}, latent_ce={len(latent_ce)}"
    )

for idx, start_epoch in enumerate(boundaries):
    end_epoch = boundaries[idx + 1] if idx + 1 < num_stages else total_epochs
    print(
        "\t".join(
            [
                str(idx + 1),
                str(start_epoch),
                str(end_epoch),
                loss_types[idx],
                vae_trainable[idx],
                lora_trainable[idx],
                aux_source[idx],
                latent_ce[idx],
            ]
        )
    )
PY
}

_resolve_total_num_train_epochs() {
    local main_total_epochs="$1"
    local curriculum_epochs="$2"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$PYTHON_BIN" - "$main_total_epochs" "$curriculum_epochs" <<'PY'
import sys

main_total_epochs_str, curriculum_epochs = sys.argv[1:]
main_total_epochs = float(main_total_epochs_str)

if "," not in curriculum_epochs:
    print(main_total_epochs_str)
    raise SystemExit(0)

boundaries = [float(x.strip()) for x in curriculum_epochs.split(",") if x.strip()]
if not boundaries:
    raise SystemExit("Curriculum split requires at least one stage boundary.")
if boundaries[0] != 0.0:
    raise SystemExit(f"Curriculum split requires boundaries to start at 0, got {boundaries[0]}.")
if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
    raise SystemExit(f"Curriculum boundaries must be strictly increasing: {boundaries}")

if len(boundaries) == 1:
    print("1.0")
    raise SystemExit(0)

# Boundary lists encode stage starts only. Infer the total epoch budget by
# extending the final stage with the same width as the previous stage.
last_span = boundaries[-1] - boundaries[-2]
if last_span <= 0:
    raise SystemExit(f"Invalid final curriculum span from boundaries: {boundaries}")

derived_total_epochs = boundaries[-1] + last_span
print(str(derived_total_epochs))
PY
}

_compute_epoch_span() {
    local start_epoch="$1"
    local end_epoch="$2"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$PYTHON_BIN" - "$start_epoch" "$end_epoch" <<'PY'
import sys

start_epoch = float(sys.argv[1])
end_epoch = float(sys.argv[2])
span = end_epoch - start_epoch
if span <= 0:
    raise SystemExit(f"Invalid stage epoch span: start={start_epoch}, end={end_epoch}")
print(span)
PY
}

_build_curriculum_stage_dataset_plan() {
    local stage_datasets_str="$1"
    local default_dataset_spec="$2"
    local num_stages="$3"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$PYTHON_BIN" - "$stage_datasets_str" "$default_dataset_spec" "$num_stages" <<'PY'
import sys

raw_stage_datasets, default_dataset_spec, num_stages_str = sys.argv[1:]
num_stages = int(num_stages_str)

if num_stages <= 0:
    raise SystemExit("Curriculum stage dataset plan requires num_stages > 0.")

raw_stage_datasets = raw_stage_datasets.strip()
if not raw_stage_datasets:
    specs = [default_dataset_spec] * num_stages
else:
    specs = [item.strip() for item in raw_stage_datasets.split("|")]
    if len(specs) == 1 and num_stages > 1:
        specs = specs * num_stages

if len(specs) != num_stages:
    raise SystemExit(
        f"Curriculum stage dataset spec length mismatch: got {len(specs)} spec(s) for {num_stages} stages."
    )

for spec in specs:
    if not spec:
        raise SystemExit("Curriculum stage dataset spec entries must be non-empty.")
    print(spec)
PY
}

_list_checkpoint_dirs() {
    qwen3vl_list_checkpoint_dirs "$PYTHON_BIN" "$CKPT_DIR"
}

# Export env vars that Python code needs (canonicalized in qwen3vl_runtime_env.yaml)
_set_env_from_runtime "QWEN3VL_LATENT_SUPERVISION" "latent_supervision" "1"
_set_env_from_runtime "QWEN3VL_LATENT_TOKEN_ID" "latent_token_id" "151669"
_set_env_from_runtime "QWEN3VL_THINKING_START_ID" "thinking_start_id" "151667"
_set_env_from_runtime "QWEN3VL_THINKING_END_ID" "thinking_end_id" "151668"
_set_env_from_runtime "QWEN3VL_THINKING_SEP_ID" "thinking_sep_id" "151670"
_set_env_from_runtime "QWEN3VL_LOSS_TYPE" "loss_type" "ce+vae"
_set_env_from_runtime "QWEN3VL_LATENT_AUX_LOSS_SOURCE" "latent_aux_loss_source" "hidden"
_set_env_from_runtime "QWEN3VL_MATCH_STRATEGY" "match_strategy" "truncate"
qwen3vl_export_env_from_main_then_runtime "$PYTHON_BIN" "$CONFIG_PATH" "generation.max_new_tokens" "$QWEN3VL_RUNTIME_ENV_CONFIG" "max_new_tokens" "QWEN3VL_MAX_NEW_TOKENS" "40960"
_set_env_from_runtime "QWEN3VL_VAE_INTERMEDIATE_SIZE" "vae_intermediate_size" "512"
_set_env_from_runtime "QWEN3VL_CURRICULUM_ENABLE" "curriculum_enable" "1"
_set_env_from_runtime "QWEN3VL_CURRICULUM_EPOCHS" "curriculum_epochs" "0,1,2"
_set_env_from_runtime "QWEN3VL_CURRICULUM_LOSS_TYPES" "curriculum_loss_types" "ce+mse:0.4+ot:0.4,vae,ce+vae"
_set_env_from_runtime "QWEN3VL_CURRICULUM_VAE_TRAINABLE" "curriculum_vae_trainable" "0,1,0"
_set_env_from_runtime "QWEN3VL_CURRICULUM_AUX_SOURCE" "curriculum_aux_source" "hidden"
_set_env_from_runtime "QWEN3VL_CURRICULUM_LORA_TRAINABLE" "curriculum_lora_trainable" "1,0,1"
_set_env_from_runtime "QWEN3VL_CURRICULUM_LATENT_CE" "curriculum_latent_ce" "1,0,0"
_set_env_from_runtime "QWEN3VL_CURRICULUM_STAGE_DATASETS" "curriculum_stage_datasets" ""
_set_env_from_runtime "QWEN3VL_PREPARE_CURRICULUM_DATASETS" "prepare_curriculum_datasets" "0"
_set_env_from_runtime "QWEN3VL_LATENT_CE_TOKEN" "latent_ce_token" "0"
_set_env_from_runtime "QWEN3VL_HIDDEN_STATES_HOOK" "hidden_states_hook" "1"
_set_env_from_main "DATALOADER_NUM_WORKERS" "dataloader_num_workers" "4"

# Runtime behavior
_set_env_from_runtime "QWEN3VL_TRANSPARENT_EVAL_MAX_NEW_TOKENS" "eval_max_new_tokens" "8192"
_set_env_from_runtime "QWEN3VL_COMPILE_VISION_ONLY" "compile_vision_only" "1"
_set_env_from_runtime "PYTORCH_CUDA_ALLOC_CONF" "cuda_alloc_conf" "expandable_segments:True"

# vLLM plugin runtime flags (used by backfill / benchmark subprocesses)
_set_env_from_runtime "VLLM_THINKING" "vllm_thinking" "1"
_set_env_from_runtime "VLLM_ENFORCE_EAGER" "vllm_enforce_eager" "0"
qwen3vl_export_env_from_main_then_runtime "$PYTHON_BIN" "$CONFIG_PATH" "generation.min_continuous_steps" "$QWEN3VL_RUNTIME_ENV_CONFIG" "min_continuous_steps" "MIN_CONTINUOUS_STEPS" "0"
qwen3vl_export_max_continuous_steps_from_yaml "$PYTHON_BIN" "$CONFIG_PATH" "$QWEN3VL_RUNTIME_ENV_CONFIG"

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
FINAL_HANDOFF_DIR="$CKPT_DIR/checkpoint_latest"

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
qwen3vl_snapshot_run_configs "$CONFIG_PATH" "$QWEN3VL_RUNTIME_ENV_CONFIG" "$RUN_DIR"
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

QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" CUDA_VISIBLE_DEVICES="\$BACKFILL_GPUS" $PYTHON_BIN Qwen/inference/backfill_transparent_eval.py --checkpoint_dir "$CKPT_DIR" --checkpoint checkpoint_latest --gpu_memory_utilization \${GPU_MEMORY_UTILIZATION:-0.9} 2>&1 | tee \${BACKFILL_LOG:-/tmp/backfill_thinking_debug.log}
BENCH_DIR="$RUN_DIR/bench"
mkdir -p "\$BENCH_DIR"
QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" CUDA_VISIBLE_DEVICES="\$BENCH_GPUS" $PYTHON_BIN Qwen/evaluation/run_all_benchmarks.py --start-server --gpus "\$BENCH_GPUS" --benchmarks \${BENCHMARKS:-MathVision,MMMU,RealWorldQA} --num-samples \${BENCHMARK_NUM_SAMPLES:-$BENCHMARK_NUM_SAMPLES} --lora-path "$CKPT_DIR/checkpoint_latest" --run-dir "\$BENCH_DIR"
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
    export MASTER_PORT="$(qwen3vl_resolve_master_port "$PYTHON_BIN")"
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
MAIN_NUM_TRAIN_EPOCHS="$(_get_main_config "num_train_epochs" "1.0")"
QWEN3VL_SPLIT_CURRICULUM_STAGES="${QWEN3VL_SPLIT_CURRICULUM_STAGES:-1}"
TOTAL_NUM_TRAIN_EPOCHS="$(_resolve_total_num_train_epochs "$MAIN_NUM_TRAIN_EPOCHS" "$QWEN3VL_CURRICULUM_EPOCHS")"

STAGE_PLAN_LINES=()
STAGE_DATASET_SPECS=()
if [ "$QWEN3VL_SPLIT_CURRICULUM_STAGES" = "1" ] && [ "$QWEN3VL_CURRICULUM_ENABLE" = "1" ]; then
    mapfile -t STAGE_PLAN_LINES < <(
        _build_curriculum_stage_plan \
            "$QWEN3VL_CURRICULUM_EPOCHS" \
            "$QWEN3VL_CURRICULUM_LOSS_TYPES" \
            "$QWEN3VL_CURRICULUM_VAE_TRAINABLE" \
            "$QWEN3VL_CURRICULUM_LORA_TRAINABLE" \
            "$QWEN3VL_CURRICULUM_AUX_SOURCE" \
            "$QWEN3VL_CURRICULUM_LATENT_CE" \
            "$TOTAL_NUM_TRAIN_EPOCHS"
    )
    mapfile -t STAGE_DATASET_SPECS < <(
        _build_curriculum_stage_dataset_plan \
            "$QWEN3VL_CURRICULUM_STAGE_DATASETS" \
            "$DATASET_SPEC" \
            "${#STAGE_PLAN_LINES[@]}"
    )
fi

if [ "$QWEN3VL_PREPARE_CURRICULUM_DATASETS" = "1" ]; then
    _log_wrapper_status "prepare_curriculum_datasets start"
    "$PYTHON_BIN" "$REPO_ROOT/Qwen/data/prepare_qwen3vl_curriculum_datasets.py" 2>&1 | tee -a "$LOG_FILE"
    _log_wrapper_status "prepare_curriculum_datasets done"
fi


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
echo "  - Base spec: $DATASET_SPEC"
echo "  - Mix strategy: ${DATASET_MIX_STRATEGY:-concat}"
if [ "${DATASET_MIX_STRATEGY:-concat}" = "concat" ]; then
  echo "  - Joint shuffle: enabled via concat + Trainer random sampler"
else
  echo "  - Joint shuffle: Trainer random sampler enabled; mix ordering follows ${DATASET_MIX_STRATEGY}"
fi
echo "  - Stage dataset overrides: ${QWEN3VL_CURRICULUM_STAGE_DATASETS:-none}"
echo "  - Auto-prepare derived datasets: $QWEN3VL_PREPARE_CURRICULUM_DATASETS"
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
echo "  - Latent step CE: $QWEN3VL_CURRICULUM_LATENT_CE"
echo "  - Split into separate runs: $QWEN3VL_SPLIT_CURRICULUM_STAGES"
if [ "${#STAGE_PLAN_LINES[@]}" -gt 0 ]; then
  for idx in "${!STAGE_PLAN_LINES[@]}"; do
    stage_line="${STAGE_PLAN_LINES[$idx]}"
    IFS=$'\t' read -r stage_num stage_start stage_end stage_loss stage_vae_trainable stage_lora_trainable stage_aux_source stage_latent_ce <<< "$stage_line"
    stage_dataset_spec="${STAGE_DATASET_SPECS[$idx]:-$DATASET_SPEC}"
    echo "    Stage $stage_num: epochs [$stage_start, $stage_end) loss=$stage_loss vae_trainable=$stage_vae_trainable lora_trainable=$stage_lora_trainable aux_source=$stage_aux_source latent_ce=$stage_latent_ce dataset=$stage_dataset_spec"
  done
fi
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

_log_wrapper_status "wrapper_start config=$CONFIG_PATH run_dir=$RUN_DIR ckpt_dir=$CKPT_DIR cuda_visible_devices=$CUDA_VISIBLE_DEVICES run_backfill=$RUN_BACKFILL run_benchmark=$RUN_BENCHMARK split_curriculum=$QWEN3VL_SPLIT_CURRICULUM_STAGES"

# Run llamafactory CLI with tee for logging
# Use PIPESTATUS to preserve exit code from llamafactory-cli
NPROC_PER_NODE_DEFAULT="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$NPROC_PER_NODE_DEFAULT}"

TRAIN_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == output_dir=* ]]; then
        continue
    fi
    if [[ "$arg" == swanlab_run_name=* ]]; then
        continue
    fi
    if [[ "$arg" == dataset=* || "$arg" == dataset_dir=* || "$arg" == mix_strategy=* || "$arg" == num_train_epochs=* ]]; then
        continue
    fi
    TRAIN_ARGS+=("$arg")
done

_run_llamafactory_stage() {
    local stage_num="$1"
    local stage_start="$2"
    local stage_end="$3"
    local stage_loss="$4"
    local stage_vae_trainable="$5"
    local stage_lora_trainable="$6"
    local stage_aux_source="$7"
    local stage_latent_ce="$8"
    local stage_resume="$9"
    local stage_dataset_spec="${10}"
    local stage_output_dir="$CKPT_DIR/stage_${stage_num}"
    local stage_num_train_epochs
    stage_num_train_epochs="$(_compute_epoch_span "$stage_start" "$stage_end")"

    eval "$(qwen3vl_materialize_dataset_mix "$PYTHON_BIN" "$DATASET_DIR" "$stage_dataset_spec" "$stage_output_dir")"
    local effective_dataset_spec="$QWEN3VL_EFFECTIVE_DATASET_SPEC"
    local effective_dataset_dir="$QWEN3VL_EFFECTIVE_DATASET_DIR"
    local stage_dataset_ratio_applied="${QWEN3VL_DATASET_RATIO_APPLIED:-0}"
    local stage_dataset_mix_summary_path="${QWEN3VL_DATASET_MIX_SUMMARY_PATH:-}"
    local stage_dataset_count="${QWEN3VL_DATASET_COUNT:-1}"

    if [ "$stage_dataset_count" -gt 1 ]; then
      case "${DATASET_STREAMING,,}" in
        1|true|yes)
          echo "❌ Multi-dataset SFT requires streaming=false so the merged dataset can be jointly shuffled." >&2
          return 1
          ;;
      esac
      if [[ -z "$DATASET_MIX_STRATEGY" || "$DATASET_MIX_STRATEGY" == "null" ]]; then
        DATASET_MIX_STRATEGY="concat"
      fi
    fi

    local stage_args=("${TRAIN_ARGS[@]}")
    stage_args+=("output_dir=$stage_output_dir")
    stage_args+=("num_train_epochs=$stage_num_train_epochs")
    stage_args+=("dataset=$effective_dataset_spec")
    stage_args+=("dataset_dir=$effective_dataset_dir")
    if [ "$stage_dataset_count" -gt 1 ]; then
        stage_args+=("mix_strategy=$DATASET_MIX_STRATEGY")
    fi
    if [ -n "$stage_resume" ]; then
        stage_args+=("adapter_name_or_path=$stage_resume")
        stage_args+=("resume_from_checkpoint=null")
    fi
    if [[ "$USE_SWANLAB_CONFIG" == "true" || "$USE_SWANLAB_CONFIG" == "True" || "$USE_SWANLAB_CONFIG" == "1" ]]; then
        local base_run_name="${SWANLAB_RUN_NAME:-$(basename "$RUN_DIR")}"
        stage_args+=("swanlab_run_name=${base_run_name}-stage${stage_num}")
    fi

    export QWEN3VL_CURRICULUM_ENABLE=0
    export QWEN3VL_LOSS_TYPE="$stage_loss"
    export QWEN3VL_VAE_TRAINABLE="$stage_vae_trainable"
    export QWEN3VL_LORA_TRAINABLE="$stage_lora_trainable"
    export QWEN3VL_LATENT_AUX_LOSS_SOURCE="$stage_aux_source"
    export QWEN3VL_LATENT_CE_ACTIVE="$stage_latent_ce"
    if [ -n "$stage_resume" ] && [ -f "$stage_resume/vae.safetensors" ]; then
        export QWEN3VL_VAE_CHECKPOINT_PATH="$stage_resume/vae.safetensors"
    else
        unset QWEN3VL_VAE_CHECKPOINT_PATH || true
    fi
    if [[ "$stage_loss" == *"vae_ce"* ]]; then
        export QWEN3VL_VAE_CE_ENABLE=1
    else
        export QWEN3VL_VAE_CE_ENABLE=0
    fi

    mkdir -p "$stage_output_dir"
    _log_wrapper_status "stage_start stage=$stage_num start_epoch=$stage_start end_epoch=$stage_end stage_num_train_epochs=$stage_num_train_epochs loss_type=$stage_loss handoff=${stage_resume:-none} output_dir=$stage_output_dir"
    echo ""
    echo "========================================================================"
    echo "Launching stage $stage_num"
    echo "  Epoch window: [$stage_start, $stage_end)"
    echo "  Train epochs this stage: $stage_num_train_epochs"
    echo "  Loss type: $stage_loss"
    echo "  VAE trainable: $stage_vae_trainable"
    echo "  LoRA trainable: $stage_lora_trainable"
    echo "  Aux source: $stage_aux_source"
    echo "  Latent CE: $stage_latent_ce"
    echo "  VAE handoff: ${QWEN3VL_VAE_CHECKPOINT_PATH:-none}"
    echo "  Handoff weights: ${stage_resume:-none}"
    echo "  Dataset spec: $stage_dataset_spec"
    echo "  Effective dataset spec: $effective_dataset_spec"
    echo "  Effective dataset dir: $effective_dataset_dir"
    echo "  Dataset entries: $stage_dataset_count"
    echo "  Ratio parsing: $([ "$stage_dataset_ratio_applied" = "1" ] && echo "enabled" || echo "disabled")"
    if [ -n "$stage_dataset_mix_summary_path" ]; then
        echo "  Dataset summary: $stage_dataset_mix_summary_path"
    fi
    echo "  Stage output dir: $stage_output_dir"
    echo "========================================================================"

    local stage_config_path="$CONFIG_PATH"
    local exit_code

    if [ "${NPROC_PER_NODE:-1}" = "1" ]; then
        if [ "$USE_FSDP" = true ]; then
            stage_config_path="$stage_output_dir/runtime_single_gpu_no_fsdp.yaml"
            qwen3vl_materialize_single_gpu_runtime_config "$PYTHON_BIN" "$CONFIG_PATH" "$stage_config_path"
            echo "⚠️  Single visible GPU detected. Using a runtime config without FSDP because the current torch FSDP NO_SHARD path is broken for Qwen3VL tied weights."
            _log_wrapper_status "single_gpu_fsdp_fallback stage=$stage_num source_config=$CONFIG_PATH runtime_config=$stage_config_path"
        fi
    fi

    set +e
    "$PYTHON_BIN" -m llamafactory.cli train "$stage_config_path" "${stage_args[@]}" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    set -e

    _log_wrapper_status "stage_finished stage=$stage_num exit_code=$exit_code"

    if [ "$exit_code" -eq 0 ]; then
        local stage_handoff_dir="$stage_output_dir/checkpoint_latest"
        if [ ! -d "$stage_handoff_dir" ]; then
            echo "❌ Stage $stage_num finished without checkpoint_latest: $stage_handoff_dir" >&2
            _log_wrapper_status "stage_handoff_missing stage=$stage_num path=$stage_handoff_dir"
            return 1
        fi

        rm -rf "$FINAL_HANDOFF_DIR"
        cp -a "$stage_handoff_dir" "$FINAL_HANDOFF_DIR"
        _log_wrapper_status "stage_handoff_ready stage=$stage_num path=$FINAL_HANDOFF_DIR"
    fi

    return "$exit_code"
}

train_exit_code=0
if [ "${#STAGE_PLAN_LINES[@]}" -gt 0 ]; then
    for stage_line in "${STAGE_PLAN_LINES[@]}"; do
        IFS=$'\t' read -r stage_num stage_start stage_end stage_loss stage_vae_trainable stage_lora_trainable stage_aux_source stage_latent_ce <<< "$stage_line"
        stage_resume=""
        if [ "$stage_num" -gt 1 ]; then
            stage_resume="$FINAL_HANDOFF_DIR"
            if [ ! -d "$stage_resume" ]; then
                echo "❌ Missing handoff checkpoint before stage $stage_num: $stage_resume" >&2
                _log_wrapper_status "stage_resume_missing stage=$stage_num path=$stage_resume"
                train_exit_code=1
                break
            fi
        fi
        _run_llamafactory_stage \
            "$stage_num" \
            "$stage_start" \
            "$stage_end" \
            "$stage_loss" \
            "$stage_vae_trainable" \
            "$stage_lora_trainable" \
            "$stage_aux_source" \
            "$stage_latent_ce" \
            "$stage_resume" \
            "${STAGE_DATASET_SPECS[$((stage_num - 1))]:-$DATASET_SPEC}"
        stage_exit_code=$?
        if [ "$stage_exit_code" -ne 0 ]; then
            train_exit_code=$stage_exit_code
            break
        fi
    done
else
    _run_llamafactory_stage \
        "1" \
        "0" \
        "$TOTAL_NUM_TRAIN_EPOCHS" \
        "$QWEN3VL_LOSS_TYPE" \
        "${QWEN3VL_VAE_TRAINABLE:-1}" \
        "${QWEN3VL_LORA_TRAINABLE:-1}" \
        "$QWEN3VL_LATENT_AUX_LOSS_SOURCE" \
        "${QWEN3VL_LATENT_CE_ACTIVE:-1}" \
        "" \
        "$DATASET_SPEC"
    stage_exit_code=$?
    if [ "$stage_exit_code" -ne 0 ]; then
        train_exit_code=$stage_exit_code
    fi
fi
_log_wrapper_status "training_finished exit_code=$train_exit_code"

# ============================================================================
# Post-training: Run backfill transparent eval on all checkpoints
# Uses all training GPUs in parallel (round-robin, M jobs at a time)
# ============================================================================
if [ "$train_exit_code" -eq 0 ] && [ "$RUN_BACKFILL" = "1" ]; then
    _log_wrapper_status "backfill_gate entered exit_code=$train_exit_code run_backfill=$RUN_BACKFILL"
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
    while IFS= read -r CHECKPOINT_NAME; do
        [ -n "$CHECKPOINT_NAME" ] || continue
        CHECKPOINT="$CKPT_DIR/$CHECKPOINT_NAME"

        # Skip if already has results
        if [ -d "$CHECKPOINT/eval_results" ] && [ "$(ls "$CHECKPOINT/eval_results"/backfill_*.json 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  Skipping $CHECKPOINT_NAME - already has backfill results"
            continue
        fi

        CHECKPOINT_LIST+=("$CHECKPOINT_NAME")
    done < <(_list_checkpoint_dirs)

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
                    if QWEN3VL_RUNTIME_ENV_CONFIG="$QWEN3VL_RUNTIME_ENV_CONFIG" CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" Qwen/inference/backfill_transparent_eval.py \
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
from Qwen.inference.vllm_utils import cleanup_vllm_engine_processes
cleanup_vllm_engine_processes()
PY
}

if [ "$train_exit_code" -eq 0 ] && [ "$RUN_BENCHMARK" = "1" ]; then
    _log_wrapper_status "benchmark_gate entered exit_code=$train_exit_code run_benchmark=$RUN_BENCHMARK"
    cleanup_vllm_benchmark_processes
    LATEST_CHECKPOINT="$FINAL_HANDOFF_DIR"
    if [ ! -d "$LATEST_CHECKPOINT" ]; then
        latest_checkpoint_rel="$(_list_checkpoint_dirs | tail -n 1 || true)"
        if [ -n "${latest_checkpoint_rel:-}" ]; then
            LATEST_CHECKPOINT="$CKPT_DIR/$latest_checkpoint_rel"
        fi
    fi
    if [ -z "${LATEST_CHECKPOINT:-}" ] || [ ! -d "$LATEST_CHECKPOINT" ]; then
        _log_wrapper_status "benchmark_skip reason=no_checkpoint"
        echo "⚠️  Benchmark skipped: no checkpoint found in $CKPT_DIR"
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
            --explore \
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
