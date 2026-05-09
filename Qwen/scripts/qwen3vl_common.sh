#!/bin/bash

qwen3vl_resolve_root_dir() {
    local repo_root_input="${1:-${REPO_ROOT:-}}"
    local repo_root_abs=""
    if [[ -n "$repo_root_input" ]]; then
        repo_root_abs="$(cd "$repo_root_input" && pwd)"
    else
        repo_root_abs="$(pwd)"
    fi
    (
        cd "$repo_root_abs/../.." && pwd
    )
}


qwen3vl_init_root_dir() {
    if [[ -n "${ROOT_DIR:-}" ]]; then
        export ROOT_DIR
        return 0
    fi
    ROOT_DIR="$(qwen3vl_resolve_root_dir "${1:-${REPO_ROOT:-}}")"
    export ROOT_DIR
}


qwen3vl_require_python_bin() {
    local _deepseek_ocr_dir="$1"
    qwen3vl_init_root_dir "$_deepseek_ocr_dir"
    local python_bin="$ROOT_DIR/envs/ocrflow/bin/python"
    if [ ! -x "$python_bin" ]; then
        echo "❌ Python not found or not executable: $python_bin" >&2
        echo "   Please ensure OCRFlow env exists at: $ROOT_DIR/envs/ocrflow" >&2
        return 1
    fi
    printf '%s\n' "$python_bin"
}


if [[ -z "${QWEN3VL_ROOT_DIR_INITIALIZED:-}" ]]; then
    qwen3vl_init_root_dir "${REPO_ROOT:-}"
    export QWEN3VL_ROOT_DIR_INITIALIZED=1
fi


qwen3vl_get_yaml_value() {
    local python_bin="$1"
    local file_path="$2"
    local key="$3"
    local default_value="${4:-}"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$python_bin" - "$file_path" "$key" "$default_value" <<'PY'
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


qwen3vl_export_env_from_yaml() {
    local python_bin="$1"
    local file_path="$2"
    local env_name="$3"
    local yaml_key="$4"
    local default_value="$5"
    local current_value="${!env_name:-}"
    if [[ -n "$current_value" ]]; then
        export "$env_name=$current_value"
    else
        export "$env_name=$(qwen3vl_get_yaml_value "$python_bin" "$file_path" "$yaml_key" "$default_value")"
    fi
}


qwen3vl_export_env_from_main_then_runtime() {
    local python_bin="$1"
    local main_file_path="$2"
    local main_yaml_key="$3"
    local runtime_file_path="$4"
    local runtime_yaml_key="$5"
    local env_name="$6"
    local default_value="$7"
    local current_value="${!env_name:-}"
    if [[ -n "$current_value" ]]; then
        export "$env_name=$current_value"
        return 0
    fi

    local resolved_value=""
    resolved_value="$(qwen3vl_get_yaml_value "$python_bin" "$main_file_path" "$main_yaml_key" "")"
    if [[ -z "$resolved_value" ]]; then
        resolved_value="$(qwen3vl_get_yaml_value "$python_bin" "$runtime_file_path" "$runtime_yaml_key" "$default_value")"
    fi
    export "$env_name=$resolved_value"
}


qwen3vl_resolve_path() {
    local base_dir="$1"
    local raw_path="$2"

    if [[ -z "$raw_path" ]]; then
        printf '%s\n' ""
        return 0
    fi
    if [[ "$raw_path" = /* ]]; then
        printf '%s\n' "$raw_path"
        return 0
    fi

    local resolved_dir=""
    resolved_dir="$(
        cd "$base_dir" && cd "$(dirname "$raw_path")" && pwd
    )"
    printf '%s\n' "$resolved_dir/$(basename "$raw_path")"
}


qwen3vl_resolve_max_continuous_steps() {
    local python_bin="$1"
    local main_file_path="$2"
    local runtime_file_path="$3"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$python_bin" - "$main_file_path" "$runtime_file_path" <<'PY'
import sys
from pathlib import Path
import yaml

main_file_path = Path(sys.argv[1])
runtime_file_path = Path(sys.argv[2])

try:
    main_cfg = yaml.safe_load(main_file_path.read_text()) or {}
except Exception:
    main_cfg = {}

try:
    runtime_cfg = yaml.safe_load(runtime_file_path.read_text()) or {}
except Exception:
    runtime_cfg = {}

generation_cfg = main_cfg.get("generation") if isinstance(main_cfg.get("generation"), dict) else {}
data_cfg = main_cfg.get("data") if isinstance(main_cfg.get("data"), dict) else {}

raw_max = generation_cfg.get("max_continuous_steps", runtime_cfg.get("max_continuous_steps"))
raw_tokens = generation_cfg.get(
    "max_new_tokens",
    data_cfg.get("max_response_length", runtime_cfg.get("max_new_tokens", 40960)),
)

try:
    max_new_tokens = int(raw_tokens)
except Exception:
    max_new_tokens = 40960

derived = max(1, max_new_tokens // 2 if max_new_tokens > 1 else 1)

if raw_max is None or str(raw_max).strip() in {"", "null", "None"}:
    print(derived)
else:
    try:
        print(max(0, int(raw_max)))
    except Exception:
        print(derived)
PY
}


qwen3vl_export_max_continuous_steps_from_yaml() {
    local python_bin="$1"
    local main_file_path="$2"
    local runtime_file_path="$3"
    local current_value="${MAX_CONTINUOUS_STEPS:-}"
    if [[ -z "$current_value" ]]; then
        current_value="$(qwen3vl_resolve_max_continuous_steps "$python_bin" "$main_file_path" "$runtime_file_path")"
    fi
    export "MAX_CONTINUOUS_STEPS=$current_value"
}


qwen3vl_resolve_master_port() {
    local python_bin="$1"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$python_bin" - "${MASTER_PORT:-}" <<'PY'
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


qwen3vl_snapshot_run_configs() {
    local config_path="$1"
    local runtime_env_config="$2"
    local dest_dir="$3"
    cp -f "$config_path" "$dest_dir/$(basename "$config_path")"
    cp -f "$runtime_env_config" "$dest_dir/$(basename "$runtime_env_config")"
}


qwen3vl_list_checkpoint_dirs() {
    local python_bin="$1"
    local ckpt_dir="$2"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$python_bin" - "$ckpt_dir" <<'PY'
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1]).resolve()
candidates = []
for path in ckpt_dir.glob("checkpoint-*"):
    if path.is_dir():
        candidates.append(path)
for stage_dir in ckpt_dir.glob("stage_*"):
    if not stage_dir.is_dir():
        continue
    for path in stage_dir.glob("checkpoint-*"):
        if path.is_dir():
            candidates.append(path)

for path in sorted(candidates, key=lambda p: str(p.relative_to(ckpt_dir))):
    print(path.relative_to(ckpt_dir))
PY
}


qwen3vl_count_visible_gpus() {
    local python_bin="$1"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}"
        return
    fi
    env PYTHONPATH= PYTHONSAFEPATH=1 "$python_bin" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
}


qwen3vl_gcd() {
    local a="${1:-0}"
    local b="${2:-0}"
    if ! [[ "$a" =~ ^[0-9]+$ ]]; then
        a=0
    fi
    if ! [[ "$b" =~ ^[0-9]+$ ]]; then
        b=0
    fi
    while [[ "$b" -ne 0 ]]; do
        local remainder=$((a % b))
        a="$b"
        b="$remainder"
    done
    printf '%s\n' "$a"
}


qwen3vl_lcm() {
    local a="${1:-1}"
    local b="${2:-1}"
    if ! [[ "$a" =~ ^[0-9]+$ ]] || [[ "$a" -lt 1 ]]; then
        a=1
    fi
    if ! [[ "$b" =~ ^[0-9]+$ ]] || [[ "$b" -lt 1 ]]; then
        b=1
    fi
    local gcd_value
    gcd_value="$(qwen3vl_gcd "$a" "$b")"
    printf '%s\n' $(((a / gcd_value) * b))
}


qwen3vl_resolve_generation_batch_size() {
    local num_generations="${1:-1}"
    local per_device_train_batch_size="${2:-1}"
    local num_processes="${3:-1}"
    local requested_min="${4:-}"

    if ! [[ "$num_generations" =~ ^[0-9]+$ ]] || [[ "$num_generations" -lt 1 ]]; then
        num_generations=1
    fi
    if ! [[ "$per_device_train_batch_size" =~ ^[0-9]+$ ]] || [[ "$per_device_train_batch_size" -lt 1 ]]; then
        per_device_train_batch_size=1
    fi
    if ! [[ "$num_processes" =~ ^[0-9]+$ ]] || [[ "$num_processes" -lt 1 ]]; then
        num_processes=1
    fi

    local global_batch_size=$((per_device_train_batch_size * num_processes))
    local base_generation_batch_size=$((num_generations * global_batch_size))

    if ! [[ "$requested_min" =~ ^[0-9]+$ ]] || [[ "$requested_min" -lt 1 ]]; then
        printf '%s\n' "$base_generation_batch_size"
        return 0
    fi

    local aligned_min
    aligned_min="$(qwen3vl_lcm "$requested_min" "$global_batch_size")"
    aligned_min="$(qwen3vl_lcm "$aligned_min" "$num_generations")"
    if [[ "$aligned_min" -gt "$base_generation_batch_size" ]]; then
        printf '%s\n' "$aligned_min"
        return 0
    fi
    printf '%s\n' "$base_generation_batch_size"
}


qwen3vl_configure_ray_noset_cuda_visible_devices() {
    unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES || true
}


qwen3vl_sanitize_log_stream() {
    stdbuf -oL -eL perl -MIO::Handle -ne 'BEGIN { STDOUT->autoflush(1) } s/\e\[[0-9;]*[[:alpha:]]//g; s/\r/\n/g; print'
}


qwen3vl_copy_if_present() {
    local src="$1"
    local dest_dir="$2"
    if [[ -n "$src" && -f "$src" ]]; then
        cp "$src" "$dest_dir/$(basename "$src")"
    fi
}


qwen3vl_prepare_thinking_tokenizer_dir() {
    local python_bin="$1"
    local source_path="$2"
    local output_dir="$3"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$python_bin" - "$source_path" "$output_dir" <<'PY'
import sys
from pathlib import Path

from transformers import AutoTokenizer

source_path = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(str(source_path), trust_remote_code=True)
for token in ("<latent>", "<think_sep>"):
    if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [token]},
            replace_additional_special_tokens=False,
        )
tokenizer.save_pretrained(str(output_dir))
for filename in (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "image_processor_config.json",
    "feature_extractor.json",
):
    src = source_path / filename
    dst = output_dir / filename
    if src.exists() and not dst.exists():
        dst.write_bytes(src.read_bytes())
print(str(output_dir))
PY
}


qwen3vl_materialize_single_gpu_runtime_config() {
    local python_bin="$1"
    local src_config="$2"
    local dest_config="$3"
    env PYTHONPATH= PYTHONSAFEPATH=1 "$python_bin" - "$src_config" "$dest_config" <<'PY'
import sys
from pathlib import Path

import yaml

src_config = Path(sys.argv[1])
dest_config = Path(sys.argv[2])
data = yaml.safe_load(src_config.read_text()) or {}

# Current PyTorch FSDP clamps single-process world size to NO_SHARD. For
# Qwen3VL's tied embed/lm_head weights, that upstream path crashes under
# use_orig_params.
data.pop("fsdp", None)
data.pop("fsdp_config", None)

dest_config.parent.mkdir(parents=True, exist_ok=True)
dest_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY
}
