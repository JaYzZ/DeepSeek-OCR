#!/bin/bash

qwen3vl_require_python_bin() {
    local _deepseek_ocr_dir="$1"
    local project_root="${ROOT_DIR:-/share/project/xiyan}"
    local python_bin="$project_root/envs/ocrflow/bin/python"
    if [ ! -x "$python_bin" ]; then
        echo "❌ Python not found or not executable: $python_bin" >&2
        echo "   Please ensure OCRFlow env exists at: $project_root/envs/ocrflow" >&2
        return 1
    fi
    printf '%s\n' "$python_bin"
}


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
