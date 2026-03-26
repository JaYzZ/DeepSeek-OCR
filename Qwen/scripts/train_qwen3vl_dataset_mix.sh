#!/bin/bash

qwen3vl_materialize_dataset_mix() {
  local python_bin="$1"
  local dataset_dir="$2"
  local dataset_spec="$3"
  local output_dir="$4"

  "$python_bin" - "$dataset_dir" "$dataset_spec" "$output_dir" <<'PY'
import json
import math
import os
import random
import shlex
import sys
from pathlib import Path


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def select_count(total: int, ratio: float) -> int:
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"Dataset ratio must be in (0, 1], got {ratio}.")
    if ratio >= 1.0:
        return total
    return max(1, min(total, int(total * ratio + 0.5)))


def subset_jsonl(src_path: Path, dst_path: Path, keep_count: int, seed: int) -> None:
    total = line_count(src_path)
    if keep_count >= total:
        dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        return

    chosen = sorted(random.Random(seed).sample(range(total), keep_count))
    chosen_iter = iter(chosen)
    next_idx = next(chosen_iter, None)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if idx == next_idx:
                dst.write(line)
                next_idx = next(chosen_iter, None)
                if next_idx is None:
                    break


dataset_dir = Path(sys.argv[1]).resolve()
dataset_spec = sys.argv[2]
output_dir = Path(sys.argv[3]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
dataset_info_path = dataset_dir / "dataset_info.json"
seed = int(os.environ.get("QWEN3VL_DATASET_MIX_SEED", "42"))

with dataset_info_path.open("r", encoding="utf-8") as f:
    dataset_info = json.load(f)

entries = []
ratio_applied = False
for raw_item in dataset_spec.split(","):
    item = raw_item.strip()
    if not item:
        continue

    name = item
    ratio = None
    if ":" in item:
        maybe_name, maybe_ratio = item.rsplit(":", 1)
        maybe_name = maybe_name.strip()
        maybe_ratio = maybe_ratio.strip()
        try:
            ratio = float(maybe_ratio)
            name = maybe_name
            ratio_applied = True
        except ValueError:
            ratio = None

    if name not in dataset_info:
        raise ValueError(f"Undefined dataset {name} in {dataset_info_path}.")

    entry = dict(dataset_info[name])
    raw_file_name = entry.get("file_name")
    if not raw_file_name:
        raise ValueError(f"Dataset {name} is missing file_name in {dataset_info_path}.")

    source_path = Path(raw_file_name)
    if not source_path.is_absolute():
        source_path = (dataset_dir / source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file for {name} not found: {source_path}")
    if source_path.is_dir():
        raise ValueError(f"Dataset ratio parsing currently expects file datasets, got directory: {source_path}")

    total_samples = line_count(source_path)
    selected_samples = total_samples
    subset_path = source_path
    if ratio is not None:
        selected_samples = select_count(total_samples, ratio)
        if selected_samples < total_samples:
            runtime_dir = output_dir / "runtime_dataset_dir"
            subset_name = f"{name}__ratio_{ratio:.6f}".replace("/", "_") + source_path.suffix
            subset_path = runtime_dir / subset_name
            subset_jsonl(source_path, subset_path, selected_samples, seed + len(entries))

    entry["file_name"] = str(subset_path)
    entry.pop("num_samples", None)
    entries.append(
        {
            "name": name,
            "ratio": ratio,
            "total_samples": total_samples,
            "selected_samples": selected_samples,
            "source_path": str(source_path),
            "materialized_path": str(subset_path),
            "config": entry,
        }
    )

summary_path = output_dir / "dataset_mix_summary.json"
summary_payload = {
    "dataset_spec": dataset_spec,
    "seed": seed,
    "ratio_applied": ratio_applied,
    "datasets": [
        {
            "name": entry["name"],
            "ratio": entry["ratio"],
            "total_samples": entry["total_samples"],
            "selected_samples": entry["selected_samples"],
            "source_path": entry["source_path"],
            "materialized_path": entry["materialized_path"],
        }
        for entry in entries
    ],
}
summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

effective_dataset_dir = dataset_dir
effective_dataset_spec = ",".join(entry["name"] for entry in entries)
if ratio_applied:
    runtime_dir = output_dir / "runtime_dataset_dir"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dataset_info = {entry["name"]: entry["config"] for entry in entries}
    (runtime_dir / "dataset_info.json").write_text(json.dumps(runtime_dataset_info, indent=2), encoding="utf-8")
    effective_dataset_dir = runtime_dir

dataset_count = len(entries)
print(f"QWEN3VL_EFFECTIVE_DATASET_DIR={shlex.quote(str(effective_dataset_dir))}")
print(f"QWEN3VL_EFFECTIVE_DATASET_SPEC={shlex.quote(effective_dataset_spec)}")
print(f"QWEN3VL_DATASET_RATIO_APPLIED={int(ratio_applied)}")
print(f"QWEN3VL_DATASET_MIX_SUMMARY_PATH={shlex.quote(str(summary_path))}")
print(f"QWEN3VL_DATASET_COUNT={dataset_count}")
PY
}
