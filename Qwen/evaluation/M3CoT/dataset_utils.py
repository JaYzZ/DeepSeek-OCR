import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_M3COT_DATA_ROOT = "/share/project/xiyan/huggingface/LightChen2333/M3CoT/data"


def _resolve_data_root() -> str:
    return os.environ.get("M3COT_DATA_ROOT", DEFAULT_M3COT_DATA_ROOT)


def _resolve_split(dataset_name: str) -> str:
    normalized = str(dataset_name or "M3CoT").strip().upper()
    if normalized in {"M3COT", "M3COT_TEST", "TEST"}:
        return "test"
    if normalized in {"M3COT_VAL", "M3COT_VALIDATION", "VALIDATION", "VAL", "DEV"}:
        return "validation"
    raise ValueError(f"Unsupported M3CoT dataset name: {dataset_name}")


def _glob_split_files(data_root: str, split: str) -> list[Path]:
    root = Path(data_root)
    pattern = {
        "test": "test-*.parquet",
        "validation": "validation-*.parquet",
    }[split]
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No M3CoT parquet shards found for split={split} under {root}")
    return files


def _normalize_choices(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]


def load_dataset(dataset_name: str = "M3CoT") -> pd.DataFrame:
    split = _resolve_split(dataset_name)
    files = _glob_split_files(_resolve_data_root(), split)
    data = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    data["choices"] = data["choices"].apply(_normalize_choices)
    data["index"] = data["id"].astype(str)
    return data


def _stable_hash(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(lambda x: hashlib.md5(x.encode()).hexdigest())


def deterministic_limit(data: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0 or limit >= len(data):
        return data

    # Keep M3CoT subsampling deterministic while matching the original split's
    # broad distribution as closely as possible.
    stratify_col = None
    for candidate in ("domain", "topic", "category"):
        if candidate in data.columns:
            stratify_col = candidate
            break

    limited = data.copy()
    limited["_hash"] = _stable_hash(limited["index"])
    if stratify_col is None:
        return limited.sort_values("_hash").head(limit).drop(columns=["_hash"])

    groups = []
    for group_name, group_df in limited.groupby(stratify_col, sort=True):
        group_sorted = group_df.sort_values("_hash").reset_index(drop=True)
        groups.append((str(group_name), group_sorted))

    num_groups = len(groups)
    if num_groups == 0:
        return limited.sort_values("_hash").head(limit).drop(columns=["_hash"])

    total_count = len(limited)
    group_sizes = {group_name: len(group_df) for group_name, group_df in groups}
    exact_targets = {
        group_name: (limit * group_sizes[group_name]) / total_count
        for group_name, _ in groups
    }
    take_counts = {
        group_name: min(group_sizes[group_name], int(exact_targets[group_name]))
        for group_name, _ in groups
    }
    remainder = limit - sum(take_counts.values())

    selected_parts = []
    leftovers = []
    for group_name, group_df in groups:
        take = take_counts[group_name]
        if take > 0:
            selected_parts.append(group_df.head(take))
        remaining_df = group_df.iloc[take:].copy()
        if not remaining_df.empty:
            remaining_df["_group_name"] = group_name
            leftovers.append(remaining_df)

    if remainder > 0 and leftovers:
        group_priority = sorted(
            (
                (
                    exact_targets[group_name] - take_counts[group_name],
                    group_name,
                )
                for group_name, _ in groups
                if take_counts[group_name] < group_sizes[group_name]
            ),
            key=lambda item: (-item[0], item[1]),
        )
        extra_take_counts = {group_name: 0 for group_name, _ in groups}
        for _, group_name in group_priority:
            if remainder <= 0:
                break
            extra_take_counts[group_name] += 1
            remainder -= 1

        for group_name, group_df in groups:
            extra_take = extra_take_counts[group_name]
            if extra_take > 0:
                selected_parts.append(group_df.iloc[take_counts[group_name]: take_counts[group_name] + extra_take])

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else limited.head(0).copy()
    if len(selected) < limit:
        selected_ids = set(selected["index"].astype(str).tolist())
        fallback = limited[~limited["index"].astype(str).isin(selected_ids)].sort_values("_hash").head(limit - len(selected))
        if not fallback.empty:
            selected = pd.concat([selected, fallback], ignore_index=True)

    return selected.drop(columns=["_hash"], errors="ignore")


def dump_image(line: dict[str, Any] | pd.Series, img_root: str) -> list[str]:
    os.makedirs(img_root, exist_ok=True)
    image_obj = line["image"]
    if not isinstance(image_obj, dict) or "bytes" not in image_obj:
        raise ValueError("M3CoT image field must be a dict containing raw image bytes")

    image_id = str(line.get("id") or line.get("index"))
    path_hint = str(image_obj.get("path") or "")
    suffix = Path(path_hint).suffix or ".png"
    out_path = Path(img_root) / f"{image_id}{suffix}"
    if not out_path.exists():
        with open(out_path, "wb") as f:
            f.write(image_obj["bytes"])
    return [str(out_path)]
