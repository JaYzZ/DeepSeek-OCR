#!/usr/bin/env python3
"""Convert DeepVision-103K parquet files into VERL RL parquet format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import datasets
import pandas as pd


DEFAULT_DATA_DIR = Path("/share/project/xiyan/huggingface/skylenage/DeepVision-103K")
DEFAULT_OUTPUT_DIR = Path("/share/project/xiyan/sources/DeepSeek-OCR/Qwen/data/deepvision_103k_verl")


def _to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray, dict, list, tuple)):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_python(v) for v in value]
    if isinstance(value, list):
        return [_to_python(v) for v in value]
    return value


def _ensure_image_token(prompt: list[dict[str, Any]], has_image: bool) -> list[dict[str, Any]]:
    if not has_image:
        return prompt

    prompt = [dict(message) for message in prompt]
    for idx in range(len(prompt) - 1, -1, -1):
        if prompt[idx].get("role") != "user":
            continue
        content = str(prompt[idx].get("content") or "")
        if "<image>" not in content:
            prompt[idx]["content"] = f"<image>\n{content}".strip()
        return prompt

    prompt.append({"role": "user", "content": "<image>"})
    return prompt


def _normalize_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = _to_python(row.get("prompt"))
    messages: list[dict[str, str]] = []
    if isinstance(prompt, list):
        for message in prompt:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip() or "user"
            content = str(message.get("content") or "")
            messages.append({"role": role, "content": content})

    if not messages:
        question = str(row.get("question") or "").strip()
        messages = [{"role": "user", "content": f"<image>\n{question}".strip()}]

    images = _to_python(row.get("images")) or []
    return _ensure_image_token(messages, has_image=bool(images))


def _normalize_equivalent_answers(reward_model: dict[str, Any]) -> list[str]:
    answers = _to_python(reward_model.get("equivalent_answers")) or []
    normalized: list[str] = []
    for answer in answers:
        text = str(answer).strip()
        if not text or text == "__EMPTY__":
            continue
        normalized.append(text)
    return normalized


def _parse_pass_rate(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            denom = float(right)
            if denom == 0:
                return None
            return float(left) / denom
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _build_record(row: dict[str, Any], source_name: str, row_idx: int) -> dict[str, Any] | None:
    reward_model = _to_python(row.get("reward_model")) or {}
    ground_truth = str(reward_model.get("ground_truth") or "").strip()
    if not ground_truth:
        return None

    images = _to_python(row.get("images")) or []
    if not images:
        return None

    data_source = str(row.get("data_source") or source_name)
    question = str(row.get("question") or "").strip()
    pass_rate_raw = row.get("mimo-pass_rate", row.get("pass_rate"))

    return {
        "data_source": f"deepvision::{data_source}",
        "prompt": _normalize_prompt(row),
        "images": images,
        "ability": str(row.get("ability") or "math"),
        "reward_model": {
            "style": str(reward_model.get("style") or "rule"),
            "ground_truth": ground_truth,
            "equivalent_answers": _normalize_equivalent_answers(reward_model),
        },
        "extra_info": {
            "split": source_name,
            "index": row_idx,
            "question": question,
            "pass_rate": _parse_pass_rate(pass_rate_raw),
            "pass_rate_raw": None if pass_rate_raw is None else str(pass_rate_raw),
            "reward_model": reward_model,
        },
    }


def _load_records(data_dir: Path, include_sources: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        source_name = parquet_path.stem
        if include_sources and source_name not in include_sources:
            continue
        frame = pd.read_parquet(parquet_path)
        for row_idx, (_, row) in enumerate(frame.iterrows()):
            record = _build_record(_to_python(row.to_dict()), source_name=source_name, row_idx=row_idx)
            if record is not None:
                records.append(record)
    return records


def _split_records(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["data_source"], []).append(record)

    rng = random.Random(seed)
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    for source, source_records in sorted(grouped.items()):
        source_records = list(source_records)
        rng.shuffle(source_records)
        val_count = max(1, int(round(len(source_records) * val_ratio)))
        val_count = min(val_count, len(source_records) - 1) if len(source_records) > 1 else len(source_records)
        val_records.extend(source_records[:val_count])
        train_records.extend(source_records[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_val_samples is not None:
        val_records = val_records[:max_val_samples]
    return train_records, val_records


def _write_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(records).to_parquet(str(output_path))


def _parse_sources(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-file", default="train.parquet")
    parser.add_argument("--val-file", default="val.parquet")
    parser.add_argument("--sources", default="math-77k,visual_logic-26k")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    include_sources = _parse_sources(args.sources)
    records = _load_records(args.data_dir, include_sources=include_sources)
    train_records, val_records = _split_records(
        records=records,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    train_path = args.output_dir / args.train_file
    val_path = args.output_dir / args.val_file
    _write_parquet(train_records, train_path)
    _write_parquet(val_records, val_path)

    summary = {
        "data_dir": str(args.data_dir),
        "sources": include_sources,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "train_file": str(train_path),
        "val_file": str(val_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
