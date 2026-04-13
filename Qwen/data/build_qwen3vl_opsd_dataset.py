#!/usr/bin/env python3
"""Build OPSD manifests from existing Qwen3-VL thinking datasets.

This script reuses the already-rendered question / rationale PNGs produced by the
thinking-dataset builders. It does not regenerate PNGs or latent `.pt` caches.

Input sources are the existing thinking JSONLs registered in `Qwen/data/dataset_info.json`.
For each row, the builder emits a compact manifest for OPSD training:

- `sample_id`
- `source_dataset`
- `task`
- `question_images`
- `teacher_rationale_images`
- `student_user_text`
- `assistant_target`
- `answer_text`
- `latent_ground_truth`
- `latent_supervision`
- `latent_seq_lens`
- `num_latent_steps`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Qwen" / "data"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"


def _load_dataset_info() -> dict[str, Any]:
    with DATASET_INFO_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dataset_path(dataset_name: str, dataset_info: dict[str, Any]) -> Path:
    entry = dataset_info.get(dataset_name)
    if not entry:
        raise KeyError(f"Unknown dataset '{dataset_name}' in {DATASET_INFO_PATH}")
    file_name = entry.get("file_name")
    if not file_name:
        raise ValueError(f"Dataset '{dataset_name}' is missing file_name in {DATASET_INFO_PATH}")
    path = Path(file_name)
    if not path.is_absolute():
        path = (DATA_DIR / path).resolve()
    return path


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    return content if isinstance(content, str) else ""


def _user_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    content = messages[0].get("content", "")
    return content if isinstance(content, str) else ""


def _split_answer(text: str) -> str:
    marker = "</think>"
    idx = text.find(marker)
    if idx < 0:
        return text.strip()
    return text[idx + len(marker):].strip()


def _strip_leading_image_placeholder(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("<image>"):
        stripped = stripped[len("<image>"):].lstrip()
    return stripped


def _latent_path_to_png(latent_path: str) -> str:
    pt_path = Path(latent_path)
    if pt_path.suffix != ".pt":
        raise ValueError(f"Expected latent cache path ending in .pt, got: {latent_path}")
    if pt_path.parent.name != ".feature_cache":
        raise ValueError(
            f"Expected latent cache under a .feature_cache directory, got: {latent_path}"
        )
    png_path = pt_path.parent.parent / f"{pt_path.stem}.png"
    return str(png_path)


def _build_row(source_dataset: str, row_idx: int, row: dict[str, Any]) -> dict[str, Any]:
    question_images = list(row.get("images") or [])
    rationale_latents = list(row.get("latent_ground_truth") or [])
    rationale_images = [_latent_path_to_png(path) for path in rationale_latents]

    for image_path in question_images + rationale_images:
        if not Path(image_path).exists():
            raise FileNotFoundError(
                f"Missing rendered image required for OPSD manifest: {image_path}"
            )

    sample_id = f"{source_dataset}_{row_idx:07d}"
    return {
        "sample_id": sample_id,
        "source_dataset": source_dataset,
        "task": row.get("task", source_dataset),
        "question_images": question_images,
        "teacher_rationale_images": rationale_images,
        "student_user_text": _strip_leading_image_placeholder(_user_text(row)),
        "assistant_target": _assistant_text(row),
        "answer_text": _split_answer(_assistant_text(row)),
        "latent_ground_truth": rationale_latents,
        "latent_supervision": list(row.get("latent_supervision") or []),
        "latent_seq_lens": list(row.get("latent_seq_lens") or []),
        "num_latent_steps": int(row.get("num_latent_steps", len(rationale_latents))),
    }


def build_manifest(dataset_names: list[str], output_path: Path) -> tuple[int, dict[str, int]]:
    dataset_info = _load_dataset_info()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_dataset_counts: dict[str, int] = {}
    total = 0

    with output_path.open("w", encoding="utf-8") as f_out:
        for dataset_name in dataset_names:
            src_path = _resolve_dataset_path(dataset_name, dataset_info)
            kept = 0
            for row_idx, row in enumerate(_iter_jsonl(src_path)):
                built = _build_row(dataset_name, row_idx, row)
                f_out.write(json.dumps(built, ensure_ascii=False) + "\n")
                kept += 1
            per_dataset_counts[dataset_name] = kept
            total += kept

    return total, per_dataset_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OPSD manifest from existing thinking JSONLs.")
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names from Qwen/data/dataset_info.json",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output OPSD manifest JSONL path",
    )
    args = parser.parse_args()

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not dataset_names:
        raise SystemExit("No datasets specified.")

    output_path = Path(args.output_jsonl)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    total, per_dataset = build_manifest(dataset_names, output_path)
    print(f"[opsd-dataset] wrote {total} rows -> {output_path}")
    print(f"[opsd-dataset] breakdown: {per_dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
