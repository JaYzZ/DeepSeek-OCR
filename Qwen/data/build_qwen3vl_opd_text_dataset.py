#!/usr/bin/env python3
"""Build text-privileged OPD manifests from existing Qwen3-VL thinking datasets.

This variant keeps the same latent supervision fields as the default OPD builder,
but replaces teacher-side reasoning images with teacher-side reasoning text.
The teacher prompt can then consume the original textual rationale directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from Qwen.data.build_qwen3vl_opsd_dataset import (
    REPO_ROOT,
    _assistant_text,
    _load_dataset_info,
    _resolve_dataset_path,
    _split_answer,
    _strip_leading_image_placeholder,
    _user_text,
)
from Qwen.data.utils import extract_thinking_and_answer


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _teacher_rationale_text(row: dict[str, Any]) -> str:
    cot = str(row.get("cot") or "").strip()
    if cot:
        thinking, _ = extract_thinking_and_answer(cot, max_chars=None, return_chunks=False)
        if thinking:
            return thinking.strip()
        return cot.replace("<think>", "").replace("</think>", "").strip()

    assistant = _assistant_text(row)
    thinking, _ = extract_thinking_and_answer(assistant, max_chars=None, return_chunks=False)
    return thinking.strip()


def _build_row(source_dataset: str, row_idx: int, row: dict[str, Any]) -> dict[str, Any]:
    question_images = list(row.get("images") or [])
    for image_path in question_images:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Missing question image required for OPD text manifest: {image_path}")

    sample_id = f"{source_dataset}_{row_idx:07d}"
    return {
        "sample_id": sample_id,
        "source_dataset": source_dataset,
        "task": row.get("task", source_dataset),
        "question_images": question_images,
        "teacher_rationale_images": [],
        "teacher_rationale_text": _teacher_rationale_text(row),
        "student_user_text": _strip_leading_image_placeholder(_user_text(row)),
        "assistant_target": _assistant_text(row),
        "answer_text": _split_answer(_assistant_text(row)),
        "latent_ground_truth": list(row.get("latent_ground_truth") or []),
        "latent_supervision": list(row.get("latent_supervision") or []),
        "latent_seq_lens": list(row.get("latent_seq_lens") or []),
        "num_latent_steps": int(row.get("num_latent_steps", len(row.get("latent_ground_truth") or []))),
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
    parser = argparse.ArgumentParser(description="Build text-privileged OPD manifest from existing thinking JSONLs.")
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names from Qwen/data/dataset_info.json",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output OPD text manifest JSONL path",
    )
    args = parser.parse_args()

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not dataset_names:
        raise SystemExit("No datasets specified.")

    output_path = Path(args.output_jsonl)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    total, per_dataset = build_manifest(dataset_names, output_path)
    print(f"[opd-text-dataset] wrote {total} rows -> {output_path}")
    print(f"[opd-text-dataset] breakdown: {per_dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
