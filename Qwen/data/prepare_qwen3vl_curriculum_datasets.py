#!/usr/bin/env python3
"""
Prepare late-stage curriculum datasets for Qwen3VL SFT.

This script derives task-focused views from existing local datasets:
- OCR strict: exact OCR / bbox / markdown reconstruction
- DeepVision thinking concise: short-form VQA / MCQ / exact-answer supervision
-   while preserving the same latent thinking path
- R1OV MCQ concise: same image/question/latent path, but answer shortened to option-first format
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
DATA_DIR = REPO_ROOT / "Qwen" / "data"
SFT_DIR = DATA_DIR / "sft"

OCR_SUPPLEMENT = SFT_DIR / "ocrvl_ocr_supplement.jsonl"
OCR_STRICT = SFT_DIR / "ocrvl_ocr_strict.jsonl"
DEEPVISION_THINKING_SRC = SFT_DIR / "deepvision_thinking.jsonl"
DEEPVISION_THINKING_CONCISE = SFT_DIR / "deepvision_thinking_concise.jsonl"
R1OV_SRC = SFT_DIR / "r1ov_thinking.jsonl"
R1OV_THINKING_CONCISE = SFT_DIR / "r1ov_thinking_concise.jsonl"

STRICT_OCR_TASKS = {"bbox_ocr", "full_document_ocr", "markdown_conversion"}
MCQ_PATTERNS = (
    "\nOptions:",
    "Please select the correct answer from the options above.",
    "provide the correct option letter",
    "Choose the correct option",
    "\nA.",
    "\n(A)",
)


def _needs_rebuild(output_path: Path, input_paths: Iterable[Path]) -> bool:
    if not output_path.exists():
        return True
    out_mtime = output_path.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > out_mtime for path in input_paths)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _assistant_text(row: dict) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    return content if isinstance(content, str) else ""


def _user_text(row: dict) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    content = messages[0].get("content", "")
    return content if isinstance(content, str) else ""


def _split_think_answer(text: str) -> tuple[str, str]:
    close_tag = "</think>"
    close_idx = text.find(close_tag)
    if close_idx < 0:
        return "", text.strip()
    prefix = text[: close_idx + len(close_tag)]
    answer = text[close_idx + len(close_tag) :].strip()
    return prefix, answer


def _extract_option_answer(answer: str) -> str | None:
    stripped = answer.strip()
    if re.fullmatch(r"\(?[A-H]\)?", stripped):
        return stripped.strip("()")

    patterns = [
        r"(?i)\banswer\s*[:：]?\s*\(?([A-H])\)?\b",
        r"(?i)\boption\s*[:：]?\s*\(?([A-H])\)?\b",
        r"(?i)\bcorrect\s+answer\s*[:：]?\s*\(?([A-H])\)?\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, stripped)
        if matches:
            return matches[-1].upper()
    return None


def _is_mcq_prompt(user_text: str) -> bool:
    return any(pattern in user_text for pattern in MCQ_PATTERNS)


def _extract_option_text_from_prompt(user_text: str, option_letter: str) -> str | None:
    patterns = [
        rf"(?im)^\s*\(?{re.escape(option_letter)}[\)\.\:]\s*(.+)$",
        rf"(?im)^\s*{re.escape(option_letter)}\s*[-]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_text)
        if match:
            option_text = match.group(1).strip()
            if option_text:
                return option_text
    return None


def build_ocr_strict() -> tuple[int, Counter]:
    stats = Counter()

    def rows():
        for row in _iter_jsonl(OCR_SUPPLEMENT):
            task = row.get("task", "unknown")
            if task in STRICT_OCR_TASKS:
                stats[task] += 1
                yield row

    count = _write_jsonl(OCR_STRICT, rows())
    return count, stats


def build_deepvision_thinking_concise(max_chars: int = 96, max_lines: int = 2) -> tuple[int, Counter]:
    stats = Counter()

    def rows():
        for row in _iter_jsonl(DEEPVISION_THINKING_SRC):
            assistant = _assistant_text(row).strip()
            think_prefix, answer = _split_think_answer(assistant)
            if not think_prefix or not answer:
                stats["skip_no_think_or_answer"] += 1
                continue
            if len(answer) > max_chars:
                stats["skip_long_answer"] += 1
                continue
            if answer.count("\n") >= max_lines:
                stats["skip_multiline_answer"] += 1
                continue

            new_row = dict(row)
            new_messages = [dict(msg) for msg in row.get("messages", [])]
            new_messages[-1]["content"] = f"{think_prefix}{answer.strip()}"
            new_row["messages"] = new_messages
            new_row["task"] = "deepvision_thinking_concise"
            new_row["answer_style"] = "concise_final_answer"
            stats["kept"] += 1
            yield new_row

    count = _write_jsonl(DEEPVISION_THINKING_CONCISE, rows())
    return count, stats


def build_r1onevision_thinking_concise() -> tuple[int, Counter]:
    stats = Counter()

    def rows():
        for row in _iter_jsonl(R1OV_SRC):
            user_text = _user_text(row)
            if not _is_mcq_prompt(user_text):
                continue

            assistant = _assistant_text(row)
            think_prefix, answer = _split_think_answer(assistant)
            if not think_prefix or not answer:
                stats["skip_no_think_or_answer"] += 1
                continue

            concise_answer = _extract_option_answer(answer)
            if concise_answer is None:
                stats["skip_unparsed_answer"] += 1
                continue
            option_text = _extract_option_text_from_prompt(user_text, concise_answer)

            new_row = dict(row)
            new_messages = [dict(msg) for msg in row.get("messages", [])]
            if option_text:
                new_messages[-1]["content"] = f"{think_prefix}{concise_answer}. {option_text}"
                new_row["answer_style"] = "concise_option_with_text"
            else:
                new_messages[-1]["content"] = f"{think_prefix}{concise_answer}"
                new_row["answer_style"] = "concise_option_only"
            new_row["messages"] = new_messages
            new_row["task"] = "r1ov_thinking_concise"
            stats["kept"] += 1
            yield new_row

    count = _write_jsonl(R1OV_THINKING_CONCISE, rows())
    return count, stats


def main() -> int:
    jobs = [
        ("ocr_strict", OCR_STRICT, [OCR_SUPPLEMENT, SCRIPT_PATH], build_ocr_strict),
        (
            "deepvision_thinking_concise",
            DEEPVISION_THINKING_CONCISE,
            [DEEPVISION_THINKING_SRC, SCRIPT_PATH],
            build_deepvision_thinking_concise,
        ),
        (
            "r1ov_thinking_concise",
            R1OV_THINKING_CONCISE,
            [R1OV_SRC, SCRIPT_PATH],
            build_r1onevision_thinking_concise,
        ),
    ]

    for name, output_path, input_paths, builder in jobs:
        if not _needs_rebuild(output_path, input_paths):
            print(f"[curriculum-datasets] skip {name}: up to date -> {output_path}")
            continue

        count, stats = builder()
        print(f"[curriculum-datasets] built {name}: {count} rows -> {output_path}")
        if stats:
            print(f"[curriculum-datasets] stats {name}: {dict(stats)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
