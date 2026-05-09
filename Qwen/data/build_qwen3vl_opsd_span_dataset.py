#!/usr/bin/env python3
"""Build a unified r1ov manifest for warmup and main OPSD-span training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen.data.utils import extract_thinking_and_answer

DATA_DIR = REPO_ROOT / "Qwen" / "data"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"

DEFAULT_DATASET_NAME = "r1ov_thinking"
LATENT_TOKEN = "<latent>"
THINK_START = "<think>"
THINK_END = "</think>"
MAX_GLOBAL_SPANS = 8
MIN_COMPRESSED_SPAN_STEPS = 2
MIN_SHORT_COMPRESSED_SPAN_STEPS = 1

_ANCHOR_HINTS = (
    "observe",
    "looking at",
    "look at",
    "therefore",
    "thus",
    "so ",
    "hence",
    "because",
    "compare",
    "suppose",
    "substitute",
    "verify",
    "排除",
    "观察",
    "因此",
    "所以",
    "可得",
    "代入",
    "验证",
)

_INTRO_PATTERNS = (
    r"^(okay|ok)\b",
    r"^so\b",
    r"^i need to\b",
    r"^i'm trying to\b",
    r"^let me\b",
    r"^first\b",
    r"^题目",
    r"^我需要",
    r"^先",
)

_OBSERVE_PATTERNS = (
    r"\b(as seen|seen in the image|looking at|look at the image|the image shows)\b",
    r"\b(the diagram|the figure|the map|the chart|the table)\b",
    r"\b(image|picture|map|diagram|figure|table|graph)\b",
    r"(图中|图片中|从图中|观察到)",
)

_LEADING_LATENT_OPEN_PATTERNS = (
    r"^(the image shows|image shows|the picture shows|picture shows)\b",
    r"^(looking at the image|looking at this image|from the image|from this image)\b",
    r"^(the chart shows|the graph shows|the table shows|the figure shows|the diagram shows)\b",
    r"^(图中|图片中|从图中|从图片中|观察图中|观察图片)",
)

_KNOWLEDGE_PATTERNS = (
    r"\b(from the lecture|i know that|recall that|remember that)\b",
    r"\b(means|is defined as|the formula|concentration is|temperature is)\b",
    r"(根据题意|根据定义|已知|公式)",
)

_OPTION_PATTERNS = (
    r"\b(option|choice|choices)\b",
    r"\b[a-d]\.\b",
    r"\b[a-d]\s+is\b",
    r"(选项|答案选项)",
)

_CALC_PATTERNS = (
    r"\b(calculate|comput(e|ing)|multiply|divid(e|ing)|subtract|sum|total)\b",
    r"[0-9]+\s*[\+\-\*/=]\s*[0-9]+",
    r"(计算|代入|相加|相减|相乘|相除)",
)

_VERIFY_PATTERNS = (
    r"\b(check|double-check|verify)\b",
    r"(验证|检查)",
)

_CONCLUDE_PATTERNS = (
    r"\b(therefore|thus|hence)\b",
    r"\b(the answer is|the correct answer is)\b",
    r"\bso,? the answer\b",
    r"(因此|所以|故|答案是|正确答案)",
)

_CLAUSE_START_MARKERS = (
    "because",
    "since",
    "therefore",
    "thus",
    "hence",
    "however",
    "while",
    "given that",
    "which means",
    "that means",
    "as seen",
    "looking at",
    "now",
    "next",
    "then",
    "finally",
    "option a",
    "option b",
    "option c",
    "option d",
    "because",
    "因为",
    "所以",
    "因此",
    "由于",
    "说明",
    "这意味着",
    "由此可见",
    "再看",
    "接着",
    "选项a",
    "选项b",
    "选项c",
    "选项d",
)


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


def _strip_leading_image_placeholder(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("<image>"):
        stripped = stripped[len("<image>") :].lstrip()
    return stripped


def _split_answer(text: str) -> str:
    idx = text.find(THINK_END)
    if idx < 0:
        return text.strip()
    return text[idx + len(THINK_END) :].strip()


def _extract_thinking(text: str) -> str:
    thinking, _ = extract_thinking_and_answer(text, max_chars=None, return_chunks=False)
    return thinking.strip()


def _normalize_span_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    raw_parts = re.split(r"(?<=[\.\!\?。！？])\s+|\n+", text)
    spans = [_normalize_span_text(part) for part in raw_parts if _normalize_span_text(part)]
    if spans:
        clauses: list[str] = []
        for span in spans:
            clauses.extend(_split_clauses(span))
        return clauses

    fallback = re.split(r"(?<=[,;:，；：])\s*", text)
    spans = [_normalize_span_text(part) for part in fallback if _normalize_span_text(part)]
    if spans:
        clauses: list[str] = []
        for span in spans:
            clauses.extend(_split_clauses(span))
        return clauses
    return [_normalize_span_text(text)]


def _split_on_clause_markers(text: str) -> list[str]:
    if not text:
        return []
    pattern = "|".join(re.escape(marker) for marker in _CLAUSE_START_MARKERS)
    if not pattern:
        return [text]
    marked = re.sub(
        rf"\s+(?=(?:{pattern})\b)",
        " ||| ",
        text,
        flags=re.IGNORECASE,
    )
    return [part.strip(" ,;:，；：") for part in marked.split("|||") if part.strip(" ,;:，；：")]


def _merge_short_clauses(parts: list[str]) -> list[str]:
    merged: list[str] = []
    for part in parts:
        normalized = _normalize_span_text(part)
        if not normalized:
            continue
        word_count = len(normalized.split())
        if merged and word_count <= 4:
            merged[-1] = f"{merged[-1]} {normalized}".strip()
        else:
            merged.append(normalized)
    return merged


def _split_clauses(sentence: str) -> list[str]:
    text = _normalize_span_text(sentence)
    if not text:
        return []

    comma_rich = len(re.findall(r"[,;:，；：]", text)) >= 2
    long_text = len(text) >= 160 or len(text.split()) >= 28
    has_marker = any(marker in text.lower() for marker in _CLAUSE_START_MARKERS if marker.isascii())
    has_marker = has_marker or any(marker in text for marker in _CLAUSE_START_MARKERS if not marker.isascii())

    if not comma_rich and not long_text and not has_marker:
        return [text]

    coarse_parts = re.split(r"(?<=[,;:，；：])\s*", text)
    split_parts: list[str] = []
    for coarse in coarse_parts:
        split_parts.extend(_split_on_clause_markers(coarse))
    merged = _merge_short_clauses(split_parts)
    return merged or [text]


def _match_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _classify_sentence_role(sentence: str) -> str:
    lowered = sentence.lower().strip()
    if not lowered:
        return "other"
    if _match_any(lowered, _INTRO_PATTERNS):
        return "intro"
    if _match_any(lowered, _CONCLUDE_PATTERNS):
        return "conclude"
    if _match_any(lowered, _VERIFY_PATTERNS):
        return "verify"
    if _match_any(lowered, _CALC_PATTERNS):
        return "calc"
    if _match_any(lowered, _OPTION_PATTERNS):
        return "option"
    if _match_any(lowered, _OBSERVE_PATTERNS):
        return "observe"
    if _match_any(lowered, _KNOWLEDGE_PATTERNS):
        return "knowledge"
    return "reason"


def _classify_span_role(span: str) -> str:
    return _classify_sentence_role(span)


def _merge_score(left: dict[str, Any], right: dict[str, Any]) -> tuple[int, int]:
    left_role = str(left["role"])
    right_role = str(right["role"])
    compatible_pairs = {
        ("intro", "knowledge"),
        ("intro", "observe"),
        ("observe", "knowledge"),
        ("knowledge", "observe"),
        ("observe", "option"),
        ("option", "reason"),
        ("option", "calc"),
        ("reason", "calc"),
        ("calc", "verify"),
        ("verify", "conclude"),
        ("reason", "conclude"),
    }
    if left_role == right_role:
        penalty = 0
    elif left_role == "other" or right_role == "other":
        penalty = 1
    elif (left_role, right_role) in compatible_pairs:
        penalty = 2
    else:
        penalty = 3
    char_cost = len(str(left["text"])) + len(str(right["text"]))
    return penalty, char_cost


def _merge_groups_to_limit(groups: list[dict[str, Any]], max_groups: int) -> list[dict[str, Any]]:
    merged = [dict(group) for group in groups]
    while len(merged) > max_groups:
        best_idx = 0
        best_score = _merge_score(merged[0], merged[1])
        for idx in range(1, len(merged) - 1):
            score = _merge_score(merged[idx], merged[idx + 1])
            if score < best_score:
                best_score = score
                best_idx = idx
        left = merged[best_idx]
        right = merged[best_idx + 1]
        merged[best_idx : best_idx + 2] = [
            {
                "role": left["role"] if best_score[0] == 0 else "mixed",
                "text": f"{left['text']} {right['text']}".strip(),
            }
        ]
    return merged


def _rebuild_pieces_from_compressed_spans(
    spans: list[str],
    compressed_spans: list[list[int]],
) -> tuple[list[str], list[int]]:
    if not spans:
        return [], []

    normalized_ranges = sorted(
        [
            [max(0, int(start)), min(len(spans) - 1, int(end))]
            for start, end in compressed_spans
            if int(end) >= int(start)
        ],
        key=lambda item: (item[0], item[1]),
    )

    pieces: list[str] = []
    anchor_indices: list[int] = []
    range_idx = 0
    span_idx = 0
    while span_idx < len(spans):
        if range_idx < len(normalized_ranges):
            start, end = normalized_ranges[range_idx]
            if span_idx == start:
                pieces.append(LATENT_TOKEN)
                span_idx = end + 1
                range_idx += 1
                continue
        pieces.append(spans[span_idx])
        anchor_indices.append(span_idx)
        span_idx += 1

    return pieces, anchor_indices


def _split_thinking_into_spans(thinking: str) -> list[str]:
    text = thinking.strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    groups: list[dict[str, Any]] = []
    for sentence in sentences:
        role = _classify_sentence_role(sentence)
        if groups and groups[-1]["role"] == role:
            groups[-1]["text"] = f"{groups[-1]['text']} {sentence}".strip()
        else:
            groups.append({"role": role, "text": sentence})

    groups = _merge_groups_to_limit(groups, MAX_GLOBAL_SPANS)
    spans = [_normalize_span_text(group["text"]) for group in groups if _normalize_span_text(group["text"])]
    return spans[:MAX_GLOBAL_SPANS]


def _should_use_leading_latent_opening(spans: list[str]) -> bool:
    if len(spans) < 2:
        return False
    opening = _normalize_span_text(spans[0])
    if not opening:
        return False
    return _match_any(opening.lower(), _LEADING_LATENT_OPEN_PATTERNS)


def _is_anchor_span(span: str) -> bool:
    lowered = span.lower()
    return any(hint in lowered for hint in _ANCHOR_HINTS)


def _choose_anchor_indices(spans: list[str]) -> list[int]:
    if not spans:
        return []
    roles = [_classify_span_role(span) for span in spans]

    anchors = {len(spans) - 1}

    for idx, span in enumerate(spans[1:-1], start=1):
        role = roles[idx]
        if role in {"conclude", "option", "verify"}:
            anchors.add(idx)
        elif role != "observe" and _is_anchor_span(span):
            anchors.add(idx)

    if len(spans) >= 3 and len(anchors) == 1:
        preferred = next(
            (idx for idx, role in enumerate(roles[:-1]) if role in {"intro", "reason", "option"}),
            0,
        )
        anchors.add(preferred)

    if not any(role in {"intro", "reason"} for idx, role in enumerate(roles) if idx in anchors):
        preferred_setup = next(
            (idx for idx, role in enumerate(roles[:-1]) if role in {"intro", "reason"}),
            0,
        )
        anchors.add(preferred_setup)

    if len(spans) >= 3 and len(anchors) == 2:
        middle_candidates = [
            idx
            for idx in range(1, len(spans) - 1)
            if idx not in anchors and roles[idx] in {"reason", "option", "intro"}
        ]
        if middle_candidates:
            anchors.add(middle_candidates[len(middle_candidates) // 2])

    if 0 not in anchors and roles[0] == "observe":
        visible_before_latent = [idx for idx in sorted(anchors) if idx < len(spans) - 1]
        if not visible_before_latent:
            preferred_prefix = next(
                (idx for idx, role in enumerate(roles[:-1]) if role in {"intro", "reason", "option"}),
                None,
            )
            if preferred_prefix is not None:
                anchors.add(preferred_prefix)
            else:
                anchors.add(0)
    return sorted(anchors)


def _build_compressed_target(thinking: str, answer_text: str) -> tuple[str, dict[str, Any]]:
    spans = _split_thinking_into_spans(thinking)
    if not spans:
        target = f"{THINK_START}{LATENT_TOKEN}{THINK_END}{answer_text}".strip()
        trace = {
            "spans": [],
            "anchors": [],
            "compressed_spans": [[0, 0]],
            "num_latent": 1,
            "num_spans": 0,
        }
        return target, trace

    anchor_indices = _choose_anchor_indices(spans)
    anchor_set = set(anchor_indices)
    pieces: list[str] = []
    compressed_spans: list[list[int]] = []
    pending_start: int | None = None
    min_steps = MIN_COMPRESSED_SPAN_STEPS
    if len(spans) <= 3:
        min_steps = MIN_SHORT_COMPRESSED_SPAN_STEPS

    for idx, span in enumerate(spans):
        if idx in anchor_set:
            if pending_start is not None:
                if idx - pending_start >= min_steps:
                    pieces.append(LATENT_TOKEN)
                    compressed_spans.append([pending_start, idx - 1])
                else:
                    pieces.extend(spans[pending_start:idx])
                pending_start = None
            pieces.append(span)
        elif pending_start is None:
            pending_start = idx

    if pending_start is not None:
        if len(spans) - pending_start >= min_steps:
            pieces.append(LATENT_TOKEN)
            compressed_spans.append([pending_start, len(spans) - 1])
        else:
            pieces.extend(spans[pending_start:])

    use_leading_latent = _should_use_leading_latent_opening(spans)
    if use_leading_latent and compressed_spans:
        compressed_spans[0][0] = 0
        pieces, anchor_indices = _rebuild_pieces_from_compressed_spans(spans, compressed_spans)

    if not use_leading_latent and compressed_spans and compressed_spans[0][0] == 0:
        first_end = compressed_spans[0][1]
        pieces = [spans[0]]
        revised_compressed: list[list[int]] = []
        if first_end >= 1:
            pieces.append(LATENT_TOKEN)
            revised_compressed.append([1, first_end])
        for start, end in compressed_spans[1:]:
            pieces.append(LATENT_TOKEN)
            revised_compressed.append([start, end])
        tail_start = first_end + 1
        for idx in range(tail_start, len(spans)):
            if idx not in anchor_set:
                continue
            pieces.append(spans[idx])
        compressed_spans = revised_compressed
        anchor_indices = sorted({0} | {idx for idx in anchor_indices if idx > first_end})

    if LATENT_TOKEN not in pieces:
        if len(spans) >= 4:
            middle_end = len(spans) - 2
            pieces = [spans[0], LATENT_TOKEN, spans[-1]]
            compressed_spans = [[1, middle_end]]
            anchor_indices = [0, len(spans) - 1]
        elif len(spans) == 3:
            pieces = [spans[0], LATENT_TOKEN, spans[2]]
            compressed_spans = [[1, 1]]
            anchor_indices = [0, 2]
        elif len(spans) == 2:
            pieces = [spans[0], LATENT_TOKEN]
            compressed_spans = [[1, 1]]
            anchor_indices = [0]
        elif len(spans) == 1:
            pieces = [LATENT_TOKEN, spans[0]]
            compressed_spans = [[0, 0]]
            anchor_indices = [0]
        else:
            pieces = list(spans)
            compressed_spans = []

    think_body = "\n".join(piece for piece in pieces if piece)
    target = f"{THINK_START}{think_body}{THINK_END}{answer_text}".strip()
    trace = {
        "spans": spans,
        "anchors": anchor_indices,
        "compressed_spans": compressed_spans,
        "num_latent": sum(1 for piece in pieces if piece == LATENT_TOKEN),
        "num_spans": len(spans),
        "max_global_spans": MAX_GLOBAL_SPANS,
        "min_compressed_span_steps": MIN_COMPRESSED_SPAN_STEPS,
        "min_short_compressed_span_steps": MIN_SHORT_COMPRESSED_SPAN_STEPS,
    }
    return target, trace


def _build_row(source_dataset: str, row_idx: int, row: dict[str, Any]) -> dict[str, Any]:
    assistant_target = _assistant_text(row).strip()
    answer_text = _split_answer(assistant_target)
    thinking_text = _extract_thinking(str(row.get("cot") or assistant_target))
    compressed_target, compressed_trace = _build_compressed_target(thinking_text, answer_text)
    spans = [str(span) for span in list(compressed_trace.get("spans") or []) if str(span).strip()]
    think_body = "\n".join(spans) if spans else thinking_text
    full_target = f"{THINK_START}{think_body}{THINK_END}{answer_text}".strip()

    return {
        "sample_id": f"{source_dataset}_{row_idx:07d}",
        "source_dataset": source_dataset,
        "task": row.get("task", source_dataset),
        "question_images": list(row.get("images") or []),
        "student_user_text": _strip_leading_image_placeholder(_user_text(row)),
        "assistant_target": assistant_target,
        "answer_text": answer_text,
        "teacher_solution_text": thinking_text,
        "full_target": full_target,
        "compressed_target": compressed_target,
        "compressed_trace": compressed_trace,
        "teacher_reference_target": compressed_target,
    }


def build_manifest(output_path: Path, dataset_name: str = DEFAULT_DATASET_NAME) -> tuple[int, dict[str, int]]:
    dataset_info = _load_dataset_info()
    src_path = _resolve_dataset_path(dataset_name, dataset_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_path.open("w", encoding="utf-8") as f_out:
        for row_idx, row in enumerate(_iter_jsonl(src_path)):
            built = _build_row(dataset_name, row_idx, row)
            f_out.write(json.dumps(built, ensure_ascii=False) + "\n")
            total += 1

    return total, {dataset_name: total}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified OPSD-span manifest from r1ov thinking data.")
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output OPSD-span manifest JSONL path",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help=f"Dataset name from Qwen/data/dataset_info.json (default: {DEFAULT_DATASET_NAME})",
    )
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    total, per_dataset = build_manifest(output_path, args.dataset)
    print(f"[opsd-span-dataset] wrote {total} rows -> {output_path}")
    print(f"[opsd-span-dataset] breakdown: {per_dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
