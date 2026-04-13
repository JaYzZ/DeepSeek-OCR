#!/usr/bin/env python3
"""Rule-based reward function for DeepVision-103K GSPO runs."""

from __future__ import annotations

import ast
import os
import re
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from typing import Any

import sympy
from latex2sympy2 import latex2sympy

# Number of workers for parallel reward computation
_REWARD_WORKERS = int(os.environ.get("REWARD_COMPUTATION_WORKERS", "16"))


BOXED_RE = re.compile(r"\\boxed\s*{")
FINAL_ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
OPTION_BLOCK_RE = re.compile(r"options?\s*:\s*(\[[^\]]+\])", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_boxed_spans(text: str) -> list[str]:
    matches: list[str] = []
    for match in BOXED_RE.finditer(text):
        start = match.end()
        depth = 1
        idx = start
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            matches.append(text[start : idx - 1].strip())
    return matches


def _strip_reasoning_prefix(text: str) -> str:
    stripped = THINK_RE.sub(" ", text)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    return stripped.strip()


def _extract_candidate_answers(solution_str: str) -> list[str]:
    text = _strip_reasoning_prefix(solution_str)
    candidates = [item for item in _extract_boxed_spans(text) if item]

    final_match = FINAL_ANSWER_RE.search(text)
    if final_match:
        candidates.append(final_match.group(1).strip())

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidates.append(lines[-1])

    raw = text.strip()
    if raw:
        candidates.append(raw)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\$+|\$+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?\n\t")


def _normalize_choice(text: str) -> str:
    text = _normalize_text(text)
    match = re.fullmatch(r"(?:option\s*)?([A-D])", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text


@lru_cache(maxsize=4096)
def _load_option_map(question: str) -> dict[str, str]:
    match = OPTION_BLOCK_RE.search(question)
    if not match:
        return {}
    try:
        options = ast.literal_eval(match.group(1))
    except Exception:
        return {}
    if not isinstance(options, list):
        return {}

    option_map: dict[str, str] = {}
    for idx, option in enumerate(options):
        letter = chr(ord("A") + idx)
        option_map[_normalize_text(str(option)).lower()] = letter
    return option_map


@lru_cache(maxsize=16384)
def _latex_to_sympy(expr: str) -> sympy.Expr | None:
    try:
        return latex2sympy(expr)
    except Exception:
        return None


@lru_cache(maxsize=16384)
def _text_to_sympy(expr: str) -> sympy.Expr | None:
    expr = _normalize_text(expr)
    expr = expr.replace("^", "**")
    expr = expr.replace("×", "*").replace("·", "*")
    expr = expr.replace("−", "-").replace("–", "-")
    expr = expr.replace("%", "/100")
    expr = re.sub(r"\\left|\\right", "", expr)
    expr = re.sub(r"\\,", "", expr)
    expr = re.sub(r"\\boxed\s*{(.+)}", r"\1", expr)

    parsed = _latex_to_sympy(expr)
    if parsed is not None:
        return parsed

    # `sympify()` on raw LaTeX-like strings can emit Python 3.12 SyntaxWarnings
    # (for example `\Z`, `\(`) and is never useful if latex parsing already failed.
    if "\\" in expr:
        return None

    try:
        return sympy.sympify(expr)
    except Exception:
        return None


def _answers_equivalent(pred: str, gold: str) -> bool:
    pred_norm = _normalize_choice(pred)
    gold_norm = _normalize_choice(gold)
    if pred_norm == gold_norm:
        return True

    pred_expr = _text_to_sympy(pred_norm)
    gold_expr = _text_to_sympy(gold_norm)
    if pred_expr is None or gold_expr is None:
        return False

    try:
        return bool(sympy.simplify(pred_expr - gold_expr) == 0)
    except Exception:
        return False


def _build_gold_candidates(ground_truth: str, extra_info: dict[str, Any]) -> list[str]:
    candidates = [ground_truth]
    reward_model = extra_info.get("reward_model") or {}
    for answer in reward_model.get("equivalent_answers") or []:
        text = str(answer).strip()
        if text and text != "__EMPTY__":
            candidates.append(text)

    question = str(extra_info.get("question") or "")
    option_map = _load_option_map(question)
    for answer in list(candidates):
        key = _normalize_text(answer).lower()
        if key in option_map:
            candidates.append(option_map[key])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _compute_single_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    del data_source
    extra_info = extra_info or {}

    pred_candidates = _extract_candidate_answers(solution_str)
    gold_candidates = _build_gold_candidates(ground_truth, extra_info)

    for pred in pred_candidates:
        for gold in gold_candidates:
            if _answers_equivalent(pred, gold):
                return 1.0
    return 0.0


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> float:
    return _compute_single_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]] | None = None,
) -> list[float]:
    """Compute rewards for a batch of DeepVision samples.

    Uses thread pool for parallel symbolic equivalence checking.
    SymPy operations are CPU-intensive and release the GIL, so threads provide
    true parallelism.
    """
    if extra_infos is None:
        extra_infos = [{} for _ in solution_strs]

    batch_size = len(solution_strs)
    if len(data_sources) != batch_size or len(ground_truths) != batch_size or len(extra_infos) != batch_size:
        raise ValueError(
            "Batch reward inputs must have matching lengths: "
            f"{len(data_sources)=}, {len(solution_strs)=}, {len(ground_truths)=}, {len(extra_infos)=}"
        )

    # Use thread pool for parallel computation (SymPy releases GIL)
    # Small batches (< 32) use sequential to avoid overhead
    if batch_size < 32:
        return [
            _compute_single_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            for data_source, solution_str, ground_truth, extra_info in zip(
                data_sources,
                solution_strs,
                ground_truths,
                extra_infos,
                strict=False,
            )
        ]

    # Parallel computation for larger batches
    with ThreadPool(processes=min(_REWARD_WORKERS, batch_size)) as pool:
        results = pool.starmap(
            _compute_single_score,
            [
                (data_source, solution_str, ground_truth, extra_info)
                for data_source, solution_str, ground_truth, extra_info in zip(
                    data_sources,
                    solution_strs,
                    ground_truths,
                    extra_infos,
                    strict=False,
                )
            ],
        )
    return results
