#!/usr/bin/env python3
"""Shared data utilities for Qwen3VL thinking datasets and replay paths."""

from __future__ import annotations

import re
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from Renderer.skia_renderer import (
    SkiaRenderer,
    measure_finalized_text_canvas,
    prepare_text_for_rendering,
    snap_canvas_to_grid,
)


def compress_newlines(text: str) -> str:
    """Compress multiple consecutive newlines to a single newline."""
    return re.sub(r"\n{2,}", "\n", (text or ""))


def format_cot_subsequences(thinking_chunks: Optional[list[str]]) -> str:
    """
    Format CoT chunks with <think_sep> separators.

    Output shape:
      <think>[chunk1]<think_sep>[chunk2]...</think>
    """
    chunks = [c.strip() for c in (thinking_chunks or []) if c and c.strip()]
    if not chunks:
        return ""
    return f"<think>{'<think_sep>'.join(chunks)}</think>"


def chunk_thinking_text(thinking: str, max_chars: int) -> list[str]:
    """
    Chunk long thinking text into balanced segments.

    Rules:
    - Prefer sentence/newline boundaries
    - Keep chunk sizes roughly balanced
    - Ensure each chunk does not exceed max_chars
    """
    thinking = compress_newlines(thinking)
    if len(thinking) <= max_chars:
        return [thinking] if thinking else []

    num_chunks = (len(thinking) + max_chars - 1) // max_chars
    fragments = re.split(r"(\. \n|\. |\n)", thinking)

    sentences = []
    for i in range(0, len(fragments) - 1, 2):
        if i + 1 < len(fragments):
            sentences.append(fragments[i] + fragments[i + 1])
        else:
            sentences.append(fragments[i])
    if fragments and fragments[-1]:
        sentences.append(fragments[-1])
    sentences = [s for s in sentences if s.strip()]

    target_size = len(thinking) / max(num_chunks, 1)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        test_chunk = current_chunk + sentence
        if len(test_chunk) <= target_size * 1.2 or not current_chunk:
            current_chunk = test_chunk
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)

    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            split_point = chunk.rfind(". ", 0, max_chars)
            if split_point == -1:
                split_point = max_chars
            final_chunks.append(chunk[: split_point + 1].strip())
            chunk = chunk[split_point + 1 :].strip()
        if chunk:
            final_chunks.append(chunk)

    return [c for c in final_chunks if c]


@dataclass(frozen=True)
class SkiaRenderConfig:
    min_size: int = 32
    max_size: int = 4096
    vit_divisor: int = 32
    padding: int = 12
    thinking_padding: int = 8
    short_line_wrap_threshold: int = 20
    short_line_min_lines: int = 8


@dataclass(frozen=True)
class ThinkingReplayRenderConfig:
    strip_leading_indentation: bool = False
    collapse_multi_blank_lines: bool = False
    normalize_bullet_prefixes: bool = False
    collapse_all_whitespace: bool = False
    compact_layout: bool = False


_LIST_PREFIX_RE = re.compile(r"^(\s*)([-*+•●▪◦‣·]+)\s+")
_NUMBERED_PREFIX_RE = re.compile(r"^(\s*)(\d+)\s*[\.\)]\s+")
_ALPHA_PREFIX_RE = re.compile(r"^(\s*)([A-Za-z])\s*[\.\)]\s+")


def _atomic_save_png(image: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=output_path.suffix,
        prefix=f"{output_path.stem}.",
        dir=output_path.parent,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        Image.fromarray(image).save(tmp_path, format="PNG")
        if not tmp_path.exists():
            raise FileNotFoundError(f"Temporary render output missing after save: {tmp_path}")
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class AdaptiveSkiaRenderer:
    """Skia renderer with adaptive preprocessing matching the builders."""

    def __init__(
        self,
        min_font_size: float = 10.0,
        max_font_size: float = 10.0,
        config: Optional[SkiaRenderConfig] = None,
        compact_thinking_layout: bool = False,
    ):
        self._config = config or SkiaRenderConfig()
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.compact_thinking_layout = compact_thinking_layout
        self._renderer_cache: dict[tuple[int, int, int, bool, bool], SkiaRenderer] = {}

    def _get_renderer(self, width: int, height: int, padding: int, preserve_newlines: bool) -> SkiaRenderer:
        key = (width, height, padding, preserve_newlines, False)
        renderer = self._renderer_cache.get(key)
        if renderer is None:
            renderer = SkiaRenderer(
                width=width,
                height=height,
                padding=padding,
                min_font_size=self.min_font_size,
                max_font_size=self.max_font_size,
                preserve_newlines=preserve_newlines,
            )
            self._renderer_cache[key] = renderer
        return renderer

    def _measure_canvas(self, prepared_text: str, padding: int) -> tuple[int, int, int, int]:
        raw_width, raw_height = measure_finalized_text_canvas(
            prepared_text,
            padding=padding,
            font_size=self.min_font_size,
            min_size=self._config.min_size,
        )
        snapped_width, snapped_height = snap_canvas_to_grid(
            raw_width,
            raw_height,
            divisor=self._config.vit_divisor,
            min_size=self._config.min_size,
            max_size=self._config.max_size,
        )
        return raw_width, raw_height, snapped_width, snapped_height

    def _prepare_compact_thinking_render(self, text: str, padding: int) -> tuple[str, dict[str, Any]]:
        compact_text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not compact_text:
            return "", {
                "layout_type": "failed",
                "preserve_newlines": True,
                "num_lines": 0,
                "max_line_chars": 0,
            }

        words = compact_text.split()
        if not words:
            return "", {
                "layout_type": "failed",
                "preserve_newlines": True,
                "num_lines": 0,
                "max_line_chars": 0,
            }

        longest_word_len = max(len(word) for word in words)
        max_char_width = min(1000, max(longest_word_len, len(compact_text)))
        min_char_width = max(8, longest_word_len)

        wrap_measure_cache: dict[int, tuple[list[str], int, int]] = {}
        score_cache: dict[int, tuple[float, float, int]] = {}

        def wrap_and_measure(char_width: int) -> tuple[list[str], int, int]:
            cached = wrap_measure_cache.get(char_width)
            if cached is not None:
                return cached

            wrapped = textwrap.wrap(
                compact_text,
                width=max(1, char_width),
                break_long_words=False,
                break_on_hyphens=False,
            )
            if not wrapped:
                wrapped = [compact_text]

            prepared_text = "\n".join(wrapped)
            _, _, snapped_width, snapped_height = self._measure_canvas(prepared_text, padding)
            result = (wrapped, snapped_width, snapped_height)
            wrap_measure_cache[char_width] = result
            return result

        def get_score(char_width: int) -> tuple[float, float, int]:
            cached = score_cache.get(char_width)
            if cached is not None:
                return cached
            _, snapped_width, snapped_height = wrap_and_measure(char_width)
            if snapped_width > self._config.max_size or snapped_height > self._config.max_size:
                result = (float("inf"), float("inf"), char_width)
            else:
                longer = max(snapped_width, snapped_height)
                shorter = max(1, min(snapped_width, snapped_height))
                result = (snapped_width * snapped_height, longer / shorter, char_width)
            score_cache[char_width] = result
            return result

        lo, hi = min_char_width, max_char_width
        min_valid = None
        while lo <= hi:
            mid = (lo + hi) // 2
            _, snapped_width, snapped_height = wrap_and_measure(mid)
            if snapped_width <= self._config.max_size and snapped_height <= self._config.max_size:
                min_valid = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if min_valid is None:
            best_wrapped, best_width, best_height = wrap_and_measure(max_char_width)
        else:
            best_wrapped = None
            best_width = best_height = 0
            best_area = float("inf")
            left, right = min_valid, max_char_width
            while right - left > 3:
                mid = (left + right) // 2
                ranked = sorted(
                    [
                        (left, get_score(left)),
                        (mid, get_score(mid)),
                        (right, get_score(right)),
                    ],
                    key=lambda item: item[1],
                )
                chosen_widths = {ranked[0][0], ranked[1][0]}
                if chosen_widths == {left, mid}:
                    right = mid
                elif chosen_widths == {mid, right}:
                    left = mid
                else:
                    left_score = (ranked[0][1][1], ranked[0][1][0], left)
                    right_score = (ranked[-1][1][1], ranked[-1][1][0], right)
                    if left_score <= right_score:
                        right = mid
                    else:
                        left = mid

            for char_width in range(left, right + 1):
                wrapped, snapped_width, snapped_height = wrap_and_measure(char_width)
                if snapped_width <= self._config.max_size and snapped_height <= self._config.max_size:
                    area = snapped_width * snapped_height
                    if area < best_area:
                        best_area = area
                        best_wrapped = wrapped
                        best_width = snapped_width
                        best_height = snapped_height

            if best_wrapped is None:
                best_wrapped, best_width, best_height = wrap_and_measure(min_valid)

        prepared_text = "\n".join(best_wrapped)
        return prepared_text, {
            "layout_type": "compact_flow",
            "preserve_newlines": True,
            "num_lines": len(best_wrapped),
            "max_line_chars": max((len(line) for line in best_wrapped), default=0),
            "snapped_width": best_width,
            "snapped_height": best_height,
        }

    def _prepare_render(self, text: str, thinking_mode: bool) -> tuple[str, dict[str, Any], int]:
        padding = self._config.thinking_padding if thinking_mode else self._config.padding
        if thinking_mode and self.compact_thinking_layout:
            prepared_text, layout_info = self._prepare_compact_thinking_render(text, padding)
            return prepared_text, layout_info, padding
        prepared_text, layout_info = prepare_text_for_rendering(
            text,
            short_line_threshold=self._config.short_line_wrap_threshold,
            min_lines_for_reflow=self._config.short_line_min_lines,
            max_canvas_size=self._config.max_size,
            measurement_padding=padding,
            measurement_font_size=self.min_font_size,
            min_canvas_size=self._config.min_size,
            measurement_divisor=self._config.vit_divisor,
        )
        return prepared_text, layout_info, padding

    def measure_text(self, text: str, thinking_mode: bool) -> tuple[int, int, int, int]:
        prepared_text, _, padding = self._prepare_render(text, thinking_mode)
        return self._measure_canvas(prepared_text, padding)

    def _render_one(self, text: str, thinking_mode: bool):
        prepared_text, layout_info, padding = self._prepare_render(text, thinking_mode)
        if not prepared_text or layout_info.get("layout_type") == "failed":
            return None
        _, _, width, height = self._measure_canvas(prepared_text, padding)
        return self._get_renderer(width, height, padding, layout_info["preserve_newlines"]).render_batch(
            [prepared_text]
        )[0]

    def render_batch(self, texts: list[str], thinking_mode: bool = False) -> list[Any]:
        prepared_specs = []
        for text in texts:
            prepared_text, layout_info, padding = self._prepare_render(text, thinking_mode)
            prepared_specs.append((prepared_text, padding, layout_info["preserve_newlines"]))

        grouped: dict[tuple[int, int, int, bool], list[tuple[int, str]]] = {}
        for idx, (prepared_text, padding, preserve_newlines) in enumerate(prepared_specs):
            _, _, width, height = self._measure_canvas(prepared_text, padding)
            grouped.setdefault((width, height, padding, preserve_newlines), []).append((idx, prepared_text))

        results: list[Any] = [None] * len(texts)
        for (width, height, padding, preserve_newlines), items in grouped.items():
            renderer = self._get_renderer(width, height, padding, preserve_newlines)
            images = renderer.render_batch([text for _, text in items])
            for (idx, _), image in zip(items, images):
                results[idx] = image
        return results

    def render(self, text: str, output_path: str, thinking_mode: bool = False) -> bool:
        if not text or not text.strip():
            return False
        image = self._render_one(text, thinking_mode)
        if image is None:
            return False
        _atomic_save_png(image, Path(output_path))
        return True

    def shutdown(self) -> None:
        for renderer in self._renderer_cache.values():
            renderer.shutdown()
        self._renderer_cache.clear()


def load_cached_latent_seq_len(
    cache_path: str,
    seq_len_cache: dict[str, int],
    *,
    min_tensor_ndim: int = 1,
    fallback_seq_len: int = 0,
) -> int:
    cached = seq_len_cache.get(cache_path)
    if cached is not None:
        return cached

    payload = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(payload, dict):
        tensor = payload.get("latent")
        if tensor is None:
            tensor = payload.get("l_features")
    else:
        tensor = payload

    if isinstance(tensor, torch.Tensor) and tensor.ndim >= min_tensor_ndim:
        seq_len = int(tensor.shape[0])
    else:
        seq_len = int(fallback_seq_len)
    seq_len_cache[cache_path] = seq_len
    return seq_len


def build_cot_chunk_token_ids(
    tokenizer,
    thinking_chunks: list[str],
    *,
    normalize_chunks: bool = True,
    empty_chunk_fallback_text: Optional[str] = None,
) -> list[list[int]]:
    chunk_token_ids: list[list[int]] = []
    for chunk in thinking_chunks or []:
        text = str(chunk or "")
        if normalize_chunks:
            text = compress_newlines(text).strip()

        if not text:
            if empty_chunk_fallback_text is None:
                continue
            text = empty_chunk_fallback_text

        chunk_token_ids.append(tokenizer.encode(text, add_special_tokens=False))
    return chunk_token_ids


def extract_thinking_and_answer(
    content: str,
    max_chars: Optional[int] = 4800,
    return_chunks: bool = True,
):
    open_tag = "<think>"
    close_tag = "</think>"

    open_pos = content.find(open_tag)
    if open_pos < 0:
        return "", content.strip()

    close_pos = content.find(close_tag, open_pos + len(open_tag))
    if close_pos < 0:
        return "", content.strip()

    thinking_start = open_pos + len(open_tag)
    thinking_raw = content[thinking_start:close_pos].strip()

    answer_start = close_pos + len(close_tag)
    answer = content[answer_start:].strip()
    answer = re.sub(r"<\/?image>", "", answer).strip()

    if max_chars is not None and len(thinking_raw) > max_chars:
        thinking_chunks = chunk_thinking_text(thinking_raw, max_chars)
        if return_chunks:
            return thinking_chunks, answer
        return thinking_chunks[0], answer

    thinking = compress_newlines(thinking_raw)
    if return_chunks:
        return [thinking], answer
    return thinking, answer


def _normalize_bullet_prefix(line: str) -> str:
    line = _LIST_PREFIX_RE.sub("- ", line)
    line = _NUMBERED_PREFIX_RE.sub(r"\2. ", line)
    line = _ALPHA_PREFIX_RE.sub(r"\2. ", line)
    return line


def normalize_thinking_text_for_rendering(
    text: str,
    config: Optional[ThinkingReplayRenderConfig] = None,
) -> str:
    config = config or ThinkingReplayRenderConfig()
    normalized = str(text or "")
    if not normalized:
        return ""

    if config.strip_leading_indentation:
        normalized = textwrap.dedent(normalized)

    lines = normalized.splitlines()
    if config.strip_leading_indentation:
        lines = [line.lstrip() for line in lines]
    if config.normalize_bullet_prefixes:
        lines = [_normalize_bullet_prefix(line) for line in lines]

    if config.collapse_multi_blank_lines:
        collapsed_lines: list[str] = []
        prev_blank = False
        for line in lines:
            stripped = line.rstrip()
            is_blank = not stripped
            if is_blank and prev_blank:
                continue
            collapsed_lines.append("" if is_blank else stripped)
            prev_blank = is_blank
        lines = collapsed_lines
    else:
        lines = [line.rstrip() for line in lines]

    normalized = "\n".join(lines).strip()
    if config.collapse_all_whitespace or config.compact_layout:
        normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _split_text_for_rendering(text: str, *, preserve_line_boundaries: bool = True) -> tuple[str, str]:
    text = str(text or "").strip()
    if not text:
        return "", ""

    lines = text.splitlines()
    if preserve_line_boundaries and len(lines) >= 4:
        midpoint = len(lines) // 2
        candidates: list[tuple[int, int]] = []
        for idx in range(1, len(lines)):
            stripped = lines[idx].strip()
            prev_stripped = lines[idx - 1].strip()
            boundary_score = 0
            if not prev_stripped or not stripped:
                boundary_score -= 4
            if re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+|[A-Za-z][\.\)]\s+)", stripped):
                boundary_score -= 2
            if re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+|[A-Za-z][\.\)]\s+)", prev_stripped):
                boundary_score -= 1
            candidates.append((abs(idx - midpoint) + boundary_score, idx))
        if candidates:
            _, split_idx = min(candidates)
            left = "\n".join(lines[:split_idx]).strip()
            right = "\n".join(lines[split_idx:]).strip()
            if left and right:
                return left, right

    sentence_split = re.split(r"(?<=[.!?])\s+", text)
    if len(sentence_split) >= 2:
        midpoint = len(sentence_split) // 2
        left = " ".join(sentence_split[:midpoint]).strip()
        right = " ".join(sentence_split[midpoint:]).strip()
        if left and right:
            return left, right

    words = text.split()
    if len(words) >= 2:
        midpoint = len(words) // 2
        left = " ".join(words[:midpoint]).strip()
        right = " ".join(words[midpoint:]).strip()
        if left and right:
            return left, right

    midpoint = len(text) // 2
    return text[:midpoint].strip(), text[midpoint:].strip()


def _ensure_thinking_chunks_fit_renderer(
    renderer: AdaptiveSkiaRenderer,
    thinking_chunks: list[str],
    *,
    allow_unsplittable: bool = True,
    preserve_line_boundaries: bool = True,
) -> list[str]:
    queue_chunks = [str(chunk or "").strip() for chunk in thinking_chunks if str(chunk or "").strip()]
    fitted_chunks: list[str] = []

    while queue_chunks:
        chunk = queue_chunks.pop(0)
        raw_width, raw_height, _, _ = renderer.measure_text(chunk, thinking_mode=True)
        if raw_width <= renderer._config.max_size and raw_height <= renderer._config.max_size:
            fitted_chunks.append(chunk)
            continue

        left, right = _split_text_for_rendering(chunk, preserve_line_boundaries=preserve_line_boundaries)
        if not left or not right or left == chunk or right == chunk:
            if allow_unsplittable:
                fitted_chunks.append(chunk)
                continue
            raise ValueError(f"Failed to split chunk for renderer safety: {chunk[:200]!r}")
        queue_chunks = [left, right, *queue_chunks]

    return fitted_chunks
