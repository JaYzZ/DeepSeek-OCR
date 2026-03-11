#!/usr/bin/env python3
"""Shared utility helpers for dataset builder scripts."""

import re
from typing import List, Optional


def compress_newlines(text: str) -> str:
    """Compress multiple consecutive newlines to a single newline."""
    return re.sub(r"\n{2,}", "\n", (text or ""))


def format_cot_subsequences(thinking_chunks: Optional[List[str]]) -> str:
    """
    Format CoT chunks with <think_sep> separators.

    Output shape:
      <think>[chunk1]<think_sep>[chunk2]...</think>
    """
    chunks = [c.strip() for c in (thinking_chunks or []) if c and c.strip()]
    if not chunks:
        return ""
    return f"<think>{'<think_sep>'.join(chunks)}</think>"


def chunk_thinking_text(thinking: str, max_chars: int) -> List[str]:
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
