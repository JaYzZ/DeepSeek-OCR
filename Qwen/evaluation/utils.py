#!/usr/bin/env python3
"""Shared helpers for evaluation scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


_THINK_BLOCK_PATTERNS = (
    r"<think\b[^>]*>.*?</think>",
    r"<thinking\b[^>]*>.*?</thinking>",
    r"<\|thinking\|>.*?<\|/thinking\|>",
)

_THINK_OPEN_TO_EOF_PATTERNS = (
    r"<think\b[^>]*>.*$",
    r"<thinking\b[^>]*>.*$",
    r"<\|thinking\|>.*$",
)

_THINK_TAG_PATTERNS = (
    r"</?think\b[^>]*>",
    r"</?thinking\b[^>]*>",
    r"<\|/?thinking\|>",
)


def strip_thinking_tokens(prediction: Any) -> str:
    """Remove known thinking wrappers and their contents from evaluation text."""
    text = str(prediction)

    changed = True
    while changed:
        changed = False
        for pattern in _THINK_BLOCK_PATTERNS:
            updated = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
            if updated != text:
                text = updated
                changed = True

    for pattern in _THINK_OPEN_TO_EOF_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    for pattern in _THINK_TAG_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def normalize_chat_api_url(api_url: str) -> str:
    """Accept either a server root or a full chat-completions endpoint."""
    trimmed = api_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    return f"{trimmed}/v1/chat/completions"


def parse_cuda_visible_devices(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def infer_tensor_parallel_size(value: Optional[str], fallback: int = 1) -> int:
    visible = parse_cuda_visible_devices(value)
    return len(visible) if visible else fallback


def get_model_attention_heads(model_path: str) -> Optional[int]:
    config_path = Path(model_path) / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_json = json.load(f)
    except Exception:
        return None

    text_cfg = config_json.get("text_config")
    if isinstance(text_cfg, dict):
        heads = text_cfg.get("num_attention_heads")
        if isinstance(heads, int) and heads > 0:
            return heads

    heads = config_json.get("num_attention_heads")
    if isinstance(heads, int) and heads > 0:
        return heads
    return None


def compatible_tensor_parallel_candidates(max_requested_tp: int) -> list[int]:
    upper = max(1, max_requested_tp)
    return list(range(upper, 0, -1))


def capped_tensor_parallel_candidates(max_requested_tp: int) -> list[int]:
    upper = max(1, max_requested_tp)
    return [tp for tp in (4, 2, 1) if tp <= upper] or [1]


def pick_compatible_tensor_parallel_size(
    model_path: str,
    requested_tp: int,
    *,
    capped: bool = False,
) -> int:
    attn_heads = get_model_attention_heads(model_path)
    allowed_tps = (
        capped_tensor_parallel_candidates(requested_tp)
        if capped
        else compatible_tensor_parallel_candidates(requested_tp)
    )
    if not attn_heads:
        return allowed_tps[0]

    for tp in allowed_tps:
        if attn_heads % tp == 0:
            return tp
    return 1


def select_compatible_tensor_parallel_gpus(
    model_path: str,
    gpus: list[int],
    *,
    capped: bool = False,
) -> tuple[list[int], int]:
    if not gpus:
        return [], 1

    tensor_parallel_size = pick_compatible_tensor_parallel_size(
        model_path,
        len(gpus),
        capped=capped,
    )
    return list(gpus[:tensor_parallel_size]), tensor_parallel_size
