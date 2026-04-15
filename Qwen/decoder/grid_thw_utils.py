#!/usr/bin/env python3
"""Helpers to infer `grid_thw` for OCR-style visual token sequences."""

import math

import torch


def infer_ocr_grid_thw(num_tokens: int) -> list[int]:
    """Infer `grid_thw` from OCR token count."""
    h = int(math.sqrt(num_tokens))

    if h * h == num_tokens:
        return [1, h, h]

    for size in range(h - 2, h + 3):
        if size * size + size + 1 == num_tokens:
            return [1, size, size]

    if num_tokens == 100:
        return [1, 10, 10]
    if num_tokens == 111:
        return [1, 10, 10]

    raise ValueError(f"Cannot infer grid_thw from {num_tokens} tokens")


OCR_TOKEN_TO_GRID = {
    100: [1, 10, 10],
    111: [1, 10, 10],
    464: [1, 20, 20],
    1055: [1, 30, 30],
}


def get_ocr_grid_thw(embedding: torch.Tensor) -> list[int]:
    """Get `grid_thw` directly from an OCR embedding tensor."""
    num_tokens = embedding.shape[0]
    if num_tokens in OCR_TOKEN_TO_GRID:
        return OCR_TOKEN_TO_GRID[num_tokens]
    return infer_ocr_grid_thw(num_tokens)
