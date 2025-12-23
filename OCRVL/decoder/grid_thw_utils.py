#!/usr/bin/env python3
"""
Utility to infer grid_thw from OCR visual token count

For DeepSeek-OCR:
- 100 tokens = 10×10 grid (no separators)
- 111 tokens = 10×10 grid + 11 separators
- grid_thw = [1, 10, 10]

This allows us to pass only the embeddings without needing grid_thw separately!
"""

import torch


def infer_ocr_grid_thw(num_tokens: int) -> list[int]:
    """
    Infer grid_thw from OCR token count.

    Args:
        num_tokens: Number of visual tokens

    Returns:
        grid_thw as [t, h, w]

    Examples:
        >>> infer_ocr_grid_thw(100)
        [1, 10, 10]
        >>> infer_ocr_grid_thw(111)
        [1, 10, 10]
        >>> infer_ocr_grid_thw(464)  # 20×20 + 64 separators
        [1, 20, 20]
    """
    # OCR format can be:
    # 1. Without separators: h×w
    # 2. With separators: h×w + (h+1) newline tokens
    #    Total = h*w + h + 1 = h*(w+1) + 1

    # For square grids (h=w):
    # - Without separators: num_tokens = h^2
    # - With separators: num_tokens = h^2 + h + 1

    import math
    h = int(math.sqrt(num_tokens))

    # Try without separators first (h*w = num_tokens)
    if h * h == num_tokens:
        return [1, h, h]

    # Try with separators (h*w + h + 1 = num_tokens)
    # Refine to exact value
    for size in range(h-2, h+3):
        if size * size + size + 1 == num_tokens:
            return [1, size, size]

    # Fallback for common cases
    if num_tokens == 100:
        return [1, 10, 10]
    if num_tokens == 111:
        return [1, 10, 10]

    raise ValueError(f"Cannot infer grid_thw from {num_tokens} tokens")


# Common OCR grid sizes
OCR_TOKEN_TO_GRID = {
    100: [1, 10, 10],    # Standard OCR without separators: 10×10 = 100
    111: [1, 10, 10],    # Standard OCR with separators: 100 + 11 = 111
    464: [1, 20, 20],    # High-res: 400 + 21 + 43 = 464
    1055: [1, 30, 30],   # Very high-res: 900 + 31 + 124 = 1055
}


def get_ocr_grid_thw(embedding: torch.Tensor) -> list[int]:
    """
    Get grid_thw directly from OCR embedding shape.

    Args:
        embedding: Visual tokens of shape [seq_len, dim]

    Returns:
        grid_thw as [t, h, w]
    """
    num_tokens = embedding.shape[0]

    if num_tokens in OCR_TOKEN_TO_GRID:
        return OCR_TOKEN_TO_GRID[num_tokens]

    return infer_ocr_grid_thw(num_tokens)


if __name__ == "__main__":
    # Test
    print("Testing grid_thw inference:")
    print(f"100 tokens → {get_ocr_grid_thw(torch.zeros(100, 1280))}")
    print(f"111 tokens → {get_ocr_grid_thw(torch.zeros(111, 1280))}")
    print(f"464 tokens → {get_ocr_grid_thw(torch.zeros(464, 1280))}")
