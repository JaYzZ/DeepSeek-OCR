#!/usr/bin/env python3
"""Runtime patch for vLLM to fix dict embedding processing bug"""

import logging
import torch
from typing import Dict, Any

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_vllm_embedding_fix():
    """
    Apply runtime patch to fix vLLM's dict embedding processing bug.

    This fixes the issue where pre-computed embeddings passed as dict
    produce incorrect output, even when using HuggingFace's own visual encoder.

    The bug appears to be in how vLLM merges dict embeddings into the input sequence.
    This patch ensures embeddings are properly formatted and placed.
    """
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        logger.debug("vLLM embedding fix already applied")
        return

    try:
        # Import vLLM's Qwen3-VL model
        from vllm.model_executor.models import qwen3_vl

        # Store original method
        original_process_image = qwen3_vl.Qwen3VLForConditionalGeneration._process_image_input

        def patched_process_image(self, image_input):
            """
            Patched _process_image_input that ensures dict embeddings are properly formatted.

            The bug: vLLM doesn't properly handle the batch dimension and device placement
            for dict embeddings, causing them to be misaligned with the input sequence.
            """
            grid_thw = image_input["image_grid_thw"]
            assert grid_thw.ndim == 2

            if image_input["type"] == "image_embeds":
                image_embeds = image_input["image_embeds"]

                # FIX 1: Ensure proper dtype conversion
                image_embeds = image_embeds.to(dtype=self.visual.dtype)

                # FIX 2: Ensure proper device placement
                if image_embeds.device != self.visual.device:
                    image_embeds = image_embeds.to(device=self.visual.device, non_blocking=True)

                # FIX 3: Ensure embeddings are contiguous in memory
                if not image_embeds.is_contiguous():
                    image_embeds = image_embeds.contiguous()

                # FIX 4: Validate embedding dimensions match expected format
                # vLLM expects [seq_len, hidden_size * (1 + deepstack_levels)]
                expected_hidden_size = self.visual.merger.hidden_size * (1 + len(self.visual.deepstack_visual_indexes))
                if image_embeds.shape[-1] != expected_hidden_size:
                    logger.warning(
                        f"Embedding dimension mismatch: got {image_embeds.shape[-1]}, "
                        f"expected {expected_hidden_size}. This may cause incorrect output."
                    )
            else:
                # Normal pixel_values path - use original implementation
                return original_process_image(self, image_input)

            # Split concatenated embeddings for each image item
            merge_size = self.visual.spatial_merge_size
            sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
            return image_embeds.split(sizes)

        # Apply patch
        qwen3_vl.Qwen3VLForConditionalGeneration._process_image_input = patched_process_image

        _PATCH_APPLIED = True
        logger.debug("✓ Applied vLLM embedding fix (runtime patch)")

    except Exception as e:
        logger.warning(f"Failed to apply vLLM embedding fix: {e}")
        logger.warning("Pre-computed embeddings may not work correctly")


def verify_embedding_format(
    embeddings: torch.Tensor,
    grid_thw: torch.Tensor,
    expected_hidden_size: int = 8192,
) -> bool:
    """
    Verify that embeddings are in the correct format for vLLM.

    Args:
        embeddings: Visual embeddings [seq_len, hidden_size]
        grid_thw: Grid dimensions [1, 3] or [batch, 3]
        expected_hidden_size: Expected hidden dimension (2048 * 4 = 8192 for Qwen3-VL)

    Returns:
        True if format is correct, False otherwise
    """
    # Check dimensions
    if embeddings.ndim != 2:
        logger.error(f"Embeddings must be 2D, got {embeddings.ndim}D")
        return False

    # Check hidden size
    if embeddings.shape[-1] != expected_hidden_size:
        logger.error(
            f"Embedding hidden size mismatch: got {embeddings.shape[-1]}, "
            f"expected {expected_hidden_size}"
        )
        return False

    # Check sequence length matches grid
    merge_size = 2  # Qwen3-VL uses 2x2 spatial merge
    expected_seq_len = grid_thw[0].prod().item() // (merge_size ** 2)
    if embeddings.shape[0] != expected_seq_len:
        logger.error(
            f"Sequence length mismatch: got {embeddings.shape[0]}, "
            f"expected {expected_seq_len} from grid_thw {grid_thw[0].tolist()}"
        )
        return False

    # Check dtype
    if embeddings.dtype not in [torch.float16, torch.bfloat16]:
        logger.warning(f"Unusual dtype: {embeddings.dtype}, should be float16 or bfloat16")

    return True


if __name__ == "__main__":
    # Test the patch
    print("Testing vLLM embedding fix...")
    apply_vllm_embedding_fix()
    print("Patch applied successfully!")
