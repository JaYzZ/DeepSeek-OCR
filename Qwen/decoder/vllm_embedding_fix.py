#!/usr/bin/env python3
"""Runtime patch for the vLLM Qwen3-VL dict-embedding path."""

from typing import Any, Dict
import logging

import torch

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_vllm_embedding_fix():
    """Patch vLLM so dict-style precomputed embeddings are handled consistently."""
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        logger.debug("vLLM embedding fix already applied")
        return

    try:
        from vllm.model_executor.models import qwen3_vl

        original_process_image = qwen3_vl.Qwen3VLForConditionalGeneration._process_image_input

        def patched_process_image(self, image_input):
            grid_thw = image_input["image_grid_thw"]
            assert grid_thw.ndim == 2

            if image_input["type"] == "image_embeds":
                image_embeds = image_input["image_embeds"]
                image_embeds = image_embeds.to(dtype=self.visual.dtype)

                if image_embeds.device != self.visual.device:
                    image_embeds = image_embeds.to(device=self.visual.device, non_blocking=True)

                if not image_embeds.is_contiguous():
                    image_embeds = image_embeds.contiguous()

                expected_hidden_size = self.visual.merger.hidden_size * (
                    1 + len(self.visual.deepstack_visual_indexes)
                )
                if image_embeds.shape[-1] != expected_hidden_size:
                    logger.warning(
                        "Embedding dimension mismatch: got %s, expected %s. "
                        "This may cause incorrect output.",
                        image_embeds.shape[-1],
                        expected_hidden_size,
                    )
            else:
                return original_process_image(self, image_input)

            merge_size = self.visual.spatial_merge_size
            sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
            return image_embeds.split(sizes)

        qwen3_vl.Qwen3VLForConditionalGeneration._process_image_input = patched_process_image
        _PATCH_APPLIED = True
        logger.debug("Applied vLLM embedding fix")

    except Exception as exc:
        logger.warning("Failed to apply vLLM embedding fix: %s", exc)
        logger.warning("Precomputed embeddings may not work correctly")


def verify_embedding_format(
    embeddings: torch.Tensor,
    grid_thw: torch.Tensor,
    expected_hidden_size: int = 8192,
) -> bool:
    """Validate the format used for vLLM precomputed embeddings."""
    if embeddings.ndim != 2:
        logger.error("Embeddings must be 2D, got %sD", embeddings.ndim)
        return False

    if embeddings.shape[-1] != expected_hidden_size:
        logger.error(
            "Embedding hidden size mismatch: got %s, expected %s",
            embeddings.shape[-1],
            expected_hidden_size,
        )
        return False

    merge_size = 2
    expected_seq_len = grid_thw[0].prod().item() // (merge_size ** 2)
    if embeddings.shape[0] != expected_seq_len:
        logger.error(
            "Sequence length mismatch: got %s, expected %s from grid_thw %s",
            embeddings.shape[0],
            expected_seq_len,
            grid_thw[0].tolist(),
        )
        return False

    if embeddings.dtype not in [torch.float16, torch.bfloat16]:
        logger.warning("Unusual dtype: %s, expected float16 or bfloat16", embeddings.dtype)

    return True
