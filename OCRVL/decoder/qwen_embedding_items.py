#!/usr/bin/env python3
"""
Custom Embedding Items for Qwen VL models that preserve grid_thw

This class extends vLLM's EmbeddingItems to pass both image_embeds and image_grid_thw
through the passthrough pipeline, bypassing the HF processor.
"""

from typing import Mapping
import torch
from vllm.multimodal.parse import EmbeddingItems


class QwenVLEmbeddingItems(EmbeddingItems):
    """
    Embedding items for Qwen VL that include grid_thw metadata.

    Unlike standard EmbeddingItems which only pass embeddings, this class
    also passes image_grid_thw through the passthrough pipeline.
    """

    def __init__(
        self,
        embeddings: torch.Tensor | list[torch.Tensor],
        grid_thw: torch.Tensor,
        modality: str = "image",
    ):
        """
        Args:
            embeddings: Visual embeddings
                - Single: [seq_len, dim]
                - Batch: list of [seq_len, dim]
            grid_thw: MRoPE grid for all embeddings
                - Shape: [num_images, 3] for batch
                - Each row: [temporal, height, width]
            modality: "image" or "video"
        """
        super().__init__(embeddings, modality)
        self.grid_thw = grid_thw

    def get_passthrough_data(self) -> Mapping[str, object]:
        """
        Pass both embeddings and grid_thw to the model.

        Returns:
            Dict with image_embeds and image_grid_thw keys
        """
        return {
            f"{self.modality}_embeds": self.data,
            f"{self.modality}_grid_thw": self.grid_thw,
        }

