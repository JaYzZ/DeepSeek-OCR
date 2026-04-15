#!/usr/bin/env python3
"""Embedding item helpers for Qwen VL multimodal passthrough."""

from typing import Mapping

import torch
from vllm.multimodal.parse import EmbeddingItems


class QwenVLEmbeddingItems(EmbeddingItems):
    """Embedding items that preserve `image_grid_thw` passthrough metadata."""

    def __init__(
        self,
        embeddings: torch.Tensor | list[torch.Tensor],
        grid_thw: torch.Tensor,
        modality: str = "image",
    ):
        super().__init__(embeddings, modality)
        self.grid_thw = grid_thw

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {
            f"{self.modality}_embeds": self.data,
            f"{self.modality}_grid_thw": self.grid_thw,
        }
