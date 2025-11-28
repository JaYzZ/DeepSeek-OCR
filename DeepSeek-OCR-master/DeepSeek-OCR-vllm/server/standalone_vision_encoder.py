"""
Standalone Vision Encoder for DeepSeek OCR (V1 Compatible)

This module provides a standalone vision encoder service that works with vLLM V1 engine.
It uses native PyTorch/HuggingFace models instead of vLLM-wrapped versions to avoid
tensor parallel initialization requirements.

Architecture:
- Uses the same vision processing pipeline as DeepSeek-OCR
- Direct weight loading from HuggingFace checkpoint
- No dependency on vLLM's model instance or parallel state
"""

import torch
import torch.nn as nn
from typing import List, Optional
from PIL import Image
import numpy as np


class StandaloneVisionEncoder:
    """
    Standalone vision encoder that works independently of vLLM V1 engine.

    This encoder uses the DeepSeek-OCR model's vision encoding directly
    by loading the model instance and accessing its vision processing methods.
    """

    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize standalone vision encoder

        Args:
            model_path: Path to DeepSeek-OCR model (HuggingFace or local)
            device: Device to load model on
            dtype: Data type for model weights
        """
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.model = None

        # Load the model directly using transformers
        self._load_model()

    def _load_model(self):
        """Load DeepSeek-OCR model directly"""
        from transformers import AutoModel

        print(f"Loading DeepSeek-OCR model from {self.model_path}...")
        print("This may take a few minutes...")

        # Load the full model using AutoModel with trust_remote_code
        # This will use the custom model class from the repository
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )

        self.model.eval()
        print("✓ Model loaded successfully")

    @torch.no_grad()
    def encode_images(
        self,
        images: List[Image.Image],
        return_global: bool = True,
        return_local: bool = False
    ) -> List[torch.Tensor]:
        """
        Encode images to visual embeddings directly through vision encoders

        Uses the model's vision_model (CLIP) and sam_model directly

        Args:
            images: List of PIL Images (rendered text images at 640x640)
            return_global: If True, return CLS token (global image representation)
            return_local: If True, return local patch features (10x10 grid)

        Returns:
            List of visual embedding tensors, one per image
            - If return_global=True, return_local=False: [1, 1280] CLS token only
            - If return_global=False, return_local=True: [111, 1280] local patches
            - If both True: [112, 1280] CLS + local patches
        """
        from process.image_process import DeepseekOCRProcessor
        from PIL import ImageOps
        import numpy as np

        processor = DeepseekOCRProcessor()

        # Process images directly using vision transforms (no text tokenization!)
        pixel_values_list = []
        images_spatial_crop_list = []

        for image in images:
            # Pad image to base_size (typically 1024)
            global_view = ImageOps.pad(
                image,
                (processor.base_size, processor.base_size),
                color=tuple(int(x * 255) for x in processor.image_transform.mean)
            )

            # Apply vision transform (resize, normalize, to tensor)
            pixel_values = processor.image_transform(global_view)
            pixel_values_list.append(pixel_values)

            # For rendered text images at 640x640, no tiling needed
            images_spatial_crop_list.append([1, 1])  # Single tile

        # Stack into batch tensors
        pixel_values = torch.stack(pixel_values_list, dim=0).to(
            device=self.device, dtype=self.dtype
        )
        images_spatial_crop = torch.tensor(
            images_spatial_crop_list, dtype=torch.long
        ).to(device=self.device)

        batch_size = len(images)

        # Encode using model's vision encoders directly - BATCHED for GPU efficiency

        # Step 1: SAM encoding - BATCHED (processes all images at once)
        sam_features = self.model.model.sam_model(pixel_values)  # [B, C, H, W]

        # Step 2: CLIP encoding with SAM features - BATCHED
        clip_features = self.model.model.vision_model(pixel_values, sam_features)  # [B, seq, dim]
        # clip_features[:, 0] = CLS token (global representation)
        # clip_features[:, 1:] = patch tokens (local representation)

        embeddings_list = []

        if return_global and not return_local:
            # GLOBAL FEATURES ONLY: Mean pool the projected local features
            # This creates a single global representation from all local patches
            #
            # Note: DeepSeek-OCR doesn't have a native CLS-only pathway because
            # the projector expects 2048 dim (CLIP 1024 + SAM 1024), not just CLS.
            # We create global features by mean-pooling the projected local features.

            # Concatenate CLIP patch features + SAM features (same as local)
            features = torch.cat(
                (
                    clip_features[:, 1:],  # Skip CLS token [B, seq-1, dim1]
                    sam_features.flatten(2).permute(0, 2, 1),  # Flatten spatial [B, hw, dim2]
                ),
                dim=-1,
            )  # [B, hw, dim1+dim2]

            # Project features
            features = self.model.model.projector(features)  # [B, hw, hidden_dim]

            # Mean pool across all patches to get global representation
            for jdx in range(batch_size):
                img_features = features[jdx]  # [hw, hidden_dim]
                global_feature = img_features.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                embeddings_list.append(global_feature)

        elif return_local and not return_global:
            # LOCAL FEATURES ONLY: Return patch tokens (original behavior)
            # Concatenate CLIP patch features + SAM features
            features = torch.cat(
                (
                    clip_features[:, 1:],  # Skip CLS token [B, seq-1, dim1]
                    sam_features.flatten(2).permute(0, 2, 1),  # Flatten spatial [B, hw, dim2]
                ),
                dim=-1,
            )  # [B, hw, dim1+dim2]

            # Project features
            features = self.model.model.projector(features)  # [B, hw, hidden_dim]

            # Add newline tokens and view separator per image
            _, hw, dim = features.shape
            side = int(hw**0.5)

            for jdx in range(batch_size):
                img_features = features[jdx]  # [hw, hidden_dim]

                # Reshape and add newline tokens
                img_features = img_features.view(side, side, dim)
                newline = self.model.model.image_newline[None, None, :].expand(side, 1, dim)
                img_features = torch.cat([img_features, newline], dim=1)
                img_features = img_features.view(-1, dim)  # [seq_len, hidden_dim]

                # Add view separator
                combined = torch.cat(
                    [img_features, self.model.model.view_seperator[None, :]],
                    dim=0
                )

                embeddings_list.append(combined)

        else:
            # BOTH or NEITHER: Not implemented, raise error
            raise ValueError(
                f"Invalid feature combination: return_global={return_global}, return_local={return_local}. "
                "Must set exactly one of return_global or return_local to True."
            )

        return embeddings_list


def create_vision_encoder(
    model_path: str = "deepseek-ai/DeepSeek-OCR",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
) -> StandaloneVisionEncoder:
    """
    Factory function to create standalone vision encoder

    Args:
        model_path: Path to DeepSeek-OCR model
        device: Device to load on
        dtype: Data type for weights

    Returns:
        Initialized StandaloneVisionEncoder
    """
    return StandaloneVisionEncoder(
        model_path=model_path,
        device=device,
        dtype=dtype
    )
