"""
Image Token Encoder using DeepSeek OCR Vision Models

This module extracts image tokens using DeepSeek OCR's dual vision encoders:
- SAM (Segment Anything Model) ViT-B for spatial features
- CLIP-L for semantic features

The encoded features are concatenated and projected to create image tokens
suitable for rectified flow training.

IMPORTANT: This is the encoder part of the frozen codec (like VQGAN in LDM).
All components (SAM, CLIP, Projector) are frozen during training.
Only the DiT flow model is trained.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

# Add DeepSeek OCR path
deepseek_vllm_path = os.path.join(
    os.path.dirname(__file__),
    "../../DeepSeek-OCR-master/DeepSeek-OCR-vllm"
)
sys.path.insert(0, os.path.abspath(deepseek_vllm_path))

from deepencoder.sam_vary_sdpa import build_sam_vit_b
from deepencoder.clip_sdpa import build_clip_l
from deepencoder.build_linear import MlpProjector
from addict import Dict


class ImageTokenEncoder(nn.Module):
    """
    Encodes images into tokens using DeepSeek OCR's vision models.

    Architecture:
        Image → SAM ViT-B + CLIP-L → Concat → MLP Projection → Image Tokens

    The output tokens are [B, num_tokens, token_dim] where:
        - num_tokens = (H/patch_size/downsample_ratio)^2
        - token_dim is the projection dimension (default 1280)

    Args:
        token_dim: Output token dimension (default 1280 for DeepSeek OCR)
        image_size: Input image size (default 640)
        patch_size: Patch size for tokenization (default 16)
        downsample_ratio: Downsampling ratio (default 4)
        freeze: Whether to freeze the vision encoders
        device: Device to load models on
        dtype: Data type for model weights
    """

    def __init__(
        self,
        token_dim: int = 1280,
        image_size: int = 640,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        freeze: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.dtype = dtype

        # Calculate number of tokens
        self.num_tokens_per_side = (image_size // patch_size) // downsample_ratio
        self.num_tokens = self.num_tokens_per_side ** 2

        # Initialize vision encoders
        self.sam_model = build_sam_vit_b().to(device).to(dtype)
        self.vision_model = build_clip_l().to(device).to(dtype)

        # Projection layer (SAM: 1024 channels, CLIP: 1024 channels → total 2048)
        self.projector = MlpProjector(
            Dict(
                projector_type="linear",
                input_dim=2048,  # SAM (1024) + CLIP (1024)
                n_embed=token_dim
            )
        ).to(device).to(dtype)

        # Freeze vision encoders if specified
        if freeze:
            self.freeze_encoders()

    def freeze_encoders(self):
        """Freeze the vision encoder parameters AND projector (complete codec freeze)."""
        for param in self.sam_model.parameters():
            param.requires_grad = False
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Freeze projector too - entire codec should be frozen (like VQGAN in LDM)
        for param in self.projector.parameters():
            param.requires_grad = False

        self.sam_model.eval()
        self.vision_model.eval()
        self.projector.eval()

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        return_2d: bool = False
    ) -> torch.Tensor:
        """
        Encode image to tokens.

        Args:
            pixel_values: Image tensor [B, 3, H, W] in range [0, 1] or [-1, 1]
            return_2d: If True, return tokens as [B, H_tokens, W_tokens, D]
                      If False, return as [B, N_tokens, D] where N = H*W

        Returns:
            Image tokens [B, num_tokens, token_dim] or [B, H, W, token_dim]
        """
        # Ensure correct dtype
        pixel_values = pixel_values.to(self.dtype)

        # Encode with SAM
        sam_features = self.sam_model(pixel_values)  # [B, 1024, H/16, W/16]

        # Encode with CLIP (conditioned on SAM features)
        clip_features = self.vision_model(pixel_values, sam_features)  # [B, 1+N, 1024]

        # Remove CLS token from CLIP output
        clip_features = clip_features[:, 1:, :]  # [B, N, 1024]

        # Reshape SAM features to match CLIP
        sam_features_flat = sam_features.flatten(2).permute(0, 2, 1)  # [B, N, 1024]

        # Concatenate features
        combined_features = torch.cat([clip_features, sam_features_flat], dim=-1)  # [B, N, 2048]

        # Project to token dimension
        tokens = self.projector(combined_features)  # [B, N, token_dim]

        if return_2d:
            # Reshape to 2D grid
            B, N, D = tokens.shape
            H = W = int(N ** 0.5)
            tokens = tokens.view(B, H, W, D)

        return tokens

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_2d: bool = False
    ) -> torch.Tensor:
        """Forward pass - encode image to tokens."""
        return self.encode_image(pixel_values, return_2d=return_2d)

    def get_num_tokens(self) -> int:
        """Get number of tokens per image."""
        return self.num_tokens

    def get_token_dim(self) -> int:
        """Get token dimension."""
        return self.token_dim


def create_image_token_encoder(
    token_dim: int = 1280,
    image_size: int = 640,
    freeze: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> ImageTokenEncoder:
    """
    Factory function to create an ImageTokenEncoder.

    Args:
        token_dim: Output token dimension
        image_size: Input image size
        freeze: Whether to freeze vision encoders
        device: Device to load on
        dtype: Model dtype

    Returns:
        ImageTokenEncoder instance

    Example:
        >>> encoder = create_image_token_encoder(token_dim=1280, image_size=640)
        >>> images = torch.randn(2, 3, 640, 640).cuda().bfloat16()
        >>> tokens = encoder(images)
        >>> tokens.shape
        torch.Size([2, 1600, 1280])
    """
    return ImageTokenEncoder(
        token_dim=token_dim,
        image_size=image_size,
        freeze=freeze,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    # Test the encoder
    print("Testing ImageTokenEncoder...")

    encoder = create_image_token_encoder(
        token_dim=1280,
        image_size=640,
        freeze=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"Number of tokens: {encoder.get_num_tokens()}")
    print(f"Token dimension: {encoder.get_token_dim()}")

    # Test with dummy image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_image = torch.randn(2, 3, 640, 640).to(device).bfloat16()

    with torch.no_grad():
        tokens = encoder(dummy_image)

    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {tokens.shape}")
    print(f"Expected shape: [2, 1600, 1280]")
    print("Test passed!" if tokens.shape == (2, 1600, 1280) else "Test failed!")
