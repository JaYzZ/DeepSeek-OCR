"""
MMDiT with SANA Pretrained Weight Transfer

This module adapts SANA's efficient DiT architecture for OCRFlow by:
1. Using channel-to-spatial unpatch/patch operations (like Qwen's patchify)
2. Transferring pretrained weights from SANA (600M or 1.6B parameters)
3. Only training two small projection layers

SANA Architecture:
- 600M model: 20 layers, 2240 hidden dim (70 heads × 32)
- 1.6B model: 28 layers, 2240 hidden dim (70 heads × 32)
- Uses linear attention (more efficient than standard attention)
- Trained on text-to-image generation

Benefits over Qwen-Image:
- 33× smaller (600M vs 20B)
- Much faster training and inference
- Less memory required
- Still pretrained on image generation
"""

import torch
import torch.nn as nn
from diffusers import SanaTransformer2DModel
from typing import Optional, Tuple


def unpatch_channels_to_spatial(x: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """
    Convert channels to spatial dimensions (reverse of patchify).

    Args:
        x: [B, C, H, W] input tensor
        patch_size: Size of patches (default: 4)

    Returns:
        [B, C//(patch_size**2), H*patch_size, W*patch_size]

    Example:
        [B, 1280, 10, 10] → [B, 80, 40, 40]
        Uses 16 channels to encode each 4×4 spatial patch
    """
    B, C, H, W = x.shape
    C_out = C // (patch_size * patch_size)

    # Reshape: [B, C, H, W] → [B, C_out, patch_size, patch_size, H, W]
    x = x.reshape(B, C_out, patch_size, patch_size, H, W)

    # Permute: [B, C_out, patch_size, patch_size, H, W] → [B, C_out, H, patch_size, W, patch_size]
    x = x.permute(0, 1, 4, 2, 5, 3)

    # Reshape: [B, C_out, H, patch_size, W, patch_size] → [B, C_out, H*patch_size, W*patch_size]
    x = x.reshape(B, C_out, H * patch_size, W * patch_size)

    return x


def patch_spatial_to_channels(x: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """
    Convert spatial dimensions to channels (patchify operation).

    Args:
        x: [B, C, H, W] input tensor
        patch_size: Size of patches (default: 4)

    Returns:
        [B, C*(patch_size**2), H//patch_size, W//patch_size]

    Example:
        [B, 80, 40, 40] → [B, 1280, 10, 10]
        Packs 4×4 spatial into 16 channels
    """
    B, C, H, W = x.shape
    C_out = C * (patch_size * patch_size)

    # Reshape: [B, C, H, W] → [B, C, H//patch_size, patch_size, W//patch_size, patch_size]
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)

    # Permute: [B, C, H//patch_size, patch_size, W//patch_size, patch_size] → [B, C, patch_size, patch_size, H//patch_size, W//patch_size]
    x = x.permute(0, 1, 3, 5, 2, 4)

    # Reshape: [B, C, patch_size, patch_size, H//patch_size, W//patch_size] → [B, C_out, H//patch_size, W//patch_size]
    x = x.reshape(B, C_out, H // patch_size, W // patch_size)

    return x


class MMDiTOCRFlowSANA(nn.Module):
    """
    OCRFlow MMDiT initialized from SANA pretrained weights.

    Architecture:
        Visual tokens [B, 111, 1280]
        ↓ Reshape to [B, 1280, 10, 10]
        ↓ Unpatch: [B, 1280, 10, 10] → [B, 80, 40, 40] (channel→spatial)
        ↓ Project: 80 → 32 (SANA input channels, NEW)
        ↓ SANA transformer: 20 or 28 layers (PRETRAINED)
        ↓ Project: 32 → 80 (NEW)
        ↓ Patch: [B, 80, 40, 40] → [B, 1280, 10, 10] (spatial→channel)
        ↓ Reshape back to [B, 100, 1280] + append 11 structural tokens

    Args:
        sana_model_path: Path to SANA model (600M or 1.6B)
        freeze_backbone: Whether to freeze pretrained layers
        patch_size: Patch size for unpatch/patch operations (default: 4)
    """

    def __init__(
        self,
        sana_model_path: str = "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        freeze_backbone: bool = False,
        patch_size: int = 4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_image_tokens = 111
        self.num_trainable_tokens = 100
        self.token_dim = 1280
        self.spatial_size = 10  # sqrt(100) = 10

        # Load pretrained SANA transformer
        print(f"Loading pretrained SANA from {sana_model_path}...")
        self.sana_transformer = SanaTransformer2DModel.from_pretrained(
            sana_model_path,
            subfolder="transformer",
            torch_dtype=torch.float32  # Use float32 for training
        )

        # Extract architecture info
        config = self.sana_transformer.config
        self.sana_in_channels = config.in_channels  # Typically 32
        self.hidden_size = config.num_attention_heads * config.attention_head_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_attention_heads
        self.caption_channels = config.caption_channels  # For text conditioning

        print(f"Loaded SANA: {self.num_layers} layers, {self.num_heads} heads, {self.hidden_size} hidden")
        print(f"  in_channels: {self.sana_in_channels}")
        print(f"  caption_channels: {self.caption_channels}")

        # Calculate channel dimension after unpatch
        # 1280 channels with 10×10 spatial → 80 channels with 40×40 spatial
        self.unpatched_channels = self.token_dim // (patch_size * patch_size)  # 1280 / 16 = 80
        self.unpatched_size = self.spatial_size * patch_size  # 10 * 4 = 40

        print(f"Unpatch: [{self.token_dim}, {self.spatial_size}, {self.spatial_size}] → [{self.unpatched_channels}, {self.unpatched_size}, {self.unpatched_size}]")

        # NEW: Input projection (80 → 32 for SANA)
        self.img_in = nn.Conv2d(
            self.unpatched_channels,
            self.sana_in_channels,
            kernel_size=1,
            bias=True
        )

        # NEW: Output projection (32 → 80)
        self.norm_out = nn.LayerNorm(self.sana_in_channels, eps=1e-6)
        self.proj_out = nn.Conv2d(
            self.sana_in_channels,
            self.unpatched_channels,
            kernel_size=1,
            bias=True
        )

        # NEW: Text projection (need to match SANA's caption_channels)
        # T5 gives 768-dim, CLIP gives 768-dim pooled
        # We'll concatenate them and project to caption_channels
        self.text_projection = nn.Linear(768 + 768, self.caption_channels)

        # Initialize new layers
        self._initialize_new_layers()

        # Optionally freeze pretrained backbone
        if freeze_backbone:
            self._freeze_backbone()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pretrained_params = sum(p.numel() for p in self.sana_transformer.parameters())

        print(f"\n{'='*70}")
        print(f"OCRFlow MMDiT with SANA Pretrained Weights")
        print(f"{'='*70}")
        print(f"Total parameters: {total_params / 1e9:.2f}B")
        print(f"Pretrained parameters: {pretrained_params / 1e6:.1f}M ({pretrained_params/total_params*100:.1f}%)")
        print(f"New parameters: {(total_params-pretrained_params) / 1e6:.1f}M ({(1-pretrained_params/total_params)*100:.1f}%)")
        print(f"Trainable parameters: {trainable_params / 1e9:.2f}B")
        print(f"{'='*70}\n")

    def _initialize_new_layers(self):
        """Initialize newly added layers."""
        # Xavier uniform for input projection
        nn.init.xavier_uniform_(self.img_in.weight)
        nn.init.zeros_(self.img_in.bias)

        # Zero init for output projection (flow matching convention)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # Xavier for text projection
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)

        print("✓ Initialized new projection layers")

    def _freeze_backbone(self):
        """Freeze all SANA pretrained weights."""
        for param in self.sana_transformer.parameters():
            param.requires_grad = False
        print("✓ Froze pretrained SANA weights")

    def reshape_tokens_to_spatial(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape visual tokens to spatial format.

        Args:
            tokens: [B, 111, 1280] visual tokens

        Returns:
            spatial: [B, 1280, 10, 10] spatial features
            structural: [B, 11, 1280] structural tokens
        """
        B = tokens.shape[0]

        # Split: first 100 tokens = trainable, last 11 = structural
        trainable = tokens[:, :self.num_trainable_tokens, :]  # [B, 100, 1280]
        structural = tokens[:, self.num_trainable_tokens:, :]  # [B, 11, 1280]

        # Reshape trainable to spatial [B, 1280, 10, 10]
        spatial = trainable.transpose(1, 2).reshape(B, self.token_dim, self.spatial_size, self.spatial_size)

        return spatial, structural

    def reshape_spatial_to_tokens(self, spatial: torch.Tensor, structural: torch.Tensor) -> torch.Tensor:
        """
        Reshape spatial features back to token sequence.

        Args:
            spatial: [B, 1280, 10, 10] spatial features
            structural: [B, 11, 1280] structural tokens

        Returns:
            tokens: [B, 111, 1280] visual tokens
        """
        B = spatial.shape[0]

        # Flatten spatial [B, 1280, 10, 10] → [B, 100, 1280]
        trainable = spatial.reshape(B, self.token_dim, -1).transpose(1, 2)

        # Concatenate with structural tokens
        tokens = torch.cat([trainable, structural], dim=1)  # [B, 111, 1280]

        return tokens

    def forward(
        self,
        x_t: torch.Tensor,  # [B, 111, 1280] noisy visual tokens
        t: torch.Tensor,  # [B] timesteps
        text_seq_embeds: torch.Tensor,  # [B, L, 768] T5 embeddings
        text_pooled_embeds: torch.Tensor,  # [B, 768] CLIP embeddings
        cfg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting velocity for flow matching.

        Args:
            x_t: Noisy visual tokens [B, 111, 1280]
            t: Timesteps [B]
            text_seq_embeds: Text sequence embeddings [B, L, 768]
            text_pooled_embeds: Pooled text embeddings [B, 768]
            cfg_mask: Classifier-free guidance mask [B]

        Returns:
            Predicted velocity [B, 111, 1280]
        """
        B = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype

        # 1. Reshape tokens to spatial format
        spatial, structural = self.reshape_tokens_to_spatial(x_t)  # [B, 1280, 10, 10], [B, 11, 1280]

        # 2. Unpatch: Convert channels to spatial (1280ch, 10×10) → (80ch, 40×40)
        unpatched = unpatch_channels_to_spatial(spatial, self.patch_size)  # [B, 80, 40, 40]

        # 3. Project to SANA input channels (80 → 32)
        hidden = self.img_in(unpatched)  # [B, 32, 40, 40]

        # 4. Prepare text conditioning
        # Concatenate T5 pooled (mean of sequence) + CLIP pooled
        t5_pooled = text_seq_embeds.mean(dim=1)  # [B, 768]
        text_combined = torch.cat([t5_pooled, text_pooled_embeds], dim=-1)  # [B, 1536]
        text_embeds = self.text_projection(text_combined)  # [B, caption_channels]

        # Apply CFG dropout if mask provided
        if cfg_mask is not None:
            text_embeds = text_embeds * cfg_mask.view(-1, 1)

        # 5. Use SANA transformer backbone
        # SANA expects:
        # - hidden_states: [B, C, H, W]
        # - encoder_hidden_states: [B, caption_channels] or [B, L, caption_channels]
        # - timestep: [B] or scalar
        output = self.sana_transformer(
            hidden_states=hidden,
            encoder_hidden_states=text_embeds.unsqueeze(1),  # [B, 1, caption_channels]
            timestep=t,
            return_dict=True
        ).sample  # [B, 32, 40, 40]

        # 6. Project back to unpatched dimension (32 → 80)
        output = self.norm_out(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        v_unpatched = self.proj_out(output)  # [B, 80, 40, 40]

        # 7. Patch: Convert spatial back to channels (80ch, 40×40) → (1280ch, 10×10)
        v_spatial = patch_spatial_to_channels(v_unpatched, self.patch_size)  # [B, 1280, 10, 10]

        # 8. Reshape back to tokens
        # For structural tokens, predict zero velocity (no change)
        structural_v = torch.zeros_like(structural)
        v_pred = self.reshape_spatial_to_tokens(v_spatial, structural_v)  # [B, 111, 1280]

        return v_pred


def create_mmdit_ocrflow_sana(
    sana_model_path: str = "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    freeze_backbone: bool = False,
    patch_size: int = 4,
) -> MMDiTOCRFlowSANA:
    """
    Create OCRFlow MMDiT with pretrained SANA weights.

    Args:
        sana_model_path: Path to SANA model (600M or 1.6B)
            - Sana_600M_1024px_diffusers (600M params)
            - SANA1.5_1.6B_1024px_diffusers (1.6B params)
        freeze_backbone: Whether to freeze pretrained layers
        patch_size: Patch size for unpatch/patch operations

    Returns:
        MMDiTOCRFlowSANA model with pretrained SANA weights

    Example:
        >>> # Use 600M model
        >>> model = create_mmdit_ocrflow_sana(
        ...     sana_model_path="Efficient-Large-Model/Sana_600M_1024px_diffusers"
        ... )
        >>>
        >>> # Or use 1.6B model (as in BLIP3o-NEXT)
        >>> model = create_mmdit_ocrflow_sana(
        ...     sana_model_path="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
        ... )
    """
    model = MMDiTOCRFlowSANA(
        sana_model_path=sana_model_path,
        freeze_backbone=freeze_backbone,
        patch_size=patch_size,
    )

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing MMDiTOCRFlowSANA...\n")

    model = create_mmdit_ocrflow_sana()

    # Test forward pass
    B = 2
    x_t = torch.randn(B, 111, 1280)
    t = torch.rand(B)
    text_seq = torch.randn(B, 77, 768)
    text_pooled = torch.randn(B, 768)

    print(f"\nTesting forward pass...")
    print(f"  Input: {list(x_t.shape)}")

    with torch.no_grad():
        v_pred = model(x_t, t, text_seq, text_pooled)

    print(f"  Output: {list(v_pred.shape)}")
    print(f"\n✓ Test passed!")

    # Test unpatch/patch operations
    print(f"\nTesting unpatch/patch operations...")
    spatial = torch.randn(2, 1280, 10, 10)
    print(f"  Input: {list(spatial.shape)}")

    unpatched = unpatch_channels_to_spatial(spatial, patch_size=4)
    print(f"  After unpatch: {list(unpatched.shape)}")

    patched = patch_spatial_to_channels(unpatched, patch_size=4)
    print(f"  After patch: {list(patched.shape)}")

    print(f"  Reversible: {torch.allclose(spatial, patched, atol=1e-6)}")
    print(f"\n✓ Unpatch/patch operations are reversible!")
