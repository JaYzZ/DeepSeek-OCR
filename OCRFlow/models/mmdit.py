"""
MMDiT (Multimodal Diffusion Transformer) for OCRFlow

Implements a multimodal diffusion transformer that learns to generate image tokens
from text inputs using rectified flow matching. Architecture is based on UniFlow/SD3.5.

Key features:
- Joint attention between text and image token modalities
- Rectified flow matching for training (predicts constant velocity)
- Adaptive layer normalization (AdaLN) with timestep conditioning
- Multiple model sizes from Tiny (150M) to XL (2.5B) parameters
- Classifier-free guidance support for inference

Architecture:
    Text → Text Encoders (T5 + CLIP) → Text Embeddings
    Noise x_0 → MMDiT (conditioned on text + timestep) → Velocity pred
    x_{t+dt} = x_t + v * dt → ... → Image Tokens x_1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from ..configs.model_config import ModelConfig


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings.

    Similar to transformer positional encodings but for continuous timesteps.
    """

    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: 1-D tensor of N indices, one per batch element
            dim: Embedding dimension
            max_period: Controls frequency range
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] timesteps in [0, 1]
        Returns:
            [B, hidden_size] embeddings
        """
        t_freq = self.timestep_embedding(t, self.freq_embed_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply affine modulation: scale * x + shift
    Used in AdaLN (Adaptive Layer Normalization).
    """
    return x * (1 + scale) + shift


class MLP(nn.Module):
    """
    Simple MLP with GELU activation.
    """

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class JointAttention(nn.Module):
    """
    Joint multi-head attention for text and image tokens.

    Both modalities attend to each other in a single attention operation,
    enabling rich cross-modal interactions.

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        qk_norm: Whether to normalize Q and K (improves stability)
    """

    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Separate Q/K/V projections for image and text
        self.qkv_img = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.qkv_txt = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

        # QK normalization (from SD3)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False, eps=1e-6)

        # Output projections
        self.proj_img = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj_txt = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        x_img: torch.Tensor,  # [B, N_img, D]
        x_txt: torch.Tensor,  # [B, N_txt, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute joint attention.

        Returns:
            (attended_img, attended_txt)
        """
        B, N_img, D = x_img.shape
        _, N_txt, _ = x_txt.shape

        # Compute Q/K/V for both modalities
        qkv_img = self.qkv_img(x_img).reshape(B, N_img, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_txt = self.qkv_txt(x_txt).reshape(B, N_txt, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q_img, k_img, v_img = qkv_img[0], qkv_img[1], qkv_img[2]  # Each: [B, H, N, head_dim]
        q_txt, k_txt, v_txt = qkv_txt[0], qkv_txt[1], qkv_txt[2]

        # QK normalization
        if self.qk_norm:
            q_img = self.q_norm(q_img)
            k_img = self.k_norm(k_img)
            q_txt = self.q_norm(q_txt)
            k_txt = self.k_norm(k_txt)

        # Concatenate keys and values for joint attention
        k_joint = torch.cat([k_img, k_txt], dim=2)  # [B, H, N_img + N_txt, head_dim]
        v_joint = torch.cat([v_img, v_txt], dim=2)

        # Image queries attend to all tokens (img + txt)
        attn_img = F.scaled_dot_product_attention(
            q_img, k_joint, v_joint,
            dropout_p=0.0, is_causal=False
        )  # [B, H, N_img, head_dim]

        # Text queries attend to all tokens (img + txt)
        attn_txt = F.scaled_dot_product_attention(
            q_txt, k_joint, v_joint,
            dropout_p=0.0, is_causal=False
        )  # [B, H, N_txt, head_dim]

        # Reshape and project
        attn_img = attn_img.transpose(1, 2).reshape(B, N_img, D)
        attn_txt = attn_txt.transpose(1, 2).reshape(B, N_txt, D)

        out_img = self.proj_img(attn_img)
        out_txt = self.proj_txt(attn_txt)

        return out_img, out_txt


class MMDiTBlock(nn.Module):
    """
    Multimodal DiT block with joint attention and adaptive layer norm (AdaLN).

    Implements the SD3-style architecture with separate AdaLN modulation
    for text and image modalities.

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio (typically 4.0)
        qk_norm: Whether to apply QK normalization
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Separate layer norms for image and text (no affine parameters, modulated by AdaLN)
        self.norm1_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Joint attention
        self.attn = JointAttention(hidden_size, num_heads, qk_norm=qk_norm)

        # Separate norms for MLP
        self.norm2_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Separate MLPs for image and text
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_img = MLP(hidden_size, mlp_hidden)
        self.mlp_txt = MLP(hidden_size, mlp_hidden)

        # AdaLN modulation for image tokens
        # Outputs 6 parameters: shift/scale for norm1, gate for attn, shift/scale for norm2, gate for mlp
        self.adaLN_modulation_img = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # AdaLN modulation for text tokens
        self.adaLN_modulation_txt = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x_img: torch.Tensor,  # [B, N_img, D]
        x_txt: torch.Tensor,  # [B, N_txt, D]
        c: torch.Tensor,      # [B, D] conditioning (timestep + pooled text)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_img: Image latent tokens
            x_txt: Text tokens
            c: Conditioning vector (combines timestep and pooled text embeddings)

        Returns:
            (updated_x_img, updated_x_txt)
        """
        # Get modulation parameters for image
        shift_msa_img, scale_msa_img, gate_msa_img, shift_mlp_img, scale_mlp_img, gate_mlp_img = \
            self.adaLN_modulation_img(c).chunk(6, dim=-1)

        # Get modulation parameters for text
        shift_msa_txt, scale_msa_txt, gate_msa_txt, shift_mlp_txt, scale_mlp_txt, gate_mlp_txt = \
            self.adaLN_modulation_txt(c).chunk(6, dim=-1)

        # Pre-norm + modulation for attention
        norm_img = modulate(
            self.norm1_img(x_img),
            shift_msa_img.unsqueeze(1),
            scale_msa_img.unsqueeze(1)
        )
        norm_txt = modulate(
            self.norm1_txt(x_txt),
            shift_msa_txt.unsqueeze(1),
            scale_msa_txt.unsqueeze(1)
        )

        # Joint attention
        attn_img, attn_txt = self.attn(norm_img, norm_txt)

        # Add gated residual connection
        x_img = x_img + gate_msa_img.unsqueeze(1) * attn_img
        x_txt = x_txt + gate_msa_txt.unsqueeze(1) * attn_txt

        # Pre-norm + modulation for MLP
        norm_img_mlp = modulate(
            self.norm2_img(x_img),
            shift_mlp_img.unsqueeze(1),
            scale_mlp_img.unsqueeze(1)
        )
        norm_txt_mlp = modulate(
            self.norm2_txt(x_txt),
            shift_mlp_txt.unsqueeze(1),
            scale_mlp_txt.unsqueeze(1)
        )

        # MLP + gated residual
        x_img = x_img + gate_mlp_img.unsqueeze(1) * self.mlp_img(norm_img_mlp)
        x_txt = x_txt + gate_mlp_txt.unsqueeze(1) * self.mlp_txt(norm_txt_mlp)

        return x_img, x_txt


class FinalLayer(nn.Module):
    """
    Final layer that predicts the velocity for rectified flow.

    Uses AdaLN to modulate the final layer norm based on timestep.
    """

    def __init__(self, hidden_size: int, output_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input features
            c: [B, D] conditioning

        Returns:
            [B, N, output_channels] predicted velocity
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        x = self.linear(x)
        return x


class MMDiTOCRFlow(nn.Module):
    """
    Multimodal Diffusion Transformer for OCRFlow.

    Predicts velocity in rectified flow to generate image tokens from text.

    Model Sizes:
        - Tiny: 384 dim, 12 layers, 6 heads → ~150M params
        - Small: 768 dim, 12 layers, 12 heads → ~350M params
        - Base: 1024 dim, 18 layers, 16 heads → ~700M params
        - Large: 1536 dim, 24 layers, 24 heads → ~1.7B params
        - XL: 1536 dim, 30 layers, 24 heads → ~2.5B params

    Args:
        config: ModelConfig instance with all model hyperparameters
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.num_image_tokens = config.num_image_tokens
        self.token_dim = config.token_dim

        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(
            hidden_size=config.time_embed_dim,
            freq_embed_size=config.freq_embed_size
        )

        # Text conditioning projections
        if config.use_dual_text_encoders:
            # T5 sequence embeddings
            self.text_seq_proj = nn.Linear(config.t5_dim, config.hidden_size)
            # CLIP pooled embeddings
            self.text_pooled_proj = nn.Linear(config.clip_pooled_dim, config.hidden_size)
        else:
            # Single text encoder
            self.text_seq_proj = nn.Linear(config.t5_dim, config.hidden_size)
            self.text_pooled_proj = None

        # Image token projection (from DeepSeek OCR token_dim to hidden_size)
        self.img_proj = nn.Linear(config.token_dim, config.hidden_size)

        # Learnable positional embeddings for image tokens
        self.pos_embed_img = nn.Parameter(
            torch.randn(1, config.num_image_tokens, config.hidden_size) * 0.02
        )

        # MMDiT blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qk_norm=config.qk_norm,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer (predicts velocity)
        self.final_layer = FinalLayer(config.hidden_size, config.token_dim)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights following DiT initialization scheme.

        Key ideas:
        - Xavier uniform for most layers
        - Zero-out AdaLN modulation final layers (ensures model starts as identity)
        - Zero-out final prediction layer (ensures predictions start near zero)
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)

        # Zero-out AdaLN modulation layers in blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation_img[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_img[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation_txt[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_txt[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x_t: torch.Tensor,                    # [B, N, token_dim] noisy image tokens at timestep t
        t: torch.Tensor,                      # [B] timesteps in [0, 1]
        text_seq_embeds: torch.Tensor,        # [B, L_txt, t5_dim] T5 sequence embeddings
        text_pooled_embeds: torch.Tensor,     # [B, clip_dim] CLIP pooled embeddings
        cfg_mask: Optional[torch.Tensor] = None, # [B] mask for classifier-free guidance (0 = drop text)
    ) -> torch.Tensor:
        """
        Forward pass of MMDiT to predict velocity.

        Args:
            x_t: Noisy image tokens at timestep t [B, num_tokens, token_dim]
            t: Timesteps [B] in range [0, 1]
            text_seq_embeds: Text sequence embeddings from T5 [B, seq_len, t5_dim]
            text_pooled_embeds: Pooled text embeddings from CLIP [B, clip_dim]
            cfg_mask: Optional mask for CFG training (0 = unconditional, 1 = conditional)

        Returns:
            Predicted velocity [B, num_tokens, token_dim]
        """
        B = x_t.shape[0]

        # Project image tokens to hidden dimension and add positional embeddings
        x_img = self.img_proj(x_t) + self.pos_embed_img  # [B, N_img, D]

        # Project text sequence embeddings
        x_txt = self.text_seq_proj(text_seq_embeds)  # [B, N_txt, D]

        # Get timestep embeddings
        t_emb = self.timestep_embedder(t)  # [B, D]

        # Get pooled text conditioning
        if self.text_pooled_proj is not None:
            pooled_emb = self.text_pooled_proj(text_pooled_embeds)  # [B, D]
        else:
            # Mean pool over sequence if no CLIP encoder
            pooled_emb = x_txt.mean(dim=1)  # [B, D]

        # Combine timestep and pooled text for conditioning
        c = t_emb + pooled_emb  # [B, D]

        # Apply CFG mask if provided (zero out text conditioning for unconditional samples)
        if cfg_mask is not None:
            # cfg_mask is [B], 0 = unconditional, 1 = conditional
            # Expand to [B, 1] and multiply with pooled embedding
            mask = cfg_mask.unsqueeze(1).to(c.dtype)  # [B, 1]
            c = t_emb + pooled_emb * mask

            # Also mask text tokens
            x_txt = x_txt * mask.unsqueeze(2)  # [B, N_txt, D]

        # Pass through MMDiT blocks
        for block in self.blocks:
            x_img, x_txt = block(x_img, x_txt, c)

        # Final layer: predict velocity
        v_pred = self.final_layer(x_img, c)  # [B, N_img, token_dim]

        return v_pred

    @torch.no_grad()
    def sample(
        self,
        text_seq_embeds: torch.Tensor,
        text_pooled_embeds: torch.Tensor,
        num_steps: int = 20,
        cfg_scale: float = 3.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Sample image tokens using Euler sampling.

        Args:
            text_seq_embeds: Text sequence embeddings [B, L, D_t5]
            text_pooled_embeds: Pooled text embeddings [B, D_clip]
            num_steps: Number of Euler steps
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            device: Device for computation
            dtype: Data type

        Returns:
            Generated image tokens [B, num_tokens, token_dim]
        """
        B = text_seq_embeds.shape[0]

        # Start from noise
        x_t = torch.randn(
            B, self.num_image_tokens, self.token_dim,
            device=device, dtype=dtype
        )

        # Euler sampling
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((B,), step * dt, device=device, dtype=dtype)

            if cfg_scale > 1.0:
                # Conditional prediction
                v_cond = self(
                    x_t, t, text_seq_embeds, text_pooled_embeds,
                    cfg_mask=torch.ones(B, device=device)
                )

                # Unconditional prediction
                v_uncond = self(
                    x_t, t, text_seq_embeds, text_pooled_embeds,
                    cfg_mask=torch.zeros(B, device=device)
                )

                # Classifier-free guidance
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                # No guidance
                v = self(x_t, t, text_seq_embeds, text_pooled_embeds)

            # Euler step
            x_t = x_t + v * dt

        return x_t


def create_mmdit_ocrflow(
    model_size: str = "small",
    **kwargs
) -> MMDiTOCRFlow:
    """
    Factory function to create MMDiTOCRFlow model.

    Args:
        model_size: One of ["tiny", "small", "base", "large", "xl"]
        **kwargs: Additional config overrides

    Returns:
        MMDiTOCRFlow instance

    Example:
        >>> model = create_mmdit_ocrflow("small")
        >>> # Test forward pass
        >>> x_t = torch.randn(2, 1600, 1280).cuda().bfloat16()
        >>> t = torch.rand(2).cuda().bfloat16()
        >>> text_seq = torch.randn(2, 512, 768).cuda().bfloat16()
        >>> text_pool = torch.randn(2, 768).cuda().bfloat16()
        >>> v = model(x_t, t, text_seq, text_pool)
        >>> v.shape
        torch.Size([2, 1600, 1280])
    """
    from ..configs.model_config import get_model_config

    config = get_model_config(model_size, **kwargs)
    model = MMDiTOCRFlow(config)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Created MMDiTOCRFlow ({model_size}):")
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing MMDiTOCRFlow...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create small model
    model = create_mmdit_ocrflow("small")
    model = model.to(device).to(torch.bfloat16)

    # Test inputs
    B, N, D = 2, 1600, 1280
    x_t = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)
    t = torch.rand(B, device=device, dtype=torch.bfloat16)
    text_seq = torch.randn(B, 512, 768, device=device, dtype=torch.bfloat16)
    text_pool = torch.randn(B, 768, device=device, dtype=torch.bfloat16)

    # Forward pass
    with torch.no_grad():
        v_pred = model(x_t, t, text_seq, text_pool)

    print(f"\nInput shape: {x_t.shape}")
    print(f"Output shape: {v_pred.shape}")
    print(f"Expected: [2, 1600, 1280]")
    print("Test passed!" if v_pred.shape == x_t.shape else "Test failed!")

    # Test sampling
    print("\nTesting sampling...")
    with torch.no_grad():
        samples = model.sample(
            text_seq_embeds=text_seq,
            text_pooled_embeds=text_pool,
            num_steps=10,
            cfg_scale=2.0,
            device=device,
            dtype=torch.bfloat16,
        )

    print(f"Sampled shape: {samples.shape}")
    print("Sampling test passed!" if samples.shape == (B, N, D) else "Sampling test failed!")
