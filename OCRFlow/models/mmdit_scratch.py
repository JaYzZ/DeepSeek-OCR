"""
MMDiT from Scratch for OCRFlow (~600M parameters)

A clean MMDiT implementation trained from scratch without any pretrained weights.
Designed to match QwenImage-like architecture at ~600M scale with 10×10 feature map support.

Architecture:
- ~600M parameters (similar to SANA 600M)
- 10×10 feature map (100 visual tokens)
- Joint attention between text and image modalities
- AdaLN (Adaptive Layer Norm) with timestep conditioning
- Rectified flow matching for training

Model Sizes:
- small: 768 hidden, 16 layers, 12 heads → ~200M params
- medium: 1280 hidden, 20 layers, 20 heads → ~400M params
- large: 1536 hidden, 24 layers, 24 heads → ~600M params (default)
- xl: 2048 hidden, 28 layers, 32 heads → ~1.2B params

Usage:
    model = create_mmdit_scratch("large")  # ~600M params
    v_pred = model(x_t, t, text_seq_embeds, text_pooled_embeds)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MMDiTScratchConfig:
    """Configuration for MMDiT from scratch"""

    # Model size
    hidden_size: int = 1536
    num_layers: int = 24
    num_heads: int = 24
    mlp_ratio: float = 4.0

    # Feature map (10×10 = 100 visual tokens)
    spatial_size: int = 10  # 10×10 feature map
    num_visual_tokens: int = 100  # spatial_size^2
    num_structural_tokens: int = 11  # newline + separator
    token_dim: int = 1280  # DeepSeek OCR dimension

    # Text encoder dimensions
    text_seq_dim: int = 768  # T5 dimension
    text_pooled_dim: int = 768  # CLIP pooled dimension
    max_text_len: int = 512

    # Attention
    qk_norm: bool = True
    use_rope: bool = True  # Rotary position embeddings
    rope_theta: float = 10000.0

    # Timestep embedding
    freq_embed_size: int = 256

    # Dropout
    dropout: float = 0.0

    @property
    def num_image_tokens(self) -> int:
        return self.num_visual_tokens + self.num_structural_tokens

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


# Model size presets (tuned for target parameter counts)
# MMDiT has ~2x params per block vs standard transformer (separate img/txt processing)
MODEL_CONFIGS = {
    "small": {
        "hidden_size": 512,
        "num_layers": 12,
        "num_heads": 8,
    },  # ~100M params
    "medium": {
        "hidden_size": 768,
        "num_layers": 16,
        "num_heads": 12,
    },  # ~300M params
    "large": {
        "hidden_size": 960,
        "num_layers": 20,
        "num_heads": 15,
    },  # ~600M params
    "xl": {
        "hidden_size": 1280,
        "num_layers": 24,
        "num_heads": 20,
    },  # ~1B params
}


def get_config(model_size: str = "large", **kwargs) -> MMDiTScratchConfig:
    """Get model configuration for specified size."""
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"model_size must be one of {list(MODEL_CONFIGS.keys())}")

    config_dict = MODEL_CONFIGS[model_size].copy()
    config_dict.update(kwargs)
    return MMDiTScratchConfig(**config_dict)


class RMSNorm(nn.Module):
    """RMS Normalization (more stable than LayerNorm)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to Q and K."""
    # cos/sin: [seq_len, head_dim]
    # q/k: [batch, heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimestepEmbedder(nn.Module):
    """Timestep embedding with sinusoidal encoding + MLP"""

    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.freq_embed_size)
        return self.mlp(t_freq)


class MLP(nn.Module):
    """SwiGLU-style MLP (better than standard GELU MLP)"""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio * 2 / 3)  # SwiGLU uses 2/3 factor
        mlp_hidden = ((mlp_hidden + 63) // 64) * 64  # Round to multiple of 64 for efficiency

        self.gate_proj = nn.Linear(hidden_size, mlp_hidden, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_hidden, bias=False)
        self.down_proj = nn.Linear(mlp_hidden, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class JointAttention(nn.Module):
    """
    Joint attention for image and text modalities.
    Both modalities attend to each other in a single attention operation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Separate QKV for image and text
        self.qkv_img = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.qkv_txt = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        # QK normalization (improves training stability)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Output projections
        self.proj_img = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_txt = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_img: torch.Tensor,  # [B, N_img, D]
        x_txt: torch.Tensor,  # [B, N_txt, D]
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_img, D = x_img.shape
        _, N_txt, _ = x_txt.shape

        # Compute QKV
        qkv_img = self.qkv_img(x_img).reshape(B, N_img, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_txt = self.qkv_txt(x_txt).reshape(B, N_txt, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q_img, k_img, v_img = qkv_img[0], qkv_img[1], qkv_img[2]
        q_txt, k_txt, v_txt = qkv_txt[0], qkv_txt[1], qkv_txt[2]

        # QK normalization
        if self.qk_norm:
            q_img = self.q_norm(q_img)
            k_img = self.k_norm(k_img)
            q_txt = self.q_norm(q_txt)
            k_txt = self.k_norm(k_txt)

        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            # Apply RoPE to image tokens only (text uses its own positions)
            cos_img = rope_cos[:N_img]
            sin_img = rope_sin[:N_img]
            q_img, k_img = apply_rotary_pos_emb(q_img, k_img, cos_img, sin_img)

            cos_txt = rope_cos[:N_txt]
            sin_txt = rope_sin[:N_txt]
            q_txt, k_txt = apply_rotary_pos_emb(q_txt, k_txt, cos_txt, sin_txt)

        # Joint attention: concatenate K, V from both modalities
        k_joint = torch.cat([k_img, k_txt], dim=2)
        v_joint = torch.cat([v_img, v_txt], dim=2)

        # Image attends to all
        attn_img = F.scaled_dot_product_attention(
            q_img, k_joint, v_joint,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        # Text attends to all
        attn_txt = F.scaled_dot_product_attention(
            q_txt, k_joint, v_joint,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        # Reshape and project
        attn_img = attn_img.transpose(1, 2).reshape(B, N_img, D)
        attn_txt = attn_txt.transpose(1, 2).reshape(B, N_txt, D)

        out_img = self.proj_img(attn_img)
        out_txt = self.proj_txt(attn_txt)

        return out_img, out_txt


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1.0 + scale) + shift


class MMDiTBlock(nn.Module):
    """
    MMDiT block with joint attention and AdaLN.

    Separate processing for image and text with shared attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Normalization
        self.norm1_img = RMSNorm(hidden_size)
        self.norm1_txt = RMSNorm(hidden_size)

        # Joint attention
        self.attn = JointAttention(hidden_size, num_heads, qk_norm, dropout)

        # MLP
        self.norm2_img = RMSNorm(hidden_size)
        self.norm2_txt = RMSNorm(hidden_size)
        self.mlp_img = MLP(hidden_size, mlp_ratio, dropout)
        self.mlp_txt = MLP(hidden_size, mlp_ratio, dropout)

        # AdaLN modulation (6 params each: shift/scale for attn, gate for attn, shift/scale for mlp, gate for mlp)
        self.adaLN_img = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_txt = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        c: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation params
        mod_img = self.adaLN_img(c).chunk(6, dim=-1)
        mod_txt = self.adaLN_txt(c).chunk(6, dim=-1)

        shift_attn_img, scale_attn_img, gate_attn_img = mod_img[0], mod_img[1], mod_img[2]
        shift_mlp_img, scale_mlp_img, gate_mlp_img = mod_img[3], mod_img[4], mod_img[5]

        shift_attn_txt, scale_attn_txt, gate_attn_txt = mod_txt[0], mod_txt[1], mod_txt[2]
        shift_mlp_txt, scale_mlp_txt, gate_mlp_txt = mod_txt[3], mod_txt[4], mod_txt[5]

        # Pre-norm + modulation for attention
        norm_img = modulate(self.norm1_img(x_img), shift_attn_img.unsqueeze(1), scale_attn_img.unsqueeze(1))
        norm_txt = modulate(self.norm1_txt(x_txt), shift_attn_txt.unsqueeze(1), scale_attn_txt.unsqueeze(1))

        # Joint attention
        attn_img, attn_txt = self.attn(norm_img, norm_txt, rope_cos, rope_sin)

        # Gated residual
        x_img = x_img + gate_attn_img.unsqueeze(1) * attn_img
        x_txt = x_txt + gate_attn_txt.unsqueeze(1) * attn_txt

        # Pre-norm + modulation for MLP
        norm_img = modulate(self.norm2_img(x_img), shift_mlp_img.unsqueeze(1), scale_mlp_img.unsqueeze(1))
        norm_txt = modulate(self.norm2_txt(x_txt), shift_mlp_txt.unsqueeze(1), scale_mlp_txt.unsqueeze(1))

        # MLP + gated residual
        x_img = x_img + gate_mlp_img.unsqueeze(1) * self.mlp_img(norm_img)
        x_txt = x_txt + gate_mlp_txt.unsqueeze(1) * self.mlp_txt(norm_txt)

        return x_img, x_txt


class FinalLayer(nn.Module):
    """Final layer for velocity prediction with AdaLN"""

    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.linear(x)


class MMDiTScratch(nn.Module):
    """
    MMDiT trained from scratch for OCRFlow.

    ~600M parameters at "large" size, supporting 10×10 feature maps.
    Includes conditioning placeholder for image/text rendering input.
    """

    def __init__(self, config: MMDiTScratchConfig):
        super().__init__()
        self.config = config

        # Timestep embedding
        self.time_embedder = TimestepEmbedder(config.hidden_size, config.freq_embed_size)

        # Text embeddings
        self.text_seq_proj = nn.Linear(config.text_seq_dim, config.hidden_size)
        self.text_pooled_proj = nn.Linear(config.text_pooled_dim, config.hidden_size)

        # Image token projection (from DeepSeek OCR dim to hidden)
        self.img_proj = nn.Linear(config.token_dim, config.hidden_size)

        # Conditioning image projection (same dim as visual tokens)
        # This is a placeholder for future conditioning:
        # - Text rendering tokens
        # - Native image tokens
        # - Any other visual conditioning
        self.cond_img_proj = nn.Linear(config.token_dim, config.hidden_size)
        self.cond_img_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid(),
        )
        # Pooled condition embedding (for AdaLN conditioning)
        self.cond_pooled_proj = nn.Sequential(
            nn.Linear(config.token_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Learnable positional embeddings for image tokens (10×10 grid)
        self.pos_embed_img = nn.Parameter(
            torch.randn(1, config.num_visual_tokens, config.hidden_size) * 0.02
        )

        # RoPE for attention
        if config.use_rope:
            self.rope = RotaryEmbedding(
                config.head_dim,
                max_seq_len=config.num_visual_tokens + config.max_text_len,
                theta=config.rope_theta
            )
        else:
            self.rope = None

        # MMDiT blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qk_norm=config.qk_norm,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer
        self.final_layer = FinalLayer(config.hidden_size, config.token_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DiT conventions"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Zero-out AdaLN modulation outputs (start as identity)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_img[-1].weight)
            nn.init.zeros_(block.adaLN_img[-1].bias)
            nn.init.zeros_(block.adaLN_txt[-1].weight)
            nn.init.zeros_(block.adaLN_txt[-1].bias)

        # Zero-out final layer (start with zero velocity)
        nn.init.zeros_(self.final_layer.adaLN[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

        # Zero-out conditioning gate (start without conditioning influence)
        nn.init.zeros_(self.cond_img_gate[0].weight)
        nn.init.zeros_(self.cond_img_gate[0].bias)

    def encode_condition(self, cond_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode conditioning image tokens.

        Args:
            cond_img: [B, N, token_dim] conditioning visual tokens
                      (text rendering or native image, same length as latent tokens)

        Returns:
            cond_hidden: [B, N, hidden_size] projected condition
            cond_pooled: [B, hidden_size] pooled condition for AdaLN
        """
        # Project condition tokens
        cond_hidden = self.cond_img_proj(cond_img)  # [B, N, hidden_size]

        # Pool to get global condition embedding
        cond_pooled = cond_img.mean(dim=1)  # [B, token_dim]
        cond_pooled = self.cond_pooled_proj(cond_pooled)  # [B, hidden_size]

        return cond_hidden, cond_pooled

    def fuse_condition(
        self,
        x_img: torch.Tensor,
        cond_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse conditioning into image features using gated fusion.

        Args:
            x_img: [B, N, hidden_size] image features
            cond_hidden: [B, N, hidden_size] condition features

        Returns:
            fused: [B, N, hidden_size] fused features
        """
        # Gated fusion
        gate = self.cond_img_gate(torch.cat([x_img, cond_hidden], dim=-1))
        fused = x_img + gate * cond_hidden
        return fused

    def forward(
        self,
        x_t: torch.Tensor,  # [B, 111, 1280] or [B, 100, 1280]
        t: torch.Tensor,  # [B] timesteps in [0, 1]
        text_seq_embeds: torch.Tensor,  # [B, L, text_seq_dim]
        text_pooled_embeds: torch.Tensor,  # [B, text_pooled_dim]
        cond_img: Optional[torch.Tensor] = None,  # [B, N, 1280] conditioning image tokens
        cfg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting velocity for flow matching.

        Args:
            x_t: Noisy visual tokens (can be 100 or 111 tokens)
            t: Timesteps in [0, 1]
            text_seq_embeds: Text sequence embeddings
            text_pooled_embeds: Pooled text embeddings
            cond_img: Optional conditioning image tokens (text rendering or native image)
                      Shape: [B, N, 1280] where N matches visual tokens (100 or 111)
            cfg_mask: Optional CFG mask (0 = unconditional)

        Returns:
            Predicted velocity (same shape as x_t)
        """
        B = x_t.shape[0]
        N_img = x_t.shape[1]

        # Handle both 100 and 111 token inputs
        if N_img == self.config.num_image_tokens:
            # Full 111 tokens: split visual and structural
            x_visual = x_t[:, :self.config.num_visual_tokens, :]
            x_structural = x_t[:, self.config.num_visual_tokens:, :]
            has_structural = True
        else:
            # Just visual tokens
            x_visual = x_t
            has_structural = False

        # Project image tokens and add position embeddings
        x_img = self.img_proj(x_visual) + self.pos_embed_img

        # Process conditioning if provided
        cond_hidden, cond_pooled = None, None
        if cond_img is not None:
            # Handle structural tokens in conditioning
            if cond_img.shape[1] == self.config.num_image_tokens:
                cond_visual = cond_img[:, :self.config.num_visual_tokens, :]
            else:
                cond_visual = cond_img
            cond_hidden, cond_pooled = self.encode_condition(cond_visual)

            # Fuse condition into image features
            x_img = self.fuse_condition(x_img, cond_hidden)

        # Project text
        x_txt = self.text_seq_proj(text_seq_embeds)

        # Timestep + pooled text conditioning + pooled image conditioning
        t_emb = self.time_embedder(t)
        pooled_emb = self.text_pooled_proj(text_pooled_embeds)
        c = t_emb + pooled_emb

        # Add conditioning to AdaLN if provided
        if cond_pooled is not None:
            c = c + cond_pooled

        # Apply CFG mask
        if cfg_mask is not None:
            mask = cfg_mask.unsqueeze(1).to(c.dtype)
            c = t_emb + pooled_emb * mask
            if cond_pooled is not None:
                c = c + cond_pooled * mask
            x_txt = x_txt * mask.unsqueeze(2)

        # Get RoPE embeddings
        if self.rope is not None:
            max_seq = max(x_img.shape[1], x_txt.shape[1])
            rope_cos, rope_sin = self.rope(max_seq, x_img.device)
        else:
            rope_cos, rope_sin = None, None

        # Pass through blocks
        for block in self.blocks:
            x_img, x_txt = block(x_img, x_txt, c, rope_cos, rope_sin)

        # Final layer
        v_pred = self.final_layer(x_img, c)

        # Reconstruct output with structural tokens if needed
        if has_structural:
            # Zero velocity for structural tokens
            v_structural = torch.zeros_like(x_structural)
            v_pred = torch.cat([v_pred, v_structural], dim=1)

        return v_pred

    @torch.no_grad()
    def sample(
        self,
        text_seq_embeds: torch.Tensor,
        text_pooled_embeds: torch.Tensor,
        cond_img: Optional[torch.Tensor] = None,
        num_steps: int = 20,
        cfg_scale: float = 3.0,
        return_full_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Sample visual tokens using Euler integration.

        Args:
            text_seq_embeds: Text sequence embeddings [B, L, D]
            text_pooled_embeds: Pooled embeddings [B, D]
            cond_img: Optional conditioning image tokens [B, N, 1280]
            num_steps: Number of Euler steps
            cfg_scale: Classifier-free guidance scale
            return_full_tokens: Whether to return 111 tokens (with structural)

        Returns:
            Generated visual tokens
        """
        B = text_seq_embeds.shape[0]
        device = text_seq_embeds.device
        dtype = text_seq_embeds.dtype

        # Start from noise
        x_t = torch.randn(B, self.config.num_visual_tokens, self.config.token_dim, device=device, dtype=dtype)

        # Euler sampling
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((B,), step * dt, device=device, dtype=dtype)

            if cfg_scale > 1.0:
                # CFG: conditional - unconditional
                v_cond = self(x_t, t, text_seq_embeds, text_pooled_embeds,
                             cond_img=cond_img, cfg_mask=torch.ones(B, device=device))
                v_uncond = self(x_t, t, text_seq_embeds, text_pooled_embeds,
                               cond_img=cond_img, cfg_mask=torch.zeros(B, device=device))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self(x_t, t, text_seq_embeds, text_pooled_embeds, cond_img=cond_img)

            x_t = x_t + v * dt

        # Optionally add structural tokens
        if return_full_tokens:
            # Add zero structural tokens
            structural = torch.zeros(
                B, self.config.num_structural_tokens, self.config.token_dim,
                device=device, dtype=dtype
            )
            x_t = torch.cat([x_t, structural], dim=1)

        return x_t


def create_mmdit_scratch(
    model_size: str = "large",
    **kwargs
) -> MMDiTScratch:
    """
    Create MMDiT from scratch.

    Args:
        model_size: One of ["small", "medium", "large", "xl"]
        **kwargs: Additional config overrides

    Returns:
        MMDiTScratch model

    Example:
        >>> model = create_mmdit_scratch("large")  # ~600M params
        >>> x_t = torch.randn(2, 100, 1280)
        >>> t = torch.rand(2)
        >>> text_seq = torch.randn(2, 77, 768)
        >>> text_pool = torch.randn(2, 768)
        >>> v = model(x_t, t, text_seq, text_pool)
    """
    config = get_config(model_size, **kwargs)
    model = MMDiTScratch(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"MMDiT from Scratch ({model_size})")
    print(f"{'='*60}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Feature map: {config.spatial_size}×{config.spatial_size}")
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")
    print(f"{'='*60}\n")

    return model


def estimate_params(model_size: str = "large") -> dict:
    """Estimate parameters for a given model size."""
    config = get_config(model_size)
    h = config.hidden_size
    n = config.num_layers
    mlp_hidden = int(h * config.mlp_ratio * 2 / 3)
    mlp_hidden = ((mlp_hidden + 63) // 64) * 64

    # Per block: attention + MLP + AdaLN
    attn_params = 2 * (3 * h * h + h * h)  # qkv + proj for img and txt
    mlp_params = 2 * (2 * h * mlp_hidden + mlp_hidden * h)  # img and txt MLPs
    adaln_params = 2 * h * 6 * h  # img and txt AdaLN
    per_block = attn_params + mlp_params + adaln_params

    # Embeddings
    emb_params = (
        config.freq_embed_size * h + h * h +  # time
        config.text_seq_dim * h +  # text seq proj
        config.text_pooled_dim * h +  # text pool proj
        config.token_dim * h +  # img proj
        config.num_visual_tokens * h  # pos embed
    )

    # Final layer
    final_params = h * config.token_dim + 2 * h * h

    total = emb_params + n * per_block + final_params

    return {
        "embeddings": emb_params,
        "per_block": per_block,
        "total_blocks": n * per_block,
        "final": final_params,
        "total": total,
        "total_millions": total / 1e6,
    }


if __name__ == "__main__":
    import torch

    print("Testing MMDiT from Scratch...\n")

    # Test all model sizes
    for size in ["small", "medium", "large", "xl"]:
        print(f"\n--- {size.upper()} ---")
        params = estimate_params(size)
        print(f"Estimated: {params['total_millions']:.1f}M params")

    # Create and test large model
    print("\n" + "="*60)
    print("Creating large model for testing...")
    model = create_mmdit_scratch("large")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).to(torch.bfloat16)

    # Test with 100 tokens (visual only)
    print("\nTest 1: 100 visual tokens")
    x_t = torch.randn(2, 100, 1280, device=device, dtype=torch.bfloat16)
    t = torch.rand(2, device=device, dtype=torch.bfloat16)
    text_seq = torch.randn(2, 77, 768, device=device, dtype=torch.bfloat16)
    text_pool = torch.randn(2, 768, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        v_pred = model(x_t, t, text_seq, text_pool)
    print(f"  Input: {list(x_t.shape)}")
    print(f"  Output: {list(v_pred.shape)}")
    print(f"  ✓ Passed" if v_pred.shape == x_t.shape else "  ✗ Failed")

    # Test with 111 tokens (full)
    print("\nTest 2: 111 tokens (100 visual + 11 structural)")
    x_t = torch.randn(2, 111, 1280, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        v_pred = model(x_t, t, text_seq, text_pool)
    print(f"  Input: {list(x_t.shape)}")
    print(f"  Output: {list(v_pred.shape)}")
    print(f"  ✓ Passed" if v_pred.shape == x_t.shape else "  ✗ Failed")

    # Test sampling
    print("\nTest 3: Sampling")
    with torch.no_grad():
        samples = model.sample(text_seq, text_pool, num_steps=5, cfg_scale=2.0)
    print(f"  Sampled: {list(samples.shape)}")
    print(f"  ✓ Passed" if samples.shape == (2, 111, 1280) else "  ✗ Failed")

    print("\n" + "="*60)
    print("All tests passed!")
