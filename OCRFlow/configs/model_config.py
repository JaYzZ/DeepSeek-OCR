"""
OCRFlow Model Configuration

Defines model architecture configurations for different MMDiT sizes,
adapted from UniFlow for OCR image token generation.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Configuration for MMDiT OCRFlow model"""

    # Model size identifier
    model_size: Literal["tiny", "small", "base", "large", "xl"] = "small"

    # Image token configuration (from DeepSeek OCR)
    # Image tokens are [B, num_tokens, token_dim]
    # For DeepSeek OCR with 640x640 image and 16px patches with 4x downsample:
    # num_tokens = (640/16/4) * (640/16/4) = 40 * 40 = 1600 tokens
    num_image_tokens: int = 1600  # Number of image tokens per image
    token_dim: int = 1280  # DeepSeek OCR projection dimension

    # Text encoder configuration
    use_dual_text_encoders: bool = True  # T5 + CLIP like UniFlow
    t5_model: str = "google/t5-v1_1-base"  # or DistillT5
    clip_model: str = "openai/clip-vit-large-patch14"  # or SigLIP2
    t5_seq_len: int = 512
    t5_dim: int = 768
    clip_pooled_dim: int = 768

    # MMDiT architecture (based on model_size)
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # Attention configuration
    use_flash_attn: bool = True
    qkv_bias: bool = True
    qk_norm: bool = True

    # Normalization and activation
    norm_type: Literal["layer_norm", "rms_norm"] = "layer_norm"
    norm_eps: float = 1e-6
    activation: str = "gelu"

    # Timestep embedding
    time_embed_dim: Optional[int] = None  # If None, uses hidden_size
    freq_embed_size: int = 256

    # Flow matching configuration
    num_sampling_steps: int = 20  # Euler sampling steps for inference
    cfg_scale: float = 3.0  # Classifier-free guidance scale

    # Dropout
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # Initialization
    init_std: float = 0.02

    def __post_init__(self):
        """Set time_embed_dim if not provided"""
        if self.time_embed_dim is None:
            self.time_embed_dim = self.hidden_size


# Predefined model size configurations (similar to UniFlow)
MODEL_CONFIGS = {
    "tiny": {
        "hidden_size": 384,
        "num_layers": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "small": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "base": {
        "hidden_size": 1024,
        "num_layers": 18,
        "num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "large": {
        "hidden_size": 1536,
        "num_layers": 24,
        "num_heads": 24,
        "mlp_ratio": 4.0,
    },
    "xl": {
        "hidden_size": 1536,
        "num_layers": 30,
        "num_heads": 24,
        "mlp_ratio": 4.0,
    },
}


def get_model_config(
    model_size: str = "small",
    num_image_tokens: int = 1600,
    token_dim: int = 1280,
    **kwargs
) -> ModelConfig:
    """
    Get model configuration for specified size.

    Args:
        model_size: One of ["tiny", "small", "base", "large", "xl"]
        num_image_tokens: Number of image tokens (depends on image size and patch size)
        token_dim: Dimension of image tokens (DeepSeek OCR projection dim)
        **kwargs: Additional config overrides

    Returns:
        ModelConfig instance

    Example:
        >>> config = get_model_config("small", num_image_tokens=1600, token_dim=1280)
        >>> config.hidden_size
        768
        >>> config.num_params_millions()
        350
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"model_size must be one of {list(MODEL_CONFIGS.keys())}, got {model_size}"
        )

    # Get base config for model size
    size_config = MODEL_CONFIGS[model_size].copy()

    # Merge with custom kwargs
    size_config.update(kwargs)

    # Create config
    config = ModelConfig(
        model_size=model_size,
        num_image_tokens=num_image_tokens,
        token_dim=token_dim,
        **size_config
    )

    return config


def estimate_model_params(config: ModelConfig) -> dict:
    """
    Estimate model parameters for MMDiT.

    Args:
        config: ModelConfig instance

    Returns:
        Dictionary with parameter counts
    """
    h = config.hidden_size
    n_layers = config.num_layers
    mlp_hidden = int(h * config.mlp_ratio)

    # Embedding projections
    token_proj_params = config.token_dim * h  # Project image tokens to hidden
    time_embed_params = config.freq_embed_size * config.time_embed_dim * 2  # Sin/cos + MLP

    # Text encoder projections (if dual encoders)
    if config.use_dual_text_encoders:
        t5_proj_params = config.t5_dim * h
        clip_proj_params = config.clip_pooled_dim * h
    else:
        t5_proj_params = clip_proj_params = 0

    # Per-layer parameters
    # AdaLN modulation: 6 * hidden (scale/shift for QKV, MLP, etc.)
    adaln_params = 6 * h * config.time_embed_dim

    # Self-attention: QKV + output projection
    attn_params = 4 * h * h

    # MLP: two linear layers
    mlp_params = h * mlp_hidden + mlp_hidden * h

    # Layer norm parameters (negligible, but included for completeness)
    ln_params = 4 * h  # 2 layer norms per block

    per_layer_params = adaln_params + attn_params + mlp_params + ln_params
    total_layer_params = per_layer_params * n_layers

    # Final layer: predict velocity
    final_params = h * config.token_dim

    # Total
    total_params = (
        token_proj_params +
        time_embed_params +
        t5_proj_params +
        clip_proj_params +
        total_layer_params +
        final_params
    )

    return {
        "total": total_params,
        "total_millions": total_params / 1e6,
        "embedding": token_proj_params + time_embed_params + t5_proj_params + clip_proj_params,
        "transformer_layers": total_layer_params,
        "final_layer": final_params,
    }


if __name__ == "__main__":
    # Test configurations
    for size in ["tiny", "small", "base", "large", "xl"]:
        config = get_model_config(size)
        params = estimate_model_params(config)
        print(f"\n{size.upper()} Model:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Heads: {config.num_heads}")
        print(f"  Parameters: {params['total_millions']:.1f}M")
