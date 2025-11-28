"""
Euler Sampling for OCRFlow Inference

Implements Euler method for sampling from rectified flow models.
"""

import torch
from typing import Optional, Callable
from tqdm import tqdm


@torch.no_grad()
def euler_sampling(
    model: torch.nn.Module,
    text_seq_embeds: torch.Tensor,
    text_pooled_embeds: torch.Tensor,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    x_init: Optional[torch.Tensor] = None,
    num_tokens: int = 1600,
    token_dim: int = 1280,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
    callback: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Sample image tokens using Euler method for rectified flow.

    The rectified flow ODE is: dx/dt = v_theta(x_t, t, text)
    where v is the velocity predicted by the model.

    Args:
        model: MMDiT model
        text_seq_embeds: Text sequence embeddings [B, L, D_t5]
        text_pooled_embeds: Pooled text embeddings [B, D_clip]
        num_steps: Number of Euler integration steps
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
        x_init: Initial noise [B, N, D]. If None, sample from N(0, I)
        num_tokens: Number of image tokens (only used if x_init is None)
        token_dim: Token dimension (only used if x_init is None)
        device: Device for computation
        dtype: Data type
        verbose: Whether to show progress bar
        callback: Optional callback function called after each step with (step, x_t)

    Returns:
        Generated image tokens [B, num_tokens, token_dim]

    Example:
        >>> from OCRFlow.models import create_mmdit_ocrflow
        >>> from OCRFlow.models.text_encoders import load_text_encoders, encode_text_dual
        >>>
        >>> # Load model and text encoders
        >>> model = create_mmdit_ocrflow("small").cuda().eval()
        >>> encoders = load_text_encoders()
        >>>
        >>> # Encode text
        >>> prompts = ["<image>\n<|grounding|>Convert the document to markdown."]
        >>> t5_emb, clip_emb, _ = encode_text_dual(prompts, **encoders)
        >>>
        >>> # Sample
        >>> tokens = euler_sampling(
        ...     model, t5_emb, clip_emb, num_steps=20, cfg_scale=3.0
        ... )
    """
    model.eval()
    B = text_seq_embeds.shape[0]

    # Initialize from noise if not provided
    if x_init is None:
        x_t = torch.randn(
            B, num_tokens, token_dim,
            device=device,
            dtype=dtype
        )
    else:
        x_t = x_init.to(device).to(dtype)

    # Time step size
    dt = 1.0 / num_steps

    # Euler integration
    iterator = range(num_steps)
    if verbose:
        iterator = tqdm(iterator, desc="Sampling", unit="step")

    for step in iterator:
        # Current timestep
        t = torch.full((B,), step * dt, device=device, dtype=dtype)

        if cfg_scale > 1.0:
            # Conditional prediction
            v_cond = model(
                x_t, t, text_seq_embeds, text_pooled_embeds,
                cfg_mask=torch.ones(B, device=device)
            )

            # Unconditional prediction
            v_uncond = model(
                x_t, t, text_seq_embeds, text_pooled_embeds,
                cfg_mask=torch.zeros(B, device=device)
            )

            # Classifier-free guidance
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            # No guidance
            v = model(x_t, t, text_seq_embeds, text_pooled_embeds)

        # Euler step: x_{t+dt} = x_t + v * dt
        x_t = x_t + v * dt

        # Optional callback
        if callback is not None:
            callback(step, x_t)

    return x_t


@torch.no_grad()
def heun_sampling(
    model: torch.nn.Module,
    text_seq_embeds: torch.Tensor,
    text_pooled_embeds: torch.Tensor,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    x_init: Optional[torch.Tensor] = None,
    num_tokens: int = 1600,
    token_dim: int = 1280,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Sample using Heun's method (2nd order Runge-Kutta).

    More accurate than Euler but requires 2x model evaluations.

    Args:
        (same as euler_sampling)

    Returns:
        Generated image tokens [B, num_tokens, token_dim]
    """
    model.eval()
    B = text_seq_embeds.shape[0]

    # Initialize from noise if not provided
    if x_init is None:
        x_t = torch.randn(
            B, num_tokens, token_dim,
            device=device,
            dtype=dtype
        )
    else:
        x_t = x_init.to(device).to(dtype)

    dt = 1.0 / num_steps

    iterator = range(num_steps)
    if verbose:
        iterator = tqdm(iterator, desc="Sampling (Heun)", unit="step")

    for step in iterator:
        t = torch.full((B,), step * dt, device=device, dtype=dtype)

        # Helper function to get velocity with CFG
        def get_velocity(x, t):
            if cfg_scale > 1.0:
                v_cond = model(
                    x, t, text_seq_embeds, text_pooled_embeds,
                    cfg_mask=torch.ones(B, device=device)
                )
                v_uncond = model(
                    x, t, text_seq_embeds, text_pooled_embeds,
                    cfg_mask=torch.zeros(B, device=device)
                )
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                return model(x, t, text_seq_embeds, text_pooled_embeds)

        # First prediction (Euler step)
        v1 = get_velocity(x_t, t)
        x_euler = x_t + v1 * dt

        # Second prediction at next timestep
        t_next = torch.full((B,), (step + 1) * dt, device=device, dtype=dtype)
        v2 = get_velocity(x_euler, t_next)

        # Heun's method: average the two slopes
        x_t = x_t + 0.5 * (v1 + v2) * dt

    return x_t


if __name__ == "__main__":
    print("Testing sampling functions...")

    # Mock model for testing
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1280, 1280)

        def forward(self, x_t, t, text_seq, text_pool, cfg_mask=None):
            # Simple mock: just return small perturbation
            return torch.randn_like(x_t) * 0.1

    model = MockModel().cuda().eval()

    # Test data
    text_seq = torch.randn(2, 512, 768).cuda().bfloat16()
    text_pool = torch.randn(2, 768).cuda().bfloat16()

    # Test Euler sampling
    print("\nTesting Euler sampling...")
    samples_euler = euler_sampling(
        model, text_seq, text_pool,
        num_steps=5,
        cfg_scale=1.0,
        num_tokens=100,
        token_dim=1280,
        verbose=False
    )
    print(f"  Output shape: {samples_euler.shape}")
    print(f"  Expected: [2, 100, 1280]")
    assert samples_euler.shape == (2, 100, 1280)
    print("  ✓ Euler sampling test passed")

    # Test Heun sampling
    print("\nTesting Heun sampling...")
    samples_heun = heun_sampling(
        model, text_seq, text_pool,
        num_steps=5,
        cfg_scale=1.0,
        num_tokens=100,
        token_dim=1280,
        verbose=False
    )
    print(f"  Output shape: {samples_heun.shape}")
    assert samples_heun.shape == (2, 100, 1280)
    print("  ✓ Heun sampling test passed")

    print("\nAll sampling tests passed!")
