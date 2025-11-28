"""
Rectified Flow Loss for OCRFlow Training

Implements loss functions for training MMDiT with rectified flow matching.

Rectified flow uses a linear interpolation path:
    x_t = (1-t)*x_0 + t*x_1
    where x_0 ~ N(0, I) and x_1 = encoder(image)

The model predicts constant velocity: v = x_1 - x_0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class RectifiedFlowLoss(nn.Module):
    """
    Loss for rectified flow matching.

    The model predicts velocity v_theta(x_t, t, text) where:
        - x_t = (1-t)*x_0 + t*x_1 is the interpolated state
        - x_0 ~ N(0, I) is Gaussian noise
        - x_1 are the target image tokens
        - Ground truth velocity: v_true = x_1 - x_0

    Args:
        loss_type: Type of loss ('mse', 'l1', or 'huber')
        huber_delta: Delta parameter for Huber loss
    """

    def __init__(
        self,
        loss_type: Literal["mse", "l1", "huber"] = "huber",
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def forward(
        self,
        v_pred: torch.Tensor,  # [B, N, D] predicted velocity
        v_true: torch.Tensor,  # [B, N, D] ground truth velocity = x_1 - x_0
    ) -> torch.Tensor:
        """
        Compute loss between predicted and true velocity.

        Args:
            v_pred: Predicted velocity from model
            v_true: Ground truth velocity (x_1 - x_0)

        Returns:
            Scalar loss
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(v_pred, v_true, reduction="mean")
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_pred, v_true, reduction="mean")
        elif self.loss_type == "huber":
            loss = F.huber_loss(
                v_pred, v_true,
                reduction="mean",
                delta=self.huber_delta
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


def sample_timesteps(
    batch_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    sampling: Literal["uniform", "logit_normal"] = "uniform",
) -> torch.Tensor:
    """
    Sample timesteps for flow matching training.

    Args:
        batch_size: Number of timesteps to sample
        device: Device for tensors
        dtype: Tensor dtype
        sampling: Sampling strategy
            - 'uniform': Uniform sampling in [0, 1]
            - 'logit_normal': Logit-normal sampling (more focus on middle timesteps)

    Returns:
        Timesteps [B] in range [0, 1]
    """
    if sampling == "uniform":
        t = torch.rand(batch_size, device=device, dtype=dtype)
    elif sampling == "logit_normal":
        # Sample from logit-normal distribution
        # This concentrates samples around t=0.5
        z = torch.randn(batch_size, device=device, dtype=dtype)
        t = torch.sigmoid(z)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")

    return t


def compute_flow_matching_loss(
    model: nn.Module,
    x_0: torch.Tensor,  # [B, N, D] noise
    x_1: torch.Tensor,  # [B, N, D] target image tokens
    text_seq_embeds: torch.Tensor,  # [B, L, D_txt]
    text_pooled_embeds: torch.Tensor,  # [B, D_pool]
    loss_fn: RectifiedFlowLoss,
    cfg_dropout_prob: float = 0.1,
    timestep_sampling: str = "uniform",
) -> torch.Tensor:
    """
    Compute flow matching loss for a batch.

    Args:
        model: MMDiT model
        x_0: Gaussian noise
        x_1: Target image tokens (from encoder)
        text_seq_embeds: Text sequence embeddings
        text_pooled_embeds: Pooled text embeddings
        loss_fn: Loss function
        cfg_dropout_prob: Probability of dropping text conditioning (for CFG training)
        timestep_sampling: Timestep sampling strategy

    Returns:
        Scalar loss
    """
    B = x_0.shape[0]
    device = x_0.device
    dtype = x_0.dtype

    # Sample timesteps
    t = sample_timesteps(
        B, device=device, dtype=dtype, sampling=timestep_sampling
    )

    # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
    t_expand = t.view(B, 1, 1)  # [B, 1, 1] for broadcasting
    x_t = (1 - t_expand) * x_0 + t_expand * x_1

    # Ground truth velocity (constant in rectified flow)
    v_true = x_1 - x_0

    # Classifier-free guidance: randomly drop text conditioning
    cfg_mask = None
    if cfg_dropout_prob > 0:
        cfg_mask = (torch.rand(B, device=device) > cfg_dropout_prob).float()

    # Predict velocity
    v_pred = model(
        x_t=x_t,
        t=t,
        text_seq_embeds=text_seq_embeds,
        text_pooled_embeds=text_pooled_embeds,
        cfg_mask=cfg_mask,
    )

    # Compute loss
    loss = loss_fn(v_pred, v_true)

    return loss


if __name__ == "__main__":
    # Test loss functions
    print("Testing RectifiedFlowLoss...")

    # Create loss function
    loss_fn = RectifiedFlowLoss(loss_type="huber", huber_delta=1.0)

    # Test data
    B, N, D = 4, 1600, 1280
    v_pred = torch.randn(B, N, D)
    v_true = torch.randn(B, N, D)

    # Compute loss
    loss = loss_fn(v_pred, v_true)

    print(f"Predicted velocity shape: {v_pred.shape}")
    print(f"True velocity shape: {v_true.shape}")
    print(f"Loss value: {loss.item():.4f}")
    print("Test passed!")

    # Test timestep sampling
    print("\nTesting timestep sampling...")
    t_uniform = sample_timesteps(100, device="cpu", sampling="uniform")
    t_logit = sample_timesteps(100, device="cpu", sampling="logit_normal")

    print(f"Uniform t range: [{t_uniform.min():.3f}, {t_uniform.max():.3f}]")
    print(f"Logit-normal t range: [{t_logit.min():.3f}, {t_logit.max():.3f}]")
    print(f"Logit-normal t mean: {t_logit.mean():.3f} (should be ~0.5)")
    print("Sampling test passed!")
