"""
MAR (Masked Autoregressive) Training for Vision Tokens

Based on "Autoregressive Image Generation without Vector Quantization" (Kaiming He et al.)
Adapted for continuous vision embeddings without discrete codebook.

Key Ideas:
1. Randomly mask a subset of visual tokens
2. Predict masked tokens autoregressively in random order
3. Use L2 loss on continuous embeddings (no VQ needed)
4. Provides auxiliary self-supervised signal alongside next-token prediction

Architecture:
    Input: Visual tokens [B, 111, 1280]
    ↓
    Random masking (e.g., 30-70% tokens)
    ↓
    Autoregressive prediction with causal attention
    ↓
    L2 loss between predicted and target embeddings

Benefits:
- Self-supervised signal from vision modality
- No text needed (pure vision learning)
- Complements next-token text prediction
- Better vision representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import random


class MARMaskedPrediction:
    """
    Masked Autoregressive Prediction for vision tokens.

    Follows MAR paper but adapted for continuous embeddings:
    - No VQ codebook required
    - Direct L2 regression on embedding space
    - Random masking ratios for robustness
    """

    def __init__(
        self,
        mask_ratio_min: float = 0.3,
        mask_ratio_max: float = 0.7,
        loss_weight: float = 0.1,
        use_random_order: bool = True,
    ):
        """
        Args:
            mask_ratio_min: Minimum fraction of tokens to mask
            mask_ratio_max: Maximum fraction of tokens to mask
            loss_weight: Weight for MAR loss (vs main next-token loss)
            use_random_order: Whether to predict in random order (True) or sequential
        """
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.loss_weight = loss_weight
        self.use_random_order = use_random_order

    def create_masked_targets(
        self,
        visual_tokens: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masked input and targets for MAR training.

        Args:
            visual_tokens: [B, N, D] vision embeddings
            mask_ratio: Fraction to mask (random if None)

        Returns:
            masked_tokens: [B, N, D] with masked positions zeroed
            mask: [B, N] binary mask (1 = masked, 0 = visible)
            prediction_order: [B, N] order to predict masked tokens
        """
        B, N, D = visual_tokens.shape
        device = visual_tokens.device

        # Random mask ratio per batch
        if mask_ratio is None:
            mask_ratio = random.uniform(self.mask_ratio_min, self.mask_ratio_max)

        num_masked = int(N * mask_ratio)

        # Create random mask for each sample in batch
        mask = torch.zeros(B, N, device=device)
        masked_tokens = visual_tokens.clone()
        prediction_order = torch.zeros(B, N, dtype=torch.long, device=device)

        for b in range(B):
            # Random positions to mask
            masked_indices = torch.randperm(N, device=device)[:num_masked]
            mask[b, masked_indices] = 1

            # Zero out masked positions
            masked_tokens[b, masked_indices] = 0

            # Random prediction order (key to MAR)
            if self.use_random_order:
                order = torch.randperm(N, device=device)
            else:
                order = torch.arange(N, device=device)

            prediction_order[b] = order

        return masked_tokens, mask, prediction_order

    def compute_mar_loss(
        self,
        model,
        visual_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute MAR loss for masked token prediction.

        Args:
            model: Language model with vision input
            visual_tokens: [B, N, D] original vision embeddings
            text_tokens: Optional text tokens (not used in pure vision MAR)

        Returns:
            loss: Scalar MAR loss
            metrics: Dict with detailed metrics
        """
        B, N, D = visual_tokens.shape

        # Create masked input
        masked_tokens, mask, prediction_order = self.create_masked_targets(visual_tokens)

        # Get model's predicted embeddings
        # Assuming model has a vision_projection or similar
        # This part depends on your model architecture
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Forward pass with masked tokens
            # NOTE: This is pseudo-code - adapt to your model API
            if hasattr(model, 'predict_vision_tokens'):
                predicted_tokens = model.predict_vision_tokens(masked_tokens, mask)
            else:
                # Fallback: use hidden states from forward pass
                outputs = model(inputs_embeds=masked_tokens, output_hidden_states=True)
                predicted_tokens = outputs.hidden_states[-1][:, :N, :]  # [B, N, D]

        # Compute L2 loss only on masked positions
        # Shape: [B, N, D]
        diff = predicted_tokens - visual_tokens
        diff_squared = diff ** 2

        # Mask loss (only compute for masked tokens)
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        masked_loss = (diff_squared * mask_expanded).sum() / (mask.sum() * D + 1e-8)

        # Metrics
        num_masked = mask.sum().item()
        metrics = {
            'mar_loss': masked_loss.item(),
            'num_masked_tokens': num_masked,
            'mask_ratio': num_masked / (B * N),
            'mse_per_token': masked_loss.item() / (num_masked / (B * N) + 1e-8),
        }

        return masked_loss * self.loss_weight, metrics

    @torch.no_grad()
    def evaluate_reconstruction(
        self,
        model,
        visual_tokens: torch.Tensor,
        num_samples: int = 4,
    ) -> dict:
        """
        Evaluate MAR reconstruction quality.

        Returns metrics on how well the model can reconstruct masked tokens.
        """
        B, N, D = visual_tokens.shape

        masked_tokens, mask, _ = self.create_masked_targets(
            visual_tokens[:num_samples],
            mask_ratio=0.5
        )

        # Get predictions
        if hasattr(model, 'predict_vision_tokens'):
            predicted_tokens = model.predict_vision_tokens(masked_tokens, mask)
        else:
            outputs = model(inputs_embeds=masked_tokens, output_hidden_states=True)
            predicted_tokens = outputs.hidden_states[-1][:, :N, :]

        # Compute reconstruction error
        mse = F.mse_loss(predicted_tokens[mask == 1], visual_tokens[:num_samples][mask == 1])

        # Cosine similarity
        pred_flat = predicted_tokens[mask == 1]
        target_flat = visual_tokens[:num_samples][mask == 1]
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()

        return {
            'reconstruction_mse': mse.item(),
            'reconstruction_cosine_sim': cos_sim.item(),
        }


class MARTrainingWrapper:
    """
    Wrapper to combine standard next-token prediction with MAR.

    Usage:
        mar_wrapper = MARTrainingWrapper(model, mar_config)

        for batch in dataloader:
            loss, metrics = mar_wrapper.compute_combined_loss(
                visual_tokens=batch['visual_tokens'],
                text_tokens=batch['text_tokens'],
                labels=batch['labels'],
            )
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        mar_loss_weight: float = 0.1,
        mask_ratio_min: float = 0.3,
        mask_ratio_max: float = 0.7,
        enable_mar: bool = True,
    ):
        self.model = model
        self.enable_mar = enable_mar

        if enable_mar:
            self.mar = MARMaskedPrediction(
                mask_ratio_min=mask_ratio_min,
                mask_ratio_max=mask_ratio_max,
                loss_weight=mar_loss_weight,
            )

    def compute_combined_loss(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss: next-token prediction + MAR.

        Args:
            visual_tokens: [B, N_vis, D] vision embeddings
            text_tokens: [B, N_text] text token IDs
            labels: [B, N_text] target labels

        Returns:
            total_loss: Combined loss
            metrics: Dict with breakdown
        """
        # Standard next-token prediction loss
        outputs = self.model(
            inputs_embeds=visual_tokens,
            input_ids=text_tokens,
            labels=labels,
        )

        lm_loss = outputs.loss
        metrics = {'lm_loss': lm_loss.item()}

        # Add MAR loss if enabled
        if self.enable_mar:
            mar_loss, mar_metrics = self.mar.compute_mar_loss(
                self.model,
                visual_tokens,
            )
            total_loss = lm_loss + mar_loss
            metrics.update(mar_metrics)
            metrics['total_loss'] = total_loss.item()
        else:
            total_loss = lm_loss
            metrics['total_loss'] = lm_loss.item()

        return total_loss, metrics


# Example integration with training loop
def example_training_loop():
    """
    Example of how to integrate MAR into training.
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # Initialize model and MAR wrapper
    # model = YourOCRModel(...)
    # mar_wrapper = MARTrainingWrapper(
    #     model=model,
    #     mar_loss_weight=0.1,  # 10% weight for MAR loss
    #     mask_ratio_min=0.3,
    #     mask_ratio_max=0.7,
    # )

    # Training loop
    # for step, batch in enumerate(dataloader):
    #     visual_tokens = encoder.encode_texts(batch['texts'])  # [B, 111, 1280]
    #
    #     loss, metrics = mar_wrapper.compute_combined_loss(
    #         visual_tokens=visual_tokens,
    #         text_tokens=batch['input_ids'],
    #         labels=batch['labels'],
    #     )
    #
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #
    #     if step % 100 == 0:
    #         logger.info(f"Step {step}: "
    #                    f"Total={metrics['total_loss']:.4f}, "
    #                    f"LM={metrics['lm_loss']:.4f}, "
    #                    f"MAR={metrics.get('mar_loss', 0):.4f}")

    pass


if __name__ == "__main__":
    # Quick test
    print("Testing MAR implementation...")

    # Create dummy visual tokens
    B, N, D = 4, 111, 1280
    visual_tokens = torch.randn(B, N, D)

    # Test masking
    mar = MARMaskedPrediction(mask_ratio_min=0.5, mask_ratio_max=0.5)
    masked, mask, order = mar.create_masked_targets(visual_tokens)

    print(f"Visual tokens: {visual_tokens.shape}")
    print(f"Masked tokens: {masked.shape}")
    print(f"Mask: {mask.shape}, Masked ratio: {mask.mean():.2f}")
    print(f"Prediction order: {order.shape}")

    # Verify masking
    assert (masked[mask == 1] == 0).all(), "Masked positions should be zero"
    assert (masked[mask == 0] == visual_tokens[mask == 0]).all(), "Unmasked should match"

    print("✓ MAR implementation test passed!")
