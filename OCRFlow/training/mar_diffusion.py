"""
MAR (Masked Autoregressive) Training with Diffusion Loss for Vision Tokens

Based on "Autoregressive Image Generation without Vector Quantization" (Kaiming He et al.)
Properly adapted using DiffLoss for continuous vision token prediction.

Key Architecture (from original MAR):
1. Random masking (70-100% of tokens)
2. Encoder processes only UNMASKED tokens
3. Decoder reconstructs full sequence with mask tokens
4. DiffLoss predicts masked tokens via diffusion

For OCRFlow visual tokens [B, 111, 1280]:
- Treat as continuous latents (like VAE latents in original MAR)
- Use diffusion to model distribution of masked tokens
- Condition on encoded context from unmasked tokens
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import random
import numpy as np
import scipy.stats as stats

from .diffloss import DiffLoss


class MARDiffusionTraining:
    """
    MAR training with proper diffusion loss for visual tokens.

    Follows original MAR architecture:
    - Random masking (high ratio: 70-100%)
    - Process only unmasked tokens
    - Diffusion loss to predict masked tokens
    """

    def __init__(
        self,
        token_dim: int = 1280,
        summary_dim: int = 512,
        num_visual_tokens: int = 100,  # Only pure image features (exclude newlines+separator)
        mask_ratio_min: float = 0.7,  # MAR uses high mask ratios
        loss_weight: float = 1.0,
        diffloss_depth: int = 3,
        diffloss_width: int = 1024,
        num_sampling_steps: str = "100",
        diffusion_batch_mul: int = 4,
    ):
        """
        Args:
            token_dim: Dimension of visual tokens (1280 for DeepSeek-OCR)
            summary_dim: Dimension of encoded summary from chunk_encoder
            num_visual_tokens: Number of pure visual tokens (100 for 10x10 grid)
            mask_ratio_min: Minimum mask ratio (MAR paper uses 0.7-1.0)
            loss_weight: Weight for MAR loss
            diffloss_depth: Depth of diffusion MLP
            diffloss_width: Width of diffusion MLP
            num_sampling_steps: Number of diffusion steps for generation
            diffusion_batch_mul: Batch multiplier for diffusion training
        """
        self.token_dim = token_dim
        self.summary_dim = summary_dim
        self.num_visual_tokens = num_visual_tokens
        self.loss_weight = loss_weight
        self.diffusion_batch_mul = diffusion_batch_mul

        # Truncated Gaussian for mask ratio (centered at 100% with std 0.25)
        # This is the key from MAR paper
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25,
            0,
            loc=1.0,
            scale=0.25
        )

        # Diffusion loss to predict masked tokens
        self.diffloss = DiffLoss(
            target_channels=token_dim,  # Predict 1280-dim tokens
            z_channels=summary_dim,  # Condition on summary_dim from encoder
            depth=diffloss_depth,
            width=diffloss_width,
            num_sampling_steps=num_sampling_steps,
        )

    def extract_visual_tokens(self, full_tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract pure visual tokens from full 111-token sequence.

        Full sequence structure (111 tokens):
        - Tokens 0-109: Image features (10x10 grid) + newlines (interleaved)
          - Each row: 10 image tokens + 1 newline = 11 tokens per row
          - 10 rows × 11 = 110 tokens
        - Token 110: View separator

        We extract only the 100 pure image features (10×10 grid).

        Args:
            full_tokens: [B, 111, D] full visual tokens

        Returns:
            visual_tokens: [B, 100, D] pure image features only
        """
        B, N, D = full_tokens.shape
        assert N == 111, f"Expected 111 tokens, got {N}"

        # Extract pure visual tokens (skip newlines and separator)
        # Pattern: Take first 10 tokens from each of 10 rows
        visual_indices = []
        for row in range(10):
            row_start = row * 11  # Each row starts at 0, 11, 22, ..., 99
            visual_indices.extend(range(row_start, row_start + 10))

        visual_indices = torch.tensor(visual_indices, device=full_tokens.device)  # [100]
        visual_tokens = full_tokens[:, visual_indices, :]  # [B, 100, D]

        return visual_tokens

    def reconstruct_full_tokens(
        self,
        visual_tokens: torch.Tensor,
        newline_token: torch.Tensor,
        separator_token: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct full 111-token sequence from 100 visual tokens.

        Args:
            visual_tokens: [B, 100, D] pure image features
            newline_token: [D] learned newline embedding
            separator_token: [D] learned separator embedding

        Returns:
            full_tokens: [B, 111, D] full sequence
        """
        B, N, D = visual_tokens.shape
        assert N == 100, f"Expected 100 visual tokens, got {N}"

        device = visual_tokens.device
        full_tokens = torch.zeros(B, 111, D, device=device, dtype=visual_tokens.dtype)

        # Reshape visual tokens to grid
        visual_grid = visual_tokens.view(B, 10, 10, D)  # [B, 10, 10, D]

        # Insert visual tokens and newlines
        for row in range(10):
            row_start = row * 11
            full_tokens[:, row_start:row_start+10, :] = visual_grid[:, row, :, :]  # Image features
            full_tokens[:, row_start+10, :] = newline_token.unsqueeze(0).expand(B, -1)  # Newline

        # Add separator
        full_tokens[:, 110, :] = separator_token.unsqueeze(0).expand(B, -1)

        return full_tokens

    def random_masking(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random mask following MAR paper.

        Args:
            x: [B, N, D] visual tokens (N=100 pure visual tokens)

        Returns:
            mask: [B, N] binary mask (1 = masked, 0 = visible)
            orders: [B, N] random prediction order for each sample
        """
        B, N, D = x.shape
        device = x.device
        assert N == self.num_visual_tokens, f"Expected {self.num_visual_tokens} tokens, got {N}"

        # Sample mask ratio from truncated Gaussian
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked = int(np.ceil(N * mask_rate))

        # Random order for each sample
        mask = torch.zeros(B, N, device=device)
        orders = torch.zeros(B, N, dtype=torch.long, device=device)

        for b in range(B):
            # Random shuffle
            order = torch.randperm(N, device=device)
            orders[b] = order

            # Mask first num_masked tokens in random order
            mask[b, order[:num_masked]] = 1

        return mask, orders

    def forward_loss(
        self,
        model,
        full_visual_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute MAR diffusion loss on pure visual tokens (excluding newlines/separator).

        Args:
            model: MarkovianChunkDecoder with chunk_encoder and chunk_decoder
            full_visual_tokens: [B, 111, D] full visual tokens with newlines+separator

        Returns:
            loss: Diffusion loss on masked tokens
            metrics: Dictionary with loss breakdown
        """
        B, N_full, D = full_visual_tokens.shape
        assert N_full == 111, f"Expected 111 tokens, got {N_full}"
        assert D == self.token_dim, f"Token dim mismatch: {D} vs {self.token_dim}"

        # 1. Extract only pure visual tokens (100 tokens, no newlines/separator)
        visual_tokens = self.extract_visual_tokens(full_visual_tokens)  # [B, 100, D]

        # 2. Generate random mask (70-100% masked)
        mask, orders = self.random_masking(visual_tokens)  # [B, 100]

        # Create masked visual tokens (zero out masked positions)
        masked_visual = visual_tokens * (1 - mask.unsqueeze(-1))  # [B, 100, D]

        # 3. Encoder: Process masked input with full sequence structure
        # Reconstruct full 111-token sequence with masked visual tokens + original structural tokens
        # Extract original structural tokens (newlines and separator)
        newline_indices = [row * 11 + 10 for row in range(10)]  # Indices 10, 21, 32, ..., 109
        separator_index = 110

        # Create full masked sequence
        masked_full = full_visual_tokens.clone()

        # Replace pure visual tokens with masked versions
        for row in range(10):
            row_start = row * 11
            masked_full[:, row_start:row_start+10, :] = masked_visual[:, row*10:(row+1)*10, :]

        # Encode full masked sequence (chunk_encoder: [B, 111, D] → [B, summary_dim])
        summary = model.chunk_encoder(masked_full)  # [B, summary_dim]

        # 4. Decoder: Reconstruct full visual tokens
        # Decode summary back to full 111-token sequence
        decoder_output_full = model.chunk_decoder(summary)  # [B, 111, D]

        # Extract only the pure visual tokens from decoder output
        decoder_output = self.extract_visual_tokens(decoder_output_full)  # [B, 100, D]

        # 5. DiffLoss: Predict masked visual tokens via diffusion
        # Flatten batch and sequence for diffusion
        target_flat = visual_tokens.reshape(B * 100, D)  # [B*100, D]
        z_flat = decoder_output.reshape(B * 100, D)  # [B*100, D]
        mask_flat = mask.reshape(B * 100)  # [B*100]

        # Repeat for diffusion batch multiplication
        target_flat = target_flat.repeat(self.diffusion_batch_mul, 1)  # [B*100*mul, D]
        z_flat = z_flat.repeat(self.diffusion_batch_mul, 1)  # [B*100*mul, D]
        mask_flat = mask_flat.repeat(self.diffusion_batch_mul)  # [B*100*mul]

        # Diffusion loss (only on masked positions)
        diffloss = self.diffloss(target=target_flat, z=z_flat, mask=mask_flat)

        # Metrics
        metrics = {
            'mar_diffloss': diffloss.item(),
            'mask_ratio': mask.mean().item(),
            'num_masked': mask.sum().item(),
            'num_visual_tokens': 100,
        }

        return diffloss * self.loss_weight, metrics


def example_usage():
    """Example of integrating MAR diffusion training"""
    print("Example MAR Diffusion Training Setup:")
    print("=" * 60)

    # Assume we have:
    # - model: MarkovianChunkDecoder with chunk_encoder & chunk_decoder
    # - visual_tokens: [B, 111, 1280] from vision encoder

    # Create MAR trainer
    mar = MARDiffusionTraining(
        token_dim=1280,
        summary_dim=512,  # Adjust based on your model
        mask_ratio_min=0.7,
        loss_weight=0.1,
        diffloss_depth=3,
        diffloss_width=1024,
    )

    print(f"MAR Configuration:")
    print(f"  Token dim: {mar.token_dim}")
    print(f"  Summary dim: {mar.summary_dim}")
    print(f"  Mask ratio: {mar.mask_ratio_generator.mean():.2f} ± {mar.mask_ratio_generator.std():.2f}")
    print(f"  Loss weight: {mar.loss_weight}")
    print()

    # In training loop:
    # chunk_loss, metrics = model.compute_loss(chunk_sequences)
    #
    # if enable_mar:
    #     mar_loss, mar_metrics = mar.forward_loss(
    #         model=model,
    #         visual_tokens=inputs,  # [B, 111, 1280]
    #     )
    #     total_loss = chunk_loss + mar_loss
    #     metrics.update(mar_metrics)
    # else:
    #     total_loss = chunk_loss

    print("Integration with training:")
    print("  1. Compute standard chunk loss")
    print("  2. Compute MAR diffusion loss on visual tokens")
    print("  3. Combine: total_loss = chunk_loss + mar_loss")
    print()
    print("Expected behavior:")
    print("  - High mask ratio (70-100%)")
    print("  - Diffusion predicts masked tokens")
    print("  - Improves encoder/decoder representations")


if __name__ == "__main__":
    example_usage()
