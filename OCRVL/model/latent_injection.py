"""
Latent Token Injection Module

This module provides utilities for injecting pre-encoded OCR features at specific
latent token positions during the forward pass. This enables minimal-change thinking
training where:

1. Data format uses special latent tokens: <think><|latent_step|>*k</think>
2. Pre-encoded OCR features are provided via latent_supervision parameter
3. During forward pass, latent token embeddings are replaced with OCR features
4. Existing thinking_projection MLP provides reconstruction supervision

OCR Feature Format:
    - Each latent supervision chunk is [100, 1280] (10×10 grid, 1280-dim)
    - Mean-pooled to [1280] before projection through OCR connector
    - Projected to [hidden_dim] (e.g., 4096) via existing OCR connector

Key Functions:
    inject_latent_features(): Replace latent token embeddings with OCR features
    find_latent_positions(): Locate latent tokens in input_ids
    prepare_latent_supervision(): Format OCR features for injection

Usage:
    # In dataset collate function:
    batch = {
        'input_ids': ...,
        'latent_supervision': [[feat1, feat2], [feat3]],  # Variable per sample
        'latent_positions': tensor,  # Boolean mask
        ...
    }

    # In model forward():
    inputs_embeds = self.get_input_embeddings()(input_ids)
    inputs_embeds = inject_latent_features(
        inputs_embeds,
        latent_supervision,
        latent_positions,
        ocr_connector=self.ocr_connector,
    )
"""

import logging
from typing import List, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def find_latent_positions(
    input_ids: torch.LongTensor,
    latent_token_id: int,
    start_token_id: int,
    end_token_id: int,
) -> torch.BoolTensor:
    """Find positions of latent step tokens between thinking start/end markers.

    Format: <think><|latent_step|>[<|thinking_sep|><|latent_step|>]*</think>

    This is simpler than the old format - we just need to find all <|latent_step|> tokens
    that appear between <think> and </think>.

    Args:
        input_ids: [batch_size, seq_len] Input token IDs
        latent_token_id: Token ID for <|latent_step|>
        start_token_id: Token ID for <think>
        end_token_id: Token ID for </think>

    Returns:
        [batch_size, seq_len] Boolean mask where True indicates a latent step position

    Example with 3 steps:
        Input:  [Q, <think>, <|latent_step|>, <|thinking_sep|>, <|latent_step|>, <|thinking_sep|>, <|latent_step|>, </think>, A]
        Output: [F,       F,               T,                F,               T,                F,               T,        F, F]
    """
    batch_size, seq_len = input_ids.shape
    latent_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for b in range(batch_size):
        # Find start token position
        start_positions = (input_ids[b] == start_token_id).nonzero(as_tuple=False)

        # Find end token position
        end_positions = (input_ids[b] == end_token_id).nonzero(as_tuple=False)

        if len(start_positions) == 0 or len(end_positions) == 0:
            continue

        start_pos = start_positions[0].item()
        end_pos = end_positions[0].item()

        # Mark all latent_step tokens between start and end
        if start_pos < end_pos:
            in_thinking_section = input_ids[b, start_pos+1:end_pos] == latent_token_id
            latent_mask[b, start_pos+1:end_pos] = in_thinking_section

    return latent_mask


def inject_latent_features(
    inputs_embeds: torch.FloatTensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
    ocr_connector: nn.Module,
) -> torch.FloatTensor:
    """Inject pre-encoded OCR features at latent token positions.

    This function:
    1. Takes pre-encoded OCR features (list of [100, 1280] tensors per sample)
    2. Mean-pools each to [1280] vector
    3. Passes through OCR connector to get [hidden_dim] embeddings
    4. Replaces the embeddings at latent token positions

    Args:
        inputs_embeds: [batch_size, seq_len, hidden_dim] Input embeddings
        latent_supervision: List of lists of OCR features
            - Outer list: per sample in batch
            - Inner list: per latent chunk in sample
            - Each tensor: [100, 1280] (10×10 grid, unless otherwise specified)
        latent_positions: [batch_size, seq_len] Boolean mask for latent positions
        ocr_connector: OCR connector (projects 1280 → hidden_dim)

    Returns:
        [batch_size, seq_len, hidden_dim] Embeddings with latents injected

    Note:
        Each [100, 1280] feature chunk is mean-pooled to [1280] before projection.
        This means each latent token represents one rendered thinking image.
    """
    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    # Clone to avoid modifying the original (in-place modifications can cause issues)
    injected_embeds = inputs_embeds.clone()

    for b in range(batch_size):
        # Get latent positions for this sample
        sample_latent_mask = latent_positions[b]
        num_latent_tokens = sample_latent_mask.sum().item()

        if num_latent_tokens == 0:
            continue  # No latents in this sample

        # Get supervision features for this sample
        if b >= len(latent_supervision):
            logger.warning(f"Sample {b} has latent positions but no supervision provided")
            continue

        sample_supervision = latent_supervision[b]
        if len(sample_supervision) != num_latent_tokens:
            logger.warning(
                f"Sample {b}: has {num_latent_tokens} latent tokens but "
                f"{len(sample_supervision)} supervision chunks. "
                f"Using minimum."
            )
            num_latent_tokens = min(num_latent_tokens, len(sample_supervision))

        # Process each supervision chunk and inject at corresponding position
        latent_indices = sample_latent_mask.nonzero(as_tuple=False).squeeze(dim=-1)

        for i, latent_idx in enumerate(latent_indices[:num_latent_tokens]):
            # Get supervision feature for this latent
            feat = sample_supervision[i]  # [T, 1280]

            # Ensure feature is on correct device
            if feat.device != device:
                feat = feat.to(device=device, dtype=dtype)

            # Mean-pool spatial tokens to get single [1280] vector
            if feat.ndim == 2:
                pooled_feat = feat.mean(dim=0)  # [1280]
            else:
                pooled_feat = feat  # Already [1280]

            # Project through OCR connector: [1280] → [hidden_dim]
            projected = ocr_connector(pooled_feat.unsqueeze(0))  # [1, hidden_dim]
            projected = projected.squeeze(0)  # [hidden_dim]

            # Inject at latent position
            injected_embeds[b, latent_idx] = projected

    return injected_embeds


def prepare_latent_supervision(
    raw_features: List[List[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> List[List[torch.Tensor]]:
    """Prepare raw OCR features for injection.

    This is a utility to ensure features are on the correct device and dtype.

    Args:
        raw_features: List of lists of OCR features (can be on CPU or GPU)
        device: Target device
        dtype: Target dtype

    Returns:
        List of lists of OCR features on correct device/dtype
    """
    prepared = []
    for sample_feats in raw_features:
        sample_prepared = []
        for feat in sample_feats:
            if isinstance(feat, torch.Tensor):
                feat = feat.to(device=device, dtype=dtype)
            sample_prepared.append(feat)
        prepared.append(sample_prepared)
    return prepared


def validate_latent_inputs(
    input_ids: torch.LongTensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> bool:
    """Validate that latent inputs are correctly formatted.

    Args:
        input_ids: [batch_size, seq_len] Input token IDs
        latent_supervision: List of lists of OCR features
        latent_positions: [batch_size, seq_len] Boolean mask

    Returns:
        True if valid, raises ValueError otherwise

    Raises:
        ValueError: If inputs are incorrectly formatted
    """
    batch_size = input_ids.shape[0]

    if len(latent_supervision) != batch_size:
        raise ValueError(
            f"latent_supervision has {len(latent_supervision)} samples "
            f"but input_ids has {batch_size} samples"
        )

    if latent_positions.shape[0] != batch_size:
        raise ValueError(
            f"latent_positions has {latent_positions.shape[0]} samples "
            f"but input_ids has {batch_size} samples"
        )

    if latent_positions.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"latent_positions seq_len {latent_positions.shape[1]} "
            f"doesn't match input_ids seq_len {input_ids.shape[1]}"
        )

    # Check that supervision count matches latent positions count
    for b in range(batch_size):
        num_latent_tokens = latent_positions[b].sum().item()
        num_supervision = len(latent_supervision[b])

        if num_latent_tokens > 0 and num_supervision == 0:
            raise ValueError(
                f"Sample {b} has {num_latent_tokens} latent tokens but no supervision"
            )

        if num_supervision > 0 and num_latent_tokens == 0:
            raise ValueError(
                f"Sample {b} has supervision but no latent tokens"
            )

    return True


# Convenience class for batch-level injection
class LatentInjector:
    """Helper class to manage latent injection with connector reference.

    This class stores the OCR connector and provides a cleaner interface
    for injection in the model forward pass.

    Example:
        # In model __init__:
        self.latent_injector = LatentInjector(ocr_connector=self.ocr_connector)

        # In model forward():
        if latent_supervision is not None:
            inputs_embeds = self.latent_injector.inject(
                inputs_embeds, latent_supervision, latent_positions
            )
    """

    def __init__(self, ocr_connector: nn.Module):
        """Initialize with OCR connector.

        Args:
            ocr_connector: OCR connector (projects 1280 → hidden_dim)
        """
        self.ocr_connector = ocr_connector

    def inject(
        self,
        inputs_embeds: torch.FloatTensor,
        latent_supervision: List[List[torch.Tensor]],
        latent_positions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Inject latent features using stored connector.

        Args:
            inputs_embeds: Input embeddings
            latent_supervision: OCR features for latents
            latent_positions: Boolean mask for latent positions

        Returns:
            Embeddings with latents injected
        """
        return inject_latent_features(
            inputs_embeds=inputs_embeds,
            latent_supervision=latent_supervision,
            latent_positions=latent_positions,
            ocr_connector=self.ocr_connector,
        )
