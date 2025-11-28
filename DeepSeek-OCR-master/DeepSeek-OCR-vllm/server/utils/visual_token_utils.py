"""
Utility functions for extracting pure visual tokens from DeepSeek-OCR output.

For OCRFlow and other downstream tasks that need only the content-dependent
visual features without structural tokens (newlines and view separator).
"""

import numpy as np
import torch
from typing import Union


def extract_pure_visual_tokens(
    vistok_111: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Extract 100 pure visual tokens from 111-token DeepSeek-OCR output.

    Removes:
    - 10 newline markers (indices: 10, 21, 32, 43, 54, 65, 76, 87, 98, 109)
    - 1 view separator (index: 110)

    Keeps:
    - 100 visual patch tokens (10x10 grid) that vary with input content

    Args:
        vistok_111: Visual tokens from DeepSeek-OCR
                   Shape: [111, 1280] or [batch, 111, 1280]

    Returns:
        pure_visual: Pure content-dependent visual tokens
                    Shape: [100, 1280] or [batch, 100, 1280]
                    Same type (numpy/torch) as input

    Example:
        >>> # From API response
        >>> vistok_111 = np.frombuffer(response.content, dtype=np.float32)
        >>> vistok_111 = vistok_111.reshape(-1, 111, 1280)
        >>> vistok_100 = extract_pure_visual_tokens(vistok_111)
        >>> print(vistok_100.shape)  # (batch, 100, 1280)
    """
    is_torch = isinstance(vistok_111, torch.Tensor)

    # Build index list: keep visual tokens, skip newlines and separator
    # Pattern: rows 0-9, each row has 11 tokens [10 visual + 1 newline]
    visual_indices = []
    for row in range(10):
        row_start = row * 11  # Start of each row
        visual_indices.extend(range(row_start, row_start + 10))

    # visual_indices = [0,1,...,9, 11,12,...,20, 22,...,31, ..., 99,100,...,108]
    # Skipped: 10, 21, 32, 43, 54, 65, 76, 87, 98, 109 (newlines), 110 (separator)

    if is_torch:
        if vistok_111.dim() == 2:  # Single sample
            return vistok_111[visual_indices]
        elif vistok_111.dim() == 3:  # Batch
            return vistok_111[:, visual_indices, :]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {vistok_111.shape}")
    else:  # numpy
        if vistok_111.ndim == 2:  # Single sample
            return vistok_111[visual_indices]
        elif vistok_111.ndim == 3:  # Batch
            return vistok_111[:, visual_indices, :]
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {vistok_111.shape}")


def reshape_to_spatial_grid(
    pure_visual_100: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reshape 100 visual tokens into 10x10 spatial grid.

    Useful for:
    - Spatially-aware processing
    - Convolutional operations
    - Visualization as 2D feature maps

    Args:
        pure_visual_100: Pure visual tokens
                        Shape: [100, 1280] or [batch, 100, 1280]

    Returns:
        spatial_grid: Spatial arrangement of tokens
                     Shape: [10, 10, 1280] or [batch, 10, 10, 1280]

    Example:
        >>> vistok_100 = extract_pure_visual_tokens(vistok_111)
        >>> spatial = reshape_to_spatial_grid(vistok_100)
        >>> print(spatial.shape)  # (batch, 10, 10, 1280)
        >>> # Access top-left patch: spatial[0, 0, 0, :]
        >>> # Access bottom-right patch: spatial[0, 9, 9, :]
    """
    is_torch = isinstance(pure_visual_100, torch.Tensor)

    if is_torch:
        if pure_visual_100.dim() == 2:  # Single sample [100, 1280]
            return pure_visual_100.view(10, 10, -1)
        elif pure_visual_100.dim() == 3:  # Batch [batch, 100, 1280]
            batch_size = pure_visual_100.shape[0]
            return pure_visual_100.view(batch_size, 10, 10, -1)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {pure_visual_100.shape}")
    else:  # numpy
        if pure_visual_100.ndim == 2:  # Single sample [100, 1280]
            return pure_visual_100.reshape(10, 10, -1)
        elif pure_visual_100.ndim == 3:  # Batch [batch, 100, 1280]
            batch_size = pure_visual_100.shape[0]
            return pure_visual_100.reshape(batch_size, 10, 10, -1)
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {pure_visual_100.shape}")


def flatten_spatial_grid(
    spatial_grid: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Flatten 10x10 spatial grid back to 100 sequential tokens.

    Inverse of reshape_to_spatial_grid().

    Args:
        spatial_grid: Spatial grid of tokens
                     Shape: [10, 10, 1280] or [batch, 10, 10, 1280]

    Returns:
        flat_tokens: Flattened sequence
                    Shape: [100, 1280] or [batch, 100, 1280]
    """
    is_torch = isinstance(spatial_grid, torch.Tensor)

    if is_torch:
        if spatial_grid.dim() == 3:  # Single sample [10, 10, 1280]
            return spatial_grid.view(100, -1)
        elif spatial_grid.dim() == 4:  # Batch [batch, 10, 10, 1280]
            batch_size = spatial_grid.shape[0]
            return spatial_grid.view(batch_size, 100, -1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {spatial_grid.shape}")
    else:  # numpy
        if spatial_grid.ndim == 3:  # Single sample [10, 10, 1280]
            return spatial_grid.reshape(100, -1)
        elif spatial_grid.ndim == 4:  # Batch [batch, 10, 10, 1280]
            batch_size = spatial_grid.shape[0]
            return spatial_grid.reshape(batch_size, 100, -1)
        else:
            raise ValueError(f"Expected 3D or 4D array, got shape {spatial_grid.shape}")


def add_structural_tokens(
    pure_visual_100: Union[np.ndarray, torch.Tensor],
    newline_vector: Union[np.ndarray, torch.Tensor],
    separator_vector: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Add structural tokens to pure visual tokens to reconstruct 111-token format.

    This is the inverse of extract_pure_visual_tokens().
    Useful if you need to feed modified visual tokens back to DeepSeek-OCR decoder.

    Args:
        pure_visual_100: Pure visual tokens [100, 1280] or [batch, 100, 1280]
        newline_vector: Newline marker vector [1280]
        separator_vector: View separator vector [1280]

    Returns:
        vistok_111: Full token sequence [111, 1280] or [batch, 111, 1280]

    Note:
        To get the original newline and separator vectors from the model:
        >>> from server.standalone_vision_encoder import create_vision_encoder
        >>> encoder = create_vision_encoder()
        >>> newline_vec = encoder.model.model.image_newline.cpu().numpy()
        >>> separator_vec = encoder.model.model.view_seperator.cpu().numpy()
    """
    is_torch = isinstance(pure_visual_100, torch.Tensor)

    if is_torch:
        newline_vector = torch.as_tensor(newline_vector, device=pure_visual_100.device, dtype=pure_visual_100.dtype)
        separator_vector = torch.as_tensor(separator_vector, device=pure_visual_100.device, dtype=pure_visual_100.dtype)

        if pure_visual_100.dim() == 2:  # Single sample
            spatial = pure_visual_100.view(10, 10, -1)  # [10, 10, 1280]
            newlines = newline_vector[None, None, :].expand(10, 1, -1)  # [10, 1, 1280]
            with_newlines = torch.cat([spatial, newlines], dim=1)  # [10, 11, 1280]
            flat = with_newlines.view(110, -1)  # [110, 1280]
            return torch.cat([flat, separator_vector[None, :]], dim=0)  # [111, 1280]

        elif pure_visual_100.dim() == 3:  # Batch
            batch_size = pure_visual_100.shape[0]
            spatial = pure_visual_100.view(batch_size, 10, 10, -1)  # [B, 10, 10, 1280]
            newlines = newline_vector[None, None, None, :].expand(batch_size, 10, 1, -1)  # [B, 10, 1, 1280]
            with_newlines = torch.cat([spatial, newlines], dim=2)  # [B, 10, 11, 1280]
            flat = with_newlines.view(batch_size, 110, -1)  # [B, 110, 1280]
            separator = separator_vector[None, None, :].expand(batch_size, 1, -1)  # [B, 1, 1280]
            return torch.cat([flat, separator], dim=1)  # [B, 111, 1280]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {pure_visual_100.shape}")

    else:  # numpy
        if pure_visual_100.ndim == 2:  # Single sample
            spatial = pure_visual_100.reshape(10, 10, -1)  # [10, 10, 1280]
            newlines = np.broadcast_to(newline_vector[None, None, :], (10, 1, spatial.shape[2]))
            with_newlines = np.concatenate([spatial, newlines], axis=1)  # [10, 11, 1280]
            flat = with_newlines.reshape(110, -1)  # [110, 1280]
            return np.concatenate([flat, separator_vector[None, :]], axis=0)  # [111, 1280]

        elif pure_visual_100.ndim == 3:  # Batch
            batch_size = pure_visual_100.shape[0]
            spatial = pure_visual_100.reshape(batch_size, 10, 10, -1)  # [B, 10, 10, 1280]
            newlines = np.broadcast_to(newline_vector[None, None, None, :], (batch_size, 10, 1, spatial.shape[3]))
            with_newlines = np.concatenate([spatial, newlines], axis=2)  # [B, 10, 11, 1280]
            flat = with_newlines.reshape(batch_size, 110, -1)  # [B, 110, 1280]
            separator = np.broadcast_to(separator_vector[None, None, :], (batch_size, 1, spatial.shape[3]))
            return np.concatenate([flat, separator], axis=1)  # [B, 111, 1280]
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {pure_visual_100.shape}")
