"""
Visualization utilities for Qwen3-VL thinking mode.
t-SNE computation, attention heatmap helpers, token display utilities.
"""

import numpy as np
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Optional


def compute_tsne(
    hidden_states: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    max_samples: int = 500,
) -> np.ndarray:
    """Compute t-SNE on hidden states.

    Args:
        hidden_states: (seq_len, hidden_size) array
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed
        max_samples: Maximum number of samples to compute t-SNE on

    Returns:
        (seq_len, 2) t-SNE coordinates
    """
    # Downsample if sequence is too long
    if hidden_states.shape[0] > max_samples:
        indices = np.linspace(0, hidden_states.shape[0] - 1, max_samples, dtype=int)
        sample_hidden = hidden_states[indices]
    else:
        indices = None
        sample_hidden = hidden_states

    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    )
    sample_2d = tsne.fit_transform(sample_hidden)

    # If we downsampled, interpolate back to full sequence
    if indices is not None:
        full_2d = np.zeros((hidden_states.shape[0], 2))
        for i, idx in enumerate(indices):
            full_2d[idx] = sample_2d[i]
        # Interpolate missing positions
        for i in range(full_2d.shape[0]):
            if full_2d[i].sum() == 0:
                # Find nearest computed positions
                computed = indices[indices <= i]
                if len(computed) > 0:
                    prev_idx = computed[-1]
                    prev_pos = np.where(indices == prev_idx)[0][0]
                    if i < indices[-1]:
                        next_idx = indices[indices > i][0]
                        next_pos = np.where(indices == next_idx)[0][0]
                        # Linear interpolation
                        t = (i - prev_idx) / (next_idx - prev_idx)
                        full_2d[i] = sample_2d[prev_pos] * (1 - t) + sample_2d[next_pos] * t
                    else:
                        full_2d[i] = sample_2d[prev_pos]
        return full_2d

    return sample_2d


def prepare_token_data(
    tokens: List[Dict[str, Any]],
    hidden_states: np.ndarray,
    continuous_mask: List[bool],
) -> Dict[str, Any]:
    """Prepare token data for visualization.

    Args:
        tokens: List of token dicts with id, text, position
        hidden_states: (seq_len, hidden_size) array
        continuous_mask: List marking continuous positions

    Returns:
        Dict with token types, colors, labels for t-SNE
    """
    token_types = []
    token_labels = []

    for i, token in enumerate(tokens):
        if i < len(continuous_mask) and continuous_mask[i]:
            token_types.append('continuous')
        elif 'image' in token.get('text', '').lower() or token.get('is_image', False):
            token_types.append('image')
        elif i < 10:  # Early tokens are likely question/prompt
            token_types.append('question')
        else:
            token_types.append('answer')

        token_labels.append(token.get('text', f"token_{i}"))

    # Color mapping
    color_map = {
        'image': 'red',
        'question': 'blue',
        'continuous': 'green',
        'answer': 'orange',
    }
    colors = [color_map.get(t, 'gray') for t in token_types]

    return {
        'types': token_types,
        'colors': colors,
        'labels': token_labels,
    }


def aggregate_attention(
    attention_weights: np.ndarray,
    method: str = 'mean_last_layer',
) -> np.ndarray:
    """Aggregate attention weights across layers and heads.

    Args:
        attention_weights: (num_layers, num_heads, seq_len, seq_len) or
                          (num_layers, seq_len, seq_len)
        method: Aggregation method
            - 'mean_last_layer': Mean across heads of last layer
            - 'mean_all_layers': Mean across all layers and heads
            - 'max_last_layer': Max across heads of last layer

    Returns:
        (seq_len, seq_len) aggregated attention matrix
    """
    if attention_weights.ndim == 4:
        # (num_layers, num_heads, seq_len, seq_len)
        if method == 'mean_last_layer':
            return attention_weights[-1].mean(axis=0)
        elif method == 'mean_all_layers':
            return attention_weights.mean(axis=0).mean(axis=0)
        elif method == 'max_last_layer':
            return attention_weights[-1].max(axis=0)
        else:
            return attention_weights[-1].mean(axis=0)
    elif attention_weights.ndim == 3:
        # (num_layers, seq_len, seq_len)
        if method == 'mean_last_layer':
            return attention_weights[-1]
        elif method == 'mean_all_layers':
            return attention_weights.mean(axis=0)
        else:
            return attention_weights[-1]
    else:
        # Already (seq_len, seq_len)
        return attention_weights


def create_attention_heatmap_data(
    attention_matrix: np.ndarray,
    tokens: List[str],
    selected_token_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Create Plotly heatmap data for attention visualization.

    Args:
        attention_matrix: (seq_len, seq_len) attention weights
        tokens: List of token strings
        selected_token_idx: If specified, show only this token's attention

    Returns:
        Dict with z (values), x (labels), y (labels)
    """
    if selected_token_idx is not None and 0 <= selected_token_idx < attention_matrix.shape[0]:
        # Show single row
        z = attention_matrix[selected_token_idx:selected_token_idx+1, :]
        y = [tokens[selected_token_idx]]
    else:
        z = attention_matrix
        y = tokens

    # Truncate labels if too long
    max_label_len = 30
    x = [t[:max_label_len] + '...' if len(t) > max_label_len else t for t in tokens]

    return {
        'z': z.tolist(),
        'x': x,
        'y': y,
    }


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string
    """
    import base64
    from pathlib import Path

    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode('utf-8')


def decode_base64_to_image(base64_str: str) -> bytes:
    """Decode base64 string to image bytes.

    Args:
        base64_str: Base64 encoded string

    Returns:
        Image bytes
    """
    import base64
    return base64.b64decode(base64_str)
