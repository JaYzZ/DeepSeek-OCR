"""
vLLM inference utilities for Qwen3-VL thinking mode.
Handles model loading, inference with hidden states extraction, and trace collection.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from PIL import Image
import base64
from io import BytesIO


def load_runtime_env(config_path: str = "Qwen/configs/qwen3vl_runtime_env.yaml") -> Dict[str, Any]:
    """Load runtime environment configuration.

    Args:
        config_path: Path to runtime env config

    Returns:
        Config dictionary
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_vllm_thinking_env(
    model_path: str,
    lora_checkpoint_path: Optional[str] = None,
    enable_thinking: bool = True,
) -> None:
    """Set up environment variables for vLLM thinking mode.

    Args:
        model_path: Path to base model
        lora_checkpoint_path: Path to LoRA checkpoint
        enable_thinking: Enable thinking mode
    """
    os.environ["VLLM_MODEL_PATH"] = model_path
    if lora_checkpoint_path:
        os.environ["VLLM_LORA_CHECKPOINT_PATH"] = lora_checkpoint_path

    if enable_thinking:
        os.environ["VLLM_THINKING_MODE"] = "1"
        os.environ["VLLM_THINKING_DEBUG"] = "1"

    # Enable thinking-specific token IDs (Qwen3-VL specific)
    os.environ["QWEN3VL_THINKING_START_ID"] = "151667"
    os.environ["QWEN3VL_THINKING_END_ID"] = "151668"
    os.environ["QWEN3VL_LATENT_TOKEN_ID"] = "151669"
    os.environ["QWEN3VL_THINKING_SEP_ID"] = "151670"


def prepare_image_prompt(image_path: Optional[str] = None, image_base64: Optional[str] = None) -> List[Dict[str, Any]]:
    """Prepare image prompt for Qwen3-VL.

    Args:
        image_path: Path to image file
        image_base64: Base64 encoded image

    Returns:
        List of message dicts for the model
    """
    if image_base64:
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_bytes))
    elif image_path:
        img = Image.open(image_path)
    else:
        return []

    # Qwen3-VL image format
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
        ],
    }]


def prepare_text_prompt(text: str) -> List[Dict[str, Any]]:
    """Prepare text prompt for Qwen3-VL.

    Args:
        text: Text prompt

    Returns:
        List of message dicts
    """
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": text},
        ],
    }]


def prepare_multimodal_prompt(
    text: str,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Prepare multimodal prompt for Qwen3-VL.

    Args:
        text: Text prompt/question
        image_path: Path to image file
        image_base64: Base64 encoded image

    Returns:
        List of message dicts
    """
    content = []

    if image_path or image_base64:
        if image_base64:
            img_bytes = base64.b64decode(image_base64)
            img = Image.open(BytesIO(img_bytes))
        else:
            img = Image.open(image_path)
        content.append({"type": "image", "image": img})

    content.append({"type": "text", "text": text})

    return [{
        "role": "user",
        "content": content,
    }]


def collect_trace_data(
    trace_store: Dict[str, Any],
    tokenizer: Any,
) -> Dict[str, Any]:
    """Collect and process trace data for visualization.

    Args:
        trace_store: Trace data from trace_store
        tokenizer: Tokenizer for decoding token IDs

    Returns:
        Processed data with tokens, hidden states, attention, etc.
    """
    from vllm_thinking.trace_store import get_request_trace

    trace = get_request_trace("current")  # or get by specific request ID
    if not trace:
        return {}

    # Extract continuous tokens
    continuous_embeddings = trace.get('continuous_latent_embeddings', [])
    continuous_mask = trace.get('continuous_token_mask', [])

    # Extract all hidden states if available
    all_hidden_states = trace.get('all_hidden_states', [])
    all_token_ids = trace.get('all_token_ids', [])

    # Extract attention weights if available
    attention_weights = trace.get('attention_weights', [])

    # Decode tokens
    tokens = []
    if all_token_ids:
        for token_id in all_token_ids:
            token_text = tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            tokens.append({
                'id': int(token_id),
                'text': token_text,
            })

    # Convert to numpy arrays
    hidden_states = None
    if all_hidden_states is not None and np.asarray(all_hidden_states).size > 0:
        hidden_states = np.asarray(all_hidden_states)

    attention = None
    if attention_weights is not None and np.asarray(attention_weights).size > 0:
        attention = np.asarray(attention_weights)

    return {
        'tokens': tokens,
        'hidden_states': hidden_states,
        'continuous_mask': continuous_mask,
        'attention_weights': attention,
        'continuous_embeddings': continuous_embeddings,
    }


def extract_image_token_embeddings(
    model: Any,
    image_path: str,
) -> np.ndarray:
    """Extract image token embeddings from vision encoder.

    Args:
        model: Qwen3-VL model
        image_path: Path to image

    Returns:
        (num_image_tokens, hidden_size) array of embeddings
    """
    img = Image.open(image_path)

    # This is model-specific - Qwen3-VL uses a vision encoder
    # The exact implementation depends on the model architecture
    # Placeholder for actual implementation

    # For Qwen3-VL, image tokens are processed through the visual encoder
    # and embedded into the language model's hidden space

    # This would typically involve:
    # 1. Processing image through visual encoder
    # 2. Getting image embeddings
    # 3. Projecting to language model hidden size

    raise NotImplementedError(
        "Image token extraction requires specific model architecture handling. "
        "This should be implemented based on how Qwen3-VL processes images."
    )


def get_default_deepvision_example() -> Dict[str, str]:
    """Get default deepvision example for visualization.

    Returns:
        Dict with image_path and question
    """
    return {
        "image_path": "Qwen/data/deepvision_images/deepvision_math-77k_0000000.jpg",
        "question": "In the figure, triangle ABC is a right triangle with angle C = 90°. If point D is on side AB such that CD is perpendicular to AB, and the lengths of AC = 8 and BC = 6, find the length of CD.",
    }
