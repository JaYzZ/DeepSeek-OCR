"""Visualization utilities for Qwen3-VL thinking mode."""

from .visualization_utils import (
    compute_tsne,
    prepare_token_data,
    aggregate_attention,
    create_attention_heatmap_data,
    encode_image_to_base64,
    decode_base64_to_image,
)

from .vllm_inference import (
    load_runtime_env,
    setup_vllm_thinking_env,
    prepare_image_prompt,
    prepare_text_prompt,
    prepare_multimodal_prompt,
    collect_trace_data,
    get_default_deepvision_example,
)

__all__ = [
    # visualization_utils
    "compute_tsne",
    "prepare_token_data",
    "aggregate_attention",
    "create_attention_heatmap_data",
    "encode_image_to_base64",
    "decode_base64_to_image",
    # vllm_inference
    "load_runtime_env",
    "setup_vllm_thinking_env",
    "prepare_image_prompt",
    "prepare_text_prompt",
    "prepare_multimodal_prompt",
    "collect_trace_data",
    "get_default_deepvision_example",
]
