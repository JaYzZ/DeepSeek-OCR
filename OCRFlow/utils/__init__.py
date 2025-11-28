"""
OCRFlow Utilities Module

Contains utility functions:
- sampling: Euler sampling for inference
- helpers: General helper functions
- vision_encoder: Memory-efficient vision encoder (~400M params)
- vistok_decoder: DeepSeek-OCR decoder for vistok-to-text transcription
- fast_renderer: High-throughput text rendering
- text_rendering: Basic text rendering utilities
"""

from .sampling import euler_sampling
from .helpers import set_seed, get_device
from .vision_encoder import VisionEncoderOnly, create_vision_encoder, get_vision_encoder
from .vistok_decoder import VistokDecoder, create_vistok_decoder
from .fast_renderer import FastBatchRenderer, render_text_optimized

__all__ = [
    "euler_sampling",
    "set_seed",
    "get_device",
    "VisionEncoderOnly",
    "create_vision_encoder",
    "get_vision_encoder",
    "VistokDecoder",
    "create_vistok_decoder",
    "FastBatchRenderer",
    "render_text_optimized",
]
