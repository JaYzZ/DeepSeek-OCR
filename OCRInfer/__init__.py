"""
OCRInfer - Inference toolkit for DeepSeek OCR with encoder-decoder separation

This package provides:
- Standalone vision encoder (DPSKOCREncoder, ~400M params)
- Optional multi-level feature extraction (DeepStack/Qwen3-VL style)
- vLLM-based decoder with embedding support
- Round-trip testing utilities
"""

__version__ = "1.0.0"

from .encoder import DPSKOCREncoder, EncoderOutput
from .decoder import VLLMEmbeddingDecoder

__all__ = [
    "DPSKOCREncoder",
    "EncoderOutput",
    "VLLMEmbeddingDecoder",
]
