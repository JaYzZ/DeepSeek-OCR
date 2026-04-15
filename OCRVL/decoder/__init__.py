"""OCRVL-specific vLLM processors."""

from .ocrvl_vllm_processor import OCRVLProcessor
from .ocr_qwen3vl_vllm_processor import OCRQwen3VLProcessor

__all__ = [
    "OCRVLProcessor",
    "OCRQwen3VLProcessor",
]
