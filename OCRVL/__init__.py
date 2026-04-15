"""OCRVL

Lightweight adaptation layer for multimodal LLMs to consume DeepSeek-OCR's
visual-token-first inputs. It currently provides:

- OCR-aware LLaVA wrappers (LLaMA/Mistral) that respect repeated image tokens.
- OCR-aware Qwen VL wrappers (Qwen3-VL, Qwen2.5-VL) that accept OCR visual tokens.
- OCR text/image adapters for Qwen2.5-VL and Qwen3-VL built on DeepSeek-OCR encodings.
- OCRVL-specific vLLM processors for OCR-aware inference paths.
"""

from .model.language_model.ocr_qwen25_vl import (
    OCRQwen25VLForConditionalGeneration,
    Qwen25VLOCRTextAdapter,
)
from .model.language_model.ocr_qwen3_vl import (
    OCRQwen3VLForConditionalGeneration,
    Qwen3VLOCRTextAdapter,
)
