"""OCRVL

Lightweight adaptation layer for multimodal LLMs to consume DeepSeek-OCR's
visual-token-first inputs. It currently provides:

- OCR-aware LLaVA wrappers (LLaMA/Mistral) that respect repeated image tokens.
- OCR-aware Qwen VL wrappers (Qwen3-VL, Qwen2.5-VL) that accept OCR visual tokens.
- OCR text/image adapters for Qwen2.5-VL and Qwen3-VL built on DeepSeek-OCR encodings.
- vLLM decoders for Qwen VL models (skeleton implementations).
"""

import os
from sys_path import _add_sys_path

# Ensure the sibling LLaVA repo is importable (../LLaVA)
_HERE = os.path.dirname(__file__)
_SOURCES_DIR = os.path.dirname(os.path.dirname(_HERE))  # .../sources
_LLAVA_DIR = os.path.join(_SOURCES_DIR, 'LLaVA')
_add_sys_path(_LLAVA_DIR)

# Re-export the most common entry points
from .model.language_model.ocr_llava_llama import OCRLlavaLlamaForCausalLM  # noqa: E402,F401
from .model.language_model.ocr_qwen25_vl import (  # noqa: E402,F401
    OCRQwen25VLForConditionalGeneration,
    Qwen25VLOCRTextAdapter,
)
from .model.language_model.ocr_qwen3_vl import (  # noqa: E402,F401
    OCRQwen3VLForConditionalGeneration,
    Qwen3VLOCRTextAdapter,
)

# Decoders (vLLM-based, skeleton implementations)
# NOTE: These are not yet fully implemented - use model.language_model.* for production
try:
    from .decoder import Qwen3VLDecoder, Qwen25VLDecoder  # noqa: E402,F401
except ImportError:
    # vLLM not installed, decoders unavailable
    pass
