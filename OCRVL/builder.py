"""OCR-LLaVA Builder Helpers

These thin wrappers ensure the OCR-aware LLaVA classes are registered with
HuggingFace's AutoModel registry before delegating to the official LLaVA
builder. This lets you load standard LLaVA checkpoints while using the OCR
image-token alignment logic.
"""

from __future__ import annotations

from typing import Any, Tuple


def load_pretrained_model(*args: Any, use_ocr_llava: bool = True, **kwargs: Any) -> Tuple:
    """Drop-in replacement for llava.model.builder.load_pretrained_model.

    If `use_ocr_llava` is True (default), import the OCR-aligned LLaVA wrapper
    to override AutoModel mapping and then call upstream loader.
    Returns (tokenizer, model, image_processor, context_len), identical to
    upstream.
    """
    if use_ocr_llava:
        # Import to trigger AutoModel registration override for LlavaConfig
        from OCRVL.model.language_model import ocr_llava_llama  # noqa: F401

    from llava.model.builder import load_pretrained_model as _llava_load

    return _llava_load(*args, **kwargs)

