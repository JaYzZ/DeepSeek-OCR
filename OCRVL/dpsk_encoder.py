from __future__ import annotations

import os
from typing import Optional

import torch

from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
from OCRInfer.utils.model_paths import resolve_model_path

_ENCODER_SINGLETON: Optional[DPSKOCREncoder] = None


def _default_device() -> str:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return f"cuda:{local_rank}"
    return "cpu"


def _parse_dtype(value: str) -> torch.dtype:
    v = value.strip().lower()
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value!r} (expected bf16|fp16|fp32)")


def get_dpsk_encoder() -> DPSKOCREncoder:
    """
    Lazily initialize the DeepSeek-OCR vision encoder in the *current process*.

    Important: only call this from the training/inference process (i.e., not inside
    PyTorch DataLoader workers), because model loading may touch CUDA.
    """
    global _ENCODER_SINGLETON
    if _ENCODER_SINGLETON is not None:
        return _ENCODER_SINGLETON

    model_path = os.environ.get("DPSK_MODEL_PATH", "deepseek-ai/DeepSeek-OCR")
    model_path = resolve_model_path(model_path)
    device = os.environ.get("DPSK_DEVICE", _default_device())
    dtype = _parse_dtype(os.environ.get("DPSK_DTYPE", "bf16"))

    require_deepstack = os.environ.get("OCRVL_DPSK_DEEPSTACK", "1").strip() != "0"
    intermediate_layer_indices = [] if require_deepstack else None

    remove_separators = os.environ.get("OCRVL_DPSK_REMOVE_SEPARATORS", "1").strip() != "0"

    _ENCODER_SINGLETON = DPSKOCREncoder(
        model_path=model_path,
        device=device,
        dtype=dtype,
        intermediate_layer_indices=intermediate_layer_indices,
        remove_separators=remove_separators,
    )
    return _ENCODER_SINGLETON

