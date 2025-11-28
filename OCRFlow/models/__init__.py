"""
OCRFlow Models Module

Contains model implementations:
- markovian_chunk_decoder: GPT-like chunk-to-chunk decoder (autoregressive at chunk level)
- image_token_encoder: DeepSeek OCR-based image token encoder
- mmdit: Multimodal Diffusion Transformer
- mmdit_pretrained: MMDiT with Qwen-Image pretrained weights (20B)
- mmdit_sana: MMDiT with SANA pretrained weights (600M/1.6B)
- mmdit_scratch: MMDiT trained from scratch (~600M params, 10x10 feature map)
- text_encoders: Dual text encoder setup (T5 + CLIP)
"""

from .markovian_chunk_decoder import (
    MarkovianChunkDecoder,
    create_chunk_decoder,
)
from .image_token_encoder import ImageTokenEncoder
from .mmdit import MMDiTOCRFlow, create_mmdit_ocrflow
from .mmdit_pretrained import MMDiTOCRFlowPretrained, create_mmdit_ocrflow_pretrained
from .mmdit_sana import MMDiTOCRFlowSANA, create_mmdit_ocrflow_sana
from .mmdit_scratch import MMDiTScratch, create_mmdit_scratch
from .text_encoders import load_text_encoders, encode_text_dual

__all__ = [
    "MarkovianChunkDecoder",
    "create_chunk_decoder",
    "ImageTokenEncoder",
    "MMDiTOCRFlow",
    "create_mmdit_ocrflow",
    "MMDiTOCRFlowPretrained",
    "create_mmdit_ocrflow_pretrained",
    "MMDiTOCRFlowSANA",
    "create_mmdit_ocrflow_sana",
    "MMDiTScratch",
    "create_mmdit_scratch",
    "load_text_encoders",
    "encode_text_dual"
]
