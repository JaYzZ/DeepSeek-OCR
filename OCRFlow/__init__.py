"""
OCRFlow: Rectified Flow Model for Text-to-Image Token Mapping

This package implements a rectified flow model using MMDiT transformer architecture
to learn text-to-image token mappings, inspired by UniFlow but adapted for OCR tasks.

Architecture:
- Image Token Encoder: Uses DeepSeek OCR's vision models (SAM + CLIP) to encode images
- Text Encoder: Dual text encoders (T5 + CLIP) for rich text conditioning
- MMDiT Transformer: Multimodal diffusion transformer predicting velocity in flow matching
- Rectified Flow: Linear interpolation path for faster, higher-quality generation

Author: Based on UniFlow architecture
"""

__version__ = "0.1.0"

from . import models
from . import training
from . import utils
from . import configs

__all__ = ["models", "training", "utils", "configs"]
