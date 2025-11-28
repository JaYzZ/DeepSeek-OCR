"""
OCRFlow Training Module

Contains training utilities:
- dataset: OCR dataset loaders
- loss: Flow matching loss functions
"""

from .loss import RectifiedFlowLoss
from .dataset import OCRDataset

__all__ = ["RectifiedFlowLoss", "OCRDataset"]
