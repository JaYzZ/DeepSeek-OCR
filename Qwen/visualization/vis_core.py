#!/usr/bin/env python3
"""
Core utilities and base classes for visualization framework.

Provides common functionality for:
- Encoder initialization and management
- Image preprocessing
- Feature caching
- Configuration management
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from project_paths import hf_path

from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
from Qwen.encoder import Qwen25VLEncoder, Qwen3VLEncoder


class EncoderType(Enum):
    """Supported encoder types."""
    DPSK = "dpsk"
    QWEN3VL = "qwen3vl"
    QWEN25VL = "qwen25vl"


@dataclass
class EncoderConfig:
    """Configuration for vision encoder."""
    encoder_type: EncoderType
    model_path: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    use_vllm_kernels: bool = True
    intermediate_layers: Optional[List[int]] = field(default_factory=lambda: [6, 12, 18])
    remove_separators: bool = True
    keep_cls_intermediate: bool = True


@dataclass
class FeatureSpec:
    """Specification for feature extraction."""
    name: str
    layer: str  # 'final', 'intermediate', or specific layer index
    feature_type: str  # 'cls', 'visual', 'sam_raw'
    pooling: str = 'mean'  # 'mean', 'attention', 'none'


class EncoderManager:
    """Manage encoder initialization and caching."""

    _encoders: Dict[EncoderType, Any] = {}

    @classmethod
    def get_encoder(cls, config: EncoderConfig):
        """Get or create encoder instance."""
        if config.encoder_type in cls._encoders:
            return cls._encoders[config.encoder_type]

        encoder = cls._create_encoder(config)
        cls._encoders[config.encoder_type] = encoder
        return encoder

    @classmethod
    def _create_encoder(cls, config: EncoderConfig):
        """Create encoder based on type."""
        if config.encoder_type == EncoderType.DPSK:
            return DPSKOCREncoder(
                model_path=config.model_path,
                device=config.device,
                dtype=config.dtype,
                intermediate_layer_indices=config.intermediate_layers,
                remove_separators=config.remove_separators,
                keep_cls_intermediate=config.keep_cls_intermediate,
            )
        elif config.encoder_type == EncoderType.QWEN3VL:
            return Qwen3VLEncoder(
                model_name_or_path=config.model_path,
                device=config.device,
                dtype=config.dtype,
                use_vllm_kernels=config.use_vllm_kernels,
            )
        elif config.encoder_type == EncoderType.QWEN25VL:
            return Qwen25VLEncoder(
                model_name_or_path=config.model_path,
                device=config.device,
                dtype=config.dtype,
            )
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")

    @classmethod
    def cleanup(cls):
        """Clean up cached encoders."""
        for encoder in cls._encoders.values():
            del encoder
        cls._encoders.clear()


class ImagePreprocessor:
    """Image preprocessing utilities."""

    @staticmethod
    def load_image(path: Union[str, Path]) -> Image.Image:
        """Load image as RGB."""
        return Image.open(path).convert('RGB')

    @staticmethod
    def pad_to_base_size(image: Image.Image, base_size: int = 640) -> Image.Image:
        """Pad image to base_size maintaining aspect ratio."""
        return ImageOps.pad(image, (base_size, base_size), color=(128, 128, 128))

    @staticmethod
    def to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
        """Convert PIL image to tensor."""
        if normalize:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.4814546, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
            ])
        else:
            transform = T.ToTensor()

        return transform(image)

    @staticmethod
    def preprocess_for_encoder(image: Image.Image, encoder_type: EncoderType) -> torch.Tensor:
        """Preprocess image for specific encoder."""
        # Most encoders expect 640x640 padded images
        padded = ImagePreprocessor.pad_to_base_size(image, 640)
        tensor = ImagePreprocessor.to_tensor(padded)
        return tensor.unsqueeze(0)  # Add batch dimension


# FeatureCache removed - not used effectively and adds complexity


class ConfigManager:
    """Manage visualization configurations."""

    DEFAULT_ENCODERS = {
        EncoderType.DPSK: "deepseek-ai/DeepSeek-OCR",
        EncoderType.QWEN3VL: str(hf_path("Qwen", "Qwen3-VL-2B-Instruct")),
        EncoderType.QWEN25VL: str(hf_path("Qwen", "Qwen2.5-VL-7B-Instruct")),
    }

    @classmethod
    def get_default_config(cls, encoder_type: EncoderType) -> EncoderConfig:
        """Get default configuration for encoder type."""
        return EncoderConfig(
            encoder_type=encoder_type,
            model_path=cls.DEFAULT_ENCODERS[encoder_type],
        )

    @classmethod
    def load_config(cls, config_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    @classmethod
    def save_config(cls, config: Dict[str, Any], output_path: Path):
        """Save configuration to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)


class PathManager:
    """Manage output paths for visualizations."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent / "results" / "feature_vis"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_output_dir(self, vis_type: str, create: bool = True) -> Path:
        """Get output directory for visualization type."""
        # Don't add 'plots' if base_dir already ends with it
        if self.base_dir.name == "plots":
            output_dir = self.base_dir / vis_type
        else:
            output_dir = self.base_dir / "plots" / vis_type
        if create:
            output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_features_dir(self, create: bool = True) -> Path:
        """Get features directory."""
        features_dir = self.base_dir / "features"
        if create:
            features_dir.mkdir(parents=True, exist_ok=True)
        return features_dir

    def get_images_dir(self, create: bool = True) -> Path:
        """Get images directory."""
        images_dir = self.base_dir / "images"
        if create:
            images_dir.mkdir(parents=True, exist_ok=True)
        return images_dir


def get_device(device_str: str = "cuda:0") -> torch.device:
    """Get torch device with validation."""
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available, using CPU")
            return torch.device("cpu")
    return torch.device(device_str)


def setup_logging(verbose: bool = False):
    """Setup logging for visualization."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(message)s'
    )
    return logging.getLogger(__name__)
