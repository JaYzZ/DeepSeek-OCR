"""
OCRFlow Configuration Module

Contains configuration files for:
- Model architectures (MMDiT sizes)
- Training hyperparameters
- Data paths and preprocessing
"""

from .model_config import ModelConfig, get_model_config
from .train_config import TrainConfig, get_default_train_config

__all__ = ["ModelConfig", "get_model_config", "TrainConfig", "get_default_train_config"]
