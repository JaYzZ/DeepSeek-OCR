"""
DeepSeek-OCR Model Wrapper

This wrapper defines the DeepSeek-OCR model architecture so that OCRInfer
can load it directly from safetensors. The model code is self-managed in this repository.
"""
import torch
import torch.nn as nn
from typing import Optional


class DeepSeekOCRWrapper(nn.Module):
    """Wrapper for DeepSeek-OCR model that can load weights from safetensors."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dtype = dtype

        # Import DeepSeek-OCR components from our local copy
        from OCRInfer.model.deepseek_ocr.modeling_deepseekocr import (
            DeepseekOCRModel,
            DeepseekOCRConfig
        )

        # Create a minimal config for the vision model
        # We only need vision components, not the full language model
        config = DeepseekOCRConfig(
            vocab_size=129280,  # From config.json
            hidden_size=1280,
            intermediate_size=6848,
            num_hidden_layers=12,
            num_attention_heads=10,
            num_key_value_heads=10,
            dtype=dtype,
        )

        # Create the model with correct architecture
        self.model = DeepseekOCRModel(config)

        # Convert to target dtype
        self.to(dtype)

    def forward(self, pixel_values, attention_mask=None):
        """Forward pass - not used during inference, only for loading weights."""
        return self.model(pixel_values=pixel_values, attention_mask=attention_mask)


# Export key classes from our local copy
from OCRInfer.model.deepseek_ocr.modeling_deepseekocr import (
    DeepseekOCRConfig,
    DeepseekOCRModel,
    DeepseekOCRForCausalLM
)

__all__ = ['DeepSeekOCRWrapper', 'DeepseekOCRConfig', 'DeepseekOCRModel', 'DeepseekOCRForCausalLM']
