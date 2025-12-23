#!/usr/bin/env python3
"""Qwen2.5-VL Vision Encoder - Standalone encoder for benchmarking.

Default path uses vLLM's inference-only ViT implementation to match native
vLLM speed (FlashAttention-varlen + packed seqlens).
Falls back to HF visual module if vLLM sources are unavailable.
"""

import torch
from typing import List, Union
from PIL import Image
import numpy as np
from dataclasses import dataclass

from transformers import AutoModel, AutoProcessor

try:
    from .vllm_qwen_vit import build_qwen25_vit
except ImportError:  # pragma: no cover
    from vllm_qwen_vit import build_qwen25_vit


@dataclass
class Qwen25VLEncoderOutput:
    """Output from Qwen2.5-VL vision encoder."""
    features: torch.Tensor
    grid_thw: torch.Tensor

    def as_vllm_mm_dict(self) -> dict[str, torch.Tensor]:
        """Return a vLLM-compatible multimodal dict for decoder-only inference."""
        return {
            "image_embeds": self.features,
            "image_grid_thw": self.grid_thw,
        }


class Qwen25VLEncoder:
    """Standalone Qwen2.5-VL vision encoder."""

    def __init__(
        self,
        model_name_or_path: str = "/share/project/xiyan/huggingface/Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        use_vllm_kernels: bool = True,
        compile: bool = False,
        attn_backend_override: str | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_vllm_kernels = use_vllm_kernels
        self.compile = compile
        self.attn_backend_override = attn_backend_override
        self._vision_model = None
        self._processor = None

    @property
    def vision_model(self):
        if self._vision_model is None:
            self._load_model()
        return self._vision_model

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
        return self._processor

    def _load_model(self):
        print(f"Loading Qwen2.5-VL vision encoder from {self.model_name_or_path}...")
        if self.use_vllm_kernels:
            try:
                self._vision_model = build_qwen25_vit(
                    self.model_name_or_path,
                    device=self.device,
                    dtype=self.dtype,
                    compile=self.compile,
                    attn_backend_override=self.attn_backend_override,
                )
                print(f"Vision encoder loaded with vLLM kernels to {self.device}")
                return
            except Exception as e:
                import traceback
                print(f"vLLM vision load failed, falling back to HF visual: {e}")
                traceback.print_exc()

        hf_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
            "device_map": "cpu",
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            hf_kwargs["attn_implementation"] = "flash_attention_2"
        full_model = AutoModel.from_pretrained(self.model_name_or_path, **hf_kwargs)
        self._vision_model = full_model.visual.to(self.device)
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._vision_model.eval()
        print(f"Vision encoder loaded to {self.device} (HF fallback)")
        print(
            f"Parameters: {sum(p.numel() for p in self._vision_model.parameters()):,}"
        )

    @torch.no_grad()
    def encode_images(
        self,
        images: Union[List[Image.Image], List[np.ndarray]]
    ) -> Qwen25VLEncoderOutput:
        if not isinstance(images, list):
            images = [images]

        # Convert numpy to PIL
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            pil_images.append(img)

        # Use processor
        processed = self.processor.image_processor(pil_images, return_tensors='pt')
        pixel_values = processed['pixel_values'].to(device=self.device, dtype=self.dtype)
        grid_thw = processed['image_grid_thw'].to(device=self.device)

        # Run vision encoder
        features = self.vision_model(pixel_values, grid_thw)

        return Qwen25VLEncoderOutput(
            features=features,
            grid_thw=grid_thw
        )


if __name__ == "__main__":
    print("Testing Qwen25VLEncoder...")
    encoder = Qwen25VLEncoder(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16
    )
    test_image = Image.new('RGB', (640, 640), color=(255, 0, 0))
    output = encoder.encode_images([test_image])
    print(f"\nOutput:")
    print(f"  Features shape: {output.features.shape}")
    print(f"  Grid THW: {output.grid_thw}")
    print("\nEncoder test passed!")
