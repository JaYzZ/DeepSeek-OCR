#!/usr/bin/env python3
"""Qwen3-VL Vision Encoder - Standalone encoder for benchmarking.

Default path uses vLLM's inference-only ViT implementation to match native
vLLM speed (FlashAttention-varlen, packed seqlens, vLLM linear/conv layers).
Falls back to HF visual module if vLLM sources are unavailable.
"""

import torch
from typing import List, Union
from PIL import Image
import numpy as np
from dataclasses import dataclass

from transformers import AutoModel, AutoProcessor

# Support both package import and direct script execution.
try:
    from .vllm_qwen_vit import build_qwen3_vit
except ImportError:  # pragma: no cover
    from vllm_qwen_vit import build_qwen3_vit


@dataclass
class Qwen3VLEncoderOutput:
    """Output from Qwen3-VL vision encoder."""
    features: torch.Tensor
    deepstack_features: List[torch.Tensor]
    grid_thw: torch.Tensor

    def as_vllm_mm_dict(self) -> dict[str, torch.Tensor]:
        """Return a vLLM-compatible multimodal dict for decoder-only inference.

        vLLM Qwen3-VL expects image_embeds concatenated as:
        [main_features | deepstack_0 | deepstack_1 | deepstack_2]
        with shape [seq_len, 2048 * 4] = [seq_len, 8192]
        """
        if self.deepstack_features:
            image_embeds = torch.cat([self.features] + self.deepstack_features, dim=-1)
        else:
            image_embeds = self.features
        return {
            "image_embeds": image_embeds,
            "image_grid_thw": self.grid_thw,
        }


class Qwen3VLEncoder:
    """Standalone Qwen3-VL vision encoder."""

    def __init__(
        self,
        model_name_or_path: str = "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Instruct",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        use_vllm_kernels: bool = True,  # Use vLLM's optimized ViT kernels for better performance
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
        self._out_hidden_size = None
        self._num_deepstack = None

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
        print(f"Loading Qwen3-VL vision encoder from {self.model_name_or_path}...")
        if self.use_vllm_kernels:
            try:
                self._vision_model = build_qwen3_vit(
                    self.model_name_or_path,
                    device=self.device,
                    dtype=self.dtype,
                    compile=self.compile,
                    attn_backend_override=self.attn_backend_override,
                )
                # cache split info
                # vLLM vision config fields: out_hidden_size, deepstack_visual_indexes
                from transformers import AutoConfig

                cfg = AutoConfig.from_pretrained(
                    self.model_name_or_path, trust_remote_code=True
                ).vision_config
                self._out_hidden_size = cfg.out_hidden_size
                self._num_deepstack = len(getattr(cfg, "deepstack_visual_indexes", []))
                print(
                    f"Vision encoder loaded with vLLM kernels to {self.device} "
                    f"(deepstack={self._num_deepstack})"
                )
                return
            except Exception as e:
                import traceback
                print(f"vLLM vision load failed, falling back to HF visual: {e}")
                traceback.print_exc()

        hf_kwargs = {
            "trust_remote_code": True,
            "dtype": self.dtype,
            "device_map": str(self.device),  # Load directly to target device
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            hf_kwargs["attn_implementation"] = "flash_attention_2"
        full_model = AutoModel.from_pretrained(self.model_name_or_path, **hf_kwargs)
        self._vision_model = full_model.visual
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._vision_model.eval()
        print(f"Vision encoder loaded successfully to {self.device} (HF fallback)")
        print(
            f"Vision encoder parameters: "
            f"{sum(p.numel() for p in self._vision_model.parameters()):,}"
        )

    @torch.no_grad()
    def encode_images(
        self,
        images: Union[List[Image.Image], List[np.ndarray]]
    ) -> Qwen3VLEncoderOutput:
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

        # Run vision encoder.
        out = self.vision_model(pixel_values, grid_thw)
        if self._out_hidden_size is not None:
            # vLLM ViT returns concatenated [main | deepstack...] along last dim.
            chunks = torch.split(out, self._out_hidden_size, dim=1)
            features = chunks[0]
            deepstack_features = list(chunks[1:])
        else:
            # HF visual model returns (features, deepstack_features)
            # features: [total_tokens, hidden_dim] - flattened across batch
            # deepstack_features: List of [total_tokens, hidden_dim]
            features, deepstack_features = out

        # Reshape from flattened batch to proper batch structure
        # Split by grid_thw to get per-image features
        batch_size = grid_thw.shape[0]
        merge_size = 2  # Qwen3-VL uses 2x2 spatial merge

        # Calculate tokens per image from grid_thw
        tokens_per_image = []
        for i in range(batch_size):
            t, h, w = grid_thw[i].tolist()
            num_tokens = (t * h * w) // (merge_size ** 2)
            tokens_per_image.append(num_tokens)

        # Split features by token counts
        features_list = []
        deepstack_list = [[] for _ in range(batch_size)]

        start_idx = 0
        for img_idx, num_tokens in enumerate(tokens_per_image):
            end_idx = start_idx + num_tokens

            # Extract features for this image
            img_features = features[start_idx:end_idx]
            features_list.append(img_features)

            # Extract deepstack for this image
            for ds_idx, ds_features in enumerate(deepstack_features):
                img_ds = ds_features[start_idx:end_idx]
                deepstack_list[img_idx].append(img_ds)

            start_idx = end_idx

        # Stack back into batched tensors for convenience
        # features: [batch, max_tokens, hidden_dim] with padding if needed
        max_tokens = max(tokens_per_image)
        batched_features = torch.zeros(
            (batch_size, max_tokens, features_list[0].shape[-1]),
            dtype=features_list[0].dtype,
            device=features_list[0].device
        )
        batched_deepstack = [
            torch.zeros_like(batched_features)
            for _ in range(len(deepstack_features))
        ]

        for img_idx in range(batch_size):
            num_tokens = tokens_per_image[img_idx]
            batched_features[img_idx, :num_tokens] = features_list[img_idx]
            for ds_idx in range(len(deepstack_features)):
                batched_deepstack[ds_idx][img_idx, :num_tokens] = deepstack_list[img_idx][ds_idx]

        return Qwen3VLEncoderOutput(
            features=batched_features,
            deepstack_features=batched_deepstack,
            grid_thw=grid_thw
        )


if __name__ == "__main__":
    print("Testing Qwen3VLEncoder...")
    encoder = Qwen3VLEncoder(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16
    )
    test_image = Image.new('RGB', (640, 640), color=(255, 0, 0))
    output = encoder.encode_images([test_image])
    print(f"\nOutput:")
    print(f"  Features shape: {output.features.shape}")
    print(f"  Deepstack features: {len(output.deepstack_features)} layers")
    for i, feat in enumerate(output.deepstack_features):
        print(f"    Layer {i}: {feat.shape}")
    print(f"  Grid THW: {output.grid_thw}")
    print("\nEncoder test passed!")
