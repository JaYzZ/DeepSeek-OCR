#!/usr/bin/env python3
"""Qwen3-VL vision encoder for benchmarking.

Default path uses vLLM's inference-only ViT implementation to match native
vLLM speed (FlashAttention-varlen, packed seqlens, vLLM linear/conv layers).
Falls back to HF visual module if vLLM sources are unavailable.
"""

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from project_paths import hf_path

from .vllm_qwen_vit import build_qwen3_vit


@dataclass
class Qwen3VLEncoderOutput:
    """Output from Qwen3-VL vision encoder.

    Attributes:
        features: List of per-image feature tensors, each [actual_tokens, hidden_dim]
            No padding - each image has its actual token count based on resolution.
        deepstack_features: List of lists, where each inner list contains per-image
            deepstack features for that layer.
        grid_thw: Tensor of shape [batch_size, 3] with (temporal, height, width) grid
            dimensions for each image.
    """
    features: List[torch.Tensor]  # Changed from Tensor to List[Tensor]
    deepstack_features: List[List[torch.Tensor]]  # Changed from List[Tensor]
    grid_thw: torch.Tensor

    def as_vllm_mm_dict(self) -> dict[str, torch.Tensor]:
        """Return a vLLM-compatible multimodal dict for decoder-only inference.

        Concatenates all per-image features into a single flattened tensor for vLLM.

        vLLM Qwen3-VL expects image_embeds concatenated as:
        [img0_features | img1_features | ...]
        with shape [total_tokens, 2048]
        """
        # Concatenate all per-image features along sequence dimension
        image_embeds = torch.cat(self.features, dim=0)  # [total_tokens, hidden_dim]

        return {
            "image_embeds": image_embeds,
            "image_grid_thw": self.grid_thw,
        }


class Qwen3VLEncoder:
    """Standalone Qwen3-VL vision encoder."""

    def __init__(
        self,
        model_name_or_path: str = str(hf_path("Qwen", "Qwen3-VL-2B-Instruct")),
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
                # Check if this is expected (DDP conflict) vs unexpected error
                if "PyTorch DDP" in str(e) or "world_group is not initialized" in str(e):
                    print(f"vLLM vision kernels incompatible with multi-GPU DDP, falling back to HF vision model")
                else:
                    import traceback
                    print(f"vLLM vision load failed, falling back to HF visual: {e}")
                    traceback.print_exc()

        # Load only vision model weights (skip language model)
        from transformers import AutoConfig, Qwen3VLVisionModel

        cfg = AutoConfig.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )

        # Initialize vision model from config
        vision_config = cfg.vision_config
        self._vision_model = Qwen3VLVisionModel(vision_config)

        # Load only visual.* weights from checkpoint
        print(f"Loading vision weights from {self.model_name_or_path}...")
        from transformers import AutoModel

        # Load state dict on CPU first to filter
        full_model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            dtype=torch.float32,  # Load as float32 to CPU
            device_map="cpu",
        )

        # Extract only visual weights
        vision_state_dict = {
            k.replace("visual.", ""): v
            for k, v in full_model.state_dict().items()
            if k.startswith("visual.")
        }

        # Clean up full model
        del full_model
        import gc
        gc.collect()

        # Load vision weights
        self._vision_model.load_state_dict(vision_state_dict, strict=True)
        self._vision_model = self._vision_model.to(dtype=self.dtype, device=self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._vision_model.eval()
        print(f"Vision encoder loaded successfully to {self.device} (vision-only)")
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
        grid_thw_cpu = processed['image_grid_thw']

        # Run vision encoder.
        vision_model = self.vision_model
        grid_thw = grid_thw_cpu if self._out_hidden_size is not None else grid_thw_cpu.to(device=self.device)
        out = vision_model(pixel_values, grid_thw)
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
        batch_size = grid_thw_cpu.shape[0]
        merge_size = 2  # Qwen3-VL uses 2x2 spatial merge

        # Calculate tokens per image from grid_thw
        tokens_per_image = []
        for i in range(batch_size):
            t, h, w = grid_thw_cpu[i].tolist()
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

        # Return unpadded per-image features as a list
        # This preserves true any-resolution efficiency without padding overhead
        # features_list: List of [actual_tokens, hidden_dim] tensors
        return Qwen3VLEncoderOutput(
            features=features_list,  # List[Tensor], not batched!
            deepstack_features=deepstack_list,  # List of lists
            grid_thw=grid_thw_cpu
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
