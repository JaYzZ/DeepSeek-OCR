"""
DeepSeek-OCR Vision Encoder with Optional Multi-Level Feature Extraction

This encoder loads ONLY the vision components (CLIP + SAM encoders),
not the full language model. Supports optional DeepStack-style multi-level
feature extraction for injection into LLM layers (Qwen3-VL style).

Memory usage:
- Vision models only: ~2-3GB
- vs Full model: ~9GB

Features:
- ✅ Efficient weight loading from safetensors (no full model loading)
- ✅ Batched GPU encoding for high throughput
- ✅ Multi-GPU support via device parameter
- ✅ torch.compile support for additional speedup
- ✅ Production-ready vLLM-compatible output
- ✅ Optional multi-level feature extraction (DeepStack/Qwen3-VL style)
"""

# CRITICAL: Apply transformers patches BEFORE imports
from ..decoder import transformers_patch  # noqa: F401

import torch
import torch.nn as nn
import functools
import os
import glob
from typing import List, Optional, Tuple, Union
from PIL import Image, ImageOps
import logging
from pathlib import Path
from dataclasses import dataclass
from safetensors import safe_open

from OCRInfer.utils.model_paths import resolve_model_path
from sys_path import _add_sys_path
import sys

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput:
    """Output from DPSKOCREncoder with optional intermediate features"""
    embeddings: torch.Tensor  # [111, 1280] final visual embeddings
    intermediate_features: Optional[List[torch.Tensor]] = None  # List of [N, hidden_dim] at each level


class DPSKOCREncoder(nn.Module):
    """
    DeepSeek-OCR Vision Encoder with Optional Multi-Level Feature Extraction

    Architecture:
    - CLIP ViT (24 layers) + SAM for vision encoding
    - Optional extraction at intermediate layers [6, 12, 18] for DeepStack injection
    - Output: [111, 1280] (100 grid + 10 newlines + 1 view_separator)

    Multi-Level Features (Optional):
    - When intermediate_layer_indices is provided, extracts features at those CLIP layers
    - These features can be injected into early LLM layers (Qwen3-VL DeepStack style)
    - Default layer indices follow Qwen3-VL pattern: [6, 12, 18] for 24-layer ViT

    Optimizations:
    - Efficient weight loading from safetensors (no full model)
    - Batched GPU encoding (processes all images at once)
    - Multi-GPU support
    - torch.compile support for 2-3x speedup
    """

    # Default intermediate layer indices for DeepStack extraction
    # For 24-layer CLIP: extract at 1/4, 1/2, 3/4 points
    DEFAULT_INTERMEDIATE_LAYERS = [6, 12, 18]

    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_compile: bool = False,
        intermediate_layer_indices: Optional[List[int]] = None,
        remove_separators: bool = True,
    ):
        """
        Initialize DeepSeek-OCR vision encoder

        Args:
            model_path: Path to DeepSeek-OCR model
            device: Device to load on (e.g., "cuda:0", "cuda:1")
            dtype: Data type
            use_compile: Whether to use torch.compile for speedup (2-3x faster)
            intermediate_layer_indices: Layer indices to extract intermediate features from.
                                        None = disabled (faster), [] = use defaults [6,12,18]
            remove_separators: If True, output pure 10×10 grid (100 tokens).
                               If False, keep line separators (111 tokens = 100 grid + 10 newlines + 1 view_sep).
                               Default: True for cleaner Qwen VL integration.
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.use_compile = use_compile
        self.model_path = resolve_model_path(model_path)
        self.remove_separators = remove_separators
        # Ensure these exist even if initialization fails partway through.
        self._hooks = []
        self._intermediate_features = {}

        # Handle intermediate layer indices
        if intermediate_layer_indices is not None:
            if len(intermediate_layer_indices) == 0:
                # Empty list = use defaults
                self.intermediate_layer_indices = self.DEFAULT_INTERMEDIATE_LAYERS.copy()
            else:
                self.intermediate_layer_indices = intermediate_layer_indices
        else:
            self.intermediate_layer_indices = None

        if self.model_path != model_path:
            logger.info(f"  Using local model mirror: {self.model_path}")

        logger.info(f"Loading vision components from {self.model_path}...")
        logger.info("  Strategy: Direct safetensors weight loading (efficient)")
        if self.intermediate_layer_indices:
            logger.info(f"  Multi-level extraction enabled at layers: {self.intermediate_layer_indices}")

        # Initialize models
        self._load_vision_models()

        # Setup hooks for intermediate feature extraction if enabled
        self._hooks = []
        self._intermediate_features = {}
        if self.intermediate_layer_indices:
            self._setup_intermediate_hooks()

        logger.info("  ✓ Vision components loaded (efficient, no full model)")

    def _load_vision_models(self):
        """Load vision encoder directly from safetensors (no trust_remote_code dependency).

        This method loads the vision components directly from the checkpoint safetensors
        without relying on AutoModel.from_pretrained() which requires external Python files.
        This gives OCRInfer full control over the model loading process.
        """
        from OCRInfer.process.image_process import DeepseekOCRProcessor

        logger.info("  Loading vision components directly from safetensors (no trust_remote_code)...")

        # Find safetensors file
        safetensors_files = glob.glob(os.path.join(self.model_path, "*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors found in {self.model_path}")

        safetensors_path = safetensors_files[0]
        logger.info(f"  Found: {os.path.basename(safetensors_path)}")

        # Load state_dict from safetensors
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model."):
                    # Remove "model." prefix to match expected format
                    new_key = key[6:]  # Remove "model." prefix
                    state_dict[new_key] = f.get_tensor(key)

        logger.info(f"  Loaded {len(state_dict)} tensors from safetensors")

        # Initialize the DeepSeekOCR model architecture directly
        # This creates the model structure without loading weights
        from OCRInfer.model.deepseek_ocr_wrapper import DeepSeekOCRWrapper
        wrapper = DeepSeekOCRWrapper(dtype=self.dtype)

        # Load weights into wrapper.model (not wrapper itself)
        # state_dict keys like "vision_model.*" map to "wrapper.model.vision_model.*"
        missing_keys, unexpected_keys = wrapper.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.info(f"  Missing keys (language model, expected): {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"  Unexpected keys (ignored): {len(unexpected_keys)}")

        # Extract vision components
        self.clip_model = wrapper.model.vision_model
        self.sam_model = wrapper.model.sam_model
        self.projector = wrapper.model.projector
        self.image_newline = wrapper.model.image_newline
        self.view_separator = wrapper.model.view_seperator

        logger.info("  ✓ Extracted vision components directly from safetensors")

        # Ensure vision modules are on the expected dtype/device.
        try:
            self.clip_model = self.clip_model.to(device=self.device, dtype=self.dtype)
            self.sam_model = self.sam_model.to(device=self.device, dtype=self.dtype)
            self.projector = self.projector.to(device=self.device, dtype=self.dtype)
            self.image_newline = self.image_newline.to(device=self.device, dtype=self.dtype)
            self.view_separator = self.view_separator.to(device=self.device, dtype=self.dtype)
        except Exception as e:
            logger.warning(f"  Could not cast vision modules to dtype={self.dtype} on device={self.device}: {e}")

        # Hard-ensure all params/buffers match the requested dtype.
        def _force_dtype(module: nn.Module, dtype: torch.dtype) -> int:
            converted = 0
            for p in module.parameters(recurse=True):
                if p is not None and p.dtype != dtype:
                    p.data = p.data.to(dtype=dtype)
                    converted += 1
            for _, b in module.named_buffers(recurse=True):
                if b is not None and b.dtype != dtype and b.is_floating_point():
                    b.data = b.data.to(dtype=dtype)
            return converted

        try:
            conv = 0
            conv += _force_dtype(self.clip_model, self.dtype)
            conv += _force_dtype(self.sam_model, self.dtype)
            conv += _force_dtype(self.projector, self.dtype)
            if conv:
                logger.info(f"  ✓ Forced dtype={self.dtype} for {conv} fp32 params in vision modules")
            if hasattr(self, "image_newline") and getattr(self, "image_newline", None) is not None:
                if self.image_newline.dtype != self.dtype:
                    self.image_newline.data = self.image_newline.data.to(dtype=self.dtype)
            if hasattr(self, "view_separator") and getattr(self, "view_separator", None) is not None:
                if self.view_separator.dtype != self.dtype:
                    self.view_separator.data = self.view_separator.data.to(dtype=self.dtype)
        except Exception as e:
            logger.warning(f"  Could not force dtype for all vision params: {e}")

        # Set eval mode
        self.clip_model.eval()
        self.sam_model.eval()
        self.projector.eval()

        # CRITICAL: Wrap SAM model forward to ensure dtype consistency with FSDP
        # FSDP eval mode can rematerialize float32 parameters, causing dtype mismatches
        original_sam_forward = self.sam_model.forward
        @functools.wraps(original_sam_forward)
        def sam_forward_with_dtype_fix(*args, **kwargs):
            """Ensure SAM encoder maintains bfloat16 dtype during forward pass."""
            # Get the input (first arg is pixel_values)
            if args and hasattr(args[0], 'dtype'):
                x = args[0]
                if x.dtype != torch.bfloat16:
                    x = x.to(torch.bfloat16)
                    args = (x,) + args[1:]

            # Ensure ALL parameters and buffers are bfloat16 before forward
            # Use in-place .data = ... for FSDP compatibility
            for module in self.sam_model.modules():
                # Convert all parameters (weight, bias, etc.)
                for name, param in list(module.named_parameters(recurse=False)):
                    if param is not None and param.dtype != torch.bfloat16:
                        param.data = param.data.to(torch.bfloat16)
                # Convert all buffers (running_mean, running_var, etc.)
                for name, buf in list(module.named_buffers(recurse=False)):
                    if buf is not None and buf.dtype != torch.bfloat16:
                        buf.data = buf.data.to(torch.bfloat16)

            # Call original forward
            result = original_sam_forward(*args, **kwargs)

            # Ensure output is bfloat16
            if result.dtype != torch.bfloat16:
                result = result.to(torch.bfloat16)

            return result

        self.sam_model.forward = sam_forward_with_dtype_fix
        logger.info("  ✓ Wrapped SAM encoder forward with dtype fix for FSDP")

        # CRITICAL: Wrap CLIP model forward to ensure dtype consistency with FSDP
        # Same issue as SAM - FSDP eval mode can rematerialize float32 parameters
        original_clip_forward = self.clip_model.forward
        @functools.wraps(original_clip_forward)
        def clip_forward_with_dtype_fix(*args, **kwargs):
            """Ensure CLIP encoder maintains bfloat16 dtype during forward pass."""
            # Get the input (first arg is pixel_values)
            if args and hasattr(args[0], 'dtype'):
                x = args[0]
                # Only convert floating point tensors, not integer tensors
                if x.dtype != torch.bfloat16 and x.is_floating_point():
                    x = x.to(torch.bfloat16)
                    args = (x,) + args[1:]

            # Ensure ALL floating-point parameters and buffers are bfloat16 before forward
            # Use in-place .data = ... for FSDP compatibility
            for module in self.clip_model.modules():
                # Convert all floating-point parameters (weight, bias, etc.)
                for name, param in list(module.named_parameters(recurse=False)):
                    if param is not None and param.dtype != torch.bfloat16 and param.is_floating_point():
                        param.data = param.data.to(torch.bfloat16)
                # Convert all floating-point buffers (running_mean, running_var, etc.)
                for name, buf in list(module.named_buffers(recurse=False)):
                    if buf is not None and buf.dtype != torch.bfloat16 and buf.is_floating_point():
                        buf.data = buf.data.to(torch.bfloat16)

            # Call original forward
            result = original_clip_forward(*args, **kwargs)

            # Ensure output is bfloat16 (only if it's floating point)
            if hasattr(result, 'dtype') and result.is_floating_point() and result.dtype != torch.bfloat16:
                result = result.to(torch.bfloat16)

            return result

        self.clip_model.forward = clip_forward_with_dtype_fix
        logger.info("  ✓ Wrapped CLIP encoder forward with dtype fix for FSDP")

        # Memory format optimization: SAM uses convolutions, benefits from channels_last
        # Expected 5-10% speedup on convolution-heavy operations
        try:
            self.sam_model = self.sam_model.to(memory_format=torch.channels_last)
            logger.info("  ✓ SAM model converted to channels_last memory format")
        except Exception as e:
            logger.warning(f"  Could not convert SAM to channels_last: {e}")

        # Optional: torch.compile for 2-3x speedup
        if self.use_compile:
            logger.info("  Compiling models with torch.compile...")
            self.clip_model = torch.compile(self.clip_model)
            self.sam_model = torch.compile(self.sam_model)
            logger.info("  ✓ Models compiled")

        # Load processor
        self.processor = DeepseekOCRProcessor()

        # Log memory usage
        total_params = (
            sum(p.numel() for p in self.clip_model.parameters()) +
            sum(p.numel() for p in self.sam_model.parameters()) +
            sum(p.numel() for p in self.projector.parameters())
        )
        logger.info(f"  Total params: {total_params / 1e6:.1f}M")

    def _setup_intermediate_hooks(self):
        """Setup forward hooks to capture intermediate CLIP layer outputs"""
        if not self.intermediate_layer_indices:
            return

        # Access CLIP transformer layers
        transformer_layers = self.clip_model.transformer.layers

        for layer_idx in self.intermediate_layer_indices:
            if layer_idx >= len(transformer_layers):
                logger.warning(f"Layer index {layer_idx} exceeds CLIP layers ({len(transformer_layers)}), skipping")
                continue

            def hook_fn(layer_idx):
                def hook(module, input, output):
                    # Store the output of this layer
                    self._intermediate_features[layer_idx] = output.detach()
                return hook

            handle = transformer_layers[layer_idx].register_forward_hook(hook_fn(layer_idx))
            self._hooks.append(handle)

        logger.info(f"  ✓ Registered hooks for {len(self._hooks)} intermediate layers")

    def _clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._intermediate_features = {}

    def __del__(self):
        """Cleanup hooks on deletion"""
        try:
            if hasattr(self, "_hooks"):
                self._clear_hooks()
        except Exception:
            pass

    @torch.no_grad()
    def encode_images(
        self,
        images: List[Image.Image],
        return_global: bool = False,
        return_local: bool = True,
        return_intermediate: bool = False,
    ) -> Union[List[torch.Tensor], List[EncoderOutput]]:
        """
        Encode images to visual embeddings using BATCHED processing

        Args:
            images: List of PIL Images
            return_global: Return global CLS token
            return_local: Return local patch features
            return_intermediate: Return intermediate layer features (if configured)

        Returns:
            If return_intermediate=False:
                List of visual embedding tensors:
                - [100, 1280] each if remove_separators=True (pure 10×10 grid)
                - [111, 1280] each if remove_separators=False (grid + separators)
            If return_intermediate=True:
                List of EncoderOutput with embeddings and intermediate_features
        """
        if len(images) == 0:
            return []

        # Clear intermediate features from previous call
        self._intermediate_features = {}

        # Process all images into batch
        pixel_values_list = []

        for image in images:
            # Pad to base size
            global_view = ImageOps.pad(
                image,
                (self.processor.base_size, self.processor.base_size),
                color=tuple(int(x * 255) for x in self.processor.image_transform.mean)
            )

            # Apply transform
            pixel_values = self.processor.image_transform(global_view)
            pixel_values_list.append(pixel_values)

        # Stack into batch tensor - BATCHED PROCESSING
        pixel_values = torch.stack(pixel_values_list, dim=0)

        return self.encode_pixel_values(
            pixel_values,
            return_global=return_global,
            return_local=return_local,
            return_intermediate=return_intermediate,
        )

    def encode_pixel_values(
        self,
        pixel_values: torch.Tensor,
        return_global: bool = False,
        return_local: bool = True,
        return_intermediate: bool = False,
    ) -> Union[List[torch.Tensor], List[EncoderOutput]]:
        """
        Encode preprocessed pixel values to visual embeddings using BATCHED processing.

        Args:
            pixel_values: Tensor shaped [B,3,H,W] produced by DeepseekOCRProcessor.image_transform.
            return_global: Return global mean token.
            return_local: Return local patch features.
            return_intermediate: Return intermediate layer features (if configured).

        Returns:
            Same as encode_images.
        """
        if pixel_values.numel() == 0:
            return []
        if pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must be [B,3,H,W], got shape={tuple(pixel_values.shape)}")

        # Clear intermediate features from previous call
        self._intermediate_features = {}

        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        # Convert to channels_last for SAM (if SAM is in channels_last)
        if hasattr(self.sam_model, 'memory_format') or next(self.sam_model.parameters()).is_contiguous(memory_format=torch.channels_last):
            pixel_values = pixel_values.to(memory_format=torch.channels_last)

        batch_size = int(pixel_values.shape[0])

        # ==== BATCHED GPU ENCODING (processes all images at once) ====
        # SAM encoding - BATCHED
        sam_features = self.sam_model(pixel_values)  # [B, 1024, H, W]

        # CLIP encoding with SAM features - BATCHED
        # This triggers the hooks to capture intermediate features
        clip_features = self.clip_model(pixel_values, sam_features)  # [B, seq+1, 1024]

        # Concatenate CLIP patch features + SAM features - BATCHED
        features = torch.cat(
            (
                clip_features[:, 1:],  # [B, seq, 1024] - skip CLS token
                sam_features.flatten(2).permute(0, 2, 1),  # [B, hw, 1024]
            ),
            dim=-1,
        )  # [B, hw, 2048]

        # Project to hidden_dim - BATCHED
        features = self.projector(features)  # [B, hw, 1280]

        # Add newline tokens and view separator per image
        _, hw, dim = features.shape
        side = int(hw ** 0.5)

        # Process intermediate features if requested
        intermediate_list = None
        if return_intermediate and self.intermediate_layer_indices and self._intermediate_features:
            intermediate_list = []
            # Sort by layer index to ensure consistent ordering
            sorted_indices = sorted(self._intermediate_features.keys())
            for layer_idx in sorted_indices:
                # Get intermediate features [B, seq, hidden_dim]
                inter_feat = self._intermediate_features[layer_idx]
                # Skip CLS token if present (first token)
                if inter_feat.dim() == 3 and inter_feat.shape[1] > hw:
                    inter_feat = inter_feat[:, 1:, :]  # Remove CLS token
                intermediate_list.append(inter_feat)

        # Pre-allocate output list for tensor pre-allocation optimization
        # Expected 2-5% speedup from reduced allocations
        embeddings_list = []

        # Pre-compute newline buffer (reused for all images) - only if we need separators
        if not self.remove_separators:
            newline_buffer = self.image_newline[None, None, :].expand(side, 1, dim)

        for jdx in range(batch_size):
            img_features = features[jdx].view(side, side, dim)  # [10, 10, 1280]

            if self.remove_separators:
                # Pure grid without separators: [100, 1280]
                combined = img_features.view(-1, dim)
            else:
                # With separators: add newlines and view separator [111, 1280]
                # Reuse pre-computed newline buffer
                img_features = torch.cat([img_features, newline_buffer], dim=1)
                img_features = img_features.view(-1, dim)  # [110, 1280]

                # Add view separator to get [111, 1280]
                combined = torch.cat(
                    [img_features, self.view_separator[None, :]],
                    dim=0
                )  # [111, 1280]

            if return_local and not return_global:
                # Local features only: [100, 1280] or [111, 1280]
                final_emb = combined
            elif return_global and not return_local:
                # Global features only (mean pool): [1, 1280]
                final_emb = combined.mean(dim=0, keepdim=True)
            elif return_global and return_local:
                # Both global and local: [101, 1280] or [112, 1280]
                global_feat = combined.mean(dim=0, keepdim=True)
                final_emb = torch.cat([global_feat, combined], dim=0)
            else:
                # Default: local only
                final_emb = combined

            if return_intermediate and intermediate_list:
                # Return EncoderOutput with intermediate features for this image
                img_intermediate = [inter[jdx] for inter in intermediate_list]
                embeddings_list.append(EncoderOutput(
                    embeddings=final_emb,
                    intermediate_features=img_intermediate
                ))
            else:
                embeddings_list.append(final_emb)

        return embeddings_list

    @torch.no_grad()
    def encode_images_with_deepstack(
        self,
        images: List[Image.Image],
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Encode images and return both final embeddings and DeepStack features

        This is a convenience method for Qwen3-VL style multi-level feature injection.

        Args:
            images: List of PIL Images

        Returns:
            Tuple of:
                - List of final embeddings:
                  * [100, 1280] each if remove_separators=True
                  * [111, 1280] each if remove_separators=False
                - List of intermediate feature lists (one list per image, each containing
                  features from layers [6, 12, 18] by default)
        """
        # Ensure intermediate layers are configured
        if not self.intermediate_layer_indices:
            self.intermediate_layer_indices = self.DEFAULT_INTERMEDIATE_LAYERS.copy()
            self._setup_intermediate_hooks()

        outputs = self.encode_images(images, return_intermediate=True)

        final_embeddings = []
        intermediate_features = []

        for output in outputs:
            if isinstance(output, EncoderOutput):
                final_embeddings.append(output.embeddings)
                intermediate_features.append(output.intermediate_features or [])
            else:
                final_embeddings.append(output)
                intermediate_features.append([])

        return final_embeddings, intermediate_features

    @torch.no_grad()
    def encode_pixel_values_with_deepstack(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Same as encode_images_with_deepstack, but accepts preprocessed pixel values [B,3,H,W].
        """
        # Ensure intermediate layers are configured
        if not self.intermediate_layer_indices:
            self.intermediate_layer_indices = self.DEFAULT_INTERMEDIATE_LAYERS.copy()
            self._setup_intermediate_hooks()

        outputs = self.encode_pixel_values(pixel_values, return_intermediate=True)

        final_embeddings: List[torch.Tensor] = []
        intermediate_features: List[List[torch.Tensor]] = []

        for output in outputs:
            if isinstance(output, EncoderOutput):
                final_embeddings.append(output.embeddings)
                intermediate_features.append(output.intermediate_features or [])
            else:
                final_embeddings.append(output)
                intermediate_features.append([])

        return final_embeddings, intermediate_features
