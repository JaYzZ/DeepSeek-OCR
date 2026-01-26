"""
DeepSeek-OCR Cross-Attention Encoder for Latent Supervision

This encoder implements a training-free feature combination mechanism using
pretrained cross-attention from the final CLIP transformer layer.

Key Idea:
- Query (Q): Image features from L-1 layer output (original question image)
- Key/Value (K=V): CoT text features from L-1 layer output (rendered thinking)
- Cross-attention: Final pretrained CLIP layer combines Q with K=V
- Output: Always 100 tokens (same as single image encoding)

Use Case:
- Combine image I and thinking text T for latent token supervision
- Training-free: Uses pretrained cross-attention weights
- Handles multiple images:
  * Multiple T: Cross-attention attends to all, output 100 tokens
  * Multiple I: Select primary as Q, put others in K=V

Example:
    encoder = DPSKOCRCrossAttentionEncoder(device="cuda:0")
    query_images = [image1, image2]  # Primary images (Q)
    kv_images = [cot_image1, cot_image2]  # CoT rendered images (K=V)

    # Output: list of [100, 1280] tensors
    features = encoder.combine_images(query_images, kv_images)
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
class CrossAttentionOutput:
    """Output from cross-attention combination"""
    combined_features: torch.Tensor  # [100, 1280] combined features
    query_features: Optional[torch.Tensor] = None  # [100, 1280] Q features (L-1)
    kv_features: Optional[torch.Tensor] = None  # [N_kv, 1280] K=V features (L-1)
    attention_weights: Optional[torch.Tensor] = None  # [100, N_kv] attention map


class DPSKOCRCrossAttentionEncoder(nn.Module):
    """
    DeepSeek-OCR Cross-Attention Encoder for Feature Combination

    This encoder uses the pretrained CLIP transformer's final layer as a
    cross-attention mechanism to combine query image features with key-value
    features from rendered thinking text.

    Architecture:
    1. Extract L-1 layer outputs from CLIP for both Q and K=V
    2. Use final CLIP transformer layer as cross-attention:
       - Q = L-1 output from query images
       - K = V = L-1 output from KV images
    3. Project to 1280-dim via DPSK projector
    4. Output 100 tokens (10x10 grid, training-free)

    Key Properties:
    - Training-free: Uses only pretrained weights
    - No gradient computation: @torch.no_grad()
    - Preserves 100-token output: Same shape as single image
    - Handles variable numbers of Q and K=V images
    """

    # CLIP ViT-L/14 has 24 layers
    NUM_CLIP_LAYERS = 24
    # L-1 layer index (23 for 0-indexed)
    L_MINUS_1_LAYER = NUM_CLIP_LAYERS - 2  # Layer 22 (0-indexed)

    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        primary_query_idx: int = 0,
    ):
        """
        Initialize cross-attention encoder

        Args:
            model_path: Path to DeepSeek-OCR model
            device: Device to load on
            dtype: Data type
            primary_query_idx: When multiple query images, which one to use as Q
                             (others go into K=V)
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_path = resolve_model_path(model_path)
        self.primary_query_idx = primary_query_idx
        self.num_clip_layers = self.NUM_CLIP_LAYERS
        self.l_minus_1_layer = self.L_MINUS_1_LAYER

        # Hook storage for L-1 layer outputs
        self._l_minus_1_output = None
        self._hooks = []

        logger.info(f"Initializing Cross-Attention Encoder from {self.model_path}")
        logger.info(f"  Device: {device}, dtype: {dtype}")
        logger.info(f"  L-1 layer: {self.l_minus_1_layer}/{self.num_clip_layers}")

        # Load vision components
        self._load_vision_models()

        logger.info("  ✓ Cross-Attention encoder ready")

    def _load_vision_models(self):
        """Load vision encoder components from safetensors"""
        from OCRInfer.process.image_process import DeepseekOCRProcessor

        logger.info("  Loading vision components from safetensors...")

        # Find safetensors file
        safetensors_files = glob.glob(os.path.join(self.model_path, "*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors found in {self.model_path}")

        safetensors_path = safetensors_files[0]
        logger.info(f"  Found: {os.path.basename(safetensors_path)}")

        # Load state_dict
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model."):
                    new_key = key[6:]  # Remove "model." prefix
                    state_dict[new_key] = f.get_tensor(key)

        # Initialize model wrapper
        from OCRInfer.model.deepseek_ocr_wrapper import DeepSeekOCRWrapper
        wrapper = DeepSeekOCRWrapper(dtype=self.dtype)
        wrapper.model.load_state_dict(state_dict, strict=False)

        # Extract components
        self.clip_model = wrapper.model.vision_model
        self.sam_model = wrapper.model.sam_model
        self.projector = wrapper.model.projector

        # Move to device and set dtype
        self.clip_model = self.clip_model.to(device=self.device, dtype=self.dtype)
        self.sam_model = self.sam_model.to(device=self.device, dtype=self.dtype)
        self.projector = self.projector.to(device=self.device, dtype=self.dtype)

        # Set eval mode
        self.clip_model.eval()
        self.sam_model.eval()
        self.projector.eval()

        # Wrap forward passes for dtype consistency
        self._wrap_forward_passes()

        # Load processor
        self.processor = DeepseekOCRProcessor()

        logger.info("  ✓ Vision components loaded")

    def _wrap_forward_passes(self):
        """Wrap forward passes for dtype consistency"""
        original_sam_forward = self.sam_model.forward
        @functools.wraps(original_sam_forward)
        def sam_forward_with_dtype_fix(*args, **kwargs):
            if args and hasattr(args[0], 'dtype'):
                x = args[0]
                if x.dtype != torch.bfloat16:
                    x = x.to(torch.bfloat16)
                    args = (x,) + args[1:]
            for module in self.sam_model.modules():
                for name, param in list(module.named_parameters(recurse=False)):
                    if param is not None and param.dtype != torch.bfloat16:
                        param.data = param.data.to(torch.bfloat16)
                for name, buf in list(module.named_buffers(recurse=False)):
                    if buf is not None and buf.dtype != torch.bfloat16:
                        buf.data = buf.data.to(torch.bfloat16)
            result = original_sam_forward(*args, **kwargs)
            if result.dtype != torch.bfloat16:
                result = result.to(torch.bfloat16)
            return result
        self.sam_model.forward = sam_forward_with_dtype_fix

        original_clip_forward = self.clip_model.forward
        @functools.wraps(original_clip_forward)
        def clip_forward_with_dtype_fix(*args, **kwargs):
            if args and hasattr(args[0], 'dtype'):
                x = args[0]
                if x.dtype != torch.bfloat16 and x.is_floating_point():
                    x = x.to(torch.bfloat16)
                    args = (x,) + args[1:]
            for module in self.clip_model.modules():
                for name, param in list(module.named_parameters(recurse=False)):
                    if param is not None and param.dtype != torch.bfloat16 and param.is_floating_point():
                        param.data = param.data.to(torch.bfloat16)
                for name, buf in list(module.named_buffers(recurse=False)):
                    if buf is not None and buf.dtype != torch.bfloat16 and buf.is_floating_point():
                        buf.data = buf.data.to(torch.bfloat16)
            result = original_clip_forward(*args, **kwargs)
            if hasattr(result, 'dtype') and result.is_floating_point() and result.dtype != torch.bfloat16:
                result = result.to(torch.bfloat16)
            return result
        self.clip_model.forward = clip_forward_with_dtype_fix

    def _register_l_minus_1_hook(self):
        """Register forward hook to capture L-1 layer output"""
        if self._hooks:
            return  # Already registered

        transformer_layers = self.clip_model.transformer.layers
        l_minus_1_idx = self.l_minus_1_layer

        def hook_fn(module, input, output):
            self._l_minus_1_output = output.detach()

        handle = transformer_layers[l_minus_1_idx].register_forward_hook(hook_fn)
        self._hooks.append(handle)
        logger.debug(f"  Registered L-1 hook at layer {l_minus_1_idx}")

    def _clear_hooks(self):
        """Remove all hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._l_minus_1_output = None

    @torch.no_grad()
    def _encode_to_l_minus_1(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images and extract L-1 layer output

        Args:
            pixel_values: [B, 3, H, W] preprocessed images

        Returns:
            [B, seq, 1024] L-1 layer output (before final transformer layer)
        """
        # Register hook to capture L-1 output
        self._register_l_minus_1_hook()
        self._l_minus_1_output = None

        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        # SAM encoding
        sam_features = self.sam_model(pixel_values)  # [B, 1024, H, W]

        # CLIP encoding (hook captures L-1 output)
        # Note: We need to run through L-1 layer, not the full encoder
        clip_features = self.clip_model(pixel_values, sam_features)  # [B, seq+1, 1024]

        # Return L-1 output (captured by hook)
        # If hook didn't capture (shouldn't happen), return final output
        if self._l_minus_1_output is not None:
            return self._l_minus_1_output
        else:
            logger.warning("L-1 hook didn't capture output, using final layer")
            return clip_features

    @torch.no_grad()
    def _cross_attention_combine(
        self,
        query_features: torch.Tensor,
        kv_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Combine Q and K=V features using pretrained final layer as cross-attention

        Args:
            query_features: [B_q, seq_q, hidden] Query features (L-1 output)
            kv_features: [B_kv, seq_kv, hidden] Key/Value features (L-1 output)
            return_attention: Return attention weights for visualization

        Returns:
            Tuple of:
                - combined: [B_q, 100, 1280] Combined features (projected)
                - attn_weights: [B_q, 100, seq_kv] Attention weights (if requested)
        """
        # Get the final transformer layer (pretrained cross-attention)
        final_layer = self.clip_model.transformer.layers[-1]

        # Reshape for cross-attention
        # Standard CLIP expects [batch, seq, hidden]
        # We need to manually compute Q@K^T @ V

        hidden_dim = query_features.shape[-1]  # 1024 for CLIP
        num_heads = final_layer.num_heads  # Usually 16
        head_dim = hidden_dim // num_heads

        # Project Q, K, V using pretrained weights
        # Q = query_features (already projected by q_proj)
        # K = V = kv_features (already projected by k_proj, v_proj)

        # Get Q projection
        q = final_layer.q_proj(query_features)  # [B_q, seq_q, hidden]
        q = q.view(query_features.shape[0], query_features.shape[1], num_heads, head_dim)
        q = q.transpose(1, 2)  # [B_q, num_heads, seq_q, head_dim]

        # Get K, V projections
        k = final_layer.k_proj(kv_features)  # [B_kv, seq_kv, hidden]
        k = k.view(kv_features.shape[0], kv_features.shape[1], num_heads, head_dim)
        k = k.transpose(1, 2)  # [B_kv, num_heads, seq_kv, head_dim]

        v = final_layer.v_proj(kv_features)  # [B_kv, seq_kv, hidden]
        v = v.view(kv_features.shape[0], kv_features.shape[1], num_heads, head_dim)
        v = v.transpose(1, 2)  # [B_kv, num_heads, seq_kv, head_dim]

        # Handle batch mismatch: if B_q != B_kv, we need to broadcast
        # For cross-attention: each Q attends to all K=V
        if q.shape[0] != k.shape[0]:
            # Broadcast K and V to match Q batch
            k = k.expand(q.shape[0], -1, -1, -1)
            v = v.expand(q.shape[0], -1, -1, -1)

        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B_q, num_heads, seq_q, seq_kv]
        attn_scores = attn_scores / (head_dim ** 0.5)

        # Apply softmax (no causal mask for cross-attention)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention to V: attn_weights @ V
        attn_output = torch.matmul(attn_weights, v)  # [B_q, num_heads, seq_q, head_dim]

        # Reshape back: [B_q, seq_q, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(q.shape[0], q.shape[2], hidden_dim)

        # Apply output projection (pretrained)
        combined_features = final_layer.out_proj(attn_output)  # [B_q, seq_q, hidden]

        # Remove CLS token if present (index 0)
        if combined_features.shape[1] > 100:
            combined_features = combined_features[:, 1:, :]  # [B_q, 100, hidden]

        # Concatenate with SAM features and project to 1280
        # For now, we only have CLIP features at L-1, need SAM features too
        # Actually, for combination, we should project CLIP-only to 1280
        # The projector expects [B, seq, 2048] = CLIP + SAM

        # For simplicity: project CLIP features directly using a sub-layer of projector
        # The projector is: Linear(2048 -> 1280)
        # We need to handle 1024 -> 1280 for CLIP-only

        # Use first half of projector (CLIP part) or create separate projection
        # Projector weight shape: [1280, 2048]
        # We'll use a simple linear projection for 1024 -> 1280
        if not hasattr(self, '_clip_projector'):
            self._clip_projector = nn.Linear(1024, 1280, bias=False).to(
                device=self.device, dtype=self.dtype
            )
            # Initialize from first 1024 columns of projector
            with torch.no_grad():
                projector_weight = self.projector.weight[:, :1024]  # [1280, 1024]
                self._clip_projector.weight.copy_(projector_weight)
            self._clip_projector.eval()

        combined_features = self._clip_projector(combined_features)  # [B_q, 100, 1280]

        if return_attention:
            # Average attention across heads: [B_q, seq_q, seq_kv]
            attn_weights_avg = attn_weights.mean(dim=1)
            # Limit to first 100 positions (Q side)
            attn_weights_avg = attn_weights_avg[:, :100, :]
            return combined_features, attn_weights_avg
        else:
            return combined_features, None

    @torch.no_grad()
    def combine_images(
        self,
        query_images: List[Image.Image],
        kv_images: List[Image.Image],
        return_attention: bool = False,
        return_components: bool = False,
    ) -> List[Union[torch.Tensor, CrossAttentionOutput]]:
        """
        Combine query images and KV images using pretrained cross-attention

        Args:
            query_images: List of query images (Q) - original question images
            kv_images: List of key-value images (K=V) - rendered CoT text
            return_attention: Return attention weights for visualization
            return_components: Return separate Q and K=V features

        Returns:
            List of combined features [100, 1280] or CrossAttentionOutput
        """
        if not query_images:
            raise ValueError("query_images cannot be empty")
        if not kv_images:
            raise ValueError("kv_images cannot be empty")

        # Preprocess query images
        pixel_values_list = []
        for image in query_images:
            global_view = ImageOps.pad(
                image,
                (self.processor.base_size, self.processor.base_size),
                color=tuple(int(x * 255) for x in self.processor.image_transform.mean)
            )
            pixel_values = self.processor.image_transform(global_view)
            pixel_values_list.append(pixel_values)
        query_pixel_values = torch.stack(pixel_values_list, dim=0)

        # Preprocess KV images
        pixel_values_list = []
        for image in kv_images:
            global_view = ImageOps.pad(
                image,
                (self.processor.base_size, self.processor.base_size),
                color=tuple(int(x * 255) for x in self.processor.image_transform.mean)
            )
            pixel_values = self.processor.image_transform(global_view)
            pixel_values_list.append(pixel_values)
        kv_pixel_values = torch.stack(pixel_values_list, dim=0)

        return self.combine_pixel_values(
            query_pixel_values,
            kv_pixel_values,
            return_attention=return_attention,
            return_components=return_components,
        )

    @torch.no_grad()
    def combine_pixel_values(
        self,
        query_pixel_values: torch.Tensor,
        kv_pixel_values: torch.Tensor,
        return_attention: bool = False,
        return_components: bool = False,
    ) -> List[Union[torch.Tensor, CrossAttentionOutput]]:
        """
        Combine preprocessed query and KV pixel values

        Args:
            query_pixel_values: [B_q, 3, H, W] query images
            kv_pixel_values: [B_kv, 3, H, W] key-value images
            return_attention: Return attention weights
            return_components: Return separate Q and K=V features

        Returns:
            List of combined features [100, 1280] or CrossAttentionOutput
        """
        # Encode queries to L-1
        query_features_l1 = self._encode_to_l_minus_1(query_pixel_values)  # [B_q, seq, 1024]

        # Encode KV to L-1
        kv_features_l1 = self._encode_to_l_minus_1(kv_pixel_values)  # [B_kv, seq, 1024]

        # Remove CLS token for cross-attention
        if query_features_l1.shape[1] > 100:
            query_features_l1 = query_features_l1[:, :101, :]  # Keep CLS + 100 tokens
        else:
            query_features_l1 = query_features_l1[:, :, :]

        if kv_features_l1.shape[1] > 100:
            kv_features_l1 = kv_features_l1[:, :101, :]  # Keep CLS + 100 tokens
        else:
            kv_features_l1 = kv_features_l1[:, :, :]

        # Cross-attention combine
        combined, attn_weights = self._cross_attention_combine(
            query_features_l1,
            kv_features_l1,
            return_attention=return_attention,
        )

        # Convert to list of outputs
        outputs = []
        batch_size = combined.shape[0]

        for i in range(batch_size):
            if return_components:
                # Extract per-sample features
                q_feat = query_features_l1[i]  # [seq, 1024]
                kv_feat = kv_features_l1  # [B_kv, seq, 1024] - all KV

                # Remove CLS from Q for output
                q_feat_out = q_feat[1:101, :] if q_feat.shape[0] > 100 else q_feat[:, :]

                outputs.append(CrossAttentionOutput(
                    combined_features=combined[i],  # [100, 1280]
                    query_features=q_feat_out,  # [100, 1024]
                    kv_features=kv_feat,  # [B_kv, seq, 1024]
                    attention_weights=attn_weights[i] if attn_weights is not None else None,
                ))
            else:
                outputs.append(combined[i])  # [100, 1280]

        return outputs

    def __del__(self):
        """Cleanup hooks on deletion"""
        try:
            if hasattr(self, "_hooks"):
                self._clear_hooks()
        except Exception:
            pass


def main():
    """CLI for testing cross-attention encoder"""
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(
        description="Test DPSK Cross-Attention Encoder"
    )
    parser.add_argument(
        "--query-image",
        type=str,
        required=True,
        help="Path to query image (original question image)"
    )
    parser.add_argument(
        "--kv-image",
        type=str,
        required=True,
        help="Path to KV image (rendered CoT text)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Path to DeepSeek-OCR model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save combined features (.pt)"
    )

    args = parser.parse_args()

    # Load images
    query_img = Image.open(args.query_image).convert("RGB")
    kv_img = Image.open(args.kv_image).convert("RGB")

    # Initialize encoder
    encoder = DPSKOCRCrossAttentionEncoder(
        model_path=args.model_path,
        device=args.device,
    )

    # Combine
    logger.info("Combining features...")
    results = encoder.combine_images(
        query_images=[query_img],
        kv_images=[kv_img],
        return_attention=True,
        return_components=True,
    )

    # Output
    result = results[0]
    logger.info(f"Combined features shape: {result.combined_features.shape}")
    logger.info(f"Query features shape: {result.query_features.shape}")
    logger.info(f"KV features shape: {result.kv_features.shape}")
    if result.attention_weights is not None:
        logger.info(f"Attention weights shape: {result.attention_weights.shape}")

    # Save if requested
    if args.output:
        torch.save({
            "combined": result.combined_features,
            "query": result.query_features,
            "kv": result.kv_features,
            "attention": result.attention_weights,
        }, args.output)
        logger.info(f"Saved to {args.output}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
