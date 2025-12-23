"""OCR-aligned Qwen3-VL wrappers + OCR text/image adapter.

These wrappers keep the official Qwen3-VL architecture intact while making it
work with DeepSeek-OCR's visual-token-first inputs:

- Allows feeding pre-encoded OCR visual tokens via `ocr_image_features`
  (e.g., when text has been rendered into visual tokens already).
- Optionally scales vision/text embeddings (`vision_scale`, `text_scale`) to
  bias the model toward visual evidence while keeping short textual prompts.
- Optionally computes an alignment loss against Qwen3-VL's native ViT features
  (training only; during inference you can omit pixel inputs entirely).
- Registers with HuggingFace AutoModel so loading Qwen3-VL checkpoints will
  instantiate this OCR-aware variant once the module is imported.

This module also contains `Qwen3VLOCRTextAdapter`, which renders dense text into
images, encodes them with DeepSeek-OCR vision, and returns `(input_ids,
ocr_image_features)` so callers can keep prompts short while feeding content
through the OCR encoder.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLCausalLMOutputWithPast,
)
from transformers.utils import is_torchdynamo_compiling

from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
from OCRInfer.utils.model_paths import resolve_model_path
from sys_path import _add_sys_path

# Try to import the text renderer from the local DeepSeek-OCR vLLM server utilities.
_THIS_DIR = os.path.dirname(__file__)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))  # .../DeepSeek-OCR
_DS_OCR_SERVER = os.path.join(_ROOT, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm", "server")
_add_sys_path(_DS_OCR_SERVER)

try:
    from text_renderer import chunk_text_by_tokens, render_text_to_image
except Exception:  # pragma: no cover
    chunk_text_by_tokens = None
    render_text_to_image = None

try:
    from Renderer import VelloRenderer  # type: ignore
except Exception:  # pragma: no cover
    VelloRenderer = None

try:
    from Renderer import SkiaRenderer  # type: ignore
except Exception:  # pragma: no cover
    SkiaRenderer = None


def _ensure_renderer():
    if chunk_text_by_tokens is None or render_text_to_image is None:
        raise ImportError(
            "text_renderer utilities not found. Ensure DeepSeek-OCR-vllm/server is available."
        )
    return chunk_text_by_tokens, render_text_to_image


def _normalize_ocr_features(
    feats: Union[torch.Tensor, List[torch.Tensor], tuple],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[List[torch.Tensor], Optional[List[List[torch.Tensor]]]]:
    """Convert various feature container types into a list of [T, C] tensors + optional deepstack."""
    deepstack: Optional[List[List[torch.Tensor]]] = None
    if isinstance(feats, tuple) and len(feats) == 2:
        feats, deepstack = feats  # type: ignore[assignment]

    if isinstance(feats, torch.Tensor):
        if feats.ndim == 2:
            feats_list = [feats]
        elif feats.ndim == 3:
            feats_list = [feats[i] for i in range(feats.shape[0])]
        else:
            raise ValueError("ocr_image_features must be [T,C] or [B,T,C]")
    elif isinstance(feats, list):
        feats_list = [torch.as_tensor(f) for f in feats]
    else:
        raise ValueError("Unsupported ocr_image_features type")

    feats_list = [f.to(device=device, dtype=dtype) for f in feats_list]

    deepstack_list = None
    if deepstack is not None:
        deepstack_list = []
        for per_img in deepstack:
            deepstack_list.append([torch.as_tensor(t, device=device, dtype=dtype) for t in per_img])

    return feats_list, deepstack_list


def _infer_square_grid(num_tokens: int, spatial_merge_size: int, *, device, dtype) -> torch.Tensor:
    """Best-effort guess for (t,h,w) grid when caller skips image_grid_thw.

    Qwen3-VL expects image_grid_thw to compute rotary positions. If it's not
    provided alongside pre-encoded features, we approximate a square grid so
    downstream position ids remain roughly aligned.
    """
    patches = num_tokens * (spatial_merge_size ** 2)
    side = int(round(patches ** 0.5))
    if side * side != patches:
        # Fall back to a degenerate 1xN strip if perfect square not found.
        return torch.tensor([1, patches, 1], device=device, dtype=dtype)
    return torch.tensor([1, side, side], device=device, dtype=dtype)


def _default_ocr_grid_thw(config, *, device) -> torch.LongTensor:
    """OCR features are 111 tokens: 10x10 visual + separators; treat grid as 10x10 by default."""
    grid = getattr(config, "ocr_grid_thw", None)
    if grid is not None and len(grid) == 3:
        t, h, w = grid
        return torch.tensor([t, h, w], device=device, dtype=torch.long)
    size = getattr(config, "ocr_grid_size", None)
    if size is not None:
        return torch.tensor([1, int(size), int(size)], device=device, dtype=torch.long)
    return torch.tensor([1, 10, 10], device=device, dtype=torch.long)


def _extract_visual_tokens(
    embeds: torch.Tensor, grid_thw: torch.LongTensor, *, device: torch.device
) -> torch.Tensor:
    """Return visual-only tokens (drop newline/view-sep tokens for OCR layout)."""
    _, h, w = (int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2]))
    visual_len = h * w
    total = embeds.shape[0]
    # Common OCR layout: visual_len + h newlines + 1 view separator.
    if total == visual_len + h + 1:
        row_stride = w + 1
        idxs = []
        for r in range(h):
            start = r * row_stride
            idxs.extend(range(start, start + w))
        return embeds.index_select(0, torch.tensor(idxs, device=device))
    return embeds[:visual_len]


def _pool_ref_to_ocr_grid(
    ref_embeds: torch.Tensor, ref_grid_thw: torch.LongTensor, ocr_grid_thw: torch.LongTensor
) -> torch.Tensor:
    """Average-pool reference ViT tokens to OCR visual grid size."""
    import torch.nn.functional as F

    _, ref_h, ref_w = (int(ref_grid_thw[0]), int(ref_grid_thw[1]), int(ref_grid_thw[2]))
    _, ocr_h, ocr_w = (int(ocr_grid_thw[0]), int(ocr_grid_thw[1]), int(ocr_grid_thw[2]))
    visual_len = ref_h * ref_w
    ref_visual = ref_embeds[:visual_len]
    ref_visual = ref_visual.view(ref_h, ref_w, -1).permute(2, 0, 1).unsqueeze(0)  # [1,C,ref_h,ref_w]
    pooled = F.adaptive_avg_pool2d(ref_visual, (ocr_h, ocr_w))
    pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(ocr_h * ocr_w, -1)
    return pooled


class OCRQwen3VLModel(Qwen3VLModel):
    """Drop-in Qwen3-VL model that accepts OCR visual features and scaling."""

    config_class = Qwen3VLConfig

    def _init_ocr_connector(self, in_dim: int, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Build connector following LLaVA-1.5/1.6 practice (mlp2x_gelu)."""
        target_dim = self.config.text_config.hidden_size
        if in_dim == target_dim:
            connector: nn.Module = nn.Identity()
        else:
            # LLaVA-1.5/1.6 standard: mlp2x_gelu (Linear → GELU → Linear)
            connector = nn.Sequential(
                nn.Linear(in_dim, target_dim),
                nn.GELU(),
                nn.Linear(target_dim, target_dim)
            )
        return connector.to(device=device, dtype=dtype)

    def _maybe_get_ocr_connector(self, in_dim: int, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        if not hasattr(self, "ocr_connector") or self.ocr_connector is None:  # type: ignore[attr-defined]
            self.ocr_connector = self._init_ocr_connector(in_dim, device=device, dtype=dtype)  # type: ignore[attr-defined]
        else:
            # If dims changed (unlikely), rebuild.
            conn = self.ocr_connector  # type: ignore[attr-defined]
            if isinstance(conn, nn.Linear) and conn.in_features != in_dim:
                self.ocr_connector = self._init_ocr_connector(in_dim, device=device, dtype=dtype)  # type: ignore[attr-defined]
        return self.ocr_connector  # type: ignore[attr-defined]

    def _maybe_get_deepstack_connector(self, in_dim: int, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Deepstack levels connector - also uses LLaVA-1.5/1.6 mlp2x_gelu."""
        if not hasattr(self, "_ocr_deepstack_connectors") or self._ocr_deepstack_connectors is None:  # type: ignore[attr-defined]
            self._ocr_deepstack_connectors = nn.ModuleDict()  # type: ignore[attr-defined]
        key = str(in_dim)
        connectors: nn.ModuleDict = self._ocr_deepstack_connectors  # type: ignore[attr-defined]
        if key not in connectors:
            target_dim = self.config.text_config.hidden_size
            if in_dim == target_dim:
                connectors[key] = nn.Identity()
            else:
                # LLaVA-1.5/1.6 standard: mlp2x_gelu
                connectors[key] = nn.Sequential(
                    nn.Linear(in_dim, target_dim),
                    nn.GELU(),
                    nn.Linear(target_dim, target_dim)
                )
            connectors[key] = connectors[key].to(device=device, dtype=dtype)
        return connectors[key]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ocr_image_features: Optional[Union[torch.Tensor, List[torch.Tensor], tuple]] = None,
        # Alignment-only inputs: keep ViT on for reference features during training.
        pixel_values_ref: Optional[torch.Tensor] = None,
        ocr_alignment: Optional[bool] = None,
        ocr_alignment_weight: Optional[float] = None,
        vision_scale: Optional[float] = None,
        text_scale: Optional[float] = None,
        **kwargs: Any,
    ):
        cfg_vision_scale = float(getattr(self.config, "ocr_vision_scale", 1.0))
        cfg_text_scale = float(getattr(self.config, "ocr_text_scale", 1.0))
        vision_scale = float(vision_scale) if vision_scale is not None else cfg_vision_scale
        text_scale = float(text_scale) if text_scale is not None else cfg_text_scale

        # This OCR-Qwen wrapper consumes OCR pre-encoded visual tokens.
        # Native pixel_values -> ViT -> LM path is only used as a *reference* for alignment.
        if ocr_image_features is None:
            raise ValueError(
                "ocr_image_features is required. Provide OCR features (and <image> placeholders in input_ids). "
                "For training-time alignment, also provide pixel_values/pixel_values_ref with ocr_alignment=True."
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        ocr_alignment = bool(ocr_alignment) if ocr_alignment is not None else False

        if pixel_values is not None and not ocr_alignment:
            raise ValueError(
                "pixel_values is only accepted for alignment (set ocr_alignment=True). "
                "For the LM path, provide ocr_image_features."
            )

        ocr_image_embeds = None
        ref_image_embeds = None
        ref_deepstack_embeds = None

        if ocr_image_features is not None:
            image_feats, deepstack_feats = _normalize_ocr_features(
                ocr_image_features,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

            require_deepstack = bool(getattr(self.config, "ocr_require_deepstack", True))
            if require_deepstack:
                if deepstack_feats is None:
                    raise ValueError(
                        "Qwen3-VL OCR path requires deepstack features. "
                        "Pass ocr_image_features=(final_feats, deepstack_feats) where deepstack_feats has 3 levels per image."
                    )
                for per_img in deepstack_feats:
                    if len(per_img) != 3:
                        raise ValueError(f"Expected 3 deepstack levels per image for Qwen3-VL, got {len(per_img)}.")

            conn_final = self._maybe_get_ocr_connector(
                image_feats[0].shape[-1], device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            image_feats = [conn_final(f) for f in image_feats]
            image_embeds = torch.cat(image_feats, dim=0)
            ocr_image_embeds = image_embeds
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            if vision_scale != 1.0:
                image_embeds = image_embeds * vision_scale
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # If caller skipped image_grid_thw, try to approximate it so RoPE stays consistent.
            if image_grid_thw is None:
                base_grid = _default_ocr_grid_thw(self.config, device=inputs_embeds.device)
                image_grid_thw = base_grid.unsqueeze(0).expand(len(image_feats), -1).contiguous()
            if deepstack_feats is not None:
                # deepstack_feats is a list of per-image features
                # Each per-image is a list of levels: [[level0, level1, level2], [level0, level1, level2], ...]
                # We need to transpose to: [all_level0s, all_level1s, all_level2s]
                per_level_lists = []
                num_levels = len(deepstack_feats[0]) if deepstack_feats else 0

                for level_idx in range(num_levels):
                    level_features = []
                    for per_img in deepstack_feats:
                        t = per_img[level_idx].to(inputs_embeds.device, inputs_embeds.dtype)
                        # Check if this is the same dimension as final features
                        # If not, use deepstack-specific connector
                        if hasattr(conn_final, 'in_features') and t.shape[-1] == conn_final.in_features:
                            # Same dimension as final features, can reuse connector
                            level_features.append(conn_final(t))
                        else:
                            # Different dimension, need deepstack connector
                            conn_level = self._maybe_get_deepstack_connector(
                                t.shape[-1], device=inputs_embeds.device, dtype=inputs_embeds.dtype
                            )
                            level_features.append(conn_level(t))
                    # Concatenate all images for this level
                    per_level_lists.append(torch.cat(level_features, dim=0))

                deepstack_image_embeds = per_level_lists
        # Note: we intentionally do not support injecting native ViT features into the LM path here.
        # If you want native ViT outputs, compute them via `ocr_alignment=True` reference path below.

        # Reference ViT path for alignment (does not affect inputs_embeds).
        if ocr_alignment:
            ref_pixels = pixel_values_ref if pixel_values_ref is not None else pixel_values
            if ref_pixels is not None:
                ref_image_list, ref_deepstack = self.get_image_features(ref_pixels, image_grid_thw)
                ref_image_embeds = torch.cat(ref_image_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                if ref_deepstack is not None:
                    ref_deepstack_embeds = [
                        torch.cat(level, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                        for level in ref_deepstack
                    ]

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            if vision_scale != 1.0:
                video_embeds = video_embeds * vision_scale
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if text_scale != 1.0:
            text_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
            if image_mask is not None:
                text_mask = text_mask & (~image_mask[..., 0])
            if video_mask is not None:
                text_mask = text_mask & (~video_mask[..., 0])
            inputs_embeds[text_mask] = inputs_embeds[text_mask] * text_scale

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        out = Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

        # Attach alignment artifacts for the wrapper to consume.
        if ocr_alignment and ocr_image_embeds is not None and ref_image_embeds is not None:
            w = float(ocr_alignment_weight) if ocr_alignment_weight is not None else float(
                getattr(self.config, "ocr_alignment_weight", 1.0)
            )
            # Pool reference ViT tokens to OCR visual grid and drop OCR separators.
            ocr_grid = _default_ocr_grid_thw(self.config, device=ocr_image_embeds.device)
            ref_grid = None
            if image_grid_thw is not None and image_grid_thw.shape[0] > 0:
                ref_grid = image_grid_thw[0]
            else:
                # Best-effort fallback to square guess from ref length.
                ref_grid = _infer_square_grid(
                    ref_image_embeds.shape[0],
                    spatial_merge_size=self.visual.spatial_merge_size,
                    device=ref_image_embeds.device,
                    dtype=torch.long,
                )
            ref_pooled = _pool_ref_to_ocr_grid(ref_image_embeds, ref_grid, ocr_grid)
            ocr_visual = _extract_visual_tokens(ocr_image_embeds, ocr_grid, device=ocr_image_embeds.device)
            n = min(ocr_visual.shape[0], ref_pooled.shape[0])
            align_loss = torch.mean((ocr_visual[:n] - ref_pooled[:n]) ** 2)
            if deepstack_visual_embeds is not None and ref_deepstack_embeds is not None:
                # deepstack_visual_embeds shape can vary depending on upstream; only align when levels are tensors.
                if all(isinstance(t, torch.Tensor) for t in deepstack_visual_embeds):
                    ds_losses = []
                    for ocr_level, ref_level in zip(deepstack_visual_embeds, ref_deepstack_embeds):
                        ref_level_pooled = _pool_ref_to_ocr_grid(ref_level, ref_grid, ocr_grid)
                        ocr_level_visual = _extract_visual_tokens(ocr_level, ocr_grid, device=ocr_level.device)
                        m = min(ocr_level_visual.shape[0], ref_level_pooled.shape[0])
                        ds_losses.append(torch.mean((ocr_level_visual[:m] - ref_level_pooled[:m]) ** 2))
                    if ds_losses:
                        align_loss = align_loss + torch.stack(ds_losses).mean()
            out.ocr_alignment_loss = align_loss * w  # type: ignore[attr-defined]
            out.ocr_image_embeds = ocr_image_embeds  # type: ignore[attr-defined]
            out.ref_image_embeds = ref_image_embeds  # type: ignore[attr-defined]

        return out


class OCRQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL causal LM that is friendly to OCR-style visual token inputs."""

    config_class = Qwen3VLConfig

    def __init__(self, config):
        super(Qwen3VLForConditionalGeneration, self).__init__(config)
        self.model = OCRQwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        ocr_image_features: Optional[Union[torch.Tensor, List[torch.Tensor], tuple]] = None,
        pixel_values_ref: Optional[torch.Tensor] = None,
        ocr_alignment: Optional[bool] = None,
        ocr_alignment_weight: Optional[float] = None,
        vision_scale: Optional[float] = None,
        text_scale: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            ocr_image_features=ocr_image_features,
            pixel_values_ref=pixel_values_ref,
            ocr_alignment=ocr_alignment,
            ocr_alignment_weight=ocr_alignment_weight,
            vision_scale=vision_scale,
            text_scale=text_scale,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        alignment_loss = getattr(outputs, "ocr_alignment_loss", None)
        if alignment_loss is not None:
            loss = alignment_loss if loss is None else (loss + alignment_loss)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        ocr_image_features: Optional[Union[torch.Tensor, List[torch.Tensor], tuple]] = None,
        vision_scale: Optional[float] = None,
        text_scale: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # Handle input_ids in kwargs (common pattern: model.generate(input_ids=...))
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        model_inputs = {
            "input_ids": inputs,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "ocr_image_features": ocr_image_features,
            "vision_scale": vision_scale,
            "text_scale": text_scale,
        }
        return super().generate(**model_inputs, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        ocr_image_features=None,
        deepstack_features=None,
        vision_scale=None,
        text_scale=None,
        **kwargs: Any,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            ocr_image_features=ocr_image_features,
            deepstack_features=deepstack_features,
            vision_scale=vision_scale,
            text_scale=text_scale,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position is not None and cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["ocr_image_features"] = None
            model_inputs["deepstack_features"] = None

        if vision_scale is not None:
            model_inputs["vision_scale"] = vision_scale
        if text_scale is not None:
            model_inputs["text_scale"] = text_scale

        return model_inputs


# Ensure AutoModel loads the OCR-aware variant for Qwen3-VL checkpoints.
AutoModelForCausalLM.register(Qwen3VLConfig, OCRQwen3VLForConditionalGeneration)


class Qwen3VLOCRTextAdapter:
    """Render + encode long text/images into DeepSeek-OCR visual tokens for Qwen3-VL OCR wrapper."""

    def __init__(
        self,
        *,
        encoder: Optional[object] = None,
        encoder_model_path: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        chunk_tokens: int = 1000,
        render_width: int = 640,
        render_height: int = 640,
        render_font_size: int = 18,
        render_padding: int = 30,
        use_vello_renderer: bool = True,
        render_texts=None,
        use_deepstack: bool = True,
    ) -> None:
        from PIL import Image  # local import keeps import-time optional deps minimal

        self.chunk_tokens = chunk_tokens
        self.render_width = render_width
        self.render_height = render_height
        self.render_font_size = render_font_size
        self.render_padding = render_padding
        self.device = device
        self.dtype = dtype
        self.encoder_model_path = resolve_model_path(encoder_model_path)
        self.use_deepstack = use_deepstack

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = DPSKOCREncoder(
                model_path=self.encoder_model_path,
                device=device,
                dtype=dtype,
            )

        self._Image = Image
        self._render_texts = render_texts or self._build_renderer(use_vello_renderer)

    def _build_renderer(self, use_vello_renderer: bool):
        from PIL import Image

        if use_vello_renderer and VelloRenderer is not None:
            try:
                vello = VelloRenderer(
                    width=self.render_width,
                    height=self.render_height,
                    padding=self.render_padding,
                    min_font_size=float(self.render_font_size) * 0.6,
                    max_font_size=float(self.render_font_size) * 1.5,
                )

                def _render_with_vello(chunks: Sequence[str]):
                    arrays = vello.render_batch(list(chunks))
                    return [Image.fromarray(arr) for arr in arrays]

                return _render_with_vello
            except Exception:
                pass

        if SkiaRenderer is not None:
            try:
                skia = SkiaRenderer(
                    width=self.render_width,
                    height=self.render_height,
                    padding=self.render_padding,
                    min_font_size=float(self.render_font_size) * 0.6,
                    max_font_size=float(self.render_font_size) * 1.5,
                )

                def _render_with_skia(chunks: Sequence[str]):
                    arrays = skia.render_batch(list(chunks))
                    return [Image.fromarray(arr) for arr in arrays]

                return _render_with_skia
            except Exception:
                pass

        chunk_fn, render_fn = _ensure_renderer()

        def _render_with_pil(chunks: Sequence[str]):
            return [
                render_fn(
                    chunk,
                    width=self.render_width,
                    height=self.render_height,
                    font_size=self.render_font_size,
                    padding=self.render_padding,
                )
                for chunk in chunks
            ]

        return _render_with_pil

    def text_to_ocr_features(self, text: str, tokenizer=None):
        chunk_fn, _ = _ensure_renderer()
        tokenizer_arg = tokenizer if tokenizer is not None and hasattr(tokenizer, "encode") else None
        chunks = chunk_fn(text, chunk_size=self.chunk_tokens, tokenizer=tokenizer_arg)
        images = self._render_texts(chunks)
        if self.use_deepstack and hasattr(self.encoder, "encode_images_with_deepstack"):
            final_embeds, deepstack_feats = self.encoder.encode_images_with_deepstack(images)
            return final_embeds, deepstack_feats
        return self.encoder.encode_images(images, return_global=False, return_local=True)

    def images_to_ocr_features(self, images: Iterable):
        img_list = list(images)
        if self.use_deepstack and hasattr(self.encoder, "encode_images_with_deepstack"):
            return self.encoder.encode_images_with_deepstack(img_list)
        return self.encoder.encode_images(img_list, return_global=False, return_local=True)

    @staticmethod
    def build_prompt_with_placeholders(
        instruction: str,
        num_chunks: int,
        *,
        placeholder_token: str = "<image>",
        joiner: str = "\n",
    ) -> str:
        placeholders = joiner.join([placeholder_token] * num_chunks)
        if instruction.strip() and placeholders:
            return f"{instruction.rstrip()}{joiner}{placeholders}"
        if instruction.strip():
            return instruction
        return placeholders

    def prepare_qwen_inputs(
        self,
        *,
        instruction: str,
        dense_text: str,
        tokenizer,
        placeholder_token: str = "<image>",  # Deprecated, uses Qwen3-VL format now
        return_deepstack: bool = True,
        render_instruction: bool = False,  # NEW: render instruction as image
    ):
        """Prepare inputs in Qwen3-VL native format.

        Uses <|vision_start|><|image_pad|>×N<|vision_end|> for each image.

        Args:
            render_instruction: If True, render instruction text as image and encode it.
                               This makes input fully vision tokens (instruction + content).
        """
        ocr_feats = self.text_to_ocr_features(dense_text, tokenizer=tokenizer)
        deepstack_feats = None
        if isinstance(ocr_feats, tuple):
            ocr_feats, deepstack_feats = ocr_feats

        # Build prompt with Qwen3-VL vision tokens
        vision_blocks = []
        instruction_feats = []
        instruction_deepstack = None

        # Render instruction as image if requested
        if render_instruction and instruction.strip():
            instruction_feats = self.text_to_ocr_features(instruction, tokenizer=tokenizer)
            if isinstance(instruction_feats, tuple):
                instruction_feats, instruction_deepstack = instruction_feats

            # Add instruction as first vision block
            for feat in instruction_feats:
                num_visual_tokens = feat.shape[0]
                vision_block = "<|vision_start|>" + "<|image_pad|>" * num_visual_tokens + "<|vision_end|>"
                vision_blocks.append(vision_block)

            # Prepend instruction features to content features
            ocr_feats = instruction_feats + ocr_feats
            if deepstack_feats is not None and instruction_deepstack is not None:
                deepstack_feats = instruction_deepstack + deepstack_feats

        # Add content vision blocks
        content_start_idx = len(instruction_feats) if (render_instruction and instruction.strip()) else 0
        for feat in ocr_feats[content_start_idx:]:
            num_visual_tokens = feat.shape[0]  # Should be 100 for DPSK OCR
            vision_block = "<|vision_start|>" + "<|image_pad|>" * num_visual_tokens + "<|vision_end|>"
            vision_blocks.append(vision_block)

        # Combine: either text instruction or pure vision blocks
        if render_instruction or not instruction.strip():
            prompt = "\n".join(vision_blocks)  # Pure vision input
        else:
            prompt = instruction.rstrip() + "\n" + "\n".join(vision_blocks)  # Text instruction + vision

        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        if return_deepstack and deepstack_feats is not None:
            return input_ids, (ocr_feats, deepstack_feats)
        return input_ids, ocr_feats

    def prepare_qwen_inputs_from_images(
        self,
        *,
        instruction: str,
        images: Sequence,
        tokenizer,
        placeholder_token: str = "<image>",  # Deprecated, uses Qwen3-VL format now
        return_deepstack: bool = True,
        render_instruction: bool = False,  # NEW: render instruction as image
    ):
        """Prepare inputs from images in Qwen3-VL native format.

        Uses <|vision_start|><|image_pad|>×N<|vision_end|> for each image.

        Args:
            render_instruction: If True, render instruction text as image and encode it.
                               This makes input fully vision tokens (instruction + content).
        """
        ocr_feats = self.images_to_ocr_features(images)
        deepstack_feats = None
        if isinstance(ocr_feats, tuple):
            ocr_feats, deepstack_feats = ocr_feats

        # Build prompt with Qwen3-VL vision tokens
        vision_blocks = []
        instruction_feats = []
        instruction_deepstack = None

        # Render instruction as image if requested
        if render_instruction and instruction.strip():
            instruction_feats = self.text_to_ocr_features(instruction, tokenizer=tokenizer)
            if isinstance(instruction_feats, tuple):
                instruction_feats, instruction_deepstack = instruction_feats

            # Add instruction as first vision block(s)
            for feat in instruction_feats:
                num_visual_tokens = feat.shape[0]
                vision_block = "<|vision_start|>" + "<|image_pad|>" * num_visual_tokens + "<|vision_end|>"
                vision_blocks.append(vision_block)

            # Prepend instruction features to content features
            ocr_feats = instruction_feats + ocr_feats
            if deepstack_feats is not None and instruction_deepstack is not None:
                deepstack_feats = instruction_deepstack + deepstack_feats

        # Add content vision blocks
        content_start_idx = len(instruction_feats) if (render_instruction and instruction.strip()) else 0
        for feat in ocr_feats[content_start_idx:]:
            num_visual_tokens = feat.shape[0]  # Should be 100 for DPSK OCR
            vision_block = "<|vision_start|>" + "<|image_pad|>" * num_visual_tokens + "<|vision_end|>"
            vision_blocks.append(vision_block)

        # Combine: either text instruction or pure vision blocks
        if render_instruction or not instruction.strip():
            prompt = "\n".join(vision_blocks)  # Pure vision input
        else:
            prompt = instruction.rstrip() + "\n" + "\n".join(vision_blocks)  # Text instruction + vision

        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        if return_deepstack and deepstack_feats is not None:
            return input_ids, (ocr_feats, deepstack_feats)
        return input_ids, ocr_feats
