"""OCR-aligned LLaVA Meta For CausalLM

This file adapts the official LLaVA arch (../LLaVA/llava/model/llava_arch.py)
to accept OCR-encoded image tokens in the input_ids.

Key difference vs upstream:
- Upstream expects exactly one IMAGE_TOKEN per image, then it inserts the whole
  image embedding sequence at that single position.
- DeepSeek-OCR's processors can expand a single <image> placeholder into a
  block of repeated IMAGE_TOKENs whose length equals the number of visual
  tokens for that image (global grid + optional local tiles + line breaks).

This module changes only the multimodal preparation logic so that each
contiguous run of IMAGE_TOKEN_INDEX is treated as a single image placeholder.
The entire run is replaced by one encoded image feature sequence, and labels
for these positions are set to IGNORE_INDEX as usual.

Everything else (vision tower, projector, tokenizer init) remains identical to
upstream LLaVA and is reused.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# Reuse upstream utilities and constants
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, unpad_image  # noqa: F401


def _maybe_get_image_token_id_from_config(config) -> Optional[int]:
    """Try to read an OCR image token id from config if provided by caller.

    If not present, return None and the caller should fall back to
    IMAGE_TOKEN_INDEX or heuristic detection.
    """
    # Common attribute names to probe
    for key in (
        'ocr_image_token_id',
        'image_token_id',
        'mm_image_token_id',
    ):
        if hasattr(config, key):
            val = getattr(config, key)
            if isinstance(val, int):
                return val
    return None


def _find_image_token_runs(ids_1d: torch.Tensor, image_token_id: int) -> List[Tuple[int, int]]:
    """Return list of [start, end) indices for contiguous IMAGE_TOKEN runs.

    ids_1d must be a 1-D tensor on any device/dtype (long preferred).
    """
    if ids_1d.numel() == 0:
        return []
    eq = (ids_1d == image_token_id).to(torch.bool)
    if not torch.any(eq):
        return []
    # Convert to CPU for simple scanning if needed (length is small per sequence)
    eq_cpu = eq.detach().to("cpu")
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(eq_cpu.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, int(eq_cpu.numel())))
    return runs


class OCRLlavaMetaForCausalLM(LlavaMetaForCausalLM):
    """Override only the multimodal packing to support OCR image-token blocks.

    Supports two sources of visual inputs:
    - images (raw pixels) → use upstream vision tower + mm_projector
    - ocr_image_features (pre-encoded visual tokens) → map with mm_projector

    Also allows optional scaling (`vision_scale`, `text_scale`) to bias the model
    so that vision carries the majority of information and text stays as
    instruction-only.
    """

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
        images: Optional[torch.FloatTensor],
        image_sizes: Optional[List[List[int]]] = None,
        ocr_image_features: Optional[torch.Tensor] = None,
        vision_scale: Optional[float] = None,
        text_scale: Optional[float] = None,
    ):
        vision_tower = self.get_vision_tower()
        # Text-only step or single-token decoding step. Allow pre-encoded OCR
        # features to bypass vision tower checks even if `images` is None.
        if (vision_tower is None and ocr_image_features is None) or \
           (images is None and ocr_image_features is None) or \
           (input_ids.shape[1] == 1):
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # If caller provides pre-encoded OCR features, use them; else, encode images
        def _infer_projector_in_dim(mm_projector: nn.Module) -> Optional[int]:
            # Best effort: find first Linear and use its in_features
            if isinstance(mm_projector, nn.Linear):
                return mm_projector.in_features
            for m in mm_projector.modules():
                if isinstance(m, nn.Linear):
                    return m.in_features
            return None

        if ocr_image_features is not None:
            # Normalize to list per image of [T, C]
            if isinstance(ocr_image_features, torch.Tensor):
                if ocr_image_features.ndim == 2:
                    ocr_feats_list = [ocr_image_features]
                elif ocr_image_features.ndim == 3:
                    ocr_feats_list = [ocr_image_features[i] for i in range(ocr_image_features.shape[0])]
                else:
                    raise ValueError("ocr_image_features must be [T,C] or [B,T,C]")
            elif isinstance(ocr_image_features, list):
                ocr_feats_list = ocr_image_features
            else:
                raise ValueError("Unsupported ocr_image_features type")

            # Adapt dims if needed before projector
            projector = self.get_model().mm_projector
            proj_in = _infer_projector_in_dim(projector)
            adapted_feats = []
            for feats in ocr_feats_list:
                if proj_in is not None and feats.shape[-1] != proj_in:
                    # lazily create feature adapter on the model
                    model = self.get_model()
                    if not hasattr(model, 'ocr_feature_adapter') or getattr(model, 'ocr_feature_adapter_in', None) != feats.shape[-1]:
                        adapter = nn.Linear(feats.shape[-1], proj_in, bias=False).to(device=feats.device, dtype=feats.dtype)
                        # store for reuse
                        model.ocr_feature_adapter = adapter
                        model.ocr_feature_adapter_in = feats.shape[-1]
                    feats = model.ocr_feature_adapter(feats)
                adapted_feats.append(feats)

            # Project to LLM hidden using the existing mm_projector
            image_features = [projector(f.unsqueeze(0)).squeeze(0) for f in adapted_feats]
        else:
            # Encode images using upstream pipeline (handles anyres/unpad, etc.)
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                concat_images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)

                mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
                image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

                if mm_patch_merge_type == 'flat':
                    image_features = [x.flatten(0, 1) for x in image_features]
                elif mm_patch_merge_type.startswith('spatial'):
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    self.get_vision_tower().config.image_size
                                )
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                image_features = self.encode_images(images)

        # From here on, diverge slightly from upstream: treat contiguous IMAGE_TOKEN blocks
        # as a single placeholder per image.

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove padding using attention mask
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []
        cur_image_idx = 0

        # Normalize image_features to a flat list per image
        if isinstance(image_features, torch.Tensor):
            image_features_list: List[torch.Tensor] = [image_features]
        else:
            image_features_list = list(image_features)

        # Choose which token id to look for. Prefer explicit config id;
        # otherwise support LLaVA's negative sentinel; if neither present,
        # detect long repeated runs as a heuristic fallback.
        preferred_img_token_id = _maybe_get_image_token_id_from_config(self.config)

        # resolve scales
        if vision_scale is None:
            vision_scale = float(getattr(self.config, 'ocr_vision_scale', 1.0))
        if text_scale is None:
            text_scale = float(getattr(self.config, 'ocr_text_scale', 1.0))

        for batch_idx, cur_input_ids in enumerate(input_ids):
            runs: List[Tuple[int, int]] = []
            if preferred_img_token_id is not None:
                runs = _find_image_token_runs(cur_input_ids, preferred_img_token_id)
            if not runs:
                # Fall back to sentinel used by upstream LLaVA data pipelines
                sentinel_runs = _find_image_token_runs(cur_input_ids, IMAGE_TOKEN_INDEX)
                runs = sentinel_runs
            if not runs:
                # Final fallback: heuristic detection of very long same-id blocks
                # (typical OCR visual-token blocks are 100+ repeated ids)
                unique_ids = torch.unique(cur_input_ids)
                for uid in unique_ids.tolist():
                    if uid < 0:
                        continue
                    cand_runs = _find_image_token_runs(cur_input_ids, int(uid))
                    runs.extend([r for r in cand_runs if (r[1] - r[0]) >= 16])
                # Keep runs sorted and non-overlapping
                runs.sort(key=lambda x: x[0])

            if not runs:
                # Text-only for this sample
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue

            cur_labels = labels[batch_idx]
            cur_new_input_embeds: List[torch.Tensor] = []
            cur_new_labels: List[torch.Tensor] = []

            prev_end = 0
            for (start, end) in runs:
                # Text segment before this image block
                if start > prev_end:
                    seg_ids = cur_input_ids[prev_end:start]
                    seg_labels = cur_labels[prev_end:start]
                    if seg_ids.numel() > 0:
                        seg_embeds = self.get_model().embed_tokens(seg_ids)
                        if text_scale != 1.0:
                            seg_embeds = seg_embeds * text_scale
                        cur_new_input_embeds.append(seg_embeds)
                        cur_new_labels.append(seg_labels)

                # Image features for this block
                if cur_image_idx >= len(image_features_list):
                    raise RuntimeError(
                        f"Mismatch: found {len(runs)} image-token blocks but only "
                        f"{len(image_features_list) - cur_image_idx} image features left in batch."
                    )
                cur_img_feat = image_features_list[cur_image_idx]
                cur_image_idx += 1
                if vision_scale != 1.0:
                    cur_img_feat = cur_img_feat * vision_scale
                cur_new_input_embeds.append(cur_img_feat)
                cur_new_labels.append(torch.full((cur_img_feat.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                prev_end = end

            # Trailing text after last image block
            if prev_end < cur_input_ids.shape[0]:
                seg_ids = cur_input_ids[prev_end:]
                seg_labels = cur_labels[prev_end:]
                if seg_ids.numel() > 0:
                    seg_embeds = self.get_model().embed_tokens(seg_ids)
                    if text_scale != 1.0:
                        seg_embeds = seg_embeds * text_scale
                    cur_new_input_embeds.append(seg_embeds)
                    cur_new_labels.append(seg_labels)

            # Finalize this sample
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate if needed
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad to same length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded: List[torch.Tensor] = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
