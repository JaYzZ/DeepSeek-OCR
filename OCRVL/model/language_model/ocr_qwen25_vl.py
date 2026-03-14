"""OCR-aligned Qwen2.5-VL wrappers + OCR text/image adapter.

These wrappers mirror the Qwen3-VL integration so you can feed DeepSeek-OCR
visual tokens to Qwen2.5-VL while keeping textual prompts short:

- Accepts pre-encoded OCR visual tokens via `ocr_image_features`.
- Optional scaling of vision/text embeddings (`vision_scale`, `text_scale`).
- Registers with HuggingFace AutoModel so Qwen2.5-VL checkpoints load this
  OCR-aware variant once imported.

This module also contains `Qwen25VLOCRTextAdapter`, which renders dense text into
images, encodes them with DeepSeek-OCR vision, and returns `(input_ids,
ocr_image_features)` so callers can keep prompts short while feeding content
through the OCR encoder.
"""

from __future__ import annotations

import os
import importlib.util
from typing import Any, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from transformers.utils import is_torchdynamo_compiling

from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
from OCRInfer.utils.model_paths import resolve_model_path

def _load_text_renderer():
    """Load text_renderer.py directly from the vendored DeepSeek-OCR server tree."""
    this_dir = os.path.dirname(__file__)
    root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))  # .../DeepSeek-OCR
    module_path = os.path.join(root, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm", "server", "text_renderer.py")
    if not os.path.isfile(module_path):
        return None, None
    spec = importlib.util.spec_from_file_location("deepseek_ocr_text_renderer", module_path)
    if spec is None or spec.loader is None:
        return None, None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "chunk_text_by_tokens", None), getattr(module, "render_text_to_image", None)


chunk_text_by_tokens, render_text_to_image = _load_text_renderer()

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
    feats: Union[torch.Tensor, List[torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Convert various feature container types into a list of [T, C] tensors."""
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

    return [f.to(device=device, dtype=dtype) for f in feats_list]


def _infer_square_grid(num_tokens: int, spatial_merge_size: int, *, device, dtype) -> torch.Tensor:
    """Best-effort guess for (t,h,w) grid when caller skips image_grid_thw."""
    patches = num_tokens * (spatial_merge_size ** 2)
    side = int(round(patches ** 0.5))
    if side * side != patches:
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
    ref_visual = ref_visual.view(ref_h, ref_w, -1).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(ref_visual, (ocr_h, ocr_w))
    pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(ocr_h * ocr_w, -1)
    return pooled


class OCRQwen25VLModel(Qwen2_5_VLModel):
    """Drop-in Qwen2.5-VL model that accepts OCR visual features and scaling."""

    config_class = Qwen2_5_VLConfig

    def _init_ocr_connector(self, in_dim: int, proj_in_dim: int, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Lazily build a lightweight connector from OCR dim -> vision projector input dim."""
        if in_dim == proj_in_dim:
            connector: nn.Module = nn.Identity()
        else:
            connector = nn.Linear(in_dim, proj_in_dim, bias=False)
        return connector.to(device=device, dtype=dtype)

    def _maybe_get_ocr_connector(
        self, in_dim: int, proj_in_dim: int, *, device: torch.device, dtype: torch.dtype
    ) -> nn.Module:
        if not hasattr(self, "ocr_connector") or self.ocr_connector is None:  # type: ignore[attr-defined]
            self.ocr_connector = self._init_ocr_connector(in_dim, proj_in_dim, device=device, dtype=dtype)  # type: ignore[attr-defined]
        else:
            conn = self.ocr_connector  # type: ignore[attr-defined]
            if isinstance(conn, nn.Linear) and conn.in_features != in_dim:
                self.ocr_connector = self._init_ocr_connector(in_dim, proj_in_dim, device=device, dtype=dtype)  # type: ignore[attr-defined]
        return self.ocr_connector  # type: ignore[attr-defined]

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
        ocr_image_features: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        ocr_pixel_values: Optional[torch.Tensor] = None,
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
        if ocr_image_features is None and ocr_pixel_values is None:
            raise ValueError(
                "ocr_image_features (or ocr_pixel_values) is required. Provide OCR features (and <image> placeholders in input_ids). "
                "For training-time alignment, also provide pixel_values/pixel_values_ref with ocr_alignment=True."
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        ocr_alignment = bool(ocr_alignment) if ocr_alignment is not None else False

        if pixel_values is not None and not ocr_alignment:
            raise ValueError(
                "pixel_values is only accepted for alignment (set ocr_alignment=True). "
                "For the LM path, provide ocr_image_features."
            )

        image_mask = None
        ocr_image_embeds = None
        ref_image_embeds = None
        if ocr_image_features is None and ocr_pixel_values is not None:
            from OCRVL.dpsk_encoder import get_dpsk_encoder

            encoder = get_dpsk_encoder()
            with torch.no_grad():
                feats = encoder.encode_pixel_values(ocr_pixel_values)
            ocr_image_features = feats

        if ocr_image_features is not None:
            feats = _normalize_ocr_features(
                ocr_image_features, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            num_images = len(feats)
            if image_grid_thw is None:
                base_grid = _default_ocr_grid_thw(self.config, device=inputs_embeds.device)
                image_grid_thw = base_grid.unsqueeze(0).expand(num_images, -1).contiguous()
            else:
                image_grid_thw = image_grid_thw.to(inputs_embeds.device)

            image_mask = torch.zeros(
                (inputs_embeds.shape[0], num_images),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

            proj = self.visual_projector
            if isinstance(proj, nn.ModuleList):
                proj = proj[0]
            proj = proj.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            proj_in_dim = getattr(proj, "in_features", feats[0].shape[-1])
            conn = self._maybe_get_ocr_connector(
                feats[0].shape[-1], proj_in_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

            emb_lists = []
            for idx, (feat, grid) in enumerate(zip(feats, image_grid_thw)):
                feat = conn(feat)
                emb = proj(feat.unsqueeze(0))
                emb_lists.append(emb[0])
                image_mask[:, idx] = 1
                if grid.shape[0] == 1:
                    image_grid_thw[idx] = grid
                else:
                    image_grid_thw[idx] = grid[0]

            image_features = torch.cat(emb_lists, dim=0)
            ocr_image_embeds = image_features
            image_features = image_features * vision_scale
        else:
            image_features = None

        text_features = inputs_embeds * text_scale

        # Reference ViT path for alignment (does not affect inputs_embeds).
        if ocr_alignment and ocr_image_embeds is not None:
            ref_pixels = pixel_values_ref if pixel_values_ref is not None else pixel_values
            if ref_pixels is not None:
                ref_list = self.get_image_features(ref_pixels, image_grid_thw)
                # get_image_features returns list per image for Qwen2.5.
                ref_image_embeds = torch.cat(ref_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        # When OCR features are provided, don't also inject raw pixels into the LM path.
        pixel_values_forward = None if ocr_image_features is not None else pixel_values

        out = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=text_features,
            pixel_values=pixel_values_forward,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            image_features=image_features,
            image_mask=image_mask,
            **kwargs,
        )

        if ocr_alignment and ocr_image_embeds is not None and ref_image_embeds is not None:
            w = float(ocr_alignment_weight) if ocr_alignment_weight is not None else float(
                getattr(self.config, "ocr_alignment_weight", 1.0)
            )
            ocr_grid = _default_ocr_grid_thw(self.config, device=ocr_image_embeds.device)
            if image_grid_thw is not None and image_grid_thw.shape[0] > 0:
                ref_grid = image_grid_thw[0]
            else:
                ref_grid = _infer_square_grid(
                    ref_image_embeds.shape[0],
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                    device=ref_image_embeds.device,
                    dtype=torch.long,
                )
            ref_pooled = _pool_ref_to_ocr_grid(ref_image_embeds, ref_grid, ocr_grid)
            ocr_visual = _extract_visual_tokens(ocr_image_embeds, ocr_grid, device=ocr_image_embeds.device)
            n = min(ocr_visual.shape[0], ref_pooled.shape[0])
            align_loss = torch.mean((ocr_visual[:n] - ref_pooled[:n]) ** 2) * w
            try:
                out.ocr_alignment_loss = align_loss  # type: ignore[attr-defined]
                out.ocr_image_embeds = ocr_image_embeds  # type: ignore[attr-defined]
                out.ref_image_embeds = ref_image_embeds  # type: ignore[attr-defined]
            except Exception:
                # If upstream output type is strict, silently skip attaching.
                pass

        return out


class OCRQwen25VLCausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """Match parent type so downstream callers see the OCR subclass."""

    pass


class OCRQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLConfig
    _supports_cache_class = True

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.model = OCRQwen25VLModel(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ocr_image_features: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        pixel_values_ref: Optional[torch.Tensor] = None,
        ocr_alignment: Optional[bool] = None,
        ocr_alignment_weight: Optional[float] = None,
        vision_scale: Optional[float] = None,
        text_scale: Optional[float] = None,
        **kwargs,
    ) -> Union[Qwen2_5_VLCausalLMOutputWithPast, tuple]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ocr_image_features=ocr_image_features,
            pixel_values_ref=pixel_values_ref,
            ocr_alignment=ocr_alignment,
            ocr_alignment_weight=ocr_alignment_weight,
            vision_scale=vision_scale,
            text_scale=text_scale,
            **kwargs,
        )

        hidden_states = outputs[0]
        alignment_loss = getattr(outputs, "ocr_alignment_loss", None)

        if labels is not None:
            labels = labels.to(hidden_states.device)
            logits = self.lm_head(hidden_states).float()
            loss = None
            if not is_torchdynamo_compiling():
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = self.loss_function(hidden_states, labels)
            if alignment_loss is not None:
                loss = loss + alignment_loss
            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output

            return OCRQwen25VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return OCRQwen25VLCausalLMOutputWithPast(
            loss=alignment_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        ocr_image_features: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        vision_scale: Optional[float] = None,
        text_scale: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        model_inputs = {
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
            vision_scale=vision_scale,
            text_scale=text_scale,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position is not None and cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["ocr_image_features"] = None

        if vision_scale is not None:
            model_inputs["vision_scale"] = vision_scale
        if text_scale is not None:
            model_inputs["text_scale"] = text_scale

        return model_inputs


# Ensure AutoModel loads the OCR-aware variant for Qwen2.5-VL checkpoints.
AutoModelForCausalLM.register(Qwen2_5_VLConfig, OCRQwen25VLForConditionalGeneration)


class Qwen25VLOCRTextAdapter:
    """Render + encode long text/images into DeepSeek-OCR visual tokens for Qwen2.5-VL OCR wrapper."""

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
        self.use_deepstack = False  # Qwen2.5 uses final-layer features only

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
        return self.encoder.encode_images(images, return_global=False, return_local=True)

    def images_to_ocr_features(self, images: Iterable):
        img_list = list(images)
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
        placeholder_token: str = "<image>",
    ):
        ocr_feats = self.text_to_ocr_features(dense_text, tokenizer=tokenizer)
        prompt = self.build_prompt_with_placeholders(
            instruction, len(ocr_feats), placeholder_token=placeholder_token
        )
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        return input_ids, ocr_feats

    def prepare_qwen_inputs_from_images(
        self,
        *,
        instruction: str,
        images: Sequence,
        tokenizer,
        placeholder_token: str = "<image>",
    ):
        ocr_feats = self.images_to_ocr_features(images)
        prompt = self.build_prompt_with_placeholders(
            instruction, len(ocr_feats), placeholder_token=placeholder_token
        )
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        return input_ids, ocr_feats
