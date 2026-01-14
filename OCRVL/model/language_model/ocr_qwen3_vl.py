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

import logging
import os
import shutil
from typing import Any, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

logger = logging.getLogger(__name__)


class OCRQwen3VLConfig(Qwen3VLConfig):
    """OCR-aware Qwen3-VL config that auto-loads OCRQwen3VLForConditionalGeneration.

    This config extends Qwen3VLConfig with DPSK OCR encoder path configuration.
    When loaded via AutoConfig, it will automatically instantiate OCRQwen3VLForConditionalGeneration.
    """

    model_type = "ocr_qwen3_vl"

    def __init__(self, *args, dpsk_ocr_model_name_or_path: Optional[str] = None, **kwargs):
        """Initialize config with optional DPSK OCR encoder path.

        Args:
            dpsk_ocr_model_name_or_path: Path to DeepSeek-OCR model for DPSK encoder.
                                        Falls back to DPSK_MODEL_PATH env var or "deepseek-ai/DeepSeek-OCR".
        """
        super().__init__(*args, **kwargs)
        self.dpsk_ocr_model_name_or_path = dpsk_ocr_model_name_or_path
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


class DPSKVisionTowerAdapter(nn.Module):
    """DPSK OCR encoder adapter that produces features compatible with Qwen3-VL's vision tower.

    This wrapper makes the DPSK encoder compatible with Qwen3-VL's vision tower interface.
    With FSDP and freeze_vision_tower: true, LlamaFactory handles freezing automatically.

    The DPSK encoder has a different interface than Qwen3-VL's vision tower:
    - DPSK: encode_pixel_values_with_deepstack() -> (final_embeddings, intermediate_embeddings)
    - Qwen3-VL: forward(pixel_values, grid_thw) -> (last_hidden_state, pooler_output, deepstack_outputs)

    NOTE: Unlike DDP, with FSDP we can make this a regular module since:
    - freeze_vision_tower: true will handle freezing via requires_grad=False
    - FSDP will properly manage device placement for all child modules
    """

    def __init__(self, dpsk_encoder: DPSKOCREncoder):
        super().__init__()

        # Store as regular child module so FSDP can manage device placement
        self._dpsk_encoder = dpsk_encoder

        # Copy attributes that Qwen3-VL vision tower has
        # Use object.__setattr__ to avoid triggering any custom __setattr__ or __getattr__
        object.__setattr__(self, 'config', getattr(dpsk_encoder, 'config', None))
        object.__setattr__(self, 'blocks', getattr(dpsk_encoder, 'blocks', []))
        object.__setattr__(self, 'merger', getattr(dpsk_encoder, 'merger', None))
        object.__setattr__(self, 'deepstack_merger_list', getattr(dpsk_encoder, 'deepstack_merger_list', []))
        object.__setattr__(self, 'deepstack_visual_indexes', getattr(dpsk_encoder, 'deepstack_visual_indexes', []))

    @property
    def dpsk_encoder(self) -> DPSKOCREncoder:
        """Property to access encoder."""
        return self._dpsk_encoder

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor = None):
        """Forward pass compatible with Qwen3-VL's vision tower interface.

        NOTE: With FSDP + freeze_vision_tower: true, we don't need @torch.no_grad() or .detach()
        since LlamaFactory handles freezing by setting requires_grad=False on vision tower.

        Args:
            pixel_values: Input images [B, C, H, W]
            grid_thw: Grid dimensions (ignored by DPSK but kept for interface compatibility)

        Returns:
            Tuple matching Qwen3-VL vision tower output:
            (last_hidden_state, pooler_output, deepstack_outputs)
        """
        # AGGRESSIVE dtype enforcement: Fix any float32 parameters on EVERY forward pass
        # This is necessary because FSDP + eval can reset dtypes on different ranks
        # The check is fast (only checks dtype, doesn't convert if already correct)
        import torch.nn as nn
        fixed = 0

        # Convert the DPSK encoder itself
        for module in self.dpsk_encoder.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
                if hasattr(module, 'bias') and module.bias is not None:
                    if module.bias.dtype != torch.bfloat16:
                        module.bias.data = module.bias.data.to(torch.bfloat16)
                        fixed += 1
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype != torch.bfloat16:
                        module.weight.data = module.weight.data.to(torch.bfloat16)
                        fixed += 1

        # Also directly convert the nested SAM and CLIP models (FSDP might hide them)
        for model in [self.dpsk_encoder.sam_model, self.dpsk_encoder.clip_model]:
            if model is not None:
                for module in model.modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
                        if hasattr(module, 'bias') and module.bias is not None:
                            if module.bias.dtype != torch.bfloat16:
                                module.bias.data = module.bias.data.to(torch.bfloat16)
                                fixed += 1
                        if hasattr(module, 'weight') and module.weight is not None:
                            if module.weight.dtype != torch.bfloat16:
                                module.weight.data = module.weight.data.to(torch.bfloat16)
                                fixed += 1

        if fixed > 0:
            logger.info(f"[OCRVL] Fixed {fixed} vision encoder params to bfloat16 (forward pass)")

        # DPSK encoder returns: (final_embeddings, intermediate_embeddings)
        final_embeddings, intermediate_embeddings = self.dpsk_encoder.encode_pixel_values_with_deepstack(
            pixel_values
        )

        # Convert to format expected by Qwen3-VL
        if isinstance(final_embeddings, (list, tuple)):
            last_hidden_state = torch.cat(final_embeddings, dim=0)
        else:
            last_hidden_state = final_embeddings

        pooler_output = None

        # deepstack_outputs
        if intermediate_embeddings is not None and len(intermediate_embeddings) > 0:
            num_levels = len(intermediate_embeddings[0])
            deepstack_outputs = []
            for level_idx in range(num_levels):
                level_features = [per_img[level_idx] for per_img in intermediate_embeddings]
                level_cat = torch.cat(level_features, dim=0)
                deepstack_outputs.append(level_cat)
        else:
            deepstack_outputs = []

        return last_hidden_state, pooler_output, deepstack_outputs

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

    config_class = OCRQwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)

        # Defer connector initialization to avoid meta tensor issues during from_pretrained.
        # Connectors will be initialized lazily on first use or explicitly after model loading.
        self.ocr_connector: Optional[nn.Module] = None
        self.thinking_projection: Optional[nn.Module] = None

        # Deepstack connector for 1024-dim intermediate features (individual module, not ModuleDict)
        # Stored as individual module for PEFT modules_to_save compatibility
        self.ocr_deepstack_connector: Optional[nn.Module] = None

        # Note: DPSK encoder is now stored in self.visual (wrapped by DPSKVisionTowerAdapter)
        # instead of as a separate self.dpsk_encoder attribute. This allows LlamaFactory's
        # freeze_vision_tower mechanism to work naturally without DDP issues.

    def _initialize_ocr_connectors_early(self, include_thinking: bool = False) -> None:
        """Force early initialization of OCR connectors for training setup.

        This ensures that connector modules exist when LlamaFactory tries to mark
        them as trainable via `additional_target`. Without this, the lazy initialization
        means connectors are None during training setup, so they remain frozen.

        Args:
            include_thinking: If True, also initialize thinking_projection. This should
                only be set when training with latent supervision, as thinking_projection
                doesn't get gradients during normal training and causes DDP errors.

        Note: Deepstack connectors are stored as individual modules (not ModuleDict)
        because PEFT's modules_to_save doesn't support ModuleDict.
        """
        # Get current device and dtype from the model
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Initialize ocr_connector if not already initialized
        # This is the main connector used for OCR features (1280-dim)
        if self.ocr_connector is None:
            # OCR features are 1280-dim
            self.ocr_connector = self._init_ocr_connector(1280, device=device, dtype=dtype)

        # Initialize deepstack connector if not already initialized
        # Stored as individual module for PEFT compatibility (not ModuleDict).
        # Only initialize if deepstack is enabled; otherwise keep it absent to avoid DDP "unused params" hazards.
        if bool(getattr(self.config, "ocr_require_deepstack", True)) and self.ocr_deepstack_connector is None:
            self.ocr_deepstack_connector = self._init_ocr_connector(1024, device=device, dtype=dtype)

        # NOTE: thinking_projection is only initialized if include_thinking=True
        # This projection is ONLY used when training with latent supervision.
        # When not used, it has no gradient flow and causes DDP errors.
        # It uses lazy initialization (_maybe_get_thinking_projection) when needed.
        if include_thinking and self.thinking_projection is None:
            self.thinking_projection = self._init_thinking_projection(device=device, dtype=dtype)

    def _offload_visual_to_cpu(self) -> None:
        # Handle DDP wrapping: if self is DDP-wrapped, access self.module.visual
        actual_model = self._get_actual_model()
        visual = getattr(actual_model, "visual", None)
        if isinstance(visual, nn.Module):
            visual.to(device=torch.device("cpu"))
            visual.requires_grad_(False)

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
        # Only move to device/dtype if not a meta device (avoid meta tensor issues during init)
        if device.type != 'meta':
            connector = connector.to(device=device, dtype=dtype)
        return connector

    def _maybe_get_ocr_connector(self, in_dim: int, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        if not hasattr(self, "ocr_connector") or self.ocr_connector is None:  # type: ignore[attr-defined]
            self.ocr_connector = self._init_ocr_connector(in_dim, device=device, dtype=dtype)  # type: ignore[attr-defined]
        else:
            # If dims changed (unlikely), rebuild.
            conn = self.ocr_connector  # type: ignore[attr-defined]
            if isinstance(conn, nn.Linear) and conn.in_features != in_dim:
                self.ocr_connector = self._init_ocr_connector(in_dim, device=device, dtype=dtype)  # type: ignore[attr-defined]
        self.ocr_connector = self.ocr_connector.to(device=device, dtype=dtype)  # type: ignore[attr-defined]

        # When using additional_target with PEFT, the connector is wrapped in ModulesToSaveWrapper
        # The wrapper's forward() method handles the call correctly, so we return the wrapper itself
        # Do NOT access modules_to_save['default'] directly - this bypasses the wrapper
        return self.ocr_connector  # type: ignore[attr-defined]

    def _init_thinking_projection(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Build thinking projection MLP: hidden_dim → latent_dim (4096 → 1280)

        This projection maps transformer hidden states to OCR latent space for
        thinking-with-latent-tokens training. Architecture follows connector pattern.

        Parameters: ~10.5M (Linear 4096×2048=8.4M + Linear 2048×1280=2.6M + biases)
        """
        hidden_dim = self.config.text_config.hidden_size  # 4096 for Qwen3-VL
        latent_dim = 1280  # DPSK OCR latent dimension

        projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 4096 → 2048
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),  # 2048 → 1280
        )

        # Only move to device/dtype if not a meta device
        if device.type != 'meta':
            projection = projection.to(device=device, dtype=dtype)
        return projection

    def _maybe_get_thinking_projection(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Lazy initialization of thinking projection"""
        if not hasattr(self, "thinking_projection") or self.thinking_projection is None:  # type: ignore[attr-defined]
            self.thinking_projection = self._init_thinking_projection(device=device, dtype=dtype)  # type: ignore[attr-defined]
        self.thinking_projection = self.thinking_projection.to(device=device, dtype=dtype)  # type: ignore[attr-defined]
        return self.thinking_projection  # type: ignore[attr-defined]

    def _maybe_get_deepstack_connector(self, in_dim: int, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Deepstack levels connector - also uses LLaVA-1.5/1.6 mlp2x_gelu.

        Note: This is simplified to handle only 1024-dim deepstack features.
        The connector is stored as an individual module (ocr_deepstack_connector)
        for PEFT additional_target compatibility.
        """
        if in_dim != 1024:
            raise ValueError(f"Deepstack connector currently only supports 1024-dim, got {in_dim}")

        if self.ocr_deepstack_connector is None:  # type: ignore[attr-defined]
            self.ocr_deepstack_connector = self._init_ocr_connector(1024, device=device, dtype=dtype)  # type: ignore[attr-defined]
        else:
            # Ensure device/dtype are correct
            self.ocr_deepstack_connector = self.ocr_deepstack_connector.to(device=device, dtype=dtype)  # type: ignore[attr-defined]

        # When using additional_target with PEFT, return the wrapper itself
        # The wrapper's forward() method handles the call correctly
        return self.ocr_deepstack_connector  # type: ignore[attr-defined]

    def _get_actual_model(self) -> "OCRQwen3VLModel":
        """Return the actual model, unwrapping DDP if present.

        When using DistributedDataParallel (DDP), self is the DDP wrapper
        and the actual model is at self.module. This helper provides safe access
        to the underlying model regardless of whether DDP is enabled.

        Returns:
            The actual OCRQwen3VLModel instance (unwrapped from DDP if present).
        """
        if isinstance(self, DistributedDataParallel):
            return self.module
        return self

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
        # Native pixel_values -> ViT -> LM path is only used for dual-path alignment with ocr_alignment=True.

        # During generation, cache_position indicates if this is first pass (=0) or subsequent (>0)
        # Only require vision features on first pass
        is_first_pass = (cache_position is None or
                        (hasattr(cache_position, '__len__') and len(cache_position) > 0 and cache_position[0] == 0) or
                        (not hasattr(cache_position, '__len__') and cache_position == 0))

        if ocr_image_features is None and pixel_values is None and is_first_pass:
            raise ValueError(
                "Vision input is required. Provide one of: "
                "- pixel_values (will be processed by DPSK encoder via vision tower), "
                "- ocr_image_features (pre-encoded features, useful for cached inference). "
                "For training-time alignment, also provide pixel_values_ref with ocr_alignment=True."
            )

        # On subsequent passes, ocr_image_features can be None (vision already in KV cache)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        ocr_alignment = bool(ocr_alignment) if ocr_alignment is not None else False

        ocr_image_embeds = None
        ref_image_embeds = None
        ref_deepstack_embeds = None

        # Handle pixel_values from LlamaFactory (now that DPSKOCRImageProcessor returns pixel_values)
        # Route through the vision tower adapter which wraps the DPSK encoder
        if ocr_image_features is None and pixel_values is not None and is_first_pass and not ocr_alignment:
            actual_model = self._get_actual_model()
            visual = getattr(actual_model, "visual", None)

            # Handle FSDP-wrapped modules (CheckpointWrapper) by checking for dpsk_encoder attribute
            # instead of relying on isinstance() which fails with FSDP wrapping
            is_dpsk_adapter = isinstance(visual, DPSKVisionTowerAdapter) or (
                visual is not None and hasattr(visual, 'dpsk_encoder')
            )

            if visual is not None and not is_dpsk_adapter:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"[OCRVL] visual type check failed: type(visual)={type(visual)}, hasattr dpsk_encoder={hasattr(visual, 'dpsk_encoder') if visual is not None else 'N/A'}")
                raise ValueError(f"self.visual is not a DPSKVisionTowerAdapter (got {type(visual).__name__}). DPSK encoder initialization failed.")

            if visual is None:
                raise ValueError("actual_model.visual not found. DPSK encoder should have been initialized during model loading.")

            # Call the DPSKVisionTowerAdapter with pixel_values
            # The adapter will route through the DPSK encoder and return (last_hidden_state, pooler_output, deepstack_outputs)
            if is_dpsk_adapter:
                with torch.no_grad():
                    last_hidden_state, pooler_output, deepstack_outputs = visual(pixel_values, grid_thw=image_grid_thw)

                # Convert to ocr_image_features format: (final_embeddings, intermediate_embeddings)
                # final_embeddings: list of per-image [T, C] tensors
                # intermediate_embeddings: list of per-image lists of [T_level, C] tensors
                batch_size = int(pixel_values.shape[0])
                final_embeddings = []
                intermediate_embeddings = []

                total_tokens = int(last_hidden_state.shape[0])
                if batch_size <= 0 or total_tokens % batch_size != 0:
                    raise ValueError(
                        f"Unexpected vision output shape: total_tokens={total_tokens}, batch_size={batch_size}."
                    )
                tokens_per_image = total_tokens // batch_size

                for i in range(batch_size):
                    start_idx = i * tokens_per_image
                    end_idx = start_idx + tokens_per_image
                    img_feats = last_hidden_state[start_idx:end_idx].detach()  # [T, C]
                    final_embeddings.append(img_feats)

                    if deepstack_outputs:
                        img_intermediate = []
                        for level_output in deepstack_outputs:
                            img_intermediate.append(level_output[start_idx:end_idx].detach())
                        intermediate_embeddings.append(img_intermediate)
                    else:
                        intermediate_embeddings.append([])

                ocr_image_features = (final_embeddings, intermediate_embeddings)

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
            # Alignment needs the original Qwen3-VL vision tower; keep it offloaded unless needed.
            # Handle DDP wrapping: if self is DDP-wrapped, access self.module.visual
            actual_model = self._get_actual_model()
            visual = getattr(actual_model, "visual", None)
            if isinstance(visual, nn.Module):
                maybe_param = next(visual.parameters(), None)
                if maybe_param is not None and maybe_param.device != inputs_embeds.device:
                    visual.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
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
            output_hidden_states=True,  # Enable hidden states for thinking projection
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

    config_class = OCRQwen3VLConfig

    # Override _no_split_modules to only include Qwen3VLTextDecoderLayer
    # We removed Qwen3VLVisionBlock since we replaced the vision tower with DPSKVisionTowerAdapter
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OCRQwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

        self._dpsk_source_path: Optional[str] = None

    @staticmethod
    def _is_main_process() -> bool:
        return int(os.environ.get("RANK", "0")) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0

    def _get_actual_model(self) -> "OCRQwen3VLForConditionalGeneration":
        """Return the actual model, unwrapping DDP if present.

        When using DistributedDataParallel (DDP), self is the DDP wrapper
        and the actual model is at self.module. This helper provides safe access
        to the underlying model regardless of whether DDP is enabled.

        Returns:
            The actual OCRQwen3VLForConditionalGeneration instance (unwrapped from DDP if present).
        """
        if isinstance(self, DistributedDataParallel):
            return self.module
        return self

    @staticmethod
    def _resolve_dpsk_path(config: Any, pretrained_model_name_or_path: str) -> str:
        # 1) Prefer bundled folder inside a checkpoint dir.
        if os.path.isdir(pretrained_model_name_or_path):
            bundled = os.path.join(pretrained_model_name_or_path, "dpsk_ocr_encoder")
            if os.path.isdir(bundled):
                return bundled

        # 2) Config override (supports relative path when loading from a local dir).
        cfg_path = getattr(config, "dpsk_ocr_model_name_or_path", None)
        if isinstance(cfg_path, str) and cfg_path:
            if os.path.isdir(pretrained_model_name_or_path) and not os.path.isabs(cfg_path):
                candidate = os.path.join(pretrained_model_name_or_path, cfg_path)
                if os.path.isdir(candidate):
                    return candidate
            return cfg_path

        # 3) Environment default.
        return os.environ.get("DPSK_MODEL_PATH", "deepseek-ai/DeepSeek-OCR")

    @staticmethod
    def _hardlink_or_copytree(src: str, dst: str) -> None:
        os.makedirs(dst, exist_ok=True)
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            out_root = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(out_root, exist_ok=True)
            for d in dirs:
                os.makedirs(os.path.join(out_root, d), exist_ok=True)
            for f in files:
                s = os.path.join(root, f)
                t = os.path.join(out_root, f)
                if os.path.exists(t):
                    continue
                try:
                    os.link(s, t)
                except Exception:
                    shutil.copy2(s, t)

    def _init_dpsk_encoder_from_config(self, pretrained_model_name_or_path: str) -> None:
        """Replace model.visual with DPSK OCR encoder (wrapped as vision tower).

        This replaces Qwen3-VL's vision tower with the DPSK encoder so that:
        1. LlamaFactory's freeze_vision_tower mechanism works naturally
        2. No DDP issues with frozen parameters
        3. The original ViT is kept as backup for alignment training
        """
        cfg = self.config
        dpsk_path = self._resolve_dpsk_path(cfg, pretrained_model_name_or_path)
        dpsk_path = resolve_model_path(dpsk_path)

        self._dpsk_source_path = dpsk_path
        cfg.dpsk_ocr_model_name_or_path = dpsk_path

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = os.environ.get("DPSK_DEVICE", f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        dtype_str = os.environ.get("DPSK_DTYPE", "bf16").strip().lower()
        if dtype_str in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif dtype_str in {"fp16", "float16"}:
            dtype = torch.float16
        else:
            dtype = torch.float32

        require_deepstack = bool(getattr(cfg, "ocr_require_deepstack", True))
        # Use the same intermediate layer indices as Qwen3-VL's deepstack_visual_indexes: [5, 11, 17]
        # These correspond to early, middle, and late layers in the SAM encoder
        intermediate_layer_indices = [5, 11, 17] if require_deepstack else None
        remove_separators = os.environ.get("OCRVL_DPSK_REMOVE_SEPARATORS", "1").strip() != "0"

        # Create DPSK encoder
        dpsk_encoder = DPSKOCREncoder(
            model_path=dpsk_path,
            device=device,
            dtype=dtype,
            intermediate_layer_indices=intermediate_layer_indices,
            remove_separators=remove_separators,
        )

        # Keep original Qwen3-VL ViT as backup (for alignment training)
        # Store it in a non-module attribute to prevent DDP from tracking its parameters
        original_visual_state_dict = None
        # Handle DDP wrapping: if self is DDP-wrapped, access self.module.model.visual
        actual_model = self._get_actual_model()
        if hasattr(actual_model.model, 'visual') and isinstance(actual_model.model.visual, nn.Module):
            # Save state_dict (not the module itself) to avoid DDP tracking
            original_visual_state_dict = {
                'state_dict': actual_model.model.visual.state_dict(),
                'config': actual_model.model.visual.config if hasattr(actual_model.model.visual, 'config') else None,
            }
            logger.info("[OCRVL] Saved original Qwen3-VL vision tower state_dict for potential future use")
        # Store in a private non-module attribute (won't be tracked by DDP)
        actual_model._visual_qwen3vl_backup_state = original_visual_state_dict

        # REPLACE vision tower with DPSK encoder (wrapped as frozen vision tower)
        # The adapter handles freezing (eval mode + requires_grad=False + @torch.no_grad())
        # This ensures encoder outputs are detached from the autograd graph
        actual_model.model.visual = DPSKVisionTowerAdapter(dpsk_encoder)
        logger.info("[OCRVL] ✓ Replaced vision tower with frozen DPSK encoder (DDP-safe)")

        # CRITICAL: Sync dtype with rest of model to prevent FSDP dtype mismatch
        # The language model was loaded from checkpoint with consistent dtype,
        # but DPSK encoder was created separately. Force all DPSK params to match.
        target_dtype = next(actual_model.parameters()).dtype
        logger.info(f"[OCRVL] Syncing DPSK encoder dtype to match language model: {target_dtype}")
        converted = 0
        for param in dpsk_encoder.parameters():
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
                converted += 1
        if converted > 0:
            logger.warning(f"[OCRVL] Converted {converted} DPSK params to {target_dtype}")

    def _init_dpsk_encoder_from_saved_state(self, checkpoint_path: str) -> None:
        """Load DPSK encoder from saved checkpoint state (self-contained model).

        This is used when resuming training from a checkpoint that was saved with
        save_pretrained(). The encoder weights are part of the checkpoint, not
        loaded from external DeepSeek-OCR files.

        This ensures FSDP treats the entire model uniformly since all components
        were loaded from the same checkpoint.
        """
        import torch.nn as nn

        cfg = self.config
        state_path = os.path.join(checkpoint_path, cfg.dpsk_encoder_state_path)

        logger.info(f"[OCRVL] Loading DPSK encoder from saved state: {state_path}")

        # Load the saved state dict
        dpsk_state_dict = torch.load(state_path, map_location='cpu')

        # Get target dtype from the language model (already loaded)
        actual_model = self._get_actual_model()
        target_dtype = next(actual_model.parameters()).dtype
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = os.environ.get("DPSK_DEVICE", f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        require_deepstack = bool(getattr(cfg, "ocr_require_deepstack", True))
        intermediate_layer_indices = [5, 11, 17] if require_deepstack else None
        remove_separators = os.environ.get("OCRVL_DPSK_REMOVE_SEPARATORS", "1").strip() != "0"

        # Create a new DPSK encoder instance
        dpsk_encoder = DPSKOCREncoder(
            model_path="dummy",  # Path doesn't matter since we'll load state dict
            device='cpu',  # Load on CPU first, then move to device
            dtype=target_dtype,  # Use same dtype as language model
            intermediate_layer_indices=intermediate_layer_indices,
            remove_separators=remove_separators,
        )

        # Load the saved state dict
        dpsk_encoder.load_state_dict(dpsk_state_dict, strict=True)
        dpsk_encoder.to(device)

        logger.info(f"[OCRVL] ✓ Loaded DPSK encoder state_dict ({sum(p.numel() for p in dpsk_encoder.parameters()):,} params)")

        # Keep original Qwen3-VL ViT as backup (for alignment training)
        original_visual_state_dict = None
        if hasattr(actual_model.model, 'visual') and isinstance(actual_model.model.visual, nn.Module):
            original_visual_state_dict = {
                'state_dict': actual_model.model.visual.state_dict(),
                'config': actual_model.model.visual.config if hasattr(actual_model.model.visual, 'config') else None,
            }
            logger.info("[OCRVL] Saved original Qwen3-VL vision tower state_dict for potential future use")
        actual_model._visual_qwen3vl_backup_state = original_visual_state_dict

        # REPLACE vision tower with loaded DPSK encoder
        actual_model.model.visual = DPSKVisionTowerAdapter(dpsk_encoder)
        logger.info("[OCRVL] ✓ Replaced vision tower with loaded DPSK encoder (from checkpoint)")

        # Verify dtype matches
        encoder_dtype = next(dpsk_encoder.parameters()).dtype
        if encoder_dtype != target_dtype:
            logger.warning(f"[OCRVL] DPSK encoder dtype {encoder_dtype} != target {target_dtype}, converting...")
            for param in dpsk_encoder.parameters():
                param.data = param.data.to(target_dtype)

    def _freeze_dpsk_encoder(self) -> None:
        """Freeze all DPSK OCR encoder parameters to prevent vision encoder modification during training.

        NOTE: This method is kept for backward compatibility but is NO LONGER USED by default.
        LlamaFactory's freeze_vision_tower mechanism now handles freezing the DPSK encoder
        (which is stored in model.visual, wrapped by DPSKVisionTowerAdapter).

        The DeepSeek-OCR encoder should remain frozen to preserve its pre-trained OCR capabilities.
        Only the OCR connectors and LoRA adapters should be trained.

        Override by setting environment variable: OCRVL_UNFREEZE_DPSK_ENCODER=1
        """
        # Allow unfreezing via environment variable for debugging/special cases
        unfreeze_env = os.environ.get("OCRVL_UNFREEZE_DPSK_ENCODER", "0").strip() in {"1", "true", "yes"}
        if unfreeze_env:
            if self._is_main_process():
                import warnings
                warnings.warn(
                    "OCRVL_UNFREEZE_DPSK_ENCODER is set - DPSK encoder will be trainable. "
                    "This is NOT recommended for normal training as it will destroy the pre-trained OCR capabilities."
                )
            return

        # DPSK encoder is now in model.visual (wrapped by DPSKVisionTowerAdapter)
        # Handle DDP wrapping: if self is DDP-wrapped, access self.module.model.visual
        actual_model = self._get_actual_model()
        visual = actual_model.model.visual if hasattr(actual_model.model, "visual") else None
        if visual is not None and isinstance(visual, DPSKVisionTowerAdapter):
            visual.dpsk_encoder.requires_grad_(False)
            if self._is_main_process():
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    "[OCRVL] ✓ Frozen DPSK OCR encoder (401M params). "
                    "This preserves pre-trained OCR capabilities during training. "
                    "Set OCRVL_UNFREEZE_DPSK_ENCODER=1 to disable (not recommended)."
                )

    def verify_dpsk_frozen(self) -> dict[str, Any]:
        """Verify that the DPSK encoder (now in model.visual) is properly frozen and return status.

        Returns:
            Dictionary with frozen status and parameter counts.
        """
        result = {
            "has_dpsk_encoder": False,
            "frozen": False,
            "total_params": 0,
            "trainable_params": 0,
            "frozen_params": 0,
        }

        # DPSK encoder is now in model.visual (wrapped by DPSKVisionTowerAdapter)
        # Handle DDP wrapping: if self is DDP-wrapped, access self.module.model.visual
        actual_model = self._get_actual_model()
        visual = actual_model.model.visual if hasattr(actual_model.model, "visual") else None
        if visual is not None and isinstance(visual, DPSKVisionTowerAdapter):
            result["has_dpsk_encoder"] = True

            # Count parameters from the wrapped DPSK encoder
            dpsk_encoder = visual.dpsk_encoder
            for param in dpsk_encoder.parameters():
                result["total_params"] += param.numel()
                if param.requires_grad:
                    result["trainable_params"] += param.numel()
                else:
                    result["frozen_params"] += param.numel()

            result["frozen"] = result["trainable_params"] == 0

        return result

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any):  # type: ignore[override]
        # Log that our from_pretrained is being called
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[OCRVL] OCRQwen3VLForConditionalGeneration.from_pretrained called with: {pretrained_model_name_or_path}")
        logger.info(f"[OCRVL] Loading model with DPSK encoder freeze and connector early initialization")

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if not hasattr(model.config, "ocr_offload_vit_to_cpu"):
            model.config.ocr_offload_vit_to_cpu = True

        # Check if this is a resuming from a checkpoint with saved DPSK encoder state
        has_saved_dpsk = (
            hasattr(model.config, 'dpsk_encoder_state_path') and
            model.config.dpsk_encoder_state_path and
            os.path.exists(os.path.join(pretrained_model_name_or_path, model.config.dpsk_encoder_state_path))
        )

        if has_saved_dpsk:
            # Load from saved state_dict (resuming training - self-contained checkpoint)
            try:
                model._init_dpsk_encoder_from_saved_state(pretrained_model_name_or_path)  # type: ignore[attr-defined]
                logger.info(f"[OCRVL] ✓ Loaded DPSK encoder from checkpoint state (self-contained)")
            except Exception as e:
                logger.error(f"[OCRVL] ✗ Failed to load saved DPSK encoder: {e}")
                raise
        else:
            # Create new DPSK encoder from DeepSeek-OCR path (initial training)
            try:
                model._init_dpsk_encoder_from_config(pretrained_model_name_or_path)  # type: ignore[attr-defined]
                logger.info(f"[OCRVL] ✓ DPSK encoder initialized from DeepSeek-OCR checkpoint")
            except Exception as e:
                logger.error(f"[OCRVL] ✗ Failed to initialize DPSK encoder: {e}")
                raise

        if bool(getattr(model.config, "ocr_offload_vit_to_cpu", True)):
            try:
                model.model._offload_visual_to_cpu()  # type: ignore[attr-defined]
            except Exception:
                pass

        # Early initialize OCR connectors so they can be marked trainable by LlamaFactory
        # This must happen after model loading but before training setup
        try:
            model.model._initialize_ocr_connectors_early()  # type: ignore[attr-defined]

            # Explicitly enable gradients for connector parameters
            # This must happen BEFORE FSDP wrapping for proper sharding
            for name, param in model.named_parameters():
                if 'ocr_connector' in name:
                    param.requires_grad = True
                    logger.debug(f"[OCRVL] Enabled gradient for connector: {name}")

            # Count trainable connector params
            trainable_connector_params = sum(
                p.numel() for n, p in model.named_parameters()
                if 'ocr_connector' in n and p.requires_grad
            )
            logger.info("[OCRVL] ✓ Early initialized OCR connectors for training (ocr_connector, ocr_deepstack_connector)")
            logger.info(f"[OCRVL] ✓ Enabled gradients for {trainable_connector_params:,} connector parameters")
        except Exception as e:
            logger.error(f"[OCRVL] ✗ Failed to early initialize OCR connectors: {e}")

        logger.info(f"[OCRVL] Model loading complete, DPSK encoder frozen: {model.verify_dpsk_frozen()['frozen']}")

        return model

    def save_pretrained(self, save_directory: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        # CRITICAL: Ensure config.json has model_type="ocr_qwen3_vl" so that loading
        # this checkpoint will instantiate OCRQwen3VLForConditionalGeneration with the
        # DPSKVisionTowerAdapter wrapper. Without this, the checkpoint loads as base
        # Qwen3VLForConditionalGeneration and loses the custom vision tower.
        self.config.model_type = "ocr_qwen3_vl"

        # Save DPSK encoder state_dict as part of the main checkpoint
        # This makes the model truly self-contained (no external dependency)
        if hasattr(self, 'model') and hasattr(self.model, 'visual'):
            visual = self.model.visual
            if isinstance(visual, DPSKVisionTowerAdapter) and hasattr(visual, 'dpsk_encoder'):
                # Save the DPSK encoder's state_dict
                dpsk_state_dict = visual.dpsk_encoder.state_dict()
                dpsk_path = os.path.join(save_directory, "dpsk_vision_encoder.pt")
                if self._is_main_process():
                    torch.save(dpsk_state_dict, dpsk_path)
                    logger.info(f"[OCRVL] Saved DPSK encoder state_dict to {dpsk_path}")
                # Store the relative path in config for loading
                self.config.dpsk_encoder_state_path = "dpsk_vision_encoder.pt"

        super().save_pretrained(save_directory, *args, **kwargs)

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
        ocr_pixel_values: Optional[torch.Tensor] = None,
        pixel_values_ref: Optional[torch.Tensor] = None,
        ocr_alignment: Optional[bool] = None,
        ocr_alignment_weight: Optional[float] = None,
        vision_scale: Optional[float] = None,
        text_scale: Optional[float] = None,
        latent_supervision: Optional[List[List[torch.Tensor]]] = None,
        latent_positions: Optional[torch.BoolTensor] = None,
        thinking_loss_weight: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        # Filter out output_hidden_states from kwargs to prevent conflicts with LoRA
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'output_hidden_states'}

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
            ocr_pixel_values=ocr_pixel_values,
            pixel_values_ref=pixel_values_ref,
            ocr_alignment=ocr_alignment,
            ocr_alignment_weight=ocr_alignment_weight,
            vision_scale=vision_scale,
            text_scale=text_scale,
            **filtered_kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        # Thinking loss: align projected hidden states with OCR-encoded supervision
        thinking_loss = None
        if latent_supervision is not None and latent_positions is not None:
            # Get thinking projection MLP
            thinking_proj = self.model._maybe_get_thinking_projection(
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )

            batch_size = hidden_states.shape[0]
            thinking_losses = []

            for b in range(batch_size):
                latent_mask = latent_positions[b]  # [seq_len]
                if not latent_mask.any():
                    continue

                # Extract hidden states at latent token positions
                sample_hidden = hidden_states[b][latent_mask]  # [num_latents, hidden_dim]

                # Project to latent space
                predicted_latents = thinking_proj(sample_hidden)  # [num_latents, 1280]

                # Get supervision (each chunk is [111, 1280], mean-pool to [1280])
                supervision_list = latent_supervision[b]
                supervision_latents = []
                for sup_tensor in supervision_list:
                    # Mean-pool spatial tokens to get 1280-dim representation
                    supervision_latents.append(sup_tensor.mean(dim=0))  # [1280]

                supervision_tensor = torch.stack(supervision_latents, dim=0)  # [num_latents, 1280]
                supervision_tensor = supervision_tensor.to(predicted_latents.device, predicted_latents.dtype)

                # MSE loss
                min_count = min(predicted_latents.shape[0], supervision_tensor.shape[0])
                sample_loss = torch.mean((predicted_latents[:min_count] - supervision_tensor[:min_count]) ** 2)
                thinking_losses.append(sample_loss)

            if thinking_losses:
                thinking_loss = torch.stack(thinking_losses).mean()
                weight = float(thinking_loss_weight) if thinking_loss_weight is not None else 1.0
                thinking_loss = thinking_loss * weight

        # Combine all losses
        if thinking_loss is not None:
            loss = thinking_loss if loss is None else (loss + thinking_loss)

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
        ocr_pixel_values: Optional[torch.Tensor] = None,
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

        # Build model_inputs for standard parameters
        model_inputs = {
            "input_ids": inputs,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
        }

        # Store OCR-specific parameters as temporary model attributes
        # (workaround: HuggingFace generate() doesn't pass custom kwargs to prepare_inputs_for_generation)
        self._temp_ocr_image_features = ocr_image_features
        self._temp_ocr_pixel_values = ocr_pixel_values
        self._temp_vision_scale = vision_scale
        self._temp_text_scale = text_scale

        try:
            result = super().generate(**model_inputs, **kwargs)
        finally:
            # Clean up temporary attributes
            if hasattr(self, '_temp_ocr_image_features'):
                delattr(self, '_temp_ocr_image_features')
            if hasattr(self, '_temp_ocr_pixel_values'):
                delattr(self, '_temp_ocr_pixel_values')
            if hasattr(self, '_temp_vision_scale'):
                delattr(self, '_temp_vision_scale')
            if hasattr(self, '_temp_text_scale'):
                delattr(self, '_temp_text_scale')

        return result

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
        ocr_pixel_values=None,
        deepstack_features=None,
        vision_scale=None,
        text_scale=None,
        **kwargs: Any,
    ):
        # Don't pass OCR-specific parameters to parent (it doesn't understand them)
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
            **kwargs,
        )

        model_inputs["position_ids"] = None

        # Ensure OCR features are preserved (parent class may not include them)
        if cache_position is None or cache_position[0] == 0:
            # First forward pass: use provided OCR features or temporary attributes from generate()
            if ocr_image_features is not None:
                model_inputs["ocr_image_features"] = ocr_image_features
            elif hasattr(self, '_temp_ocr_image_features') and self._temp_ocr_image_features is not None:
                model_inputs["ocr_image_features"] = self._temp_ocr_image_features
            if ocr_pixel_values is not None:
                model_inputs["ocr_pixel_values"] = ocr_pixel_values
            elif hasattr(self, '_temp_ocr_pixel_values') and self._temp_ocr_pixel_values is not None:
                model_inputs["ocr_pixel_values"] = self._temp_ocr_pixel_values

            if deepstack_features is not None:
                model_inputs["deepstack_features"] = deepstack_features
        else:
            # Subsequent passes: clear vision inputs (already processed)
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["ocr_image_features"] = None
            model_inputs["deepstack_features"] = None
            model_inputs["ocr_pixel_values"] = None

        # Use temporary attributes from generate() if parameters not provided
        if vision_scale is not None:
            model_inputs["vision_scale"] = vision_scale
        elif hasattr(self, '_temp_vision_scale') and self._temp_vision_scale is not None:
            model_inputs["vision_scale"] = self._temp_vision_scale

        if text_scale is not None:
            model_inputs["text_scale"] = text_scale
        elif hasattr(self, '_temp_text_scale') and self._temp_text_scale is not None:
            model_inputs["text_scale"] = self._temp_text_scale

        return model_inputs


# Register OCRQwen3VLForConditionalGeneration for OCRQwen3VLConfig
# sitecustomize.py handles converting Qwen3VLConfig -> OCRQwen3VLConfig automatically
# AutoModelForCausalLM.register(OCRQwen3VLConfig, OCRQwen3VLForConditionalGeneration)


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
