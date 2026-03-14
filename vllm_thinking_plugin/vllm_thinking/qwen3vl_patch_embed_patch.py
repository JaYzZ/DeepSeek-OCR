"""vLLM Qwen3-VL compatibility patches used by the thinking plugin."""

from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def _patch_embed_weight_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
) -> None:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    if not isinstance(loaded_weight, torch.Tensor):
        loaded_weight = loaded_weight[:]

    if loaded_weight.dim() == 2 and param.dim() == 5:
        expected_flat_dim = (
            param.shape[1] * param.shape[2] * param.shape[3] * param.shape[4]
        )
        if loaded_weight.shape == (param.shape[0], expected_flat_dim):
            loaded_weight = loaded_weight.reshape_as(param)

    default_weight_loader(param, loaded_weight)


def apply_qwen3vl_linear_patch_embed_patch() -> None:
    """Allow vLLM Qwen3-VL to load linearized patch-embed weights."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from vllm.model_executor.models import qwen3_vl

    patch_cls = qwen3_vl.Qwen3_VisionPatchEmbed
    if getattr(patch_cls, "_deepseek_linear_patch_embed_patch", False):
        _PATCH_APPLIED = True
        return

    original_init = patch_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.proj.weight.weight_loader = _patch_embed_weight_loader

    patch_cls.__init__ = patched_init
    patch_cls._deepseek_linear_patch_embed_patch = True
    _PATCH_APPLIED = True
    logger.info(
        "Applied Qwen3-VL linear patch-embed compatibility patch for vLLM"
    )
