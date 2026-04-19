"""Shared Qwen3-VL patch-embed monkey patch helpers."""

from __future__ import annotations

import functools
import logging
import os

import torch
import torch.nn as nn
from llamafactory.extras import packages as lf_packages
from llamafactory.model import loader as lf_loader

logger = logging.getLogger(__name__)


def _patch_mode() -> str:
    """Return the selected patch-embed mode.

    Modes:
    - ``monkey``: patch original Qwen3-VL Conv3d patch-embed with matmul
    - ``off``: disable the monkey patch
    - ``linearized``: keep compatibility mode only for pre-converted checkpoints
    """
    return os.environ.get("QWEN3VL_PATCH_EMBED_MODE", "monkey").strip().lower()


def qwen3vl_patch_embed_monkey_enabled() -> bool:
    return _patch_mode() == "monkey"


def apply_all_qwen3vl_patch_embed_fixes() -> None:
    """Apply the repo-wide Qwen3-VL patch-embed related fixes."""
    apply_llamafactory_qwen3vl_conv3d_guard_patch()
    apply_transformers_qwen3vl_patch_embed_patch()


def apply_llamafactory_qwen3vl_conv3d_guard_patch() -> None:
    """Bypass LlamaFactory's torch-2.9 Conv3d hard stop for monkey-patched Qwen3-VL."""
    if not qwen3vl_patch_embed_monkey_enabled():
        return

    if getattr(lf_packages, "_deepseek_qwen3vl_conv3d_guard_patch", False):
        return

    original_version_check = lf_packages.is_torch_version_greater_than

    @functools.wraps(original_version_check)
    def patched_version_check(content: str):
        value = str(content).strip()
        if value in {"2.9.0", "2.10.0"} and qwen3vl_patch_embed_monkey_enabled():
            return False
        return original_version_check(content)

    lf_packages.is_torch_version_greater_than = patched_version_check
    lf_packages._deepseek_qwen3vl_conv3d_guard_patch = True

    lf_loader.is_torch_version_greater_than = patched_version_check

    logger.info("Patched LlamaFactory torch-2.9 Conv3d guard for monkey-patched Qwen3-VL")


def apply_transformers_qwen3vl_patch_embed_patch() -> None:
    """Patch HF Qwen3-VL patch-embed to bypass the pathological Conv3d path."""
    if not qwen3vl_patch_embed_monkey_enabled():
        return

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

    if getattr(Qwen3VLVisionPatchEmbed, "_deepseek_monkey_patch_embed_patch", False):
        return

    original_forward = Qwen3VLVisionPatchEmbed.forward

    @functools.wraps(original_forward)
    def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        proj = getattr(self, "proj", None)
        if not isinstance(proj, nn.Conv3d):
            return original_forward(self, hidden_states)

        weight = proj.weight.view(proj.out_channels, -1)
        out = hidden_states.to(weight.dtype) @ weight.t()
        if proj.bias is not None:
            out = out + proj.bias
        return out

    Qwen3VLVisionPatchEmbed.forward = patched_forward
    Qwen3VLVisionPatchEmbed._deepseek_monkey_patch_embed_patch = True
    logger.info("Applied Qwen3-VL patch-embed monkey patch for Transformers")


__all__ = [
    "apply_all_qwen3vl_patch_embed_fixes",
    "apply_llamafactory_qwen3vl_conv3d_guard_patch",
    "apply_transformers_qwen3vl_patch_embed_patch",
    "qwen3vl_patch_embed_monkey_enabled",
]
