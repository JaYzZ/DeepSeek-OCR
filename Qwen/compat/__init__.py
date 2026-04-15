"""Compatibility helpers for Qwen runtime patches."""

from .patch_embed import (
    apply_all_qwen3vl_patch_embed_fixes,
    apply_llamafactory_qwen3vl_conv3d_guard_patch,
    apply_transformers_qwen3vl_patch_embed_patch,
    qwen3vl_patch_embed_monkey_enabled,
)

__all__ = [
    "apply_all_qwen3vl_patch_embed_fixes",
    "apply_llamafactory_qwen3vl_conv3d_guard_patch",
    "apply_transformers_qwen3vl_patch_embed_patch",
    "qwen3vl_patch_embed_monkey_enabled",
]
