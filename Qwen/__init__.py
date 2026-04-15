"""Qwen package entrypoint.

Keep package import side effects minimal so non-training processes
(for example Ray agents) do not accidentally import the heavy
LLaMA-Factory latent integration path.
"""

from __future__ import annotations

import os


def _latent_patches_enabled() -> bool:
    return os.environ.get("QWEN3VL_LATENT_SUPERVISION", "0") == "1"


if _latent_patches_enabled():
    from .llamafactory.integration import _patch_once  # noqa: F401


__all__ = []
