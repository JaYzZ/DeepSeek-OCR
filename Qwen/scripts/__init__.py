"""
Qwen training scripts package.

This package contains scripts for training Qwen3VL models with
latent supervision support.
"""

from .llamafactory_integration import _patch_once

__all__ = ["_patch_once"]
