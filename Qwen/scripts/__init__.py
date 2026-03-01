"""
Qwen training scripts package.

This package contains scripts for training Qwen3VL models with
latent supervision support.

Note: The latent supervision integration has been migrated to Qwen/llamafactory/integration.py
"""

from Qwen.llamafactory.integration import _patch_once

__all__ = ["_patch_once"]
