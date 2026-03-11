"""
Qwen3VL training package with latent supervision support.

This package provides integration with LlamaFactory for training
the official Qwen3VL thinking model with latent supervision.

Latent supervision is automatically enabled when:
- QWEN3VL_LATENT_SUPERVISION=1 environment variable is set
- Training config name contains "thinking"

The integration handles:
1. Latent injection at <latent> positions
2. Latent supervision loading from .latent.pt files
3. Thinking loss computation (REPA/Contrastive/OT loss)
4. Model patching for thinking_projection module
"""

import warnings

# Import the integration on package import
# This applies patches when PYTHONPATH includes the repo root
try:
    from .llamafactory.integration import _patch_once
    # Patches are applied automatically in integration.py
except ImportError as exc:
    warnings.warn(
        f"[Qwen3VL] Failed to import latent integration patches: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )

__all__ = []
