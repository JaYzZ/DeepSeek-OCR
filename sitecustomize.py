"""
LlamaFactory integration hub.

This is the main entry point for all model-specific patches.
It delegates to:
- Qwen.llamafactory.integration: Qwen3VL-specific patches
- OCRVL.llamafactory.integration: OCRVL-specific patches

sitecustomize.py is loaded automatically by Python's site-packages mechanism,
making it the ideal place to apply global patches before any model imports.
"""

from __future__ import annotations

import logging
import os
import torch.multiprocessing as mp

# FD limit: Use file_system sharing strategy to avoid FD-per-tensor
mp.set_sharing_strategy("file_system")
if os.environ.get("LOCAL_RANK", "0") == "0":
    logging.warning("[sitecustomize] Set torch multiprocessing sharing strategy to 'file_system' to avoid FD limits")

# Import Qwen patches (for Qwen3VL training)
try:
    from Qwen.llamafactory import integration as qwen_integration
except ImportError:
    qwen_integration = None

# Import OCRVL patches (for OCRVL models)
try:
    from OCRVL.llamafactory import integration as ocrvl_integration
except ImportError:
    ocrvl_integration = None


def _patch_once() -> None:
    """Apply all patches."""
    logger = logging.getLogger(__name__)

    # Apply Qwen patches (latent supervision, callbacks, tokenizer, etc.)
    if qwen_integration is not None:
        try:
            if hasattr(qwen_integration, 'apply_qwen_patches'):
                qwen_integration.apply_qwen_patches(logger)
            elif hasattr(qwen_integration, '_patch_once'):
                qwen_integration._patch_once()
        except Exception as e:
            logger.warning(f"[sitecustomize] Failed to apply Qwen patches: {e}")

    # Apply OCRVL patches (OCRVL model support)
    if ocrvl_integration is not None:
        try:
            if hasattr(ocrvl_integration, 'apply_ocrvl_patches'):
                ocrvl_integration.apply_ocrvl_patches(logger)
        except Exception as e:
            logger.warning(f"[sitecustomize] Failed to apply OCRVL patches: {e}")


# Apply patches on import
_patch_once()
