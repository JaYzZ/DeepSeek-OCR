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


def _defer_cuda_touching_patches() -> bool:
    return os.environ.get("QWEN3VL_DEFER_VERL_PATCHES") == "1"


def _configure_torch_sharing_strategy() -> None:
    import torch.multiprocessing as mp

    # FD limit: Use file_system sharing strategy to avoid FD-per-tensor
    mp.set_sharing_strategy("file_system")
    if os.environ.get("LOCAL_RANK", "0") == "0":
        logging.warning("[sitecustomize] Set torch multiprocessing sharing strategy to 'file_system' to avoid FD limits")


if not _defer_cuda_touching_patches():
    _configure_torch_sharing_strategy()


def _qwen_patches_enabled() -> bool:
    """Whether Qwen LlamaFactory patches should be applied in this process."""
    return os.environ.get("QWEN3VL_LATENT_SUPERVISION", "0") == "1"


def _ocrvl_patches_enabled() -> bool:
    """Whether OCRVL sitecustomize patches should be applied in this process."""
    flag = os.environ.get("OCRVL_APPLY_PATCHES")
    if flag is None:
        return False
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def _verl_patches_enabled() -> bool:
    flag = os.environ.get("QWEN3VL_APPLY_VERL_PATCHES")
    if flag is None:
        return False
    if os.environ.get("QWEN3VL_DEFER_VERL_PATCHES") == "1":
        return False
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def _patch_once() -> None:
    """Apply all patches."""
    if _defer_cuda_touching_patches():
        return

    logger = logging.getLogger(__name__)

    try:
        from Qwen.compat.patch_embed import apply_all_qwen3vl_patch_embed_fixes

        apply_all_qwen3vl_patch_embed_fixes()
    except ImportError as e:
        logger.warning(f"[sitecustomize] Failed to import Qwen patch-embed fix: {e}")
    except Exception as e:
        logger.warning(f"[sitecustomize] Failed to apply Qwen patch-embed fix: {e}")

    # Apply Qwen patches (latent supervision, callbacks, tokenizer, etc.)
    if _qwen_patches_enabled():
        try:
            from Qwen.llamafactory import integration as qwen_integration

            if hasattr(qwen_integration, 'apply_qwen_patches'):
                qwen_integration.apply_qwen_patches(logger)
            elif hasattr(qwen_integration, '_patch_once'):
                qwen_integration._patch_once()
        except ImportError as e:
            logger.warning(f"[sitecustomize] Failed to import Qwen patches: {e}")
        except Exception as e:
            logger.warning(f"[sitecustomize] Failed to apply Qwen patches: {e}")

    # Apply OCRVL patches (OCRVL model support)
    if _ocrvl_patches_enabled():
        try:
            from OCRVL.llamafactory import integration as ocrvl_integration

            if hasattr(ocrvl_integration, 'apply_ocrvl_patches'):
                ocrvl_integration.apply_ocrvl_patches(logger)
        except ImportError as e:
            logger.warning(f"[sitecustomize] Failed to import OCRVL patches: {e}")
        except Exception as e:
            logger.warning(f"[sitecustomize] Failed to apply OCRVL patches: {e}")

    if _verl_patches_enabled():
        try:
            from verl_compat import apply_runtime_compat_patches

            apply_runtime_compat_patches()
            import verl_compat.reward_manager  # noqa: F401
        except ImportError as e:
            logger.warning(f"[sitecustomize] Failed to import VERL patches: {e}")
        except Exception as e:
            logger.warning(f"[sitecustomize] Failed to apply VERL patches: {e}")


# Apply patches on import
_patch_once()
