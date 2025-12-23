"""
Transformers Compatibility Patch for DeepSeek-OCR

This module patches the incompatible imports in DeepSeek-OCR's model code
to work with newer transformers versions.

The issue: DeepSeek-OCR tries to import LlamaFlashAttention2 which doesn't exist
in transformers 4.46+. We patch it to use the correct class names.
"""

import sys
import logging
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)


def patch_transformers_imports():
    """
    Patch transformers imports to make DeepSeek-OCR model code work

    This adds missing classes that DeepSeek-OCR expects but aren't in newer transformers.
    """
    try:
        import transformers.models.llama.modeling_llama as llama_module

        # Check if LlamaFlashAttention2 exists
        if not hasattr(llama_module, 'LlamaFlashAttention2'):
            logger.info("Patching transformers: Adding LlamaFlashAttention2")

            # Use Llama attention as the base
            if hasattr(llama_module, 'LlamaAttention'):
                llama_module.LlamaFlashAttention2 = llama_module.LlamaAttention
            else:
                # Create a mock class as fallback
                class LlamaFlashAttention2:
                    pass
                llama_module.LlamaFlashAttention2 = LlamaFlashAttention2

            logger.info("  ✓ LlamaFlashAttention2 patched")

        # Check for other potentially missing classes
        if not hasattr(llama_module, 'LlamaSdpaAttention'):
            logger.info("Patching transformers: Adding LlamaSdpaAttention")
            if hasattr(llama_module, 'LlamaAttention'):
                llama_module.LlamaSdpaAttention = llama_module.LlamaAttention
            logger.info("  ✓ LlamaSdpaAttention patched")

        return True

    except Exception as e:
        logger.error(f"Failed to patch transformers: {e}")
        return False


def apply_patches():
    """Apply all necessary patches"""
    logger.info("Applying transformers compatibility patches...")
    success = patch_transformers_imports()

    if success:
        logger.info("✓ All patches applied successfully")
    else:
        logger.warning("⚠ Some patches failed - model loading may fail")

    return success


# Auto-apply patches when module is imported
apply_patches()
