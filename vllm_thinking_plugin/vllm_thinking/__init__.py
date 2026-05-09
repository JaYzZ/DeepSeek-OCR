"""
vLLM thinking/latent decode plugin.

This plugin patches GPUModelRunner to support:
- continuous hidden-state AR when `VLLM_THINKING=1`
- discrete latent-token carry, where sampling `<latent>` injects
  `embed(<latent>) + hidden_state` into the next decode step
"""

import logging
import os

logger = logging.getLogger(__name__)

# Plugin version
__version__ = "0.1.0"

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def vllm_thinking_plugin():
    """
    Main entry point for vLLM plugin system.

    This function is called by vLLM in all processes during initialization.
    It applies the thinking mode patch to GPUModelRunner.
    """
    try:
        from vllm_thinking.qwen3vl_patch_embed_patch import (
            apply_qwen3vl_linear_patch_embed_patch,
        )

        apply_qwen3vl_linear_patch_embed_patch()
    except Exception as e:
        logger.error(f"[vLLM Thinking Plugin] Failed to apply Qwen3-VL patch-embed patch: {e}")
        raise

    enabled = _env_flag("VLLM_THINKING", default=False)
    if "VLLM_THINKING" in os.environ:
        logger.info(f"[PLUGIN] VLLM_THINKING={enabled}")
    if "OPSD_DELTA_MEMORY_ENABLED" in os.environ:
        logger.info(
            "[PLUGIN] OPSD_DELTA_MEMORY_ENABLED=%s OPSD_DELTA_MEMORY_GAMMA=%s",
            os.environ.get("OPSD_DELTA_MEMORY_ENABLED", "0"),
            os.environ.get("OPSD_DELTA_MEMORY_GAMMA", "0.5"),
        )
    logger.info("[vLLM Thinking Plugin] Initializing decode patch...")

    try:
        from vllm_thinking.runner_patch import apply_thinking_mode_patch
        apply_thinking_mode_patch()
        logger.info("[vLLM Thinking Plugin] ✓ Successfully applied decode patch")
    except Exception as e:
        logger.error(f"[vLLM Thinking Plugin] Failed to apply patch: {e}")
        raise


# For backward compatibility - also support direct import
def load_plugin():
    """Alternative entry point that can be called directly."""
    vllm_thinking_plugin()


__all__ = ['vllm_thinking_plugin', 'load_plugin', '__version__']
