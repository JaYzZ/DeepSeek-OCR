"""OCRVL ↔ LlamaFactory integration helpers.

Keep this package lightweight: prefer registering plugins/templates via official
LlamaFactory hooks instead of monkeypatching internal functions.
"""

from __future__ import annotations


def register_ocrvl_templates() -> None:
    """Register OCRVL Qwen3-VL templates (including qwen3vl_latent for thinking training)."""
    from .qwen3_vl_ocrvl_template import register_ocrvl_qwen3_vl_template

    register_ocrvl_qwen3_vl_template()

