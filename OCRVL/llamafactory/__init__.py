"""OCRVL ↔ LlamaFactory integration helpers.

Keep this package lightweight: prefer registering plugins/templates via official
LlamaFactory hooks instead of monkeypatching internal functions.
"""

from __future__ import annotations


def register_llamafactory_extensions() -> None:
    """Register OCRVL multimodal plugins/templates."""
    from .qwen3_vl_ocrvl_template import register_ocrvl_qwen3_vl_template

    register_ocrvl_qwen3_vl_template()

