"""Shared FlashAttention compatibility helpers."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _native_varlen_requires_num_splits(flash_attn_gpu) -> bool:
    """Return whether the native varlen binding expects the trailing num_splits arg."""
    doc = getattr(getattr(flash_attn_gpu, "varlen_fwd", None), "__doc__", "") or ""
    return "arg21" in doc


def apply_flash_attn_varlen_compat_patch() -> None:
    """Patch flash-attn varlen forward when the Python wrapper lags the native binding.

    Some flash-attn builds expose ``varlen_fwd(..., gen_, num_splits=0)`` in the
    CUDA extension while the Python wrapper still calls it without the trailing
    ``num_splits`` argument. Qwen3VL training hits that varlen path and fails
    with a ``TypeError``. This patch keeps the fix local to our repo.
    """
    try:
        import flash_attn.flash_attn_interface as flash_attn_interface
    except Exception:
        return

    if getattr(flash_attn_interface, "_deepseek_varlen_compat_patch", False):
        return

    flash_attn_gpu = getattr(flash_attn_interface, "flash_attn_gpu", None)
    if flash_attn_gpu is None or not _native_varlen_requires_num_splits(flash_attn_gpu):
        flash_attn_interface._deepseek_varlen_compat_patch = True
        return

    maybe_contiguous = flash_attn_interface.maybe_contiguous
    def compat_flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left: int = -1,
        window_size_right: int = -1,
        softcap: float = 0.0,
        alibi_slopes=None,
        return_softmax: bool = False,
        block_table=None,
        leftpad_k=None,
        seqused_k=None,
        zero_tensors: bool = False,
    ) -> Tuple:
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        return flash_attn_gpu.varlen_fwd(
            q,
            k,
            v,
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            leftpad_k,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            zero_tensors,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            return_softmax,
            None,
            0,
        )

    compat_flash_attn_varlen_forward._deepseek_compat_patch = True

    flash_attn_interface._flash_attn_varlen_forward = compat_flash_attn_varlen_forward
    flash_attn_interface._wrapped_flash_attn_varlen_forward = compat_flash_attn_varlen_forward
    flash_attn_interface._deepseek_varlen_compat_patch = True

    logger.info("Applied FlashAttention varlen compat patch with explicit num_splits=0")


__all__ = ["apply_flash_attn_varlen_compat_patch"]
