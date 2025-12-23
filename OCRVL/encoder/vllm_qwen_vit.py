#!/usr/bin/env python3
"""Helpers to run Qwen-VL vision encoders with vLLM kernels.

We reuse vLLM's inference-only ViT implementations to match production
vLLM throughput/latency. This module is intentionally small; it only
constructs the vision transformer and loads HF visual weights into it.

If vLLM sources are available at ../vllm, they will be imported directly.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Tuple

import torch
from transformers import AutoConfig, AutoModel


def _ensure_vllm_on_path() -> None:
    # Add sibling ../vllm to sys.path for local-source reuse.
    here = os.path.dirname(os.path.abspath(__file__))
    vllm_root = os.path.abspath(os.path.join(here, "..", "..", "..", "vllm"))
    if vllm_root not in sys.path:
        sys.path.insert(0, vllm_root)


def _iter_visual_weights(
    hf_model: torch.nn.Module,
    prefix: str = "visual.",
) -> Iterable[Tuple[str, torch.Tensor]]:
    # vLLM vision models expect weight names without the leading "visual."
    for name, tensor in hf_model.state_dict().items():
        if not name.startswith(prefix):
            continue
        yield name[len(prefix) :], tensor


def build_qwen3_vit(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    compile: bool = False,
    attn_backend_override: str | None = None,
) -> torch.nn.Module:
    """Build vLLM Qwen3-VL vision transformer and load HF visual weights."""
    _ensure_vllm_on_path()
    from vllm.attention.backends.registry import AttentionBackendEnum
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
    from vllm.distributed import initialize_model_parallel, init_distributed_environment
    import traceback

    # Initialize vLLM distributed environment (required even for single GPU)
    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://localhost:29500",
            local_rank=0,
            backend="nccl" if torch.cuda.is_available() else "gloo",
        )

    # Initialize vLLM model parallel groups
    try:
        initialize_model_parallel()
    except Exception as e:
        print(f"Failed to initialize vLLM distributed: {e}")
        traceback.print_exc()
        raise

    hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    vision_config = hf_config.vision_config

    # vLLM uses torch.get_default_dtype() during init; align it with target dtype.
    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        backend = None
        if attn_backend_override:
            # Accept either enum name ("FLASH_ATTN") or value.
            backend = (
                AttentionBackendEnum[attn_backend_override]
                if attn_backend_override in AttentionBackendEnum.__members__
                else AttentionBackendEnum(attn_backend_override)
            )
        vit = Qwen3_VisionTransformer(
            vision_config=vision_config,
            use_data_parallel=True,  # avoid TP init; matches single-GPU fast path
            attn_backend_override=backend,
        )
    finally:
        torch.set_default_dtype(prev_default)

    vit.to(device=device, dtype=dtype)
    vit.eval()

    # Load only visual weights from HF.
    hf_model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=dtype,
    )
    vit.load_weights(_iter_visual_weights(hf_model))
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if compile and hasattr(torch, "compile"):
        # Wrap whole model; vLLM uses custom ops for ViT attention.
        vit = torch.compile(vit, mode="max-autotune")
    return vit


def build_qwen25_vit(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    compile: bool = False,
    attn_backend_override: str | None = None,
) -> torch.nn.Module:
    """Build vLLM Qwen2.5-VL vision transformer and load HF visual weights."""
    _ensure_vllm_on_path()
    from vllm.attention.backends.registry import AttentionBackendEnum
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
    from vllm.distributed import initialize_model_parallel, init_distributed_environment

    # Initialize vLLM distributed environment (required even for single GPU)
    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://localhost:29501",  # Different port than qwen3
            local_rank=0,
            backend="nccl" if torch.cuda.is_available() else "gloo",
        )

    # Initialize vLLM model parallel groups
    initialize_model_parallel()

    hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    vision_config = hf_config.vision_config

    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        backend = None
        if attn_backend_override:
            backend = (
                AttentionBackendEnum[attn_backend_override]
                if attn_backend_override in AttentionBackendEnum.__members__
                else AttentionBackendEnum(attn_backend_override)
            )
        vit = Qwen2_5_VisionTransformer(
            vision_config=vision_config,
            use_data_parallel=True,
            attn_backend_override=backend,
        )
    finally:
        torch.set_default_dtype(prev_default)

    vit.to(device=device, dtype=dtype)
    vit.eval()

    hf_model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=dtype,
    )
    vit.load_weights(_iter_visual_weights(hf_model))
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if compile and hasattr(torch, "compile"):
        vit = torch.compile(vit, mode="max-autotune")
    return vit
