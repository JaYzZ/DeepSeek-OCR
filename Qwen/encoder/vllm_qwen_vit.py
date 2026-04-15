#!/usr/bin/env python3
"""Helpers to run Qwen-VL vision encoders with installed vLLM kernels."""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import torch
from transformers import AutoConfig, AutoModel
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.distributed import initialize_model_parallel, init_distributed_environment
from vllm.distributed.parallel_state import get_tensor_model_parallel_group
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer


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

    # Check if PyTorch DDP is already running (e.g., from torchrun)
    # If so, we need to initialize vLLM's parallel state using the existing setup
    if torch.distributed.is_initialized():
        # PyTorch DDP is already initialized - adapt vLLM to use it
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))

        # First, check if vLLM model parallel is already initialized
        try:
            # If this succeeds, vLLM is already initialized, skip
            get_tensor_model_parallel_group()
        except (AssertionError, AttributeError):
            # vLLM not initialized yet - need to initialize it
            # Initialize vLLM's distributed environment using existing PyTorch setup
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method="env://",  # Use env vars from torchrun
                local_rank=local_rank,
                backend="nccl" if torch.cuda.is_available() else "gloo",
            )
            # Now initialize vLLM model parallel
            initialize_model_parallel(
                tensor_model_parallel_size=1,  # No tensor parallel for vision encoder
                pipeline_model_parallel_size=1,  # No pipeline parallel
            )
    else:
        # No distributed setup yet, initialize vLLM's distributed from scratch
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://localhost:29500",
            local_rank=0,
            backend="nccl" if torch.cuda.is_available() else "gloo",
        )

        # Initialize vLLM model parallel groups
        initialize_model_parallel()

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
        dtype=dtype,
    )
    vit.load_weights(_iter_visual_weights(hf_model))
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vit


def build_qwen25_vit(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    compile: bool = False,
    attn_backend_override: str | None = None,
) -> torch.nn.Module:
    """Build vLLM Qwen2.5-VL vision transformer and load HF visual weights."""
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
        dtype=dtype,
    )
    vit.load_weights(_iter_visual_weights(hf_model))
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if compile and hasattr(torch, "compile"):
        vit = torch.compile(vit, mode="max-autotune")
    return vit
