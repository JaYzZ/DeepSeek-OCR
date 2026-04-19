from __future__ import annotations

import pytest
import torch


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _run_flash_attn_fa2_probe(
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype_name: str = "bf16",
    causal: bool = False,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    import flash_attn
    from flash_attn import flash_attn_func

    device = torch.device("cuda:0")
    dtype = _resolve_dtype(dtype_name)
    capability = torch.cuda.get_device_capability(device)
    device_name = torch.cuda.get_device_name(device)

    shape = (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    out = flash_attn_func(q, k, v, causal=causal)
    loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)

    return {
        "status": "ok",
        "mode": "dense",
        "device_name": device_name,
        "device_capability": list(capability),
        "torch_version": torch.__version__,
        "flash_attn_version": getattr(flash_attn, "__version__", "unknown"),
        "output_shape": list(out.shape),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dtype": dtype_name,
        "causal": causal,
    }


def _run_flash_attn_varlen_probe(
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype_name: str = "bf16",
    causal: bool = False,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    import flash_attn
    from flash_attn import flash_attn_varlen_func

    device = torch.device("cuda:0")
    dtype = _resolve_dtype(dtype_name)
    capability = torch.cuda.get_device_capability(device)
    device_name = torch.cuda.get_device_name(device)

    lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    total_tokens = int(cu_seqlens[-1].item())

    shape = (total_tokens, num_heads, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        seq_len,
        seq_len,
        causal=causal,
    )
    loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)

    return {
        "status": "ok",
        "mode": "varlen",
        "device_name": device_name,
        "device_capability": list(capability),
        "torch_version": torch.__version__,
        "flash_attn_version": getattr(flash_attn, "__version__", "unknown"),
        "output_shape": list(out.shape),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dtype": dtype_name,
        "causal": causal,
        "total_tokens": total_tokens,
    }


@pytest.mark.gpu
def test_flash_attn_ready() -> None:
    result = _run_flash_attn_fa2_probe()
    assert result["status"] == "ok"


@pytest.mark.gpu
def test_flash_attn_varlen_ready() -> None:
    result = _run_flash_attn_varlen_probe()
    assert result["status"] == "ok"
