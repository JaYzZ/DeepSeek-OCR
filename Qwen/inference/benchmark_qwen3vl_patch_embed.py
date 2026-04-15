#!/usr/bin/env python3
"""Benchmark Qwen3-VL patch-embed variants on the same pretrained weights."""

from __future__ import annotations

import argparse
import gc
import json
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq
from project_paths import hf_path


@dataclass(frozen=True)
class BenchSpec:
    n: int
    warmup: int
    iters: int


class LinearizedPatchEmbed(nn.Module):
    """Equivalent to the repo's linearized patch-embed implementation."""

    def __init__(self, patch_embed) -> None:
        super().__init__()
        self.patch_size = patch_embed.patch_size
        self.temporal_patch_size = patch_embed.temporal_patch_size
        self.in_channels = patch_embed.in_channels
        self.embed_dim = patch_embed.embed_dim

        weight = patch_embed.proj.weight.detach()
        bias = patch_embed.proj.bias.detach() if patch_embed.proj.bias is not None else None
        in_dim = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        self.proj = nn.Linear(in_dim, self.embed_dim, bias=bias is not None, device=weight.device, dtype=weight.dtype)
        self.proj.weight.data.copy_(weight.reshape(self.embed_dim, -1))
        if bias is not None:
            self.proj.bias.data.copy_(bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        return self.proj(hidden_states.to(dtype=target_dtype))


def apply_monkey_patch(patch_embed) -> None:
    """Patch the original module in-place to bypass Conv3d/cudnn."""
    proj = patch_embed.proj

    def _fast_patch_embed(_self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = proj.weight.view(proj.out_channels, -1)
        out = hidden_states.to(weight.dtype) @ weight.T
        if proj.bias is not None:
            out = out + proj.bias
        return out

    patch_embed.forward = types.MethodType(_fast_patch_embed, patch_embed)


def bench_fn(fn, x: torch.Tensor, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def summarize_equivalence(orig_forward, monkey_forward, linear_forward, x_bf16: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        y_orig_bf16 = orig_forward(x_bf16)
        y_monkey_bf16 = monkey_forward(x_bf16)
        y_linear_bf16 = linear_forward(x_bf16)

        x_fp32 = x_bf16.float()
        y_orig_fp32 = orig_forward(x_fp32)
        y_monkey_fp32 = monkey_forward(x_fp32)
        y_linear_fp32 = linear_forward(x_fp32)

    return {
        "bf16_max_abs_diff_monkey_vs_orig": float((y_monkey_bf16 - y_orig_bf16).abs().max().item()),
        "bf16_max_abs_diff_linear_vs_orig": float((y_linear_bf16 - y_orig_bf16).abs().max().item()),
        "fp32_max_abs_diff_monkey_vs_orig": float((y_monkey_fp32 - y_orig_fp32).abs().max().item()),
        "fp32_max_abs_diff_linear_vs_orig": float((y_linear_fp32 - y_orig_fp32).abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=str(hf_path("Qwen", "Qwen3-VL-2B-Thinking")),
        help="Original Qwen3-VL checkpoint path with Conv3d patch embed.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Benchmark dtype.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="196,784,2048,8192",
        help="Comma-separated flattened patch counts to benchmark.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda")

    model_path = Path(args.model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cpu",
    )
    patch_embed = model.visual.patch_embed.to(device=device, dtype=dtype)
    patch_embed.eval()
    orig_forward = patch_embed.forward

    linear_patch_embed = LinearizedPatchEmbed(patch_embed).to(device=device, dtype=dtype)
    linear_patch_embed.eval()
    apply_monkey_patch(patch_embed)
    monkey_forward = patch_embed.forward

    patch_dim = (
        patch_embed.in_channels
        * patch_embed.temporal_patch_size
        * patch_embed.patch_size
        * patch_embed.patch_size
    )

    x_check = torch.randn(32, patch_dim, dtype=dtype, device=device)
    eq = summarize_equivalence(orig_forward, monkey_forward, linear_patch_embed.forward, x_check)

    specs: list[BenchSpec] = []
    for raw_n in args.batch_sizes.split(","):
        n = int(raw_n.strip())
        if n <= 0:
            continue
        if n < 512:
            specs.append(BenchSpec(n=n, warmup=3, iters=20))
        elif n < 2048:
            specs.append(BenchSpec(n=n, warmup=2, iters=10))
        elif n < 8192:
            specs.append(BenchSpec(n=n, warmup=1, iters=3))
        else:
            specs.append(BenchSpec(n=n, warmup=1, iters=1))

    results: dict[str, object] = {
        "model_path": str(model_path),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "dtype": args.dtype,
        "patch_dim": patch_dim,
        "equivalence": eq,
        "benchmarks": [],
    }

    print(json.dumps({k: v for k, v in results.items() if k != "benchmarks"}, indent=2), flush=True)
    print("", flush=True)
    print(f"{'N':>8}  {'orig(ms)':>12}  {'monkey(ms)':>12}  {'linear(ms)':>12}  {'orig/monkey':>12}  {'orig/linear':>12}", flush=True)

    for spec in specs:
        x = torch.randn(spec.n, patch_dim, dtype=dtype, device=device)
        t_orig = bench_fn(orig_forward, x, warmup=spec.warmup, iters=spec.iters)
        t_monkey = bench_fn(monkey_forward, x, warmup=spec.warmup, iters=spec.iters)
        t_linear = bench_fn(linear_patch_embed.forward, x, warmup=spec.warmup, iters=spec.iters)

        row = {
            "n": spec.n,
            "warmup": spec.warmup,
            "iters": spec.iters,
            "orig_ms": t_orig,
            "monkey_ms": t_monkey,
            "linear_ms": t_linear,
            "speedup_monkey": t_orig / t_monkey,
            "speedup_linear": t_orig / t_linear,
        }
        results["benchmarks"].append(row)
        print(
            f"{spec.n:>8}  {t_orig:>12.3f}  {t_monkey:>12.3f}  {t_linear:>12.3f}  "
            f"{row['speedup_monkey']:>12.1f}  {row['speedup_linear']:>12.1f}",
            flush=True,
        )
        del x
        gc.collect()
        torch.cuda.empty_cache()

    print("", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
