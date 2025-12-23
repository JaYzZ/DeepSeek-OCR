#!/usr/bin/env python3
"""Microbenchmark for Qwen2.5-VL / Qwen3-VL vision encoders.

Reports:
- preprocessing latency (HF processor)
- forward latency (encoder only)
- end-to-end latency (preprocess + forward)

This is meant to be run on the same GPU setup where you run vLLM serving,
so the numbers are directly comparable. On CPU, this is still functional
but not representative of production vLLM speed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from OCRVL.encoder.qwen25vl_encoder import Qwen25VLEncoder  # noqa: E402
from OCRVL.encoder.qwen3vl_encoder import Qwen3VLEncoder  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _now() -> float:
    return time.perf_counter()


def _make_images(batch: int, size: int) -> list[Image.Image]:
    # Solid-color images avoid decode overhead; we benchmark model + processor.
    return [Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)) for _ in range(batch)]


@torch.no_grad()
def bench_encoder(encoder: Any, batch: int, size: int, warmup: int, iters: int) -> dict[str, float]:
    device = encoder.device
    images = _make_images(batch, size)

    # Warmup
    for _ in range(warmup):
        _ = encoder.encode_images(images)
        _sync(device)

    # Timed runs
    pre_ms = []
    fwd_ms = []
    e2e_ms = []

    for _ in range(iters):
        # Preprocess
        t0 = _now()
        processed = encoder.processor.image_processor(images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, dtype=encoder.dtype)
        grid_thw = processed["image_grid_thw"].to(device=device)
        _sync(device)
        t1 = _now()

        # Forward (match encoder internal logic)
        out = encoder.vision_model(pixel_values, grid_thw)
        _sync(device)
        t2 = _now()

        pre_ms.append((t1 - t0) * 1e3)
        fwd_ms.append((t2 - t1) * 1e3)
        e2e_ms.append((t2 - t0) * 1e3)

        # Avoid holding onto outputs across iterations.
        del out

    def _p50(xs: list[float]) -> float:
        return float(np.percentile(xs, 50))

    def _p90(xs: list[float]) -> float:
        return float(np.percentile(xs, 90))

    return {
        "pre_ms_p50": _p50(pre_ms),
        "pre_ms_p90": _p90(pre_ms),
        "fwd_ms_p50": _p50(fwd_ms),
        "fwd_ms_p90": _p90(fwd_ms),
        "e2e_ms_p50": _p50(e2e_ms),
        "e2e_ms_p90": _p90(e2e_ms),
        "images_per_s_e2e_p50": (batch / (_p50(e2e_ms) / 1e3)),
        "images_per_s_fwd_p50": (batch / (_p50(fwd_ms) / 1e3)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen25", "qwen3"], required=True)
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--no-vllm-kernels", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    use_vllm_kernels = not args.no_vllm_kernels

    if args.model == "qwen3":
        enc = Qwen3VLEncoder(
            model_name_or_path=args.model_path
            or "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Instruct",
            device=args.device,
            dtype=dtype,
            use_vllm_kernels=use_vllm_kernels,
            compile=args.compile,
        )
    else:
        enc = Qwen25VLEncoder(
            model_name_or_path=args.model_path
            or "/share/project/xiyan/huggingface/Qwen/Qwen2.5-VL-3B-Instruct",
            device=args.device,
            dtype=dtype,
            use_vllm_kernels=use_vllm_kernels,
            compile=args.compile,
        )

    stats = bench_encoder(enc, args.batch, args.size, args.warmup, args.iters)

    print(f"model={args.model} device={args.device} dtype={args.dtype} batch={args.batch} size={args.size}")
    for k, v in stats.items():
        if k.endswith("_ms_p50") or k.endswith("_ms_p90"):
            print(f"{k}={v:.3f}")
        else:
            print(f"{k}={v:.3f}")


if __name__ == "__main__":
    main()
