#!/usr/bin/env python3
"""
Hybrid Inference Backend - Supports both official Qwen3-VL and custom Linear variant

Automatically detects model type and uses appropriate backend:
- Official Qwen3-VL → vLLM (fast, tensor parallel)
- Linear variant → HF Transformers (compatible)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Literal, Optional

# Determine which backend to use
def detect_model_type(model_path: str) -> Literal["vllm", "hf"]:
    """Detect if model uses vLLM-compatible architecture or requires HF."""
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        # Assume vLLM compatible if no config found
        return "vllm"

    with open(config_path) as f:
        config = json.load(f)

    # Check for linear patch embed (not compatible with vLLM)
    if config.get("vision_config", {}).get("patch_embed_type") == "linear":
        return "hf"

    # Check model name
    if "Linear" in model_path:
        return "hf"

    # Default to vLLM
    return "vllm"


def run_vllm_inference(
    model_path: str,
    lora_path: Optional[str],
    output_dir: str,
    num_samples: int,
    benchmarks: list,
    **kwargs
):
    """Run inference using vLLM backend."""
    print("\n" + "="*80)
    print("Using vLLM Backend (Fast)")
    print("="*80 + "\n")

    # Import vLLM module
    sys.path.insert(0, str(Path(__file__).parent))
    from run_all_inference import main as vllm_main

    # Build args for vLLM
    sys.argv = [
        "run_all_inference.py",
        "--model-path", model_path,
        "--output-dir", output_dir,
        "--num-samples", str(num_samples),
        "--benchmarks", ",".join(benchmarks),
        "--tensor-parallel-size", str(kwargs.get("tensor_parallel_size", 4)),
        "--gpu-memory-utilization", str(kwargs.get("gpu_memory_utilization", 0.75)),
    ]

    if lora_path:
        sys.argv.extend([
            "--enable-lora",
            "--lora-path", lora_path,
            "--lora-name", kwargs.get("lora_name", "default"),
        ])

    # Run vLLM inference
    return vllm_main()


def run_hf_inference(
    model_path: str,
    lora_path: Optional[str],
    output_dir: str,
    num_samples: int,
    benchmarks: list,
    **kwargs
):
    """Run inference using HF Transformers backend (for Linear variant)."""
    print("\n" + "="*80)
    print("Using HF Transformers Backend (Compatible with Linear variant)")
    print("="*80 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    # Import HF inference module (to be created)
    sys.path.insert(0, str(Path(__file__).parent))
    from run_hf_inference import main as hf_main

    return hf_main(
        model_path=model_path,
        lora_path=lora_path,
        output_dir=output_dir,
        num_samples=num_samples,
        benchmarks=benchmarks,
        **kwargs
    )


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid inference - auto-detects model type and uses appropriate backend"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--lora-name", type=str, default="default")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--benchmarks", type=str, default="MathVision,MMMU,RealWorldQA,ODinW-13")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)

    args = parser.parse_args()

    # Detect model type
    model_type = detect_model_type(args.model_path)

    print(f"\n{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Detected type: {model_type.upper()}")
    print(f"Backend: {'vLLM (fast)' if model_type == 'vllm' else 'HF Transformers (compatible)'}")
    print(f"{'='*80}\n")

    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(',')]

    # Route to appropriate backend
    kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "lora_name": args.lora_name,
    }

    if model_type == "vllm":
        return run_vllm_inference(
            model_path=args.model_path,
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            benchmarks=benchmarks,
            **kwargs
        )
    else:
        return run_hf_inference(
            model_path=args.model_path,
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            benchmarks=benchmarks,
            **kwargs
        )


if __name__ == "__main__":
    sys.exit(main())
