#!/usr/bin/env python3
"""
Unified Inference Script - Run all benchmarks with single model load

This script loads vLLM once and runs inference on all benchmarks sequentially,
then exits. This saves 3-4 minutes by avoiding redundant model loading.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Qwen.inference.vllm_utils import apply_runtime_env_for_thinking, resolve_lora_artifacts
from project_paths import hf_path

def run_vllm_inference(
    model_path: str,
    lora_path: str,
    output_dir: str,
    num_samples: int,
    benchmarks: list,
    tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.75,
    lora_name: str = "default",
):
    """Run inference using vLLM backend by calling individual benchmark scripts."""
    adapter_meta = resolve_lora_artifacts(model_path, lora_path)
    resolved_model_path = adapter_meta["model_path"] or model_path
    resolved_lora_rank = adapter_meta["lora_rank"] if adapter_meta["lora_rank"] is not None else 64

    print("\n" + "="*80)
    print("🚀 UNIFIED INFERENCE - vLLM Backend")
    print("="*80)
    print(f"Model: {resolved_model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
        print(f"LoRA rank: {resolved_lora_rank}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Total timing
    total_start = time.time()

    # Track completed benchmarks (for resume support)
    completed = []

    for benchmark in benchmarks:
        output_file = os.path.join(output_dir, f"{benchmark.lower()}_inference.jsonl")

        # Skip if already exists (resume support)
        if os.path.exists(output_file):
            print(f"⏭️  Skipping {benchmark} (output exists): {output_file}")
            completed.append(benchmark)
            continue

        # Build command for individual benchmark script
        benchmark_scripts = {
            "MathVision": "mathvision/run_mathv.py",
            "MMMU": "mmmu/run_mmmu.py",
            "RealWorldQA": "realworldqa/run_realworldqa.py",
            "ODinW-13": "odinw/run_odinw.py",
            "M3CoT": "m3cot/run_m3cot.py",
        }

        if benchmark not in benchmark_scripts:
            print(f"⚠️  Unknown benchmark: {benchmark}")
            continue

        script_path = Path(__file__).parent / benchmark_scripts[benchmark]

        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            "infer",
            "--model-path", resolved_model_path,
            "--output-file", output_file,
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
        ]

        # Add dataset argument if applicable
        datasets = {
            "MathVision": "MathVision",
            "MMMU": "MMMU_DEV_VAL",
            "RealWorldQA": "RealWorldQA",
            "M3CoT": "M3CoT",
        }
        if benchmark in datasets:
            cmd.extend(["--dataset", datasets[benchmark]])

        # Add sampling/limit args
        if num_samples > 0:
            if benchmark == "ODinW-13":
                cmd.extend(["--limit", str(num_samples)])
            elif benchmark in ("MathVision", "MMMU", "RealWorldQA", "M3CoT"):
                cmd.extend(["--num-samples", str(num_samples)])

        env = os.environ.copy()

        # Set LoRA checkpoint path for VAE loading (if provided)
        if lora_path:
            env["VLLM_LORA_CHECKPOINT_PATH"] = lora_path

        # Add LoRA arguments if provided
        if lora_path:
            cmd.extend([
                "--enable-lora",
                "--lora-path", lora_path,
                "--lora-name", lora_name,
            ])

        print(f"\n{'='*80}")
        print(f"Running {benchmark} inference")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Output: {output_file}")
        print(f"{'='*80}\n")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                cwd=Path(__file__).parent,
                capture_output=False  # Show output in real-time
            )
            elapsed = time.time() - start_time
            print(f"\n✓ {benchmark} inference completed in {elapsed:.2f}s")
            completed.append(benchmark)

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n✗ {benchmark} inference failed after {elapsed:.2f}s")
            print(f"Error: {e}")
            print(f"Return code: {e.returncode}")
            # Print stderr if available
            if e.stderr:
                print(f"STDERR: {e.stderr[:1000]}")
            # Continue to next benchmark instead of stopping
            print(f"⚠️  Continuing to next benchmark...")

    total_elapsed = time.time() - total_start

    print("\n" + "="*80)
    print("✅ ALL INFERENCE COMPLETED")
    print("="*80)
    print(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Completed benchmarks: {', '.join(completed)}")
    if len(completed) < len(benchmarks):
        skipped = [b for b in benchmarks if b not in completed]
        print(f"Skipped/Failed benchmarks: {', '.join(skipped)}")
    print(f"\nAll results saved to: {output_dir}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Inference - vLLM Backend")
    parser.add_argument("--model-path", type=str, default=str(hf_path("Qwen", "Qwen3-VL-2B-Thinking")))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--benchmarks", type=str, default="MathVision,MMMU,RealWorldQA",
                       help="Comma-separated list of benchmarks to run")

    # LoRA arguments - note: vLLM backend requires scripts to support LoRA
    parser.add_argument("--enable-lora", action="store_true", help="Enable LoRA support")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--lora-name", type=str, default="default", help="Name for LoRA adapter")
    args = parser.parse_args()
    apply_runtime_env_for_thinking(repo_root=_REPO_ROOT)

    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(',')]

    # Run inference
    run_vllm_inference(
        model_path=args.model_path,
        lora_path=args.lora_path if args.enable_lora else None,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        benchmarks=benchmarks,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora_name=args.lora_name,
    )


if __name__ == "__main__":
    main()
