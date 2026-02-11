#!/usr/bin/env python3
"""
HF Transformers Inference Backend - For custom Qwen3-VL-Linear variant

Simpler approach: Call individual benchmark scripts with HF-compatible arguments.
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict


def detect_model_type(model_path: str) -> str:
    """Detect if model uses linear patch embed (HF required) or standard (vLLM OK)."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if config.get("vision_config", {}).get("patch_embed_type") == "linear":
            return "hf"
        if "Linear" in model_path:
            return "hf"
    return "vllm"


def run_benchmark_inference_hf(
    benchmark: str,
    model_path: str,
    lora_path: str,
    output_file: str,
    num_samples: int,
    data_dir: str = None
) -> bool:
    """Run a single benchmark using its native script with HF-compatible settings."""
    benchmark_configs = {
        "MathVision": {
            "script": "MathVision/run_mathv.py",
            "dataset": "MathVision",
            "use_num_samples": True,
        },
        "MMMU": {
            "script": "mmmu/run_mmmu.py",
            "dataset": "MMMU_DEV_VAL",
            "use_num_samples": False,  # MMMU loads all, limit via env var
        },
        "RealWorldQA": {
            "script": "RealWorldQA/run_realworldqa.py",
            "dataset": "RealWorldQA",
            "use_num_samples": False,  # RealWorldQA loads all, limit via env var
        },
        "ODinW-13": {
            "script": "ODinW-13/run_odinw.py",
            "dataset": None,
            "use_num_samples": False,  # Uses --limit parameter
        }
    }

    if benchmark not in benchmark_configs:
        print(f"Error: Unknown benchmark {benchmark}")
        return False

    config = benchmark_configs[benchmark]
    script_path = Path(__file__).parent / config["script"]

    # Build command for individual benchmark script
    cmd = [
        sys.executable,
        str(script_path),
        "infer",
        "--model-path", model_path,
        "--output-file", output_file,
    ]

    if config["dataset"]:
        cmd.extend(["--dataset", config["dataset"]])

    # Add sampling/limit args
    if num_samples > 0:
        if benchmark == "ODinW-13":
            cmd.extend(["--limit", str(num_samples)])
        elif config["use_num_samples"]:
            cmd.extend(["--num-samples", str(num_samples)])
        # MMMU and RealWorldQA use environment variable

    # Set environment for sample limiting
    env = os.environ.copy()
    if num_samples > 0 and benchmark in ["MMMU", "RealWorldQA"]:
        env["EVAL_NUM_SAMPLES"] = str(num_samples)

    # Disable vLLM for HF models - these scripts will detect HF compatibility
    # The scripts should use transformers directly when model requires it
    env["USE_HF_BACKEND"] = "1"

    print(f"\n{'='*80}")
    print(f"Running {benchmark} inference with HF backend")
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
            cwd=Path(__file__).parent
        )
        elapsed = time.time() - start_time
        print(f"✓ {benchmark} inference completed in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {benchmark} inference failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False


def main(
    model_path: str,
    lora_path: str = None,
    output_dir: str = None,
    num_samples: int = 100,
    benchmarks: List[str] = None,
    **kwargs
) -> int:
    """Main HF inference entry point."""
    print("\n" + "="*80)
    print("🚀 HF TRANSFORMERS INFERENCE (Linear Variant)")
    print("="*80)
    print(f"Model: {model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    # Run each benchmark
    total_start = time.time()
    success_count = 0

    for benchmark in benchmarks:
        output_file = os.path.join(output_dir, f"{benchmark.lower()}_inference.jsonl")

        if run_benchmark_inference_hf(
            benchmark=benchmark,
            model_path=model_path,
            lora_path=lora_path,
            output_file=output_file,
            num_samples=num_samples
        ):
            success_count += 1
        else:
            print(f"Warning: {benchmark} failed, continuing...")

    total_elapsed = time.time() - total_start

    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Completed: {success_count}/{len(benchmarks)} benchmarks")
    print(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")

    return 0 if success_count == len(benchmarks) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HF Transformers inference for Linear variant")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--benchmarks", type=str, default="MathVision,MMMU,RealWorldQA,ODinW-13")

    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(',')]

    sys.exit(main(
        model_path=args.model_path,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        benchmarks=benchmarks
    ))
