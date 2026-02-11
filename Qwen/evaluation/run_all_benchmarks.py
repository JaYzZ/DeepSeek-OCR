#!/usr/bin/env python3
"""
Centralized Evaluation Script for All Qwen Benchmarks

Runs all 4 benchmarks (MathVision, MMMU, RealWorldQA, ODinW-13) with 100 samples each.
Automatically selects 4 free GPUs and saves results to timestamped directories.

Usage:
    python run_all_benchmarks.py [--num-samples N] [--gpus GPU_ID,GPU_ID,...] [--skip-infer] [--skip-eval]
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import get_data_path


def get_free_gpus(num_gpus: int = 4, min_free_mb: int = 10000) -> List[int]:
    """
    Select GPUs with the most free memory.

    Args:
        num_gpus: Number of GPUs to select
        min_free_mb: Minimum free memory in MB (default: 10GB)

    Returns:
        List of GPU IDs sorted by available memory (most free first)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not query GPU status: {e}")
        print("Falling back to GPUs 0,1,2,3")
        return list(range(num_gpus))

    gpu_info = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split(',')
        if len(parts) == 2:
            gpu_id = int(parts[0].strip())
            free_mb = int(parts[1].strip())
            if free_mb >= min_free_mb:
                gpu_info.append((gpu_id, free_mb))

    # Sort by free memory (most free first)
    gpu_info.sort(key=lambda x: x[1], reverse=True)

    selected_gpus = [gpu_id for gpu_id, _ in gpu_info[:num_gpus]]

    if len(selected_gpus) < num_gpus:
        print(f"Warning: Only found {len(selected_gpus)} GPUs with >= {min_free_mb}MB free memory")
        # If we don't have enough, just take what we have
        if len(selected_gpus) == 0:
            print("No suitable GPUs found, falling back to 0,1,2,3")
            return list(range(min(4, len(gpu_info))))

    print(f"Selected GPUs: {selected_gpus}")
    for gpu_id in selected_gpus:
        free_mb = next((mem for gid, mem in gpu_info if gid == gpu_id), 0)
        print(f"  GPU {gpu_id}: {free_mb}MB free")

    return selected_gpus


def create_timestamp_dir(base_path: str) -> str:
    """Create a timestamped directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_path) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created results directory: {run_dir}")
    return str(run_dir)


def run_unified_inference(
    benchmarks: List[str],
    run_dir: str,
    num_samples: int,
    gpus: List[int],
    model_path: str,
    lora_path: str = None,
    lora_name: str = "default",
    max_lora_rank: int = 64,
    enable_lora: bool = False
) -> Dict[str, str]:
    """
    Run inference for all benchmarks using hybrid backend (auto-detects model type).
    Supports both official Qwen3-VL (vLLM) and custom Linear variant (HF Transformers).

    Args:
        benchmarks: List of benchmark names
        run_dir: Output directory
        num_samples: Number of samples to process
        gpus: List of GPU IDs to use
        model_path: Path to the model
        lora_path: Path to LoRA adapter
        lora_name: Name for LoRA adapter
        max_lora_rank: Maximum LoRA rank
        enable_lora: Whether LoRA is enabled

    Returns:
        Dictionary mapping benchmark names to output files
    """
    script_path = Path(__file__).parent / "hybrid_inference.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--model-path", model_path,
        "--output-dir", run_dir,
        "--num-samples", str(num_samples),
        "--tensor-parallel-size", str(len(gpus)),
        "--gpu-memory-utilization", "0.75",
        "--benchmarks", ",".join(benchmarks),
        "--lora-name", lora_name,
    ]

    # Add LoRA argument if provided
    if lora_path:
        cmd.extend(["--lora-path", lora_path])

    # Set GPU environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

    print(f"\n{'='*80}")
    print(f"Running HYBRID inference for {len(benchmarks)} benchmarks")
    print(f"Auto-detects model type: vLLM (official) or HF Transformers (Linear variant)")
    print(f"LoRA: {lora_path if lora_path else 'Disabled'}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPUs: {gpus}")
    print(f"Output: {run_dir}")
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
        print(f"✓ Hybrid inference completed in {elapsed:.2f}s")

        # Return mapping of benchmark to output files
        output_files = {}
        for benchmark in benchmarks:
            output_file = os.path.join(run_dir, f"{benchmark.lower()}_inference.jsonl")
            if os.path.exists(output_file):
                output_files[benchmark] = output_file
            else:
                print(f"Warning: Expected output file not found: {output_file}")

        return output_files

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ Hybrid inference failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        return {}


def run_inference(
    benchmark: str,
    run_dir: str,
    num_samples: int,
    gpus: List[int],
    model_path: str,
    extra_args: Dict[str, str] = None
) -> Tuple[bool, str]:
    """
    Run inference for a benchmark.

    Args:
        benchmark: Benchmark name (MathVision, MMMU, RealWorldQA, ODinW-13)
        run_dir: Output directory
        num_samples: Number of samples to process
        gpus: List of GPU IDs to use
        model_path: Path to the model
        extra_args: Additional arguments to pass

    Returns:
        (success, output_file) tuple
    """
    benchmark_configs = {
        "MathVision": {
            "script": "MathVision/run_mathv.py",
            "dataset": "MathVision",
            "output": "mathvision_inference.jsonl",
            "use_num_samples": True,
            "limit_at_eval": False
        },
        "MMMU": {
            "script": "mmmu/run_mmmu.py",
            "dataset": "MMMU_DEV_VAL",
            "output": "mmmu_inference.jsonl",
            "use_num_samples": False,  # MMMU loads all data during inference
            "limit_at_eval": True      # Use --limit during eval phase
        },
        "RealWorldQA": {
            "script": "RealWorldQA/run_realworldqa.py",
            "dataset": "RealWorldQA",
            "output": "realworldqa_inference.jsonl",
            "use_num_samples": False,  # RealWorldQA loads all data during inference
            "limit_at_eval": True      # Use --limit during eval phase
        },
        "ODinW-13": {
            "script": "ODinW-13/run_odinw.py",
            "dataset": None,
            "output": "odinw_inference.jsonl",
            "use_num_samples": False,  # ODinW uses --limit instead
            "limit_at_eval": False
        }
    }

    if benchmark not in benchmark_configs:
        print(f"Error: Unknown benchmark {benchmark}")
        return False, ""

    config = benchmark_configs[benchmark]
    script_path = Path(__file__).parent / config["script"]
    output_file = os.path.join(run_dir, config["output"])

    cmd = [
        sys.executable,
        str(script_path),
        "infer",
        "--model-path", model_path,
        "--output-file", output_file,
        "--tensor-parallel-size", str(len(gpus)),
        "--gpu-memory-utilization", "0.75",  # Lower to avoid OOM
    ]

    # Add dataset argument if applicable
    if config["dataset"]:
        cmd.extend(["--dataset", config["dataset"]])

    # Add sampling/limit args
    # ODinW-13 uses --limit for inference, MathVision uses --num-samples
    # MMMU and RealWorldQA don't support limiting during inference
    if num_samples > 0:
        if benchmark == "ODinW-13":
            cmd.extend(["--limit", str(num_samples)])
        elif config["use_num_samples"]:
            cmd.extend(["--num-samples", str(num_samples)])
        # For MMMU and RealWorldQA, we'll limit during evaluation phase

    # Add extra arguments if provided
    if extra_args:
        for key, value in extra_args.items():
            cmd.extend([f"--{key}", value])

    # Set GPU environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

    # Set sample limit for MMMU and RealWorldQA via environment variable
    # These benchmarks don't support --num-samples, so we limit at dataset load time
    if num_samples > 0 and benchmark in ["MMMU", "RealWorldQA"]:
        env["EVAL_NUM_SAMPLES"] = str(num_samples)
        print(f"✓ Set EVAL_NUM_SAMPLES={num_samples} for {benchmark}")

    print(f"\n{'='*80}")
    print(f"Running inference for {benchmark}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPUs: {gpus}")
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
        return True, output_file
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {benchmark} inference failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False, ""


def run_evaluation(
    benchmark: str,
    input_file: str,
    run_dir: str,
    num_samples: int,
    data_dir: str = None
) -> Tuple[bool, str]:
    """
    Run evaluation for a benchmark.

    Args:
        benchmark: Benchmark name
        input_file: Input file with inference results
        run_dir: Output directory
        num_samples: Number of samples to evaluate
        data_dir: Data directory override

    Returns:
        (success, result_file) tuple
    """
    benchmark_configs = {
        "MathVision": {
            "script": "MathVision/run_mathv.py",
            "dataset": "MathVision",
            "output": "mathvision_eval_result.csv",
            "result_key": "mathvision_eval_result_eval_score.csv",
            "limit_at_eval": False  # Already limited during inference
        },
        "MMMU": {
            "script": "mmmu/run_mmmu.py",
            "dataset": "MMMU_DEV_VAL",
            "output": "mmmu_eval_result.csv",
            "result_key": "mmmu_eval_result_acc.json",
            "limit_at_eval": True   # Need to limit during evaluation
        },
        "RealWorldQA": {
            "script": "RealWorldQA/run_realworldqa.py",
            "dataset": "RealWorldQA",
            "output": "realworldqa_eval_result.csv",
            "result_key": "realworldqa_eval_result_acc.json",
            "limit_at_eval": True   # Need to limit during evaluation
        },
        "ODinW-13": {
            "script": "ODinW-13/run_odinw.py",
            "dataset": None,
            "output": "odinw_eval_result.json",
            "result_key": None,
            "limit_at_eval": False  # Already limited during inference
        }
    }

    if benchmark not in benchmark_configs:
        print(f"Error: Unknown benchmark {benchmark}")
        return False, ""

    config = benchmark_configs[benchmark]
    script_path = Path(__file__).parent / config["script"]
    output_file = os.path.join(run_dir, config["output"])

    # Get judge server URL from environment
    judge_url = os.environ.get('JUDGE_SERVER_URL', 'http://47.111.147.142:8600')

    cmd = [
        sys.executable,
        str(script_path),
        "eval",
        "--input-file", input_file,
        "--output-file", output_file,
    ]

    # Add API args only for benchmarks that use JUDGE server (not ODinW-13 which uses COCO eval)
    if benchmark != "ODinW-13":
        cmd.extend(["--api-type", "custom"])
        cmd.extend(["--api-url", judge_url])

    # Add dataset argument if applicable
    if config["dataset"]:
        cmd.extend(["--dataset", config["dataset"]])

    # Add data dir if specified
    if data_dir:
        cmd.extend(["--data-dir", data_dir])

    # Add limit only for benchmarks that need it (MMMU, RealWorldQA)
    if num_samples > 0 and config["limit_at_eval"]:
        cmd.extend(["--limit", str(num_samples)])

    print(f"\n{'='*80}")
    print(f"Running evaluation for {benchmark}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent
        )
        elapsed = time.time() - start_time
        print(f"✓ {benchmark} evaluation completed in {elapsed:.2f}s")

        # Return the result file (metrics file)
        if config["result_key"]:
            result_file = os.path.join(run_dir, config["result_key"])
        else:
            result_file = output_file

        return True, result_file
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {benchmark} evaluation failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False, ""


def parse_benchmark_results(benchmark: str, result_file: str) -> Dict:
    """
    Parse evaluation results from a benchmark.

    Args:
        benchmark: Benchmark name
        result_file: Path to result file

    Returns:
        Dictionary with parsed metrics
    """
    if not os.path.exists(result_file):
        return {"error": "Result file not found"}

    try:
        if benchmark == "ODinW-13":
            # ODinW outputs JSON with mAP metrics
            with open(result_file, 'r') as f:
                data = json.load(f)
            # Extract average mAP
            avg_map = data.get("Average", 0.0)
            return {
                "mAP": avg_map,
                "raw": data
            }

        elif benchmark in ["MathVision"]:
            # MathVision outputs CSV score file
            import pandas as pd
            df = pd.read_csv(result_file)
            # Look for accuracy column
            if 'accuracy' in df.columns:
                acc = df['accuracy'].iloc[0] if len(df) > 0 else 0.0
                return {"accuracy": acc, "raw": df.to_dict()}
            elif 'acc' in df.columns:
                acc = df['acc'].iloc[0] if len(df) > 0 else 0.0
                return {"accuracy": acc, "raw": df.to_dict()}
            else:
                return {"error": "No accuracy column found", "raw": df.to_dict()}

        elif benchmark in ["MMMU", "RealWorldQA"]:
            # MMMU and RealWorldQA output JSON with accuracy
            with open(result_file, 'r') as f:
                data = json.load(f)
            acc = data.get("overall_accuracy", 0.0)
            return {
                "accuracy": acc,
                "raw": data
            }

        else:
            return {"error": f"Unknown benchmark format: {benchmark}"}

    except Exception as e:
        return {"error": str(e)}


def generate_summary(run_dir: str, results: Dict[str, Dict]) -> str:
    """
    Generate a comprehensive summary of all benchmark results.

    Args:
        run_dir: Run directory path
        results: Dictionary of benchmark results

    Returns:
        Path to summary file
    """
    summary_file = os.path.join(run_dir, "SUMMARY.md")

    with open(summary_file, 'w') as f:
        f.write(f"# Comprehensive Evaluation Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Run Directory**: `{run_dir}`\n\n")

        # Table of contents
        f.write("## Results Overview\n\n")
        f.write("| Benchmark | Metric | Score |\n")
        f.write("|-----------|--------|-------|\n")

        for benchmark, result in results.items():
            if "error" in result:
                f.write(f"| {benchmark} | ERROR | {result['error']} |\n")
            elif "mAP" in result:
                f.write(f"| {benchmark} | mAP | {result['mAP']:.4f} |\n")
            elif "accuracy" in result:
                f.write(f"| {benchmark} | Accuracy | {result['accuracy']:.4f} |\n")
            else:
                f.write(f"| {benchmark} | N/A | N/A |\n")

        f.write("\n## Detailed Results\n\n")

        # Write detailed results for each benchmark
        for benchmark, result in results.items():
            f.write(f"### {benchmark}\n\n")

            if "error" in result:
                f.write(f"**Error**: {result['error']}\n\n")
            elif "mAP" in result:
                f.write(f"**mAP**: {result['mAP']:.4f}\n\n")
                if "raw" in result and isinstance(result["raw"], dict):
                    f.write("**Per-dataset mAP**:\n\n")
                    for ds, metrics in result["raw"].items():
                        if ds != "Average" and isinstance(metrics, dict):
                            f.write(f"- {ds}: {metrics.get('mAP', 0.0):.4f}\n")
                    f.write("\n")
            elif "accuracy" in result:
                f.write(f"**Accuracy**: {result['accuracy']:.4f}\n\n")
                if "raw" in result and isinstance(result["raw"], dict):
                    if "accuracy_by_split" in result["raw"]:
                        f.write("**Accuracy by Split**:\n\n")
                        for split, acc in result["raw"]["accuracy_by_split"].items():
                            f.write(f"- {split}: {acc:.4f}\n")
                        f.write("\n")
            else:
                f.write("No detailed metrics available.\n\n")

        # Calculate overall average (excluding ODinW-13 which uses mAP)
        f.write("## Overall Statistics\n\n")
        acc_scores = [
            r["accuracy"] for r in results.values()
            if "accuracy" in r and "error" not in r
        ]
        if acc_scores:
            avg_acc = sum(acc_scores) / len(acc_scores)
            f.write(f"**Average Accuracy (MathVision, MMMU, RealWorldQA)**: {avg_acc:.4f}\n\n")

        map_scores = [
            r["mAP"] for r in results.values()
            if "mAP" in r and "error" not in r
        ]
        if map_scores:
            f.write(f"**Average mAP (ODinW-13)**: {map_scores[0]:.4f}\n\n")

    print(f"\n{'='*80}")
    print(f"Summary written to: {summary_file}")
    print(f"{'='*80}\n")

    # Print summary to console
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    print(f"\n| Benchmark | Metric | Score |")
    print("|-----------|--------|-------|")
    for benchmark, result in results.items():
        if "error" in result:
            print(f"| {benchmark} | ERROR | {result['error']} |")
        elif "mAP" in result:
            print(f"| {benchmark} | mAP | {result['mAP']:.4f} |")
        elif "accuracy" in result:
            print(f"| {benchmark} | Accuracy | {result['accuracy']:.4f} |")
        else:
            print(f"| {benchmark} | N/A | N/A |")
    print("\n" + "="*80 + "\n")

    return summary_file


def main():
    parser = argparse.ArgumentParser(
        description="Centralized evaluation script for all Qwen benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with 100 samples each, auto-select 4 GPUs
  python run_all_benchmarks.py --num-samples 100

  # Run with LoRA checkpoint from training
  python run_all_benchmarks.py --num-samples 100 \\
      --model-path Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking \\
      --enable-lora \\
      --lora-path Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_20260209_235504/checkpoint-1000

  # Run only inference (skip evaluation)
  python run_all_benchmarks.py --num-samples 100 --skip-eval

  # Run only evaluation on existing inference results
  python run_all_benchmarks.py --skip-infer --run-dir results/run_20240129_120000

  # Specify GPUs manually
  python run_all_benchmarks.py --num-samples 100 --gpus 0,1,2,3
        """
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per benchmark (default: 100)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated GPU IDs (default: auto-select 4 free GPUs)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path("/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking")),
        help="Path to the model"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Existing run directory (for skip-infer mode)"
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Skip inference, only run evaluation"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation, only run inference"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="MathVision,MMMU,RealWorldQA,ODinW-13",
        help="Comma-separated list of benchmarks to run (default: all)"
    )

    # LoRA arguments
    parser.add_argument(
        "--enable-lora",
        action="store_true",
        help="Enable LoRA adapter (uses unified inference path)"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default="default",
        help="Name for LoRA adapter (default: default)"
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=64,
        help="Maximum LoRA rank (default: 64)"
    )

    args = parser.parse_args()

    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(',')]

    # GPU selection
    if args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(',')]
        print(f"Using manually specified GPUs: {gpus}")
    else:
        gpus = get_free_gpus(num_gpus=4)

    if len(gpus) == 0:
        print("Error: No GPUs available")
        return 1

    # Create run directory
    if args.run_dir:
        run_dir = args.run_dir
        print(f"Using existing run directory: {run_dir}")
    else:
        base_results_path = Path(__file__).parent / "results"
        run_dir = create_timestamp_dir(str(base_results_path))

    # Results tracking
    all_results = {}

    # Run inference
    if not args.skip_infer:
        print("\n" + "="*80)
        print("PHASE 1: INFERENCE")
        print("="*80 + "\n")

        # Use unified inference if LoRA is enabled (single model load, faster)
        if args.enable_lora:
            inference_files = run_unified_inference(
                benchmarks=benchmarks,
                run_dir=run_dir,
                num_samples=args.num_samples,
                gpus=gpus,
                model_path=args.model_path,
                lora_path=args.lora_path,
                lora_name=args.lora_name,
                max_lora_rank=args.max_lora_rank,
                enable_lora=args.enable_lora
            )
            if not inference_files:
                print("Error: Unified inference failed or produced no output files")
                return 1
        else:
            # Use individual benchmark scripts (original behavior)
            inference_files = {}
            for benchmark in benchmarks:
                success, output_file = run_inference(
                    benchmark=benchmark,
                    run_dir=run_dir,
                    num_samples=args.num_samples,
                    gpus=gpus,
                    model_path=args.model_path
                )
                if success:
                    inference_files[benchmark] = output_file
                else:
                    print(f"Warning: {benchmark} inference failed, skipping...")

    # Run evaluation
    if not args.skip_eval:
        print("\n" + "="*80)
        print("PHASE 2: EVALUATION")
        print("="*80 + "\n")

        # Determine input files
        if args.skip_infer:
            # Try to find existing inference files in run_dir
            inference_files = {}
            for benchmark in benchmarks:
                possible_files = [
                    os.path.join(run_dir, f"{benchmark.lower().replace('-', '')}_inference.jsonl"),
                    os.path.join(run_dir, f"{benchmark.lower()}_inference.jsonl"),
                ]
                for pf in possible_files:
                    if os.path.exists(pf):
                        inference_files[benchmark] = pf
                        break
                if benchmark not in inference_files:
                    print(f"Warning: Could not find inference file for {benchmark}")

        for benchmark, input_file in inference_files.items():
            success, result_file = run_evaluation(
                benchmark=benchmark,
                input_file=input_file,
                run_dir=run_dir,
                num_samples=args.num_samples
            )
            if success:
                parsed = parse_benchmark_results(benchmark, result_file)
                all_results[benchmark] = parsed
            else:
                all_results[benchmark] = {"error": "Evaluation failed"}

    # Generate summary
    if all_results:
        generate_summary(run_dir, all_results)

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
