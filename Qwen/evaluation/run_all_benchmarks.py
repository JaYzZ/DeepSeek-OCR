#!/usr/bin/env python3
"""
Centralized Evaluation Script for All Qwen Benchmarks

Runs all 4 benchmarks (MathVision, MMMU, RealWorldQA, ODinW-13) with 100 samples each.
Automatically selects 4 free GPUs and saves results to timestamped directories.

Usage:
    python run_all_benchmarks.py [--num-samples N] [--gpus GPU_ID,GPU_ID,...] [--skip-infer] [--skip-eval]
    tmux new-session -d -s benchmark -c /share/project/xiyan/sources/DeepSeek-OCR "bash -lc 'python -u Qwen/evaluation/run_all_benchmarks.py --start-server --lora-path Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_20260221_223934/checkpoint-1812 --num-samples 100 --gpus 0,1,2,3 2>&1 | tee /tmp/bench.log'"
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import time
import signal
import socket
import errno
import platform
import requests
import threading
import concurrent.futures
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from transformers import AutoProcessor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import get_data_path, QWEN3_VL_2B_THINKING


class BenchmarkLogger:
    """Logger that writes to both console and bench.log file."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.log_file = os.path.join(run_dir, "bench.log")
        self._file = None

    def __enter__(self):
        self._file = open(self.log_file, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
        return False

    def log(self, message: str, to_console: bool = True):
        """Log message to file and optionally to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        if self._file:
            self._file.write(log_line + "\n")
            self._file.flush()
        if to_console:
            print(log_line)

    def log_section(self, title: str):
        """Log a section header."""
        self.log("=" * 60)
        self.log(title)
        self.log("=" * 60)

    def log_command(self, cmd: str):
        """Log a command being executed."""
        self.log(f"COMMAND: {cmd}")

    def log_dict(self, title: str, d: dict):
        """Log a dictionary as key=value pairs."""
        self.log(f"{title}:")
        for k, v in d.items():
            self.log(f"  {k}={v}")


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
    enable_lora: bool = False,
    logger: "BenchmarkLogger" = None
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
        logger: Optional BenchmarkLogger instance

    Returns:
        Dictionary mapping benchmark names to output files
    """
    script_path = Path(__file__).parent / "run_all_inference.py"

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

    # Enable thinking mode for continuous latent AR
    env["VLLM_THINKING_MODE_ENABLED"] = "1"
    env["VLLM_THINKING_AUTO_PATCH"] = "1"

    # Set LoRA checkpoint path for VAE loading
    if lora_path:
        env["VLLM_LORA_CHECKPOINT_PATH"] = lora_path

    # Log detailed information
    if logger:
        logger.log_section("RUNNING UNIFIED INFERENCE")
        logger.log(f"Model: {model_path}")
        logger.log(f"LoRA: {lora_path if lora_path else 'Disabled'}")
        logger.log(f"Benchmarks: {', '.join(benchmarks)}")
        logger.log(f"Num samples: {num_samples}")
        logger.log(f"GPUs: {gpus}")
        logger.log(f"Tensor parallel size: {len(gpus)}")
        logger.log(f"Output dir: {run_dir}")
        logger.log_command(" ".join(cmd))
        logger.log_dict("Environment", {
            "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
            "VLLM_THINKING_MODE_ENABLED": env["VLLM_THINKING_MODE_ENABLED"],
            "VLLM_THINKING_AUTO_PATCH": env["VLLM_THINKING_AUTO_PATCH"],
        })

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
        # Capture stdout and stderr for logging
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent
        )

        # Stream output to both console and log
        if logger:
            logger.log("--- Inference output start ---")

        for line in process.stdout:
            print(line, end='')  # Print to console
            if logger:
                logger.log(f"  [subprocess] {line.rstrip()}", to_console=False)

        process.wait()

        if logger:
            logger.log("--- Inference output end ---")

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        elapsed = time.time() - start_time
        print(f"✓ Hybrid inference completed in {elapsed:.2f}s")

        if logger:
            logger.log(f"Inference completed in {elapsed:.2f}s")

        # Return mapping of benchmark to output files
        output_files = {}
        for benchmark in benchmarks:
            output_file = os.path.join(run_dir, f"{benchmark.lower()}_inference.jsonl")
            if os.path.exists(output_file):
                output_files[benchmark] = output_file
                if logger:
                    # Count lines in output file
                    with open(output_file) as f:
                        num_lines = sum(1 for _ in f)
                    logger.log(f"Output file: {output_file} ({num_lines} samples)")
            else:
                print(f"Warning: Expected output file not found: {output_file}")
                if logger:
                    logger.log(f"Warning: Output file not found: {output_file}")

        return output_files

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ Hybrid inference failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        if logger:
            logger.log(f"ERROR: Inference failed after {elapsed:.2f}s: {e}")
        return {}


def run_server_inference(
    benchmarks: List[str],
    run_dir: str,
    num_samples: int,
    server_url: str,
    gpus: List[int],
    concurrency: int = 1,
    logger: "BenchmarkLogger" = None
) -> Dict[str, str]:
    """
    Run inference for all benchmarks using existing vLLM server.

    Args:
        benchmarks: List of benchmark names
        run_dir: Output directory
        num_samples: Number of samples to process
        server_url: URL of the vLLM server
        logger: Optional BenchmarkLogger instance

    Returns:
        Dictionary mapping benchmark names to output files
    """

    if logger:
        logger.log_section("SERVER INFERENCE MODE")
        logger.log(f"Server URL: {server_url}")
        logger.log(f"Benchmarks: {', '.join(benchmarks)}")
        logger.log(f"Num samples: {num_samples}")
        logger.log(f"Concurrency: {concurrency}")

    # Check server health
    print(f"Checking server health at {server_url}/health...")
    try:
        resp = requests.get(f"{server_url}/health", timeout=10)
        print(f"Health check response: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Server health check failed: {resp.status_code}")
            return {}
        server_info = resp.json()
        print(f"Server info: {server_info}")
        if logger:
            logger.log(f"Server info: {server_info}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        import traceback
        traceback.print_exc()
        return {}

    # Import benchmark-specific functions
    sys.path.insert(0, str(Path(__file__).parent))

    # Many dataset utilities expect LMUData to be set. Default to evaluation's data dir.
    os.environ.setdefault("LMUData", str(Path(__file__).parent / "data"))

    from MathVision.dataset_utils import load_dataset as load_mathv_dataset, dump_image as mathv_dump_image
    from MathVision.run_mathv import build_mathv_prompt
    from mmmu.dataset_utils import load_dataset as load_mmmu_dataset, dump_image as mmmu_dump_image
    from mmmu.run_mmmu import build_mmmu_prompt
    from RealWorldQA.dataset_utils import load_dataset as load_realworldqa_dataset, dump_image as realworldqa_dump_image
    from RealWorldQA.run_realworldqa import build_realworldqa_prompt

    # Load processor
    model_path = server_info.get("model", QWEN3_VL_2B_THINKING)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    output_files = {}

    # Dataset configs
    dataset_configs = {
        "MathVision": {"dataset": "MathVision", "load_func": load_mathv_dataset, "prompt_func": build_mathv_prompt, "dump_image": mathv_dump_image},
        "MMMU": {"dataset": "MMMU_DEV_VAL", "load_func": load_mmmu_dataset, "prompt_func": build_mmmu_prompt, "dump_image": mmmu_dump_image},
        "RealWorldQA": {"dataset": "RealWorldQA", "load_func": load_realworldqa_dataset, "prompt_func": build_realworldqa_prompt, "dump_image": realworldqa_dump_image},
        # ODinW is handled by calling its own benchmark script in API mode.
        "ODinW-13": {"dataset": None, "load_func": None, "prompt_func": None, "dump_image": None},
    }

    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Processing benchmark: {benchmark}")
        print(f"{'='*60}")
        if logger:
            logger.log_section(f"INFERENCE: {benchmark}")

        # ODinW-13 has a more complex job generator; reuse its existing script in API mode.
        if benchmark == "ODinW-13":
            success, output_file = run_inference(
                benchmark=benchmark,
                run_dir=run_dir,
                num_samples=num_samples,
                gpus=gpus,  # only for CUDA_VISIBLE_DEVICES; inference uses server via --api-url
                model_path=model_path,
                server_url=server_url,
                logger=logger,
            )
            if not success:
                print("ODinW-13 inference failed in server mode")
                if logger:
                    logger.log("Error: ODinW-13 inference failed in server mode")
                continue
            output_files[benchmark] = output_file
            continue

        output_file = os.path.join(run_dir, f"{benchmark.lower()}_inference.jsonl")
        config = dataset_configs.get(benchmark)
        if not config:
            print(f"Unknown benchmark: {benchmark}")
            continue

        # Load dataset
        dataset_name = config["dataset"]
        load_func = config["load_func"]
        prompt_func = config["prompt_func"]
        dump_image_impl = config.get("dump_image")

        print(f"Loading dataset for {benchmark}...")
        try:
            # Most dataset loaders accept only a dataset name (or even no args). Some older
            # versions accepted a limit argument; support both signatures.
            if dataset_name:
                try:
                    data = load_func(dataset_name)
                except TypeError:
                    data = load_func(dataset_name, num_samples if num_samples > 0 else None)
            else:
                try:
                    data = load_func()
                except TypeError:
                    data = load_func(num_samples if num_samples > 0 else None)
            print(f"Loaded {len(data)} samples for {benchmark}")
        except Exception as e:
            print(f"Failed to load dataset for {benchmark}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Normalize data to a list of row-like dicts for consistent iteration.
        # Pandas DataFrame iteration yields column names, so avoid `for x in df`.
        if hasattr(data, "to_dict") and hasattr(data, "iterrows"):
            try:
                rows = data.to_dict(orient="records")
            except TypeError:
                rows = [row.to_dict() for _, row in data.iterrows()]
        else:
            rows = list(data)

        # Limit samples
        if num_samples > 0 and num_samples < len(rows):
            rows = rows[:num_samples]

        if logger:
            logger.log(f"Loaded {len(rows)} samples")

        # Set up per-benchmark image dumping and prompt construction.
        lmu_data = os.environ.get("LMUData", str(Path(__file__).parent / "data"))
        img_root = os.path.join(lmu_data, "images", dataset_name) if dataset_name else os.path.join(lmu_data, "images")
        os.makedirs(img_root, exist_ok=True)

        def dump_image_func(line):
            # Most dataset_utils.dump_image signatures are (line, img_root).
            if dump_image_impl is None:
                return None
            return dump_image_impl(line, img_root)

        # RealWorldQA prompt needs min/max pixels; use processor defaults.
        default_min_pixels = getattr(processor.image_processor, "min_pixels", 28 * 28 * 256)
        default_max_pixels = getattr(processor.image_processor, "max_pixels", 28 * 28 * 2048)

        # Process samples
        results_by_idx: Dict[int, Dict[str, object]] = {}
        start_time = time.time()

        # Build prompts sequentially (avoids races when dumping images to disk).
        request_tasks: List[Tuple[int, object, List[dict], dict]] = []
        for idx, row in tqdm(enumerate(rows), total=len(rows), desc=f"{benchmark} build"):
            try:
                if benchmark in ("MathVision", "MMMU"):
                    messages = prompt_func(row, dump_image_func, dataset_name)
                elif benchmark == "RealWorldQA":
                    messages = prompt_func(row, dump_image_func, default_min_pixels, default_max_pixels)
                else:
                    messages = prompt_func(row, dataset_name if dataset_name else None)

                # Convert to API format - extract images and text from messages
                api_messages = []
                for msg in messages:
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        processed_content = []
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "image":
                                    # Normalize local file URIs like "file:///abs/path" to paths.
                                    img = item.get("image")
                                    if isinstance(img, str) and img.startswith("file://"):
                                        item = dict(item)
                                        item["image"] = img[len("file://"):]
                                    processed_content.append(item)
                                elif item.get("type") == "text":
                                    processed_content.append(item)
                        api_messages.append({"role": msg.get("role", "user"), "content": processed_content})
                    else:
                        api_messages.append({"role": msg.get("role", "user"), "content": content})

                payload = {
                    "messages": api_messages,
                    "max_tokens": 8192,
                    "temperature": 0.0,
                }
                request_tasks.append((idx, row, messages, payload))
            except Exception as e:
                print(f"Error building prompt for sample {idx}: {e}")
                continue

        def call_server(task: Tuple[int, object, List[dict], dict]) -> Tuple[int, Optional[Dict[str, object]], Optional[str]]:
            idx, row, messages, payload = task
            url = f"{server_url}/v1/chat/completions"
            last_err = None
            for attempt in range(3):
                try:
                    resp = requests.post(url, json=payload, timeout=300)
                    if resp.status_code != 200:
                        last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        raise RuntimeError(last_err)

                    result_data = resp.json()
                    response_text = result_data["choices"][0]["message"]["content"]
                    # Keep raw model output (may include <think>...</think>) and
                    # also provide a stripped variant for evaluation.
                    response_raw = response_text
                    response_final = (
                        response_raw.split("</think>")[-1].strip()
                        if "</think>" in response_raw
                        else response_raw
                    )
                    out = {
                        "question_id": idx,
                        "annotation": row.to_dict() if hasattr(row, "to_dict") else dict(row),
                        "task": benchmark,
                        "result": {"gen": response_final, "gen_raw": response_raw},
                        "messages": messages,
                    }
                    return idx, out, None
                except Exception as e:
                    last_err = str(e)
                    # Small jittered backoff to avoid thundering herd on transient errors.
                    time.sleep((2 ** attempt) + random.random())
            return idx, None, last_err

        # Execute requests concurrently to keep vLLM busy.
        effective_concurrency = max(1, int(concurrency or 1))
        if effective_concurrency == 1:
            for task in tqdm(request_tasks, total=len(request_tasks), desc=f"{benchmark} infer"):
                idx, out, err = call_server(task)
                if out is not None:
                    results_by_idx[idx] = out
                else:
                    print(f"Error for sample {idx}: {err}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_concurrency) as ex:
                futures = [ex.submit(call_server, task) for task in request_tasks]
                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{benchmark} infer"):
                    idx, out, err = fut.result()
                    if out is not None:
                        results_by_idx[idx] = out
                    else:
                        print(f"Error for sample {idx}: {err}")

        # Save results
        with open(output_file, 'w') as f:
            for idx in sorted(results_by_idx.keys()):
                f.write(json.dumps(results_by_idx[idx]) + '\n')

        elapsed = time.time() - start_time
        print(f"✓ {benchmark} inference completed in {elapsed:.2f}s ({len(results_by_idx)} samples)")

        if logger:
            sps = (len(results_by_idx) / elapsed) if elapsed > 0 else 0.0
            logger.log(f"Completed: {len(results_by_idx)} samples in {elapsed:.2f}s ({sps:.2f} samples/s)")

        output_files[benchmark] = output_file

    return output_files


def run_inference(
    benchmark: str,
    run_dir: str,
    num_samples: int,
    gpus: List[int],
    model_path: str,
    server_url: str = None,
    extra_args: Dict[str, str] = None,
    logger: "BenchmarkLogger" = None
) -> Tuple[bool, str]:
    """
    Run inference for a benchmark.

    Args:
        benchmark: Benchmark name (MathVision, MMMU, RealWorldQA, ODinW-13)
        run_dir: Output directory
        num_samples: Number of samples to process
        gpus: List of GPU IDs to use
        model_path: Path to the model
        server_url: URL of the vLLM server (e.g., http://localhost:8016)
        extra_args: Additional arguments to pass
        logger: Optional BenchmarkLogger instance

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
            # MMMU now supports limiting inference via --num-samples / --limit.
            "use_num_samples": True,
            "limit_at_eval": True      # Keep optional eval limiting as well
        },
        "RealWorldQA": {
            "script": "RealWorldQA/run_realworldqa.py",
            "dataset": "RealWorldQA",
            "output": "realworldqa_inference.jsonl",
            # RealWorldQA now supports limiting inference via --num-samples / --limit.
            "use_num_samples": True,
            "limit_at_eval": True      # Keep optional eval limiting as well
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

    # Set LOCAL_API_URL and --api-url for benchmark scripts that support it
    benchmarks_with_api_url = ["MathVision", "MMMU", "RealWorldQA", "ODinW-13"]
    if server_url and benchmark in benchmarks_with_api_url:
        api_full_url = f"{server_url}/v1/chat/completions"
        env["LOCAL_API_URL"] = api_full_url
        # Also pass as CLI argument
        cmd.extend(["--api-url", api_full_url])
        if logger:
            logger.log(f"Set LOCAL_API_URL={api_full_url}")

    # Log detailed info
    if logger:
        logger.log_section(f"INFERENCE: {benchmark}")
        logger.log(f"Script: {script_path}")
        logger.log(f"Model: {model_path}")
        logger.log(f"Output: {output_file}")
        logger.log(f"GPUs: {gpus}")
        logger.log_command(" ".join(cmd))

    print(f"\n{'='*80}")
    print(f"Running inference for {benchmark}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPUs: {gpus}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # Capture stdout and stderr for logging
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent
        )

        # Stream output to both console and log
        if logger:
            logger.log(f"--- {benchmark} inference output start ---", to_console=False)

        for line in process.stdout:
            print(line, end='')  # Print to console
            if logger:
                logger.log(f"  [{benchmark}] {line.rstrip()}", to_console=False)

        process.wait()

        if logger:
            logger.log(f"--- {benchmark} inference output end ---", to_console=False)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        elapsed = time.time() - start_time
        print(f"✓ {benchmark} inference completed in {elapsed:.2f}s")

        if logger:
            logger.log(f"Inference completed in {elapsed:.2f}s")
            # Count lines in output file
            if os.path.exists(output_file):
                with open(output_file) as f:
                    num_lines = sum(1 for _ in f)
                logger.log(f"Output file has {num_lines} samples")

        return True, output_file
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {benchmark} inference failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        if logger:
            logger.log(f"ERROR: {benchmark} inference failed after {elapsed:.2f}s: {e}")
        return False, ""


def run_evaluation(
    benchmark: str,
    input_file: str,
    run_dir: str,
    num_samples: int,
    data_dir: str = None,
    logger: "BenchmarkLogger" = None
) -> Tuple[bool, str]:
    """
    Run evaluation for a benchmark.

    Args:
        benchmark: Benchmark name
        input_file: Input file with inference results
        run_dir: Output directory
        num_samples: Number of samples to evaluate
        data_dir: Data directory override
        logger: Optional BenchmarkLogger instance

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

    # Log detailed info
    if logger:
        logger.log_section(f"EVALUATION: {benchmark}")
        logger.log(f"Input file: {input_file}")
        logger.log(f"Output file: {output_file}")
        logger.log(f"Judge URL: {judge_url}")
        logger.log_command(" ".join(cmd))

    print(f"\n{'='*80}")
    print(f"Running evaluation for {benchmark}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # Capture stdout and stderr for logging
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent
        )

        # Stream output to both console and log
        if logger:
            logger.log(f"--- {benchmark} evaluation output start ---", to_console=False)

        for line in process.stdout:
            print(line, end='')  # Print to console
            if logger:
                logger.log(f"  [{benchmark}] {line.rstrip()}", to_console=False)

        process.wait()

        if logger:
            logger.log(f"--- {benchmark} evaluation output end ---", to_console=False)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        elapsed = time.time() - start_time
        print(f"✓ {benchmark} evaluation completed in {elapsed:.2f}s")

        if logger:
            logger.log(f"Evaluation completed in {elapsed:.2f}s")

        # Return the result file (metrics file)
        if config["result_key"]:
            result_file = os.path.join(run_dir, config["result_key"])
        else:
            result_file = output_file

        if logger:
            logger.log(f"Result file: {result_file}")

        return True, result_file
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {benchmark} evaluation failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        if logger:
            logger.log(f"ERROR: {benchmark} evaluation failed after {elapsed:.2f}s: {e}")
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
                # Normalize to percentage for consistent presentation.
                acc = float(acc)
                if 0.0 <= acc <= 1.0:
                    acc *= 100.0
                return {"accuracy": acc, "raw": df.to_dict()}
            elif 'acc' in df.columns:
                acc = df['acc'].iloc[0] if len(df) > 0 else 0.0
                acc = float(acc)
                if 0.0 <= acc <= 1.0:
                    acc *= 100.0
                return {"accuracy": acc, "raw": df.to_dict()}
            else:
                return {"error": "No accuracy column found", "raw": df.to_dict()}

        elif benchmark in ["MMMU", "RealWorldQA"]:
            # MMMU and RealWorldQA output JSON with accuracy
            with open(result_file, 'r') as f:
                data = json.load(f)
            acc = data.get("overall_accuracy", 0.0)
            acc = float(acc)
            if 0.0 <= acc <= 1.0:
                acc *= 100.0
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

    def _format_map_score(val: float) -> str:
        """Format mAP consistently with accuracy-style reporting.

        ODinW-13 mAP is typically in [0, 1]; downstream consumers here prefer
        'points' (e.g. 0.2304 -> 23.04) for readability in reports.
        """
        try:
            v = float(val)
        except Exception:
            return str(val)
        if 0.0 <= v <= 1.0:
            v *= 100.0
        return f"{v:.2f}"

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
                f.write(f"| {benchmark} | mAP | {_format_map_score(result['mAP'])} |\n")
            elif "accuracy" in result:
                f.write(f"| {benchmark} | Accuracy | {result['accuracy']:.2f}% |\n")
            else:
                f.write(f"| {benchmark} | N/A | N/A |\n")

        f.write("\n## Detailed Results\n\n")

        # Write detailed results for each benchmark
        for benchmark, result in results.items():
            f.write(f"### {benchmark}\n\n")

            if "error" in result:
                f.write(f"**Error**: {result['error']}\n\n")
            elif "mAP" in result:
                f.write(f"**mAP**: {_format_map_score(result['mAP'])}\n\n")
                if "raw" in result and isinstance(result["raw"], dict):
                    f.write("**Per-dataset mAP**:\n\n")
                    for ds, metrics in result["raw"].items():
                        if ds != "Average" and isinstance(metrics, dict):
                            f.write(f"- {ds}: {_format_map_score(metrics.get('mAP', 0.0))}\n")
                    f.write("\n")
            elif "accuracy" in result:
                f.write(f"**Accuracy**: {result['accuracy']:.2f}%\n\n")
                if "raw" in result and isinstance(result["raw"], dict):
                    if "accuracy_by_split" in result["raw"]:
                        f.write("**Accuracy by Split**:\n\n")
                        for split, acc in result["raw"]["accuracy_by_split"].items():
                            try:
                                acc_val = float(acc)
                                if 0.0 <= acc_val <= 1.0:
                                    acc_val *= 100.0
                                f.write(f"- {split}: {acc_val:.2f}%\n")
                            except Exception:
                                f.write(f"- {split}: {acc}\n")
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
            f.write(f"**Average Accuracy (MathVision, MMMU, RealWorldQA)**: {avg_acc:.2f}%\n\n")

        map_scores = [
            r["mAP"] for r in results.values()
            if "mAP" in r and "error" not in r
        ]
        if map_scores:
            f.write(f"**Average mAP (ODinW-13)**: {_format_map_score(map_scores[0])}\n\n")

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
            print(f"| {benchmark} | mAP | {_format_map_score(result['mAP'])} |")
        elif "accuracy" in result:
            print(f"| {benchmark} | Accuracy | {result['accuracy']:.2f}% |")
        else:
            print(f"| {benchmark} | N/A | N/A |")
    print("\n" + "="*80 + "\n")

    return summary_file


def start_vllm_server(
    model_path: str,
    lora_path: str,
    gpus: List[int],
    gpu_memory_utilization: float,
    port: int = 8016,  # default port, will auto-find if taken
    logger: "BenchmarkLogger" = None
) -> str:
    """
    Start vLLM server as a background process.

    Args:
        model_path: Path to the model
        lora_path: Path to LoRA adapter
        gpus: List of GPU IDs to use
        gpu_memory_utilization: GPU memory utilization
        port: Server port
        logger: Optional logger

    Returns:
        Tuple of (Server URL, Process) or (None, None) on failure
    """
    # Find free port
    def find_free_port(start_port):
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(('', port))
                    # Also verify we can connect to it
                    s.listen(1)
                    return port
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    continue
                raise
        return start_port

    # Always find a free port to avoid conflicts
    port = find_free_port(port)
    print(f"Using port: {port}")

    server_url = f"http://localhost:{port}"
    tensor_parallel_size = len(gpus)

    if logger:
        logger.log_section("STARTING vLLM SERVER")
        logger.log(f"Model: {model_path}")
        logger.log(f"LoRA: {lora_path}")
        logger.log(f"GPUs: {gpus}")
        logger.log(f"Tensor parallel: {tensor_parallel_size}")
        logger.log(f"GPU memory util: {gpu_memory_utilization}")
        logger.log(f"Port: {port}")

    print(f"\n{'='*80}")
    print(f"Starting vLLM server...")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"LoRA: {lora_path}")
    print(f"GPUs: {gpus}")
    print(f"Tensor parallel: {tensor_parallel_size}")
    print(f"GPU memory util: {gpu_memory_utilization}")
    print(f"Port: {port}")
    print(f"{'='*80}\n")

    # Build command
    script_path = Path(__file__).parent.parent / "scripts" / "vllm_server.py"

    # Resolve paths to absolute to avoid working directory issues
    if model_path and not os.path.isabs(model_path):
        model_path = str(Path(__file__).parent.parent.parent / model_path)
    if lora_path and not os.path.isabs(lora_path):
        lora_path = str(Path(__file__).parent.parent.parent / lora_path)

    cmd = [
        sys.executable,
        str(script_path),
        "--model-path", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    if lora_path:
        cmd.extend(["--lora-path", lora_path])

    # Set environment - CRITICAL: pass GPU IDs to server
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["VLLM_THINKING_MODE_ENABLED"] = "1"
    env["VLLM_THINKING_AUTO_PATCH"] = "1"
    if lora_path:
        env["VLLM_LORA_CHECKPOINT_PATH"] = lora_path

    # Start server process - stream output to see progress.
    # Use a new process group/session so we can reliably terminate the whole tree.
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(Path(__file__).parent.parent.parent),
        start_new_session=True,
    )

    print(f"Server process started (PID: {process.pid})")

    # Stream server output for visibility (cap to avoid unbounded memory growth)
    output_lines = deque(maxlen=400)
    def stream_output():
        for line in process.stdout:
            print(f"  [server] {line.rstrip()}")
            output_lines.append(line)
    stream_thread = threading.Thread(target=stream_output, daemon=True)
    stream_thread.start()

    print(f"Waiting for server to be ready...")

    # Wait for server to be ready
    max_wait = 600  # 10 minutes (vLLM loading can take time)
    start_wait = time.time()
    server_ready = False
    wait_count = 0

    while time.time() - start_wait < max_wait:
        wait_count += 1
        if wait_count % 6 == 0:  # Print every ~30 seconds
            elapsed = time.time() - start_wait
            print(f"Still waiting for server... ({elapsed:.0f}s elapsed)")
        try:
            resp = requests.get(f"{server_url}/health", timeout=5)
            if resp.status_code == 200:
                # Verify it's our vLLM server (check for vLLM-specific response)
                info = resp.json()
                if "status" in info and info.get("status") == "healthy":
                    server_ready = True
                    print(f"✓ Server ready at {server_url}")
                    if logger:
                        logger.log(f"Server ready at {server_url}")
                    break
                else:
                    # Port is occupied by another service!
                    print(f"✗ Port {port} is used by another service: {info}")
                    process.kill()
                    return None, None
        except Exception:
            pass
        time.sleep(5)

    if not server_ready:
        print(f"✗ Server failed to start within {max_wait}s")
        process.kill()
        return None, None

    return server_url, process


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

    # Server mode arguments
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Use vLLM server at this URL. If not provided and --start-server is set, "
             "will start server automatically. Example: http://localhost:8016"
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start vLLM server automatically before running benchmarks"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8016,
        help="Port for vLLM server (default: 8016)"
    )
    parser.add_argument(
        "--server-concurrency",
        type=int,
        default=16,
        help="Number of concurrent in-flight requests to the vLLM server (default: 16)"
    )

    args = parser.parse_args()

    # Derive enable_lora from lora_path (if lora_path is provided, use LoRA)
    args.enable_lora = bool(args.lora_path)

    # Ensure dataset root is set for dataset utilities (some require LMUData).
    os.environ.setdefault("LMUData", str(Path(__file__).parent / "data"))

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

    # Initialize logger
    with BenchmarkLogger(run_dir) as logger:
        # Track server we start (if any) so we can always clean it up, even on early returns.
        server_process = None
        # Log initial configuration
        logger.log_section("BENCHMARK RUN START")

        # Log system info
        logger.log_section("System Information")
        logger.log(f"Hostname: {platform.node()}")
        logger.log(f"Platform: {platform.platform()}")
        logger.log(f"Python: {platform.python_version()}")

        # Log GPU status
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            logger.log("GPU Status:")
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        logger.log(f"  GPU {parts[0]}: {parts[1]} | Total: {parts[2]} | Free: {parts[3]} | Used: {parts[4]} | Util: {parts[5]}")
        except Exception as e:
            logger.log(f"Could not query GPU status: {e}")

        # Log relevant environment variables
        env_vars = [
            "CUDA_VISIBLE_DEVICES",
            "VLLM_THINKING_MODE_ENABLED",
            "VLLM_THINKING_AUTO_PATCH",
            "VLLM_LORA_CHECKPOINT_PATH",
            "JUDGE_SERVER_URL",
            "TRANSFORMERS_CACHE",
        ]
        logger.log_dict("Environment Variables", {
            k: os.environ.get(k, "(not set)") for k in env_vars
        })

        # Log main configuration
        logger.log_dict("Configuration", {
            "num_samples": args.num_samples,
            "gpus": str(gpus),
            "model_path": args.model_path,
            "enable_lora": args.enable_lora,
            "lora_path": args.lora_path or "None",
            "lora_name": args.lora_name,
            "max_lora_rank": args.max_lora_rank,
            "benchmarks": ",".join(benchmarks),
            "skip_infer": args.skip_infer,
            "skip_eval": args.skip_eval,
        })

        try:
            # Results tracking
            all_results = {}
            inference_files = {}

            # Run inference
            if not args.skip_infer:
                logger.log_section("PHASE 1: INFERENCE")

                # Start server if requested
                if args.start_server:
                    server_url, server_process = start_vllm_server(
                        model_path=args.model_path,
                        lora_path=args.lora_path,
                        gpus=gpus,
                        gpu_memory_utilization=0.85,
                        port=args.port,
                        logger=logger
                    )
                    if not server_url:
                        logger.log("Error: Failed to start vLLM server")
                        return 1
                    args.server_url = server_url

                # Use server mode if server-url is provided
                if args.server_url:
                    inference_files = run_server_inference(
                        benchmarks=benchmarks,
                        run_dir=run_dir,
                        num_samples=args.num_samples,
                        server_url=args.server_url,
                        gpus=gpus,
                        concurrency=args.server_concurrency,
                        logger=logger
                    )
                    if not inference_files:
                        logger.log("Error: Server inference failed or produced no output files")
                        return 1
                # Use unified inference if LoRA is enabled (single model load, faster)
                elif args.enable_lora:
                    inference_files = run_unified_inference(
                        benchmarks=benchmarks,
                        run_dir=run_dir,
                        num_samples=args.num_samples,
                        gpus=gpus,
                        model_path=args.model_path,
                        lora_path=args.lora_path,
                        lora_name=args.lora_name,
                        max_lora_rank=args.max_lora_rank,
                        enable_lora=args.enable_lora,
                        logger=logger
                    )
                    if not inference_files:
                        logger.log("Error: Unified inference failed or produced no output files")
                        return 1
                else:
                    # Use individual benchmark scripts (original behavior)
                    for benchmark in benchmarks:
                        success, output_file = run_inference(
                            benchmark=benchmark,
                            run_dir=run_dir,
                            num_samples=args.num_samples,
                            gpus=gpus,
                            model_path=args.model_path,
                            server_url=args.server_url,
                            logger=logger
                        )
                        if success:
                            inference_files[benchmark] = output_file
                            logger.log(f"Inference completed: {benchmark} -> {output_file}")
                        else:
                            logger.log(f"Warning: {benchmark} inference failed, skipping...")
            elif args.skip_infer and not args.skip_eval:
                # Try to find existing inference files in run_dir
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
                        logger.log(f"Warning: Could not find inference file for {benchmark}")

            # Run evaluation
            if not args.skip_eval:
                logger.log_section("PHASE 2: EVALUATION")

                for benchmark, input_file in inference_files.items():
                    success, result_file = run_evaluation(
                        benchmark=benchmark,
                        input_file=input_file,
                        run_dir=run_dir,
                        num_samples=args.num_samples,
                        logger=logger
                    )
                    if success:
                        parsed = parse_benchmark_results(benchmark, result_file)
                        all_results[benchmark] = parsed
                        logger.log(f"Evaluation completed: {benchmark} -> {parsed}")
                    else:
                        all_results[benchmark] = {"error": "Evaluation failed"}
                        logger.log(f"Evaluation failed: {benchmark}")

            # Generate summary
            if all_results:
                generate_summary(run_dir, all_results)
                logger.log_section("BENCHMARK RUN COMPLETE")
                logger.log(f"Results saved to: {run_dir}")
        finally:
            # Cleanup: kill the server process if we started it (even on early return / exceptions)
            if server_process is not None:
                print(f"\nStopping vLLM server (PID: {server_process.pid})...")
                logger.log(f"Stopping vLLM server (PID: {server_process.pid})")
                try:
                    if os.name != "nt":
                        os.killpg(server_process.pid, signal.SIGTERM)
                    else:
                        server_process.terminate()
                    server_process.wait(timeout=30)
                except Exception as e:
                    print(f"Warning: Error stopping server: {e}")
                    try:
                        if os.name != "nt":
                            os.killpg(server_process.pid, signal.SIGKILL)
                        else:
                            server_process.kill()
                    except Exception:
                        pass
                print("✓ Server stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
