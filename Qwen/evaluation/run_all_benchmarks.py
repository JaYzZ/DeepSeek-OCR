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
import pandas as pd

# Add repo/evaluation directories to path for imports
_EVAL_DIR = Path(__file__).parent
_REPO_ROOT = _EVAL_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EVAL_DIR))
from config import get_data_path, QWEN3_VL_2B_THINKING
from utils import select_compatible_tensor_parallel_gpus
from Qwen.scripts.vllm_utils import (
    apply_runtime_env_for_thinking,
    cleanup_vllm_engine_processes,
    normalize_media_path,
    resolve_lora_artifacts,
)

LOCAL_JUDGE_DEFAULT_MODEL = "/share/project/xiyan/huggingface/Qwen/Qwen2.5-VL-7B-Instruct"
EXPLORE_TEMPERATURE = 0.7
EXPLORE_MAX_TOKENS = 8192
EXPLORE_N = 8


def _get_runtime_yaml_value(key: str, default):
    cfg_path = os.environ.get(
        "QWEN3VL_RUNTIME_ENV_CONFIG",
        str(_REPO_ROOT / "Qwen/configs/qwen3vl_runtime_env.yaml"),
    )
    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get(key, default)
    except Exception:
        return default


class BenchmarkLogger:
    """Logger that writes to both console and benchmark.log file."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.log_file = os.path.join(run_dir, "benchmark.log")
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


def get_free_gpus(num_gpus: int = 8, min_free_mb: int = 10000) -> List[int]:
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
        print("Falling back to GPUs 0-7")
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
            print("No suitable GPUs found, falling back to 0-7")
            return list(range(min(num_gpus, len(gpu_info))))

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


def normalize_run_path(path: str | Path) -> str:
    """Resolve run paths relative to repo root so subprocess cwd changes do not break them."""
    path = Path(path)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return str(path)


_TOKENIZER_CACHE: dict[str, object] = {}


def _load_counting_tokenizer(model_path: str):
    model_path = normalize_run_path(model_path)
    if model_path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_path]
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "encode"):
        tokenizer = processor
    _TOKENIZER_CACHE[model_path] = tokenizer
    return tokenizer


def _count_text_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer.encode(str(text), add_special_tokens=False))
    except Exception:
        return 0


def _split_generation_sections(raw_text: str, answer_text: str) -> tuple[str, str]:
    raw = str(raw_text or "")
    answer = str(answer_text or "").strip()
    if "</think>" in raw:
        thinking_part, answer_part = raw.rsplit("</think>", 1)
        return thinking_part + "</think>", answer_part.strip()
    if answer and raw.endswith(answer):
        return raw[: -len(answer)], answer
    return "", answer or raw


def _get_inference_file(run_dir: str, benchmark: str) -> str | None:
    candidates = {
        "MMMU": ["mmmu_inference.jsonl"],
        "MathVision": ["mathvision_inference.jsonl"],
        "RealWorldQA": ["realworldqa_inference.jsonl"],
        "ODinW-13": ["odinw_inference.jsonl"],
    }.get(benchmark, [])
    for name in candidates:
        path = Path(run_dir) / name
        if path.exists():
            return str(path)
    return None


def collect_benchmark_token_stats(run_dir: str, benchmark: str, model_path: str) -> Dict:
    inference_file = _get_inference_file(run_dir, benchmark)
    if not inference_file or not os.path.exists(inference_file):
        return {}

    tokenizer = _load_counting_tokenizer(model_path)
    sample_count = 0
    total_tokens = 0
    thinking_tokens = 0
    answer_tokens = 0

    with open(inference_file, "r") as f:
        for line in f:
            row = json.loads(line)
            result = row.get("result", {}) or {}
            raw_text = result.get("gen_raw", "")
            answer_text = result.get("gen", "")
            thinking_text, final_answer_text = _split_generation_sections(raw_text, answer_text)

            total_tokens += _count_text_tokens(tokenizer, raw_text)
            thinking_tokens += _count_text_tokens(tokenizer, thinking_text)
            answer_tokens += _count_text_tokens(tokenizer, final_answer_text)
            sample_count += 1

    if sample_count == 0:
        return {}

    return {
        "samples": sample_count,
        "avg_generated_tokens": total_tokens / sample_count,
        "avg_thinking_tokens": thinking_tokens / sample_count,
        "avg_answer_tokens": answer_tokens / sample_count,
    }


def find_available_port(start_port: int, search_span: int = 100) -> int:
    for port in range(start_port, start_port + search_span):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", port))
                sock.listen(1)
                return port
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                continue
            raise
    return start_port


def stop_managed_process(
    process: subprocess.Popen | None,
    label: str,
    logger: "BenchmarkLogger" = None,
) -> None:
    if process is None:
        return

    print(f"\nStopping {label} (PID: {process.pid})...")
    if logger:
        logger.log(f"Stopping {label} (PID: {process.pid})")
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=30)
    except Exception as exc:
        warn_msg = f"Warning: Error stopping {label}: {exc}"
        print(warn_msg)
        if logger:
            logger.log(warn_msg)
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except Exception as kill_err:
            warn_msg = f"Warning: Failed to force-kill {label} process {process.pid}: {kill_err}"
            print(warn_msg)
            if logger:
                logger.log(warn_msg)
    print(f"✓ {label} stopped")


def _stream_managed_process_output(
    process: subprocess.Popen,
    prefix: str,
    output_lines: deque,
    logger: "BenchmarkLogger" = None,
) -> threading.Thread:
    """Mirror managed child output to console, benchmark.log, and an in-memory tail."""

    def _stream() -> None:
        for line in process.stdout:
            rendered = line.rstrip()
            print(f"  [{prefix}] {rendered}")
            output_lines.append(rendered)
            if logger:
                logger.log(f"[{prefix}] {rendered}", to_console=False)

    thread = threading.Thread(target=_stream, daemon=True)
    thread.start()
    return thread


def _log_managed_process_failure(
    process: subprocess.Popen,
    label: str,
    output_lines: deque,
    logger: "BenchmarkLogger" = None,
) -> None:
    """Persist the child exit code and recent output when startup fails."""
    returncode = process.poll()
    msg = f"{label} exited before readiness check completed"
    if returncode is not None:
        msg += f" (returncode={returncode})"
    print(f"✗ {msg}")
    if logger:
        logger.log(f"ERROR: {msg}")
        if output_lines:
            logger.log(f"Recent {label} output tail:", to_console=False)
            for line in output_lines:
                logger.log(f"  [{label}] {line}", to_console=False)


def requires_judge(benchmark: str) -> bool:
    return benchmark != "ODinW-13"


def run_unified_inference(
    benchmarks: List[str],
    run_dir: str,
    num_samples: int,
    gpus: List[int],
    model_path: str,
    lora_path: str = None,
    lora_name: str = "default",
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
        enable_lora: Whether LoRA is enabled
        logger: Optional BenchmarkLogger instance

    Returns:
        Dictionary mapping benchmark names to output files
    """
    adapter_meta = resolve_lora_artifacts(model_path, lora_path)
    resolved_model_path = adapter_meta["model_path"] or model_path
    resolved_lora_rank = adapter_meta["lora_rank"] if adapter_meta["lora_rank"] is not None else 64
    script_path = Path(__file__).parent / "run_all_inference.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--model-path", resolved_model_path,
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

    # Set LoRA checkpoint path for VAE loading
    if lora_path:
        env["VLLM_LORA_CHECKPOINT_PATH"] = lora_path

    # Log detailed information
    if logger:
        logger.log_section("RUNNING UNIFIED INFERENCE")
        logger.log(f"Requested model: {model_path}")
        logger.log(f"Resolved model: {resolved_model_path}")
        logger.log(f"LoRA: {lora_path if lora_path else 'Disabled'}")
        if lora_path:
            logger.log(f"Resolved LoRA rank: {resolved_lora_rank}")
        logger.log(f"Benchmarks: {', '.join(benchmarks)}")
        logger.log(f"Num samples: {num_samples}")
        logger.log(f"GPUs: {gpus}")
        logger.log(f"Tensor parallel size: {len(gpus)}")
        logger.log(f"Output dir: {run_dir}")
        logger.log_command(" ".join(cmd))
        logger.log_dict("Environment", {
            "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
            "VLLM_THINKING": env["VLLM_THINKING"],
        })

    print(f"\n{'='*80}")
    print(f"Running HYBRID inference for {len(benchmarks)} benchmarks")
    print(f"Auto-detects model type: vLLM (official) or HF Transformers (Linear variant)")
    print(f"Model: {resolved_model_path}")
    print(f"LoRA: {lora_path if lora_path else 'Disabled'}")
    if lora_path:
        print(f"LoRA rank: {resolved_lora_rank}")
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
    max_tokens: int,
    concurrency: int = 1,
    logger: "BenchmarkLogger" = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n: int = 1,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    output_suffix: str = "",
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
        logger.log(f"Max tokens: {max_tokens}")
        logger.log(f"Concurrency: {concurrency}")
        logger.log(f"Temperature: {temperature}")
        logger.log(f"Top-p: {top_p}")
        logger.log(f"Samples per prompt: {n}")

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

    try:
        # Many dataset utilities expect LMUData to be set. Default to evaluation's data dir.
        os.environ.setdefault("LMUData", str(Path(__file__).parent / "data"))

        # Import through the full package path so relative imports inside benchmark modules work.
        from Qwen.evaluation.MathVision.dataset_utils import (
            load_dataset as load_mathv_dataset,
            dump_image as mathv_dump_image,
        )
        from Qwen.evaluation.MathVision.run_mathv import build_mathv_prompt
        from Qwen.evaluation.MathVision.eval_utils import post_check as mathvision_post_check
        from Qwen.evaluation.mmmu.dataset_utils import (
            load_dataset as load_mmmu_dataset,
            dump_image as mmmu_dump_image,
        )
        from Qwen.evaluation.mmmu.run_mmmu import build_mmmu_prompt
        from Qwen.evaluation.RealWorldQA.dataset_utils import (
            load_dataset as load_realworldqa_dataset,
            dump_image as realworldqa_dump_image,
        )
        from Qwen.evaluation.RealWorldQA.run_realworldqa import build_realworldqa_prompt

        # Load processor
        model_path = server_info.get("model", QWEN3_VL_2B_THINKING)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed during server inference setup: {e}")
        traceback.print_exc()
        if logger:
            logger.log(f"Error: server inference setup failed: {e}")
            logger.log(traceback.format_exc(), to_console=False)
        return {}

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

        output_file = os.path.join(run_dir, f"{benchmark.lower()}{output_suffix}_inference.jsonl")
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
        total_rows = len(rows)
        progress_step = max(1, min(10, total_rows // 10 if total_rows >= 10 else 1))

        def log_benchmark_progress(phase: str, completed: int, total: int, force: bool = False):
            if not logger or total <= 0:
                return
            if not force and completed % progress_step != 0 and completed != total:
                return
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = total - completed
            eta_seconds = remaining / rate if rate > 0 else 0.0
            logger.log(
                f"{phase} progress: {completed}/{total} "
                f"({(completed / total) * 100:.1f}%) | "
                f"elapsed={elapsed:.1f}s | rate={rate:.2f} samples/s | eta={eta_seconds:.1f}s"
            )

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
                                    img = item.get("image")
                                    normalized_img = normalize_media_path(img)
                                    if normalized_img != img:
                                        item = dict(item)
                                        item["image"] = normalized_img
                                    processed_content.append(item)
                                elif item.get("type") == "text":
                                    processed_content.append(item)
                        api_messages.append({"role": msg.get("role", "user"), "content": processed_content})
                    else:
                        api_messages.append({"role": msg.get("role", "user"), "content": content})

                payload = {
                    "messages": api_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "n": n,
                    "presence_penalty": presence_penalty,
                    "repetition_penalty": repetition_penalty,
                }
                request_tasks.append((idx, row, messages, payload))
                log_benchmark_progress("Prompt build", len(request_tasks), total_rows)
            except Exception as e:
                print(f"Error building prompt for sample {idx}: {e}")
                continue

        log_benchmark_progress("Prompt build", len(request_tasks), total_rows, force=True)

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
                    choices = result_data.get("choices", [])
                    if not choices:
                        raise RuntimeError("No choices returned from server")

                    candidate_results = []
                    for choice in choices:
                        response_raw = choice["message"]["content"]
                        response_final = (
                            response_raw.split("</think>")[-1].strip()
                            if "</think>" in response_raw
                            else response_raw
                        )
                        candidate_results.append(
                            {
                                "gen": response_final,
                                "gen_raw": response_raw,
                                "finish_reason": choice.get("finish_reason", "stop"),
                            }
                        )

                    primary = candidate_results[0]
                    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
                    out = {
                        "question_id": idx,
                        "annotation": row_dict,
                        "task": benchmark,
                        "result": {"gen": primary["gen"], "gen_raw": primary["gen_raw"]},
                        "messages": messages,
                    }
                    if len(candidate_results) > 1:
                        out["candidates"] = candidate_results
                        out["candidate_stats"] = {
                            "num_candidates": len(candidate_results),
                            "num_unique_final_answers": len({c["gen"] for c in candidate_results}),
                            "num_unique_raw_answers": len({c["gen_raw"] for c in candidate_results}),
                        }
                    if "usage" in result_data:
                        out["usage"] = result_data["usage"]
                    return idx, out, None
                except Exception as e:
                    last_err = str(e)
                    # Small jittered backoff to avoid thundering herd on transient errors.
                    time.sleep((2 ** attempt) + random.random())
            return idx, None, last_err

        # Execute requests concurrently to keep vLLM busy.
        effective_concurrency = max(1, int(concurrency or 1))
        if effective_concurrency == 1:
            completed_requests = 0
            for task in tqdm(request_tasks, total=len(request_tasks), desc=f"{benchmark} infer"):
                idx, out, err = call_server(task)
                if out is not None:
                    results_by_idx[idx] = out
                else:
                    print(f"Error for sample {idx}: {err}")
                completed_requests += 1
                log_benchmark_progress("Inference", completed_requests, len(request_tasks))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_concurrency) as ex:
                futures = [ex.submit(call_server, task) for task in request_tasks]
                completed_requests = 0
                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{benchmark} infer"):
                    idx, out, err = fut.result()
                    if out is not None:
                        results_by_idx[idx] = out
                    else:
                        print(f"Error for sample {idx}: {err}")
                    completed_requests += 1
                    log_benchmark_progress("Inference", completed_requests, len(request_tasks))

        log_benchmark_progress("Inference", len(request_tasks), len(request_tasks), force=True)

        # Save results
        with open(output_file, 'w') as f:
            for idx in sorted(results_by_idx.keys()):
                f.write(json.dumps(results_by_idx[idx]) + '\n')

        if benchmark == "MathVision" and n > 1:
            completed_rows = [results_by_idx[idx] for idx in sorted(results_by_idx.keys())]
            candidate_stats = [row.get("candidate_stats", {}) for row in completed_rows]
            stats_payload = {
                "benchmark": benchmark,
                "num_samples": len(completed_rows),
                "samples_with_candidates": sum(1 for row in completed_rows if "candidates" in row),
                "avg_unique_final_answers": (
                    sum(stats.get("num_unique_final_answers", 0) for stats in candidate_stats) / len(candidate_stats)
                    if candidate_stats else 0.0
                ),
                "avg_unique_raw_answers": (
                    sum(stats.get("num_unique_raw_answers", 0) for stats in candidate_stats) / len(candidate_stats)
                    if candidate_stats else 0.0
                ),
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": n,
                "status": "pending_judge_eval",
            }
            stats_file = os.path.join(run_dir, f"{benchmark.lower()}{output_suffix}_stats.json")
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats_payload, f, indent=2)
            if logger:
                logger.log(f"Exploration stats saved: {stats_file}")
                logger.log(f"MathVision exploration summary: {stats_payload}")

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
    output_file = normalize_run_path(Path(run_dir) / config["output"])

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
    judge_url: str = None,
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
            "limit_at_eval": False,  # Already limited during inference
            "eval_model": "gpt-4o",
        },
        "MMMU": {
            "script": "mmmu/run_mmmu.py",
            "dataset": "MMMU_DEV_VAL",
            "output": "mmmu_eval_result.csv",
            "result_key": "mmmu_eval_result_acc.json",
            "limit_at_eval": True,   # Need to limit during evaluation
            "eval_model": "gpt-3.5-turbo-0125",
        },
        "RealWorldQA": {
            "script": "RealWorldQA/run_realworldqa.py",
            "dataset": "RealWorldQA",
            "output": "realworldqa_eval_result.csv",
            "result_key": "realworldqa_eval_result_acc.json",
            "limit_at_eval": True,   # Need to limit during evaluation
            "eval_model": "gpt-4o",
        },
        "ODinW-13": {
            "script": "ODinW-13/run_odinw.py",
            "dataset": None,
            "output": "odinw_eval_result.json",
            "result_key": None,
            "limit_at_eval": False,  # Already limited during inference
            "eval_model": None,
        }
    }

    if benchmark not in benchmark_configs:
        print(f"Error: Unknown benchmark {benchmark}")
        return False, ""

    config = benchmark_configs[benchmark]
    script_path = Path(__file__).parent / config["script"]
    input_file = normalize_run_path(input_file)
    output_file = normalize_run_path(Path(run_dir) / config["output"])

    # Always use the local judge server selected by the driver for judge-based benchmarks.
    if benchmark != "ODinW-13" and not judge_url:
        judge_url = os.environ.get("JUDGE_SERVER_URL")
    if benchmark != "ODinW-13" and not judge_url:
        print(f"Error: Local judge URL not configured for {benchmark}")
        return False, ""

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
        if config.get("eval_model"):
            cmd.extend(["--eval-model", config["eval_model"]])

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
            result_file = normalize_run_path(Path(run_dir) / config["result_key"])
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


def _normalize_benchmark_name(benchmark: str) -> str:
    normalized = str(benchmark).strip()
    alias_map = {
        "realworldqa": "RealWorldQA",
        "real_world_qa": "RealWorldQA",
        "real-world-qa": "RealWorldQA",
        "mathvision": "MathVision",
        "mmmu": "MMMU",
        "odinw-13": "ODinW-13",
        "odinw13": "ODinW-13",
    }
    return alias_map.get(normalized.lower(), normalized)


def evaluate_multiple_choice_exploration(
    benchmark: str,
    inference_file: str,
    stats_file: str,
    judge_url: str | None,
    logger: "BenchmarkLogger" = None,
) -> Dict:
    benchmark = _normalize_benchmark_name(benchmark)
    if benchmark == "MMMU":
        from Qwen.evaluation.mmmu.eval_utils import build_judge as build_mc_judge, eval_single_sample as eval_mc_single_sample
    elif benchmark == "RealWorldQA":
        from Qwen.evaluation.RealWorldQA.eval_utils import build_judge as build_mc_judge, eval_single_sample as eval_mc_single_sample
    else:
        payload = {"benchmark": benchmark, "error": f"Unsupported multiple-choice exploration benchmark: {benchmark}"}
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    rows = []
    with open(inference_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        payload = {"benchmark": benchmark, "error": "No exploration inference rows found"}
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    judge_model = None
    scoring_logic = f"{benchmark} eval_single_sample candidate-wise (rule extraction only)"
    if judge_url:
        judge_model = build_mc_judge(
            model="gpt-4o",
            api_type="custom",
            api_url=judge_url,
            api_key=os.environ.get("LOCAL_API_KEY", "EMPTY"),
        )
        scoring_logic = f"{benchmark} eval_single_sample candidate-wise"

    sample_rows = []
    evaluated_rows = []
    for row in rows:
        annotation = row.get("annotation", {}) or {}
        candidates = row.get("candidates") or [row.get("result", {})]
        candidate_hits = []
        candidate_evaluations = []

        for candidate in candidates:
            eval_row = dict(annotation)
            eval_row["prediction"] = candidate.get("gen", "")
            if benchmark == "MMMU":
                eval_row["GT"] = annotation.get("answer")
            elif benchmark == "RealWorldQA":
                eval_row["answer"] = annotation.get("answer")
            eval_result = eval_mc_single_sample((judge_model, eval_row))
            hit = bool(eval_result.get("hit", 0))
            candidate_hits.append(hit)
            candidate_evaluations.append(
                {
                    "prediction": candidate.get("gen", ""),
                    "prediction_raw": candidate.get("gen_raw", ""),
                    "hit": hit,
                    "extracted_answer": eval_result.get("extracted_answer"),
                    "extraction_method": eval_result.get("extraction_method"),
                    "extraction_success": eval_result.get("extraction_success"),
                    "extraction_log": eval_result.get("extraction_log"),
                }
            )

        sample_rows.append(
            {
                "question_id": row.get("question_id"),
                "top1_hit": bool(candidate_hits[0]) if candidate_hits else False,
                "pass_at_k": any(candidate_hits),
                "num_correct": sum(1 for hit in candidate_hits if hit),
                "num_candidates": len(candidate_hits),
            }
        )
        enriched = dict(row)
        enriched["candidate_evaluations"] = candidate_evaluations
        evaluated_rows.append(enriched)

    df = pd.DataFrame(sample_rows)
    k = int(df["num_candidates"].max()) if len(df) else 0
    payload = {
        "benchmark": benchmark,
        "num_samples": int(len(df)),
        "samples_with_candidates": int((df["num_candidates"] > 0).sum()) if len(df) else 0,
        "samples_with_any_correct": int(df["pass_at_k"].sum()) if len(df) else 0,
        "top1_accuracy": float(df["top1_hit"].mean()) if len(df) else 0.0,
        f"pass@{k}": float(df["pass_at_k"].mean()) if len(df) else 0.0,
        f"mean@{k}": float((df["num_correct"] / df["num_candidates"].replace(0, 1)).mean()) if len(df) else 0.0,
        "avg_correct_candidates": float(df["num_correct"].mean()) if len(df) else 0.0,
        "avg_candidates": float(df["num_candidates"].mean()) if len(df) else 0.0,
        "judge_url": judge_url or "",
        "scoring_logic": scoring_logic,
    }

    evaluated_file = inference_file.replace("_inference.jsonl", "_evaluated.jsonl")
    with open(evaluated_file, "w", encoding="utf-8") as f:
        for row in evaluated_rows:
            f.write(json.dumps(row) + "\n")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if logger:
        logger.log(f"Exploration evaluation saved: {evaluated_file}")
        logger.log(f"Exploration stats saved: {stats_file}")
        logger.log(f"{benchmark} exploration evaluated summary: {payload}")

    return payload


def evaluate_mathvision_exploration(
    inference_file: str,
    stats_file: str,
    judge_url: str | None,
    logger: "BenchmarkLogger" = None,
) -> Dict:
    """Evaluate MathVision explore candidates using the same extraction/scoring path as benchmark eval."""
    from Qwen.evaluation.MathVision.eval_utils import (
        build_judge as build_mathvision_judge,
        eval_single_sample as mathvision_eval_single_sample,
        post_check as mathvision_post_check,
    )

    rows = []
    with open(inference_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        payload = {"benchmark": "MathVision", "error": "No exploration inference rows found"}
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    if not judge_url:
        payload = {"benchmark": "MathVision", "error": "Judge URL required for exploration evaluation"}
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    judge_model = build_mathvision_judge(
        model="gpt-4o",
        api_type="custom",
        api_url=judge_url,
        api_key=os.environ.get("LOCAL_API_KEY", "EMPTY"),
    )
    scoring_logic = "MathVision eval_single_sample + post_check"

    sample_rows = []
    evaluated_rows = []
    for row in rows:
        annotation = row.get("annotation", {}) or {}
        candidates = row.get("candidates") or [row.get("result", {})]
        candidate_hits = []
        candidate_evaluations = []

        for candidate in candidates:
            eval_row = dict(annotation)
            eval_row["prediction"] = candidate.get("gen", "")
            eval_result = mathvision_eval_single_sample((judge_model, eval_row))
            scored_row = dict(eval_row)
            scored_row.update(eval_result)
            hit = bool(mathvision_post_check(scored_row, prefetch=False))
            candidate_hits.append(hit)
            candidate_evaluations.append(
                {
                    "prediction": candidate.get("gen", ""),
                    "prediction_raw": candidate.get("gen_raw", ""),
                    "hit": hit,
                    "res": eval_result.get("res"),
                    "log": eval_result.get("log"),
                    "extract_model": eval_result.get("extract_model"),
                    "extract_flag": eval_result.get("extract_flag"),
                }
            )

        sample_rows.append(
            {
                "question_id": row.get("question_id"),
                "top1_hit": bool(candidate_hits[0]) if candidate_hits else False,
                "pass_at_k": any(candidate_hits),
                "num_correct": sum(1 for hit in candidate_hits if hit),
                "num_candidates": len(candidate_hits),
            }
        )
        enriched = dict(row)
        enriched["candidate_evaluations"] = candidate_evaluations
        evaluated_rows.append(enriched)

    df = pd.DataFrame(sample_rows)
    k = int(df["num_candidates"].max()) if len(df) else 0
    payload = {
        "benchmark": "MathVision",
        "num_samples": int(len(df)),
        "samples_with_candidates": int((df["num_candidates"] > 0).sum()) if len(df) else 0,
        "samples_with_any_correct": int(df["pass_at_k"].sum()) if len(df) else 0,
        "top1_accuracy": float(df["top1_hit"].mean()) if len(df) else 0.0,
        f"pass@{k}": float(df["pass_at_k"].mean()) if len(df) else 0.0,
        f"mean@{k}": float((df["num_correct"] / df["num_candidates"].replace(0, 1)).mean()) if len(df) else 0.0,
        "avg_correct_candidates": float(df["num_correct"].mean()) if len(df) else 0.0,
        "avg_candidates": float(df["num_candidates"].mean()) if len(df) else 0.0,
        "judge_url": judge_url or "",
        "scoring_logic": scoring_logic,
    }

    evaluated_file = inference_file.replace("_inference.jsonl", "_evaluated.jsonl")
    with open(evaluated_file, "w", encoding="utf-8") as f:
        for row in evaluated_rows:
            f.write(json.dumps(row) + "\n")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if logger:
        logger.log(f"Exploration evaluation saved: {evaluated_file}")
        logger.log(f"Exploration stats saved: {stats_file}")
        logger.log(f"MathVision exploration evaluated summary: {payload}")

    return payload


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

    explore_stats = []
    try:
        for name in sorted(os.listdir(run_dir)):
            if not name.endswith("_explore_stats.json"):
                continue
            path = os.path.join(run_dir, name)
            with open(path, "r", encoding="utf-8") as ef:
                explore_stats.append((name, json.load(ef)))
    except Exception:
        pass

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
                token_stats = result.get("token_stats")
                if token_stats:
                    f.write("**Average Generated Tokens per Sample**:\n\n")
                    f.write(f"- total: {token_stats['avg_generated_tokens']:.2f}\n")
                    f.write(f"- thinking: {token_stats['avg_thinking_tokens']:.2f}\n")
                    f.write(f"- answer: {token_stats['avg_answer_tokens']:.2f}\n")
                    f.write(f"- samples: {token_stats['samples']}\n\n")
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

        if explore_stats:
            f.write("## Exploration Results\n\n")
            for name, payload in explore_stats:
                benchmark = payload.get("benchmark", name.replace("_explore_stats.json", ""))
                f.write(f"### {benchmark} Explore\n\n")
                for key, value in payload.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

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
    adapter_meta = resolve_lora_artifacts(model_path, lora_path)
    resolved_model_path = adapter_meta["model_path"] or model_path
    resolved_lora_rank = adapter_meta["lora_rank"] if adapter_meta["lora_rank"] is not None else 64
    # Always find a free port to avoid conflicts
    port = find_available_port(port)
    print(f"Using port: {port}")

    server_url = f"http://localhost:{port}"
    server_gpus, tensor_parallel_size = select_compatible_tensor_parallel_gpus(
        model_path,
        gpus,
        capped=False,
    )
    if not server_gpus:
        server_gpus = list(gpus)

    if logger:
        logger.log_section("STARTING vLLM SERVER")
        logger.log(f"Requested model: {model_path}")
        logger.log(f"Resolved model: {resolved_model_path}")
        logger.log(f"LoRA: {lora_path}")
        if lora_path:
            logger.log(f"Resolved LoRA rank: {resolved_lora_rank}")
        logger.log(f"GPUs: {server_gpus}")
        logger.log(f"Tensor parallel: {tensor_parallel_size}")
        logger.log(f"GPU memory util: {gpu_memory_utilization}")
        logger.log(f"Port: {port}")

    print(f"\n{'='*80}")
    print(f"Starting vLLM server...")
    print(f"{'='*80}")
    print(f"Model: {resolved_model_path}")
    print(f"LoRA: {lora_path}")
    if lora_path:
        print(f"LoRA rank: {resolved_lora_rank}")
    print(f"GPUs: {server_gpus}")
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
        "--model-path", resolved_model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    if lora_path:
        cmd.extend(["--lora-path", lora_path])

    # Set environment - CRITICAL: pass GPU IDs to server
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, server_gpus))
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
    _stream_managed_process_output(process, "server", output_lines, logger)

    print(f"Waiting for server to be ready...")

    # Wait for server to be ready
    max_wait = 600  # 10 minutes (vLLM loading can take time)
    start_wait = time.time()
    server_ready = False
    wait_count = 0

    while time.time() - start_wait < max_wait:
        wait_count += 1
        if process.poll() is not None:
            _log_managed_process_failure(process, "vLLM server", output_lines, logger)
            stop_managed_process(process, "vLLM server", logger)
            return None, None
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
        except requests.RequestException:
            # Expected while server is still booting.
            pass
        except Exception as e:
            msg = f"Warning: Unexpected error while polling server health: {e}"
            print(msg)
            if logger:
                logger.log(msg)
        time.sleep(5)

    if not server_ready:
        print(f"✗ Server failed to start within {max_wait}s")
        if logger and output_lines:
            logger.log("Recent vLLM server output tail before timeout:", to_console=False)
            for line in output_lines:
                logger.log(f"  [vLLM server] {line}", to_console=False)
        process.kill()
        return None, None

    return server_url, process


def start_local_judge_server(
    gpus: List[int],
    port: int = 8600,
    logger: "BenchmarkLogger" = None,
) -> Tuple[str | None, subprocess.Popen | None]:
    model_path = normalize_run_path(
        os.environ.get("JUDGE_MODEL_PATH", LOCAL_JUDGE_DEFAULT_MODEL)
    )
    gpu_memory_utilization = float(os.environ.get("LOCAL_JUDGE_GPU_MEMORY_UTILIZATION", "0.9"))
    max_model_len = int(os.environ.get("LOCAL_JUDGE_MAX_MODEL_LEN", "32768"))
    judge_gpus, tensor_parallel_size = select_compatible_tensor_parallel_gpus(
        model_path,
        gpus,
        capped=True,
    )
    if not judge_gpus:
        judge_gpus = list(gpus)
    port = find_available_port(port)
    judge_url = f"http://127.0.0.1:{port}"
    script_path = Path(__file__).parent / "judge_server.py"

    if logger:
        logger.log_section("STARTING LOCAL JUDGE SERVER")
        logger.log(f"Judge model: {model_path}")
        logger.log(f"GPUs: {judge_gpus}")
        logger.log(f"Tensor parallel: {tensor_parallel_size}")
        logger.log(f"GPU memory util: {gpu_memory_utilization}")
        logger.log(f"Max model len: {max_model_len}")
        logger.log(f"Port: {port}")

    print(f"\n{'='*80}")
    print("Starting local judge server...")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"GPUs: {judge_gpus}")
    print(f"Tensor parallel: {tensor_parallel_size}")
    print(f"GPU memory util: {gpu_memory_utilization}")
    print(f"Max model len: {max_model_len}")
    print(f"Port: {port}")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable,
        str(script_path),
        "--model-path", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, judge_gpus))

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

    print(f"Judge server process started (PID: {process.pid})")

    output_lines = deque(maxlen=400)

    _stream_managed_process_output(process, "judge", output_lines, logger)

    max_wait = 600
    start_wait = time.time()
    while time.time() - start_wait < max_wait:
        if process.poll() is not None:
            _log_managed_process_failure(process, "local judge server", output_lines, logger)
            stop_managed_process(process, "local judge server", logger)
            return None, None
        try:
            resp = requests.get(f"{judge_url}/health", timeout=5)
            if resp.status_code == 200:
                info = resp.json()
                if info.get("status") == "healthy":
                    if logger:
                        logger.log(f"Local judge server ready at {judge_url}")
                    print(f"✓ Local judge server ready at {judge_url}")
                    return judge_url, process
        except requests.RequestException:
            pass
        time.sleep(5)

    print(f"✗ Local judge server failed to start within {max_wait}s")
    if logger and output_lines:
        logger.log("Recent local judge output tail before timeout:", to_console=False)
        for line in output_lines:
            logger.log(f"  [local judge server] {line}", to_console=False)
    stop_managed_process(process, "local judge server", logger)
    return None, None


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
        help="Run directory. Defaults to <lora-path>/bench when --lora-path is set; otherwise creates a timestamped directory under evaluation/results"
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
        default="MathVision,MMMU,RealWorldQA",
        help="Comma-separated list of benchmarks to run (default: MathVision,MMMU,RealWorldQA)"
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
        default=None,
        help="Number of concurrent in-flight requests to the vLLM server (default: 16)"
    )
    parser.add_argument(
        "--server-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for server-mode inference (default: 0.0)"
    )
    parser.add_argument(
        "--server-top-p",
        type=float,
        default=1.0,
        help="Top-p for server-mode inference (default: 1.0)"
    )
    parser.add_argument(
        "--server-n",
        type=int,
        default=1,
        help="Number of completions per prompt for server-mode inference (default: 1)"
    )
    parser.add_argument(
        "--server-presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty for server-mode inference (default: 0.0)"
    )
    parser.add_argument(
        "--server-repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for server-mode inference (default: 1.0)"
    )
    parser.add_argument(
        "--server-max-tokens",
        type=int,
        default=None,
        help="Override max tokens for server-mode inference"
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Run explore-only inference/evaluation on the provided benchmarks"
    )
    args = parser.parse_args()
    apply_runtime_env_for_thinking(repo_root=Path(__file__).resolve().parents[2])
    if args.server_concurrency is None:
        args.server_concurrency = int(_get_runtime_yaml_value("benchmark_server_concurrency", 16))
    if args.skip_eval:
        print("Error: evaluation is required for both benchmark and explore runs")
        return 1

    # Derive enable_lora from lora_path (if lora_path is provided, use LoRA)
    args.enable_lora = bool(args.lora_path)
    adapter_meta = resolve_lora_artifacts(args.model_path, args.lora_path)
    resolved_model_path = adapter_meta["model_path"] or args.model_path
    resolved_lora_rank = adapter_meta["lora_rank"] if adapter_meta["lora_rank"] is not None else 64

    # Ensure dataset root is set for dataset utilities (some require LMUData).
    os.environ.setdefault("LMUData", str(Path(__file__).parent / "data"))

    # Parse benchmarks
    benchmarks = [_normalize_benchmark_name(b) for b in args.benchmarks.split(',') if b.strip()]

    # GPU selection
    if args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(',')]
        print(f"Using manually specified GPUs: {gpus}")
    else:
        gpus = get_free_gpus(num_gpus=8)

    if len(gpus) == 0:
        print("Error: No GPUs available")
        return 1

    # Create or resolve run directory.
    if args.run_dir:
        run_dir = normalize_run_path(args.run_dir)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        print(f"Using specified run directory: {run_dir}")
    else:
        if args.lora_path:
            run_dir = normalize_run_path(Path(args.lora_path) / "bench")
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            print(f"Using default LoRA benchmark directory: {run_dir}")
        else:
            base_results_path = Path(__file__).parent / "results"
            run_dir = normalize_run_path(create_timestamp_dir(str(base_results_path)))

    # Initialize logger
    with BenchmarkLogger(run_dir) as logger:
        # Track server we start (if any) so we can always clean it up, even on early returns.
        server_process = None
        judge_process = None
        judge_url = None
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
            "VLLM_THINKING",
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
            "resolved_model_path": resolved_model_path,
            "enable_lora": args.enable_lora,
            "lora_path": args.lora_path or "None",
            "lora_name": args.lora_name,
            "resolved_lora_rank": resolved_lora_rank,
            "benchmarks": ",".join(benchmarks),
            "skip_infer": args.skip_infer,
            "skip_eval": args.skip_eval,
            "server_concurrency": args.server_concurrency,
            "explore_only": args.explore,
            "explore_temperature": EXPLORE_TEMPERATURE,
            "explore_max_tokens": EXPLORE_MAX_TOKENS,
            "explore_n": EXPLORE_N,
        })

        try:
            # Results tracking
            all_results = {}
            inference_files = {}
            explore_inference_files = {}
            run_benchmark = not args.explore
            run_explore = args.explore

            # Run inference
            if not args.skip_infer:
                logger.log_section("PHASE 1: INFERENCE")

                # Start server if requested
                if args.start_server:
                    server_url, server_process = start_vllm_server(
                        model_path=resolved_model_path,
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

                if not args.server_url:
                    logger.log("Error: server mode is required for benchmark/explore runs")
                    return 1

                if run_benchmark:
                    server_max_tokens = (
                        int(args.server_max_tokens)
                        if args.server_max_tokens is not None
                        else int(
                            os.environ.get(
                                "QWEN3VL_TRANSPARENT_EVAL_MAX_NEW_TOKENS",
                                _get_runtime_yaml_value("eval_max_new_tokens", 8192),
                            )
                        )
                    )
                    inference_files = run_server_inference(
                        benchmarks=benchmarks,
                        run_dir=run_dir,
                        num_samples=args.num_samples,
                        server_url=args.server_url,
                        gpus=gpus,
                        max_tokens=server_max_tokens,
                        concurrency=args.server_concurrency,
                        logger=logger,
                        temperature=args.server_temperature,
                        top_p=args.server_top_p,
                        n=args.server_n,
                        presence_penalty=args.server_presence_penalty,
                        repetition_penalty=args.server_repetition_penalty,
                        output_suffix="",
                    )
                    if not inference_files:
                        logger.log("Error: Server inference failed or produced no output files")
                        return 1

                if run_explore:
                    logger.log_section("PHASE 1B: EXPLORATION")
                    explore_inference_files = run_server_inference(
                        benchmarks=benchmarks,
                        run_dir=run_dir,
                        num_samples=args.num_samples,
                        server_url=args.server_url,
                        gpus=gpus,
                        max_tokens=EXPLORE_MAX_TOKENS,
                        concurrency=args.server_concurrency,
                        logger=logger,
                        temperature=EXPLORE_TEMPERATURE,
                        top_p=1.0,
                        n=EXPLORE_N,
                        presence_penalty=0.0,
                        repetition_penalty=1.0,
                        output_suffix="_explore",
                    )
                    if not explore_inference_files:
                        logger.log("Warning: Exploration mode produced no output files")
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
                if not inference_files:
                    logger.log("Error: No inference files found for evaluation")
                    return 1

            # Run evaluation
            if not args.skip_eval:
                # Inference is complete; free the training GPU set before bringing up the judge.
                if server_process is not None:
                    stop_managed_process(server_process, "vLLM server", logger)
                    cleanup_vllm_engine_processes(logger)
                    server_process = None

                logger.log_section("PHASE 2: EVALUATION")
                eval_inputs = explore_inference_files if args.explore else inference_files
                judge_benchmarks = [benchmark for benchmark in eval_inputs if requires_judge(benchmark)]
                if judge_benchmarks:
                    judge_url, judge_process = start_local_judge_server(
                        gpus=gpus,
                        port=8600,
                        logger=logger,
                    )
                    if not judge_url:
                        logger.log("Error: Failed to start local judge server")
                        return 1
                    os.environ["JUDGE_SERVER_URL"] = judge_url
                    logger.log(f"Using local judge server for final scoring: {judge_url}")

                for benchmark, input_file in eval_inputs.items():
                    success, result_file = run_evaluation(
                        benchmark=benchmark,
                        input_file=input_file,
                        run_dir=run_dir,
                        num_samples=args.num_samples,
                        judge_url=judge_url if requires_judge(benchmark) else None,
                        logger=logger
                    )
                    if success:
                        parsed = parse_benchmark_results(benchmark, result_file)
                        token_stats = collect_benchmark_token_stats(run_dir, benchmark, resolved_model_path)
                        if token_stats:
                            parsed["token_stats"] = token_stats
                        all_results[benchmark] = parsed
                        logger.log(f"Evaluation completed: {benchmark} -> {parsed}")
                    else:
                        all_results[benchmark] = {"error": "Evaluation failed"}
                        logger.log(f"Evaluation failed: {benchmark}")

                if "MathVision" in explore_inference_files:
                    stats_file = os.path.join(run_dir, "mathvision_explore_stats.json")
                    evaluate_mathvision_exploration(
                        inference_file=explore_inference_files["MathVision"],
                        stats_file=stats_file,
                        judge_url=judge_url if requires_judge("MathVision") else None,
                        logger=logger,
                    )
                if "MMMU" in explore_inference_files:
                    stats_file = os.path.join(run_dir, "mmmu_explore_stats.json")
                    evaluate_multiple_choice_exploration(
                        benchmark="MMMU",
                        inference_file=explore_inference_files["MMMU"],
                        stats_file=stats_file,
                        judge_url=judge_url if requires_judge("MMMU") else None,
                        logger=logger,
                    )
                if "RealWorldQA" in explore_inference_files:
                    stats_file = os.path.join(run_dir, "realworldqa_explore_stats.json")
                    evaluate_multiple_choice_exploration(
                        benchmark="RealWorldQA",
                        inference_file=explore_inference_files["RealWorldQA"],
                        stats_file=stats_file,
                        judge_url=judge_url if requires_judge("RealWorldQA") else None,
                        logger=logger,
                    )

            # Generate summary
            if all_results or explore_inference_files:
                generate_summary(run_dir, all_results)
                logger.log_section("BENCHMARK RUN COMPLETE")
                logger.log(f"Results saved to: {run_dir}")
        finally:
            if judge_process is not None:
                stop_managed_process(judge_process, "local judge server", logger)
                cleanup_vllm_engine_processes(logger)
            if server_process is not None:
                stop_managed_process(server_process, "vLLM server", logger)
                cleanup_vllm_engine_processes(logger)

    return 0


if __name__ == "__main__":
    sys.exit(main())
