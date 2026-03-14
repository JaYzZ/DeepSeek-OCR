#!/usr/bin/env python3
"""Shared helpers for vLLM serving/backfill scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def parse_cuda_visible_devices(cuda_visible_devices: Optional[str]) -> List[str]:
    """Parse CUDA_VISIBLE_DEVICES into a normalized GPU id list."""
    if not cuda_visible_devices:
        return []
    return [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]


def infer_tensor_parallel_size(cuda_visible_devices: Optional[str], fallback: int = 1) -> int:
    """Infer TP size from CUDA_VISIBLE_DEVICES."""
    gpus = parse_cuda_visible_devices(cuda_visible_devices)
    return len(gpus) if gpus else max(1, int(fallback))


def normalize_checkpoint_name(checkpoint: str) -> str:
    """Return the trailing checkpoint directory name."""
    checkpoint = checkpoint.rstrip("/")
    return checkpoint.split("/")[-1]


def apply_runtime_env_for_thinking(
    *,
    repo_root: Optional[Path] = None,
    logger=None,
) -> None:
    """Set shared vLLM thinking env vars from runtime env yaml unless already set."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    cfg_path = os.environ.get(
        "QWEN3VL_RUNTIME_ENV_CONFIG",
        str(repo_root / "Qwen/configs/qwen3vl_runtime_env.yaml"),
    )

    def _emit_info(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def _emit_warn(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")

    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        loaded = []

        if not os.environ.get("MIN_CONTINUOUS_STEPS"):
            val = cfg.get("min_continuous_steps", 0)
            os.environ["MIN_CONTINUOUS_STEPS"] = str(int(val))
            loaded.append(f"MIN_CONTINUOUS_STEPS={os.environ['MIN_CONTINUOUS_STEPS']}")

        if not os.environ.get("VLLM_FORCE_THINK"):
            val = cfg.get("vllm_force_think", 0)
            os.environ["VLLM_FORCE_THINK"] = "1" if str(val).strip().lower() in {"1", "true", "yes", "on"} else "0"
            loaded.append(f"VLLM_FORCE_THINK={os.environ['VLLM_FORCE_THINK']}")

        if not os.environ.get("VLLM_THINKING"):
            val = cfg.get("vllm_thinking", 1)
            os.environ["VLLM_THINKING"] = "1" if str(val).strip().lower() in {"1", "true", "yes", "on"} else "0"
            loaded.append(f"VLLM_THINKING={os.environ['VLLM_THINKING']}")

        if not os.environ.get("VLLM_ENFORCE_EAGER"):
            val = cfg.get("vllm_enforce_eager", 0)
            os.environ["VLLM_ENFORCE_EAGER"] = "1" if str(val).strip().lower() in {"1", "true", "yes", "on"} else "0"
            loaded.append(f"VLLM_ENFORCE_EAGER={os.environ['VLLM_ENFORCE_EAGER']}")

        if loaded:
            _emit_info(f"{', '.join(loaded)} (from {cfg_path})")
    except Exception as e:
        os.environ.setdefault("MIN_CONTINUOUS_STEPS", "0")
        os.environ.setdefault("VLLM_THINKING", "1")
        os.environ.setdefault("VLLM_FORCE_THINK", "0")
        os.environ.setdefault("VLLM_ENFORCE_EAGER", "0")
        _emit_warn(
            f"failed to load vLLM thinking runtime env from {cfg_path} ({e}); "
            f"fallback MIN_CONTINUOUS_STEPS={os.environ['MIN_CONTINUOUS_STEPS']}, "
            f"VLLM_THINKING={os.environ['VLLM_THINKING']}, "
            f"VLLM_FORCE_THINK={os.environ['VLLM_FORCE_THINK']}, "
            f"VLLM_ENFORCE_EAGER={os.environ['VLLM_ENFORCE_EAGER']}"
        )
