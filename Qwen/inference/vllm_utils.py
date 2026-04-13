#!/usr/bin/env python3
"""Shared helpers for vLLM serving/backfill scripts."""

from __future__ import annotations

import os
import subprocess
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def normalize_media_path(value: Any) -> Any:
    """Normalize local file URIs to plain filesystem paths."""
    if isinstance(value, str) and value.startswith("file://"):
        return value[len("file://"):]
    return value


def resolve_lora_artifacts(
    model_path: Optional[str],
    lora_path: Optional[str],
) -> Dict[str, Any]:
    """Resolve base model path and adapter rank from a LoRA checkpoint if present."""
    resolved_model_path = model_path
    resolved_lora_rank = None
    adapter_config_path = None

    if not lora_path:
        return {
            "model_path": resolved_model_path,
            "lora_rank": resolved_lora_rank,
            "adapter_config_path": adapter_config_path,
        }

    adapter_config_path = Path(lora_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        print(
            f"WARNING: Missing adapter_config.json at {adapter_config_path}. "
            f"Falling back to model_path={resolved_model_path!r} and default max LoRA rank handling."
        )
        return {
            "model_path": resolved_model_path,
            "lora_rank": resolved_lora_rank,
            "adapter_config_path": str(adapter_config_path),
        }

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    resolved_model_path = adapter_config.get("base_model_name_or_path") or resolved_model_path
    rank_value = adapter_config.get("r")
    if rank_value is not None:
        resolved_lora_rank = int(rank_value)

    return {
        "model_path": resolved_model_path,
        "lora_rank": resolved_lora_rank,
        "adapter_config_path": str(adapter_config_path),
    }


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


def cleanup_vllm_engine_processes(logger=None) -> None:
    """Kill leaked vLLM engine-core workers that can survive parent shutdown."""

    def _emit(msg: str) -> None:
        if logger is not None:
            if hasattr(logger, "log"):
                logger.log(msg)
            elif hasattr(logger, "info"):
                logger.info(msg)
        else:
            print(msg)

    try:
        subprocess.run(
            ["pkill", "-9", "-f", "VLLM::EngineCore"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _emit("Cleaned up leaked vLLM engine-core processes")
    except Exception as e:
        _emit(f"Warning: Failed to clean vLLM engine-core processes: {e}")
