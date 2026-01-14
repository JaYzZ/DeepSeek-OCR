#!/usr/bin/env python3
"""
Resolve OCRVL train_llava.sh settings from a LlamaFactory-style YAML.

Priority (highest -> lowest):
  1) key=value CLI overrides (dot paths)
  2) environment variables (ENABLE_PHASE3, NUM_GPUS, GPU_IDS, ...)
  3) YAML config
  4) built-in defaults

Outputs `export KEY='VALUE'` lines for bash to `eval`.
"""

from __future__ import annotations

import os
import shlex
import sys
from datetime import datetime
from typing import Any, Dict, List

import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def _set_dot(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = [p for p in key.split(".") if p]
    if not parts:
        raise ValueError(f"invalid override key: {key!r}")
    cur: Any = cfg
    for part in parts[:-1]:
        if part not in cur or cur[part] is None:
            cur[part] = {}
        if not isinstance(cur[part], dict):
            raise ValueError(f"cannot set {key!r}: {part!r} is not a mapping")
        cur = cur[part]
    cur[parts[-1]] = value


def _parse_overrides(args: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"override must be key=value, got: {arg!r}")
        k, v = arg.split("=", 1)
        overrides[k.strip()] = yaml.safe_load(v)
    return overrides


def _get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _count_devices(cuda_visible_devices: str) -> int:
    parts = [p.strip() for p in str(cuda_visible_devices).split(",") if p.strip()]
    return len(parts)


def _format_root(template: str, *, timestamp: str, run_name: str) -> str:
    root_dir = template.replace("{timestamp}", timestamp).replace("{run_name}", run_name)
    root_dir = root_dir.replace("{root_dir}", root_dir)  # no-op but keeps templates simple
    return root_dir


def _emit_exports(env: Dict[str, Any]) -> None:
    for k, v in env.items():
        if v is None:
            continue
        print(f"export {k}={shlex.quote(str(v))}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: resolve_llava_config.py <config.yaml> [key=value...]", file=sys.stderr)
        return 2

    config_path = sys.argv[1]
    override_args = sys.argv[2:]

    cfg = _load_yaml(config_path)
    overrides = _parse_overrides(override_args)
    for k, v in overrides.items():
        _set_dot(cfg, k, v)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_run_name = _get(cfg, "run.run_name", "llava_{timestamp}")
    run_name = str(cfg_run_name).replace("{timestamp}", timestamp)
    root_dir_template = _get(cfg, "run.root_dir", "OCRVL/checkpoints/{run_name}")
    root_dir = _format_root(str(root_dir_template), timestamp=timestamp, run_name=run_name)

    # System: prefer CUDA_VISIBLE_DEVICES (LF convention), then config, then env GPU_IDS, then default.
    cfg_cvd = _get(cfg, "system.cuda_visible_devices")
    env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    env_gpu_ids = os.environ.get("GPU_IDS")
    resolved_gpu_ids = env_cvd or (str(cfg_cvd) if cfg_cvd is not None else None) or env_gpu_ids or "0,1,2,3,4,5,6,7"
    resolved_num_gpus = os.environ.get("NUM_GPUS") or str(_count_devices(resolved_gpu_ids))

    # Phase enabling
    enable_phase3 = os.environ.get("ENABLE_PHASE3")
    if enable_phase3 is None:
        enable_phase3 = "true" if _truthy(_get(cfg, "phase3.enabled", False)) else "false"

    def phase_env(phase: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        phase_raw = _get(cfg, phase, {}) or {}
        if not isinstance(phase_raw, dict):
            raise ValueError(f"{phase} must be a mapping")

        # Support LlamaFactory-like sections (method/dataset/train/output) while also allowing flat keys.
        combined: Dict[str, Any] = {}
        for section in ("method", "dataset", "train", "output"):
            sec = phase_raw.get(section, {})
            if sec is None:
                continue
            if not isinstance(sec, dict):
                raise ValueError(f"{phase}.{section} must be a mapping")
            combined.update(sec)
        # Flat keys override section values.
        for k, v in phase_raw.items():
            if k not in {"method", "dataset", "train", "output"}:
                combined[k] = v

        out: Dict[str, Any] = {}

        finetuning_type = str(combined.get("finetuning_type", defaults["finetuning_type"]))
        lora = "1" if finetuning_type.lower() == "lora" else "0"
        out[f"{phase.upper()}_LORA"] = os.environ.get(f"{phase.upper()}_LORA", lora)

        for key, dflt in defaults.items():
            if key == "finetuning_type":
                continue
            env_name = f"{phase.upper()}_{key.upper()}"
            cfg_val = combined.get(key, dflt)
            out[env_name] = os.environ.get(env_name, cfg_val)

        # output_dir templating
        output_dir = combined.get("output_dir", defaults["output_dir"])
        output_dir = str(output_dir).replace("{root_dir}", root_dir).replace("{run_name}", run_name).replace("{timestamp}", timestamp)
        out[f"{phase.upper()}_OUTPUT_DIR"] = os.environ.get(f"{phase.upper()}_OUTPUT_DIR", output_dir)
        return out

    # Defaults are aligned with the original train_llava.sh.
    p1 = phase_env(
        "phase1",
        {
            "finetuning_type": "full",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "learning_rate": "1e-3",
            "num_train_epochs": 1,
            "dataset_type": "blip3o",
            "blip3o_dataset": "long",
            "dataset_pct": 0.05,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "output_dir": "{root_dir}/alignment",
        },
    )
    p2 = phase_env(
        "phase2",
        {
            "finetuning_type": "lora",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "learning_rate": "2e-4",
            "num_train_epochs": 3,
            "llava_json_path": "/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",
            "llava_image_dir": "/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/images",
            "render_questions": True,
            "per_device_train_batch_size": 12,
            "gradient_accumulation_steps": 4,
            "output_dir": "{root_dir}/instruction",
        },
    )
    p3 = phase_env(
        "phase3",
        {
            "finetuning_type": "lora",
            "learning_rate": "1e-4",
            "num_train_epochs": 1,
            "thinking_loss_weight": 1.0,
            "max_samples": 100000,
            "thinking_jsonl": "/share/project/xiyan/huggingface/Xkev/LLaVA-CoT-100k/train.jsonl",
            "thinking_image_dir": "/share/project/xiyan/huggingface",
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "output_dir": "{root_dir}/thinking",
        },
    )

    env: Dict[str, Any] = {
        "TIMESTAMP": timestamp,
        "RUN_NAME": run_name,
        "ROOT_DIR": root_dir,
        "GPU_IDS": resolved_gpu_ids,
        "NUM_GPUS": resolved_num_gpus,
        "CUDA_VISIBLE_DEVICES": resolved_gpu_ids,
        "TMUX_SESSION_NAME": os.environ.get("TMUX_SESSION_NAME") or _get(cfg, "system.tmux_session_name", "ocrvl_llava"),
        "ENABLE_PHASE3": enable_phase3,
        "VALIDATION_IMAGES": os.environ.get("VALIDATION_IMAGES")
        or ("true" if _truthy(_get(cfg, "validation.validation_images", True)) else "false"),
        "LLAVA_IMAGE_BASE": os.environ.get("LLAVA_IMAGE_BASE") or _get(cfg, "validation.llava_image_base"),
        "DOCLAYNET_BASE": os.environ.get("DOCLAYNET_BASE") or _get(cfg, "validation.doclaynet_png_base"),
        # Phase 2 render mode mapping to existing scripts.
        "PHASE2_RENDER": "1" if _truthy(p2.get("PHASE2_RENDER_QUESTIONS", p2.get("PHASE2_RENDER_QUESTIONS", True))) else "0",
    }

    # Normalize to existing script variable names.
    def rename(prefix: str, mapping: Dict[str, str]) -> None:
        for src, dst in mapping.items():
            if src in env:
                env[dst] = env.pop(src)

    rename(
        "",
        {
            "TMUX_SESSION_NAME": "TMUX_SESSION_NAME",
        },
    )

    # Bridge phase keys to the environment-variable names train_llava.sh already uses.
    def bridge_phase(phase_prefix: str, phase_vals: Dict[str, Any]) -> None:
        # These are internal outputs; remap into the train_llava.sh names.
        for k, v in list(phase_vals.items()):
            env[k] = v

    bridge_phase("PHASE1", p1)
    bridge_phase("PHASE2", p2)
    bridge_phase("PHASE3", p3)

    # Extra bridge fields expected by train_llava.sh.
    env.setdefault("PHASE1_BLIP3O_DATASET", env.get("PHASE1_BLIP3O_DATASET", "long"))
    env.setdefault("PHASE1_DATASET_PCT", env.get("PHASE1_DATASET_PCT", 0.05))
    env.setdefault("PHASE1_NUM_EPOCHS", env.pop("PHASE1_NUM_TRAIN_EPOCHS", None) or env.get("PHASE1_NUM_EPOCHS") or 1)
    env.setdefault("PHASE2_NUM_EPOCHS", env.pop("PHASE2_NUM_TRAIN_EPOCHS", None) or env.get("PHASE2_NUM_EPOCHS") or 3)
    env.setdefault("PHASE3_NUM_EPOCHS", env.pop("PHASE3_NUM_TRAIN_EPOCHS", None) or env.get("PHASE3_NUM_EPOCHS") or 1)

    env.setdefault("PHASE1_LR", env.pop("PHASE1_LEARNING_RATE", None) or env.get("PHASE1_LR") or "1e-3")
    env.setdefault("PHASE2_LR", env.pop("PHASE2_LEARNING_RATE", None) or env.get("PHASE2_LR") or "2e-4")
    env.setdefault("PHASE3_LR", env.pop("PHASE3_LEARNING_RATE", None) or env.get("PHASE3_LR") or "1e-4")

    env.setdefault("PHASE3_THINKING_LOSS_WEIGHT", env.pop("PHASE3_THINKING_LOSS_WEIGHT", None) or _get(cfg, "phase3.train.thinking_loss_weight", 1.0))
    env.setdefault("PHASE3_MAX_SAMPLES", env.pop("PHASE3_MAX_SAMPLES", None) or _get(cfg, "phase3.train.max_samples", 100000))

    # Cleanup: remove helper keys that we don't want to leak.
    for k in list(env.keys()):
        if k.endswith("_RENDER_QUESTIONS"):
            env.pop(k, None)
        if k.endswith("_PER_DEVICE_TRAIN_BATCH_SIZE") or k.endswith("_GRADIENT_ACCUMULATION_STEPS"):
            # Not directly used by train_llava.sh (it delegates to per-stage scripts).
            pass

    try:
        _emit_exports(env)
    except BrokenPipeError:
        # Common when users pipe into `head`; treat as successful.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
