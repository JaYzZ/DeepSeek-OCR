#!/usr/bin/env python3
"""Shared helpers for vLLM serving/backfill scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from peft import PeftModel
from tokenizers import AddedToken
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
import yaml

DEFAULT_OPSD_SPAN_STUDENT_TEMPLATE = """Answer the question.

Use <think>...</think> for reasoning. Inside the thinking span, <latent> means a compressed reasoning segment that stands for omitted thinking content. Only use <latent> inside <think>...</think>, and use it when you compress part of the reasoning instead of writing it out fully.

Question:
{question_text}"""

THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
LATENT_TOKEN = "<latent>"
THINK_SEP_TOKEN = "<think_sep>"


def vllm_thinking_enabled(default: str = "1") -> bool:
    return os.environ.get("VLLM_THINKING", default).strip().lower() in {"1", "true", "yes", "on"}


def should_force_think_prompt(default: str = "1") -> bool:
    # Discrete AR should be prompted into <think>; continuous AR should not.
    return not vllm_thinking_enabled(default=default)


def append_forced_think_prompt(prompt_text: str, *, default: str = "1") -> str:
    if should_force_think_prompt(default=default):
        return prompt_text + "<think>"
    return prompt_text


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


def build_opsd_span_student_prompt(question_text: str) -> str:
    template = _resolve_opsd_span_student_template()
    return template.format(question_text=(question_text or "").strip()).strip()


def opsd_span_prompt_enabled() -> bool:
    if os.environ.get("OPSD_STUDENT_TEMPLATE"):
        return True

    config_path = os.environ.get("OPSD_SPAN_PROMPT_CONFIG_PATH", "").strip()
    if not config_path:
        return False

    return Path(config_path).exists()


def _resolve_opsd_span_student_template() -> str:
    env_template = os.environ.get("OPSD_STUDENT_TEMPLATE")
    if env_template:
        return str(env_template)

    config_path = os.environ.get("OPSD_SPAN_PROMPT_CONFIG_PATH", "").strip()
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with config_file.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            prompts = config.get("prompts") or {}
            template = prompts.get("student_user")
            if template:
                return str(template)

    return DEFAULT_OPSD_SPAN_STUDENT_TEMPLATE


def resolve_lora_artifacts(
    model_path: Optional[str],
    lora_path: Optional[str],
) -> Dict[str, Any]:
    """Resolve adapter metadata without treating PEFT provenance as runtime config."""
    resolved_model_path = model_path
    resolved_lora_rank = None
    adapter_config_path = None

    if not lora_path:
        return {
            "model_path": resolved_model_path,
            "lora_path": lora_path,
            "lora_rank": resolved_lora_rank,
            "adapter_config_path": adapter_config_path,
        }

    adapter_config_path = Path(lora_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        print(
            f"WARNING: Missing adapter_config.json at {adapter_config_path}. "
            f"Using model_path={resolved_model_path!r} and default max LoRA rank handling."
        )
        return {
            "model_path": resolved_model_path,
            "lora_path": lora_path,
            "lora_rank": resolved_lora_rank,
            "adapter_config_path": str(adapter_config_path),
        }

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    rank_value = adapter_config.get("r")
    if rank_value is not None:
        resolved_lora_rank = int(rank_value)

    trainable_token_indices = _extract_trainable_token_indices(adapter_config)
    if trainable_token_indices:
        merged_model_path = _materialize_vllm_merged_model(
            model_path=resolved_model_path,
            lora_path=lora_path,
            adapter_config=adapter_config,
            trainable_token_indices=trainable_token_indices,
        )
        return {
            "model_path": merged_model_path,
            "lora_path": None,
            "lora_rank": None,
            "adapter_config_path": str(adapter_config_path),
        }

    return {
        "model_path": resolved_model_path,
        "lora_path": lora_path,
        "lora_rank": resolved_lora_rank,
        "adapter_config_path": str(adapter_config_path),
    }


def _extract_trainable_token_indices(adapter_config: Dict[str, Any]) -> List[int]:
    raw_indices = adapter_config.get("trainable_token_indices")
    if raw_indices is None:
        return []

    if isinstance(raw_indices, dict):
        values: List[int] = []
        for token_ids in raw_indices.values():
            if isinstance(token_ids, list):
                values.extend(int(token_id) for token_id in token_ids)
        return sorted(set(values))

    if isinstance(raw_indices, list):
        return sorted(set(int(token_id) for token_id in raw_indices))

    return []


def prepare_inference_tokenizer(
    tokenizer_path: str,
    *,
    logger=None,
) -> tuple[Any, Dict[str, int]]:
    """Load and validate the tokenizer used by vLLM-side inference.

    This keeps training/backfill/benchmark on the same <latent>/<think_sep>
    token IDs and fails fast if the tokenizer regresses to multi-token splits or
    collides with the built-in thinking markers.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    for token in (LATENT_TOKEN, THINK_SEP_TOKEN):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) > 1:
            tokenizer.add_tokens([token], special_tokens=False)

    for token in (LATENT_TOKEN, THINK_SEP_TOKEN):
        token_id = tokenizer.convert_tokens_to_ids(token)
        added = tokenizer.added_tokens_decoder.get(token_id)
        if added is not None and getattr(added, "special", False):
            tokenizer._tokenizer.add_tokens([AddedToken(token, special=False)])

    token_ids: Dict[str, int] = {}
    for token in (THINK_START_TOKEN, THINK_END_TOKEN, LATENT_TOKEN, THINK_SEP_TOKEN):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Inference tokenizer must encode {token!r} as exactly one token, got {encoded} from {tokenizer_path}"
            )
        token_ids[token] = int(encoded[0])

    if token_ids[LATENT_TOKEN] in {token_ids[THINK_START_TOKEN], token_ids[THINK_END_TOKEN]}:
        raise ValueError(
            "Inference tokenizer maps <latent> onto <think> or </think>; tokenizer artifacts are inconsistent."
        )
    if token_ids[THINK_SEP_TOKEN] in {
        token_ids[THINK_START_TOKEN],
        token_ids[THINK_END_TOKEN],
        token_ids[LATENT_TOKEN],
    }:
        raise ValueError(
            "Inference tokenizer maps <think_sep> onto an existing thinking marker; tokenizer artifacts are inconsistent."
        )

    os.environ["QWEN3VL_LATENT_TOKEN_ID"] = str(token_ids[LATENT_TOKEN])
    os.environ["QWEN3VL_THINKING_SEP_ID"] = str(token_ids[THINK_SEP_TOKEN])

    if logger is not None:
        logger.info(
            "Validated inference tokenizer IDs: <think>=%s </think>=%s <latent>=%s <think_sep>=%s",
            token_ids[THINK_START_TOKEN],
            token_ids[THINK_END_TOKEN],
            token_ids[LATENT_TOKEN],
            token_ids[THINK_SEP_TOKEN],
        )

    return tokenizer, token_ids


def _materialize_vllm_merged_model(
    *,
    model_path: Optional[str],
    lora_path: str,
    adapter_config: Dict[str, Any],
    trainable_token_indices: List[int],
) -> str:
    if not model_path:
        raise ValueError("model_path is required to materialize a merged vLLM model")

    adapter_dir = Path(lora_path)
    merged_dir = adapter_dir / "merged_vllm"
    merged_meta_path = merged_dir / "merged_from_trainable_tokens.json"
    adapter_model_path = adapter_dir / "adapter_model.safetensors"
    adapter_mtime = max(
        adapter_model_path.stat().st_mtime if adapter_model_path.exists() else 0.0,
        (adapter_dir / "adapter_config.json").stat().st_mtime,
    )

    if merged_meta_path.exists() and merged_dir.exists() and merged_dir.joinpath("config.json").exists():
        if merged_meta_path.stat().st_mtime >= adapter_mtime:
            return str(merged_dir)

    base_model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    input_embeddings = base_model.get_input_embeddings()
    target_vocab_size = max(int(input_embeddings.weight.shape[0]), max(trainable_token_indices) + 1)
    if target_vocab_size > int(input_embeddings.weight.shape[0]):
        base_model.resize_token_embeddings(target_vocab_size)

    peft_model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
    merged_model = peft_model.merge_and_unload()

    tmp_dir = Path(tempfile.mkdtemp(prefix="merged_vllm_", dir=str(adapter_dir)))
    try:
        merged_model.save_pretrained(tmp_dir, safe_serialization=True)

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.save_pretrained(tmp_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        for token in ("<latent>", "<think_sep>"):
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) > 1:
                tokenizer.add_tokens([token], special_tokens=False)
        tokenizer.save_pretrained(tmp_dir)

        tmp_meta_path = tmp_dir / merged_meta_path.name
        with tmp_meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_adapter_path": str(adapter_dir),
                    "base_model_path": model_path,
                    "trainable_token_indices": trainable_token_indices,
                    "lora_rank": adapter_config.get("r"),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        shutil.move(str(tmp_dir), str(merged_dir))
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return str(merged_dir)


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

        if not os.environ.get("MAX_CONTINUOUS_STEPS"):
            raw_max = cfg.get("max_continuous_steps")
            raw_tokens = cfg.get("max_new_tokens", 40960)
            try:
                max_new_tokens = int(raw_tokens)
            except Exception:
                max_new_tokens = 40960
            derived = max(1, max_new_tokens // 2 if max_new_tokens > 1 else 1)
            if raw_max is None or str(raw_max).strip() == "":
                os.environ["MAX_CONTINUOUS_STEPS"] = str(derived)
            else:
                try:
                    os.environ["MAX_CONTINUOUS_STEPS"] = str(max(0, int(raw_max)))
                except Exception:
                    os.environ["MAX_CONTINUOUS_STEPS"] = str(derived)
            loaded.append(f"MAX_CONTINUOUS_STEPS={os.environ['MAX_CONTINUOUS_STEPS']}")

        if not os.environ.get("VLLM_THINKING"):
            val = cfg.get("vllm_thinking", 1)
            os.environ["VLLM_THINKING"] = "1" if str(val).strip().lower() in {"1", "true", "yes", "on"} else "0"
            loaded.append(f"VLLM_THINKING={os.environ['VLLM_THINKING']}")

        if not os.environ.get("VLLM_ENFORCE_EAGER"):
            val = cfg.get("vllm_enforce_eager", 0)
            os.environ["VLLM_ENFORCE_EAGER"] = "1" if str(val).strip().lower() in {"1", "true", "yes", "on"} else "0"
            loaded.append(f"VLLM_ENFORCE_EAGER={os.environ['VLLM_ENFORCE_EAGER']}")

        if not os.environ.get("DISABLE_VERSION_CHECK"):
            val = cfg.get("disable_version_check", 1)
            os.environ["DISABLE_VERSION_CHECK"] = (
                "1" if str(val).strip().lower() in {"1", "true", "yes", "on"} else "0"
            )
            loaded.append(f"DISABLE_VERSION_CHECK={os.environ['DISABLE_VERSION_CHECK']}")

        if not os.environ.get("VLLM_TOKENIZER_PATH") and os.environ.get("VLLM_MODEL_PATH"):
            os.environ["VLLM_TOKENIZER_PATH"] = os.environ["VLLM_MODEL_PATH"]
            loaded.append(f"VLLM_TOKENIZER_PATH={os.environ['VLLM_TOKENIZER_PATH']}")

        if loaded:
            _emit_info(f"{', '.join(loaded)} (from {cfg_path})")
    except Exception as e:
        os.environ.setdefault("MIN_CONTINUOUS_STEPS", "0")
        os.environ.setdefault("MAX_CONTINUOUS_STEPS", "20480")
        os.environ.setdefault("VLLM_THINKING", "1")
        os.environ.setdefault("VLLM_ENFORCE_EAGER", "0")
        os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
        _emit_warn(
            f"failed to load vLLM thinking runtime env from {cfg_path} ({e}); "
            f"fallback MIN_CONTINUOUS_STEPS={os.environ['MIN_CONTINUOUS_STEPS']}, "
            f"MAX_CONTINUOUS_STEPS={os.environ['MAX_CONTINUOUS_STEPS']}, "
            f"VLLM_THINKING={os.environ['VLLM_THINKING']}, "
            f"VLLM_ENFORCE_EAGER={os.environ['VLLM_ENFORCE_EAGER']}, "
            f"DISABLE_VERSION_CHECK={os.environ['DISABLE_VERSION_CHECK']}"
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
