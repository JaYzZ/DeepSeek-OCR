#!/usr/bin/env python3
"""Continuous-thinking OPSD trainer for Qwen3-VL.

This keeps the core OPSD loop:
1. Student generates an on-policy rollout from question-only context.
2. Teacher replays the same rollout under privileged context.
3. Student is optimized against teacher token probabilities on that rollout.

The VLM-specific teacher privilege is extra rationale images already rendered by
the thinking dataset builders. No images or latent caches are regenerated here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
import logging
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, GenerationConfig

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:  # pragma: no cover - environment-specific
    LoraConfig = None
    PeftModel = None
    get_peft_model = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen.llamafactory.integration import (
    LatentVAE,
    _align_module_to_model_dtype_device,
    _resolve_latent_vae_module,
    load_vae_checkpoint,
    save_vae_checkpoint,
)


class _DropMessageFilter(logging.Filter):
    def __init__(self, substrings: tuple[str, ...]):
        super().__init__()
        self.substrings = substrings

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(text in message for text in self.substrings)


def _configure_quiet_logging() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"The argument `trust_remote_code` is to be used with Auto classes\. It has no effect here and is ignored\.",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"None of the inputs have requires_grad=True\. Gradients will be None",
    )

    drop_filter = _DropMessageFilter(
        (
            "No serialization method found for <class 'NoneType'>. Falling back to pickle.",
            "[Thinking] No VAE found",
            "[Thinking] vLLM continuous AR active:",
            "Successfully reset prefix cache",
            "[Thinking] Token IDs resolved:",
        )
    )
    for logger_name in (
        "",
        "vllm",
        "vllm_thinking.runner_patch",
        "vllm_thinking",
    ):
        logging.getLogger(logger_name).addFilter(drop_filter)

    for noisy_logger_name in (
        "vllm.multimodal.hasher",
        "vllm.v1.core.block_pool",
    ):
        noisy_logger = logging.getLogger(noisy_logger_name)
        noisy_logger.handlers.clear()
        noisy_logger.propagate = False
        noisy_logger.disabled = True

    logging.getLogger("vllm_thinking.runner_patch").setLevel(logging.WARNING)
    logging.getLogger("vllm_thinking").setLevel(logging.WARNING)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[normalized]


def _parse_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _resolve_path(value: str | None, *, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _default_manifest_path(dataset_names: str) -> Path:
    parts = [item.strip() for item in dataset_names.split(",") if item.strip()]
    if not parts:
        raise ValueError("data.dataset_names must be non-empty")
    slug = "__".join(parts)
    return (REPO_ROOT / "Qwen" / "data" / "opsd" / f"{slug}_opsd.jsonl").resolve()


def _assistant_answer(text: str) -> str:
    marker = "</think>"
    idx = text.find(marker)
    if idx < 0:
        return text.strip()
    return text[idx + len(marker) :].strip()


def _format_user_prompt(
    template: str,
    *,
    num_question_images: int,
    num_rationale_images: int,
    question_text: str,
) -> str:
    return template.format(
        num_question_images=num_question_images,
        num_rationale_images=num_rationale_images,
        question_text=question_text,
    ).strip()


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _maybe_limit_images(images: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return images
    return images[:limit]


def _apply_multimodal_processor_limits(processor, config: dict[str, Any]) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return

    image_max_pixels = int(config.get("image_max_pixels", 262144))
    video_max_pixels = int(config.get("video_max_pixels", 16384))
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = image_max_pixels
    if hasattr(image_processor, "video_max_pixels"):
        image_processor.video_max_pixels = video_max_pixels


class OpsdManifestDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.rows: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class OpsdVlmCollator:
    def __init__(self, student_processor, teacher_processor, config: dict[str, Any]):
        self.student_processor = student_processor
        self.teacher_processor = teacher_processor
        self.config = config
        self.system_prompt = str(config.get("system_prompt", "") or "").strip()
        self.student_template = str(config["prompts"]["student_user"])
        self.teacher_template = str(config["prompts"]["teacher_user"])
        self.max_length = int(config["training"]["max_length"])
        self.max_teacher_rationale_images = int(config["prompts"].get("max_teacher_rationale_images", 0))

    @staticmethod
    def _with_image_fields(messages: list[dict[str, Any]], image_paths: list[str]) -> list[dict[str, Any]]:
        idx = 0
        enriched: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                enriched.append(message)
                continue
            new_content: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_item = dict(item)
                    image_item["image"] = image_paths[idx]
                    idx += 1
                    new_content.append(image_item)
                else:
                    new_content.append(item)
            new_message = dict(message)
            new_message["content"] = new_content
            enriched.append(new_message)
        return enriched

    def _build_student_messages(
        self,
        user_text: str,
        *,
        num_question_images: int,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if num_question_images > 0:
            content.append({"type": "text", "text": "Image(s):"})
            content.extend({"type": "image"} for _ in range(num_question_images))
        content.append({"type": "text", "text": user_text})
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _build_teacher_messages(
        self,
        user_text: str,
        *,
        num_question_images: int,
        num_rationale_images: int,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if num_question_images > 0:
            content.append({"type": "text", "text": "Image(s):"})
            content.extend({"type": "image"} for _ in range(num_question_images))
        if num_rationale_images > 0:
            content.append({"type": "text", "text": "Reference image(s):"})
            content.extend({"type": "image"} for _ in range(num_rationale_images))
        content.append({"type": "text", "text": user_text})
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _encode_with_processor(self, processor, texts: list[str], images: list[list[Image.Image]]) -> dict[str, torch.Tensor]:
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        prompt_length = int(batch["input_ids"].shape[1])
        if prompt_length > self.max_length:
            raise ValueError(
                f"Prompt length {prompt_length} exceeds training.max_length={self.max_length}. "
                "Increase training.max_length; question text remains untruncated."
            )
        batch["prompt_lengths_per_example"] = batch["attention_mask"].sum(dim=1)
        batch["prompt_length"] = prompt_length
        return batch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        student_texts: list[str] = []
        teacher_texts: list[str] = []
        student_images: list[list[Image.Image]] = []
        teacher_images: list[list[Image.Image]] = []
        metadata: list[dict[str, Any]] = []

        for feature in features:
            question_paths = list(feature.get("question_images") or [])
            rationale_paths = _maybe_limit_images(
                list(feature.get("teacher_rationale_images") or []),
                self.max_teacher_rationale_images,
            )
            question_text = str(feature.get("student_user_text") or "").strip()
            loaded_question_images = [_load_image(path) for path in question_paths]
            loaded_teacher_images = loaded_question_images + [_load_image(path) for path in rationale_paths]
            student_user = _format_user_prompt(
                self.student_template,
                num_question_images=len(question_paths),
                num_rationale_images=0,
                question_text=question_text,
            )
            teacher_user = _format_user_prompt(
                self.teacher_template,
                num_question_images=len(question_paths),
                num_rationale_images=len(rationale_paths),
                question_text=question_text,
            )

            student_messages = self._build_student_messages(
                student_user,
                num_question_images=len(question_paths),
            )
            teacher_messages = self._build_teacher_messages(
                teacher_user,
                num_question_images=len(question_paths),
                num_rationale_images=len(rationale_paths),
            )

            student_prompt_text = self.student_processor.apply_chat_template(
                student_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            teacher_prompt_text = self.teacher_processor.apply_chat_template(
                teacher_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            student_texts.append(student_prompt_text)
            teacher_texts.append(teacher_prompt_text)

            student_images.append(loaded_question_images)
            teacher_images.append(loaded_teacher_images)

            student_vllm_messages = self._with_image_fields(student_messages, question_paths)

            metadata.append(
                {
                    "sample_id": feature.get("sample_id"),
                    "source_dataset": feature.get("source_dataset"),
                    "task": feature.get("task"),
                    "answer_text": feature.get("answer_text")
                    or _assistant_answer(feature.get("assistant_target", "")),
                    "question_images": question_paths,
                    "teacher_rationale_images": rationale_paths,
                    "student_vllm_messages": student_vllm_messages,
                    "student_prompt_text": student_prompt_text,
                    "teacher_prompt_text": teacher_prompt_text,
                }
            )

        return {
            "student_prompt": self._encode_with_processor(self.student_processor, student_texts, student_images),
            "teacher_prompt": self._encode_with_processor(self.teacher_processor, teacher_texts, teacher_images),
            "metadata": metadata,
        }


def _move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=model_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _build_generation_config(config: dict[str, Any]) -> GenerationConfig:
    generation_cfg = config["generation"]
    return GenerationConfig(
        max_new_tokens=int(generation_cfg["max_new_tokens"]),
        do_sample=bool(generation_cfg.get("do_sample", True)),
        temperature=float(generation_cfg.get("temperature", 1.0)),
        top_p=float(generation_cfg.get("top_p", 1.0)),
        top_k=int(generation_cfg.get("top_k", 20)),
        repetition_penalty=float(generation_cfg.get("repetition_penalty", 1.0)),
        use_cache=True,
        pad_token_id=int(config["tokens"]["pad_token_id"]),
    )


def _generalized_jsd_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    temperature: float,
    top_k: int,
    token_clip: float,
) -> torch.Tensor:
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    if top_k and top_k > 0:
        _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
        student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log1p(-beta_tensor),
                teacher_log_probs + torch.log(beta_tensor),
            ]
        ),
        dim=0,
    )
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
    jsd = beta_tensor * kl_teacher + (1.0 - beta_tensor) * kl_student
    jsd = jsd.sum(dim=-1)

    if token_clip and token_clip > 0:
        jsd = torch.clamp(jsd, max=token_clip)

    mask = labels != -100
    if mask.any():
        return jsd[mask].mean()
    return jsd.mean()


def _sampled_reverse_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs_sampled = torch.gather(
        student_log_probs,
        dim=-1,
        index=sampled_token_ids.unsqueeze(-1),
    ).squeeze(-1)
    teacher_log_probs_sampled = torch.gather(
        teacher_log_probs,
        dim=-1,
        index=sampled_token_ids.unsqueeze(-1),
    ).squeeze(-1)

    advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()
    mask = labels != -100
    if mask.any():
        loss = -(advantage[mask] * student_log_probs_sampled[mask]).mean()
        mean_advantage = float(advantage[mask].mean().detach())
        mean_student_lp = float(student_log_probs_sampled[mask].mean().detach())
        mean_teacher_lp = float(teacher_log_probs_sampled[mask].mean().detach())
    else:
        loss = -(advantage * student_log_probs_sampled).mean()
        mean_advantage = float(advantage.mean().detach())
        mean_student_lp = float(student_log_probs_sampled.mean().detach())
        mean_teacher_lp = float(teacher_log_probs_sampled.mean().detach())

    metrics = {
        "advantage": mean_advantage,
        "student_logprob": mean_student_lp,
        "teacher_logprob": mean_teacher_lp,
    }
    return loss, metrics


def _sampled_log_probs_from_logits(
    *,
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    gather_index = sampled_token_ids.unsqueeze(-1)
    if math.isclose(temperature, 1.0):
        sampled_logits = torch.gather(logits, dim=-1, index=gather_index).squeeze(-1)
        log_norm = torch.logsumexp(logits, dim=-1)
        return sampled_logits - log_norm

    scaled_logits = logits / temperature
    sampled_logits = torch.gather(scaled_logits, dim=-1, index=gather_index).squeeze(-1)
    log_norm = torch.logsumexp(scaled_logits, dim=-1)
    return sampled_logits - log_norm


def _sampled_reverse_kl_loss_from_logprobs(
    *,
    student_log_probs_sampled: torch.Tensor,
    teacher_log_probs_sampled: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()
    mask = labels != -100
    if mask.any():
        loss = -(advantage[mask] * student_log_probs_sampled[mask]).mean()
        mean_advantage = float(advantage[mask].mean().detach())
        mean_student_lp = float(student_log_probs_sampled[mask].mean().detach())
        mean_teacher_lp = float(teacher_log_probs_sampled[mask].mean().detach())
    else:
        loss = -(advantage * student_log_probs_sampled).mean()
        mean_advantage = float(advantage.mean().detach())
        mean_student_lp = float(student_log_probs_sampled.mean().detach())
        mean_teacher_lp = float(teacher_log_probs_sampled.mean().detach())

    metrics = {
        "advantage": mean_advantage,
        "student_logprob": mean_student_lp,
        "teacher_logprob": mean_teacher_lp,
    }
    return loss, metrics


def _freeze_visual_parameters(model) -> None:
    visual = getattr(model, "visual", None)
    if visual is None and hasattr(model, "model"):
        visual = getattr(model.model, "visual", None)
    if visual is None:
        return
    for param in visual.parameters():
        param.requires_grad = False


def _get_model_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        nested_hidden_size = getattr(text_config, "hidden_size", None)
        if nested_hidden_size is not None:
            return int(nested_hidden_size)

    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return _get_model_hidden_size(base_model)

    nested_model = getattr(model, "model", None)
    if nested_model is not None and nested_model is not model:
        return _get_model_hidden_size(nested_model)

    raise RuntimeError("Failed to resolve model hidden size for LatentVAE creation.")


def _ensure_resumed_vae_loaded(model, vae_checkpoint_path: Path) -> bool:
    if not vae_checkpoint_path.exists():
        return False

    vae = _resolve_latent_vae_module(model)
    if vae is None:
        hidden_size = _get_model_hidden_size(model)
        vae = LatentVAE(
            hidden_size=hidden_size,
            intermediate_size=int(os.environ.get("QWEN3VL_VAE_INTERMEDIATE_SIZE", "512")),
            deterministic=False,
        )
        _align_module_to_model_dtype_device(model, vae)
        model.register_module("latent_vae", vae)

    load_vae_checkpoint(model, str(vae_checkpoint_path))

    loaded_vae = _resolve_latent_vae_module(model)
    if loaded_vae is None:
        raise RuntimeError(f"Failed to attach/load latent_vae from {vae_checkpoint_path}")

    os.environ["QWEN3VL_VAE_CHECKPOINT_PATH"] = str(vae_checkpoint_path)
    return True


def _teacher_strategy(config: dict[str, Any]) -> str:
    teacher_cfg = config.get("teacher") or {}
    return str(teacher_cfg.get("strategy", "fixed")).strip().lower()


def _load_processor_for_model(model_path: str, config: dict[str, Any]):
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    _apply_multimodal_processor_limits(processor, config)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
    return processor


def _prepare_teacher_processor(config: dict[str, Any], student_processor):
    teacher_cfg = config.get("teacher") or {}
    strategy = _teacher_strategy(config)
    teacher_processor_path = (
        teacher_cfg.get("processor_name_or_path")
        or teacher_cfg.get("model_name_or_path")
        or config["model"]["model_name_or_path"]
    )
    if strategy != "separate" or str(teacher_processor_path) == str(config["model"]["model_name_or_path"]):
        return student_processor

    teacher_processor = _load_processor_for_model(str(teacher_processor_path), config)
    student_tokenizer = getattr(student_processor, "tokenizer", None)
    teacher_tokenizer = getattr(teacher_processor, "tokenizer", None)
    if student_tokenizer is None or teacher_tokenizer is None:
        raise RuntimeError("Both student and teacher processors must expose tokenizers for OPSD replay.")
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            "Separate teacher processor/tokenizer is not replay-compatible with the student tokenizer. "
            "Teacher replay reuses student rollout token IDs, so vocabularies must match exactly."
        )
    return teacher_processor


def _prepare_model(config: dict[str, Any], accelerator: Accelerator):
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    model_dtype = _parse_dtype(str(model_cfg.get("dtype", "bf16")))

    model = AutoModelForVision2Seq.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=True,
        dtype=model_dtype,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
    )

    if bool(model_cfg.get("freeze_vision_tower", True)):
        _freeze_visual_parameters(model)

    if bool(model_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    adapter_path = model_cfg.get("adapter_name_or_path")
    resume_vae_expected = False
    if adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required to load adapter_name_or_path")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
        )

        vae_checkpoint_path = Path(adapter_path) / "vae.safetensors"
        resume_vae_expected = _ensure_resumed_vae_loaded(model, vae_checkpoint_path)
    elif lora_cfg.get("enable", True):
        if get_peft_model is None or LoraConfig is None:
            raise ImportError("peft is required for LoRA OPSD training")
        target_modules = [item.strip() for item in str(lora_cfg["target_modules"]).split(",") if item.strip()]
        peft_config = LoraConfig(
            r=int(lora_cfg["rank"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"[opsd] model dtype={model_dtype} trainable_params={trainable:,} total_params={total:,}"
        )
        if resume_vae_expected:
            print(f"[opsd] loaded resumed VAE from {Path(adapter_path) / 'vae.safetensors'}")

    return model, model_dtype, resume_vae_expected


def _prepare_teacher_model(config: dict[str, Any], accelerator: Accelerator):
    teacher_cfg = config.get("teacher") or {}
    strategy = _teacher_strategy(config)
    if strategy != "separate":
        return None, None

    teacher_model_path = str(teacher_cfg.get("model_name_or_path") or config["model"]["model_name_or_path"])
    teacher_dtype = _parse_dtype(str(teacher_cfg.get("dtype", config["model"].get("dtype", "bf16"))))
    teacher_model = AutoModelForVision2Seq.from_pretrained(
        teacher_model_path,
        trust_remote_code=True,
        dtype=teacher_dtype,
        attn_implementation=teacher_cfg.get(
            "attn_implementation",
            config["model"].get("attn_implementation", "flash_attention_2"),
        ),
    )

    if bool(teacher_cfg.get("freeze_vision_tower", config["model"].get("freeze_vision_tower", True))):
        _freeze_visual_parameters(teacher_model)

    teacher_adapter_path = teacher_cfg.get("adapter_name_or_path")
    if teacher_adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required to load teacher.adapter_name_or_path")
        teacher_model = PeftModel.from_pretrained(
            teacher_model,
            teacher_adapter_path,
            is_trainable=False,
        )
        if bool(teacher_cfg.get("load_vae_from_adapter", True)):
            _ensure_resumed_vae_loaded(teacher_model, Path(teacher_adapter_path) / "vae.safetensors")

    for param in teacher_model.parameters():
        param.requires_grad_(False)
    teacher_model.eval()
    if hasattr(teacher_model, "config"):
        teacher_model.config.use_cache = False

    teacher_model = accelerator.prepare_model(teacher_model, evaluation_mode=True)

    if accelerator.is_main_process:
        teacher_adapter_desc = teacher_adapter_path or "<none>"
        print(
            f"[opsd] external_teacher strategy=separate model={teacher_model_path} "
            f"adapter={teacher_adapter_desc} dtype={teacher_dtype}"
        )

    return teacher_model, teacher_dtype


def _prepare_runtime_env(config: dict[str, Any]) -> None:
    runtime_env = config.get("runtime_env_config")
    if runtime_env:
        os.environ["QWEN3VL_RUNTIME_ENV_CONFIG"] = str(runtime_env)
    os.environ.setdefault("QWEN3VL_LATENT_SUPERVISION", "1")
    os.environ.setdefault("QWEN3VL_HIDDEN_STATES_HOOK", "0")
    os.environ.setdefault("VLLM_THINKING", "1")
    os.environ.setdefault("MIN_CONTINUOUS_STEPS", "0")
    os.environ.setdefault("QWEN3VL_OPSD_REPLAY_MODE", "0")

    from Qwen.llamafactory.integration import _patch_once

    _patch_once()


@contextmanager
def _opsd_replay_mode():
    previous = os.environ.get("QWEN3VL_OPSD_REPLAY_MODE")
    os.environ["QWEN3VL_OPSD_REPLAY_MODE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("QWEN3VL_OPSD_REPLAY_MODE", None)
        else:
            os.environ["QWEN3VL_OPSD_REPLAY_MODE"] = previous


def _prepare_vllm_env(config: dict[str, Any]) -> None:
    rollout_cfg = config.get("rollout") or {}
    cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" in cuda_alloc_conf:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    torch_alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "")
    if "expandable_segments:True" in torch_alloc_conf:
        os.environ.pop("PYTORCH_ALLOC_CONF", None)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from Qwen.inference.vllm_utils import apply_runtime_env_for_thinking

    apply_runtime_env_for_thinking(repo_root=REPO_ROOT)
    if "vllm_thinking" in rollout_cfg:
        os.environ["VLLM_THINKING"] = "1" if _parse_bool_flag(rollout_cfg.get("vllm_thinking")) else "0"
    if "vllm_force_think" in rollout_cfg:
        os.environ["VLLM_FORCE_THINK"] = "1" if _parse_bool_flag(rollout_cfg.get("vllm_force_think")) else "0"
    if "vllm_enforce_eager" in rollout_cfg:
        os.environ["VLLM_ENFORCE_EAGER"] = "1" if _parse_bool_flag(rollout_cfg.get("vllm_enforce_eager")) else "0"
    plugins = [p.strip() for p in os.environ.get("VLLM_PLUGINS", "").split(",") if p.strip()]
    if os.environ.get("VLLM_THINKING", "1").strip().lower() in {"1", "true", "yes", "on"}:
        if "vllm_thinking" not in plugins:
            plugins.append("vllm_thinking")
        os.environ["VLLM_PLUGINS"] = ",".join(plugins)
        from vllm_thinking.runner_patch import apply_thinking_mode_patch

        apply_thinking_mode_patch()
    else:
        plugins = [plugin for plugin in plugins if plugin != "vllm_thinking"]
        os.environ["VLLM_PLUGINS"] = ",".join(plugins)


def _prepare_rollout_tokenizer_dir(config: dict[str, Any], accelerator: Accelerator) -> Path:
    tokenizer_dir = Path(config["training"]["output_dir"]) / "vllm_tokenizer"
    if accelerator.is_main_process:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["model_name_or_path"],
            trust_remote_code=True,
        )
        added = 0
        for token in ("<latent>", "<think_sep>"):
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                added += tokenizer.add_tokens([token], special_tokens=False)
        tokenizer.save_pretrained(tokenizer_dir)
        latent_id = tokenizer.convert_tokens_to_ids("<latent>")
        sep_id = tokenizer.convert_tokens_to_ids("<think_sep>")
        print(f"[opsd] rollout tokenizer saved to {tokenizer_dir} added_tokens={added}")
    accelerator.wait_for_everyone()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    os.environ["QWEN3VL_LATENT_TOKEN_ID"] = str(tokenizer.convert_tokens_to_ids("<latent>"))
    os.environ["QWEN3VL_THINKING_SEP_ID"] = str(tokenizer.convert_tokens_to_ids("<think_sep>"))
    return tokenizer_dir


def _forbidden_rollout_token_ids(tokenizer) -> list[int]:
    forbidden_tokens = (
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    )
    blocked_ids: set[int] = set()
    for token in forbidden_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            blocked_ids.add(token_id)
    return sorted(blocked_ids)


def _prepare_messages_for_vllm(messages: list[dict[str, Any]], processor) -> dict[str, Any] | str:
    from Qwen.inference.vllm_utils import normalize_media_path

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = text + "<|im_start|>assistant\n"
    if os.environ.get("VLLM_FORCE_THINK", "0").strip().lower() in {"1", "true", "yes", "on"}:
        text = text + "<think>"

    images: list[str] = []
    videos: list[str] = []
    min_pixels = None
    max_pixels = None

    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    image_path = item.get("image")
                    if image_path:
                        images.append(normalize_media_path(image_path))
                    if min_pixels is None and "min_pixels" in item:
                        min_pixels = item.get("min_pixels")
                    if max_pixels is None and "max_pixels" in item:
                        max_pixels = item.get("max_pixels")
                elif item.get("type") == "video":
                    video_path = item.get("video")
                    if video_path:
                        videos.append(normalize_media_path(video_path))

    if not images and not videos:
        return text

    if min_pixels is None:
        min_pixels = getattr(processor.image_processor, "min_pixels", 28 * 28 * 256)
    if max_pixels is None:
        max_pixels = getattr(processor.image_processor, "max_pixels", 28 * 28 * 2048)

    mm_data: dict[str, Any] = {}
    if images:
        mm_data["image"] = images
    if videos:
        mm_data["video"] = videos
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    }


def _get_vllm_model_handle(vllm_engine):
    return vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model


def _should_skip_vllm_weight_sync(name: str, peft_prefix: str | None = None) -> bool:
    if "latent_vae" in name:
        return True
    if peft_prefix and peft_prefix in name:
        return True
    if "original_module" in name:
        return True
    return False


def _sync_training_model_to_vllm(accelerator: Accelerator, model, vllm_engine) -> None:
    if vllm_engine is None:
        return

    sleep_enabled = bool(getattr(vllm_engine, "opsd_enable_sleep_mode", False))

    if sleep_enabled and hasattr(vllm_engine, "wake_up"):
        torch.cuda.empty_cache()
        vllm_engine.wake_up()

    llm_model = _get_vllm_model_handle(vllm_engine)
    unwrapped_model = accelerator.unwrap_model(model)

    if PeftModel is not None and isinstance(unwrapped_model, PeftModel):
        with torch.no_grad():
            unwrapped_model.merge_adapter()
            try:
                for name, param in unwrapped_model.named_parameters():
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if _should_skip_vllm_weight_sync(name, peft_prefix=unwrapped_model.prefix):
                        continue
                    name = name.replace("modules_to_save.default.", "")
                    llm_model.load_weights([(name, param.data)])
            finally:
                unwrapped_model.unmerge_adapter()
    else:
        with torch.no_grad():
            for name, param in unwrapped_model.named_parameters():
                if _should_skip_vllm_weight_sync(name):
                    continue
                llm_model.load_weights([(name, param.data)])

    vllm_engine.reset_prefix_cache()
    if sleep_enabled and hasattr(vllm_engine, "sleep"):
        vllm_engine.sleep(level=2)


def _init_vllm_rollout_engine(
    *,
    accelerator: Accelerator,
    processor,
    config: dict[str, Any],
):
    rollout_cfg = config.get("rollout") or {}
    if str(rollout_cfg.get("backend", "vllm")).strip().lower() != "vllm":
        return None, None, None, []
    if not bool(rollout_cfg.get("enable", True)):
        return None, None, None, []

    _prepare_vllm_env(config)
    rollout_tokenizer_dir = _prepare_rollout_tokenizer_dir(config, accelerator)

    from vllm import LLM, SamplingParams

    tp_size = int(rollout_cfg.get("tensor_parallel_size", 1))
    if accelerator.num_processes % tp_size != 0:
        raise ValueError(
            f"rollout.tensor_parallel_size ({tp_size}) must divide world size ({accelerator.num_processes})"
        )

    if tp_size > 1:
        torch.distributed.new_subgroups_by_enumeration(
            [
                list(range(i * tp_size, (i + 1) * tp_size))
                for i in range(accelerator.num_processes // tp_size)
            ]
        )

    os.environ["RANK"] = str(accelerator.process_index)
    os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
    os.environ["WORLD_SIZE"] = str(accelerator.num_processes)

    model_path = config["model"]["model_name_or_path"]
    max_model_len = int(rollout_cfg.get("max_model_len", 0))
    if max_model_len <= 0:
        max_model_len = int(config["training"]["max_length"]) + int(config["generation"]["max_new_tokens"])

    sleep_enabled = bool(rollout_cfg.get("enable_sleep_mode", False))

    llm = LLM(
        model=model_path,
    tokenizer=str(rollout_tokenizer_dir),
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=float(rollout_cfg.get("gpu_memory_utilization", 0.35)),
        max_num_seqs=int(config["training"]["per_device_train_batch_size"]),
        max_model_len=max_model_len,
        trust_remote_code=True,
        distributed_executor_backend="external_launcher",
        seed=accelerator.process_index // tp_size,
        enable_sleep_mode=bool(rollout_cfg.get("enable_sleep_mode", True)),
        limit_mm_per_prompt={"image": int(rollout_cfg.get("max_images_per_prompt", 10))},
        enforce_eager=bool(rollout_cfg.get("enforce_eager", False)),
        disable_custom_all_reduce=True,
    )
    llm.opsd_enable_sleep_mode = sleep_enabled
    if sleep_enabled and hasattr(llm, "sleep"):
        llm.sleep(level=2)

    rollout_tokenizer = AutoTokenizer.from_pretrained(str(rollout_tokenizer_dir), trust_remote_code=True)
    blocked_rollout_token_ids = _forbidden_rollout_token_ids(rollout_tokenizer)

    sampling_params = SamplingParams(
        n=1,
        temperature=float(config["generation"].get("temperature", 1.0)),
        top_p=float(config["generation"].get("top_p", 1.0)),
        top_k=int(config["generation"].get("top_k", 20)),
        repetition_penalty=float(config["generation"].get("repetition_penalty", 1.0)),
        max_tokens=int(config["generation"]["max_new_tokens"]),
        skip_special_tokens=False,
        logit_bias={token_id: -100.0 for token_id in blocked_rollout_token_ids},
    )
    accelerator.wait_for_everyone()
    return llm, sampling_params, processor, blocked_rollout_token_ids


def _sanitize_rollout_completion_ids(
    completion_ids: torch.Tensor,
    blocked_token_ids: set[int],
) -> tuple[torch.Tensor, int]:
    if completion_ids.numel() == 0 or not blocked_token_ids:
        return completion_ids, 0

    keep_mask = torch.ones_like(completion_ids, dtype=torch.bool)
    for token_id in blocked_token_ids:
        keep_mask &= completion_ids != token_id

    removed = int((~keep_mask).sum().item())
    if removed == 0:
        return completion_ids, 0
    return completion_ids[keep_mask], removed


def _generate_with_vllm(
    *,
    vllm_engine,
    sampling_params,
    processor,
    metadata: list[dict[str, Any]],
    student_prompt_input_ids: torch.Tensor,
    pad_token_id: int,
    device: torch.device,
    blocked_rollout_token_ids: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    sleep_enabled = bool(getattr(vllm_engine, "opsd_enable_sleep_mode", False))
    if sleep_enabled and hasattr(vllm_engine, "wake_up"):
        torch.cuda.empty_cache()
        vllm_engine.wake_up()

    inputs = [_prepare_messages_for_vllm(item["student_vllm_messages"], processor) for item in metadata]
    outputs = vllm_engine.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
    completion_ids: list[torch.Tensor] = []
    removed_blocked_tokens = 0
    for output in outputs:
        completion = torch.tensor(output.outputs[0].token_ids, device=device, dtype=student_prompt_input_ids.dtype)
        completion, removed = _sanitize_rollout_completion_ids(completion, blocked_rollout_token_ids)
        removed_blocked_tokens += removed
        completion_ids.append(completion)
    if removed_blocked_tokens > 0 and os.environ.get("RANK", "0") == "0":
        print(f"[opsd] removed blocked rollout tokens count={removed_blocked_tokens}")
    completion_lengths = torch.tensor(
        [completion.numel() for completion in completion_ids],
        device=device,
        dtype=torch.long,
    )
    local_max_completion_tokens = int(completion_lengths.max().item()) if completion_ids else 0
    padded_completion_ids: list[torch.Tensor] = []
    for completion in completion_ids:
        if completion.numel() < local_max_completion_tokens:
            padding = torch.full(
                (local_max_completion_tokens - completion.numel(),),
                pad_token_id,
                device=device,
                dtype=student_prompt_input_ids.dtype,
            )
            padded_completion_ids.append(torch.cat([completion, padding], dim=0))
        else:
            padded_completion_ids.append(completion)
    if padded_completion_ids:
        generated_ids = torch.cat([student_prompt_input_ids, torch.stack(padded_completion_ids, dim=0)], dim=1)
    else:
        generated_ids = student_prompt_input_ids

    if sleep_enabled and hasattr(vllm_engine, "sleep"):
        vllm_engine.sleep(level=2)
    return generated_ids, completion_lengths


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    processor,
    output_dir: Path,
    step_label: str,
    keep_last_n: int = 0,
    require_vae: bool = False,
) -> None:
    checkpoint_dir = output_dir / step_label
    unwrapped_model = accelerator.unwrap_model(model)
    if require_vae and _resolve_latent_vae_module(unwrapped_model) is None:
        raise RuntimeError(
            f"Expected latent_vae to be present when saving checkpoint {checkpoint_dir}, but none was found."
        )

    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            state_dict=state_dict,
            save_function=accelerator.save,
        )
        processor.save_pretrained(checkpoint_dir)

    save_vae_checkpoint(model, str(checkpoint_dir))

    if not accelerator.is_main_process:
        return

    if require_vae and not (checkpoint_dir / "vae.safetensors").exists():
        raise RuntimeError(f"Expected VAE checkpoint at {checkpoint_dir / 'vae.safetensors'}, but it was not written.")

    latest_dir = output_dir / "checkpoint_latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    shutil.copytree(checkpoint_dir, latest_dir)

    if keep_last_n > 0:
        numbered_checkpoints: list[tuple[int, Path]] = []
        for path in output_dir.iterdir():
            if not path.is_dir():
                continue
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match is None:
                continue
            numbered_checkpoints.append((int(match.group(1)), path))
        numbered_checkpoints.sort(key=lambda item: item[0])
        stale = numbered_checkpoints[:-keep_last_n]
        for _, stale_path in stale:
            if stale_path.exists():
                shutil.rmtree(stale_path)


def _save_prompt_samples(output_dir: Path, metadata: list[dict[str, Any]], *, step_label: str) -> None:
    sample_path = output_dir / f"{step_label}_prompt_samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _maybe_prepare_manifest(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    manifest_path = Path(data_cfg["opsd_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not bool(data_cfg.get("overwrite_manifest", False)):
        return

    from Qwen.data.build_qwen3vl_opsd_dataset import build_manifest

    datasets = [item.strip() for item in str(data_cfg["dataset_names"]).split(",") if item.strip()]
    total, per_dataset = build_manifest(datasets, manifest_path)
    print(f"[opsd] prepared manifest {manifest_path} rows={total} breakdown={per_dataset}")


def _build_dataloader(config: dict[str, Any], student_processor, teacher_processor):
    train_cfg = config["training"]
    manifest_path = Path(config["data"]["opsd_manifest"])
    dataset = OpsdManifestDataset(manifest_path)
    collator = OpsdVlmCollator(student_processor, teacher_processor, config)
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["per_device_train_batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collator,
        drop_last=bool(train_cfg.get("drop_last", True)),
    )
    return dataset, dataloader


def _run_dry_run(config: dict[str, Any], dry_run_batches: int) -> None:
    config = json.loads(json.dumps(config))
    config.setdefault("training", {})
    config["training"]["num_workers"] = 0

    student_processor = _load_processor_for_model(str(config["model"]["model_name_or_path"]), config)
    teacher_processor = _prepare_teacher_processor(config, student_processor)
    dataset, dataloader = _build_dataloader(config, student_processor, teacher_processor)
    print(f"[opsd] dry-run dataset_size={len(dataset)} manifest={config['data']['opsd_manifest']}")

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        print(
            "[opsd] dry-run "
            f"batch={batch_idx} "
            f"student_input_ids={tuple(batch['student_prompt']['input_ids'].shape)} "
            f"teacher_input_ids={tuple(batch['teacher_prompt']['input_ids'].shape)} "
            f"student_images={tuple(batch['student_prompt'].get('image_grid_thw', torch.empty(0)).shape)} "
            f"teacher_images={tuple(batch['teacher_prompt'].get('image_grid_thw', torch.empty(0)).shape)}"
        )
        if batch_idx == 0:
            _save_prompt_samples(output_dir, batch["metadata"], step_label="dry_run")
        if batch_idx + 1 >= dry_run_batches:
            break


def _run_training(config: dict[str, Any]) -> None:
    train_cfg = config["training"]
    _configure_quiet_logging()
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        mixed_precision="bf16" if bool(train_cfg.get("bf16", True)) else "no",
    )

    _set_seed(int(train_cfg.get("seed", 42)))
    _prepare_runtime_env(config)

    processor = _load_processor_for_model(str(config["model"]["model_name_or_path"]), config)
    teacher_processor = _prepare_teacher_processor(config, processor)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Qwen3-VL processor did not expose a tokenizer.")

    config.setdefault("tokens", {})
    config["tokens"]["pad_token_id"] = tokenizer.pad_token_id

    model, model_dtype, require_vae_checkpoints = _prepare_model(config, accelerator)
    teacher_model, teacher_model_dtype = _prepare_teacher_model(config, accelerator)
    dataset, dataloader = _build_dataloader(config, processor, teacher_processor)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    num_update_steps_per_epoch = max(
        1,
        math.ceil(len(dataloader) / int(train_cfg["gradient_accumulation_steps"])),
    )
    configured_epochs = float(train_cfg.get("num_train_epochs", 0))
    configured_max_steps = int(train_cfg.get("max_steps", -1))
    use_epoch_budget = configured_epochs > 0
    if use_epoch_budget:
        total_steps = int(math.ceil(configured_epochs * num_update_steps_per_epoch))
    else:
        total_steps = configured_max_steps
        if total_steps <= 0:
            raise ValueError("Either training.num_train_epochs must be > 0 or training.max_steps must be > 0.")
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.0)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    generation_config = _build_generation_config(config)
    teacher_cfg = config["teacher"]
    loss_cfg = config["loss"]
    rollout_cfg = config.get("rollout") or {}
    rollout_sync_steps = max(1, int(rollout_cfg.get("sync_steps", 1)))
    rollout_engine, rollout_sampling_params, rollout_processor, blocked_rollout_token_ids = _init_vllm_rollout_engine(
        accelerator=accelerator,
        processor=processor,
        config=config,
    )
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(output_dir / "resolved_opsd_config.yaml", config)

    if accelerator.is_main_process:
        print(
            f"[opsd] teacher strategy={_teacher_strategy(config)} "
            f"teacher_prompt_processor={'student' if teacher_processor is processor else 'separate'}"
        )
        if use_epoch_budget and configured_max_steps > 0:
            print(
                f"[opsd] ignoring training.max_steps={configured_max_steps} "
                f"because training.num_train_epochs={configured_epochs:g} is set"
            )
        budget_mode = (
            f"epochs={configured_epochs:g}"
            if use_epoch_budget
            else f"max_steps={configured_max_steps}"
        )
        print(
            f"[opsd] dataset_size={len(dataset)} batch_size={train_cfg['per_device_train_batch_size']} "
            f"grad_accum={train_cfg['gradient_accumulation_steps']} total_steps={total_steps} "
            f"budget={budget_mode}"
        )

    global_step = 0
    start_time = time.time()
    step_stage_seconds = {
        "rollout": 0.0,
        "student": 0.0,
        "teacher": 0.0,
        "opt": 0.0,
        "sync": 0.0,
    }
    step_length_stats = {
        "student_prompt_max": 0,
        "teacher_prompt_max": 0,
        "completion_max": 0,
        "completion_tokens": 0,
    }
    last_loss_value = 0.0
    last_loss_metrics: dict[str, float] = {}
    model.train()
    if rollout_engine is not None:
        _sync_training_model_to_vllm(accelerator, model, rollout_engine)
        accelerator.wait_for_everyone()

    for epoch in range(int(train_cfg["num_train_epochs"])):
        for batch in dataloader:
            with accelerator.accumulate(model):
                student_prompt = _move_batch_to_device(batch["student_prompt"], accelerator.device, model_dtype)
                teacher_prompt = _move_batch_to_device(
                    batch["teacher_prompt"],
                    accelerator.device,
                    teacher_model_dtype or model_dtype,
                )

                stage_start = time.perf_counter()
                if rollout_engine is not None:
                    generated_ids, completion_lengths = _generate_with_vllm(
                        vllm_engine=rollout_engine,
                        sampling_params=rollout_sampling_params,
                        processor=rollout_processor,
                        metadata=batch["metadata"],
                        student_prompt_input_ids=student_prompt["input_ids"],
                        pad_token_id=int(config["tokens"]["pad_token_id"]),
                        device=accelerator.device,
                        blocked_rollout_token_ids=set(blocked_rollout_token_ids),
                    )
                else:
                    generation_inputs = {
                        "input_ids": student_prompt["input_ids"],
                        "attention_mask": student_prompt["attention_mask"],
                        "pixel_values": student_prompt.get("pixel_values"),
                        "image_grid_thw": student_prompt.get("image_grid_thw"),
                        "generation_config": generation_config,
                        "synced_gpus": accelerator.num_processes > 1,
                    }

                    model.eval()
                    with torch.no_grad():
                        generated_ids = accelerator.unwrap_model(model).generate(**generation_inputs)
                    model.train()
                    completion_lengths = torch.full(
                        (generated_ids.shape[0],),
                        generated_ids.shape[1] - student_prompt["input_ids"].shape[1],
                        device=generated_ids.device,
                        dtype=torch.long,
                    )
                step_stage_seconds["rollout"] += time.perf_counter() - stage_start

                student_prompt_len = int(student_prompt["prompt_length"])
                teacher_prompt_len = int(teacher_prompt["prompt_length"])

                generation_ids = generated_ids[:, student_prompt_len:]
                generation_mask = (
                    torch.arange(generation_ids.shape[1], device=generation_ids.device).unsqueeze(0)
                    < completion_lengths.unsqueeze(1)
                ).long()
                completion_width = int(generation_ids.shape[1])
                step_length_stats["student_prompt_max"] = max(step_length_stats["student_prompt_max"], student_prompt_len)
                step_length_stats["teacher_prompt_max"] = max(step_length_stats["teacher_prompt_max"], teacher_prompt_len)
                step_length_stats["completion_max"] = max(step_length_stats["completion_max"], int(completion_lengths.max().item()))
                step_length_stats["completion_tokens"] += int(completion_lengths.sum().item())

                student_input_ids = generated_ids
                student_attention_mask = torch.cat(
                    [student_prompt["attention_mask"], generation_mask],
                    dim=1,
                )

                teacher_input_ids = torch.cat(
                    [teacher_prompt["input_ids"], generation_ids],
                    dim=1,
                )
                teacher_attention_mask = torch.cat(
                    [teacher_prompt["attention_mask"], generation_mask],
                    dim=1,
                )

                labels = student_input_ids.clone()
                for row_idx, prompt_len in enumerate(student_prompt["prompt_lengths_per_example"].tolist()):
                    labels[row_idx, : int(prompt_len)] = -100
                labels[student_attention_mask == 0] = -100

                sampled_token_ids = student_input_ids[:, student_prompt_len:]
                shifted_labels = labels[:, student_prompt_len:]
                loss_type = str(loss_cfg.get("type", "sampled_reverse_kl")).strip().lower()

                stage_start = time.perf_counter()
                with _opsd_replay_mode():
                    student_outputs = model(
                        input_ids=student_input_ids,
                        attention_mask=student_attention_mask,
                        pixel_values=student_prompt.get("pixel_values"),
                        image_grid_thw=student_prompt.get("image_grid_thw"),
                        logits_to_keep=max(1, completion_width + 1),
                        use_cache=False,
                    )
                step_stage_seconds["student"] += time.perf_counter() - stage_start
                student_logits = student_outputs.logits[:, :-1, :]
                student_log_probs_sampled = None
                if loss_type == "sampled_reverse_kl":
                    student_log_probs_sampled = _sampled_log_probs_from_logits(
                        logits=student_logits,
                        sampled_token_ids=sampled_token_ids,
                        temperature=float(loss_cfg["temperature"]),
                    )
                    del student_logits
                    student_logits = None
                del student_outputs

                teacher_strategy = str(teacher_cfg.get("strategy", "fixed")).strip().lower()
                active_teacher_model = teacher_model
                if teacher_strategy == "separate":
                    if active_teacher_model is None:
                        raise RuntimeError("teacher.strategy=separate requires an external teacher model.")
                    teacher_context = nullcontext()
                else:
                    active_teacher_model = accelerator.unwrap_model(model)
                    if (
                        teacher_strategy == "fixed"
                        and PeftModel is not None
                        and isinstance(active_teacher_model, PeftModel)
                    ):
                        teacher_context = active_teacher_model.disable_adapter()
                    elif teacher_strategy in {"fixed", "current"}:
                        teacher_context = nullcontext()
                    else:
                        raise ValueError(f"Unsupported teacher strategy: {teacher_strategy}")

                stage_start = time.perf_counter()
                with torch.no_grad(), teacher_context, _opsd_replay_mode():
                    teacher_outputs = active_teacher_model(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask,
                        pixel_values=teacher_prompt.get("pixel_values"),
                        image_grid_thw=teacher_prompt.get("image_grid_thw"),
                        logits_to_keep=max(1, completion_width + 1),
                        use_cache=False,
                    )
                    teacher_logits = teacher_outputs.logits[:, :-1, :]
                step_stage_seconds["teacher"] += time.perf_counter() - stage_start
                teacher_log_probs_sampled = None
                if loss_type == "sampled_reverse_kl":
                    teacher_log_probs_sampled = _sampled_log_probs_from_logits(
                        logits=teacher_logits,
                        sampled_token_ids=sampled_token_ids,
                        temperature=float(loss_cfg["temperature"]),
                    )
                    del teacher_logits
                    teacher_logits = None
                del teacher_outputs

                if loss_type == "sampled_reverse_kl":
                    loss, loss_metrics = _sampled_reverse_kl_loss_from_logprobs(
                        student_log_probs_sampled=student_log_probs_sampled,
                        teacher_log_probs_sampled=teacher_log_probs_sampled,
                        labels=shifted_labels,
                    )
                    del student_log_probs_sampled
                    del teacher_log_probs_sampled
                elif loss_type == "jsd":
                    loss = _generalized_jsd_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        labels=shifted_labels,
                        beta=float(loss_cfg.get("beta", 0.5)),
                        temperature=float(loss_cfg["temperature"]),
                        top_k=int(loss_cfg.get("top_k", 0)),
                        token_clip=float(loss_cfg.get("token_clip", 0.0)),
                    )
                    loss_metrics = {}
                else:
                    raise ValueError(f"Unsupported OPSD loss type: {loss_type}")

                stage_start = time.perf_counter()
                accelerator.backward(loss)
                if float(train_cfg.get("max_grad_norm", 0.0)) > 0:
                    accelerator.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_stage_seconds["opt"] += time.perf_counter() - stage_start
                last_loss_value = float(loss.detach())
                last_loss_metrics = dict(loss_metrics)

                del completion_lengths
                del generated_ids
                del generation_ids
                del generation_mask
                del student_input_ids
                del student_attention_mask
                del teacher_input_ids
                del teacher_attention_mask
                del labels
                del sampled_token_ids
                del shifted_labels
                del student_prompt
                del teacher_prompt
                del loss

            if accelerator.sync_gradients:
                if rollout_engine is not None:
                    if global_step == 0 or (global_step + 1) % rollout_sync_steps == 0:
                        stage_start = time.perf_counter()
                        _sync_training_model_to_vllm(accelerator, model, rollout_engine)
                        accelerator.wait_for_everyone()
                        step_stage_seconds["sync"] += time.perf_counter() - stage_start
                global_step += 1

                if accelerator.is_main_process and global_step % int(train_cfg["logging_steps"]) == 0:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    progress = float(global_step) / float(max(1, total_steps))
                    remaining_steps = max(0, total_steps - global_step)
                    seconds_per_step = elapsed / float(max(1, global_step))
                    eta_seconds = seconds_per_step * remaining_steps
                    metric_suffix = ""
                    if last_loss_metrics:
                        metric_suffix = (
                            f" advantage={last_loss_metrics['advantage']:.4f}"
                            f" student_lp={last_loss_metrics['student_logprob']:.4f}"
                            f" teacher_lp={last_loss_metrics['teacher_logprob']:.4f}"
                        )
                    print(
                        f"[opsd] step={global_step}/{total_steps} progress={progress * 100.0:.1f}% "
                        f"epoch={epoch + 1}/{max(1, int(math.ceil(configured_epochs))) if configured_epochs > 0 else 1} "
                        f"loss={last_loss_value:.6f} lr={lr:.3e} "
                        f"elapsed={elapsed / 3600.0:.2f}h eta={eta_seconds / 3600.0:.2f}h{metric_suffix} "
                        f"lengths=student_prompt_max:{step_length_stats['student_prompt_max']} "
                        f"teacher_prompt_max:{step_length_stats['teacher_prompt_max']} "
                        f"completion_max:{step_length_stats['completion_max']} "
                        f"completion_tokens:{step_length_stats['completion_tokens']} "
                        f"timing=rollout:{step_stage_seconds['rollout']:.1f}s "
                        f"student:{step_stage_seconds['student']:.1f}s "
                        f"teacher:{step_stage_seconds['teacher']:.1f}s "
                        f"opt:{step_stage_seconds['opt']:.1f}s "
                        f"sync:{step_stage_seconds['sync']:.1f}s"
                    )
                if global_step % int(train_cfg["logging_steps"]) == 0:
                    step_stage_seconds = {key: 0.0 for key in step_stage_seconds}
                    step_length_stats = {key: 0 for key in step_length_stats}

                if accelerator.is_main_process and global_step == 1:
                    _save_prompt_samples(output_dir, batch["metadata"], step_label="step_1")
                if global_step == 1:
                    accelerator.wait_for_everyone()

                if global_step % int(train_cfg["save_steps"]) == 0:
                    accelerator.wait_for_everyone()
                    save_started_at = time.perf_counter()
                    if accelerator.is_main_process:
                        print(f"[opsd] checkpoint_save_start step={global_step}")
                    _save_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        processor=processor,
                        output_dir=output_dir,
                        step_label=f"checkpoint-{global_step}",
                        keep_last_n=int(train_cfg.get("save_total_limit", 0)),
                        require_vae=require_vae_checkpoints,
                    )
                    if accelerator.is_main_process:
                        print(
                            f"[opsd] checkpoint_save_done step={global_step} "
                            f"seconds={time.perf_counter() - save_started_at:.1f}"
                        )
                    accelerator.wait_for_everyone()

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    _save_checkpoint(
        accelerator=accelerator,
        model=model,
        processor=processor,
        output_dir=output_dir,
        step_label="checkpoint-final",
        keep_last_n=int(train_cfg.get("save_total_limit", 0)),
        require_vae=require_vae_checkpoints,
    )


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))

    if args.datasets:
        config.setdefault("data", {})
        config["data"]["dataset_names"] = args.datasets
    if args.manifest_path:
        config.setdefault("data", {})
        config["data"]["opsd_manifest"] = args.manifest_path
    if args.output_dir:
        config.setdefault("training", {})
        config["training"]["output_dir"] = args.output_dir
    if args.ckpt_path:
        config.setdefault("model", {})
        config["model"]["adapter_name_or_path"] = args.ckpt_path
    if args.overwrite_manifest:
        config.setdefault("data", {})
        config["data"]["overwrite_manifest"] = True
    return config


def _resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    runtime_env_config = _resolve_path(config.get("runtime_env_config"), base_dir=REPO_ROOT)
    if runtime_env_config is not None:
        config["runtime_env_config"] = str(runtime_env_config)

    manifest_value = config["data"].get("opsd_manifest")
    if manifest_value:
        manifest_path = _resolve_path(manifest_value, base_dir=REPO_ROOT)
    else:
        manifest_path = _default_manifest_path(str(config["data"]["dataset_names"]))
    config["data"]["opsd_manifest"] = str(manifest_path)

    output_dir = _resolve_path(config["training"]["output_dir"], base_dir=REPO_ROOT)
    config["training"]["output_dir"] = str(output_dir)

    adapter_path = _resolve_path(config["model"].get("adapter_name_or_path"), base_dir=REPO_ROOT)
    if adapter_path is not None:
        config["model"]["adapter_name_or_path"] = str(adapter_path)

    model_path = _resolve_path(config["model"]["model_name_or_path"], base_dir=REPO_ROOT)
    config["model"]["model_name_or_path"] = str(model_path)

    teacher_cfg = config.setdefault("teacher", {})
    teacher_model_path = _resolve_path(teacher_cfg.get("model_name_or_path"), base_dir=REPO_ROOT)
    if teacher_model_path is not None:
        teacher_cfg["model_name_or_path"] = str(teacher_model_path)
    teacher_processor_path = _resolve_path(teacher_cfg.get("processor_name_or_path"), base_dir=REPO_ROOT)
    if teacher_processor_path is not None:
        teacher_cfg["processor_name_or_path"] = str(teacher_processor_path)
    teacher_adapter_path = _resolve_path(teacher_cfg.get("adapter_name_or_path"), base_dir=REPO_ROOT)
    if teacher_adapter_path is not None:
        teacher_cfg["adapter_name_or_path"] = str(teacher_adapter_path)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL OPSD with continuous thinking.")
    parser.add_argument("config", help="YAML config path")
    parser.add_argument("--datasets", help="Override config data.dataset_names")
    parser.add_argument("--manifest-path", help="Override config data.opsd_manifest")
    parser.add_argument("--output-dir", help="Override config training.output_dir")
    parser.add_argument("--ckpt-path", help="Override config model.adapter_name_or_path")
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Force regeneration of the OPSD manifest JSONL",
    )
    parser.add_argument(
        "--dry-run-batches",
        type=int,
        default=0,
        help="Build dataloader batches and save prompt samples without training",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config, base_dir=REPO_ROOT)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = _load_yaml(config_path)
    config = _apply_overrides(config, args)
    config = _resolve_config_paths(config)
    _maybe_prepare_manifest(config)

    if int(args.dry_run_batches) > 0:
        _run_dry_run(config, int(args.dry_run_batches))
        return 0

    _run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
