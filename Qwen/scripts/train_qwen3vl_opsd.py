#!/usr/bin/env python3
"""Privileged OPSD trainer for Qwen3-VL.

This trainer uses two teacher replay paths:

1. Privileged teacher-on-student replay:
   - student samples an on-policy completion
   - teacher sees privileged solution context in the prompt
   - teacher replays the exact student completion token IDs
   - token-level distillation is applied on the full completion sequence

2. Optional second teacher replay for OT:
   - teacher uses the student prompt
   - teacher replays the dataset ground-truth assistant target
   - the extracted teacher thinking hidden states become OT targets
   - OT aligns those targets to the student's continuous thinking rollout
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dependency
    PeftModel = None

from Qwen.llamafactory.integration import _compute_ot_loss, _find_latent_positions, _parse_loss_spec
from Qwen.scripts.train_qwen3vl_opd_vcr import _configure_deepspeed_runtime
from Qwen.scripts.qwen3vl_opsd_common import (
    OpsdManifestDataset,
    _build_generation_config,
    _configure_quiet_logging,
    _format_user_prompt,
    _generalized_jsd_loss,
    _generate_completions_with_vllm,
    _generate_with_vllm,
    _init_vllm_rollout_engine,
    _load_image,
    _load_processor_for_model,
    _load_yaml,
    _maybe_prepare_manifest,
    _move_batch_to_device,
    _prepare_model,
    _prepare_runtime_env,
    _prepare_teacher_model,
    _prepare_teacher_processor,
    _opsd_replay_mode,
    _resolve_config_paths,
    _resolve_path,
    _sampled_log_probs_from_logits,
    _sampled_reverse_kl_loss_from_logprobs,
    _save_checkpoint,
    _set_seed,
    _sync_training_model_to_vllm,
    _teacher_strategy,
    _write_yaml,
)


def _strip_text(text: Any) -> str:
    return str(text or "").strip()


def _build_user_messages(
    *,
    system_prompt: str,
    user_text: str,
    question_images: list[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if question_images:
        content.append({"type": "text", "text": "Image(s):"})
        content.extend({"type": "image", "image": image_path} for image_path in question_images)
    content.append({"type": "text", "text": user_text})

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


class OpsdCollator:
    def __init__(self, student_processor, teacher_processor, config: dict[str, Any]):
        self.student_processor = student_processor
        self.teacher_processor = teacher_processor
        self.system_prompt = _strip_text(config.get("system_prompt"))
        self.student_template = str(config["prompts"]["student_user"])
        self.teacher_template = str(config["prompts"]["teacher_user"])

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        student_texts: list[str] = []
        student_images: list[list[Any]] = []
        metadata: list[dict[str, Any]] = []

        for feature in features:
            question_paths = list(feature.get("question_images") or [])
            answer_text = _strip_text(feature.get("answer_text"))
            question_text = _strip_text(feature.get("student_user_text"))
            teacher_solution_text = _strip_text(feature.get("teacher_solution_text"))
            teacher_assistant_target = _strip_text(feature.get("teacher_assistant_target"))

            student_user = _format_user_prompt(
                self.student_template,
                num_question_images=len(question_paths),
                num_rationale_images=0,
                question_text=question_text,
            )
            teacher_user = _format_user_prompt(
                self.teacher_template,
                num_question_images=len(question_paths),
                num_rationale_images=0,
                question_text=question_text,
                reference_solution=teacher_solution_text,
            )

            student_messages = _build_user_messages(
                system_prompt=self.system_prompt,
                user_text=student_user,
                question_images=question_paths,
            )
            teacher_messages = _build_user_messages(
                system_prompt=self.system_prompt,
                user_text=teacher_user,
                question_images=question_paths,
            )
            student_vllm_messages = copy.deepcopy(student_messages)

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
            student_images.append([_load_image(path) for path in question_paths])
            metadata.append(
                {
                    "sample_id": feature.get("sample_id"),
                    "source_dataset": feature.get("source_dataset"),
                    "task": feature.get("task"),
                    "question_text": question_text,
                    "question_images": question_paths,
                    "answer_text": answer_text,
                    "teacher_solution_text": teacher_solution_text,
                    "teacher_assistant_target": teacher_assistant_target,
                    "latent_ground_truth_paths": list(feature.get("latent_ground_truth") or []),
                    "latent_supervision_paths": list(feature.get("latent_supervision") or []),
                    "student_vllm_messages": student_vllm_messages,
                    "student_messages": student_messages,
                    "teacher_messages": teacher_messages,
                    "student_prompt_text": student_prompt_text,
                    "teacher_prompt_text": teacher_prompt_text,
                }
            )

        student_prompt = self.student_processor(
            text=student_texts,
            images=student_images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        student_prompt["prompt_lengths_per_example"] = student_prompt["attention_mask"].sum(dim=1)
        student_prompt["prompt_length"] = int(student_prompt["input_ids"].shape[1])
        return {"student_prompt": student_prompt, "metadata": metadata}


def _build_dataloader(config: dict[str, Any], student_processor, teacher_processor):
    train_cfg = config["training"]
    dataset = OpsdManifestDataset(Path(config["data"]["opsd_manifest"]))
    collator = OpsdCollator(student_processor, teacher_processor, config)
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["per_device_train_batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collator,
        drop_last=bool(train_cfg.get("drop_last", True)),
    )
    return dataset, dataloader


_MANUAL_LOSS_NAMES = {"opsd", "ot_replay"}


def _ot_replay_config(config: dict[str, Any]) -> dict[str, Any]:
    ot_cfg = copy.deepcopy(config.get("ot_replay") or {})
    ot_cfg.setdefault("enabled", False)
    return ot_cfg


def _resolve_text_model(model):
    module = getattr(model, "module", None)
    candidates = [
        getattr(getattr(module, "model", None), "language_model", None),
        getattr(module, "language_model", None),
        getattr(getattr(getattr(module, "base_model", None), "model", None), "language_model", None),
        getattr(getattr(getattr(getattr(module, "base_model", None), "model", None), "model", None), "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "language_model", None),
        getattr(getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None), "language_model", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError("Failed to resolve the Qwen3-VL text model for hidden-state capture.")


@contextmanager
def _temporary_eval(model):
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def _forward_with_last_hidden(model, **model_kwargs) -> tuple[Any, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    text_model = _resolve_text_model(model)

    def _hook(_module, _inputs, output):
        if hasattr(output, "last_hidden_state"):
            captured["hidden"] = output.last_hidden_state
        elif isinstance(output, (tuple, list)) and output:
            captured["hidden"] = output[0]
        else:
            captured["hidden"] = output

    handle = text_model.register_forward_hook(_hook)
    try:
        outputs = model(**model_kwargs)
    finally:
        handle.remove()

    hidden_states = captured.get("hidden")
    if hidden_states is None:
        raise RuntimeError("Failed to capture last hidden states from the Qwen3-VL text model.")
    return outputs, hidden_states


def _thinking_token_span(token_ids: list[int], *, start_id: int, end_id: int) -> tuple[int, int] | None:
    start_idx = None
    for idx, token_id in enumerate(token_ids):
        if token_id == start_id:
            start_idx = idx
            break
    if start_idx is None:
        return None

    for idx in range(start_idx + 1, len(token_ids)):
        if token_ids[idx] == end_id:
            return start_idx, idx
    return None


def _teacher_forward_target(config: dict[str, Any], accelerator: Accelerator, model, teacher_model):
    strategy = _teacher_strategy(config)
    if strategy == "separate":
        if teacher_model is None:
            raise RuntimeError("teacher.strategy=separate requires an external teacher model.")
        return teacher_model, nullcontext()

    active_teacher_model = accelerator.unwrap_model(model)
    if (
        strategy == "fixed"
        and PeftModel is not None
        and isinstance(active_teacher_model, PeftModel)
    ):
        return active_teacher_model, active_teacher_model.disable_adapter()
    if strategy in {"fixed", "current"}:
        return active_teacher_model, nullcontext()
    raise ValueError(f"Unsupported teacher strategy: {strategy}")


def _load_question_images(batch_metadata: list[dict[str, Any]]) -> list[list[Any]]:
    return [[_load_image(path) for path in item["question_images"]] for item in batch_metadata]


def _tokenize_prompt_messages(
    processor,
    messages_batch: list[list[dict[str, Any]]],
    question_images_batch: list[list[Any]],
):
    prompt_texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    prompt_batch = processor(
        text=prompt_texts,
        images=question_images_batch,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    prompt_batch["prompt_length"] = int(prompt_batch["input_ids"].shape[1])
    prompt_batch["prompt_lengths_per_example"] = prompt_batch["attention_mask"].sum(dim=1)
    return prompt_batch


def _tokenize_full_sequences(
    processor,
    messages_batch: list[list[dict[str, Any]]],
    assistant_targets: list[str],
    question_images_batch: list[list[Any]],
):
    full_texts = [
        processor.apply_chat_template(
            messages + [{"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for messages, target in zip(messages_batch, assistant_targets)
    ]
    return processor(
        text=full_texts,
        images=question_images_batch,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def _generate_student_rollout(
    *,
    model,
    accelerator: Accelerator,
    student_prompt: dict[str, Any],
    generation_config,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generation_inputs = {
        "input_ids": student_prompt["input_ids"],
        "attention_mask": student_prompt["attention_mask"],
        "pixel_values": student_prompt.get("pixel_values"),
        "image_grid_thw": student_prompt.get("image_grid_thw"),
        "generation_config": generation_config,
        "synced_gpus": accelerator.num_processes > 1,
    }

    active_student_model = accelerator.unwrap_model(model)
    with torch.no_grad(), _temporary_eval(active_student_model):
        generated_ids = active_student_model.generate(**generation_inputs)

    generated_attention_mask = torch.ones_like(generated_ids, dtype=torch.long)
    generated_attention_mask[generated_ids == pad_token_id] = 0
    return generated_ids, generated_attention_mask


def _build_generation_mask_from_lengths(
    *,
    generation_ids: torch.Tensor,
    completion_lengths: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.arange(generation_ids.shape[1], device=generation_ids.device).unsqueeze(0)
        < completion_lengths.unsqueeze(1)
    ).long()


def _compose_vllm_rollout_from_completion(
    *,
    student_prompt: dict[str, Any],
    completion_ids: torch.Tensor,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = student_prompt["input_ids"]
    prompt_mask = student_prompt["attention_mask"]
    completion_ids = completion_ids.to(device=device, dtype=prompt_ids.dtype).unsqueeze(0)
    completion_mask = torch.ones_like(completion_ids, dtype=prompt_mask.dtype, device=device)
    full_generated_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    full_generated_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    if completion_ids.numel() == 0:
        pad = torch.full((1, 1), pad_token_id, device=device, dtype=prompt_ids.dtype)
        full_generated_ids = torch.cat([prompt_ids, pad], dim=1)
        full_generated_attention_mask = torch.cat([prompt_mask, torch.zeros_like(pad)], dim=1)
    return full_generated_ids, full_generated_attention_mask


def _build_teacher_student_replay_batch(
    *,
    teacher_processor,
    metadata: list[dict[str, Any]],
    generation_ids: torch.Tensor,
    generation_mask: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    question_images = _load_question_images(metadata)
    prompt_batch = _tokenize_prompt_messages(
        teacher_processor,
        [item["teacher_messages"] for item in metadata],
        question_images,
    )
    prompt_batch = _move_batch_to_device(prompt_batch, device, model_dtype)
    teacher_input_ids = torch.cat([prompt_batch["input_ids"], generation_ids], dim=1)
    teacher_attention_mask = torch.cat([prompt_batch["attention_mask"], generation_mask], dim=1)
    return prompt_batch, teacher_input_ids, teacher_attention_mask


def _extract_teacher_student_replay_hidden_sequences(
    *,
    teacher_hidden: torch.Tensor,
    teacher_prompt_batch: dict[str, Any],
    generation_ids: torch.Tensor,
    generation_mask: torch.Tensor,
) -> list[torch.Tensor | None]:
    thinking_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    thinking_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))

    sequences: list[torch.Tensor | None] = []
    batch_size = int(generation_ids.shape[0])
    for batch_idx in range(batch_size):
        prompt_len = int(teacher_prompt_batch["attention_mask"][batch_idx].sum().item())
        completion_len = int(generation_mask[batch_idx].sum().item())
        assistant_ids = generation_ids[batch_idx, :completion_len].tolist()
        span = _thinking_token_span(
            assistant_ids,
            start_id=thinking_start_id,
            end_id=thinking_end_id,
        )
        if span is None:
            sequences.append(None)
            continue

        start_idx, end_idx = span
        thinking_len = end_idx - start_idx - 1
        if thinking_len <= 0:
            sequences.append(None)
            continue

        predictor_start = prompt_len + start_idx
        predictor_end = predictor_start + thinking_len
        sequence = teacher_hidden[batch_idx, predictor_start:predictor_end].detach()
        sequences.append(sequence if sequence.numel() > 0 else None)
    return sequences


def _extract_teacher_ground_truth_hidden_sequences(
    *,
    teacher_model,
    teacher_context,
    teacher_processor,
    metadata: list[dict[str, Any]],
    device: torch.device,
    model_dtype: torch.dtype,
    ot_replay_cfg: dict[str, Any],
) -> list[torch.Tensor | None]:
    thinking_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    thinking_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))

    question_images = _load_question_images(metadata)
    replay_messages = [item["student_messages"] for item in metadata]
    prompt_batch = _tokenize_prompt_messages(
        teacher_processor,
        replay_messages,
        question_images,
    )
    full_batch = _tokenize_full_sequences(
        teacher_processor,
        replay_messages,
        [_strip_text(item.get("teacher_assistant_target")) for item in metadata],
        question_images,
    )
    prompt_batch = _move_batch_to_device(prompt_batch, device, model_dtype)
    full_batch = _move_batch_to_device(full_batch, device, model_dtype)

    with torch.no_grad(), teacher_context, _temporary_eval(teacher_model):
        with _opsd_replay_mode():
            _, teacher_hidden = _forward_with_last_hidden(
                teacher_model,
                input_ids=full_batch["input_ids"],
                attention_mask=full_batch["attention_mask"],
                pixel_values=full_batch.get("pixel_values"),
                image_grid_thw=full_batch.get("image_grid_thw"),
                use_cache=False,
            )

    sequences: list[torch.Tensor | None] = []
    for batch_idx, _item in enumerate(metadata):
        prompt_len = int(prompt_batch["attention_mask"][batch_idx].sum().item())
        full_len = int(full_batch["attention_mask"][batch_idx].sum().item())
        assistant_ids = full_batch["input_ids"][batch_idx, prompt_len:full_len].tolist()
        span = _thinking_token_span(
            assistant_ids,
            start_id=thinking_start_id,
            end_id=thinking_end_id,
        )
        if span is None:
            sequences.append(None)
            continue

        start_idx, end_idx = span
        thinking_len = end_idx - start_idx - 1
        if thinking_len <= 0:
            sequences.append(None)
            continue

        assistant_offset = prompt_len
        predictor_start = assistant_offset + start_idx
        predictor_end = predictor_start + thinking_len
        sequence = teacher_hidden[batch_idx, predictor_start:predictor_end].detach()
        sequences.append(sequence if sequence.numel() > 0 else None)
    return sequences


def _loss_config(config: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = copy.deepcopy(config.get("loss") or {})
    loss_cfg.setdefault("type", "opsd")
    loss_cfg.setdefault("token_type", "sampled_reverse_kl")
    loss_cfg.setdefault("temperature", 1.0)
    loss_cfg.setdefault("beta", 0.5)
    loss_cfg.setdefault("top_k", 0)
    loss_cfg.setdefault("token_clip", 0.05)
    return loss_cfg


def _split_loss_config(loss_cfg: dict[str, Any]) -> list[tuple[str, float]]:
    loss_type = str(loss_cfg.get("type", "opsd")).strip().lower()
    parsed = _parse_loss_spec(loss_type)
    unknown = [name for name, _weight in parsed if name not in _MANUAL_LOSS_NAMES]
    if unknown:
        raise ValueError(
            f"Unsupported loss terms in loss.type: {unknown!r}. "
            "OPSD only supports 'opsd' and optional 'ot_replay'."
        )
    return [(name, float(weight)) for name, weight in parsed if name in _MANUAL_LOSS_NAMES]


def _loss_term_weight(loss_configs: list[tuple[str, float]], name: str) -> float:
    for loss_name, weight in loss_configs:
        if loss_name == name:
            return float(weight)
    return 0.0


def _prepare_opsd_runtime_env() -> None:
    os.environ["QWEN3VL_LOSS_TYPE"] = "none"
    os.environ["QWEN3VL_LATENT_CE_ACTIVE"] = "0"
    os.environ["QWEN3VL_LATENT_CE_TOKEN"] = "0"
    os.environ["QWEN3VL_HIDDEN_STATES_HOOK"] = "0"


def _token_distillation_loss(
    *,
    token_loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    generation_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if token_loss_type == "sampled_reverse_kl":
        logprob_chunk_size = int(loss_cfg.get("sampled_logprob_chunk_size", 0))
        student_log_probs_sampled = _sampled_log_probs_from_logits(
            logits=student_logits,
            sampled_token_ids=generation_ids,
            temperature=float(loss_cfg["temperature"]),
            chunk_size=logprob_chunk_size,
        )
        teacher_log_probs_sampled = _sampled_log_probs_from_logits(
            logits=teacher_logits,
            sampled_token_ids=generation_ids,
            temperature=float(loss_cfg["temperature"]),
            chunk_size=logprob_chunk_size,
        )
        return _sampled_reverse_kl_loss_from_logprobs(
            student_log_probs_sampled=student_log_probs_sampled,
            teacher_log_probs_sampled=teacher_log_probs_sampled,
            labels=labels,
        )

    if token_loss_type == "jsd":
        loss = _generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            beta=float(loss_cfg.get("beta", 0.5)),
            temperature=float(loss_cfg["temperature"]),
            top_k=int(loss_cfg.get("top_k", 0)),
            token_clip=float(loss_cfg.get("token_clip", 0.0)),
        )
        return loss, {}

    raise ValueError(f"Unsupported loss.token_type={token_loss_type!r}")


def _safe_artifact_stem(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:120] if len(text) > 120 else text


def _save_hidden_sequence_artifacts(
    *,
    output_dir: Path,
    step_label: str,
    batch_summary: list[dict[str, Any]],
    artifact_name: str,
    sequences: list[torch.Tensor | None] | None,
) -> None:
    if not sequences:
        return
    artifact_dir = output_dir / f"{step_label}_{artifact_name}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for idx, sequence in enumerate(sequences):
        if sequence is None or sequence.numel() == 0 or idx >= len(batch_summary):
            continue
        summary = batch_summary[idx]
        stem = _safe_artifact_stem(summary.get("sample_id"), f"sample_{idx}")
        tensor_path = artifact_dir / f"{stem}.pt"
        torch.save(sequence.detach().cpu(), tensor_path)
        summary[f"{artifact_name}_path"] = str(tensor_path)
        summary[f"{artifact_name}_tokens"] = int(sequence.shape[0])
        summary[f"{artifact_name}_hidden_size"] = int(sequence.shape[-1])


def _save_batch_samples(
    output_dir: Path,
    batch_summary: list[dict[str, Any]],
    step_label: str,
    *,
    teacher_student_hidden_sequences: list[torch.Tensor | None] | None = None,
    teacher_gt_hidden_sequences: list[torch.Tensor | None] | None = None,
) -> None:
    _save_hidden_sequence_artifacts(
        output_dir=output_dir,
        step_label=step_label,
        batch_summary=batch_summary,
        artifact_name="teacher_student_replay_thinking_hidden",
        sequences=teacher_student_hidden_sequences,
    )
    _save_hidden_sequence_artifacts(
        output_dir=output_dir,
        step_label=step_label,
        batch_summary=batch_summary,
        artifact_name="teacher_gt_ot_thinking_hidden",
        sequences=teacher_gt_hidden_sequences,
    )
    sample_path = output_dir / f"{step_label}_opsd_samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for item in batch_summary:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _summarize_batch(
    metadata: list[dict[str, Any]],
    *,
    generation_ids: torch.Tensor,
    generation_mask: torch.Tensor,
    teacher_student_hidden_sequences: list[torch.Tensor | None],
    teacher_hidden_sequences: list[torch.Tensor | None],
    latent_positions: torch.Tensor,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for idx, item in enumerate(metadata):
        teacher_student_hidden = (
            teacher_student_hidden_sequences[idx] if idx < len(teacher_student_hidden_sequences) else None
        )
        teacher_hidden = teacher_hidden_sequences[idx] if idx < len(teacher_hidden_sequences) else None
        summaries.append(
            {
                "sample_id": item.get("sample_id"),
                "task": item.get("task"),
                "question_text": item.get("question_text"),
                "question_images": item.get("question_images"),
                "answer_text": item.get("answer_text"),
                "teacher_solution_text": item.get("teacher_solution_text"),
                "teacher_assistant_target": item.get("teacher_assistant_target"),
                "student_completion_token_count": int(generation_mask[idx].sum().item()),
                "student_latent_token_count": int(latent_positions[idx].sum().item()),
                "teacher_student_replay_target_tokens": (
                    int(teacher_student_hidden.shape[0]) if teacher_student_hidden is not None else 0
                ),
                "teacher_ot_target_tokens": int(teacher_hidden.shape[0]) if teacher_hidden is not None else 0,
            }
        )
    return summaries


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if args.datasets:
        updated.setdefault("data", {})
        updated["data"]["dataset_names"] = args.datasets
    if args.manifest_path:
        updated.setdefault("data", {})
        updated["data"]["opsd_manifest"] = args.manifest_path
    if args.output_dir:
        updated.setdefault("training", {})
        updated["training"]["output_dir"] = args.output_dir
    if args.ckpt_path:
        updated.setdefault("model", {})
        updated["model"]["adapter_name_or_path"] = args.ckpt_path
    if args.overwrite_manifest:
        updated.setdefault("data", {})
        updated["data"]["overwrite_manifest"] = True
    return updated


def _run_dry_run(config: dict[str, Any], dry_run_batches: int) -> None:
    config = copy.deepcopy(config)
    config.setdefault("training", {})
    config["training"]["num_workers"] = 0

    _configure_quiet_logging()
    accelerator = Accelerator(mixed_precision="bf16" if bool(config["training"].get("bf16", True)) else "no")
    _configure_deepspeed_runtime(accelerator, config)
    _set_seed(int(config["training"].get("seed", 42)))
    _prepare_runtime_env(config)
    loss_cfg = _loss_config(config)
    manual_loss_configs = _split_loss_config(loss_cfg)
    _prepare_opsd_runtime_env()
    opsd_weight = _loss_term_weight(manual_loss_configs, "opsd")
    ot_replay_weight = _loss_term_weight(manual_loss_configs, "ot_replay")

    student_processor = _load_processor_for_model(str(config["model"]["model_name_or_path"]), config)
    teacher_processor = _prepare_teacher_processor(config, student_processor)
    tokenizer = getattr(student_processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Qwen3-VL processor did not expose a tokenizer.")

    config.setdefault("tokens", {})
    config["tokens"]["pad_token_id"] = tokenizer.pad_token_id

    model, model_dtype, _ = _prepare_model(config, accelerator)
    teacher_model, teacher_model_dtype = _prepare_teacher_model(config, accelerator)
    rollout_engine, rollout_sampling_params, rollout_processor, blocked_rollout_token_ids = _init_vllm_rollout_engine(
        accelerator=accelerator,
        processor=student_processor,
        config=config,
    )
    if rollout_engine is not None:
        _sync_training_model_to_vllm(accelerator, model, rollout_engine)
        accelerator.wait_for_everyone()
    dataset, dataloader = _build_dataloader(config, student_processor, teacher_processor)
    generation_config = _build_generation_config(config)
    ot_replay_cfg = _ot_replay_config(config)
    enable_teacher_student_replay = opsd_weight > 0.0
    enable_second_teacher_replay = bool(ot_replay_cfg["enabled"]) and ot_replay_weight > 0.0

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[opsd] dry-run dataset_size={len(dataset)} manifest={config['data']['opsd_manifest']}")
    print(
        f"[opsd] dry-run loss_type={loss_cfg['type']} "
        f"opsd_enabled={enable_teacher_student_replay} ot_replay_enabled={enable_second_teacher_replay}"
    )

    for batch_idx, batch in enumerate(dataloader):
        student_prompt = _move_batch_to_device(batch["student_prompt"], accelerator.device, model_dtype)
        if rollout_engine is not None:
            full_generated_ids, completion_lengths = _generate_with_vllm(
                vllm_engine=rollout_engine,
                sampling_params=rollout_sampling_params,
                processor=rollout_processor,
                metadata=batch["metadata"],
                student_prompt_input_ids=student_prompt["input_ids"],
                pad_token_id=int(config["tokens"]["pad_token_id"]),
                device=accelerator.device,
                blocked_rollout_token_ids=set(blocked_rollout_token_ids),
            )
            generation_mask = _build_generation_mask_from_lengths(
                generation_ids=full_generated_ids[:, int(student_prompt["prompt_length"]):],
                completion_lengths=completion_lengths,
            )
            full_generated_attention_mask = torch.cat(
                [student_prompt["attention_mask"], generation_mask],
                dim=1,
            )
        else:
            full_generated_ids, full_generated_attention_mask = _generate_student_rollout(
                model=model,
                accelerator=accelerator,
                student_prompt=student_prompt,
                generation_config=generation_config,
                pad_token_id=int(config["tokens"]["pad_token_id"]),
            )
        student_prompt_len = int(student_prompt["prompt_length"])
        generation_ids = full_generated_ids[:, student_prompt_len:]
        generation_mask = full_generated_attention_mask[:, student_prompt_len:]

        teacher_student_hidden_sequences = [None for _ in batch["metadata"]]
        if enable_teacher_student_replay:
            active_teacher_model, teacher_context = _teacher_forward_target(config, accelerator, model, teacher_model)
            teacher_prompt_batch, teacher_input_ids, teacher_attention_mask = _build_teacher_student_replay_batch(
                teacher_processor=teacher_processor,
                metadata=batch["metadata"],
                generation_ids=generation_ids,
                generation_mask=generation_mask,
                device=accelerator.device,
                model_dtype=teacher_model_dtype or model_dtype,
            )
            with torch.no_grad(), teacher_context, _temporary_eval(active_teacher_model):
                with _opsd_replay_mode():
                    _teacher_outputs, teacher_student_hidden = _forward_with_last_hidden(
                        active_teacher_model,
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask,
                        pixel_values=teacher_prompt_batch.get("pixel_values"),
                        image_grid_thw=teacher_prompt_batch.get("image_grid_thw"),
                        logits_to_keep=max(1, generation_ids.shape[1] + 1),
                        use_cache=False,
                    )
            teacher_student_hidden_sequences = _extract_teacher_student_replay_hidden_sequences(
                teacher_hidden=teacher_student_hidden,
                teacher_prompt_batch=teacher_prompt_batch,
                generation_ids=generation_ids,
                generation_mask=generation_mask,
            )

        active_teacher_model, teacher_context = _teacher_forward_target(config, accelerator, model, teacher_model)
        teacher_hidden_sequences = []
        if enable_second_teacher_replay:
            teacher_hidden_sequences = _extract_teacher_ground_truth_hidden_sequences(
                teacher_model=active_teacher_model,
                teacher_context=teacher_context,
                teacher_processor=teacher_processor,
                metadata=batch["metadata"],
                device=accelerator.device,
                model_dtype=teacher_model_dtype or model_dtype,
                ot_replay_cfg=ot_replay_cfg,
            )
        else:
            teacher_hidden_sequences = [None for _ in batch["metadata"]]
        latent_positions = _find_latent_positions(
            input_ids=full_generated_ids,
            latent_token_id=int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669")),
            thinking_start_id=int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667")),
            thinking_end_id=int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668")),
        )
        batch_summary = _summarize_batch(
            batch["metadata"],
            generation_ids=generation_ids,
            generation_mask=generation_mask,
            teacher_student_hidden_sequences=teacher_student_hidden_sequences,
            teacher_hidden_sequences=teacher_hidden_sequences,
            latent_positions=latent_positions,
        )
        print(
            "[opsd] dry-run "
            f"batch={batch_idx} "
            f"token_loss={loss_cfg['token_type']} "
            f"student_full_shape={tuple(full_generated_ids.shape)} "
            f"completion_width={generation_ids.shape[1]} "
            f"latent_tokens={int(latent_positions.sum().item())} "
            f"teacher_gt_tokens={sum(int(t.shape[0]) for t in teacher_hidden_sequences if t is not None)}"
        )
        if batch_idx == 0:
            _save_batch_samples(
                output_dir,
                batch_summary,
                "dry_run",
                teacher_student_hidden_sequences=teacher_student_hidden_sequences,
                teacher_gt_hidden_sequences=teacher_hidden_sequences,
            )
        if batch_idx + 1 >= dry_run_batches:
            break


def _run_training(config: dict[str, Any]) -> None:
    train_cfg = config["training"]
    _configure_quiet_logging()
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        mixed_precision="bf16" if bool(train_cfg.get("bf16", True)) else "no",
    )
    _configure_deepspeed_runtime(accelerator, config)

    _set_seed(int(train_cfg.get("seed", 42)))
    _prepare_runtime_env(config)
    loss_cfg = _loss_config(config)
    manual_loss_configs = _split_loss_config(loss_cfg)
    _prepare_opsd_runtime_env()
    opsd_weight = _loss_term_weight(manual_loss_configs, "opsd")
    ot_replay_weight = _loss_term_weight(manual_loss_configs, "ot_replay")
    enable_teacher_student_replay = opsd_weight > 0.0

    student_processor = _load_processor_for_model(str(config["model"]["model_name_or_path"]), config)
    teacher_processor = _prepare_teacher_processor(config, student_processor)
    tokenizer = getattr(student_processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Qwen3-VL processor did not expose a tokenizer.")

    config.setdefault("tokens", {})
    config["tokens"]["pad_token_id"] = tokenizer.pad_token_id

    model, model_dtype, require_vae_checkpoints = _prepare_model(config, accelerator)
    teacher_model, teacher_model_dtype = _prepare_teacher_model(config, accelerator)
    dataset, dataloader = _build_dataloader(config, student_processor, teacher_processor)
    generation_config = _build_generation_config(config)
    ot_replay_cfg = _ot_replay_config(config)
    enable_second_teacher_replay = bool(ot_replay_cfg["enabled"]) and ot_replay_weight > 0.0

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

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(output_dir / "resolved_opsd_config.yaml", config)
    rollout_engine, rollout_sampling_params, rollout_processor, blocked_rollout_token_ids = _init_vllm_rollout_engine(
        accelerator=accelerator,
        processor=student_processor,
        config=config,
    )
    if rollout_engine is not None:
        _sync_training_model_to_vllm(accelerator, model, rollout_engine)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(
            f"[opsd] teacher_strategy={_teacher_strategy(config)} "
            f"teacher_prompt_processor={'student' if teacher_processor is student_processor else 'separate'} "
            f"loss_type={loss_cfg['type']} "
            f"token_loss={loss_cfg['token_type']} opsd_enabled={enable_teacher_student_replay} "
            f"ot_replay_enabled={enable_second_teacher_replay}"
        )
        print(
            f"[opsd] dataset_size={len(dataset)} batch_size={train_cfg['per_device_train_batch_size']} "
            f"grad_accum={train_cfg['gradient_accumulation_steps']} total_steps={total_steps}"
        )

    global_step = 0
    micro_step = 0
    start_time = time.time()
    stage_seconds = {
        "rollout": 0.0,
        "teacher_student": 0.0,
        "teacher_gt": 0.0,
        "student": 0.0,
        "opt": 0.0,
    }
    window_stats = {
        "micro_steps": 0,
        "samples": 0,
        "completion_tokens": 0,
        "student_latent_tokens": 0,
        "teacher_gt_tokens": 0,
        "max_seq_len": 0,
    }
    last_loss_value = 0.0
    last_metrics: dict[str, float] = {}
    last_stats = {
        "samples": 0,
        "completion_tokens": 0,
        "student_latent_tokens": 0,
        "teacher_gt_tokens": 0,
        "seq_len": 0,
    }
    model.train()
    dataloader_iter = iter(dataloader)
    pending_rollouts: list[tuple[dict[str, Any], torch.Tensor]] = []

    def _next_batch() -> dict[str, Any] | None:
        nonlocal dataloader_iter
        try:
            return next(dataloader_iter)
        except StopIteration:
            return None

    def _fill_rollout_queue() -> bool:
        if rollout_engine is None or pending_rollouts:
            return bool(pending_rollouts)

        window_size = max(1, int((config.get("rollout") or {}).get("window_size", 1)))
        window_batches: list[dict[str, Any]] = []
        window_metadata: list[dict[str, Any]] = []
        for _ in range(window_size):
            next_batch = _next_batch()
            if next_batch is None:
                break
            window_batches.append(next_batch)
            window_metadata.extend(next_batch["metadata"])
        if not window_batches:
            return False

        stage_start = time.perf_counter()
        completions = _generate_completions_with_vllm(
            vllm_engine=rollout_engine,
            sampling_params=rollout_sampling_params,
            processor=rollout_processor,
            metadata=window_metadata,
            completion_dtype=torch.long,
            blocked_rollout_token_ids=set(blocked_rollout_token_ids),
        )
        stage_seconds["rollout"] += time.perf_counter() - stage_start

        offset = 0
        for queued_batch in window_batches:
            count = len(queued_batch["metadata"])
            for idx in range(count):
                single_prompt = {}
                for key, value in queued_batch["student_prompt"].items():
                    if torch.is_tensor(value) and value.shape[:1] == (count,):
                        single_prompt[key] = value[idx : idx + 1]
                    else:
                        single_prompt[key] = value
                single_prompt["prompt_length"] = int(queued_batch["student_prompt"]["prompt_length"])
                single_prompt["prompt_lengths_per_example"] = queued_batch["student_prompt"][
                    "prompt_lengths_per_example"
                ][idx : idx + 1]
                single_batch = {
                    "student_prompt": single_prompt,
                    "metadata": [queued_batch["metadata"][idx]],
                }
                pending_rollouts.append((single_batch, completions[offset + idx]))
            offset += count
        return bool(pending_rollouts)

    for epoch in range(int(train_cfg["num_train_epochs"])):
        while True:
            if rollout_engine is not None:
                if not _fill_rollout_queue():
                    break
                batch, rollout_completion_ids = pending_rollouts.pop(0)
            else:
                batch = _next_batch()
                rollout_completion_ids = None
                if batch is None:
                    break
            micro_step += 1
            with accelerator.accumulate(model):
                student_prompt = _move_batch_to_device(batch["student_prompt"], accelerator.device, model_dtype)

                if rollout_engine is not None:
                    full_generated_ids, full_generated_attention_mask = _compose_vllm_rollout_from_completion(
                        student_prompt=student_prompt,
                        completion_ids=rollout_completion_ids,
                        pad_token_id=int(config["tokens"]["pad_token_id"]),
                        device=accelerator.device,
                    )
                else:
                    stage_start = time.perf_counter()
                    full_generated_ids, full_generated_attention_mask = _generate_student_rollout(
                        model=model,
                        accelerator=accelerator,
                        student_prompt=student_prompt,
                        generation_config=generation_config,
                        pad_token_id=int(config["tokens"]["pad_token_id"]),
                    )
                    stage_seconds["rollout"] += time.perf_counter() - stage_start

                student_prompt_len = int(student_prompt["prompt_length"])
                generation_ids = full_generated_ids[:, student_prompt_len:]
                generation_mask = full_generated_attention_mask[:, student_prompt_len:]
                completion_width = int(generation_ids.shape[1])
                if completion_width <= 0:
                    if accelerator.is_main_process:
                        print("[opsd] skipped_batch reason=empty_generation")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                labels = generation_ids.clone()
                labels[generation_mask == 0] = -100
                latent_positions = _find_latent_positions(
                    input_ids=full_generated_ids,
                    latent_token_id=int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669")),
                    thinking_start_id=int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667")),
                    thinking_end_id=int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668")),
                ).to(device=accelerator.device)

                teacher_outputs = None
                teacher_logits = None
                teacher_student_hidden_sequences = [None for _ in batch["metadata"]]
                if enable_teacher_student_replay:
                    active_teacher_model, teacher_student_context = _teacher_forward_target(
                        config,
                        accelerator,
                        model,
                        teacher_model,
                    )
                    stage_start = time.perf_counter()
                    teacher_prompt_batch, teacher_input_ids, teacher_attention_mask = _build_teacher_student_replay_batch(
                        teacher_processor=teacher_processor,
                        metadata=batch["metadata"],
                        generation_ids=generation_ids,
                        generation_mask=generation_mask,
                        device=accelerator.device,
                        model_dtype=teacher_model_dtype or model_dtype,
                    )
                    with torch.no_grad(), teacher_student_context, _temporary_eval(active_teacher_model):
                        with _opsd_replay_mode():
                            teacher_outputs, teacher_student_hidden = _forward_with_last_hidden(
                                active_teacher_model,
                                input_ids=teacher_input_ids,
                                attention_mask=teacher_attention_mask,
                                pixel_values=teacher_prompt_batch.get("pixel_values"),
                                image_grid_thw=teacher_prompt_batch.get("image_grid_thw"),
                                logits_to_keep=max(1, completion_width + 1),
                                use_cache=False,
                            )
                    teacher_logits = teacher_outputs.logits[:, :-1, :]
                    teacher_student_hidden_sequences = _extract_teacher_student_replay_hidden_sequences(
                        teacher_hidden=teacher_student_hidden,
                        teacher_prompt_batch=teacher_prompt_batch,
                        generation_ids=generation_ids,
                        generation_mask=generation_mask,
                    )
                    stage_seconds["teacher_student"] += time.perf_counter() - stage_start

                teacher_hidden_sequences = [None for _ in batch["metadata"]]
                if enable_second_teacher_replay:
                    active_teacher_model, teacher_gt_context = _teacher_forward_target(
                        config,
                        accelerator,
                        model,
                        teacher_model,
                    )
                    stage_start = time.perf_counter()
                    teacher_hidden_sequences = _extract_teacher_ground_truth_hidden_sequences(
                        teacher_model=active_teacher_model,
                        teacher_context=teacher_gt_context,
                        teacher_processor=teacher_processor,
                        metadata=batch["metadata"],
                        device=accelerator.device,
                        model_dtype=teacher_model_dtype or model_dtype,
                        ot_replay_cfg=ot_replay_cfg,
                    )
                    stage_seconds["teacher_gt"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                with _opsd_replay_mode():
                    student_outputs, student_hidden = _forward_with_last_hidden(
                        model,
                        input_ids=full_generated_ids,
                        attention_mask=full_generated_attention_mask,
                        pixel_values=student_prompt.get("pixel_values"),
                        image_grid_thw=student_prompt.get("image_grid_thw"),
                        logits_to_keep=max(1, completion_width + 1),
                        use_cache=False,
                    )
                student_logits = student_outputs.logits[:, :-1, :]
                latent_supervision = [
                    [tensor.to(device=accelerator.device, dtype=model_dtype) for tensor in sample if tensor is not None]
                    for sample in [[sequence] if sequence is not None else [] for sequence in teacher_hidden_sequences]
                ]

                token_loss = None
                token_metrics: dict[str, float] = {}
                if enable_teacher_student_replay:
                    token_loss, token_metrics = _token_distillation_loss(
                        token_loss_type=str(loss_cfg["token_type"]).strip().lower(),
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        generation_ids=generation_ids,
                        labels=labels,
                        loss_cfg=loss_cfg,
                    )
                ot_loss = None
                if enable_second_teacher_replay:
                    ot_loss, _ = _compute_ot_loss(
                        hidden_states=student_hidden,
                        latent_supervision=latent_supervision,
                        latent_positions=latent_positions,
                    )
                stage_seconds["student"] += time.perf_counter() - stage_start

                total_loss = None
                if token_loss is not None and opsd_weight > 0.0:
                    token_term = token_loss * opsd_weight
                    total_loss = token_term if total_loss is None else total_loss + token_term
                if ot_loss is not None and ot_replay_weight > 0.0:
                    ot_term = ot_loss * ot_replay_weight
                    total_loss = ot_term if total_loss is None else total_loss + ot_term
                if total_loss is None:
                    if accelerator.is_main_process:
                        print("[opsd] skipped_batch reason=no_loss")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                stage_start = time.perf_counter()
                accelerator.backward(total_loss)
                if float(train_cfg.get("max_grad_norm", 0.0)) > 0:
                    accelerator.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                stage_seconds["opt"] += time.perf_counter() - stage_start

                last_loss_value = float(total_loss.detach())
                last_metrics = dict(token_metrics)
                last_metrics["token_loss"] = float(token_loss.detach()) if token_loss is not None else float("nan")
                last_metrics["ot_loss"] = float(ot_loss.detach()) if ot_loss is not None else float("nan")
                last_stats = {
                    "samples": len(batch["metadata"]),
                    "completion_tokens": int(generation_mask.sum().item()),
                    "student_latent_tokens": int(latent_positions.sum().item()),
                    "teacher_gt_tokens": sum(int(t.shape[0]) for t in teacher_hidden_sequences if t is not None),
                    "seq_len": int(full_generated_ids.shape[1]),
                }
                window_stats["micro_steps"] += 1
                window_stats["samples"] += last_stats["samples"]
                window_stats["completion_tokens"] += last_stats["completion_tokens"]
                window_stats["student_latent_tokens"] += last_stats["student_latent_tokens"]
                window_stats["teacher_gt_tokens"] += last_stats["teacher_gt_tokens"]
                window_stats["max_seq_len"] = max(window_stats["max_seq_len"], last_stats["seq_len"])
                batch_summary = _summarize_batch(
                    batch["metadata"],
                    generation_ids=generation_ids,
                    generation_mask=generation_mask,
                    teacher_student_hidden_sequences=teacher_student_hidden_sequences,
                    teacher_hidden_sequences=teacher_hidden_sequences,
                    latent_positions=latent_positions,
                )

                del teacher_outputs
                del teacher_logits
                if enable_teacher_student_replay:
                    del teacher_student_hidden
                del student_outputs
                del student_logits
                del student_hidden
                del token_loss
                del ot_loss
                del total_loss

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process and global_step % int(train_cfg["logging_steps"]) == 0:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    progress = float(global_step) / float(max(1, total_steps))
                    remaining_steps = max(0, total_steps - global_step)
                    eta_seconds = (elapsed / float(max(1, global_step))) * remaining_steps
                    window_micro_steps = max(1, int(window_stats["micro_steps"]))
                    metric_suffix = (
                        f" token_loss={last_metrics.get('token_loss', float('nan')):.6f}"
                        f" ot_loss={last_metrics.get('ot_loss', float('nan')):.6f}"
                    )
                    if "advantage" in last_metrics:
                        metric_suffix += (
                            f" advantage={last_metrics['advantage']:.4f}"
                            f" student_lp={last_metrics['student_logprob']:.4f}"
                            f" teacher_lp={last_metrics['teacher_logprob']:.4f}"
                        )
                    print(
                        f"[opsd] step={global_step}/{total_steps} progress={progress * 100.0:.1f}% "
                        f"micro_step={micro_step} "
                        f"epoch={epoch + 1}/{max(1, int(math.ceil(configured_epochs))) if configured_epochs > 0 else 1} "
                        f"loss={last_loss_value:.6f} lr={lr:.3e} "
                        f"elapsed={elapsed / 3600.0:.2f}h eta={eta_seconds / 3600.0:.2f}h{metric_suffix} "
                        f"last_samples={last_stats['samples']} last_completion_tokens={last_stats['completion_tokens']} "
                        f"last_student_latent_tokens={last_stats['student_latent_tokens']} "
                        f"last_teacher_gt_tokens={last_stats['teacher_gt_tokens']} last_seq_len={last_stats['seq_len']} "
                        f"window_micros={window_micro_steps} window_samples={window_stats['samples']} "
                        f"window_completion_tokens={window_stats['completion_tokens']} "
                        f"window_student_latent_tokens={window_stats['student_latent_tokens']} "
                        f"window_teacher_gt_tokens={window_stats['teacher_gt_tokens']} "
                        f"window_max_seq_len={window_stats['max_seq_len']} "
                        f"timing_sum=rollout:{stage_seconds['rollout']:.1f}s "
                        f"teacher_student:{stage_seconds['teacher_student']:.1f}s "
                        f"teacher_gt:{stage_seconds['teacher_gt']:.1f}s "
                        f"student:{stage_seconds['student']:.1f}s "
                        f"opt:{stage_seconds['opt']:.1f}s "
                        f"timing_per_micro=rollout:{stage_seconds['rollout'] / window_micro_steps:.2f}s "
                        f"teacher_student:{stage_seconds['teacher_student'] / window_micro_steps:.2f}s "
                        f"teacher_gt:{stage_seconds['teacher_gt'] / window_micro_steps:.2f}s "
                        f"student:{stage_seconds['student'] / window_micro_steps:.2f}s "
                        f"opt:{stage_seconds['opt'] / window_micro_steps:.2f}s"
                    )
                    stage_seconds = {key: 0.0 for key in stage_seconds}
                    window_stats = {key: 0 for key in window_stats}

                if accelerator.is_main_process and global_step == 1:
                    _save_batch_samples(
                        output_dir,
                        batch_summary,
                        "step_1",
                        teacher_student_hidden_sequences=teacher_student_hidden_sequences,
                        teacher_gt_hidden_sequences=teacher_hidden_sequences,
                    )
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
                        processor=student_processor,
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

                sync_steps = int((config.get("rollout") or {}).get("sync_steps", 0))
                if rollout_engine is not None and sync_steps > 0 and global_step % sync_steps == 0:
                    accelerator.wait_for_everyone()
                    sync_started_at = time.perf_counter()
                    if accelerator.is_main_process:
                        print(f"[opsd] rollout_sync_start step={global_step}")
                    _sync_training_model_to_vllm(accelerator, model, rollout_engine)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        print(
                            f"[opsd] rollout_sync_done step={global_step} "
                            f"seconds={time.perf_counter() - sync_started_at:.1f}"
                        )

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    _save_checkpoint(
        accelerator=accelerator,
        model=model,
        processor=student_processor,
        output_dir=output_dir,
        step_label="checkpoint-final",
        keep_last_n=int(train_cfg.get("save_total_limit", 0)),
        require_vae=require_vae_checkpoints,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL OPSD with configurable dual teacher replay.")
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
        help="Build dual-replay batches without optimization",
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
    else:
        _run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
