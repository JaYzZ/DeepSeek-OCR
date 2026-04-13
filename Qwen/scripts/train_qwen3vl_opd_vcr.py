#!/usr/bin/env python3
"""Teacher-rollout OPD trainer for Qwen3-VL continuous thinking.

This variant keeps the existing off-policy setup at the data/prompt level:
- teacher sees question images plus optional privileged rationale images
- student sees only question images

The key difference is the supervision target:
- teacher generates a discrete `<think> ... </think> answer` rollout online
- the teacher CoT is chunked with the same text logic as SFT
- each chunk is rendered and encoded online into Qwen3-VL latent features
- student is trained on a continuous replay sequence:
  `<think><latent><think_sep>...<latent></think>{answer}`

This reuses the native latent/VAE loss path in `Qwen.llamafactory.integration`
instead of distilling teacher token logits directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from vllm import SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen.llamafactory.integration import (
    _expand_sample_for_latent_injection,
    _find_latent_positions,
)
from Qwen.data.utils import (
    AdaptiveSkiaRenderer,
    ThinkingReplayRenderConfig,
    _ensure_thinking_chunks_fit_renderer,
    build_cot_chunk_token_ids,
    extract_thinking_and_answer as extract_thinking_and_answer_from_builder,
    format_cot_subsequences,
    normalize_thinking_text_for_rendering,
)
from Qwen.scripts.train_qwen3vl_opsd import (
    OpsdManifestDataset,
    _configure_quiet_logging,
    _format_user_prompt,
    _init_vllm_rollout_engine,
    _load_image,
    _load_processor_for_model,
    _load_yaml,
    _maybe_limit_images,
    _maybe_prepare_manifest,
    _move_batch_to_device,
    _parse_bool_flag,
    _prepare_messages_for_vllm,
    _prepare_model,
    _prepare_runtime_env,
    _prepare_teacher_processor,
    _resolve_config_paths,
    _save_checkpoint,
    _set_seed,
    _sync_training_model_to_vllm,
    _write_yaml,
)
def _strip_generation_noise(text: str) -> str:
    stripped = (text or "").strip()
    stripped = stripped.replace("<|im_end|>", "").strip()
    stripped = stripped.replace("<|endoftext|>", "").strip()
    return stripped


def _extract_thinking_and_answer(text: str, *, forced_think_prefix: bool = False) -> tuple[str, str]:
    cleaned = _strip_generation_noise(text)
    if "<think>" in cleaned:
        thinking_chunks, answer = extract_thinking_and_answer_from_builder(
            cleaned,
            max_chars=None,
            return_chunks=True,
        )
        return "\n".join(thinking_chunks).strip(), answer

    close_tag = "</think>"
    if forced_think_prefix:
        close_pos = cleaned.find(close_tag)
        if close_pos < 0:
            rebuilt = f"<think>{cleaned}</think>"
        else:
            rebuilt = f"<think>{cleaned[:close_pos].strip()}</think>{cleaned[close_pos + len(close_tag):].strip()}"
        thinking_chunks, answer = extract_thinking_and_answer_from_builder(
            rebuilt,
            max_chars=None,
            return_chunks=True,
        )
        return "\n".join(thinking_chunks).strip(), answer

    return "", cleaned


def _build_user_messages(
    *,
    system_prompt: str,
    user_text: str,
    question_images: list[str],
    rationale_images: list[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if question_images:
        content.append({"type": "text", "text": "Question image(s):"})
        content.extend({"type": "image", "image": image_path} for image_path in question_images)
    if rationale_images:
        content.append({"type": "text", "text": "Reference reasoning image(s):"})
        content.extend({"type": "image", "image": image_path} for image_path in rationale_images)
    content.append({"type": "text", "text": user_text})

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _teacher_force_think(config: dict[str, Any]) -> bool:
    return bool((config.get("teacher") or {}).get("vllm_force_think", False))


def _teacher_rollout_backend(config: dict[str, Any]) -> str:
    return str((config.get("teacher") or {}).get("rollout_backend", "hf")).strip().lower()


def _build_replay_render_config(config: dict[str, Any]) -> ThinkingReplayRenderConfig:
    replay_cfg = ((config.get("distillation") or {}).get("replay_render") or {})
    return ThinkingReplayRenderConfig(
        strip_leading_indentation=bool(replay_cfg.get("strip_leading_indentation", False)),
        collapse_multi_blank_lines=bool(replay_cfg.get("collapse_multi_blank_lines", False)),
        normalize_bullet_prefixes=bool(replay_cfg.get("normalize_bullet_prefixes", False)),
        collapse_all_whitespace=bool(replay_cfg.get("collapse_all_whitespace", False)),
        compact_layout=bool(replay_cfg.get("compact_layout", False)),
    )


def _build_replay_renderer(config: dict[str, Any]) -> AdaptiveSkiaRenderer:
    replay_cfg = ((config.get("distillation") or {}).get("replay_render") or {})
    min_font_size = float(replay_cfg.get("min_font_size", 10.0))
    max_font_size = float(replay_cfg.get("max_font_size", min_font_size))
    return AdaptiveSkiaRenderer(
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        compact_thinking_layout=bool(replay_cfg.get("compact_layout", False)),
    )


def _teacher_prompt_text(processor, messages: list[dict[str, Any]], *, force_think: bool) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = text + "<|im_start|>assistant\n"
    if force_think:
        text = text + "<think>"
    return text


def _build_teacher_prompt_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_prompt = str(config.get("system_prompt", "") or "").strip()
    teacher_template = str(config["prompts"]["teacher_user"])
    max_teacher_rationale_images = int(config["prompts"].get("max_teacher_rationale_images", 0))

    prepared: list[dict[str, Any]] = []
    for row in rows:
        question_images = list(row.get("question_images") or [])
        rationale_images = _maybe_limit_images(
            list(row.get("teacher_rationale_images") or []),
            max_teacher_rationale_images,
        )
        question_text = str(row.get("student_user_text") or "").strip()
        teacher_user = _format_user_prompt(
            teacher_template,
            num_question_images=len(question_images),
            num_rationale_images=len(rationale_images),
            question_text=question_text,
        )
        teacher_messages = _build_user_messages(
            system_prompt=system_prompt,
            user_text=teacher_user,
            question_images=question_images,
            rationale_images=rationale_images,
        )
        prepared.append(
            {
                "sample_id": row.get("sample_id"),
                "source_dataset": row.get("source_dataset"),
                "task": row.get("task"),
                "question_text": question_text,
                "question_images": question_images,
                "teacher_rationale_images": rationale_images,
                "latent_supervision": list(row.get("latent_supervision") or []),
                "teacher_messages": teacher_messages,
            }
        )
    return prepared


def _unwrap_model(module):
    current = module
    while hasattr(current, "module"):
        current = current.module
    return current


def _resolve_visual_model(module):
    unwrapped = _unwrap_model(module)
    candidates = [
        getattr(unwrapped, "visual", None),
        getattr(getattr(unwrapped, "model", None), "visual", None),
    ]

    get_base_model = getattr(unwrapped, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        candidates.extend(
            [
                getattr(base_model, "visual", None),
                getattr(getattr(base_model, "model", None), "visual", None),
            ]
        )

    base_model_attr = getattr(unwrapped, "base_model", None)
    if base_model_attr is not None:
        candidates.extend(
            [
                getattr(base_model_attr, "visual", None),
                getattr(getattr(base_model_attr, "model", None), "visual", None),
            ]
        )

    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError("Failed to resolve Qwen3-VL visual module from the loaded student model.")


def _encode_images(model, processor, images: list[Any]) -> list[torch.Tensor]:
    if not images:
        return []

    visual = _resolve_visual_model(model)
    visual_param = next(visual.parameters())
    processed = processor.image_processor(images, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device=visual_param.device, dtype=visual_param.dtype)
    grid_thw = processed["image_grid_thw"].to(device=visual_param.device)

    with torch.inference_mode():
        vision_output = visual(pixel_values, grid_thw=grid_thw)

    features = vision_output[0] if isinstance(vision_output, tuple) else vision_output
    merge_size = int(getattr(getattr(visual, "config", None), "spatial_merge_size", 2))

    tokens_per_image: list[int] = []
    for i in range(int(grid_thw.shape[0])):
        t, h, w = grid_thw[i].tolist()
        tokens_per_image.append(int((t * h * w) // (merge_size ** 2)))

    features_list: list[torch.Tensor] = []
    start_idx = 0
    for num_tokens in tokens_per_image:
        end_idx = start_idx + num_tokens
        features_list.append(features[start_idx:end_idx].detach())
        start_idx = end_idx

    if start_idx != int(features.shape[0]):
        raise RuntimeError(
            f"Rendered-image feature split mismatch: consumed {start_idx}, total {int(features.shape[0])}."
        )

    return features_list


def _prepare_student_training_batch(
    *,
    config: dict[str, Any],
    model,
    processor,
    tokenizer,
    renderer: AdaptiveSkiaRenderer,
    prepared_rows: list[dict[str, Any]],
    teacher_rollouts: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    system_prompt = str(config.get("system_prompt", "") or "").strip()
    student_template = str(config["prompts"]["student_user"])
    max_chars_per_chunk = int(config["distillation"]["max_chars_per_chunk"])
    skip_empty_thinking = bool(config["distillation"].get("skip_empty_thinking", True))
    max_length = int(config["training"]["max_length"])
    pad_token_id = tokenizer.pad_token_id
    latent_token_id = int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669"))
    thinking_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    thinking_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))
    ignore_index = -100

    valid_specs: list[dict[str, Any]] = []
    all_chunk_texts: list[str] = []
    forced_think_prefix = _teacher_force_think(config)
    replay_render_config = _build_replay_render_config(config)

    for row, rollout in zip(prepared_rows, teacher_rollouts):
        completion = str(rollout.get("decoded_text") or rollout.get("text") or "")
        thinking_text, answer_text = _extract_thinking_and_answer(
            completion,
            forced_think_prefix=forced_think_prefix,
        )
        thinking_text = normalize_thinking_text_for_rendering(
            thinking_text,
            config=replay_render_config,
        )
        answer_text = _strip_generation_noise(answer_text)
        if thinking_text:
            rebuilt = f"<think>{thinking_text}</think>{answer_text}"
            thinking_chunks, answer_text = extract_thinking_and_answer_from_builder(
                rebuilt,
                max_chars=max_chars_per_chunk,
                return_chunks=True,
            )
            thinking_chunks = _ensure_thinking_chunks_fit_renderer(
                renderer,
                thinking_chunks,
                preserve_line_boundaries=not replay_render_config.compact_layout,
            )
        else:
            thinking_chunks = []

        if not thinking_chunks and skip_empty_thinking:
            continue
        if not answer_text:
            answer_text = _strip_generation_noise(completion)

        question_images = list(row["question_images"])
        question_pils = [_load_image(path) for path in question_images]
        student_user = _format_user_prompt(
            student_template,
            num_question_images=len(question_images),
            num_rationale_images=0,
            question_text=str(row["question_text"]),
        )
        user_messages = _build_user_messages(
            system_prompt=system_prompt,
            user_text=student_user,
            question_images=question_images,
            rationale_images=[],
        )
        assistant_latent_content = (
            f"<think>{'<think_sep>'.join(['<latent>'] * len(thinking_chunks))}</think>{answer_text}"
            if thinking_chunks
            else answer_text
        )
        full_messages = user_messages + [{"role": "assistant", "content": assistant_latent_content}]
        prompt_text = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        chunk_token_ids = build_cot_chunk_token_ids(tokenizer, thinking_chunks)
        if len(chunk_token_ids) != len(thinking_chunks):
            raise ValueError(
                f"Teacher rollout chunk/token mismatch for {row['sample_id']}: "
                f"{len(chunk_token_ids)} vs {len(thinking_chunks)}"
            )

        valid_specs.append(
            {
                **row,
                "question_pils": question_pils,
                "prompt_text": prompt_text,
                "full_text": full_text,
                "teacher_completion": _strip_generation_noise(completion),
                "teacher_thinking_chunks": thinking_chunks,
                "assistant_target": assistant_latent_content,
                "answer_text": answer_text,
                "cot_chunk_token_ids": chunk_token_ids,
                "cot": format_cot_subsequences(thinking_chunks),
            }
        )
        all_chunk_texts.extend(thinking_chunks)

    if not valid_specs:
        first_rollout = teacher_rollouts[0] if teacher_rollouts else {}
        preview = _strip_generation_noise(
            str(first_rollout.get("decoded_text") or first_rollout.get("text") or "")
        )[:400]
        raise RuntimeError(
            "No valid teacher rollouts produced usable thinking chunks in this batch. "
            f"first_completion_preview={preview!r}"
        )

    rendered_images = renderer.render_batch(all_chunk_texts, thinking_mode=True) if all_chunk_texts else []
    encoded_features = _encode_images(model, processor, rendered_images) if rendered_images else []
    feature_cursor = 0

    for spec in valid_specs:
        num_chunks = len(spec["teacher_thinking_chunks"])
        latent_ground_truth = []
        latent_seq_lens = []
        if num_chunks > 0:
            latent_ground_truth = encoded_features[feature_cursor : feature_cursor + num_chunks]
            latent_seq_lens = [int(tensor.shape[0]) for tensor in latent_ground_truth]
            feature_cursor += num_chunks
        spec["latent_ground_truth_tensors"] = latent_ground_truth
        spec["latent_seq_lens"] = latent_seq_lens

    if feature_cursor != len(encoded_features):
        raise RuntimeError(
            f"Teacher rollout latent packing mismatch: consumed {feature_cursor}, encoded {len(encoded_features)}"
        )

    prompt_batch = processor(
        text=[spec["prompt_text"] for spec in valid_specs],
        images=[spec["question_pils"] for spec in valid_specs],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    full_batch = processor(
        text=[spec["full_text"] for spec in valid_specs],
        images=[spec["question_pils"] for spec in valid_specs],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    expanded_input_ids: list[list[int]] = []
    expanded_labels: list[list[int]] = []
    expanded_attention_masks: list[list[int]] = []
    expanded_vae_ce_masks: list[list[int]] = []
    replay_completion_widths: list[int] = []

    for idx, spec in enumerate(valid_specs):
        prompt_len = int(prompt_batch["attention_mask"][idx].sum().item())
        full_len = int(full_batch["attention_mask"][idx].sum().item())

        input_ids = full_batch["input_ids"][idx, :full_len].tolist()
        labels = list(input_ids)
        for pos in range(prompt_len):
            labels[pos] = ignore_index

        new_input_ids, new_labels, new_vae_ce_mask = _expand_sample_for_latent_injection(
            input_ids=input_ids,
            labels=labels,
            latent_token_id=latent_token_id,
            thinking_start_id=thinking_start_id,
            thinking_end_id=thinking_end_id,
            latent_ground_truth=spec["latent_ground_truth_tensors"],
            latent_lengths=spec["latent_seq_lens"],
            ignore_index=ignore_index,
            cot_step_token_ids=spec["cot_chunk_token_ids"],
        )

        if len(new_input_ids) > max_length:
            raise ValueError(
                f"Expanded sequence length {len(new_input_ids)} exceeds training.max_length={max_length} "
                f"for sample {spec['sample_id']}. Increase training.max_length."
            )

        expanded_input_ids.append(new_input_ids)
        expanded_labels.append(new_labels)
        expanded_attention_masks.append([1] * len(new_input_ids))
        expanded_vae_ce_masks.append(new_vae_ce_mask)
        replay_completion_widths.append(len(new_input_ids) - prompt_len)

    batch_size = len(valid_specs)
    max_seq_len = max(len(row) for row in expanded_input_ids)
    input_ids_tensor = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    labels_tensor = torch.full((batch_size, max_seq_len), ignore_index, dtype=torch.long)
    attention_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    vae_ce_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for idx in range(batch_size):
        seq_len = len(expanded_input_ids[idx])
        input_ids_tensor[idx, :seq_len] = torch.tensor(expanded_input_ids[idx], dtype=torch.long)
        labels_tensor[idx, :seq_len] = torch.tensor(expanded_labels[idx], dtype=torch.long)
        attention_mask_tensor[idx, :seq_len] = torch.tensor(expanded_attention_masks[idx], dtype=torch.long)
        vae_ce_mask_tensor[idx, :seq_len] = torch.tensor(expanded_vae_ce_masks[idx], dtype=torch.bool)

    latent_positions = _find_latent_positions(
        input_ids=input_ids_tensor,
        latent_token_id=latent_token_id,
        thinking_start_id=thinking_start_id,
        thinking_end_id=thinking_end_id,
    )

    latent_ground_truth = [spec["latent_ground_truth_tensors"] for spec in valid_specs]
    latent_supervision = [[] for _ in valid_specs]
    latent_supervision_paths = [list(spec.get("latent_supervision") or []) for spec in valid_specs]
    packed_ground_truth = []
    for sample_latents in latent_ground_truth:
        if sample_latents:
            packed_ground_truth.append(torch.cat(sample_latents, dim=0))
    latent_ground_truth_packed = torch.cat(packed_ground_truth, dim=0) if packed_ground_truth else None

    training_batch = {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "attention_mask": attention_mask_tensor,
        "vae_ce_mask": vae_ce_mask_tensor,
        "replay_completion_width": int(max(replay_completion_widths, default=0)),
        "latent_positions": latent_positions,
        "latent_ground_truth": latent_ground_truth,
        "latent_ground_truth_paths": [[] for _ in valid_specs],
        "latent_ground_truth_packed": latent_ground_truth_packed,
        "latent_supervision": latent_supervision,
        "latent_supervision_paths": latent_supervision_paths,
        "pixel_values": full_batch.get("pixel_values"),
        "image_grid_thw": full_batch.get("image_grid_thw"),
    }
    return training_batch, valid_specs


def _save_teacher_rollout_samples(output_dir: Path, specs: list[dict[str, Any]], step_label: str) -> None:
    sample_path = output_dir / f"{step_label}_teacher_rollout_samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for spec in specs:
            f.write(
                json.dumps(
                    {
                        "sample_id": spec.get("sample_id"),
                        "task": spec.get("task"),
                        "question_text": spec.get("question_text"),
                        "question_images": spec.get("question_images"),
                        "teacher_rationale_images": spec.get("teacher_rationale_images"),
                        "prompt_text": spec.get("prompt_text"),
                        "teacher_completion": spec.get("teacher_completion"),
                        "assistant_target": spec.get("assistant_target"),
                        "answer_text": spec.get("answer_text"),
                        "num_latent_steps": len(spec.get("teacher_thinking_chunks") or []),
                        "thinking_chunks": spec.get("teacher_thinking_chunks"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _prepare_teacher_rollout_runtime(config: dict[str, Any], accelerator: Accelerator, processor):
    teacher_cfg = config["teacher"]
    rollout_source = str(teacher_cfg.get("rollout_source", "shared_student")).strip().lower()
    if rollout_source != "shared_student":
        raise ValueError(
            "This trainer currently supports only teacher.rollout_source=shared_student, "
            "where teacher rollout follows the student's current LoRA weights."
        )
    rollout_cfg = copy.deepcopy(config.get("rollout") or {})
    rollout_cfg["vllm_thinking"] = bool(teacher_cfg.get("vllm_thinking", False))
    rollout_cfg["vllm_force_think"] = bool(teacher_cfg.get("vllm_force_think", False))
    rollout_cfg["enable_sleep_mode"] = bool(rollout_cfg.get("enable_sleep_mode", True))
    rollout_cfg["gpu_memory_utilization"] = float(rollout_cfg.get("gpu_memory_utilization", 0.35))
    rollout_cfg["sync_steps"] = int(rollout_cfg.get("sync_steps", 1))
    rollout_cfg["enforce_eager"] = bool(rollout_cfg.get("enforce_eager", False))
    rollout_cfg["max_images_per_prompt"] = int(rollout_cfg.get("max_images_per_prompt", 10))

    teacher_rollout_config = copy.deepcopy(config)
    teacher_rollout_config["rollout"] = rollout_cfg

    previous_vllm_thinking = os.environ.get("VLLM_THINKING")
    previous_vllm_force_think = os.environ.get("VLLM_FORCE_THINK")
    try:
        rollout_engine, sampling_params, rollout_processor, blocked_ids = _init_vllm_rollout_engine(
            accelerator=accelerator,
            processor=processor,
            config=teacher_rollout_config,
        )
    finally:
        if previous_vllm_thinking is None:
            os.environ.pop("VLLM_THINKING", None)
        else:
            os.environ["VLLM_THINKING"] = previous_vllm_thinking
        if previous_vllm_force_think is None:
            os.environ.pop("VLLM_FORCE_THINK", None)
        else:
            os.environ["VLLM_FORCE_THINK"] = previous_vllm_force_think

    if rollout_engine is None:
        raise RuntimeError("Teacher rollout engine was not initialized. Set rollout.enable=true.")

    teacher_max_new_tokens = int(teacher_cfg.get("max_new_tokens", config["generation"]["max_new_tokens"]))
    teacher_temperature = float(teacher_cfg.get("temperature", 0.0))
    teacher_top_p = float(teacher_cfg.get("top_p", 1.0))
    teacher_top_k = int(teacher_cfg.get("top_k", 1))
    teacher_repetition_penalty = float(
        teacher_cfg.get("repetition_penalty", config["generation"].get("repetition_penalty", 1.0))
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=teacher_temperature,
        top_p=teacher_top_p,
        top_k=teacher_top_k,
        repetition_penalty=teacher_repetition_penalty,
        max_tokens=teacher_max_new_tokens,
        skip_special_tokens=False,
        logit_bias={token_id: -100.0 for token_id in blocked_ids},
    )
    return rollout_engine, sampling_params, rollout_processor, int(rollout_cfg["sync_steps"])


def _generate_teacher_rollouts(
    *,
    rollout_engine,
    sampling_params: SamplingParams,
    rollout_processor,
    prepared_rows: list[dict[str, Any]],
    force_think: bool,
) -> list[dict[str, Any]]:
    sleep_enabled = bool(getattr(rollout_engine, "opsd_enable_sleep_mode", False))
    if sleep_enabled and hasattr(rollout_engine, "wake_up"):
        torch.cuda.empty_cache()
        rollout_engine.wake_up()

    previous_vllm_force_think = os.environ.get("VLLM_FORCE_THINK")
    os.environ["VLLM_FORCE_THINK"] = "1" if force_think else "0"
    try:
        inputs = [_prepare_messages_for_vllm(row["teacher_messages"], rollout_processor) for row in prepared_rows]
    finally:
        if previous_vllm_force_think is None:
            os.environ.pop("VLLM_FORCE_THINK", None)
        else:
            os.environ["VLLM_FORCE_THINK"] = previous_vllm_force_think

    outputs = rollout_engine.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
    if sleep_enabled and hasattr(rollout_engine, "sleep"):
        rollout_engine.sleep(level=2)
    tokenizer = getattr(rollout_processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Teacher rollout processor must expose a tokenizer.")

    rollouts: list[dict[str, Any]] = []
    for output in outputs:
        candidate = output.outputs[0]
        token_ids = list(candidate.token_ids or [])
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        rollouts.append(
            {
                "text": candidate.text,
                "token_ids": token_ids,
                "decoded_text": decoded_text,
            }
        )
    return rollouts


def _build_dataloader(config: dict[str, Any]):
    train_cfg = config["training"]
    manifest_path = Path(config["data"]["opsd_manifest"])
    dataset = OpsdManifestDataset(manifest_path)
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["per_device_train_batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=lambda rows: rows,
        drop_last=bool(train_cfg.get("drop_last", True)),
    )
    return dataset, dataloader


def _prepare_distillation_env(config: dict[str, Any]) -> None:
    _prepare_runtime_env(config)
    student_loss_cfg = config["student_loss"]
    os.environ["QWEN3VL_LOSS_TYPE"] = str(student_loss_cfg.get("loss_type", "ce+vae"))
    os.environ["QWEN3VL_LATENT_AUX_LOSS_SOURCE"] = str(
        student_loss_cfg.get("latent_aux_loss_source", "hidden")
    )
    os.environ["QWEN3VL_VAE_TRAINABLE"] = "1" if _parse_bool_flag(student_loss_cfg.get("vae_trainable", True)) else "0"
    os.environ["QWEN3VL_LATENT_CE_ACTIVE"] = (
        "1" if _parse_bool_flag(student_loss_cfg.get("latent_ce_active", True)) else "0"
    )
    os.environ["QWEN3VL_LATENT_CE_TOKEN"] = (
        "1" if _parse_bool_flag(student_loss_cfg.get("latent_ce_token", False)) else "0"
    )
    os.environ["QWEN3VL_HIDDEN_STATES_HOOK"] = (
        "1" if _parse_bool_flag(student_loss_cfg.get("hidden_states_hook", False)) else "0"
    )


def _configure_deepspeed_runtime(accelerator: Accelerator, config: dict[str, Any]) -> None:
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return

    train_cfg = config.get("training", {})
    ds_config = plugin.deepspeed_config
    zero_cfg = ds_config.setdefault("zero_optimization", {})

    allgather_bucket_size = int(train_cfg.get("deepspeed_allgather_bucket_size", 10_000_000))
    reduce_bucket_size = int(train_cfg.get("deepspeed_reduce_bucket_size", 10_000_000))
    zero_cfg["allgather_bucket_size"] = allgather_bucket_size
    zero_cfg["reduce_bucket_size"] = reduce_bucket_size

    if "deepspeed_contiguous_gradients" in train_cfg:
        zero_cfg["contiguous_gradients"] = bool(train_cfg["deepspeed_contiguous_gradients"])

    plugin.deepspeed_config = ds_config
    if hasattr(plugin, "hf_ds_config") and getattr(plugin.hf_ds_config, "config", None) is not None:
        plugin.hf_ds_config.config = ds_config

    accelerator.print(
        "[opd-tr] deepspeed "
        f"zero_stage={zero_cfg.get('stage')} "
        f"allgather_bucket_size={zero_cfg.get('allgather_bucket_size')} "
        f"reduce_bucket_size={zero_cfg.get('reduce_bucket_size')} "
        f"contiguous_gradients={zero_cfg.get('contiguous_gradients')}"
    )


def _run_dry_run(config: dict[str, Any], dry_run_batches: int) -> None:
    config = copy.deepcopy(config)
    config.setdefault("training", {})
    config["training"]["num_workers"] = 0

    _configure_quiet_logging()
    accelerator = Accelerator(mixed_precision="bf16" if bool(config["training"].get("bf16", True)) else "no")
    _configure_deepspeed_runtime(accelerator, config)
    _set_seed(int(config["training"].get("seed", 42)))
    _prepare_distillation_env(config)

    processor = _load_processor_for_model(str(config["model"]["model_name_or_path"]), config)
    teacher_processor = _prepare_teacher_processor(config, processor)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Qwen3-VL processor did not expose a tokenizer.")

    model, _, _ = _prepare_model(config, accelerator)
    dataset, dataloader = _build_dataloader(config)
    rollout_engine, sampling_params, rollout_processor, _ = _prepare_teacher_rollout_runtime(
        config,
        accelerator,
        teacher_processor,
    )
    renderer = _build_replay_renderer(config)

    model = accelerator.prepare_model(model)
    if rollout_engine is not None:
        _sync_training_model_to_vllm(accelerator, model, rollout_engine)
        accelerator.wait_for_everyone()

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[opd-tr] dry-run dataset_size={len(dataset)} manifest={config['data']['opsd_manifest']}")

    processed = 0
    for rows in dataloader:
        prepared_rows = _build_teacher_prompt_rows(config, rows)
        teacher_rollouts = _generate_teacher_rollouts(
            rollout_engine=rollout_engine,
            sampling_params=sampling_params,
            rollout_processor=rollout_processor,
            prepared_rows=prepared_rows,
            force_think=_teacher_force_think(config),
        )
        training_batch, valid_specs = _prepare_student_training_batch(
            config=config,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            renderer=renderer,
            prepared_rows=prepared_rows,
            teacher_rollouts=teacher_rollouts,
        )
        print(
            f"[opd-tr] dry-run batch={processed} "
            f"valid={len(valid_specs)} "
            f"seq={tuple(training_batch['input_ids'].shape)} "
            f"latents={sum(len(spec['teacher_thinking_chunks']) for spec in valid_specs)}"
        )
        if processed == 0:
            _save_teacher_rollout_samples(output_dir, valid_specs, "dry_run")
        processed += 1
        if processed >= dry_run_batches:
            break

    renderer.shutdown()


def _run_training(config: dict[str, Any]) -> None:
    train_cfg = config["training"]
    _configure_quiet_logging()
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        mixed_precision="bf16" if bool(train_cfg.get("bf16", True)) else "no",
    )
    _configure_deepspeed_runtime(accelerator, config)

    _set_seed(int(train_cfg.get("seed", 42)))
    _prepare_distillation_env(config)

    processor = _load_processor_for_model(str(config["model"]["model_name_or_path"]), config)
    teacher_processor = _prepare_teacher_processor(config, processor)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Qwen3-VL processor did not expose a tokenizer.")

    config.setdefault("tokens", {})
    config["tokens"]["pad_token_id"] = tokenizer.pad_token_id

    model, model_dtype, require_vae_checkpoints = _prepare_model(config, accelerator)
    dataset, dataloader = _build_dataloader(config)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
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

    rollout_engine, rollout_sampling_params, rollout_processor, rollout_sync_steps = _prepare_teacher_rollout_runtime(
        config,
        accelerator,
        teacher_processor,
    )
    renderer = _build_replay_renderer(config)

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(output_dir / "resolved_opsd_config.yaml", config)

    if accelerator.is_main_process:
        if use_epoch_budget and configured_max_steps > 0:
            print(
                f"[opd-tr] ignoring training.max_steps={configured_max_steps} "
                f"because training.num_train_epochs={configured_epochs:g} is set"
            )
        budget_mode = f"epochs={configured_epochs:g}" if use_epoch_budget else f"max_steps={configured_max_steps}"
        print(
            f"[opd-tr] dataset_size={len(dataset)} batch_size={train_cfg['per_device_train_batch_size']} "
            f"grad_accum={train_cfg['gradient_accumulation_steps']} total_steps={total_steps} budget={budget_mode}"
        )
        print(
            f"[opd-tr] student_loss={config['student_loss']['loss_type']} "
            f"teacher_rollout=shared_student "
            f"teacher_vllm_thinking={bool(config['teacher'].get('vllm_thinking', False))}"
        )

    global_step = 0
    micro_step = 0
    start_time = time.time()
    stage_seconds = {"rollout": 0.0, "build": 0.0, "forward": 0.0, "opt": 0.0, "sync": 0.0}
    last_loss_value = 0.0
    last_stats = {
        "samples": 0,
        "latent_steps": 0,
        "seq_len": 0,
        "replay_completion_width": 0,
        "continuous_tokens": 0,
        "continuous_tokens_max": 0,
        "continuous_tokens_avg": 0.0,
    }
    model.train()

    if rollout_engine is not None:
        _sync_training_model_to_vllm(accelerator, model, rollout_engine)
        accelerator.wait_for_everyone()

    for epoch in range(int(train_cfg["num_train_epochs"])):
        for rows in dataloader:
            micro_step += 1
            with accelerator.accumulate(model):
                prepared_rows = _build_teacher_prompt_rows(config, rows)

                stage_start = time.perf_counter()
                teacher_rollouts = _generate_teacher_rollouts(
                    rollout_engine=rollout_engine,
                    sampling_params=rollout_sampling_params,
                    rollout_processor=rollout_processor,
                    prepared_rows=prepared_rows,
                    force_think=_teacher_force_think(config),
                )
                stage_seconds["rollout"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                try:
                    training_batch, valid_specs = _prepare_student_training_batch(
                        config=config,
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        renderer=renderer,
                        prepared_rows=prepared_rows,
                        teacher_rollouts=teacher_rollouts,
                    )
                except RuntimeError as exc:
                    if "No valid teacher rollouts produced usable thinking chunks" not in str(exc):
                        raise
                    if accelerator.is_main_process:
                        print(f"[opd-tr] skipped_batch reason=no_valid_teacher_thinking details={exc}")
                    continue
                stage_seconds["build"] += time.perf_counter() - stage_start

                model_inputs = _move_batch_to_device(
                    {
                        "input_ids": training_batch["input_ids"],
                        "labels": training_batch["labels"],
                        "attention_mask": training_batch["attention_mask"],
                        "vae_ce_mask": training_batch["vae_ce_mask"],
                        "latent_positions": training_batch["latent_positions"],
                        "latent_ground_truth_packed": training_batch["latent_ground_truth_packed"],
                        "pixel_values": training_batch["pixel_values"],
                        "image_grid_thw": training_batch["image_grid_thw"],
                    },
                    accelerator.device,
                    model_dtype,
                )
                latent_ground_truth = [
                    [tensor.to(device=accelerator.device, dtype=model_dtype) for tensor in sample]
                    for sample in training_batch["latent_ground_truth"]
                ]

                stage_start = time.perf_counter()
                outputs = model(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    labels=model_inputs["labels"],
                    pixel_values=model_inputs.get("pixel_values"),
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    logits_to_keep=max(1, int(training_batch.get("replay_completion_width", 0)) + 1),
                    latent_positions=model_inputs["latent_positions"],
                    latent_ground_truth=latent_ground_truth,
                    latent_ground_truth_paths=training_batch["latent_ground_truth_paths"],
                    latent_ground_truth_packed=model_inputs["latent_ground_truth_packed"],
                    latent_supervision=training_batch["latent_supervision"],
                    latent_supervision_paths=training_batch["latent_supervision_paths"],
                    vae_ce_mask=model_inputs["vae_ce_mask"],
                    use_cache=False,
                )
                loss = outputs.loss
                stage_seconds["forward"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                accelerator.backward(loss)
                if float(train_cfg.get("max_grad_norm", 0.0)) > 0:
                    accelerator.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                stage_seconds["opt"] += time.perf_counter() - stage_start

                last_loss_value = float(loss.detach())
                per_sample_continuous_tokens = [
                    sum(int(x) for x in (spec.get("latent_seq_lens") or []))
                    for spec in valid_specs
                ]
                continuous_tokens_total = sum(per_sample_continuous_tokens)
                last_stats = {
                    "samples": len(valid_specs),
                    "latent_steps": sum(len(spec["teacher_thinking_chunks"]) for spec in valid_specs),
                    "seq_len": int(training_batch["input_ids"].shape[1]),
                    "replay_completion_width": int(training_batch.get("replay_completion_width", 0)),
                    "continuous_tokens": continuous_tokens_total,
                    "continuous_tokens_max": max(per_sample_continuous_tokens, default=0),
                    "continuous_tokens_avg": (
                        continuous_tokens_total / float(max(1, len(per_sample_continuous_tokens)))
                    ),
                }

                del loss
                del outputs
                del model_inputs
                del latent_ground_truth

            if accelerator.sync_gradients:
                if rollout_engine is not None and (global_step == 0 or (global_step + 1) % rollout_sync_steps == 0):
                    stage_start = time.perf_counter()
                    _sync_training_model_to_vllm(accelerator, model, rollout_engine)
                    accelerator.wait_for_everyone()
                    stage_seconds["sync"] += time.perf_counter() - stage_start

                global_step += 1

                if accelerator.is_main_process and global_step % int(train_cfg["logging_steps"]) == 0:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    progress = float(global_step) / float(max(1, total_steps))
                    remaining_steps = max(0, total_steps - global_step)
                    seconds_per_step = elapsed / float(max(1, global_step))
                    eta_seconds = seconds_per_step * remaining_steps
                    print(
                        f"[opd-tr] step={global_step}/{total_steps} progress={progress * 100.0:.1f}% "
                        f"micro_step={micro_step} "
                        f"epoch={epoch + 1}/{max(1, int(math.ceil(configured_epochs))) if configured_epochs > 0 else 1} "
                        f"loss={last_loss_value:.6f} lr={lr:.3e} "
                        f"elapsed={elapsed / 3600.0:.2f}h eta={eta_seconds / 3600.0:.2f}h "
                        f"samples={last_stats['samples']} latent_steps={last_stats['latent_steps']} "
                        f"seq_len={last_stats['seq_len']} "
                        f"replay_completion_width={last_stats['replay_completion_width']} "
                        f"continuous_tokens={last_stats['continuous_tokens']} "
                        f"continuous_avg={last_stats['continuous_tokens_avg']:.1f} "
                        f"continuous_max={last_stats['continuous_tokens_max']} "
                        f"timing=rollout:{stage_seconds['rollout']:.1f}s "
                        f"build:{stage_seconds['build']:.1f}s "
                        f"forward:{stage_seconds['forward']:.1f}s "
                        f"opt:{stage_seconds['opt']:.1f}s "
                        f"sync:{stage_seconds['sync']:.1f}s"
                    )
                    stage_seconds = {key: 0.0 for key in stage_seconds}

                if accelerator.is_main_process and global_step == 1:
                    _save_teacher_rollout_samples(output_dir, valid_specs, "step_1")
                if global_step == 1:
                    accelerator.wait_for_everyone()

                if global_step % int(train_cfg["save_steps"]) == 0:
                    accelerator.wait_for_everyone()
                    save_started_at = time.perf_counter()
                    if accelerator.is_main_process:
                        print(f"[opd-tr] checkpoint_save_start step={global_step}")
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
                            f"[opd-tr] checkpoint_save_done step={global_step} "
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
    renderer.shutdown()


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
    if args.max_new_tokens is not None:
        updated.setdefault("generation", {})
        updated["generation"]["max_new_tokens"] = int(args.max_new_tokens)
    if args.overwrite_manifest:
        updated.setdefault("data", {})
        updated["data"]["overwrite_manifest"] = True
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL OPD-VCR with continuous replay.")
    parser.add_argument("config", help="YAML config path")
    parser.add_argument("--datasets", help="Override config data.dataset_names")
    parser.add_argument("--manifest-path", help="Override config data.opsd_manifest")
    parser.add_argument("--output-dir", help="Override config training.output_dir")
    parser.add_argument("--ckpt-path", help="Override config model.adapter_name_or_path")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation.max_new_tokens")
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Force regeneration of the manifest JSONL",
    )
    parser.add_argument(
        "--dry-run-batches",
        type=int,
        default=0,
        help="Run the teacher rollout + continuous replay build path for N batches, then exit.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    config = _load_yaml(config_path)
    config = _apply_overrides(config, args)
    config = _resolve_config_paths(config)
    _maybe_prepare_manifest(config)

    if args.dry_run_batches > 0:
        _run_dry_run(config, args.dry_run_batches)
    else:
        _run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
