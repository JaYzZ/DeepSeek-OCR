#!/usr/bin/env python3
"""Swift dataset registration for the Qwen3-VL OPSD-span manifest."""

from __future__ import annotations

import json
import os
import re
import inspect
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import gather, gather_object, is_peft_model
from swift.dataset import RowPreprocessor, register_dataset
from swift.dataset.register import DatasetMeta
from swift.rlhf_trainers.arguments import GRPOConfig as SwiftGRPOConfig
from swift.rlhf_trainers.gkd_trainer import GKDTrainer, TeacherOutput
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.rlhf_trainers.grpo_trainer import sequence_parallel
from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin
from swift.rlhf_trainers.utils import entropy_from_logits, nanstd
from swift.rlhf_trainers.utils import pad_logps_back_to_batch
from swift.rewards.orm import ORM, orms
from swift.trainers import Seq2SeqTrainer, disable_gradient_checkpointing
from swift.tuners import Swift, peft as swift_peft
from swift.utils import get_logger
from trl import GRPOConfig as TrlGRPOConfig
from trl.trainer.grpo_trainer import nanmax, nanmin
from trl.trainer.utils import selective_log_softmax
from transformers import AutoTokenizer
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from Qwen.inference.vllm_utils import resolve_lora_artifacts

DEFAULT_SYSTEM_PROMPT = ""
DEFAULT_LATENT_TOKEN_WEIGHT = 16.0
DEFAULT_WARMUP_CONTINUATION_KL_STEPS = 4
DEFAULT_WARMUP_CONTINUATION_KL_WEIGHT = 0.1
DEFAULT_WARMUP_COMPRESSED_CE_WEIGHT = 1.0
DEFAULT_WARMUP_FULL_CE_WEIGHT = 1.0
DEFAULT_WARMUP_INJECTION_SCALE = 1.0
DEFAULT_WARMUP_STATE_LOSS_WEIGHT = 0.5
DEFAULT_WARMUP_STATE_COSINE_WEIGHT = 1.0
DEFAULT_WARMUP_STATE_HUBER_WEIGHT = 0.25
DEFAULT_WARMUP_STATE_NCE_WEIGHT = 0.1
DEFAULT_WARMUP_STATE_NCE_TEMPERATURE = 0.1
DEFAULT_WARMUP_CE_CHUNK_SIZE = 1024
DEFAULT_DELTA_MEMORY_ENABLED = True
DEFAULT_DELTA_MEMORY_GAMMA = 0.5
DEFAULT_DELTA_MEMORY_TARGET_WEIGHT = 0.25
DEFAULT_RLSD_ENABLED = False
DEFAULT_RLSD_WEIGHT = 1.0
DEFAULT_RLSD_TEMPERATURE = 1.0
DEFAULT_RLSD_MAX_WEIGHT = 4.0
DEFAULT_RLSD_MIN_WEIGHT = 0.25
DEFAULT_SINGLE_SAMPLE_GRPO_ENABLED = True
LATENT_TOKEN = "<latent>"
THINK_SEP_TOKEN = "<think_sep>"
THINK_END_TOKEN = "</think>"
THINK_START_TOKEN = "<think>"
MULTIMODAL_INPUT_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
)

logger = get_logger()


@dataclass(frozen=True)
class FormatRewardAssessment:
    reward: float
    grade: str
    answer_correct_reward: float
    answer_wrong_reward: float


@dataclass
class ReplaySampleState:
    row_idx: int
    sample_idx: int
    segments: list[tuple[int, int, str]]
    window_ids: torch.Tensor
    window_start: int = 0
    prefix_len: int = 0
    window_pos: int = 0
    segment_idx: int = 0
    boundary_logit: torch.Tensor | None = None
    pending_carry: torch.Tensor | None = None
    memory_state: torch.Tensor | None = None
    cache_state: Any = None

DEFAULT_TEACHER_TEMPLATE = """Problem:
{question_text}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step. Do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above."""

DEFAULT_OPSD_SPAN_STUDENT_TEMPLATE = """Answer the question.

Use <think>...</think> for reasoning. Inside the thinking span, <latent> means a compressed reasoning segment that stands for omitted thinking content. Only use <latent> inside <think>...</think>, and use it when you compress part of the reasoning instead of writing it out fully.

Question:
{question_text}"""


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _strip_text(value: Any) -> str:
    return str(value or "").strip()


def _gather_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Compute causal next-token log probabilities.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size)
        target_ids: Full token ids aligned to the original sequence of shape (batch, seq_len)

    Returns:
        Log probabilities of shape (batch, seq_len - 1), where position t predicts token t+1.
    """
    if logits.shape[1] != target_ids.shape[1]:
        raise ValueError(
            f"logits/target_ids sequence length mismatch: {logits.shape[1]} vs {target_ids.shape[1]}"
        )
    shifted_logits = logits[:, :-1, :]
    shifted_target_ids = target_ids[:, 1:]
    return selective_log_softmax(shifted_logits.float(), shifted_target_ids)


def _resolve_image_paths(image_paths: list[Any]) -> list[str]:
    resolved: list[str] = []
    for raw_path in image_paths:
        path = Path(str(raw_path))
        if path.is_file():
            resolved.append(str(path))
    return resolved


def _build_messages(
    *,
    system_prompt: str,
    user_text: str,
    assistant_text: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})
    return messages


def _patch_qwen3vl_gradient_checkpointing_warning() -> None:
    original_disable = getattr(Qwen3VLVisionModel, "disable_input_require_grads", None)
    if original_disable is None or getattr(original_disable, "_opsd_span_patched", False):
        return

    def disable_input_require_grads(self):
        hook = getattr(self, "_require_grads_hook", None)
        if hook is None:
            return
        hook.remove()
        self._require_grads_hook = None

    disable_input_require_grads._opsd_span_patched = True
    Qwen3VLVisionModel.disable_input_require_grads = disable_input_require_grads


def _get_latent_token_weight() -> float:
    raw = str(os.environ.get("OPSD_LATENT_TOKEN_WEIGHT", DEFAULT_LATENT_TOKEN_WEIGHT)).strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return DEFAULT_LATENT_TOKEN_WEIGHT


def _get_latent_token_id(tokenizer: Any) -> int | None:
    if tokenizer is None:
        return None
    token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    if token_id is None:
        return None
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and int(token_id) == int(unk_token_id):
        return None
    return int(token_id)


def _get_special_token_id(tokenizer: Any, token: str) -> int | None:
    if tokenizer is None:
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and int(token_id) == int(unk_token_id):
        return None
    return int(token_id)


def _get_model_input_embeddings(model: torch.nn.Module) -> Any:
    unwrapped = getattr(model, "module", model)
    return unwrapped.get_input_embeddings()


def _get_trainable_token_indices(tokenizer: Any) -> list[int]:
    token_ids: list[int] = []
    for token in (LATENT_TOKEN, THINK_SEP_TOKEN):
        token_id = _get_special_token_id(tokenizer, token)
        if token_id is not None:
            token_ids.append(int(token_id))
    return sorted(set(token_ids))


_TRAINABLE_TOKEN_INDICES_CACHE: list[int] | None = None
_REWARD_TOKENIZER_CACHE: Any | None = None
_REWARD_DEBUG_LOG_COUNT = 0


def _resolve_trainable_token_indices() -> list[int]:
    global _TRAINABLE_TOKEN_INDICES_CACHE
    if _TRAINABLE_TOKEN_INDICES_CACHE is not None:
        return _TRAINABLE_TOKEN_INDICES_CACHE

    tokenizer_path = str(os.environ.get("OPSD_SPAN_TOKENIZER_PATH", "")).strip()
    if not tokenizer_path:
        _TRAINABLE_TOKEN_INDICES_CACHE = []
        return _TRAINABLE_TOKEN_INDICES_CACHE

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _TRAINABLE_TOKEN_INDICES_CACHE = _get_trainable_token_indices(tokenizer)
    return _TRAINABLE_TOKEN_INDICES_CACHE


def _get_reward_tokenizer() -> Any | None:
    global _REWARD_TOKENIZER_CACHE
    if _REWARD_TOKENIZER_CACHE is not None:
        return _REWARD_TOKENIZER_CACHE

    tokenizer_path = str(os.environ.get("OPSD_SPAN_TOKENIZER_PATH", "")).strip()
    if not tokenizer_path:
        return None

    _REWARD_TOKENIZER_CACHE = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return _REWARD_TOKENIZER_CACHE


def _decode_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    token_ids: list[int] | None = None
    if isinstance(completion, list):
        token_ids = [int(token_id) for token_id in completion]
    elif isinstance(completion, dict):
        raw_token_ids = completion.get("token_ids")
        if isinstance(raw_token_ids, list):
            token_ids = [int(token_id) for token_id in raw_token_ids]

    if token_ids is None:
        return str(completion)

    tokenizer = _get_reward_tokenizer()
    if tokenizer is None:
        return str(completion)
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    if isinstance(tokens, str):
        tokens = [tokens]
    pieces: list[str] = []
    buffer: list[str] = []
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
    for token in tokens:
        if token in special_tokens:
            if buffer:
                pieces.append(tokenizer.convert_tokens_to_string(buffer))
                buffer = []
            pieces.append(token)
        else:
            buffer.append(token)
    if buffer:
        pieces.append(tokenizer.convert_tokens_to_string(buffer))
    return "".join(pieces)


def _log_reward_debug(
    *,
    reward_name: str,
    trainer_state: Any,
    completion_text: str,
    reasoning: str,
    answer: str,
    target: str,
    reward: float,
) -> None:
    global _REWARD_DEBUG_LOG_COUNT
    if _REWARD_DEBUG_LOG_COUNT >= 12:
        return
    step = int(getattr(trainer_state, "global_step", 0) or 0)
    if step > 2:
        return
    _REWARD_DEBUG_LOG_COUNT += 1
    logger.info(
        "[opsd_span reward debug] reward=%s step=%s value=%.4f completion=%r reasoning=%r answer=%r target=%r",
        reward_name,
        step,
        reward,
        completion_text[:500],
        reasoning[:300],
        answer[:200],
        target[:200],
    )


def _build_shifted_labels(labels: torch.Tensor) -> torch.Tensor:
    return torch.roll(labels, shifts=-1, dims=1)


def _get_env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, default)).strip()
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _get_env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _vllm_thinking_enabled(default: bool = False) -> bool:
    return _get_env_bool("VLLM_THINKING", default)


def _prompt_prefills_think() -> bool:
    return not _vllm_thinking_enabled(default=False)


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded.get("input_ids", [])
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def _split_reasoning_answer(text: str) -> tuple[str, str]:
    normalized = _strip_text(text)
    if not normalized:
        return "", ""
    match = re.search(r"<think>(.*?)</think>(.*)", normalized, re.DOTALL)
    if match:
        return _strip_text(match.group(1)), _strip_text(match.group(2))
    closing_idx = normalized.find(THINK_END_TOKEN)
    if closing_idx >= 0:
        reasoning = normalized[:closing_idx]
        answer = normalized[closing_idx + len(THINK_END_TOKEN) :]
        return _strip_text(reasoning), _strip_text(answer)
    if THINK_START_TOKEN in normalized:
        start_idx = normalized.find(THINK_START_TOKEN) + len(THINK_START_TOKEN)
        return _strip_text(normalized[start_idx:]), ""
    return normalized, ""


def _extract_reasoning_region(text: str) -> str:
    normalized = _strip_text(text)
    if not normalized:
        return ""
    start_idx = normalized.find(THINK_START_TOKEN)
    if start_idx < 0:
        start_idx = 0
    else:
        start_idx += len(THINK_START_TOKEN)
    end_idx = normalized.find(THINK_END_TOKEN, start_idx)
    if end_idx < 0:
        return _strip_text(normalized[start_idx:])
    return _strip_text(normalized[start_idx:end_idx])


def _resolve_target_answer(
    idx: int,
    *,
    answer_text: list[Any],
    compressed_targets: list[Any],
    full_targets: list[Any],
) -> str:
    candidates = [
        answer_text[idx] if idx < len(answer_text) else "",
        _split_reasoning_answer(_strip_text(full_targets[idx] if idx < len(full_targets) else ""))[1],
        _split_reasoning_answer(_strip_text(compressed_targets[idx] if idx < len(compressed_targets) else ""))[1],
    ]
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", str(candidate or "")).strip().rstrip(".")
        if normalized:
            return normalized
    return ""


def _normalize_answer_text(text: str) -> str:
    normalized = _strip_text(text)
    if not normalized:
        return ""
    normalized = normalized.replace("**", " ")
    normalized = re.sub(r"^Answer\s*:\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^The correct answer is\s*:?[\s]*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^The answer is\s*:?[\s]*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.rstrip(".")
    return normalized


def _extract_choice_letter(text: str, allowed_letters: set[str] | None = None) -> str | None:
    normalized = _normalize_answer_text(text)
    if not normalized:
        return None

    patterns = [
        r"^(?:option|choice)?\s*([A-Z])$",
        r"^(?:option|choice)?\s*([A-Z])[\.\):,\s].*$",
        r"^([A-Z])[\.\):,\s].*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).upper()
            if allowed_letters is None or candidate in allowed_letters:
                return candidate
    return None


def _extract_numeric_answer(text: str) -> str | None:
    normalized = _normalize_answer_text(text)
    if not normalized:
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
        return normalized
    return None


def _extract_short_phrase(text: str) -> str | None:
    normalized = _normalize_answer_text(text)
    if not normalized:
        return None
    if len(normalized) > 96:
        return None
    if any(marker in normalized for marker in ("\n", "Explanation:", "Reasoning:", "Step-by-Step")):
        return None
    return normalized.casefold()


def _answers_match_strict(predicted_answer: str, target_answer: str) -> bool:
    predicted = _normalize_answer_text(predicted_answer)
    target = _normalize_answer_text(target_answer)
    if not predicted or not target:
        return False

    target_choice = _extract_choice_letter(target)
    allowed_letters = {target_choice} if target_choice is not None else None
    predicted_choice = _extract_choice_letter(predicted, allowed_letters=allowed_letters)
    if predicted_choice is not None or target_choice is not None:
        return predicted_choice is not None and predicted_choice == target_choice

    predicted_number = _extract_numeric_answer(predicted)
    target_number = _extract_numeric_answer(target)
    if predicted_number is not None or target_number is not None:
        return predicted_number is not None and predicted_number == target_number

    predicted_phrase = _extract_short_phrase(predicted)
    target_phrase = _extract_short_phrase(target)
    if predicted_phrase is not None and target_phrase is not None:
        return predicted_phrase == target_phrase

    return predicted.casefold() == target.casefold()


def _count_visible_latents_in_think(text: str) -> int:
    return _extract_reasoning_region(text).count(LATENT_TOKEN)


def _count_reasoning_units(text: str) -> int:
    reasoning = _extract_reasoning_region(text)
    if not reasoning:
        return 0
    return len(re.findall(rf"{re.escape(LATENT_TOKEN)}|[^\s]+", reasoning))


def _max_consecutive_latent_run(text: str) -> int:
    reasoning = _extract_reasoning_region(text)
    if not reasoning:
        return 0
    max_run = 0
    current_run = 0
    for unit in re.findall(rf"{re.escape(LATENT_TOKEN)}|[^\s]+", reasoning):
        if unit == LATENT_TOKEN:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _missing_think_end(text: str) -> bool:
    normalized = _strip_text(text)
    if not normalized:
        return False
    return THINK_END_TOKEN not in normalized


def _has_latent_after_think(text: str) -> bool:
    normalized = _strip_text(text)
    if not normalized:
        return False
    closing_idx = normalized.find(THINK_END_TOKEN)
    if closing_idx < 0:
        return False
    trailing = normalized[closing_idx + len(THINK_END_TOKEN) :]
    return LATENT_TOKEN in trailing


def _has_balanced_think_block(text: str) -> bool:
    normalized = _strip_text(text)
    if not normalized:
        return False
    start_idx = normalized.find(THINK_START_TOKEN)
    end_idx = normalized.find(THINK_END_TOKEN)
    if end_idx < 0:
        return False
    if start_idx < 0:
        return _prompt_prefills_think()
    return end_idx > start_idx


def _assess_latent_format(
    *,
    text: str,
    reference_text: str,
) -> FormatRewardAssessment:
    has_balanced_think = _has_balanced_think_block(text)
    missing_think_end = _missing_think_end(text)
    has_latent_after_think = _has_latent_after_think(text)
    reference_latents = _count_visible_latents_in_think(reference_text)
    expect_latent = reference_latents > 0
    latent_count = _count_visible_latents_in_think(text)
    reasoning_units = _count_reasoning_units(text)
    latent_ratio = float(latent_count) / float(max(reasoning_units, 1))
    max_latent_run = _max_consecutive_latent_run(text)
    severe_format_error = missing_think_end or has_latent_after_think
    if severe_format_error:
        penalty = -1.2
        if missing_think_end:
            penalty -= 0.3
        if has_latent_after_think:
            penalty -= 0.5
        if max_latent_run >= 4:
            penalty -= min(0.5, 0.1 * float(max_latent_run - 3))
        return FormatRewardAssessment(
            reward=float(penalty),
            grade="bad",
            answer_correct_reward=0.0,
            answer_wrong_reward=0.0,
        )

    if expect_latent and latent_count == 0:
        penalty = -0.9
        if reasoning_units >= 24:
            penalty -= 0.3
        return FormatRewardAssessment(
            reward=float(penalty),
            grade="bad",
            answer_correct_reward=0.05,
            answer_wrong_reward=0.0,
        )

    if not expect_latent and latent_count > 0:
        penalty = -0.5 - min(0.5, 0.1 * float(latent_count))
        return FormatRewardAssessment(
            reward=float(penalty),
            grade="bad",
            answer_correct_reward=0.05,
            answer_wrong_reward=0.0,
        )

    if max_latent_run >= 4 or (reasoning_units > 0 and latent_ratio > 0.35):
        penalty = -0.3
        if max_latent_run >= 4:
            penalty -= min(0.4, 0.08 * float(max_latent_run - 3))
        if reasoning_units > 0 and latent_ratio > 0.35:
            penalty -= min(0.5, 1.1 * float(latent_ratio - 0.35))
        return FormatRewardAssessment(
            reward=float(penalty),
            grade="bad",
            answer_correct_reward=0.05,
            answer_wrong_reward=0.0,
        )

    gap = abs(latent_count - reference_latents)
    if expect_latent:
        if gap == 0:
            return FormatRewardAssessment(
                reward=0.8,
                grade="good",
                answer_correct_reward=1.0,
                answer_wrong_reward=0.0,
            )
        if gap <= 2:
            return FormatRewardAssessment(
                reward=0.35,
                grade="borderline",
                answer_correct_reward=0.25,
                answer_wrong_reward=0.0,
            )
        return FormatRewardAssessment(
            reward=-0.2,
            grade="bad",
            answer_correct_reward=0.1,
            answer_wrong_reward=0.0,
        )

    return FormatRewardAssessment(
        reward=0.4,
        grade="good",
        answer_correct_reward=1.0,
        answer_wrong_reward=0.0,
    )


def _find_subsequence(sequence: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    last_start = len(sequence) - len(pattern)
    for start_idx in range(last_start + 1):
        if sequence[start_idx : start_idx + len(pattern)] == pattern:
            return start_idx
    return -1


def _build_full_target(spans: list[str], answer_text: str, fallback_thinking: str) -> str:
    if spans:
        think_body = "\n".join(span for span in spans if span)
    else:
        think_body = fallback_thinking
    return f"{THINK_START_TOKEN}{think_body}{THINK_END_TOKEN}{answer_text}".strip()


def _build_full_span_token_ranges(tokenizer: Any, spans: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    prefix = THINK_START_TOKEN
    for idx, span in enumerate(spans):
        if idx > 0:
            prefix += "\n"
        start_idx = len(_tokenize_text(tokenizer, prefix))
        prefix += span
        end_idx = len(_tokenize_text(tokenizer, prefix)) - 1
        ranges.append((start_idx, end_idx))
    return ranges


def _find_response_content_start(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    assistant_token_ids: list[int],
) -> int | None:
    valid_positions = torch.nonzero(labels != -100, as_tuple=False).flatten().tolist()
    if not valid_positions or not assistant_token_ids:
        return None
    response_token_ids = [int(input_ids[pos].item()) for pos in valid_positions]
    relative_start = _find_subsequence(response_token_ids, assistant_token_ids)
    if relative_start < 0:
        return None
    return int(valid_positions[relative_start])


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _filter_model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "attention_mask",
        "cache_position",
        "image_grid_thw",
        "input_ids",
        "inputs_embeds",
        "labels",
        "logits_to_keep",
        "past_key_values",
        "pixel_values",
        "pixel_values_videos",
        "position_ids",
        "second_per_grid_ts",
        "video_grid_thw",
    }
    return {key: value for key, value in batch.items() if key in allowed_keys}


def _zero_loss_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _unwrap_model(model: Any) -> Any:
    current = model
    while hasattr(current, "module"):
        current = getattr(current, "module")
    return current


def _get_last_text_attention(model: Any) -> Any | None:
    current = _unwrap_model(model)
    qwen_model = getattr(current, "model", None)
    if qwen_model is None:
        return None
    language_model = getattr(qwen_model, "language_model", None)
    if language_model is None:
        return None
    layers = getattr(language_model, "layers", None)
    if not layers:
        return None
    last_layer = layers[-1]
    return getattr(last_layer, "self_attn", None)


def _get_module_dtype(module: Any, default: torch.dtype) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return default


def _get_linear_out_features(module: Any) -> int:
    out_features = getattr(module, "out_features", None)
    if out_features is not None:
        return int(out_features)
    base_layer = getattr(module, "base_layer", None)
    if base_layer is not None:
        out_features = getattr(base_layer, "out_features", None)
        if out_features is not None:
            return int(out_features)
    weight = getattr(module, "weight", None)
    if weight is not None:
        return int(weight.shape[0])
    if base_layer is not None:
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            return int(weight.shape[0])
    raise AttributeError("Unable to infer linear out_features")


def _normalize_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    normalized = F.layer_norm(hidden_states.float(), (hidden_states.shape[-1],))
    return F.normalize(normalized, dim=-1)


def _compute_latent_state_cosine_loss(predicted_states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
    pred = _normalize_hidden_states(predicted_states)
    tgt = _normalize_hidden_states(target_states)
    return (1.0 - (pred * tgt).sum(dim=-1)).mean()


def _attention_pool_span_hidden(query_state: torch.Tensor, span_hidden: torch.Tensor) -> torch.Tensor:
    query = query_state.float()
    span = span_hidden.float()
    scale = float(span.shape[-1]) ** 0.5
    scores = torch.matmul(span, query.unsqueeze(-1)).squeeze(-1) / scale
    weights = F.softmax(scores, dim=0)
    return torch.sum(weights.unsqueeze(-1) * span, dim=0)


def _last_block_cross_attention_pool(
    self_attn: Any,
    query_state: torch.Tensor,
    span_hidden: torch.Tensor,
) -> torch.Tensor:
    proj_dtype = _get_module_dtype(self_attn.q_proj, query_state.dtype)
    hidden_size = int(query_state.shape[-1])
    head_dim = int(getattr(self_attn, "head_dim"))
    num_heads = hidden_size // head_dim
    kv_hidden_size = _get_linear_out_features(self_attn.k_proj)
    num_key_value_heads = kv_hidden_size // head_dim
    num_key_value_groups = max(num_heads // max(num_key_value_heads, 1), 1)

    query_input = query_state.unsqueeze(0).to(dtype=proj_dtype)
    span_input = span_hidden.to(dtype=proj_dtype)

    query = self_attn.q_proj(query_input).view(1, num_heads, head_dim)
    key = self_attn.k_proj(span_input).view(span_hidden.shape[0], num_key_value_heads, head_dim)
    value = self_attn.v_proj(span_input).view(span_hidden.shape[0], num_key_value_heads, head_dim)

    query = self_attn.q_norm(query).transpose(0, 1)
    key = self_attn.k_norm(key).transpose(0, 1)
    value = value.transpose(0, 1)
    if num_key_value_heads != num_heads:
        key = key.repeat_interleave(num_key_value_groups, dim=0)
        value = value.repeat_interleave(num_key_value_groups, dim=0)

    scores = torch.matmul(query, key.transpose(-1, -2)) * float(getattr(self_attn, "scaling"))
    weights = F.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
    pooled = torch.matmul(weights, value).transpose(0, 1).reshape(1, hidden_size)
    return self_attn.o_proj(pooled).squeeze(0).to(dtype=query_state.dtype)


def _compute_multi_positive_nce_loss(
    predicted_states: torch.Tensor,
    pre_targets: torch.Tensor,
    post_targets: torch.Tensor,
) -> torch.Tensor:
    if predicted_states.shape[0] == 0:
        return _zero_loss_like(predicted_states)

    pred = _normalize_hidden_states(predicted_states)
    pre = _normalize_hidden_states(pre_targets)
    post = _normalize_hidden_states(post_targets)
    temperature = max(
        _get_env_float("OPSD_WARMUP_STATE_NCE_TEMPERATURE", DEFAULT_WARMUP_STATE_NCE_TEMPERATURE),
        1e-6,
    )
    target_bank = torch.cat([pre, post], dim=0)
    logits = torch.matmul(pred, target_bank.transpose(0, 1)) / temperature
    num_items = pred.shape[0]
    row_index = torch.arange(num_items, device=logits.device, dtype=torch.long)
    positive_logits = torch.stack(
        [
            logits[row_index, row_index],
            logits[row_index, row_index + num_items],
        ],
        dim=1,
    )
    log_pos = torch.logsumexp(positive_logits, dim=1)
    log_denom = torch.logsumexp(logits, dim=1)
    return (log_denom - log_pos).mean()


def _compute_latent_state_triplet_losses(
    predicted_states: torch.Tensor,
    pre_targets: torch.Tensor,
    post_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = _normalize_hidden_states(predicted_states)
    pre = _normalize_hidden_states(pre_targets)
    post = _normalize_hidden_states(post_targets)

    cosine_loss = 0.5 * (
        (1.0 - (pred * pre).sum(dim=-1)).mean()
        + (1.0 - (pred * post).sum(dim=-1)).mean()
    )
    huber_loss = 0.5 * (
        F.smooth_l1_loss(pred, pre, reduction="mean")
        + F.smooth_l1_loss(pred, post, reduction="mean")
    )
    nce_loss = _compute_multi_positive_nce_loss(predicted_states, pre_targets, post_targets)

    total_loss = (
        cosine_loss * _get_env_float("OPSD_WARMUP_STATE_COSINE_WEIGHT", DEFAULT_WARMUP_STATE_COSINE_WEIGHT)
        + huber_loss * _get_env_float("OPSD_WARMUP_STATE_HUBER_WEIGHT", DEFAULT_WARMUP_STATE_HUBER_WEIGHT)
        + nce_loss * _get_env_float("OPSD_WARMUP_STATE_NCE_WEIGHT", DEFAULT_WARMUP_STATE_NCE_WEIGHT)
    )
    return total_loss, cosine_loss, huber_loss, nce_loss


def _delta_memory_enabled() -> bool:
    return _get_env_bool("OPSD_DELTA_MEMORY_ENABLED", DEFAULT_DELTA_MEMORY_ENABLED)


def _delta_memory_gamma() -> float:
    return _get_env_float("OPSD_DELTA_MEMORY_GAMMA", DEFAULT_DELTA_MEMORY_GAMMA)


def _opsd_span_replay_mode() -> str:
    """Get the segment replay mode from environment."""
    return _strip_text(os.environ.get("OPSD_SPAN_REPLAY_MODE", "normal")).lower()


def _opsd_span_replay_delta_mode() -> str:
    """Get the segment replay delta mode from environment."""
    return _strip_text(os.environ.get("OPSD_SPAN_REPLAY_DELTA_MODE", "follow_global")).lower()


def _opsd_span_replay_legit_latent_count_max() -> int:
    """Get the maximum number of latent tokens considered legit for replay."""
    raw = str(os.environ.get("OPSD_SPAN_REPLAY_LEGIT_LATENT_COUNT_MAX", "12")).strip()
    try:
        return max(int(raw), 0)
    except ValueError:
        return 12


def _opsd_span_segment_replay_enabled() -> bool:
    """Check if segment replay is enabled for the current stage."""
    stage = _strip_text(os.environ.get("OPSD_SPAN_STAGE", "main")).lower()
    replay_mode = _opsd_span_replay_mode()
    return stage in {"gspo", "main"} and replay_mode == "segment_mixed"


def _apply_delta_memory_update(
    memory_state: torch.Tensor | None,
    latent_state: torch.Tensor,
) -> torch.Tensor:
    latent = latent_state.float()
    if memory_state is None:
        return latent
    gamma = _delta_memory_gamma()
    return memory_state.float() + gamma * (latent - memory_state.float())


def _compute_delta_memory_target(
    span_hidden: torch.Tensor,
) -> torch.Tensor:
    memory_state: torch.Tensor | None = None
    for idx in range(span_hidden.shape[0]):
        memory_state = _apply_delta_memory_update(memory_state, span_hidden[idx])
    if memory_state is None:
        return span_hidden.new_zeros((span_hidden.shape[-1],), dtype=torch.float32)
    return memory_state


def _compute_continuation_kl_loss(
    *,
    compressed_logits: torch.Tensor,
    full_logits: torch.Tensor,
    compressed_labels: torch.Tensor,
    compressed_keep_positions: torch.Tensor,
    full_keep_positions: torch.Tensor,
    alignments: list[dict[str, Any]],
) -> tuple[torch.Tensor, int]:
    kl_horizon = max(int(_get_env_float("OPSD_WARMUP_CONTINUATION_KL_STEPS", DEFAULT_WARMUP_CONTINUATION_KL_STEPS)), 0)
    if kl_horizon <= 0:
        return _zero_loss_like(compressed_logits), 0

    if compressed_keep_positions.numel() == 0 or full_keep_positions.numel() == 0:
        return _zero_loss_like(compressed_logits.float()), 0

    compressed_position_map = {int(pos): idx for idx, pos in enumerate(compressed_keep_positions.tolist())}
    full_position_map = {int(pos): idx for idx, pos in enumerate(full_keep_positions.tolist())}
    batch_indices: list[int] = []
    compressed_indices: list[int] = []
    full_indices: list[int] = []

    for sample in alignments:
        batch_idx = int(sample["batch_idx"])
        for item in sample["items"]:
            if not bool(item["has_continuation"]):
                continue
            cmp_start = int(item["next_position"]) - 1
            full_start = int(item["full_span_end"])
            for step_idx in range(kl_horizon):
                cmp_pos = cmp_start + step_idx
                full_pos = full_start + step_idx
                next_label_pos = cmp_pos + 1
                if cmp_pos < 0 or full_pos < 0:
                    break
                if next_label_pos >= compressed_labels.shape[1]:
                    break
                if int(compressed_labels[batch_idx, next_label_pos].item()) == -100:
                    break
                compressed_idx = compressed_position_map.get(cmp_pos)
                full_idx = full_position_map.get(full_pos)
                if compressed_idx is None or full_idx is None:
                    continue
                batch_indices.append(batch_idx)
                compressed_indices.append(compressed_idx)
                full_indices.append(full_idx)

    token_count = len(batch_indices)
    if token_count == 0:
        return _zero_loss_like(compressed_logits.float()), 0

    batch_index_tensor = torch.tensor(batch_indices, device=compressed_logits.device, dtype=torch.long)
    compressed_index_tensor = torch.tensor(compressed_indices, device=compressed_logits.device, dtype=torch.long)
    full_index_tensor = torch.tensor(full_indices, device=full_logits.device, dtype=torch.long)
    student_logits = compressed_logits[batch_index_tensor, compressed_index_tensor].float()
    teacher_logits = full_logits[batch_index_tensor, full_index_tensor].detach().float()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return loss, token_count


def _record_metric(trainer: Any, key: str, value: float) -> None:
    mode = "train" if trainer.model.training else "eval"
    custom_metrics = getattr(trainer, "custom_metrics", {}).get(mode)
    if custom_metrics is None:
        return
    custom_metrics[key].update(value)


def _record_warmup_debug(trainer: Any, reason: str) -> None:
    _record_metric(trainer, f"opsd_span/warmup_debug_{reason}", 1.0)


def _build_warmup_full_batch(trainer: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    full_messages = [row for row in list(inputs.get("opsd_full_messages") or []) if isinstance(row, list)]
    if full_messages:
        full_rows: list[dict[str, Any]] = []
        image_paths = list(inputs.get("opsd_image_paths") or [])
        for idx, messages in enumerate(full_messages):
            row = {"messages": messages}
            if idx < len(image_paths):
                row["images"] = image_paths[idx]
            full_rows.append(row)
        encoded_rows = [trainer.template.encode(row, return_length=True) for row in full_rows]
        collated = trainer.template.data_collator(encoded_rows)
        device = getattr(trainer.args, "device", None) or next(trainer.model.parameters()).device
        return _move_batch_to_device(collated, device)

    student_users = list(inputs.get("opsd_student_user") or [])
    full_targets = list(inputs.get("opsd_full_target") or [])
    image_paths = list(inputs.get("opsd_image_paths") or [])
    system_prompts = list(inputs.get("opsd_system_prompt") or [])
    if not student_users or not full_targets:
        return {}

    full_rows: list[dict[str, Any]] = []
    for idx, student_user in enumerate(student_users):
        if idx >= len(full_targets):
            break
        row = {
            "messages": _build_messages(
                system_prompt=_strip_text(system_prompts[idx] if idx < len(system_prompts) else ""),
                user_text=_strip_text(student_user),
                assistant_text=_strip_text(full_targets[idx]),
            )
        }
        if idx < len(image_paths):
            row["images"] = image_paths[idx]
        full_rows.append(row)

    encoded_rows = [trainer.template.encode(row, return_length=True) for row in full_rows]
    collated = trainer.template.data_collator(encoded_rows)
    device = getattr(trainer.args, "device", None) or next(trainer.model.parameters()).device
    return _move_batch_to_device(collated, device)


def _get_shift_label_positions(labels: torch.Tensor) -> torch.Tensor:
    valid_any = (labels[:, 1:] != -100).any(dim=0)
    return torch.nonzero(valid_any, as_tuple=False).flatten()


def _build_latent_alignment(
    *,
    tokenizer: Any,
    compressed_input_ids: torch.Tensor,
    compressed_labels: torch.Tensor,
    full_input_ids: torch.Tensor,
    full_labels: torch.Tensor,
    compressed_targets: list[str],
    full_targets: list[str],
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latent_token_id = _get_latent_token_id(tokenizer)
    think_end_token_id = _get_special_token_id(tokenizer, THINK_END_TOKEN)
    if latent_token_id is None:
        return []

    alignments: list[dict[str, Any]] = []
    batch_size = compressed_input_ids.shape[0]
    for batch_idx in range(batch_size):
        compressed_target = _strip_text(compressed_targets[batch_idx] if batch_idx < len(compressed_targets) else "")
        full_target = _strip_text(full_targets[batch_idx] if batch_idx < len(full_targets) else "")
        trace = traces[batch_idx] if batch_idx < len(traces) and isinstance(traces[batch_idx], dict) else {}
        spans = [str(span) for span in list(trace.get("spans") or []) if str(span).strip()]
        compressed_spans = list(trace.get("compressed_spans") or [])
        if not compressed_target or not full_target or not compressed_spans:
            continue

        compressed_target_ids = _tokenize_text(tokenizer, compressed_target)
        full_target_ids = _tokenize_text(tokenizer, full_target)
        compressed_content_start = _find_response_content_start(
            input_ids=compressed_input_ids[batch_idx],
            labels=compressed_labels[batch_idx],
            assistant_token_ids=compressed_target_ids,
        )
        full_content_start = _find_response_content_start(
            input_ids=full_input_ids[batch_idx],
            labels=full_labels[batch_idx],
            assistant_token_ids=full_target_ids,
        )
        if compressed_content_start is None or full_content_start is None:
            continue

        latent_offsets = [idx for idx, token_id in enumerate(compressed_target_ids) if int(token_id) == latent_token_id]
        if len(latent_offsets) != len(compressed_spans):
            continue

        full_span_ranges = _build_full_span_token_ranges(tokenizer, spans)
        sample_alignment: list[dict[str, int | bool]] = []
        for latent_idx, span_range in enumerate(compressed_spans):
            if latent_idx >= len(latent_offsets):
                break
            if not isinstance(span_range, (list, tuple)) or len(span_range) != 2:
                continue
            start_span_idx = int(span_range[0])
            end_span_idx = int(span_range[1])
            if start_span_idx < 0 or end_span_idx >= len(full_span_ranges) or start_span_idx > end_span_idx:
                continue

            latent_offset = latent_offsets[latent_idx]
            latent_position = int(compressed_content_start + latent_offset)
            next_position = latent_position + 1
            if next_position >= compressed_input_ids.shape[1]:
                has_continuation = False
            else:
                has_continuation = bool(compressed_labels[batch_idx, next_position].item() != -100)
                if has_continuation and think_end_token_id is not None:
                    has_continuation = int(compressed_input_ids[batch_idx, next_position].item()) != int(think_end_token_id)

            full_span_start = int(full_content_start + full_span_ranges[start_span_idx][0])
            full_span_end = int(full_content_start + full_span_ranges[end_span_idx][1])
            sample_alignment.append(
                {
                    "latent_position": latent_position,
                    "next_position": next_position,
                    "full_span_start": full_span_start,
                    "full_span_end": full_span_end,
                    "has_continuation": has_continuation,
                }
            )

        if sample_alignment:
            alignments.append({"batch_idx": batch_idx, "items": sample_alignment})
    return alignments


def _compute_warmup_loss(
    trainer: Any,
    model: Any,
    inputs: dict[str, Any],
) -> tuple[torch.Tensor, None]:
    tokenizer = getattr(trainer, "tokenizer", None)
    latent_token_id = _get_latent_token_id(tokenizer)
    compressed_labels = inputs.get("labels")
    compressed_input_ids = inputs.get("input_ids")
    if tokenizer is None or latent_token_id is None or compressed_labels is None or compressed_input_ids is None:
        _record_warmup_debug(trainer, "missing_core_inputs")
        return None, None

    full_batch = _build_warmup_full_batch(trainer, inputs)
    full_labels = full_batch.get("labels")
    full_input_ids = full_batch.get("input_ids")
    if full_labels is None or full_input_ids is None:
        _record_warmup_debug(trainer, "missing_full_batch")
        return None, None

    compressed_targets = list(inputs.get("opsd_compressed_target") or [])
    full_targets = list(inputs.get("opsd_full_target") or [])
    traces = list(inputs.get("opsd_compressed_trace") or [])
    alignments = _build_latent_alignment(
        tokenizer=tokenizer,
        compressed_input_ids=compressed_input_ids,
        compressed_labels=compressed_labels,
        full_input_ids=full_input_ids,
        full_labels=full_labels,
        compressed_targets=compressed_targets,
        full_targets=full_targets,
        traces=traces,
    )
    if not alignments:
        _record_warmup_debug(trainer, "empty_alignments")
        return None, None

    full_keep_positions = _get_shift_label_positions(full_labels)
    if full_keep_positions.numel() == 0:
        _record_warmup_debug(trainer, "empty_full_keep_positions")
        return None, None

    full_model_inputs = _filter_model_inputs(full_batch)
    full_model_inputs.pop("labels", None)
    full_model_inputs["output_hidden_states"] = True
    full_model_inputs["logits_to_keep"] = full_keep_positions
    with torch.no_grad():
        full_outputs = model(**full_model_inputs)
    full_hidden_states = getattr(full_outputs, "hidden_states", None)
    if not full_hidden_states:
        _record_warmup_debug(trainer, "missing_full_hidden_states")
        return None, None
    full_hidden = full_hidden_states[-1].detach()
    full_logits = full_outputs.logits
    full_outputs.hidden_states = None

    base_model = _unwrap_model(model)
    last_text_attention = _get_last_text_attention(model)
    if last_text_attention is None:
        _record_warmup_debug(trainer, "missing_last_text_attention")
        return None, None
    input_embeddings = base_model.get_input_embeddings()(compressed_input_ids)
    latent_pre_targets: list[torch.Tensor] = []
    latent_post_targets: list[torch.Tensor] = []
    latent_memory_targets: list[torch.Tensor] = []
    compressed_indices: list[tuple[int, int]] = []
    injection_indices: list[tuple[int, int]] = []
    injection_targets: list[torch.Tensor] = []
    memory_state_by_sample: dict[int, torch.Tensor] = {}
    for sample in alignments:
        batch_idx = int(sample["batch_idx"])
        for item in sample["items"]:
            start_idx = int(item["full_span_start"])
            end_idx = int(item["full_span_end"])
            if end_idx < start_idx:
                continue
            span_hidden = full_hidden[batch_idx, start_idx : end_idx + 1]
            if span_hidden.shape[0] == 0:
                continue
            pre_query_idx = max(0, start_idx - 1)
            post_query_idx = min(full_hidden.shape[1] - 1, end_idx)
            pre_target_state = _attention_pool_span_hidden(
                full_hidden[batch_idx, pre_query_idx],
                span_hidden,
            ).detach()
            post_target_state = _attention_pool_span_hidden(
                full_hidden[batch_idx, post_query_idx],
                span_hidden,
            ).detach()
            latent_position = int(item["latent_position"])
            latent_query_state = input_embeddings[batch_idx, latent_position]
            carry_target_state = _last_block_cross_attention_pool(
                last_text_attention,
                latent_query_state,
                span_hidden,
            )
            if _delta_memory_enabled():
                previous_memory = memory_state_by_sample.get(batch_idx)
                memory_target_state = _compute_delta_memory_target(span_hidden)
                updated_memory_state = _apply_delta_memory_update(previous_memory, memory_target_state)
                memory_state_by_sample[batch_idx] = updated_memory_state.detach()
                carry_target_state = updated_memory_state.to(dtype=carry_target_state.dtype, device=carry_target_state.device)
                latent_memory_targets.append(updated_memory_state.detach().to(dtype=carry_target_state.dtype, device=carry_target_state.device))
            else:
                latent_memory_targets.append(carry_target_state.detach())
            compressed_indices.append((batch_idx, latent_position))
            latent_pre_targets.append(pre_target_state)
            latent_post_targets.append(post_target_state)
            if bool(item["has_continuation"]):
                injection_indices.append((batch_idx, int(item["next_position"])))
                injection_targets.append(carry_target_state)

    del full_hidden_states
    del full_hidden

    if not latent_pre_targets or not latent_post_targets or not compressed_indices:
        _record_warmup_debug(trainer, "empty_latent_targets")
        return None, None

    latent_pre_target_tensor = torch.stack(latent_pre_targets, dim=0)
    latent_post_target_tensor = torch.stack(latent_post_targets, dim=0)
    latent_memory_target_tensor = torch.stack(latent_memory_targets, dim=0)
    injection_scale = _get_env_float("OPSD_WARMUP_INJECTION_SCALE", DEFAULT_WARMUP_INJECTION_SCALE)
    if injection_indices:
        flat_index = torch.tensor(
            [batch_idx * input_embeddings.shape[1] + token_idx for batch_idx, token_idx in injection_indices],
            device=input_embeddings.device,
            dtype=torch.long,
        )
        flat_embeds = input_embeddings.view(-1, input_embeddings.shape[-1])
        injection_target_tensor = torch.stack(injection_targets, dim=0).to(flat_embeds.dtype)
        flat_embeds.index_add_(0, flat_index, injection_target_tensor * injection_scale)
        input_embeddings = flat_embeds.view_as(input_embeddings)

    compressed_keep_positions = _get_shift_label_positions(compressed_labels)
    if compressed_keep_positions.numel() == 0:
        _record_warmup_debug(trainer, "empty_compressed_keep_positions")
        return None, None

    compressed_model_inputs = _filter_model_inputs(inputs)
    compressed_model_inputs.pop("labels", None)
    compressed_model_inputs["inputs_embeds"] = input_embeddings
    compressed_model_inputs["output_hidden_states"] = True
    compressed_model_inputs["logits_to_keep"] = compressed_keep_positions
    compressed_outputs = model(**compressed_model_inputs)
    compressed_hidden_states = getattr(compressed_outputs, "hidden_states", None)
    if not compressed_hidden_states:
        _record_warmup_debug(trainer, "missing_compressed_hidden_states")
        return None, None
    compressed_hidden = compressed_hidden_states[-1]
    compressed_logits = compressed_outputs.logits
    compressed_outputs.hidden_states = None
    compressed_ce_loss, compressed_metrics = _compute_sparse_weighted_token_ce(
        logits=compressed_logits,
        labels=compressed_labels,
        keep_positions=compressed_keep_positions,
        latent_token_id=latent_token_id,
        latent_token_weight=_get_latent_token_weight(),
    )
    gathered_states = torch.stack(
        [compressed_hidden[batch_idx, token_idx] for batch_idx, token_idx in compressed_indices],
        dim=0,
    )
    del compressed_hidden_states
    del compressed_hidden
    state_loss, state_cosine_loss, state_huber_loss, state_nce_loss = _compute_latent_state_triplet_losses(
        gathered_states,
        latent_pre_target_tensor,
        latent_post_target_tensor,
    )
    delta_memory_loss = _zero_loss_like(gathered_states.float())
    if _delta_memory_enabled():
        delta_memory_loss = _compute_latent_state_cosine_loss(gathered_states, latent_memory_target_tensor)
    continuation_kl_loss, continuation_kl_tokens = _compute_continuation_kl_loss(
        compressed_logits=compressed_logits,
        full_logits=full_logits,
        compressed_labels=compressed_labels,
        compressed_keep_positions=compressed_keep_positions,
        full_keep_positions=full_keep_positions,
        alignments=alignments,
    )

    compressed_ce_weight = _get_env_float("OPSD_WARMUP_COMPRESSED_CE_WEIGHT", DEFAULT_WARMUP_COMPRESSED_CE_WEIGHT)
    state_loss_weight = _get_env_float("OPSD_WARMUP_STATE_LOSS_WEIGHT", DEFAULT_WARMUP_STATE_LOSS_WEIGHT)
    continuation_kl_weight = _get_env_float(
        "OPSD_WARMUP_CONTINUATION_KL_WEIGHT",
        DEFAULT_WARMUP_CONTINUATION_KL_WEIGHT,
    )
    delta_memory_weight = _get_env_float(
        "OPSD_DELTA_MEMORY_TARGET_WEIGHT",
        DEFAULT_DELTA_MEMORY_TARGET_WEIGHT,
    )
    total_loss = (
        compressed_ce_loss * compressed_ce_weight
        + state_loss * state_loss_weight
        + delta_memory_loss * delta_memory_weight
        + continuation_kl_loss * continuation_kl_weight
    )

    _update_latent_metrics(trainer, compressed_metrics)
    _record_metric(trainer, "opsd_span/warmup_path_active", 1.0)
    _record_metric(trainer, "opsd_span/warmup_compressed_ce_loss", float(compressed_ce_loss.detach().item()))
    _record_metric(
        trainer,
        "opsd_span/warmup_compressed_ce_contrib",
        float((compressed_ce_loss.detach() * compressed_ce_weight).item()),
    )
    _record_metric(trainer, "opsd_span/warmup_state_loss", float(state_loss.detach().item()))
    _record_metric(
        trainer,
        "opsd_span/warmup_state_contrib",
        float((state_loss.detach() * state_loss_weight).item()),
    )
    _record_metric(trainer, "opsd_span/warmup_state_cosine_loss", float(state_cosine_loss.detach().item()))
    _record_metric(trainer, "opsd_span/warmup_state_huber_loss", float(state_huber_loss.detach().item()))
    _record_metric(trainer, "opsd_span/warmup_state_nce_loss", float(state_nce_loss.detach().item()))
    _record_metric(trainer, "opsd_span/warmup_delta_memory_enabled", 1.0 if _delta_memory_enabled() else 0.0)
    _record_metric(trainer, "opsd_span/warmup_delta_memory_loss", float(delta_memory_loss.detach().item()))
    _record_metric(
        trainer,
        "opsd_span/warmup_delta_memory_contrib",
        float((delta_memory_loss.detach() * delta_memory_weight).item()),
    )
    _record_metric(trainer, "opsd_span/warmup_continuation_kl_loss", float(continuation_kl_loss.detach().item()))
    _record_metric(
        trainer,
        "opsd_span/warmup_continuation_kl_contrib",
        float((continuation_kl_loss.detach() * continuation_kl_weight).item()),
    )
    _record_metric(trainer, "opsd_span/warmup_continuation_kl_tokens", float(continuation_kl_tokens))
    _record_metric(trainer, "opsd_span/warmup_latent_matches", float(len(compressed_indices)))
    _record_metric(trainer, "opsd_span/warmup_latent_injections", float(len(injection_indices)))
    _record_metric(
        trainer,
        "opsd_span/warmup_full_response_tokens",
        float((full_labels[:, full_keep_positions + 1] != -100).sum().item()),
    )
    _record_metric(trainer, "token_acc", float(compressed_metrics["token_acc"]))
    del full_outputs
    del compressed_outputs
    del input_embeddings
    del full_batch
    del full_model_inputs
    del compressed_model_inputs
    del compressed_logits
    return total_loss, None


def _compute_sparse_weighted_token_ce(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    keep_positions: torch.Tensor,
    latent_token_id: int | None,
    latent_token_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if keep_positions.numel() == 0:
        metrics = {
            "latent_tokens": 0.0,
            "response_tokens": 0.0,
            "latent_token_weight": float(latent_token_weight),
            "token_acc": 0.0,
        }
        return _zero_loss_like(logits.float()), metrics

    kept_labels = labels[:, keep_positions + 1]
    valid_mask = kept_labels != -100
    if not bool(valid_mask.any().item()):
        metrics = {
            "latent_tokens": 0.0,
            "response_tokens": 0.0,
            "latent_token_weight": float(latent_token_weight),
            "token_acc": 0.0,
        }
        return _zero_loss_like(logits.float()), metrics

    valid_labels = kept_labels[valid_mask]
    valid_logits = logits[valid_mask].float()

    latent_mask = torch.zeros_like(valid_labels, dtype=torch.bool)
    weights = torch.ones(valid_labels.shape[0], device=valid_labels.device, dtype=torch.float32)
    if latent_token_id is not None:
        latent_mask = valid_labels == latent_token_id
        weights = weights + latent_mask.to(dtype=weights.dtype) * (latent_token_weight - 1.0)

    chunk_size = max(int(_get_env_float("OPSD_WARMUP_CE_CHUNK_SIZE", DEFAULT_WARMUP_CE_CHUNK_SIZE)), 1)
    weighted_loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    correct_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    denom = weights.sum().clamp_min(1.0)
    token_count = max(int(valid_labels.numel()), 1)

    for start_idx in range(0, valid_logits.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, valid_logits.shape[0])
        chunk_logits = valid_logits[start_idx:end_idx]
        chunk_labels = valid_labels[start_idx:end_idx]
        chunk_weights = weights[start_idx:end_idx]
        chunk_loss = F.cross_entropy(chunk_logits, chunk_labels, reduction="none")
        weighted_loss_sum = weighted_loss_sum + (chunk_loss * chunk_weights).sum()
        chunk_preds = chunk_logits.argmax(dim=-1)
        correct_sum = correct_sum + (chunk_preds == chunk_labels).to(dtype=torch.float32).sum()

    loss = weighted_loss_sum / denom
    metrics = {
        "latent_tokens": float(latent_mask.sum().item()),
        "response_tokens": float(valid_labels.numel()),
        "latent_token_weight": float(latent_token_weight),
        "token_acc": float((correct_sum / float(token_count)).item()),
    }
    return loss, metrics


def _compute_weighted_token_ce(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    latent_token_id: int | None,
    latent_token_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    shifted_labels = labels[:, 1:].contiguous()
    shifted_logits = logits[:, :-1, :]
    flat_labels = shifted_labels.reshape(-1)
    valid_indices = torch.nonzero(flat_labels != -100, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        metrics = {
            "latent_tokens": 0.0,
            "response_tokens": 0.0,
            "latent_token_weight": float(latent_token_weight),
        }
        return _zero_loss_like(logits.float()), metrics

    flat_shifted_logits = shifted_logits.reshape(-1, shifted_logits.shape[-1])
    valid_labels = flat_labels.index_select(0, valid_indices)
    latent_mask = torch.zeros_like(valid_labels, dtype=torch.bool)
    weights = torch.ones(valid_labels.shape[0], device=valid_labels.device, dtype=torch.float32)
    if latent_token_id is not None:
        latent_mask = valid_labels == latent_token_id
        weights = weights + latent_mask.to(dtype=weights.dtype) * (latent_token_weight - 1.0)

    chunk_size = max(int(_get_env_float("OPSD_WARMUP_CE_CHUNK_SIZE", DEFAULT_WARMUP_CE_CHUNK_SIZE)), 1)
    weighted_loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    denom = weights.sum().clamp_min(1.0)
    for start_idx in range(0, valid_indices.numel(), chunk_size):
        end_idx = min(start_idx + chunk_size, valid_indices.numel())
        chunk_indices = valid_indices[start_idx:end_idx]
        chunk_logits = flat_shifted_logits.index_select(0, chunk_indices).float()
        chunk_labels = valid_labels[start_idx:end_idx]
        chunk_weights = weights[start_idx:end_idx]
        chunk_loss = F.cross_entropy(chunk_logits, chunk_labels, reduction="none")
        weighted_loss_sum = weighted_loss_sum + (chunk_loss * chunk_weights).sum()

    loss = weighted_loss_sum / denom
    metrics = {
        "latent_tokens": float(latent_mask.sum().item()),
        "response_tokens": float(valid_indices.numel()),
        "latent_token_weight": float(latent_token_weight),
    }
    return loss, metrics


def _update_latent_metrics(trainer: Any, metrics: dict[str, float]) -> None:
    if not metrics:
        return
    mode = "train" if trainer.model.training else "eval"
    custom_metrics = getattr(trainer, "custom_metrics", {}).get(mode)
    if custom_metrics is None:
        return
    for key, value in metrics.items():
        custom_metrics[f"opsd_span/{key}"].update(value)


def _rlsd_enabled() -> bool:
    return _get_env_bool("OPSD_RLSD_ENABLED", DEFAULT_RLSD_ENABLED)


def _rlsd_weight() -> float:
    return max(_get_env_float("OPSD_RLSD_WEIGHT", DEFAULT_RLSD_WEIGHT), 0.0)


def _rlsd_temperature() -> float:
    return max(_get_env_float("OPSD_RLSD_TEMPERATURE", DEFAULT_RLSD_TEMPERATURE), 1e-6)


def _rlsd_max_weight() -> float:
    return max(_get_env_float("OPSD_RLSD_MAX_WEIGHT", DEFAULT_RLSD_MAX_WEIGHT), 1.0)


def _rlsd_min_weight() -> float:
    return max(min(_get_env_float("OPSD_RLSD_MIN_WEIGHT", DEFAULT_RLSD_MIN_WEIGHT), _rlsd_max_weight()), 0.0)


def _single_sample_grpo_enabled() -> bool:
    return _get_env_bool("OPSD_SINGLE_SAMPLE_GRPO_ENABLED", DEFAULT_SINGLE_SAMPLE_GRPO_ENABLED)


class OpsdSpanLatentFormatReward(ORM):
    def __call__(self, completions, teacher_reference_target=None, **kwargs) -> list[float]:
        rewards: list[float] = []
        teacher_targets = list(teacher_reference_target or [])
        trainer_state = kwargs.get("trainer_state")
        for idx, completion in enumerate(completions):
            text = _decode_completion_text(completion)
            reference_text = str(teacher_targets[idx] if idx < len(teacher_targets) else "")
            assessment = _assess_latent_format(text=text, reference_text=reference_text)
            reasoning, answer = _split_reasoning_answer(text)
            target = reference_text
            _log_reward_debug(
                reward_name="latent_format",
                trainer_state=trainer_state,
                completion_text=text,
                reasoning=reasoning,
                answer=answer,
                target=target,
                reward=float(assessment.reward),
            )
            rewards.append(float(assessment.reward))
        return rewards


class OpsdSpanAnswerReward(ORM):
    def __call__(self, completions, answer_text=None, teacher_reference_target=None, **kwargs) -> list[float]:
        answers = list(answer_text or [])
        compressed_targets = list(kwargs.get("opsd_compressed_target") or [])
        full_targets = list(kwargs.get("opsd_full_target") or [])
        teacher_targets = list(teacher_reference_target or [])
        rewards: list[float] = []
        trainer_state = kwargs.get("trainer_state")
        for idx, completion in enumerate(completions):
            text = _decode_completion_text(completion)
            reasoning, predicted_answer = _split_reasoning_answer(text)
            predicted = _normalize_answer_text(predicted_answer)
            target = _resolve_target_answer(
                idx,
                answer_text=answers,
                compressed_targets=compressed_targets,
                full_targets=full_targets,
            )
            reference_text = str(teacher_targets[idx] if idx < len(teacher_targets) else "")
            assessment = _assess_latent_format(text=text, reference_text=reference_text)
            reward = assessment.answer_correct_reward if _answers_match_strict(predicted, target) else assessment.answer_wrong_reward
            _log_reward_debug(
                reward_name="answer",
                trainer_state=trainer_state,
                completion_text=text,
                reasoning=reasoning,
                answer=predicted,
                target=target,
                reward=float(reward),
            )
            rewards.append(reward)
        return rewards


def _compute_rlsd_token_weights(
    *,
    per_token_logps: torch.Tensor,
    teacher_per_token_logps: torch.Tensor | None,
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if teacher_per_token_logps is None:
        ones = torch.ones_like(per_token_logps, dtype=torch.float32)
        return ones, {
            "rlsd_enabled": 0.0,
            "rlsd_weight_mean": 1.0,
            "rlsd_weight_max": 1.0,
            "rlsd_weight_min": 1.0,
            "rlsd_gap_mean": 0.0,
        }

    gap = (teacher_per_token_logps - per_token_logps).detach().float()
    raw_weights = torch.exp(gap / _rlsd_temperature())
    raw_weights = torch.clamp(raw_weights, min=_rlsd_min_weight(), max=_rlsd_max_weight())
    weights = 1.0 + _rlsd_weight() * (raw_weights - 1.0)
    weights = weights * completion_mask.to(dtype=weights.dtype)
    masked = weights[completion_mask]
    masked_gap = gap[completion_mask]
    if masked.numel() == 0:
        mean_weight = 1.0
        max_weight = 1.0
        min_weight = 1.0
        mean_gap = 0.0
    else:
        mean_weight = float(masked.mean().item())
        max_weight = float(masked.max().item())
        min_weight = float(masked.min().item())
        mean_gap = float(masked_gap.mean().item())
    return weights, {
        "rlsd_enabled": 1.0,
        "rlsd_weight_mean": mean_weight,
        "rlsd_weight_max": max_weight,
        "rlsd_weight_min": min_weight,
        "rlsd_gap_mean": mean_gap,
    }


def _patch_seq2seq_latent_loss() -> None:
    if getattr(Seq2SeqTrainer, "_opsd_span_latent_loss_patched", False):
        return

    original_compute_loss = Seq2SeqTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (
            _strip_text(os.environ.get("OPSD_SPAN_STAGE", "main")).lower() == "warmup"
            and inputs.get("opsd_full_target") is not None
            and (
                inputs.get("opsd_student_user") is not None
                or inputs.get("opsd_full_messages") is not None
            )
        ):
            warmup_result = _compute_warmup_loss(self, model, inputs)
            if warmup_result[0] is not None:
                loss, _ = warmup_result
                if return_outputs:
                    return loss, SimpleNamespace(logits=None)
                return loss

        labels = inputs.get("labels")
        loss, outputs = original_compute_loss(
            self,
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        logits = getattr(outputs, "logits", None) if outputs is not None else None
        latent_token_id = _get_latent_token_id(getattr(self, "tokenizer", None))
        if labels is not None and logits is not None and latent_token_id is not None:
            loss, metrics = _compute_weighted_token_ce(
                logits=logits,
                labels=labels,
                latent_token_id=latent_token_id,
                latent_token_weight=_get_latent_token_weight(),
            )
            _update_latent_metrics(self, metrics)

        if return_outputs:
            return loss, outputs
        return loss

    Seq2SeqTrainer.compute_loss = compute_loss
    Seq2SeqTrainer._opsd_span_latent_loss_patched = True


def _patch_gkd_latent_loss() -> None:
    if getattr(GKDTrainer, "_opsd_span_latent_loss_patched", False):
        return

    original_compute_jsd_loss = GKDTrainer._compute_jsd_loss

    def _compute_jsd_loss(self, student_logits, teacher_output: TeacherOutput, labels):
        base_loss = original_compute_jsd_loss(self, student_logits, teacher_output, labels)
        latent_token_id = _get_latent_token_id(getattr(self, "tokenizer", None))
        if latent_token_id is None:
            return base_loss

        latent_loss, metrics = _compute_weighted_token_ce(
            logits=student_logits,
            labels=labels,
            latent_token_id=latent_token_id,
            latent_token_weight=_get_latent_token_weight(),
        )
        metrics["jsd_loss"] = float(base_loss.detach().item())
        metrics["latent_ce_loss"] = float(latent_loss.detach().item())
        _update_latent_metrics(self, metrics)
        return base_loss + (latent_loss - latent_loss.detach())

    GKDTrainer._compute_jsd_loss = _compute_jsd_loss
    GKDTrainer._opsd_span_latent_loss_patched = True


def _build_interleave_segments(
    *,
    latent_positions: list[int],
    seq_len: int,
) -> list[tuple[int, int, str]]:
    """
    Construct interleave segment boundaries from latent token positions.

    This function analyzes the sequence structure and segments it into alternating
    normal and latent segments. Latent segments may contain multiple consecutive
    <latent> tokens, representing compressed reasoning spans.

    Args:
        latent_positions: Sorted list of latent token positions within the sequence.
        seq_len: Total sequence length.

    Returns:
        List of tuples (start_pos, end_pos, segment_type) where:
        - start_pos: Inclusive start position of the segment
        - end_pos: Exclusive end position of the segment
        - segment_type: "normal" or "latent"
    """
    if not latent_positions:
        return [(0, seq_len, "normal")]

    segments = []
    current_pos = 0
    latent_idx = 0

    while current_pos < seq_len:
        if latent_idx < len(latent_positions) and current_pos == latent_positions[latent_idx]:
            # Start of a latent segment: collect all consecutive latent tokens
            segment_start = current_pos
            while latent_idx < len(latent_positions) and latent_positions[latent_idx] == current_pos:
                current_pos += 1
                latent_idx += 1
            segments.append((segment_start, current_pos, "latent"))
        else:
            # Normal segment: from current position to next latent (or end of sequence)
            segment_start = current_pos
            segment_end = latent_positions[latent_idx] if latent_idx < len(latent_positions) else seq_len
            segments.append((segment_start, segment_end, "normal"))
            current_pos = segment_end

    return segments


def _slice_logps_to_keep(
    *,
    logps: torch.Tensor,
    entropies: torch.Tensor | None,
    logits_to_keep: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if logits_to_keep is None or logits_to_keep >= logps.shape[1]:
        return logps, entropies
    sliced_logps = logps[:, -logits_to_keep:]
    sliced_entropies = entropies[:, -logits_to_keep:] if entropies is not None else None
    return sliced_logps, sliced_entropies


def _clone_past_key_values(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "to_legacy_cache") and hasattr(past_key_values, "from_legacy_cache"):
        return type(past_key_values).from_legacy_cache(past_key_values.to_legacy_cache())
    if isinstance(past_key_values, tuple):
        return tuple(
            tuple(tensor.clone() if isinstance(tensor, torch.Tensor) else tensor for tensor in layer)
            for layer in past_key_values
        )
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def _select_past_key_values_batch(past_key_values: Any, indices: torch.Tensor) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "batch_select_indices"):
        selected = _clone_past_key_values(past_key_values)
        selected.batch_select_indices(indices)
        return selected
    if isinstance(past_key_values, tuple):
        selected_layers = []
        for layer in past_key_values:
            selected_layers.append(
                tuple(
                    tensor.index_select(0, indices) if isinstance(tensor, torch.Tensor) else tensor
                    for tensor in layer
                )
            )
        return tuple(selected_layers)
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def _split_past_key_values_batch(past_key_values: Any, batch_size: int) -> list[Any]:
    if past_key_values is None:
        return [None] * batch_size
    split_caches: list[Any] = []
    device = None
    if isinstance(past_key_values, tuple) and past_key_values and past_key_values[0] and isinstance(past_key_values[0][0], torch.Tensor):
        device = past_key_values[0][0].device
    elif hasattr(past_key_values, "layers") and past_key_values.layers and getattr(past_key_values.layers[0], "keys", None) is not None:
        device = past_key_values.layers[0].keys.device
    for idx in range(batch_size):
        indices = torch.tensor([idx], device=device or torch.device("cpu"), dtype=torch.long)
        split_caches.append(_select_past_key_values_batch(past_key_values, indices))
    return split_caches


def _crop_past_key_values(past_key_values: Any, max_length: int) -> Any:
    if past_key_values is None:
        return None
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")
    cropped = _clone_past_key_values(past_key_values)
    if hasattr(cropped, "crop"):
        cropped.crop(max_length)
        return cropped
    if isinstance(cropped, tuple):
        return tuple(
            tuple(
                tensor[..., :max_length, :].clone() if isinstance(tensor, torch.Tensor) else tensor
                for tensor in layer
            )
            for layer in cropped
        )
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def _slice_model_inputs_for_rows(inputs: dict[str, Any], row_indices: torch.Tensor) -> dict[str, Any]:
    row_inputs: dict[str, Any] = {}
    keep_full_tensor_keys = {
        "cache_position",
        *MULTIMODAL_INPUT_KEYS,
    }
    for key, value in inputs.items():
        if key == "_origin_data":
            continue
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                row_inputs[key] = value
            elif key in keep_full_tensor_keys:
                row_inputs[key] = value
            elif key == "position_ids" and value.ndim >= 3:
                row_inputs[key] = value.index_select(1, row_indices)
            else:
                row_inputs[key] = value.index_select(0, row_indices)
        else:
            row_inputs[key] = value
    return row_inputs


def _build_replay_states(
    *,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    latent_token_id: int,
    logits_to_keep: int,
) -> tuple[list[ReplaySampleState], int]:
    states: list[ReplaySampleState] = []
    max_prefix_len = 0
    seq_len = input_ids.shape[1]
    for row_idx in range(input_ids.shape[0]):
        completion_len = int(completion_mask[row_idx].sum().item())
        if completion_len <= 0:
            continue
        window_len = min(seq_len, logits_to_keep + 1)
        window_start = seq_len - window_len
        window_ids = input_ids[row_idx, window_start:]
        latent_positions = torch.where(window_ids == latent_token_id)[0].tolist()
        segments = _build_interleave_segments(latent_positions=latent_positions, seq_len=window_len)
        prefix_len = max(window_len - logits_to_keep, 0)
        max_prefix_len = max(max_prefix_len, prefix_len)
        states.append(
            ReplaySampleState(
                row_idx=row_idx,
                sample_idx=len(states),
                segments=segments,
                window_ids=window_ids,
                window_start=window_start,
                prefix_len=prefix_len,
            )
        )
    return states, max_prefix_len


def _run_batched_replay_prefill(
    *,
    trainer: GRPOTrainer,
    model: torch.nn.Module,
    inputs: dict[str, Any],
    states: list[ReplaySampleState],
    max_prefix_len: int,
) -> Any:
    if max_prefix_len <= 0 or not states:
        return None

    row_indices = torch.tensor([state.row_idx for state in states], device=inputs["input_ids"].device, dtype=torch.long)
    batch_inputs = _slice_model_inputs_for_rows(inputs, row_indices)
    model_inputs = trainer._prepare_model_inputs(batch_inputs)
    if "input_ids" not in model_inputs:
        raise ValueError("segment replay requires input_ids in prepared model inputs")

    prefix_lengths = [state.window_start + state.prefix_len for state in states]
    max_prefill_len = max(prefix_lengths, default=0)
    prefix_input_ids = model_inputs["input_ids"][:, :max_prefill_len]
    if prefix_input_ids.shape[1] == 0:
        return None

    prefill_inputs = _filter_model_inputs(model_inputs)
    prefill_inputs["input_ids"] = prefix_input_ids
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        prefill_inputs["attention_mask"] = attention_mask[:, :max_prefill_len]
    position_ids = model_inputs.get("position_ids")
    if position_ids is not None:
        prefill_inputs["position_ids"] = position_ids[:, :, :max_prefill_len]
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
        value = model_inputs.get(key)
        if value is not None:
            prefill_inputs[key] = value
    prefill_inputs["use_cache"] = True
    outputs = model(**prefill_inputs)
    logits = outputs.logits / trainer.temperature
    for state in states:
        prefix_len = state.prefix_len
        prefill_len = state.window_start + prefix_len
        if prefill_len <= 0:
            continue
        if prefill_len > max_prefill_len:
            raise ValueError(f"prefill_len {prefill_len} exceeds prefill window {max_prefill_len}")
        state.boundary_logit = logits[state.sample_idx, prefill_len - 1 : prefill_len, :]
        state.window_pos = prefix_len
        while state.segment_idx < len(state.segments) and state.segments[state.segment_idx][1] <= prefix_len:
            state.segment_idx += 1
    split_caches = _split_past_key_values_batch(outputs.past_key_values, len(states))
    for state, cache_state in zip(states, split_caches, strict=False):
        state.cache_state = _crop_past_key_values(cache_state, state.window_start + state.prefix_len)
    return outputs.past_key_values


def _append_logit_and_entropy(
    *,
    sample: ReplaySampleState,
    logits: torch.Tensor,
    entropies: torch.Tensor | None,
    out_logps: torch.Tensor,
    out_entropies: torch.Tensor | None,
) -> None:
    if sample.window_pos <= 0:
        return
    target_token_id = sample.window_ids[sample.window_pos]
    logp = selective_log_softmax(logits.float(), target_token_id.view(1))
    out_logps[sample.row_idx, sample.window_pos - 1] = logp.squeeze(0)
    if out_entropies is not None and entropies is not None:
        out_entropies[sample.row_idx, sample.window_pos - 1] = entropies.squeeze(0)


def _finalize_latent_step(
    *,
    sample: ReplaySampleState,
    hidden_state: torch.Tensor,
) -> None:
    carry_state = hidden_state.float()
    if _delta_memory_enabled() and _opsd_span_replay_delta_mode() == "follow_global":
        updated_memory = _apply_delta_memory_update(sample.memory_state, carry_state)
        sample.memory_state = updated_memory
        sample.pending_carry = updated_memory.to(dtype=hidden_state.dtype, device=hidden_state.device)
    else:
        sample.pending_carry = hidden_state


def _group_states_by_window_pos(states: list[ReplaySampleState]) -> list[list[ReplaySampleState]]:
    grouped: dict[int, list[ReplaySampleState]] = {}
    for state in states:
        grouped.setdefault(int(state.window_pos), []).append(state)
    return [grouped[pos] for pos in sorted(grouped)]


def _group_normal_states(states: list[ReplaySampleState]) -> list[list[ReplaySampleState]]:
    grouped: dict[tuple[int, int], list[ReplaySampleState]] = {}
    for state in states:
        _, seg_end, seg_type = state.segments[state.segment_idx]
        if seg_type != "normal":
            raise ValueError("normal-state grouping received a non-normal segment")
        span_len = int(seg_end - state.window_pos)
        grouped.setdefault((int(state.window_pos), span_len), []).append(state)
    return [grouped[key] for key in sorted(grouped)]


def _run_standard_sequence_forward(
    *,
    trainer: GRPOTrainer,
    model: torch.nn.Module,
    model_inputs: dict[str, Any],
    input_ids: torch.Tensor,
    logits_to_keep: int | None,
    compute_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    forward_inputs = dict(model_inputs)
    if "logits_to_keep" in getattr(trainer, "model_kwarg_keys", {}):
        keep = int(logits_to_keep) if logits_to_keep is not None else input_ids.shape[1] - 1
        forward_inputs["logits_to_keep"] = keep + 1
    outputs = model(**forward_inputs)
    logits = outputs.logits / trainer.temperature
    if logits_to_keep is not None and logits_to_keep < input_ids.shape[1]:
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
    else:
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
    logps = selective_log_softmax(logits.float(), input_ids)
    entropies = entropy_from_logits(logits) if compute_entropy else None
    return logps, entropies


def _compute_segment_replay_logps_and_entropies(
    trainer: GRPOTrainer,
    model: torch.nn.Module,
    inputs: dict[str, Any],
    memory_states: list[torch.Tensor | None],
    compute_entropy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    completion_mask = inputs["completion_mask"]
    batch_size = input_ids.shape[0]
    logits_to_keep = int(inputs.get("logits_to_keep", 0) or input_ids.shape[1] - 1)

    # Get latent token ID
    latent_token_id = _get_latent_token_id(getattr(trainer, "tokenizer", None))
    if latent_token_id is None:
        # Fallback to standard computation
        return _compute_standard_logps_and_entropies(
            trainer=trainer,
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            compute_entropy=compute_entropy,
        )

    replay_states_all, max_prefix_len = _build_replay_states(
        input_ids=input_ids,
        completion_mask=completion_mask,
        latent_token_id=latent_token_id,
        logits_to_keep=logits_to_keep,
    )
    legit_latent_max = _opsd_span_replay_legit_latent_count_max()
    replay_states = [
        state
        for state in replay_states_all
        if int((state.window_ids == latent_token_id).sum().item()) > 0
        and int((state.window_ids == latent_token_id).sum().item()) <= legit_latent_max
    ]
    replay_row_indices = torch.tensor([state.row_idx for state in replay_states], device=input_ids.device, dtype=torch.long)
    replay_row_index_set = {int(state.row_idx) for state in replay_states}
    standard_row_values = [row_idx for row_idx in range(batch_size) if row_idx not in replay_row_index_set]
    standard_row_indices = torch.tensor(standard_row_values, device=input_ids.device, dtype=torch.long) if standard_row_values else None

    output_logps = torch.zeros((batch_size, logits_to_keep), device=input_ids.device, dtype=torch.float32)
    output_entropies = torch.zeros((batch_size, logits_to_keep), device=input_ids.device, dtype=torch.float32) if compute_entropy else None

    if standard_row_indices is not None:
        standard_inputs = _slice_model_inputs_for_rows(inputs, standard_row_indices)
        standard_model_inputs = trainer._prepare_model_inputs(standard_inputs)
        standard_input_ids = standard_model_inputs["input_ids"]
        standard_logps, standard_entropies = _run_standard_sequence_forward(
            trainer=trainer,
            model=model,
            model_inputs=standard_model_inputs,
            input_ids=standard_input_ids,
            logits_to_keep=logits_to_keep,
            compute_entropy=compute_entropy,
        )
        output_logps.index_copy_(0, standard_row_indices, standard_logps)
        if output_entropies is not None and standard_entropies is not None:
            output_entropies.index_copy_(0, standard_row_indices, standard_entropies)

    if not replay_states:
        return output_logps, output_entropies, {
            "legit_ratio": 0.0,
            "replay_segment_mixed": 0.0,
            "replay_segment_samples": 0.0,
            "replay_non_legit_ratio": 0.0,
            "replay_latent_count_mean": 0.0,
            "segment_replay_latent_steps_mean": 0.0,
            "segment_replay_latent_steps_max": 0.0,
            "segment_replay_forward_passes_mean": 1.0,
            "segment_replay_forward_passes_max": 1.0,
        }

    replay_logps = torch.zeros((len(replay_states), logits_to_keep), device=input_ids.device, dtype=torch.float32)
    replay_entropies = torch.zeros((len(replay_states), logits_to_keep), device=input_ids.device, dtype=torch.float32) if compute_entropy else None

    row_indices = replay_row_indices
    replay_inputs = _slice_model_inputs_for_rows(inputs, row_indices)
    replay_model_inputs = trainer._prepare_model_inputs(replay_inputs)
    replay_input_ids = replay_model_inputs["input_ids"]
    replay_seq_len = replay_input_ids.shape[1]
    replay_entropy_buf = replay_entropies

    for sample_idx, state in enumerate(replay_states):
        state.sample_idx = sample_idx
        if memory_states and memory_states[state.row_idx] is not None:
            state.memory_state = memory_states[state.row_idx]

    _run_batched_replay_prefill(
        trainer=trainer,
        model=model,
        inputs=replay_inputs,
        states=replay_states,
        max_prefix_len=max_prefix_len,
    )

    active = [state for state in replay_states if state.window_pos < int(state.window_ids.shape[0])]
    latent_count_sum = 0
    max_latent_steps = 0
    forward_passes = 0

    while active:
        normal_group: list[ReplaySampleState] = []
        latent_group: list[ReplaySampleState] = []
        for state in active:
            if state.segment_idx >= len(state.segments):
                continue
            seg_start, seg_end, seg_type = state.segments[state.segment_idx]
            if state.window_pos < seg_start or state.window_pos >= seg_end:
                raise ValueError(
                    f"invalid replay frontier for sample {state.row_idx}: pos={state.window_pos}, seg={state.segments[state.segment_idx]}"
                )
            if seg_type == "normal":
                normal_group.append(state)
            else:
                latent_group.append(state)

        for normal_bucket in _group_normal_states(normal_group):
            forward_passes += 1
            sample_indices = torch.tensor([state.sample_idx for state in normal_bucket], device=input_ids.device, dtype=torch.long)
            local_cache = None
            if normal_bucket[0].cache_state is not None:
                cache_states = [state.cache_state for state in normal_bucket]
                local_cache = _clone_past_key_values(cache_states[0])
                if len(cache_states) > 1:
                    legacy_layers = []
                    for layer_idx in range(len(local_cache)):
                        keys = torch.cat([cache_state[layer_idx][0] for cache_state in cache_states], dim=0)
                        values = torch.cat([cache_state[layer_idx][1] for cache_state in cache_states], dim=0)
                        legacy_layers.append((keys, values))
                    if hasattr(local_cache, "from_legacy_cache"):
                        local_cache = type(local_cache).from_legacy_cache(tuple(legacy_layers))
                    else:
                        local_cache = tuple(legacy_layers)
            chunk_lengths = []
            chunk_embeds = []
            chunk_input_ids = []
            boundary_logits = []
            for state in normal_bucket:
                _, seg_end, _ = state.segments[state.segment_idx]
                chunk_end = seg_end
                absolute_start = state.window_start + state.window_pos
                absolute_end = state.window_start + chunk_end
                chunk_ids = replay_input_ids[state.sample_idx, absolute_start:absolute_end]
                chunk_input_ids.append(chunk_ids)
                chunk_embeds.append(_get_model_input_embeddings(model)(chunk_ids.unsqueeze(0)).squeeze(0))
                chunk_lengths.append(int(chunk_ids.shape[0]))
                if state.boundary_logit is None:
                    raise ValueError(f"missing boundary logit for normal chunk sample {state.row_idx}")
                boundary_logits.append(state.boundary_logit.view(1, 1, -1))
            max_chunk = max(chunk_lengths)
            min_chunk = min(chunk_lengths)
            if min_chunk != max_chunk:
                raise ValueError("normal replay bucket mixed different span lengths")
            hidden_dim = chunk_embeds[0].shape[-1]
            inputs_embeds = torch.stack(chunk_embeds, dim=0)
            chunk_input_ids_tensor = torch.stack(chunk_input_ids, dim=0)
            local_attention_mask = replay_model_inputs["attention_mask"].index_select(0, sample_indices).clone()
            current_pos = normal_bucket[0].window_start + normal_bucket[0].window_pos
            for idx, length in enumerate(chunk_lengths):
                local_attention_mask[idx, current_pos + length :] = 0
            chunk_inputs = _filter_model_inputs(replay_model_inputs)
            for multimodal_key in MULTIMODAL_INPUT_KEYS:
                chunk_inputs.pop(multimodal_key, None)
            chunk_inputs["input_ids"] = chunk_input_ids_tensor
            chunk_inputs["inputs_embeds"] = inputs_embeds
            chunk_inputs["attention_mask"] = local_attention_mask[:, :replay_seq_len]
            position_ids = replay_model_inputs.get("position_ids")
            if position_ids is not None:
                pos_slices = []
                for state, length in zip(normal_bucket, chunk_lengths, strict=False):
                    absolute_pos = state.window_start + state.window_pos
                    pos_slice = position_ids[:, state.sample_idx : state.sample_idx + 1, absolute_pos:absolute_pos + length]
                    pos_slices.append(pos_slice)
                chunk_inputs["position_ids"] = torch.cat(pos_slices, dim=1)
            chunk_inputs["cache_position"] = torch.arange(
                current_pos,
                current_pos + max_chunk,
                device=input_ids.device,
                dtype=torch.long,
            )
            chunk_inputs["past_key_values"] = local_cache
            chunk_inputs["use_cache"] = True
            outputs = model(**chunk_inputs)
            logits = outputs.logits / trainer.temperature
            entropies = entropy_from_logits(logits) if compute_entropy else None
            updated_cache_states = _split_past_key_values_batch(outputs.past_key_values, len(normal_bucket))
            for idx, state in enumerate(normal_bucket):
                length = chunk_lengths[idx]
                full_logits = torch.cat([boundary_logits[idx], logits[idx : idx + 1, : max(length - 1, 0), :]], dim=1)
                full_entropies = None
                if entropies is not None:
                    full_entropies = torch.cat(
                        [entropy_from_logits(boundary_logits[idx]), entropies[idx : idx + 1, : max(length - 1, 0)]],
                        dim=1,
                    )
                absolute_pos = state.window_start + state.window_pos
                targets = replay_input_ids[state.sample_idx, absolute_pos:absolute_pos + length]
                replay_logps[state.sample_idx, state.window_pos - 1 : state.window_pos - 1 + length] = selective_log_softmax(
                    full_logits.float(), targets.unsqueeze(0)
                ).squeeze(0)
                if replay_entropy_buf is not None and full_entropies is not None:
                    replay_entropy_buf[state.sample_idx, state.window_pos - 1 : state.window_pos - 1 + length] = full_entropies.squeeze(0)
                state.boundary_logit = logits[idx : idx + 1, length - 1 : length, :]
                state.cache_state = updated_cache_states[idx]
                state.window_pos += length
                state.segment_idx += 1

        if latent_group:
            latent_lengths = []
            for state in latent_group:
                seg_start, seg_end, _ = state.segments[state.segment_idx]
                if state.window_pos != seg_start:
                    raise ValueError(f"latent frontier mismatch for sample {state.row_idx}")
                latent_lengths.append(seg_end - seg_start)
                latent_count_sum += seg_end - seg_start
                max_latent_steps = max(max_latent_steps, seg_end - seg_start)
            max_steps = max(latent_lengths)
            for step_idx in range(max_steps):
                active_step = [state for state, length in zip(latent_group, latent_lengths, strict=False) if step_idx < length]
                for latent_bucket in _group_states_by_window_pos(active_step):
                    if not latent_bucket:
                        continue
                    forward_passes += 1
                    sample_indices = torch.tensor([state.sample_idx for state in latent_bucket], device=input_ids.device, dtype=torch.long)
                    local_cache = None
                    if latent_bucket[0].cache_state is not None:
                        cache_states = [state.cache_state for state in latent_bucket]
                        local_cache = _clone_past_key_values(cache_states[0])
                        if len(cache_states) > 1:
                            legacy_layers = []
                            for layer_idx in range(len(local_cache)):
                                keys = torch.cat([cache_state[layer_idx][0] for cache_state in cache_states], dim=0)
                                values = torch.cat([cache_state[layer_idx][1] for cache_state in cache_states], dim=0)
                                legacy_layers.append((keys, values))
                            if hasattr(local_cache, "from_legacy_cache"):
                                local_cache = type(local_cache).from_legacy_cache(tuple(legacy_layers))
                            else:
                                local_cache = tuple(legacy_layers)
                    input_embeds = []
                    step_input_ids = []
                    boundary_logits = []
                    for state in latent_bucket:
                        absolute_pos = state.window_start + state.window_pos
                        token_id = replay_input_ids[state.sample_idx, absolute_pos]
                        step_input_ids.append(token_id)
                        embed = _get_model_input_embeddings(model)(token_id.view(1)).squeeze(0)
                        if state.pending_carry is not None:
                            embed = embed + state.pending_carry.to(dtype=embed.dtype, device=embed.device)
                            state.pending_carry = None
                        input_embeds.append(embed)
                        if state.boundary_logit is None:
                            raise ValueError(f"missing boundary logit for latent step sample {state.row_idx}")
                        boundary_logits.append(state.boundary_logit.view(1, 1, -1))
                    inputs_embeds = torch.stack(input_embeds, dim=0).unsqueeze(1)
                    step_input_ids_tensor = torch.stack(step_input_ids, dim=0).unsqueeze(1)
                    step_inputs = _filter_model_inputs(replay_model_inputs)
                    for multimodal_key in MULTIMODAL_INPUT_KEYS:
                        step_inputs.pop(multimodal_key, None)
                    step_inputs["input_ids"] = step_input_ids_tensor
                    step_inputs["inputs_embeds"] = inputs_embeds
                    local_attention_mask = replay_model_inputs["attention_mask"].index_select(0, sample_indices).clone()
                    current_pos = latent_bucket[0].window_start + latent_bucket[0].window_pos
                    for idx in range(len(latent_bucket)):
                        local_attention_mask[idx, current_pos + 1 :] = 0
                    step_inputs["attention_mask"] = local_attention_mask[:, :replay_seq_len]
                    position_ids = replay_model_inputs.get("position_ids")
                    if position_ids is not None:
                        step_pos = []
                        for state in latent_bucket:
                            absolute_pos = state.window_start + state.window_pos
                            step_pos.append(position_ids[:, state.sample_idx : state.sample_idx + 1, absolute_pos:absolute_pos + 1])
                        step_inputs["position_ids"] = torch.cat(step_pos, dim=1)
                    step_inputs["cache_position"] = torch.arange(
                        current_pos,
                        current_pos + 1,
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                    step_inputs["past_key_values"] = local_cache
                    step_inputs["use_cache"] = True
                    step_inputs["output_hidden_states"] = True
                    outputs = model(**step_inputs)
                    logits = outputs.logits / trainer.temperature
                    hidden = outputs.hidden_states[-1]
                    updated_cache_states = _split_past_key_values_batch(outputs.past_key_values, len(latent_bucket))
                    for idx, state in enumerate(latent_bucket):
                        boundary_logit_2d = boundary_logits[idx].view(1, -1)
                        replay_logps[state.sample_idx, state.window_pos - 1] = selective_log_softmax(
                            boundary_logit_2d.float(),
                            replay_input_ids[state.sample_idx, state.window_start + state.window_pos].view(1),
                        ).squeeze(0)
                        if replay_entropy_buf is not None:
                            replay_entropy_buf[state.sample_idx, state.window_pos - 1] = entropy_from_logits(boundary_logit_2d).squeeze(0)
                        state.boundary_logit = logits[idx : idx + 1, :, :]
                        _finalize_latent_step(sample=state, hidden_state=hidden[idx, 0, :])
                        state.cache_state = updated_cache_states[idx]
                        state.window_pos += 1
            for state in latent_group:
                state.segment_idx += 1

        active = [
            state
            for state in replay_states
            if state.window_pos < int(state.window_ids.shape[0]) and state.segment_idx < len(state.segments)
        ]

    replay_logps = replay_logps[:, -logits_to_keep:]
    if replay_entropy_buf is not None:
        replay_entropy_buf = replay_entropy_buf[:, -logits_to_keep:]
    output_logps.index_copy_(0, row_indices, replay_logps)
    if output_entropies is not None and replay_entropy_buf is not None:
        output_entropies.index_copy_(0, row_indices, replay_entropy_buf)

    if memory_states:
        for state in replay_states:
            memory_states[state.row_idx] = state.memory_state

    replay_segment_samples = len(replay_states)
    replay_used = replay_segment_samples > 0
    legit_ratio = replay_segment_samples / batch_size if batch_size > 0 else 0.0
    non_legit_ratio = 1.0 - legit_ratio
    latent_count_mean = latent_count_sum / max(replay_segment_samples, 1)
    latent_steps_mean = latent_count_mean
    forward_passes_mean = float(forward_passes) / max(replay_segment_samples, 1)
    metrics = {
        "legit_ratio": legit_ratio,
        "replay_segment_mixed": 1.0 if replay_used else 0.0,
        "replay_segment_samples": float(replay_segment_samples),
        "replay_non_legit_ratio": non_legit_ratio,
        "replay_latent_count_mean": latent_count_mean,
        "segment_replay_latent_steps_mean": latent_steps_mean,
        "segment_replay_latent_steps_max": float(max_latent_steps),
        "segment_replay_forward_passes_mean": forward_passes_mean,
        "segment_replay_forward_passes_max": float(forward_passes),
    }
    return output_logps, output_entropies, metrics


def _compute_standard_logps_and_entropies(
    trainer: GRPOTrainer,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    compute_entropy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """
    Standard log-probability computation without segment replay.

    Args:
        trainer: GRPO trainer instance
        model: Language model
        input_ids: Input token IDs
        attention_mask: Attention mask
        completion_mask: Completion mask
        compute_entropy: Whether to compute entropy

    Returns:
        Tuple of (logps, entropies, metrics)
    """
    model_inputs = trainer._prepare_model_inputs(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }
    )
    logps, entropies = _run_standard_sequence_forward(
        trainer=trainer,
        model=model,
        model_inputs=model_inputs,
        input_ids=input_ids,
        logits_to_keep=int(getattr(trainer.args, "logits_to_keep", 0) or input_ids.shape[1] - 1),
        compute_entropy=compute_entropy,
    )

    # Empty metrics for standard computation
    metrics = {
        "legit_ratio": 0.0,
        "replay_segment_mixed": 0.0,
        "replay_segment_samples": 0,
        "replay_non_legit_ratio": 0.0,
        "replay_latent_count_mean": 0.0,
        "segment_replay_latent_steps_mean": 0.0,
        "segment_replay_latent_steps_max": 0.0,
        "segment_replay_forward_passes_mean": 1.0,
        "segment_replay_forward_passes_max": 1.0,
    }

    return logps, entropies, metrics


def _patch_grpo_rlsd() -> None:
    if getattr(GRPOTrainer, "_opsd_span_rlsd_patched", False):
        return

    original_prepare_batch_inputs = GRPOTrainer._prepare_batch_inputs
    original_compute_loss_and_metrics = GRPOTrainer._compute_loss_and_metrics
    original_get_per_token_logps_and_entropies_single = GRPOTrainer._get_per_token_logps_and_entropies_single
    original_update_metrics = GRPOTrainer._update_metrics

    def _get_per_token_logps_and_entropies_single(self, model, inputs, compute_entropy=False):
        stage = _strip_text(os.environ.get("OPSD_SPAN_STAGE", "main")).lower()
        if stage not in {"gspo", "main"}:
            return original_get_per_token_logps_and_entropies_single(self, model, inputs, compute_entropy)
        if not _opsd_span_segment_replay_enabled():
            return original_get_per_token_logps_and_entropies_single(self, model, inputs, compute_entropy)
        if self.template.padding_free or self.template.sequence_parallel_size > 1:
            return original_get_per_token_logps_and_entropies_single(self, model, inputs, compute_entropy)

        batch_size = inputs["input_ids"].shape[0]
        memory_states: list[torch.Tensor | None] = [None] * batch_size
        if _delta_memory_enabled() and _opsd_span_replay_delta_mode() == "follow_global":
            hidden_dim = self.model.config.hidden_size
            device = inputs["input_ids"].device
            memory_states = [
                torch.zeros((hidden_dim,), dtype=torch.float32, device=device)
                for _ in range(batch_size)
            ]

        logps, entropies, replay_metrics = _compute_segment_replay_logps_and_entropies(
            trainer=self,
            model=model,
            inputs=inputs,
            memory_states=memory_states,
            compute_entropy=compute_entropy,
        )
        inputs["opsd_span_segment_replay_metrics"] = replay_metrics
        return logps, entropies

    def _prepare_batch_inputs(self, inputs):
        prepared = original_prepare_batch_inputs(self, inputs)
        if _strip_text(os.environ.get("OPSD_SPAN_STAGE", "main")).lower() != "main":
            return prepared
        if not _rlsd_enabled():
            return prepared

        for batch_encoded_inputs in prepared:
            batch_encoded_inputs["opsd_teacher_per_token_logps"] = None
            ref_adapter_name = getattr(self.args, "ref_adapter_name", None)
            if not ref_adapter_name:
                continue
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                self.model.set_adapter(ref_adapter_name)
                teacher_per_token_logps = self._get_per_token_logps_and_entropies(self.model, batch_encoded_inputs)[0]
                self.model.set_adapter(self.model_adapter_name or "default")
            batch_encoded_inputs["opsd_teacher_per_token_logps"] = teacher_per_token_logps
        return prepared

    def _compute_loss_and_metrics(self, model, inputs):
        stage = _strip_text(os.environ.get("OPSD_SPAN_STAGE", "main")).lower()
        if (
            stage not in ["gspo", "main"]
            or self.loss_type not in ["grpo", "bnpo", "dr_grpo", "dapo"]
        ):
            return original_compute_loss_and_metrics(self, model, inputs)

        mode = "train" if self.model.training else "eval"
        completion_mask = inputs["completion_mask"]
        truncated_mask = inputs["truncated_mask"]
        segment_replay_metrics = dict(inputs.get("opsd_span_segment_replay_metrics") or {})
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, inputs, compute_entropy=self.compute_entropy
        )
        segment_replay_metrics = dict(inputs.get("opsd_span_segment_replay_metrics") or segment_replay_metrics)

        entropy_mask = None
        entropy_metrics = {}
        if self.compute_entropy:
            entropies = entropies.masked_fill(completion_mask == 0, float("nan"))
            if self.args.log_entropy:
                per_completion_entropies_mean = torch.nanmean(entropies, dim=1)
                global_per_completion_entropies_mean = gather(per_completion_entropies_mean)
                entropy_metrics = {
                    "entropy_logs": global_per_completion_entropies_mean.tolist(),
                    "entropy_mean": global_per_completion_entropies_mean.nanmean().item(),
                    "entropy_max": nanmax(global_per_completion_entropies_mean).item(),
                    "entropy_min": nanmin(global_per_completion_entropies_mean).item(),
                }
            if self.args.top_entropy_quantile < 1.0:
                entropy_threshold = torch.nanquantile(entropies.flatten().float(), 1 - self.top_entropy_quantile)
                entropy_metrics["entropy_threshold"] = entropy_threshold.item()
                entropy_mask = entropies >= entropy_threshold

        if self.overlong_filter and any(truncated_mask):
            truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & (~truncated_mask)

        if self.beta != 0.0 and not self.kl_in_reward:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            safe_ratio = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)
            per_token_kl = torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)
        else:
            per_token_kl = None

        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ["sequence", "sequence_token"]:
            seq_level_log_weights = (
                (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)
            if self.importance_sampling_level == "sequence":
                log_importance_weights = seq_level_log_weights
            else:
                seq_level_log_weight = seq_level_log_weights.detach()
                log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}.")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        teacher_per_token_logps = inputs.get("opsd_teacher_per_token_logps")
        rlsd_weights, rlsd_metrics = _compute_rlsd_token_weights(
            per_token_logps=per_token_logps,
            teacher_per_token_logps=teacher_per_token_logps,
            completion_mask=completion_mask,
        )

        per_token_loss1 = coef_1 * advantages.unsqueeze(1) * rlsd_weights
        per_token_loss2 = coef_2 * advantages.unsqueeze(1) * rlsd_weights
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            batch_size = completion_mask.shape[0]
            loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            return original_compute_loss_and_metrics(self, model, inputs)

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count

        metrics_data = {
            "mode": mode,
            "entropy": entropy_metrics,
            "completion_mask": completion_mask,
            "completion_token_count": completion_token_count,
            "opsd_span_rlsd": rlsd_metrics,
            "opsd_span_segment_replay": segment_replay_metrics,
        }
        if per_token_kl is not None:
            mean_kl = masked_batch_mean(per_token_kl)
            metrics_data["kl"] = self.accelerator.gather_for_metrics(mean_kl).nanmean().item()

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())
        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        metrics_data["clipping"] = {
            "low_clip_mean": gathered_low_clip.nanmean().item(),
            "low_clip_min": nanmin(gathered_low_clip).item(),
            "high_clip_mean": gathered_high_clip.nanmean().item(),
            "high_clip_max": nanmax(gathered_high_clip).item(),
            "region_clip_mean": gathered_clip_ratio.nanmean().item(),
        }
        return loss, metrics_data

    def _update_metrics(self, metrics_data):
        original_update_metrics(self, metrics_data)
        mode = metrics_data["mode"]

        # Handle RLSD metrics
        rlsd_metrics = metrics_data.get("opsd_span_rlsd")
        if rlsd_metrics:
            for key, value in rlsd_metrics.items():
                self._metrics[mode][f"opsd_span/{key}"].append(float(value))

        # Handle segment replay metrics
        segment_replay_metrics = metrics_data.get("opsd_span_segment_replay")
        if segment_replay_metrics:
            for key, value in segment_replay_metrics.items():
                self._metrics[mode][f"opsd_span/{key}"].append(float(value))

    GRPOTrainer._get_per_token_logps_and_entropies_single = _get_per_token_logps_and_entropies_single
    GRPOTrainer._prepare_batch_inputs = _prepare_batch_inputs
    GRPOTrainer._compute_loss_and_metrics = _compute_loss_and_metrics
    GRPOTrainer._update_metrics = _update_metrics
    GRPOTrainer._opsd_span_rlsd_patched = True


def _patch_single_sample_grpo() -> None:
    if getattr(SwiftGRPOConfig, "_opsd_span_single_sample_patched", False):
        return

    original_swift_check = SwiftGRPOConfig.check_num_generations
    original_trl_post_init = TrlGRPOConfig.__post_init__
    original_compute_advantages = GRPOTrainer._compute_advantages
    original_compute_std = GRPOTrainer.compute_std

    def check_num_generations(self):
        if _single_sample_grpo_enabled() and int(self.num_generations) == 1:
            num_processes = self.world_size
            if self.generation_batch_size % self.num_generations != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                    f"({self.num_generations})."
                )
            if self.eval_strategy != "no":
                num_generations_eval = self.num_generations_eval or self.num_generations
                global_eval_batch_size = self.per_device_eval_batch_size * num_processes
                possible_values = [
                    n_gen for n_gen in range(1, global_eval_batch_size + 1) if global_eval_batch_size % n_gen == 0
                ]
                if num_generations_eval not in possible_values:
                    raise ValueError(
                        f"The global eval batch size ({num_processes} x {self.per_device_eval_batch_size}) must be "
                        f"evenly divisible by the number of generations for eval ({num_generations_eval})."
                    )
            return
        return original_swift_check(self)

    def _trl_post_init(self):
        if not (_single_sample_grpo_enabled() and int(self.num_generations) == 1):
            return original_trl_post_init(self)

        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
        super(TrlGRPOConfig, self).__post_init__()
        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)

        num_processes = self.world_size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError("'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time")

        if self.do_eval and self.eval_strategy != "no":
            num_generations = self.num_generations_eval or self.num_generations
            if (self.per_device_eval_batch_size * num_processes) % num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations})."
                )

        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.delta is not None and self.use_liger_kernel:
            raise ValueError("Liger kernel does not support two-sided GRPO loss yet.")

    def _compute_advantages(self, inputs, rewards_per_func, batch_encoded_inputs):
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else (self.num_generations_eval or self.num_generations)
        if not (_single_sample_grpo_enabled() and int(num_generations) == 1):
            return original_compute_advantages(self, inputs, rewards_per_func, batch_encoded_inputs)

        device = self.accelerator.device
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
        if self.kl_in_reward and self.beta != 0.0:
            kl_list = []
            for batch_encoded in batch_encoded_inputs:
                old_per_token_logps = batch_encoded["old_per_token_logps"]
                ref_per_token_logps = batch_encoded["ref_per_token_logps"]
                completion_mask = batch_encoded["completion_mask"]
                per_token_kl = old_per_token_logps - ref_per_token_logps
                kl = (per_token_kl * completion_mask).sum(-1)
                kl_list.append(kl)
            kl = torch.cat(kl_list, dim=0)
            kl = gather(kl)
            self._metrics[mode]["kl"].append(kl.nanmean().item())
            rewards = rewards - self.beta * kl

        rewards_mean = rewards.mean()
        advantages = rewards - rewards_mean
        if self.scale_rewards == "batch" and rewards.numel() > 1:
            rewards_std = rewards.std().expand_as(rewards)
            advantages = advantages / (rewards_std + 1e-4)

        self._metrics[mode]["reward"].append(rewards_mean.item())
        self._metrics[mode]["reward_std"].append(rewards.std().item() if rewards.numel() > 1 else 0.0)
        self._metrics[mode]["frac_reward_zero_std"].append(1.0 if rewards.numel() <= 1 else float(torch.isclose(rewards.std(), torch.zeros((), device=device)).item()))
        for i, name in enumerate(self.reward_func_names):
            col = rewards_per_func[:, i]
            self._metrics[mode][f"rewards/{name}/mean"].append(torch.nanmean(col).item())
            self._metrics[mode][f"rewards/{name}/std"].append(nanstd(col).item())
            self._logs["rewards"][name].extend(col.tolist())
        return advantages

    def compute_std(self, inputs, rewards_per_func):
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else (self.num_generations_eval or self.num_generations)
        if not (_single_sample_grpo_enabled() and int(num_generations) == 1):
            return original_compute_std(self, inputs, rewards_per_func)

        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
        if rewards.numel() > 1:
            return rewards.std().expand_as(rewards)
        return torch.zeros_like(rewards)

    SwiftGRPOConfig.check_num_generations = check_num_generations
    TrlGRPOConfig.__post_init__ = _trl_post_init
    GRPOTrainer._compute_advantages = _compute_advantages
    GRPOTrainer.compute_std = compute_std
    SwiftGRPOConfig._opsd_span_single_sample_patched = True


def _patch_lora_trainable_tokens() -> None:
    if getattr(swift_peft.LoraConfig, "_opsd_span_trainable_tokens_patched", False):
        return

    original_init = swift_peft.LoraConfig.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("trainable_token_indices") is None:
            token_indices = _resolve_trainable_token_indices()
            if token_indices:
                kwargs["trainable_token_indices"] = token_indices
                logger.info(f"opsd_span trainable_token_indices: {token_indices}")
        original_init(self, *args, **kwargs)

    swift_peft.LoraConfig.__init__ = patched_init
    swift_peft.LoraConfig._opsd_span_trainable_tokens_patched = True


def _materialize_trainable_token_state_dict(
    trainer: RolloutTrainerMixin,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    patched = dict(state_dict)
    materialized = 0

    for module_name, module in trainer.model.named_modules():
        if not isinstance(module, TrainableTokensLayer):
            continue

        clean_name = module_name.removeprefix("base_model.model.")
        clean_name = trainer._fix_param_name_to_vllm(clean_name)
        wrapper_prefix = clean_name.removesuffix(".base_layer")
        wrapper_weight_key = f"{wrapper_prefix}.weight"
        target_weight_key = f"{wrapper_prefix.removesuffix('.token_adapter')}.weight"

        base_weight = patched.pop(wrapper_weight_key, None)
        if base_weight is None:
            continue

        effective_weight = base_weight
        active_adapters = list(getattr(module, "active_adapters", []) or [])
        if not active_adapters:
            active_adapter = getattr(module, "_active_adapter", None)
            if isinstance(active_adapter, str) and active_adapter:
                active_adapters = [active_adapter]
        if not active_adapters:
            active_adapters = [str(name) for name in getattr(module, "token_indices", {}).keys()]

        for adapter_name in active_adapters:
            token_indices = getattr(module, "token_indices", {}).get(adapter_name)
            if not token_indices:
                continue
            delta_key = f"{wrapper_prefix}.trainable_tokens_delta.{adapter_name}"
            delta_weight = patched.pop(delta_key, None)
            if delta_weight is None:
                continue
            index = torch.tensor(token_indices, device=effective_weight.device, dtype=torch.long)
            effective_weight = effective_weight.index_copy(
                0,
                index,
                delta_weight.to(device=effective_weight.device, dtype=effective_weight.dtype),
            )

        patched[target_weight_key] = effective_weight
        materialized += 1

    if materialized:
        logger.info("opsd_span materialized %s trainable-token rollout weights for vLLM sync", materialized)
    return patched


def _patch_rollout_trainable_token_sync() -> None:
    if getattr(RolloutTrainerMixin, "_opsd_span_trainable_token_sync_patched", False):
        return

    original_process_state_dict = RolloutTrainerMixin._process_state_dict_for_vllm
    original_move_full_model_to_vllm = RolloutTrainerMixin._move_full_model_to_vllm

    def _process_state_dict_for_vllm(self, state_dict, is_peft, keep_lora_weights=False):
        processed = original_process_state_dict(self, state_dict, is_peft, keep_lora_weights=keep_lora_weights)
        if not processed:
            return processed
        return _materialize_trainable_token_state_dict(self, processed)

    def _move_full_model_to_vllm(self):
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "set_adapter"):
            return original_move_full_model_to_vllm(self)

        target_adapter = getattr(self, "model_adapter_name", None) or "default"
        current_adapter = getattr(model, "active_adapter", None)
        if current_adapter == target_adapter:
            return original_move_full_model_to_vllm(self)

        model.set_adapter(target_adapter)
        try:
            return original_move_full_model_to_vllm(self)
        finally:
            if current_adapter and current_adapter != target_adapter:
                model.set_adapter(current_adapter)

    RolloutTrainerMixin._process_state_dict_for_vllm = _process_state_dict_for_vllm
    RolloutTrainerMixin._move_full_model_to_vllm = _move_full_model_to_vllm
    RolloutTrainerMixin._opsd_span_trainable_token_sync_patched = True


def _resolve_rollout_bootstrap_model_dir(model: Any) -> str | None:
    model_dir = _strip_text(getattr(model, "model_dir", ""))
    adapter_path = _strip_text(os.environ.get("STUDENT_ADAPTER_PATH", ""))
    if not model_dir or not adapter_path:
        return None

    try:
        adapter_meta = resolve_lora_artifacts(model_dir, adapter_path)
    except Exception as exc:
        logger.warning(f"opsd_span rollout bootstrap model_dir resolve failed: {exc}")
        return None

    resolved_model_path = _strip_text(adapter_meta.get("model_path"))
    if not resolved_model_path or resolved_model_path == model_dir:
        return None
    if not Path(resolved_model_path).exists():
        return None
    return resolved_model_path


def _patch_swift_grpo_context_for_rollout_model_dir() -> None:
    if getattr(Swift, "_opsd_span_grpo_context_patched", False):
        return

    original_grpo_context = Swift.grpo_context

    @staticmethod
    @contextmanager
    def patched_grpo_context(model: Any, processor: Any):
        with original_grpo_context(model, processor):
            original_model_dir = _strip_text(getattr(model, "model_dir", ""))
            bootstrap_model_dir = _resolve_rollout_bootstrap_model_dir(model)
            if not original_model_dir or not bootstrap_model_dir:
                yield
                return

            setattr(model, "model_dir", bootstrap_model_dir)
            logger.info(
                "opsd_span rollout bootstrap model_dir: %s -> %s",
                original_model_dir,
                bootstrap_model_dir,
            )
            try:
                yield
            finally:
                setattr(model, "model_dir", original_model_dir)

    Swift.grpo_context = patched_grpo_context
    Swift._opsd_span_grpo_context_patched = True


class OpsdSpanPreprocessor(RowPreprocessor):
    def __init__(
        self,
        *,
        config_path: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        student_template: str = DEFAULT_OPSD_SPAN_STUDENT_TEMPLATE,
        teacher_template: str = DEFAULT_TEACHER_TEMPLATE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.system_prompt = _strip_text(os.environ.get("OPSD_SYSTEM_PROMPT", system_prompt))
        self.student_template = str(os.environ.get("OPSD_STUDENT_TEMPLATE", student_template))
        self.teacher_template = str(os.environ.get("OPSD_TEACHER_TEMPLATE", teacher_template))
        self.stage = _strip_text(os.environ.get("OPSD_SPAN_STAGE", "main")).lower() or "main"

        if config_path:
            config = _load_json(config_path)
            self.system_prompt = _strip_text(config.get("system_prompt", self.system_prompt))
            self.student_template = str(config.get("student_template", self.student_template))
            self.teacher_template = str(config.get("teacher_template", self.teacher_template))

    def preprocess(self, row: dict[str, Any]) -> dict[str, Any] | None:
        question_text = _strip_text(row.get("student_user_text"))
        answer_text = _strip_text(row.get("answer_text"))
        compressed_target = _strip_text(row.get("compressed_target"))
        teacher_reference_target = _strip_text(row.get("teacher_reference_target"))
        compressed_trace = row.get("compressed_trace") or {}
        thinking_text = _strip_text(row.get("teacher_solution_text"))
        spans = [str(span) for span in list(compressed_trace.get("spans") or []) if str(span).strip()]
        full_target = _strip_text(row.get("full_target")) or _build_full_target(spans, answer_text, thinking_text)
        if not question_text or not compressed_target or not full_target:
            return None

        image_paths = _resolve_image_paths(list(row.get("question_images") or []))
        student_user = self.student_template.format(question_text=question_text).strip()
        assistant_text = compressed_target
        record = {
            "messages": _build_messages(
                system_prompt=self.system_prompt,
                user_text=student_user,
                assistant_text=assistant_text,
            ),
            "images": image_paths,
            "answer_text": answer_text,
            "opsd_compressed_target": compressed_target,
            "opsd_compressed_trace": compressed_trace,
            "opsd_full_messages": _build_messages(
                system_prompt=self.system_prompt,
                user_text=student_user,
                assistant_text=full_target,
            ),
            "opsd_full_target": full_target,
            "opsd_image_paths": image_paths,
            "opsd_student_user": student_user,
            "opsd_system_prompt": self.system_prompt,
        }

        if self.stage != "warmup" and teacher_reference_target:
            record["teacher_prompt"] = self.teacher_template.format(
                question_text=question_text,
                reference_solution=teacher_reference_target,
            ).strip()

        return record


_DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "opsd_manifest" / "r1ov_opsd_span.jsonl"
)

_patch_qwen3vl_gradient_checkpointing_warning()
_patch_seq2seq_latent_loss()
_patch_gkd_latent_loss()
_patch_single_sample_grpo()
_patch_grpo_rlsd()
_patch_lora_trainable_tokens()
_patch_rollout_trainable_token_sync()
_patch_swift_grpo_context_for_rollout_model_dir()
orms["opsd_span_answer"] = OpsdSpanAnswerReward
orms["opsd_span_latent_format"] = OpsdSpanLatentFormatReward

register_dataset(
    DatasetMeta(
        dataset_name="qwen3vl_opsd_span",
        dataset_path=os.environ.get("OPSD_SPAN_DATASET_PATH", str(_DEFAULT_DATASET_PATH)),
        preprocess_func=OpsdSpanPreprocessor(),
    ),
    exist_ok=True,
)
