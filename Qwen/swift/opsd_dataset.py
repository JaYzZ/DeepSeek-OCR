#!/usr/bin/env python3
"""Swift dataset registration for the Qwen3-VL OPSD manifest."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from swift import TrainerCallback, callbacks_map
from swift.arguments import RLHFArguments
from swift.dataset import RowPreprocessor, register_dataset
from swift.dataset.register import DatasetMeta
from swift.infer_engine.grpo_vllm_engine import GRPOVllmEngine
from swift.infer_engine.infer_engine import InferEngine
from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin
from swift.rlhf_trainers.gkd_trainer import GKDTrainer
from transformers import PreTrainedModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from vllm_thinking.trace_store import get_request_trace

from Qwen.llamafactory.integration import (
    LatentVAE,
    _align_module_to_model_dtype_device,
    _resolve_latent_vae_module,
    save_vae_checkpoint,
)


DEFAULT_SYSTEM_PROMPT = ""

DEFAULT_STUDENT_TEMPLATE = """Answer the question.

Question:
{question_text}"""

DEFAULT_TEACHER_TEMPLATE = """Problem:
{question_text}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step. Do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above."""


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _opsd_token_loss_type() -> str:
    return str(os.environ.get("OPSD_TOKEN_LOSS_TYPE", "jsd")).strip().lower()


def _sampled_log_probs_from_logits(
    *,
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    temperature: float,
    chunk_size: int = 0,
) -> torch.Tensor:
    if chunk_size and int(chunk_size) > 0 and logits.shape[1] > int(chunk_size):
        chunks: list[torch.Tensor] = []
        step = int(chunk_size)
        for start in range(0, int(logits.shape[1]), step):
            end = min(start + step, int(logits.shape[1]))
            chunks.append(
                _sampled_log_probs_from_logits(
                    logits=logits[:, start:end, :],
                    sampled_token_ids=sampled_token_ids[:, start:end],
                    temperature=temperature,
                    chunk_size=0,
                )
            )
        return torch.cat(chunks, dim=1)

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


def _get_model_hidden_size(model: Any) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    for candidate in (
        getattr(config, "hidden_size", None),
        getattr(text_config, "hidden_size", None),
    ):
        if candidate is not None:
            return int(candidate)

    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return _get_model_hidden_size(base_model)

    nested_model = getattr(model, "model", None)
    if nested_model is not None and nested_model is not model:
        return _get_model_hidden_size(nested_model)

    raise RuntimeError("Failed to resolve model hidden size for LatentVAE creation.")


def _resolve_vae_checkpoint_path() -> Path | None:
    for env_name in ("QWEN3VL_VAE_CHECKPOINT_PATH", "OPSD_STUDENT_ADAPTER_PATH", "VLLM_LORA_CHECKPOINT_PATH"):
        raw_path = os.environ.get(env_name, "").strip()
        if not raw_path:
            continue
        checkpoint_path = Path(raw_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "vae.safetensors"
        if checkpoint_path.is_file():
            return checkpoint_path
    return None


def _ensure_swift_vae_ready(model: Any) -> dict[str, float]:
    checkpoint_path = _resolve_vae_checkpoint_path()
    explicit_enabled = os.environ.get("QWEN3VL_VAE_ENABLED", "").strip().lower()
    vae_enabled = checkpoint_path is not None or explicit_enabled in {"1", "true", "yes", "on"}
    if not vae_enabled:
        return {"vae_enabled": 0.0, "vae_trainable_params": 0.0, "vae_total_params": 0.0}

    vae = _resolve_latent_vae_module(model)
    if vae is None:
        vae = LatentVAE(
            hidden_size=_get_model_hidden_size(model),
            intermediate_size=int(os.environ.get("QWEN3VL_VAE_INTERMEDIATE_SIZE", "512")),
            deterministic=False,
        )
        _align_module_to_model_dtype_device(model, vae)
        model.register_module("latent_vae", vae)

    if checkpoint_path is not None:
        vae.load_state_dict(load_file(str(checkpoint_path)), strict=True)
        os.environ["QWEN3VL_VAE_CHECKPOINT_PATH"] = str(checkpoint_path)

    _align_module_to_model_dtype_device(model, vae)
    vae = _resolve_latent_vae_module(model)
    trainable = os.environ.get("QWEN3VL_VAE_TRAINABLE", "1").strip().lower() in {"1", "true", "yes", "on"}
    for param in vae.parameters():
        param.requires_grad_(trainable)
    vae.train(trainable)

    total_params = sum(param.numel() for param in vae.parameters())
    trainable_params = sum(param.numel() for param in vae.parameters() if param.requires_grad)
    return {
        "vae_enabled": 1.0,
        "vae_trainable_params": float(trainable_params),
        "vae_total_params": float(total_params),
    }


def _ensure_vae_in_optimizer(trainer: Any) -> None:
    optimizer = getattr(trainer, "optimizer", None)
    model = getattr(trainer, "model", None)
    if optimizer is None or model is None:
        return

    vae_params = [param for name, param in model.named_parameters() if "latent_vae" in name and param.requires_grad]
    if not vae_params:
        return

    existing_param_ids = {id(param) for group in optimizer.param_groups for param in group.get("params", [])}
    missing_params = [param for param in vae_params if id(param) not in existing_param_ids]
    if missing_params:
        optimizer.add_param_group({"params": missing_params})


def _collect_vae_grad_metrics(model: Any) -> dict[str, float]:
    vae = _resolve_latent_vae_module(model)
    if vae is None:
        return {"vae_grad_params": 0.0, "vae_nonzero_grad_params": 0.0, "vae_grad_abs_max": 0.0}

    grad_params = 0
    nonzero_grad_params = 0
    grad_abs_max = 0.0
    for param in vae.parameters():
        grad = param.grad
        if grad is None:
            continue
        grad_params += param.numel()
        current_abs_max = float(grad.detach().abs().max().item()) if grad.numel() > 0 else 0.0
        if current_abs_max > 0.0:
            nonzero_grad_params += param.numel()
        grad_abs_max = max(grad_abs_max, current_abs_max)
    return {
        "vae_grad_params": float(grad_params),
        "vae_nonzero_grad_params": float(nonzero_grad_params),
        "vae_grad_abs_max": grad_abs_max,
    }


def _to_tensor(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    tensor = tensor.to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _trace_metrics_from_data(inputs: list[dict[str, Any]]) -> dict[str, float]:
    traced_samples = 0
    active_positions = 0
    for item in inputs:
        trace = item.get("opsd_continuous_trace")
        if not isinstance(trace, dict):
            continue
        traced_samples += 1
        active_positions += int(trace.get("active_positions", 0) or 0)
    return {
        "rollout_traced_samples": float(traced_samples),
        "rollout_continuous_positions": float(active_positions),
    }


def _load_vllm_trace_for_request(request_id: str) -> dict[str, Any] | None:
    return get_request_trace(str(request_id))


def _build_trace_list(inputs: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    traces: list[dict[str, Any] | None] = []
    for item in inputs:
        trace = item.get("opsd_continuous_trace")
        traces.append(trace if isinstance(trace, dict) else None)
    return traces


def _continuous_trace_loss(
    *,
    model: Any,
    labels: torch.Tensor,
    traces: list[dict[str, Any] | None],
    metrics: dict[str, float],
) -> torch.Tensor | None:
    vae = _resolve_latent_vae_module(model)
    if vae is None or not traces:
        return None

    losses: list[torch.Tensor] = []
    traced_tokens = 0
    matched_sequences = 0
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    for trace in traces[: int(labels.shape[0])]:
        if not isinstance(trace, dict):
            continue

        mask = _to_tensor(trace.get("continuous_token_mask"), device=device, dtype=torch.bool)
        hidden_states = _to_tensor(trace.get("continuous_hidden_states"), device=device, dtype=dtype)
        latent_embeddings = _to_tensor(trace.get("continuous_latent_embeddings"), device=device, dtype=dtype)
        if mask is None or hidden_states is None or latent_embeddings is None:
            continue
        active_positions = int(mask.sum().item()) if mask.numel() > 0 else 0
        if active_positions <= 0 or hidden_states.numel() == 0 or latent_embeddings.numel() == 0:
            continue
        steps = min(active_positions, int(hidden_states.shape[0]), int(latent_embeddings.shape[0]))
        if steps <= 0:
            continue

        hidden = hidden_states[:steps].reshape(steps, -1)
        latent = latent_embeddings[:steps].reshape(steps, -1)
        dist = vae.forward(hidden, temperature=1.0)
        losses.append(-dist.log_prob(latent).mean())
        traced_tokens += steps
        matched_sequences += 1

    metrics["vae_replay_sequences"] = float(matched_sequences)
    metrics["vae_replay_tokens"] = float(traced_tokens)
    if not losses:
        return None
    return torch.stack(losses).mean()


def _append_opsd_metrics(logs: dict[str, Any], metrics: dict[str, Any] | None) -> None:
    if not metrics:
        return
    for key, value in metrics.items():
        logs[f"opsd/{key}"] = round(float(value), 8)


def _accumulate_opsd_metrics(trainer: Any, metrics: dict[str, Any] | None) -> None:
    if not metrics:
        return

    accumulator = getattr(trainer, "_opsd_metric_accumulator", None)
    if accumulator is None:
        accumulator = {
            "count": 0.0,
            "sums": {},
            "mins": {},
            "maxs": {},
            "latest": {},
        }
        trainer._opsd_metric_accumulator = accumulator

    accumulator["count"] += 1.0
    mean_prefixes = ("advantage", "student_logprob", "teacher_logprob", "response_", "teacher_prompt_", "prompt_")
    mean_keys = {
        "tokens",
        "rollout_traced_samples",
        "rollout_continuous_positions",
        "vae_replay_sequences",
        "vae_replay_tokens",
        "vae_replay_loss",
    }

    for key, raw_value in metrics.items():
        value = float(raw_value)
        if key.endswith("_min"):
            current = accumulator["mins"].get(key)
            accumulator["mins"][key] = value if current is None else min(current, value)
        elif key.endswith("_max"):
            current = accumulator["maxs"].get(key)
            accumulator["maxs"][key] = value if current is None else max(current, value)
        elif key.startswith(mean_prefixes) or key in mean_keys:
            accumulator["sums"][key] = accumulator["sums"].get(key, 0.0) + value
        else:
            accumulator["latest"][key] = value


def _finalize_opsd_metrics(trainer: Any) -> dict[str, float]:
    accumulator = getattr(trainer, "_opsd_metric_accumulator", None)
    if not accumulator or accumulator["count"] <= 0:
        return dict(getattr(trainer, "_opsd_last_metrics", {}) or {})

    count = float(accumulator["count"])
    metrics: dict[str, float] = {}
    for key, value in accumulator["sums"].items():
        metrics[key] = float(value) / count
    metrics.update({key: float(value) for key, value in accumulator["mins"].items()})
    metrics.update({key: float(value) for key, value in accumulator["maxs"].items()})
    metrics.update({key: float(value) for key, value in accumulator["latest"].items()})
    metrics["microbatches"] = count

    trainer._opsd_metric_accumulator = None
    trainer._opsd_last_metrics = metrics
    return metrics


def _patch_swift_off_policy_teacher_check() -> None:
    if getattr(RLHFArguments, "_opsd_off_policy_teacher_patched", False):
        return

    original_check_gkd = RLHFArguments._check_gkd

    def _check_gkd(self):
        if self.rlhf_type == "gkd" and _env_flag("OPSD_OFF_POLICY_MODE"):
            original_teacher_model = self.teacher_model
            original_model = self.model
            if original_teacher_model is not None and original_teacher_model == original_model:
                self.model = None
                try:
                    return original_check_gkd(self)
                finally:
                    self.model = original_model
                    self.teacher_model = original_teacher_model
                    self._teacher_use_disable_adapter = False
        return original_check_gkd(self)

    RLHFArguments._check_gkd = _check_gkd
    RLHFArguments._opsd_off_policy_teacher_patched = True


def _patch_grpo_vllm_trace_handoff() -> None:
    if getattr(GRPOVllmEngine, "_opsd_trace_handoff_patched", False):
        return

    original_create_response = GRPOVllmEngine._create_chat_completion_response

    def _create_chat_completion_response(self, result, inputs, request_config, request_id):
        response = original_create_response(self, result, inputs, request_config, request_id)
        if not request_id:
            return response

        trace = _load_vllm_trace_for_request(str(request_id))
        if trace is None:
            return response

        mask = trace.get("continuous_token_mask")
        hidden = trace.get("continuous_hidden_states")
        latent = trace.get("continuous_latent_embeddings")
        response.opsd_continuous_trace = {
            "active_positions": int(mask.sum()) if hasattr(mask, "sum") else 0,
            "mask_length": int(mask.size) if hasattr(mask, "size") else 0,
            "hidden_shape": list(hidden.shape) if hasattr(hidden, "shape") else [],
            "latent_shape": list(latent.shape) if hasattr(latent, "shape") else [],
            "continuous_token_mask": mask,
            "continuous_hidden_states": hidden,
            "continuous_latent_embeddings": latent,
        }
        return response

    GRPOVllmEngine._create_chat_completion_response = _create_chat_completion_response
    GRPOVllmEngine._opsd_trace_handoff_patched = True


def _patch_rollout_trace_postprocess() -> None:
    if getattr(RolloutTrainerMixin, "_opsd_trace_postprocess_patched", False):
        return

    original_postprocess = RolloutTrainerMixin._postprocess_rollout_outputs

    def _postprocess_rollout_outputs(self, inputs, outputs):
        processed = original_postprocess(self, inputs, outputs)
        for item, output in zip(processed, outputs):
            trace = getattr(output.response, "opsd_continuous_trace", None)
            if trace is not None:
                item["opsd_continuous_trace"] = trace
        return processed

    RolloutTrainerMixin._postprocess_rollout_outputs = _postprocess_rollout_outputs
    RolloutTrainerMixin._opsd_trace_postprocess_patched = True


def _patch_opsd_gkd_loss() -> None:
    if getattr(GKDTrainer, "_opsd_sampled_loss_patched", False):
        return

    os.environ["QWEN3VL_OPSD_REPLAY_MODE"] = "1"

    original_init = GKDTrainer.__init__
    original_create_optimizer = GKDTrainer.create_optimizer
    original_training_step = GKDTrainer.training_step
    original_fast_infer = GKDTrainer._fast_infer
    original_save_model = GKDTrainer.save_model
    original_compute_jsd_loss = GKDTrainer._compute_jsd_loss
    original_compute_loss = GKDTrainer.compute_loss
    original_log = GKDTrainer.log

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        vae_metrics = _ensure_swift_vae_ready(self.model)
        self._opsd_vae_metrics = vae_metrics
        self._opsd_metric_accumulator = None
        if not any(isinstance(callback, OpsdMetricsCallback) for callback in self.callback_handler.callbacks):
            self.add_callback(OpsdMetricsCallback(self.args, self))

    def _create_optimizer(self):
        optimizer = original_create_optimizer(self)
        _ensure_vae_in_optimizer(self)
        return optimizer

    def _fast_infer(self, inputs):
        outputs = original_fast_infer(self, inputs)
        metrics = _trace_metrics_from_data(outputs)
        self._opsd_last_traces = _build_trace_list(outputs)
        self._opsd_last_metrics = metrics
        return outputs

    def _training_step(self, model, inputs, num_items_in_batch=None):
        loss = original_training_step(self, model, inputs, num_items_in_batch=num_items_in_batch)
        metrics = dict(getattr(self, "_opsd_last_metrics", {}) or {})
        metrics.update(_collect_vae_grad_metrics(model))
        self._opsd_last_metrics = metrics
        _accumulate_opsd_metrics(self, metrics)
        return loss

    def _save_model(self, output_dir=None, _internal_call=False):
        result = original_save_model(self, output_dir=output_dir, _internal_call=_internal_call)
        vae = _resolve_latent_vae_module(self.model)
        target_dir = output_dir or self.args.output_dir
        if vae is not None and target_dir:
            save_vae_checkpoint(self.model, str(target_dir))
        return result

    def _compute_jsd_loss(self, student_logits, teacher_output, labels):
        opsd_teacher_labels = getattr(teacher_output, "opsd_teacher_labels", None)
        if opsd_teacher_labels is None or teacher_output.is_topk_mode or teacher_output.full_logits is None:
            return original_compute_jsd_loss(self, student_logits, teacher_output, labels)
        token_loss_type = _opsd_token_loss_type()
        if token_loss_type == "jsd":
            return original_compute_jsd_loss(self, student_logits, teacher_output, labels)
        if token_loss_type != "sampled_reverse_kl":
            raise ValueError(f"Unsupported OPSD token loss type: {token_loss_type!r}")

        shifted_labels = torch.roll(labels, shifts=-1, dims=1)
        shifted_teacher_labels = torch.roll(opsd_teacher_labels, shifts=-1, dims=1)
        student_mask = shifted_labels != -100
        teacher_mask = shifted_teacher_labels != -100
        assert student_mask.sum() == teacher_mask.sum(), (
            f"OPSD label count mismatch: student={student_mask.sum().item()}, "
            f"teacher={teacher_mask.sum().item()}. "
            "Student and teacher must share the same response tokens."
        )

        sampled_token_ids = shifted_labels[student_mask][None]
        student_sampled_logits = student_logits[student_mask][None]
        teacher_sampled_logits = teacher_output.full_logits[teacher_mask][None]
        chunk_size = int(os.environ.get("OPSD_LOGPROB_CHUNK_SIZE", "1024"))
        temperature = float(os.environ.get("OPSD_LOSS_TEMPERATURE", str(self.temperature)))

        student_log_probs_sampled = _sampled_log_probs_from_logits(
            logits=student_sampled_logits,
            sampled_token_ids=sampled_token_ids,
            temperature=temperature,
            chunk_size=chunk_size,
        )
        teacher_log_probs_sampled = _sampled_log_probs_from_logits(
            logits=teacher_sampled_logits,
            sampled_token_ids=sampled_token_ids,
            temperature=temperature,
            chunk_size=chunk_size,
        )
        loss, metrics = _sampled_reverse_kl_loss_from_logprobs(
            student_log_probs_sampled=student_log_probs_sampled,
            teacher_log_probs_sampled=teacher_log_probs_sampled,
            labels=sampled_token_ids,
        )
        metrics["tokens"] = float(sampled_token_ids.numel())
        self._opsd_last_metrics = metrics
        return loss

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids = inputs.get("prompt_input_ids")
        prompt_lengths = None
        if prompt_ids is not None:
            prompt_lengths = prompt_ids.ne(self.processing_class.pad_token_id).sum(dim=1)

        labels = inputs.get("labels")
        response_lengths = None
        if labels is not None:
            response_lengths = labels.ne(-100).sum(dim=1)

        opsd_teacher_inputs = inputs.get("_opsd_teacher_inputs")
        teacher_prompt_lengths = None
        if opsd_teacher_inputs is not None and opsd_teacher_inputs.get("input_ids") is not None:
            teacher_prompt_labels = opsd_teacher_inputs.get("labels")
            if teacher_prompt_labels is not None:
                teacher_prompt_lengths = teacher_prompt_labels.eq(-100).sum(dim=1)

        rollout_metrics = dict(getattr(self, "_opsd_last_metrics", {}) or {})
        self._opsd_last_metrics = {}
        result = original_compute_loss(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        metrics = dict(getattr(self, "_opsd_last_metrics", {}) or {})
        metrics.update(rollout_metrics)
        trace_loss = _continuous_trace_loss(
            model=model,
            labels=labels,
            traces=getattr(self, "_opsd_last_traces", []) or [],
            metrics=metrics,
        )
        if trace_loss is not None:
            if return_outputs:
                loss, outputs = result
                result = (loss + trace_loss, outputs)
            else:
                result = result + trace_loss
            metrics["vae_replay_loss"] = float(trace_loss.detach().item())
        metrics.update(getattr(self, "_opsd_vae_metrics", {}) or {})
        metrics["teacher_disable_adapter"] = 1.0 if getattr(self, "_teacher_use_disable_adapter", False) else 0.0
        metrics["teacher_separate_model"] = 1.0 if getattr(self, "teacher_model", None) is not None else 0.0

        if prompt_lengths is not None and prompt_lengths.numel() > 0:
            metrics["prompt_length_mean"] = float(prompt_lengths.float().mean().item())
            metrics["prompt_length_min"] = float(prompt_lengths.min().item())
            metrics["prompt_length_max"] = float(prompt_lengths.max().item())
        if response_lengths is not None and response_lengths.numel() > 0:
            max_completion = max(int(getattr(self.args, "max_completion_length", 0) or 0), 1)
            response_lengths_f = response_lengths.float()
            metrics["response_length_mean"] = float(response_lengths_f.mean().item())
            metrics["response_length_min"] = float(response_lengths.min().item())
            metrics["response_length_max"] = float(response_lengths.max().item())
            metrics["response_clip_ratio"] = float((response_lengths >= max_completion).float().mean().item())
        if teacher_prompt_lengths is not None and teacher_prompt_lengths.numel() > 0:
            teacher_prompt_lengths_f = teacher_prompt_lengths.float()
            metrics["teacher_prompt_length_mean"] = float(teacher_prompt_lengths_f.mean().item())
            metrics["teacher_prompt_length_min"] = float(teacher_prompt_lengths.min().item())
            metrics["teacher_prompt_length_max"] = float(teacher_prompt_lengths.max().item())

        self._opsd_last_metrics = metrics
        return result

    def _log(self, logs, *args, **kwargs):
        _append_opsd_metrics(logs, _finalize_opsd_metrics(self))
        return original_log(self, logs, *args, **kwargs)

    GKDTrainer.__init__ = __init__
    GKDTrainer.create_optimizer = _create_optimizer
    GKDTrainer._fast_infer = _fast_infer
    GKDTrainer.training_step = _training_step
    GKDTrainer.save_model = _save_model
    GKDTrainer._compute_jsd_loss = _compute_jsd_loss
    GKDTrainer.compute_loss = _compute_loss
    GKDTrainer.log = _log
    GKDTrainer._opsd_sampled_loss_patched = True


def _patch_qwen3vl_gradient_checkpointing_warning() -> None:
    original_disable = getattr(Qwen3VLVisionModel, "disable_input_require_grads", None)
    if original_disable is None or getattr(original_disable, "_opsd_patched", False):
        return

    def disable_input_require_grads(self):
        hook = getattr(self, "_require_grads_hook", None)
        if hook is None:
            return
        hook.remove()
        self._require_grads_hook = None

    disable_input_require_grads._opsd_patched = True
    Qwen3VLVisionModel.disable_input_require_grads = disable_input_require_grads


def _patch_multimodal_floating_point_ops_warning() -> None:
    original_estimate_tokens = PreTrainedModel.estimate_tokens
    if getattr(original_estimate_tokens, "_opsd_patched", False):
        return

    def estimate_value_tokens(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            if not torch.is_floating_point(value):
                return value.numel()
            return 0
        if isinstance(value, dict):
            total = 0
            for key in ("input_ids", "prompt_input_ids", "completion_input_ids", "labels"):
                tensor = value.get(key)
                if isinstance(tensor, torch.Tensor):
                    if key == "labels":
                        total += int(tensor.ne(-100).sum().item())
                    else:
                        total += tensor.numel()
            return total
        if isinstance(value, (list, tuple)):
            return sum(estimate_value_tokens(item) for item in value)
        return 0

    def estimate_tokens(self, input_dict):
        token_count = estimate_value_tokens(input_dict)
        if token_count > 0:
            return token_count
        return 0

    estimate_tokens._opsd_patched = True
    PreTrainedModel.estimate_tokens = estimate_tokens


def _patch_swift_logprob_keyerror() -> None:
    original_get_logprobs = InferEngine._get_logprobs
    if getattr(original_get_logprobs, "_opsd_patched", False):
        return

    def _get_logprobs(self, logprobs_list, token_ids, top_logprobs=None):
        if logprobs_list is None or len(token_ids) == 0:
            return None
        if len(token_ids) > 0:
            logprobs_list = logprobs_list[-len(token_ids):]

        res = []
        for logprobs, token_id in zip(logprobs_list, token_ids):
            token = self.tokenizer.decode(token_id)
            token_logprob = logprobs.get(token_id)
            if token_logprob is None:
                if logprobs:
                    token_logprob = max(logprobs.values())
                else:
                    token_logprob = float("-inf")
            item = {"token": token, "logprob": token_logprob, "bytes": list(token.encode("utf8"))}
            if top_logprobs is not None:
                top_items = {k: logprobs[k] for k in sorted(logprobs, key=lambda k: -logprobs[k])[:top_logprobs]}
                res_top_logprobs = []
                for k, logprob in top_items.items():
                    if logprob == float("-inf"):
                        continue
                    top_token = self.tokenizer.decode(k)
                    res_top_logprobs.append(
                        {"token": top_token, "logprob": logprob, "bytes": list(top_token.encode("utf8"))}
                    )
                item["top_logprobs"] = res_top_logprobs
            res.append(item)
        return {"content": res}

    _get_logprobs._opsd_patched = True
    InferEngine._get_logprobs = _get_logprobs


class OpsdMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            _append_opsd_metrics(logs, getattr(self.trainer, "_opsd_last_metrics", None))


callbacks_map["opsd_metrics"] = OpsdMetricsCallback


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _strip_text(value: Any) -> str:
    return str(value or "").strip()


def _strip_image_placeholder(text: str) -> str:
    text = _strip_text(text)
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip()
    return text


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


class OpsdSwiftPreprocessor(RowPreprocessor):
    def __init__(
        self,
        *,
        config_path: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        student_template: str = DEFAULT_STUDENT_TEMPLATE,
        teacher_template: str = DEFAULT_TEACHER_TEMPLATE,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.system_prompt = _strip_text(os.environ.get("OPSD_SYSTEM_PROMPT", system_prompt))
        self.student_template = str(os.environ.get("OPSD_STUDENT_TEMPLATE", student_template))
        self.teacher_template = str(os.environ.get("OPSD_TEACHER_TEMPLATE", teacher_template))
        env_image_min_pixels = os.environ.get("OPSD_IMAGE_MIN_PIXELS")
        env_image_max_pixels = os.environ.get("OPSD_IMAGE_MAX_PIXELS")
        if env_image_min_pixels not in {None, ""}:
            image_min_pixels = int(env_image_min_pixels)
        if env_image_max_pixels not in {None, ""}:
            image_max_pixels = int(env_image_max_pixels)
        self.image_min_pixels = int(image_min_pixels) if image_min_pixels is not None else None
        self.image_max_pixels = int(image_max_pixels) if image_max_pixels is not None else None

        if config_path:
            config = _load_json(config_path)
            self.system_prompt = _strip_text(config.get("system_prompt", self.system_prompt))
            self.student_template = str(config.get("student_template", self.student_template))
            self.teacher_template = str(config.get("teacher_template", self.teacher_template))
            image_min_pixels = config.get("image_min_pixels", self.image_min_pixels)
            image_max_pixels = config.get("image_max_pixels", self.image_max_pixels)
            self.image_min_pixels = int(image_min_pixels) if image_min_pixels is not None else None
            self.image_max_pixels = int(image_max_pixels) if image_max_pixels is not None else None

    def preprocess(self, row: dict[str, Any]) -> dict[str, Any] | None:
        question_text = _strip_image_placeholder(row.get("student_user_text") or "")
        assistant_target = _strip_text(row.get("assistant_target"))
        teacher_solution_text = _strip_text(row.get("teacher_solution_text"))
        if not question_text or not assistant_target or not teacher_solution_text:
            return None

        image_paths = _resolve_image_paths(list(row.get("question_images") or []))
        student_user = self.student_template.format(question_text=question_text).strip()
        teacher_prompt = self.teacher_template.format(
            question_text=question_text,
            reference_solution=teacher_solution_text,
        ).strip()

        return {
            "messages": _build_messages(
                system_prompt=self.system_prompt,
                user_text=student_user,
                assistant_text=assistant_target,
            ),
            "images": image_paths,
            "teacher_prompt": teacher_prompt,
        }


_DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "opsd_manifest" / "r1ov_deepvision_opsd.jsonl"
)

_patch_swift_off_policy_teacher_check()
_patch_grpo_vllm_trace_handoff()
_patch_rollout_trace_postprocess()
_patch_opsd_gkd_loss()
_patch_qwen3vl_gradient_checkpointing_warning()
_patch_multimodal_floating_point_ops_warning()
_patch_swift_logprob_keyerror()

register_dataset(
    DatasetMeta(
        dataset_name="qwen3vl_opsd_swift",
        dataset_path=os.environ.get("OPSD_SWIFT_DATASET_PATH", str(_DEFAULT_DATASET_PATH)),
        preprocess_func=OpsdSwiftPreprocessor(),
    ),
    exist_ok=True,
)
