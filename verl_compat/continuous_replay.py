"""Repo-local continuous replay patches for VERL GSPO."""

from __future__ import annotations

import importlib
import logging
import os
import functools
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.fs import copy_to_local
from vllm_thinking.trace_store import get_request_trace, pop_request_trace

logger = logging.getLogger(__name__)

CONTINUOUS_HIDDEN_KEY = "continuous_hidden_states"
CONTINUOUS_LATENT_KEY = "continuous_latent_embeddings"
CONTINUOUS_LATENT_LOGPROB_KEY = "continuous_latent_log_probs"
CONTINUOUS_MASK_KEY = "continuous_token_mask"
OPSD_TEACHER_PROMPT_IDS_KEY = "opsd_teacher_prompt_ids"
_PATCHED = False
_REPLAY_DTYPE_CAST_LOGGED = False
_GRAD_FLOW_LOGGED = False
_GRAD_FLOW_DEBUG = os.environ.get("QWEN3VL_CONTINUOUS_REPLAY_DEBUG", "0") == "1"
_POLICY_VAE_LOGGED = False
_ROLLOUT_CERT_LOGGED = False
_REPLAY_CERT_LOGGED = False
_REPLAY_DEBUG_LOGGED = False
_OPSD_LOSS_LOGGED = False
_VLLM_LOGPROB_FALLBACK_LOGGED = False
_OPSD_TEACHER_LOAD_LOGGED = False
_TRACE_ATTACH_LOGGED = False


def _replay_debug(message: str, *args) -> None:
    if _GRAD_FLOW_DEBUG:
        logger.warning(message, *args)


def _log_multimodal_mismatch_debug(
    *,
    modality: str,
    input_ids: torch.Tensor,
    token_id: int,
    feature_count: int,
    grid_thw: torch.Tensor | None,
) -> None:
    """Log enough sequence context to debug reserved multimodal token mismatches."""

    input_ids_cpu = input_ids.detach().to(device="cpu", dtype=torch.long)
    token_mask = input_ids_cpu == int(token_id)
    token_count = int(token_mask.sum().item())
    per_row_counts = token_mask.sum(dim=-1).tolist() if token_mask.ndim == 2 else [token_count]
    token_positions = torch.nonzero(token_mask, as_tuple=False)
    position_pairs = token_positions.tolist()
    raw_ids = input_ids_cpu.tolist()
    tail_ids = input_ids_cpu[:, -64:].tolist() if input_ids_cpu.ndim == 2 else input_ids_cpu[-64:].tolist()

    logger.error(
        "[ContinuousReplay][MM] %s token/feature mismatch: token_id=%s tokens=%s features=%s "
        "input_shape=%s grid_thw=%s per_row_counts=%s last_positions=%s tail_ids=%s raw_input_ids=%s",
        modality,
        int(token_id),
        token_count,
        int(feature_count),
        tuple(input_ids_cpu.shape),
        None if grid_thw is None else grid_thw.detach().to(device="cpu", dtype=torch.long).tolist(),
        per_row_counts,
        position_pairs[-16:],
        tail_ids,
        raw_ids,
    )
    logger.error(
        "[ContinuousReplay][MM] full_%s_token_positions=%s",
        modality,
        position_pairs,
    )


def _resolve_multimodal_prompt_mask(
    *,
    input_ids: torch.Tensor,
    prompt_positions_mask: torch.Tensor | None,
) -> torch.Tensor:
    if prompt_positions_mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)

    mask = prompt_positions_mask.to(device=input_ids.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(input_ids.shape):
        raise RuntimeError(
            "[ContinuousReplay][MM] prompt mask shape mismatch: "
            f"input_ids.shape={tuple(input_ids.shape)} prompt_mask.shape={tuple(mask.shape)}"
        )
    return mask


def _validate_multimodal_token_match(
    *,
    modality: str,
    input_ids: torch.Tensor,
    token_id: int,
    feature_count: int,
    grid_thw: torch.Tensor | None,
    prompt_positions_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Validate multimodal token/feature parity and return the exact scatter mask."""

    prompt_mask = _resolve_multimodal_prompt_mask(
        input_ids=input_ids,
        prompt_positions_mask=prompt_positions_mask,
    )
    mask = (input_ids == int(token_id)) & prompt_mask
    token_count = int(mask.sum().item())
    if token_count == feature_count:
        return mask

    _log_multimodal_mismatch_debug(
        modality=modality,
        input_ids=input_ids,
        token_id=token_id,
        feature_count=feature_count,
        grid_thw=grid_thw,
    )

    raise ValueError(
        f"{modality.capitalize()} features and {modality} tokens do not match: "
        f"tokens: {token_count}, features {feature_count}"
    )


def _inject_latent_log_probs_into_rollout(
    *,
    curr_log_prob: list[float],
    mask_row: np.ndarray,
    latent_log_probs: np.ndarray,
    request_id: str,
) -> None:
    true_positions = np.flatnonzero(mask_row)
    if latent_log_probs.shape[0] != true_positions.size:
        raise RuntimeError(
            "Continuous replay latent log_prob mismatch for request_id="
            f"{request_id}: latent_log_probs={latent_log_probs.shape[0]} active_positions={true_positions.size}"
        )

    if true_positions.size == 0:
        return

    max_position = int(true_positions.max())
    if max_position >= len(curr_log_prob):
        raise RuntimeError(
            "Continuous replay latent log_prob position overflow for request_id="
            f"{request_id}: max_position={max_position} rollout_log_probs={len(curr_log_prob)} "
            f"active_positions={true_positions.size}"
        )

    for pos_idx, position in enumerate(true_positions.tolist()):
        curr_log_prob[int(position)] = float(latent_log_probs[pos_idx])


def _trim_request_trace_to_actual_response_length(
    *,
    hidden_row: np.ndarray,
    latent_row: np.ndarray,
    mask_row: np.ndarray,
    latent_log_probs: np.ndarray | None,
    actual_response_length: int,
    request_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    trimmed_mask = np.asarray(mask_row, dtype=np.bool_).reshape(-1)
    if trimmed_mask.size > actual_response_length:
        trimmed_mask = trimmed_mask[:actual_response_length]

    kept_steps = int(trimmed_mask.sum())
    hidden_steps = int(hidden_row.shape[0]) if hidden_row.ndim == 2 else 0
    latent_steps = int(latent_row.shape[0]) if latent_row.ndim == 2 else 0
    if hidden_steps < kept_steps or latent_steps < kept_steps:
        raise RuntimeError(
            "Continuous replay trace trim underflow for request_id="
            f"{request_id}: kept_steps={kept_steps} hidden_steps={hidden_steps} latent_steps={latent_steps} "
            f"actual_response_length={actual_response_length}"
        )

    trimmed_hidden = hidden_row[:kept_steps]
    trimmed_latent = latent_row[:kept_steps]

    if latent_log_probs is None:
        return trimmed_hidden, trimmed_latent, trimmed_mask, None

    trimmed_log_probs = np.asarray(latent_log_probs, dtype=np.float32).reshape(-1)
    if trimmed_log_probs.shape[0] < kept_steps:
        raise RuntimeError(
            "Continuous replay latent log_prob trim underflow for request_id="
            f"{request_id}: kept_steps={kept_steps} latent_log_probs={trimmed_log_probs.shape[0]} "
            f"actual_response_length={actual_response_length}"
        )
    trimmed_log_probs = trimmed_log_probs[:kept_steps]
    return trimmed_hidden, trimmed_latent, trimmed_mask, trimmed_log_probs


@dataclass
class ReplayBufferPool:
    """Pre-allocated scratch buffers for continuous replay construction."""

    replay_row_ids: torch.Tensor | None = None
    hidden_buffer: torch.Tensor | None = None
    latent_buffer: torch.Tensor | None = None
    position_buffer: torch.Tensor | None = None
    max_batch_size: int = 0
    max_seq_length: int = 0
    max_continuous_steps: int = 0
    device: torch.device | None = None
    lock: Lock = Lock()

    def ensure_capacity(
        self,
        batch_size: int,
        seq_length: int,
        continuous_steps: int,
        device: torch.device,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        with self.lock:
            if (
                batch_size <= self.max_batch_size
                and seq_length <= self.max_seq_length
                and continuous_steps <= self.max_continuous_steps
                and self.device == device
                and self.hidden_buffer is not None
                and self.hidden_buffer.dtype == dtype
            ):
                return

            self.max_batch_size = max(batch_size, 1)
            self.max_seq_length = max(seq_length, 1)
            self.max_continuous_steps = max(continuous_steps, 1)
            self.device = device

            self.replay_row_ids = torch.full(
                (self.max_batch_size, self.max_seq_length),
                -1,
                dtype=torch.long,
                device=device,
            )
            self.hidden_buffer = torch.empty(
                (self.max_continuous_steps, hidden_size),
                dtype=dtype,
                device=device,
            )
            self.latent_buffer = torch.empty(
                (self.max_continuous_steps, hidden_size),
                dtype=dtype,
                device=device,
            )
            self.position_buffer = torch.empty(
                (self.max_continuous_steps,),
                dtype=torch.long,
                device=device,
            )


_buffer_pool = ReplayBufferPool()


class LatentVAE(nn.Module):
    """Minimal LatentVAE copy used for RL replay without importing SFT patches."""

    def __init__(self, hidden_size: int, intermediate_size: int = 512, deterministic: bool = False):
        super().__init__()
        self.deterministic = deterministic
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )
        self.mean = nn.Linear(intermediate_size, hidden_size)
        if not deterministic:
            self.log_std = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.distributions.Normal:
        x = self.fc(x)
        mean = self.mean(x)
        if self.deterministic:
            return torch.distributions.Normal(mean, torch.ones_like(mean) * 1e-9)
        log_std = self.log_std(x)
        std = log_std.exp() * temperature
        return torch.distributions.Normal(mean, std)


def _resolve_model_config(model) -> object | None:
    visited = set()
    queue = [model]
    while queue:
        node = queue.pop(0)
        if node is None:
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        config = getattr(node, "config", None)
        if config is not None:
            return config

        for attr_name in ("module", "_fsdp_wrapped_module", "model", "base_model", "pretrained_model"):
            child = getattr(node, attr_name, None)
            if child is not None:
                queue.append(child)
    return None


def _resolve_input_embedding_layer(model):
    visited = set()
    queue = [model]
    while queue:
        node = queue.pop(0)
        if node is None:
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        getter = getattr(node, "get_input_embeddings", None)
        if callable(getter):
            try:
                layer = getter()
            except Exception:
                layer = None
            unwrapped = _unwrap_embedding_candidate(layer)
            if unwrapped is not None:
                return unwrapped

        for attr_name in ("module", "_fsdp_wrapped_module", "model", "base_model", "pretrained_model"):
            child = getattr(node, attr_name, None)
            if child is not None:
                queue.append(child)
    return None


def _resolve_qwen3vl_model(model):
    visited = set()
    queue = [model]
    while queue:
        node = queue.pop(0)
        if node is None:
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        class_name = node.__class__.__name__
        if class_name == "Qwen3VLForConditionalGeneration":
            return node

        for attr_name in ("module", "_fsdp_wrapped_module", "model", "base_model", "pretrained_model"):
            child = getattr(node, attr_name, None)
            if child is not None:
                queue.append(child)
    return None


def _unwrap_embedding_candidate(module):
    visited = set()
    queue = [module]
    while queue:
        node = queue.pop(0)
        if node is None:
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        if isinstance(node, nn.Embedding):
            weight = getattr(node, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return node

        weight = getattr(node, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2 and callable(getattr(node, "forward", None)):
            return node

        base_layer_getter = getattr(node, "get_base_layer", None)
        if callable(base_layer_getter):
            try:
                queue.append(base_layer_getter())
            except Exception:
                pass

        for attr_name in ("_fsdp_wrapped_module", "_orig_mod", "_checkpoint_wrapped_module", "base_layer", "original_module", "module", "token_adapter", "model"):
            child = getattr(node, attr_name, None)
            if child is not None:
                queue.append(child)
    return None


def _resolve_hidden_size(model) -> int:
    config = _resolve_model_config(model)
    if config is None:
        raise RuntimeError("Could not resolve model config for latent replay")

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        text_config = getattr(config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Could not resolve hidden_size for latent replay")
    return int(hidden_size)


def _checkpoint_if_needed(function, *args):
    if not torch.is_grad_enabled():
        return function(*args)
    return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False)


def _resolve_module_compute_dtype(module, reference: torch.Tensor) -> torch.dtype:
    if reference.device.type == "cuda" and torch.is_autocast_enabled():
        try:
            return torch.get_autocast_dtype("cuda")
        except (AttributeError, TypeError):
            return torch.get_autocast_gpu_dtype()

    cached = getattr(module, "_continuous_replay_compute_dtype", None)
    if isinstance(cached, torch.dtype):
        return cached

    for candidate in (getattr(module, "language_model", None), module):
        if candidate is None:
            continue
        for tensor in itertools.chain(candidate.parameters(), candidate.buffers()):
            if tensor.is_floating_point():
                module._continuous_replay_compute_dtype = tensor.dtype
                return tensor.dtype

    module._continuous_replay_compute_dtype = reference.dtype
    return reference.dtype


def _cast_qwen3vl_inputs_for_compute_dtype(module, input_kwargs: dict) -> dict:
    global _REPLAY_DTYPE_CAST_LOGGED

    inputs_embeds = input_kwargs.get("inputs_embeds")
    if not isinstance(inputs_embeds, torch.Tensor) or not inputs_embeds.is_floating_point():
        return input_kwargs

    target_dtype = _resolve_module_compute_dtype(module, inputs_embeds)
    if target_dtype not in (torch.float16, torch.bfloat16) or inputs_embeds.dtype == target_dtype:
        return input_kwargs

    input_kwargs["inputs_embeds"] = inputs_embeds.to(dtype=target_dtype)
    deepstack_visual_embeds = input_kwargs.get("deepstack_visual_embeds")
    if isinstance(deepstack_visual_embeds, list):
        input_kwargs["deepstack_visual_embeds"] = [
            tensor.to(dtype=target_dtype) if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() else tensor
            for tensor in deepstack_visual_embeds
        ]

    if not _REPLAY_DTYPE_CAST_LOGGED:
        logger.warning(
            "[ContinuousReplay] Cast Qwen3VL replay inputs from %s to compute dtype %s.",
            inputs_embeds.dtype,
            target_dtype,
        )
        _REPLAY_DTYPE_CAST_LOGGED = True
    return input_kwargs


def _collect_param_grad_stats(params: Iterable[torch.nn.Parameter]) -> dict[str, float | int | bool]:
    trainable_params = 0
    trainable_elems = 0
    grad_params = 0
    grad_elems = 0
    nonzero_grad_params = 0
    grad_abs_max = 0.0

    for param in params:
        if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
            continue
        trainable_params += 1
        trainable_elems += int(param.numel())
        grad = param.grad
        if grad is None:
            continue
        grad_params += 1
        grad_elems += int(grad.numel())
        grad_detached = grad.detach()
        if grad_detached.numel() > 0:
            current_abs_max = float(grad_detached.abs().max().item())
            grad_abs_max = max(grad_abs_max, current_abs_max)
            if current_abs_max > 0:
                nonzero_grad_params += 1

    return {
        "trainable_params": trainable_params,
        "trainable_elems": trainable_elems,
        "grad_params": grad_params,
        "grad_elems": grad_elems,
        "nonzero_grad_params": nonzero_grad_params,
        "grad_abs_max": grad_abs_max,
        "has_any_grad": grad_params > 0,
    }


def _iter_named_trainable_actor_params(actor_module):
    seen = set()
    queue = [actor_module]
    for attr_name in ("_fsdp_wrapped_module", "module", "_orig_mod"):
        child = getattr(actor_module, attr_name, None)
        if child is not None:
            queue.append(child)

    for module in queue:
        if module is None:
            continue
        for name, param in module.named_parameters(recurse=True):
            if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            yield name, param


def _should_log_grad_flow(policy) -> bool:
    rank = getattr(policy, "rank", None)
    if rank is not None:
        return int(rank) == 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _log_grad_flow_probe(policy, micro_batch) -> None:
    global _GRAD_FLOW_LOGGED
    if _GRAD_FLOW_LOGGED or not _should_log_grad_flow(policy):
        return

    latent_vae = getattr(policy, "latent_vae", None)
    vae_stats = _collect_param_grad_stats(latent_vae.parameters()) if latent_vae is not None else None

    latent_vae_param_ids = {id(param) for param in latent_vae.parameters()} if latent_vae is not None else set()
    optimizer_non_vae_params = []
    actor_optimizer = getattr(policy, "actor_optimizer", None)
    if actor_optimizer is not None:
        for group in actor_optimizer.param_groups:
            for param in group.get("params", []):
                if id(param) not in latent_vae_param_ids:
                    optimizer_non_vae_params.append(param)
    actor_optimizer_stats = _collect_param_grad_stats(optimizer_non_vae_params)

    named_lora_params = [
        param for name, param in _iter_named_trainable_actor_params(policy.actor_module) if "lora_" in name.lower()
    ]
    named_lora_stats = _collect_param_grad_stats(named_lora_params)

    response_mask = micro_batch["response_mask"]
    response_supervised_tokens = int(response_mask.to(torch.int64).sum().item())
    continuous_positions = 0
    for mask_row in _iter_object_array_rows(micro_batch.get(CONTINUOUS_MASK_KEY)):
        if mask_row is None:
            continue
        continuous_positions += int(np.asarray(mask_row, dtype=np.bool_).sum())

    logger.warning(
        "[ContinuousReplay] Grad probe: response_tokens=%s continuous_positions=%s "
        "vae_trainable_params=%s vae_grad_params=%s vae_nonzero_grad_params=%s vae_grad_abs_max=%.6g "
        "actor_non_vae_trainable_params=%s actor_non_vae_grad_params=%s actor_non_vae_nonzero_grad_params=%s actor_non_vae_grad_abs_max=%.6g "
        "named_lora_trainable_params=%s named_lora_grad_params=%s named_lora_nonzero_grad_params=%s named_lora_grad_abs_max=%.6g",
        response_supervised_tokens,
        continuous_positions,
        0 if vae_stats is None else vae_stats["trainable_params"],
        0 if vae_stats is None else vae_stats["grad_params"],
        0 if vae_stats is None else vae_stats["nonzero_grad_params"],
        0.0 if vae_stats is None else vae_stats["grad_abs_max"],
        actor_optimizer_stats["trainable_params"],
        actor_optimizer_stats["grad_params"],
        actor_optimizer_stats["nonzero_grad_params"],
        actor_optimizer_stats["grad_abs_max"],
        named_lora_stats["trainable_params"],
        named_lora_stats["grad_params"],
        named_lora_stats["nonzero_grad_params"],
        named_lora_stats["grad_abs_max"],
    )
    _GRAD_FLOW_LOGGED = True


def _collect_grad_flow_probe_metrics(policy, micro_batch) -> dict[str, float]:
    latent_vae = getattr(policy, "latent_vae", None)
    vae_stats = _collect_param_grad_stats(latent_vae.parameters()) if latent_vae is not None else None

    response_mask = micro_batch["response_mask"]
    response_supervised_tokens = int(response_mask.to(torch.int64).sum().item())
    traced_samples = 0
    continuous_positions = 0
    for mask_row in _iter_object_array_rows(micro_batch.get(CONTINUOUS_MASK_KEY)):
        if mask_row is None:
            continue
        mask_np = np.asarray(mask_row, dtype=np.bool_).reshape(-1)
        if mask_np.size == 0:
            continue
        traced_samples += 1
        continuous_positions += int(mask_np.sum())

    return {
        "actor/replay_response_tokens": float(response_supervised_tokens),
        "actor/replay_traced_samples": float(traced_samples),
        "actor/replay_continuous_positions": float(continuous_positions),
        "actor/vae_enabled": 1.0 if latent_vae is not None else 0.0,
        "actor/vae_trainable_params": 0.0 if vae_stats is None else float(vae_stats["trainable_params"]),
        "actor/vae_grad_params": 0.0 if vae_stats is None else float(vae_stats["grad_params"]),
        "actor/vae_nonzero_grad_params": 0.0 if vae_stats is None else float(vae_stats["nonzero_grad_params"]),
        "actor/vae_grad_abs_max": 0.0 if vae_stats is None else float(vae_stats["grad_abs_max"]),
    }


def _resolve_vae_path(preferred_dir: str | None = None) -> Path | None:
    candidate_dirs = []
    if preferred_dir:
        candidate_dirs.append(preferred_dir)
    for env_name in ("VLLM_LORA_CHECKPOINT_PATH", "INIT_LORA_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidate_dirs.append(env_value)

    for candidate_dir in candidate_dirs:
        vae_path = Path(candidate_dir) / "vae.safetensors"
        if vae_path.is_file():
            return vae_path
    return None


def _load_policy_latent_vae(policy) -> None:
    global _POLICY_VAE_LOGGED
    vae_path = _resolve_vae_path()
    if vae_path is None:
        policy.latent_vae = None
        return
    vae_trainable = os.environ.get("QWEN3VL_VAE_TRAINABLE", "1") == "1"
    has_optimizer = policy.actor_optimizer is not None

    hidden_size = _resolve_hidden_size(policy.actor_module)
    vae = LatentVAE(
        hidden_size=hidden_size,
        intermediate_size=int(os.environ.get("QWEN3VL_VAE_INTERMEDIATE_SIZE", "512")),
        deterministic=False,
    )
    vae.load_state_dict(load_file(str(vae_path)), strict=True)
    vae.train(has_optimizer and vae_trainable)
    for param in vae.parameters():
        param.requires_grad = has_optimizer and vae_trainable
    policy.latent_vae = vae
    policy._qwen3vl_latent_vae_path = str(vae_path)

    if has_optimizer and vae_trainable:
        existing_param_ids = {id(param) for group in policy.actor_optimizer.param_groups for param in group["params"]}
        new_params = [param for param in policy.latent_vae.parameters() if id(param) not in existing_param_ids]
        if new_params:
            policy.actor_optimizer.add_param_group({"params": new_params})
    if not _POLICY_VAE_LOGGED:
        logger.warning(
            "[ContinuousReplay] Actor latent VAE loaded: path=%s trainable=%s params=%s",
            vae_path,
            bool(has_optimizer and vae_trainable),
            sum(param.numel() for param in policy.latent_vae.parameters()),
        )
        _POLICY_VAE_LOGGED = True


def _restore_policy_latent_vae(policy, checkpoint_dir: str | None) -> None:
    if getattr(policy, "latent_vae", None) is None:
        _load_policy_latent_vae(policy)
        if getattr(policy, "latent_vae", None) is None:
            return

    vae_path = None
    if checkpoint_dir:
        for candidate in (
            Path(checkpoint_dir) / "vae.safetensors",
            Path(checkpoint_dir) / "lora_adapter" / "vae.safetensors",
        ):
            if candidate.is_file():
                vae_path = candidate
                break
    if vae_path is None:
        vae_path = _resolve_vae_path(getattr(policy, "_qwen3vl_latent_vae_path", None))
    if vae_path is None:
        return

    policy.latent_vae.load_state_dict(load_file(str(vae_path)), strict=True)
    policy._qwen3vl_latent_vae_path = str(vae_path)


def _opsd_off_policy_enabled() -> bool:
    return str(os.environ.get("OPSD_OFF_POLICY_MODE", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _opsd_offload_teacher_enabled() -> bool:
    return str(os.environ.get("OPSD_OFFLOAD_TEACHER_MODEL", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _config_get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        value = config.get(key, default)
        return default if value is None else value
    return getattr(config, key, default)


def _module_compute_device(module: torch.nn.Module) -> torch.device:
    first_tensor = next(module.parameters(), None)
    if first_tensor is None:
        first_tensor = next(module.buffers(), None)
    if first_tensor is None:
        raise RuntimeError("Could not determine module device for OPSD external teacher.")
    return first_tensor.device


def _load_opsd_external_teacher(policy) -> None:
    global _OPSD_TEACHER_LOAD_LOGGED

    if getattr(policy, "_opsd_external_teacher_ready", False):
        return
    if not _opsd_off_policy_enabled():
        policy._opsd_external_teacher_ready = True
        policy.opsd_external_teacher = None
        return

    teacher_model_path = str(os.environ.get("OPSD_TEACHER_MODEL_PATH", "") or "").strip()
    if not teacher_model_path:
        raise RuntimeError("OPSD_OFF_POLICY_MODE requires OPSD_TEACHER_MODEL_PATH to be set.")

    actor_config = getattr(policy, "config", None)
    actor_model_cfg = _config_get(actor_config, "model", {})
    actor_module = getattr(policy, "actor_module", None)
    local_path = copy_to_local(teacher_model_path, use_shm=_config_get(actor_model_cfg, "use_shm", False))
    trust_remote_code = bool(_config_get(actor_model_cfg, "trust_remote_code", False))

    attn_implementation = "flash_attention_2"
    if not torch.cuda.is_available():
        attn_implementation = "eager"
    teacher_model_config = AutoConfig.from_pretrained(
        local_path,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    has_remote_code = hasattr(teacher_model_config, "auto_map") and any(
        teacher_model_config.architectures[0] in val for val in teacher_model_config.auto_map.values()
    )
    if has_remote_code:
        auto_class = next(
            key for key, value in teacher_model_config.auto_map.items() if teacher_model_config.architectures[0] in value
        )
        match auto_class:
            case "AutoModelForVision2Seq":
                teacher_module_class = AutoModelForVision2Seq
            case "AutoModelForCausalLM":
                teacher_module_class = AutoModelForCausalLM
            case "AutoModelForImageTextToText":
                teacher_module_class = AutoModelForImageTextToText
            case _:
                teacher_module_class = AutoModel
    else:
        if type(teacher_model_config) in AutoModelForVision2Seq._model_mapping.keys():
            teacher_module_class = AutoModelForVision2Seq
        elif type(teacher_model_config) in AutoModelForCausalLM._model_mapping.keys():
            teacher_module_class = AutoModelForCausalLM
        elif type(teacher_model_config) in AutoModelForImageTextToText._model_mapping.keys():
            teacher_module_class = AutoModelForImageTextToText
        else:
            teacher_module_class = AutoModel

    teacher_module = teacher_module_class.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch.bfloat16,
        config=teacher_model_config,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    apply_monkey_patch(
        model=teacher_module,
        use_remove_padding=bool(_config_get(actor_config, "use_remove_padding", False)),
        ulysses_sp_size=_config_get(actor_config, "ulysses_sequence_parallel_size", 1),
        use_fused_kernels=bool(_config_get(actor_model_cfg, "use_fused_kernels", False)),
        fused_kernels_backend=(_config_get(actor_model_cfg, "fused_kernel_options", {}) or {}).get(
            "impl_backend"
        ),
    )

    teacher_adapter_path = str(os.environ.get("OPSD_TEACHER_ADAPTER_PATH", "") or "").strip()
    if teacher_adapter_path:
        local_adapter_path = copy_to_local(teacher_adapter_path, use_shm=_config_get(actor_model_cfg, "use_shm", False))
        teacher_module = PeftModel.from_pretrained(teacher_module, local_adapter_path, is_trainable=False)

    for param in teacher_module.parameters():
        param.requires_grad_(False)
    teacher_module.eval()
    actor_device = next(actor_module.parameters()).device
    if actor_device.type != "cpu":
        teacher_module.to(actor_device)
    if _opsd_offload_teacher_enabled():
        teacher_module.to("cpu")

    policy.opsd_external_teacher = teacher_module
    policy._opsd_external_teacher_ready = True
    if not _OPSD_TEACHER_LOAD_LOGGED:
        logger.warning(
            "[ContinuousReplay][OPSD] Loaded external teacher model=%s adapter=%s offload=%s",
            teacher_model_path,
            teacher_adapter_path or "<none>",
            _opsd_offload_teacher_enabled(),
        )
        _OPSD_TEACHER_LOAD_LOGGED = True


def _has_continuous_replay_inputs(micro_batch) -> bool:
    return (
        CONTINUOUS_HIDDEN_KEY in micro_batch
        and CONTINUOUS_LATENT_KEY in micro_batch
        and CONTINUOUS_MASK_KEY in micro_batch
    )


def _build_prompt_positions_mask(input_ids: torch.Tensor, response_length: int) -> torch.Tensor:
    response_start = input_ids.size(1) - int(response_length)
    return (
        torch.arange(input_ids.size(1), device=input_ids.device)
        .unsqueeze(0)
        .expand(input_ids.size(0), -1)
        < response_start
    )


def _iter_object_array_rows(value) -> Iterable:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _pack_object_rows(rows: list[object]) -> np.ndarray:
    packed = np.empty((len(rows),), dtype=object)
    for idx, row in enumerate(rows):
        packed[idx] = row
    return packed


def _agent_loop_response_length(output: object) -> int:
    response_mask = getattr(output, "response_mask", None)
    if isinstance(response_mask, torch.Tensor):
        return int(response_mask.detach().to(device="cpu", dtype=torch.int64).sum().item())

    response_ids = getattr(output, "response_ids", None)
    if response_ids is None:
        return 0
    if isinstance(response_ids, torch.Tensor):
        response_ids = response_ids.detach().to(device="cpu")
        if response_ids.ndim == 0:
            return int(response_ids.numel())
        if response_ids.ndim == 1:
            return int(response_ids.numel())
        return int(response_ids.shape[-1])
    return len(response_ids)


def _unpack_agent_loop_continuous_traces(
    outputs: list[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    continuous_hidden_states = []
    continuous_latent_embeddings = []
    continuous_token_masks = []

    for output in outputs:
        trace = getattr(output, "extra_fields", {}).get("continuous_trace")
        if trace is None:
            return None

        response_length = _agent_loop_response_length(output)
        hidden_row, latent_row, mask_row = _validate_request_trace(
            trace,
            request_id=str(getattr(output, "request_id", "<unknown>")),
            response_length=response_length,
        )
        hidden_row_trimmed, latent_row_trimmed, mask_row_trimmed, _ = _trim_request_trace_to_actual_response_length(
            hidden_row=hidden_row,
            latent_row=latent_row,
            mask_row=mask_row,
            latent_log_probs=None,
            actual_response_length=response_length,
            request_id=str(getattr(output, "request_id", "<unknown>")),
        )
        continuous_hidden_states.append(hidden_row_trimmed)
        continuous_latent_embeddings.append(latent_row_trimmed)
        continuous_token_masks.append(mask_row_trimmed)

    return (
        _pack_object_rows(continuous_hidden_states),
        _pack_object_rows(continuous_latent_embeddings),
        _pack_object_rows(continuous_token_masks),
    )


def _validate_request_trace(trace: dict, request_id: str, response_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_row = np.asarray(trace[CONTINUOUS_HIDDEN_KEY], dtype=np.float16)
    latent_row = np.asarray(trace[CONTINUOUS_LATENT_KEY], dtype=np.float16)
    mask_row = np.asarray(trace[CONTINUOUS_MASK_KEY], dtype=np.bool_).reshape(-1)

    # IMPORTANT: Track original shapes before reshaping to detect corruption
    original_hidden_shape = hidden_row.shape
    original_latent_shape = latent_row.shape

    if hidden_row.ndim == 1 and hidden_row.size > 0:
        hidden_row = hidden_row.reshape(1, -1)
    if latent_row.ndim == 1 and latent_row.size > 0:
        latent_row = latent_row.reshape(1, -1)

    if mask_row.size > response_length:
        mask_row = mask_row[:response_length]
    active_positions = int(mask_row.sum())
    hidden_steps = int(hidden_row.shape[0]) if hidden_row.ndim == 2 else 0
    latent_steps = int(latent_row.shape[0]) if latent_row.ndim == 2 else 0

    # Detect shape corruption: reshape(1, -1) on 1D array with N elements creates (1, N)
    # but we need (N, hidden_dim) for proper indexing
    if hidden_row.ndim == 2 and hidden_row.shape[0] == 1 and active_positions > 1:
        logger.warning(
            "[ContinuousReplay] SHAPE WARNING for request_id=%s: "
            "mask has %d active positions but hidden_row shape is %s (original: %s). "
            "The reshape(1, -1) may have corrupted the data. Expected (%d, hidden_dim) got %s.",
            request_id,
            active_positions,
            hidden_row.shape,
            original_hidden_shape,
            active_positions,
            hidden_row.shape,
        )
    if latent_row.ndim == 2 and latent_row.shape[0] == 1 and active_positions > 1:
        logger.warning(
            "[ContinuousReplay] SHAPE WARNING for request_id=%s: "
            "mask has %d active positions but latent_row shape is %s (original: %s). "
            "The reshape(1, -1) may have corrupted the data. Expected (%d, latent_dim) got %s.",
            request_id,
            active_positions,
            latent_row.shape,
            original_latent_shape,
            active_positions,
            latent_row.shape,
        )

    if hidden_steps != active_positions or latent_steps != active_positions:
        raise RuntimeError(
            "Continuous replay trace shape mismatch for request_id="
            f"{request_id}: active_positions={active_positions} hidden_steps={hidden_steps} latent_steps={latent_steps} "
            f"hidden_shape={hidden_row.shape} latent_shape={latent_row.shape}"
        )

    return hidden_row, latent_row, mask_row


def _build_qwen3vl_input_embeds_compat(
    model,
    embed_layer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pixel_values: torch.Tensor | None,
    pixel_values_videos: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    prompt_positions_mask: torch.Tensor | None = None,
) -> dict:
    if embed_layer is None:
        raise RuntimeError("Continuous replay could not resolve Qwen3VL text embedding layer")

    inputs_embeds = embed_layer(input_ids)
    image_mask, video_mask = None, None
    deepstack_image_embeds = None
    deepstack_video_embeds = None

    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds, deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        _replay_debug(
            "[ContinuousReplay][MM] image insert input_ids=%s pixel_values=%s image_grid_thw=%s tokens=%s features=%s",
            tuple(input_ids.shape),
            tuple(pixel_values.shape),
            None if image_grid_thw is None else tuple(image_grid_thw.shape),
            int(n_image_tokens),
            int(n_image_features),
        )
        mask = _validate_multimodal_token_match(
            modality="image",
            input_ids=input_ids,
            token_id=model.config.image_token_id,
            feature_count=n_image_features,
            grid_thw=image_grid_thw,
            prompt_positions_mask=prompt_positions_mask,
        )
        image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds, deepstack_video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        _replay_debug(
            "[ContinuousReplay][MM] video insert input_ids=%s pixel_values_videos=%s video_grid_thw=%s tokens=%s features=%s",
            tuple(input_ids.shape),
            tuple(pixel_values_videos.shape),
            None if video_grid_thw is None else tuple(video_grid_thw.shape),
            int(n_video_tokens),
            int(n_video_features),
        )
        mask = _validate_multimodal_token_match(
            modality="video",
            input_ids=input_ids,
            token_id=model.config.video_token_id,
            feature_count=n_video_features,
            grid_thw=video_grid_thw,
            prompt_positions_mask=prompt_positions_mask,
        )
        video_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        dummy_pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        dummy_image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds, dummy_deepstack_image_embeds = model.visual(dummy_pixel_values, grid_thw=dummy_image_grid_thw)
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds = inputs_embeds + 0.0 * emb.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
    }


def _prepare_inputs_embeds(
    policy,
    micro_batch,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    multi_modal_inputs: dict | None,
) -> tuple[dict, dict]:
    global _REPLAY_CERT_LOGGED, _buffer_pool
    del attention_mask
    del multi_modal_inputs
    if not _has_continuous_replay_inputs(micro_batch):
        return {}, {}
    if getattr(policy, "latent_vae", None) is None:
        return {}, {}

    response_length = int(micro_batch["responses"].size(-1))
    response_start = input_ids.size(1) - response_length

    vae = policy.latent_vae.to(device=input_ids.device)
    policy.latent_vae = vae
    vae_dtype = next(vae.parameters()).dtype

    hidden_rows = _iter_object_array_rows(micro_batch.get(CONTINUOUS_HIDDEN_KEY))
    latent_rows = _iter_object_array_rows(micro_batch.get(CONTINUOUS_LATENT_KEY))
    mask_rows = _iter_object_array_rows(micro_batch.get(CONTINUOUS_MASK_KEY))

    _replay_debug(
        "[ContinuousReplay][Prepare] start batch=%s seq=%s response_len=%s hidden_size=%s",
        int(input_ids.size(0)),
        int(input_ids.size(1)),
        int(response_length),
        int(vae.hidden_size),
    )

    total_continuous_steps = 0
    batch_data = []
    for batch_idx, (hidden_row, latent_row, mask_row) in enumerate(
        itertools.zip_longest(hidden_rows, latent_rows, mask_rows, fillvalue=None)
    ):
        if mask_row is None:
            continue

        mask_np = np.asarray(mask_row, dtype=np.bool_).reshape(-1)
        if mask_np.size == 0:
            continue
        if mask_np.size > response_length:
            mask_np = mask_np[:response_length]
        true_positions = np.flatnonzero(mask_np)
        if true_positions.size == 0:
            continue

        if hidden_row is None or latent_row is None:
            continue

        hidden_np = np.array(hidden_row, dtype=np.float16, copy=True)
        if hidden_np.ndim == 1:
            hidden_np = hidden_np.reshape(1, -1)
        if hidden_np.size == 0:
            continue

        latent_np = np.array(latent_row, dtype=np.float16, copy=True)
        if latent_np.ndim == 1:
            latent_np = latent_np.reshape(1, -1)
        if latent_np.size == 0:
            continue

        n_steps = min(true_positions.size, hidden_np.shape[0], latent_np.shape[0])
        if n_steps == 0:
            continue

        _replay_debug(
            "[ContinuousReplay][Prepare] sample=%s mask_len=%s active=%s hidden_shape=%s latent_shape=%s n_steps=%s response_start=%s first_pos=%s last_pos=%s",
            batch_idx,
            int(mask_np.size),
            int(true_positions.size),
            tuple(hidden_np.shape),
            tuple(latent_np.shape),
            int(n_steps),
            int(response_start),
            int(true_positions[0]) if true_positions.size > 0 else -1,
            int(true_positions[n_steps - 1]) if n_steps > 0 else -1,
        )
        batch_data.append((batch_idx, hidden_np[:n_steps], latent_np[:n_steps], true_positions[:n_steps]))
        total_continuous_steps += n_steps

    if total_continuous_steps == 0:
        return {}, {}

    _buffer_pool.ensure_capacity(
        batch_size=input_ids.size(0),
        seq_length=input_ids.size(1),
        continuous_steps=total_continuous_steps,
        device=input_ids.device,
        hidden_size=vae.hidden_size,
        dtype=vae_dtype,
    )

    replay_row_ids = _buffer_pool.replay_row_ids[: input_ids.size(0), : input_ids.size(1)].fill_(-1)
    hidden_buffer = _buffer_pool.hidden_buffer[:total_continuous_steps]
    latent_buffer = _buffer_pool.latent_buffer[:total_continuous_steps]
    position_buffer = _buffer_pool.position_buffer[:total_continuous_steps]

    next_row_id = 0
    total_replaced_positions = 0
    for batch_idx, hidden_np, latent_np, true_positions in batch_data:
        n_steps = hidden_np.shape[0]
        start_idx = next_row_id
        end_idx = next_row_id + n_steps
        hidden_buffer[start_idx:end_idx] = torch.from_numpy(hidden_np).to(device=input_ids.device, dtype=vae_dtype)
        latent_buffer[start_idx:end_idx] = torch.from_numpy(latent_np).to(device=input_ids.device, dtype=vae_dtype)
        position_buffer[start_idx:end_idx] = torch.as_tensor(
            true_positions + response_start,
            device=input_ids.device,
            dtype=torch.long,
        )
        assign_positions = position_buffer[start_idx:end_idx]
        replay_row_ids[batch_idx, assign_positions] = torch.arange(
            start_idx,
            end_idx,
            device=input_ids.device,
            dtype=torch.long,
        )
        _replay_debug(
            "[ContinuousReplay][Prepare] assign sample=%s row_id_range=[%s,%s) seq_positions=[%s,%s]",
            batch_idx,
            int(start_idx),
            int(end_idx),
            int(assign_positions[0].item()) if n_steps > 0 else -1,
            int(assign_positions[-1].item()) if n_steps > 0 else -1,
        )

        next_row_id += n_steps
        total_replaced_positions += int(n_steps)

    if total_replaced_positions == 0:
        return {}, {}

    if not _REPLAY_CERT_LOGGED:
        logger.warning(
            "[ContinuousReplay] Actor replay active: batch=%s full_seq_len=%s response_len=%s continuous_positions=%s discrete_positions_preserved=true source=saved_rollout_latents",
            int(input_ids.size(0)),
            int(input_ids.size(1)),
            int(response_length),
            total_replaced_positions,
        )
        _REPLAY_CERT_LOGGED = True

    replay_state = {
        "continuous_replay_row_ids": replay_row_ids.clone(),
        "continuous_replay_hidden_states": hidden_buffer[:next_row_id].clone(),
        "continuous_replay_latent_embeddings": latent_buffer[:next_row_id].clone(),
        "continuous_replay_prompt_positions_mask": _build_prompt_positions_mask(
            input_ids=input_ids,
            response_length=response_length,
        ).clone(),
        "continuous_replay_latent_vae": vae,
    }
    if _GRAD_FLOW_DEBUG:
        replay_mask = replay_state["continuous_replay_row_ids"] >= 0
        selected_row_ids = replay_state["continuous_replay_row_ids"][replay_mask]
        _replay_debug(
            "[ContinuousReplay][Prepare] done total_steps=%s replaced=%s replay_mask=%s row_id_min=%s row_id_max=%s latent_shape=%s",
            int(next_row_id),
            int(total_replaced_positions),
            int(replay_mask.sum().item()),
            int(selected_row_ids.min().item()) if selected_row_ids.numel() > 0 else -1,
            int(selected_row_ids.max().item()) if selected_row_ids.numel() > 0 else -1,
            tuple(replay_state["continuous_replay_latent_embeddings"].shape),
        )
    return {
        "continuous_replay_row_ids": replay_state["continuous_replay_row_ids"],
        "continuous_replay_latent_embeddings": replay_state["continuous_replay_latent_embeddings"],
        "continuous_replay_prompt_positions_mask": replay_state["continuous_replay_prompt_positions_mask"],
    }, replay_state


def _merge_continuous_policy_stats(
    log_probs: torch.Tensor,
    entropy: torch.Tensor | None,
    replay_state: dict,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not replay_state:
        return log_probs, entropy

    row_ids = replay_state.get("continuous_replay_row_ids")
    hidden_states = replay_state.get("continuous_replay_hidden_states")
    latent_embeddings = replay_state.get("continuous_replay_latent_embeddings")
    vae = replay_state.get("continuous_replay_latent_vae")
    if row_ids is None or hidden_states is None or latent_embeddings is None or vae is None:
        return log_probs, entropy

    response_row_ids = row_ids[:, -log_probs.size(1) :].to(device=log_probs.device)
    replay_mask = response_row_ids >= 0
    if not torch.any(replay_mask):
        return log_probs, entropy

    vae = vae.to(device=log_probs.device)
    hidden_dtype = next(vae.parameters()).dtype
    hidden_states = hidden_states.to(device=log_probs.device, dtype=hidden_dtype)
    latent_embeddings = latent_embeddings.to(device=log_probs.device, dtype=hidden_dtype)
    latent_dist = vae.forward(hidden_states, temperature=1.0)
    latent_log_probs = latent_dist.log_prob(latent_embeddings).mean(dim=-1).to(dtype=log_probs.dtype)

    merged_log_probs = log_probs.clone()
    selected_row_ids = response_row_ids[replay_mask].to(dtype=torch.long)
    merged_log_probs[replay_mask] = latent_log_probs.index_select(0, selected_row_ids)

    merged_entropy = entropy
    if entropy is not None:
        latent_entropy = latent_dist.entropy().mean(dim=-1).to(dtype=entropy.dtype)
        merged_entropy = entropy.clone()
        merged_entropy[replay_mask] = latent_entropy.index_select(0, selected_row_ids)

    return merged_log_probs, merged_entropy


def _compact_replay_model_kwargs_for_rmpad(replay_model_kwargs: dict, indices, dp_actor_mod) -> dict:
    compacted = {}
    prompt_positions_mask = replay_model_kwargs.get("continuous_replay_prompt_positions_mask")
    if prompt_positions_mask is not None:
        prompt_positions_mask_rmpad = dp_actor_mod.index_first_axis(
            dp_actor_mod.rearrange(prompt_positions_mask.unsqueeze(-1), "b s ... -> (b s) ..."),
            indices,
        ).transpose(0, 1)
        compacted["continuous_replay_prompt_positions_mask"] = prompt_positions_mask_rmpad.squeeze(-1)

    replay_row_ids = replay_model_kwargs.get("continuous_replay_row_ids")
    if replay_row_ids is not None:
        replay_row_ids_rmpad = dp_actor_mod.index_first_axis(
            dp_actor_mod.rearrange(replay_row_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
            indices,
        ).transpose(0, 1)
        compacted["continuous_replay_row_ids"] = replay_row_ids_rmpad.squeeze(-1)

    latent_embeddings = replay_model_kwargs.get("continuous_replay_latent_embeddings")
    if latent_embeddings is not None:
        compacted["continuous_replay_latent_embeddings"] = latent_embeddings

    return compacted


def _patch_verl_qwen3vl_inputs_embeds_support() -> None:
    qwen3_vl_mod = importlib.import_module("verl.models.transformers.qwen3_vl")
    original_get_input_embeds = qwen3_vl_mod._get_input_embeds
    original_base_forward = qwen3_vl_mod.qwen3_vl_base_forward

    if getattr(original_base_forward, "_qwen3vl_continuous_inputs_embeds_patch", False):
        return

    def compat_get_input_embeds(
        model,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        prompt_positions_mask: torch.Tensor | None = None,
    ):
        embed_layer = _resolve_input_embedding_layer(model)
        if embed_layer is None:
            return original_get_input_embeds(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        return _build_qwen3vl_input_embeds_compat(
            model,
            embed_layer=embed_layer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            prompt_positions_mask=prompt_positions_mask,
        )

    def _apply_continuous_replay_to_inputs_embeds(
        inputs_embeds: torch.Tensor,
        continuous_replay_row_ids: torch.Tensor | None,
        continuous_replay_latent_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        if continuous_replay_row_ids is None or continuous_replay_latent_embeddings is None:
            return inputs_embeds

        row_ids = continuous_replay_row_ids.to(device=inputs_embeds.device)
        if tuple(row_ids.shape) != tuple(inputs_embeds.shape[:-1]):
            raise RuntimeError(
                "[ContinuousReplay] inputs_embeds / row_ids shape mismatch: "
                f"inputs_embeds.shape={tuple(inputs_embeds.shape)} "
                f"row_ids.shape={tuple(row_ids.shape)} "
                f"expected_row_id_shape={tuple(inputs_embeds.shape[:-1])}"
            )
        replay_mask = row_ids >= 0
        if not torch.any(replay_mask):
            return inputs_embeds

        latent_embeddings = continuous_replay_latent_embeddings.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        selected_row_ids = row_ids[replay_mask].to(dtype=torch.long)
        _replay_debug(
            "[ContinuousReplay][Inject] inputs=%s row_ids=%s mask_positions=%s latent_embeddings=%s selected_row_id_min=%s selected_row_id_max=%s",
            tuple(inputs_embeds.shape),
            tuple(row_ids.shape),
            int(replay_mask.sum().item()),
            tuple(latent_embeddings.shape),
            int(selected_row_ids.min().item()) if selected_row_ids.numel() > 0 else -1,
            int(selected_row_ids.max().item()) if selected_row_ids.numel() > 0 else -1,
        )

        outputs = inputs_embeds.clone()
        if replay_mask.sum().item() > 0:
            selected_latents = latent_embeddings.index_select(0, selected_row_ids)
            if int(replay_mask.sum().item()) != int(selected_latents.shape[0]):
                raise RuntimeError(
                    "[ContinuousReplay] replay assignment cardinality mismatch: "
                    f"mask_positions={int(replay_mask.sum().item())} selected_latents={selected_latents.shape[0]} "
                    f"inputs_embeds.shape={tuple(inputs_embeds.shape)} row_ids.shape={tuple(row_ids.shape)}"
                )
            outputs[replay_mask] = selected_latents
        return outputs

    @functools.wraps(original_base_forward)
    def compat_qwen3_vl_base_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ):
        continuous_replay_row_ids = kwargs.pop("continuous_replay_row_ids", None)
        continuous_replay_latent_embeddings = kwargs.pop("continuous_replay_latent_embeddings", None)
        continuous_replay_prompt_positions_mask = kwargs.pop("continuous_replay_prompt_positions_mask", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)

        if inputs_embeds is None:
            input_kwargs = qwen3_vl_mod._get_input_embeds(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                prompt_positions_mask=continuous_replay_prompt_positions_mask,
            )
        else:
            input_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask.to(inputs_embeds.device) if attention_mask is not None else None,
            }

        input_kwargs["inputs_embeds"] = _apply_continuous_replay_to_inputs_embeds(
            input_kwargs["inputs_embeds"],
            continuous_replay_row_ids=continuous_replay_row_ids,
            continuous_replay_latent_embeddings=continuous_replay_latent_embeddings,
        )
        input_kwargs = _cast_qwen3vl_inputs_for_compute_dtype(self, input_kwargs)
        kwargs.update(input_kwargs)
        return self.language_model(input_ids=None, **kwargs)

    compat_qwen3_vl_base_forward._qwen3vl_continuous_inputs_embeds_patch = True
    compat_get_input_embeds._qwen3vl_continuous_inputs_embeds_patch = True
    qwen3_vl_mod._get_input_embeds = compat_get_input_embeds
    qwen3_vl_mod.qwen3_vl_base_forward = compat_qwen3_vl_base_forward

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModel

    Qwen3VLModel.forward = compat_qwen3_vl_base_forward
    Qwen3VLMoeModel.forward = compat_qwen3_vl_base_forward

def apply_continuous_replay_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from verl.workers.actor import dp_actor as dp_actor_mod
    import ray
    from tensordict import TensorDict
    from verl import DataProto
    from verl.experimental.agent_loop import agent_loop as agent_loop_mod
    from verl.trainer.ppo import ray_trainer as ray_trainer_mod
    from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
    from verl.workers.rollout.vllm_rollout import vllm_rollout as rollout_mod
    from verl.workers.rollout.vllm_rollout import vllm_async_server as async_server_mod

    _patch_verl_qwen3vl_inputs_embeds_support()

    original_policy_init = dp_actor_mod.DataParallelPPOActor.__init__
    original_forward_micro_batch = dp_actor_mod.DataParallelPPOActor._forward_micro_batch
    original_agent_loop_postprocess = agent_loop_mod.AgentLoopWorker._postprocess

    def compat_policy_init(self, config, actor_module, actor_optimizer=None):
        original_policy_init(self, config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        _load_policy_latent_vae(self)
        _load_opsd_external_teacher(self)

    def compat_forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        has_continuous_replay_inputs = _has_continuous_replay_inputs(micro_batch)
        has_qwen3vl_multimodal_inputs = bool(multi_modal_inputs) and _resolve_qwen3vl_model(self.actor_module) is not None
        if not has_continuous_replay_inputs and not has_qwen3vl_multimodal_inputs:
            return original_forward_micro_batch(self, micro_batch, temperature, calculate_entropy)

        if self.use_ulysses_sp:
            raise NotImplementedError("Continuous replay does not support ulysses sequence parallelism yet")

        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            sum_pi_squared = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            replay_model_kwargs = {}
            replay_state = {}
            if has_continuous_replay_inputs:
                replay_model_kwargs, replay_state = _prepare_inputs_embeds(
                    self,
                    micro_batch,
                    input_ids,
                    attention_mask=attention_mask,
                    multi_modal_inputs=multi_modal_inputs,
                )

            if has_qwen3vl_multimodal_inputs:
                replay_model_kwargs.setdefault(
                    "continuous_replay_prompt_positions_mask",
                    _build_prompt_positions_mask(input_ids=input_ids, response_length=response_length),
                )

            extra_args = {}
            if self.use_fused_kernels:
                extra_args["temperature"] = temperature
                extra_args["return_dict"] = True

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = dp_actor_mod.unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                replay_model_kwargs_rmpad = _compact_replay_model_kwargs_for_rmpad(
                    replay_model_kwargs,
                    indices,
                    dp_actor_mod,
                )

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        dp_actor_mod.index_first_axis(
                            dp_actor_mod.rearrange(position_ids, "c b s ... -> (b s) c ..."),
                            indices,
                        )
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = dp_actor_mod.index_first_axis(
                        dp_actor_mod.rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    **replay_model_kwargs_rmpad,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)
                    inplace_backward = not calculate_entropy
                    log_probs = dp_actor_mod.logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        else:
                            entropy_rmpad = _checkpoint_if_needed(
                                self.compute_entropy_from_logits,
                                logits_rmpad,
                            )
                    if calculate_sum_pi_squared:
                        if not sum_pi_squared_checkpointing:
                            sum_pi_squared_rmpad = self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                        else:
                            sum_pi_squared_rmpad = _checkpoint_if_needed(
                                self.calculate_sum_pi_squared_from_logits,
                                logits_rmpad,
                            )

                if calculate_entropy:
                    full_entropy = dp_actor_mod.pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                full_log_probs = dp_actor_mod.pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = dp_actor_mod.pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                if calculate_sum_pi_squared:
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    **replay_model_kwargs,
                    use_cache=False,
                    **extra_args,
                )
                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1] if calculate_entropy else None
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = dp_actor_mod.logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = dp_actor_mod.verl_F.entropy_from_logits(logits)
                        else:
                            entropy = _checkpoint_if_needed(dp_actor_mod.verl_F.entropy_from_logits, logits)
                    if calculate_sum_pi_squared:
                        if not sum_pi_squared_checkpointing:
                            sum_pi_squared = self.calculate_sum_pi_squared_from_logits(logits)
                        else:
                            sum_pi_squared = _checkpoint_if_needed(
                                self.calculate_sum_pi_squared_from_logits,
                                logits,
                            )
            if has_continuous_replay_inputs:
                log_probs, entropy = _merge_continuous_policy_stats(log_probs, entropy, replay_state)

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

    def _continuous_non_tensor_keys(data) -> list[str]:
        keys = []
        if "multi_modal_inputs" in data.non_tensor_batch:
            keys.append("multi_modal_inputs")
        if OPSD_TEACHER_PROMPT_IDS_KEY in data.non_tensor_batch:
            keys.append(OPSD_TEACHER_PROMPT_IDS_KEY)
        if CONTINUOUS_HIDDEN_KEY in data.non_tensor_batch and CONTINUOUS_MASK_KEY in data.non_tensor_batch:
            keys.extend([CONTINUOUS_HIDDEN_KEY, CONTINUOUS_MASK_KEY])
        if CONTINUOUS_LATENT_KEY in data.non_tensor_batch:
            keys.append(CONTINUOUS_LATENT_KEY)
        return keys

    def _opsd_cfg_value(config, key: str, default):
        env_names = {
            "enabled": "OPSD_ENABLED",
            "weight": "OPSD_WEIGHT",
            "temperature": "OPSD_TEMPERATURE",
            "sampled_logprob_chunk_size": "OPSD_LOGPROB_CHUNK_SIZE",
        }
        env_name = env_names.get(key)
        if env_name:
            raw_value = os.environ.get(env_name)
            if raw_value is not None:
                if isinstance(default, bool):
                    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
                if isinstance(default, int):
                    return int(raw_value)
                if isinstance(default, float):
                    return float(raw_value)
                return raw_value

        cfg = getattr(config, "opsd", None)
        if cfg is not None:
            return cfg.get(key, default)
        engine_cfg = getattr(config, "engine", None)
        if engine_cfg is not None:
            opsd_cfg = engine_cfg.get("opsd", None) if hasattr(engine_cfg, "get") else None
            if opsd_cfg is not None:
                return opsd_cfg.get(key, default)
        return default

    def _opsd_enabled(config) -> bool:
        return bool(_opsd_cfg_value(config, "enabled", os.environ.get("OPSD_ENABLED", "0") == "1"))

    def _pad_prompt_response_inputs(
        *,
        prompt_rows,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = responses.device
        batch_size, response_len = responses.shape
        prompt_lists: list[list[int]] = []
        for row in _iter_object_array_rows(prompt_rows):
            if hasattr(row, "tolist"):
                row = row.tolist()
            prompt_lists.append([int(token_id) for token_id in (row or [])])
        if len(prompt_lists) != batch_size:
            raise RuntimeError(
                f"OPSD teacher prompt batch mismatch: prompts={len(prompt_lists)} responses={batch_size}"
            )
        max_prompt_len = max((len(row) for row in prompt_lists), default=0)
        if max_prompt_len <= 0:
            raise RuntimeError("OPSD teacher replay requires non-empty teacher prompt token IDs.")

        total_len = max_prompt_len + response_len
        input_ids = torch.full((batch_size, total_len), int(pad_token_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, total_len), dtype=response_mask.dtype, device=device)
        target_mask = torch.zeros((batch_size, response_len), dtype=response_mask.dtype, device=device)

        for row_idx, prompt_ids in enumerate(prompt_lists):
            prompt_len = len(prompt_ids)
            prompt_start = max_prompt_len - prompt_len
            input_ids[row_idx, prompt_start:max_prompt_len] = torch.tensor(
                prompt_ids,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row_idx, prompt_start:max_prompt_len] = 1
            input_ids[row_idx, max_prompt_len:] = responses[row_idx]
            target_mask[row_idx] = response_mask[row_idx]
            attention_mask[row_idx, max_prompt_len:] = response_mask[row_idx]

        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
        return input_ids, attention_mask, position_ids, target_mask

    def _sampled_log_probs_from_logits(
        *,
        logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        temperature: float,
        chunk_size: int,
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
        if abs(float(temperature) - 1.0) < 1e-6:
            sampled_logits = torch.gather(logits, dim=-1, index=gather_index).squeeze(-1)
            log_norm = torch.logsumexp(logits, dim=-1)
            return sampled_logits - log_norm

        scaled_logits = logits / float(temperature)
        sampled_logits = torch.gather(scaled_logits, dim=-1, index=gather_index).squeeze(-1)
        log_norm = torch.logsumexp(scaled_logits, dim=-1)
        return sampled_logits - log_norm

    def _compute_opsd_teacher_student_loss(
        self,
        *,
        model_inputs: dict,
        student_log_prob: torch.Tensor,
        temperature: float,
        pad_token_id: int,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        global _OPSD_LOSS_LOGGED

        prompt_rows = model_inputs.get(OPSD_TEACHER_PROMPT_IDS_KEY)
        if prompt_rows is None:
            return None, {}

        responses = model_inputs["responses"]
        response_mask = model_inputs["response_mask"]
        teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_target_mask = _pad_prompt_response_inputs(
            prompt_rows=prompt_rows,
            responses=responses,
            response_mask=response_mask,
            pad_token_id=pad_token_id,
        )

        teacher_module = getattr(self, "opsd_external_teacher", None) if _opsd_off_policy_enabled() else None
        if teacher_module is None:
            teacher_module = self.actor_module

        teacher_device = responses.device
        restore_device = None
        if teacher_module is not self.actor_module:
            original_device = _module_compute_device(teacher_module)
            if original_device != teacher_device:
                teacher_module = teacher_module.to(teacher_device)
                if _opsd_offload_teacher_enabled() or original_device.type == "cpu":
                    restore_device = original_device

        teacher_input_ids = teacher_input_ids.to(device=teacher_device)
        teacher_attention_mask = teacher_attention_mask.to(device=teacher_device)
        teacher_position_ids = teacher_position_ids.to(device=teacher_device)

        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            was_training = bool(getattr(teacher_module, "training", False))
            teacher_module.eval()
            try:
                output = teacher_module(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    position_ids=teacher_position_ids,
                    use_cache=False,
                )
            finally:
                if was_training:
                    teacher_module.train()
                if restore_device is not None:
                    teacher_module.to(restore_device)

        teacher_logits = output.logits[:, -responses.size(1) - 1 : -1, :]
        teacher_log_prob = _sampled_log_probs_from_logits(
            logits=teacher_logits,
            sampled_token_ids=responses,
            temperature=float(_opsd_cfg_value(self.config, "temperature", temperature)),
            chunk_size=int(_opsd_cfg_value(self.config, "sampled_logprob_chunk_size", 1024)),
        )

        mask = teacher_target_mask.to(dtype=torch.bool)
        if not mask.any():
            return None, {}

        advantage = (teacher_log_prob - student_log_prob).detach()
        loss = -(advantage[mask] * student_log_prob[mask]).mean()

        metrics = {
            "actor/opsd_loss": float(loss.detach().item()),
            "actor/opsd_advantage": float(advantage[mask].mean().detach().item()),
            "actor/opsd_student_logprob": float(student_log_prob[mask].mean().detach().item()),
            "actor/opsd_teacher_logprob": float(teacher_log_prob[mask].mean().detach().item()),
            "actor/opsd_tokens": float(mask.sum().detach().item()),
        }
        if not _OPSD_LOSS_LOGGED:
            logger.warning(
                "[ContinuousReplay][OPSD] teacher replay active: batch=%s response_len=%s prompt_max_len=%s weight=%s",
                int(responses.size(0)),
                int(responses.size(1)),
                int(teacher_input_ids.size(1) - responses.size(1)),
                float(_opsd_cfg_value(self.config, "weight", 1.0)),
            )
            _OPSD_LOSS_LOGGED = True
        return loss, metrics

    def compat_compute_log_prob(self, data, calculate_entropy=False):
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        self.actor_module.eval()
        if getattr(self, "latent_vae", None) is not None:
            self.latent_vae.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=_continuous_non_tensor_keys(data))

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = dp_actor_mod.prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(dp_actor_mod.get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if calculate_entropy else None
        sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0) if calculate_sum_pi_squared else None
        if use_dynamic_bsz:
            log_probs = dp_actor_mod.restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = dp_actor_mod.restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = dp_actor_mod.restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    def compat_update_policy(self, data):
        self.actor_module.train()
        if getattr(self, "latent_vae", None) is not None:
            self.latent_vae.train()

        temperature = data.meta_info["temperature"]
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=_continuous_non_tensor_keys(data))
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = dp_actor_mod.prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(dp_actor_mod.get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    pad_token_id = data.meta_info.get("pad_token_id", 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    calculate_entropy = entropy_coeff != 0
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    policy_loss_fn = dp_actor_mod.get_policy_loss_fn(loss_mode)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = dp_actor_mod.agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if _opsd_enabled(self.config):
                        opsd_loss, opsd_metrics = _compute_opsd_teacher_student_loss(
                            self,
                            model_inputs=model_inputs,
                            student_log_prob=log_prob,
                            temperature=temperature,
                            pad_token_id=pad_token_id,
                        )
                        if opsd_loss is not None:
                            opsd_weight = float(_opsd_cfg_value(self.config, "weight", 1.0))
                            policy_loss = policy_loss + opsd_loss * opsd_weight
                            for metric_key, metric_value in opsd_metrics.items():
                                micro_batch_metrics[metric_key] = metric_value * loss_scale_factor

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = dp_actor_mod.kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = dp_actor_mod.agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    micro_batch_metrics.update(_collect_grad_flow_probe_metrics(self, model_inputs))
                    if _GRAD_FLOW_DEBUG:
                        _log_grad_flow_probe(self, model_inputs)

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    dp_actor_mod.append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                dp_actor_mod.append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        self.actor_optimizer.zero_grad()
        return metrics

    async def compat_server_generate(self, *args, **kwargs):
        global _TRACE_ATTACH_LOGGED, _VLLM_LOGPROB_FALLBACK_LOGGED

        prompt_ids = kwargs.pop("prompt_ids", args[0] if len(args) >= 1 else None)
        sampling_params = kwargs.pop("sampling_params", args[1] if len(args) >= 2 else None)
        request_id = kwargs.pop("request_id", args[2] if len(args) >= 3 else None)
        image_data = kwargs.pop("image_data", args[3] if len(args) >= 4 else None)
        video_data = kwargs.pop("video_data", args[4] if len(args) >= 5 else None)
        priority = kwargs.pop("priority", args[5] if len(args) >= 6 else 0)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected vLLM generate kwargs: {unexpected}")
        if prompt_ids is None or sampling_params is None or request_id is None:
            raise TypeError("prompt_ids, sampling_params, and request_id are required for vLLM generate.")

        prompt_ids = async_server_mod.normalize_token_ids(prompt_ids)
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        sampling_params = dict(sampling_params)
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            max_tokens = min(
                self.config.response_length,
                self.config.prompt_length + self.config.response_length - len(prompt_ids),
            )
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = async_server_mod.SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = async_server_mod.qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)

        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        prompt = async_server_mod.TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        lora_request = None
        if self.lora_as_adapter:
            lora_loaded = async_server_mod.VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = async_server_mod.LoRARequest(
                    lora_name=async_server_mod.VLLM_LORA_NAME,
                    lora_int_id=async_server_mod.VLLM_LORA_INT_ID,
                    lora_path=async_server_mod.VLLM_LORA_PATH,
                )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        final_res = None
        async for engine_output in generator:
            final_res = engine_output
        assert final_res is not None

        vllm_output = final_res.outputs[0]
        token_ids = vllm_output.token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = []
            for token_id, logprob_map in zip(token_ids, vllm_output.logprobs or []):
                selected = logprob_map.get(token_id) if logprob_map is not None else None
                if selected is None and logprob_map:
                    selected = next(iter(logprob_map.values()))
                    if not _VLLM_LOGPROB_FALLBACK_LOGGED:
                        logger.warning(
                            "[ContinuousReplay] vLLM omitted selected token %s from logprob map; using first returned logprob.",
                            int(token_id),
                        )
                        _VLLM_LOGPROB_FALLBACK_LOGGED = True
                log_probs.append(float(getattr(selected, "logprob", 0.0)))

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = vllm_output.routed_experts

        finish_reason = vllm_output.finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason

        num_preempted = getattr(vllm_output, "num_preempted", None)
        output = async_server_mod.TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields={"global_steps": self.global_steps},
        )
        if request_id is not None:
            trace = pop_request_trace(request_id)
            if trace is None:
                trace = get_request_trace(request_id)
            if trace is not None:
                if output.log_probs is not None:
                    latent_log_probs = np.asarray(
                        trace.get(CONTINUOUS_LATENT_LOGPROB_KEY, np.empty((0,), dtype=np.float32)),
                        dtype=np.float32,
                    ).reshape(-1)
                    if latent_log_probs.size > 0:
                        hidden_row, latent_row, mask_row = _validate_request_trace(
                            trace,
                            request_id=str(request_id),
                            response_length=len(token_ids),
                        )
                        _, _, mask_row_trimmed, latent_log_probs_trimmed = _trim_request_trace_to_actual_response_length(
                            hidden_row=hidden_row,
                            latent_row=latent_row,
                            mask_row=mask_row,
                            latent_log_probs=latent_log_probs,
                            actual_response_length=len(token_ids),
                            request_id=str(request_id),
                        )
                        _inject_latent_log_probs_into_rollout(
                            curr_log_prob=output.log_probs,
                            mask_row=mask_row_trimmed,
                            latent_log_probs=latent_log_probs_trimmed,
                            request_id=str(request_id),
                        )
                output.extra_fields = dict(output.extra_fields)
                output.extra_fields["continuous_trace"] = trace
                if not _TRACE_ATTACH_LOGGED:
                    mask = np.asarray(
                        trace.get(CONTINUOUS_MASK_KEY, np.empty((0,), dtype=np.bool_)),
                        dtype=np.bool_,
                    ).reshape(-1)
                    hidden = np.asarray(
                        trace.get(CONTINUOUS_HIDDEN_KEY, np.empty((0, 0), dtype=np.float16)),
                        dtype=np.float16,
                    )
                    latent = np.asarray(
                        trace.get(CONTINUOUS_LATENT_KEY, np.empty((0, 0), dtype=np.float16)),
                        dtype=np.float16,
                    )
                    logger.warning(
                        "[ContinuousReplay] Trace attached: request_id=%s response_tokens=%s mask_len=%s active_positions=%s hidden_shape=%s latent_shape=%s",
                        request_id,
                        len(token_ids),
                        int(mask.size),
                        int(mask.sum()),
                        tuple(hidden.shape),
                        tuple(latent.shape),
                    )
                    _TRACE_ATTACH_LOGGED = True

        return output

    def compat_rollout_generate_sequences(self, prompts, **kwargs):
        global _ROLLOUT_CERT_LOGGED

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        input_ids_cpu = idx.detach().to(device="cpu")
        attention_mask_cpu = attention_mask.detach().to(device="cpu", dtype=torch.bool)
        raw_prompt_ids = [input_ids_cpu[i][attention_mask_cpu[i]].tolist() for i in range(batch_size)]

        multi_modal_rows = non_tensor_batch.get("multi_modal_data")
        if isinstance(multi_modal_rows, np.ndarray):
            multi_modal_rows = multi_modal_rows.tolist()
        elif multi_modal_rows is None:
            multi_modal_rows = [None] * batch_size

        if batch_size != len(raw_prompt_ids):
            raise RuntimeError("vLLM rollout prompt batch assembly is inconsistent.")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        if not do_sample:
            sampling_kwargs = {"best_of": 1, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "temperature": 0, "n": 1}
        elif is_validate:
            sampling_kwargs = {
                "top_k": max(0, self.config.val_kwargs.top_k),
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        else:
            sampling_kwargs = {
                "top_k": max(0, getattr(self.config, "top_k", -1)),
                "top_p": top_p,
                "temperature": temperature,
                "n": 1,
            }
            sampling_kwargs.update(kwargs)

        # Override logprobs parameter when calculate_log_probs is enabled
        if self.config.calculate_log_probs:
            sampling_kwargs["logprobs"] = 1  # Return logprobs for generated tokens

        continuous_hidden_states = []
        continuous_latent_embeddings = []
        continuous_token_masks = []
        if self.server_handle is None:
            self.server_handle = ray.get_actor(f"vllm_server_{self.replica_rank}_{self.node_rank}")

        request_prefix = f"continuous-replay-{os.getpid()}-{time.time_ns()}"
        outputs = ray.get(
            [
                self.server_handle.generate.remote(
                    prompt_ids=prompt_token_ids,
                    sampling_params=dict(sampling_kwargs),
                    request_id=f"{request_prefix}-{row_idx}",
                    image_data=(multi_modal_rows[row_idx] or {}).get("image") if multi_modal_rows[row_idx] else None,
                    video_data=(multi_modal_rows[row_idx] or {}).get("video") if multi_modal_rows[row_idx] else None,
                )
                for row_idx, prompt_token_ids in enumerate(raw_prompt_ids)
            ]
        )

        response = []
        rollout_log_probs = []
        for output in outputs:
            trace = output.extra_fields.get("continuous_trace")
            if trace is None:
                raise RuntimeError(
                    "Continuous replay trace missing for vLLM request_id="
                    f"{getattr(output, 'request_id', '<unknown>')}. "
                    "This indicates trace capture did not survive async rollout."
                )

            hidden_row, latent_row, mask_row = _validate_request_trace(
                trace,
                request_id=str(getattr(output, "request_id", "<unknown>")),
                response_length=self.config.response_length,
            )
            response_ids = output.token_ids
            response.append(response_ids)
            actual_response_length = len(response_ids)
            latent_log_probs = None
            if self.config.calculate_log_probs:
                latent_log_probs = np.asarray(
                    trace.get(CONTINUOUS_LATENT_LOGPROB_KEY, np.empty((0,), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)

            hidden_row_trimmed, latent_row_trimmed, mask_row_trimmed, latent_log_probs = (
                _trim_request_trace_to_actual_response_length(
                    hidden_row=hidden_row,
                    latent_row=latent_row,
                    mask_row=mask_row,
                    latent_log_probs=latent_log_probs,
                    actual_response_length=actual_response_length,
                    request_id=str(getattr(output, "request_id", "<unknown>")),
                )
            )

            continuous_hidden_states.append(hidden_row_trimmed)
            continuous_latent_embeddings.append(latent_row_trimmed)
            continuous_token_masks.append(mask_row_trimmed)

            if self.config.calculate_log_probs:
                curr_log_prob = list(output.log_probs or [])
                _inject_latent_log_probs_into_rollout(
                    curr_log_prob=curr_log_prob,
                    mask_row=mask_row_trimmed,
                    latent_log_probs=latent_log_probs,
                    request_id=str(getattr(output, "request_id", "<unknown>")),
                )
                rollout_log_probs.append(curr_log_prob)

        response = pad_2d_list_to_length(response, pad_token_id, max_length=self.config.response_length).to(idx.device)
        if self.config.calculate_log_probs:
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length)
            rollout_log_probs = rollout_log_probs.to(idx.device, dtype=torch.float32)
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = rollout_log_probs

        non_tensor_batch[CONTINUOUS_HIDDEN_KEY] = _pack_object_rows(continuous_hidden_states)
        non_tensor_batch[CONTINUOUS_LATENT_KEY] = _pack_object_rows(continuous_latent_embeddings)
        non_tensor_batch[CONTINUOUS_MASK_KEY] = _pack_object_rows(continuous_token_masks)
        if not _ROLLOUT_CERT_LOGGED:
            traced_samples = sum(int(getattr(item, "shape", (0,))[0] > 0) for item in continuous_hidden_states)
            active_positions = sum(int(np.asarray(mask, dtype=np.bool_).sum()) for mask in continuous_token_masks)
            saved_latents = sum(int(getattr(item, "shape", (0,))[0] > 0) for item in continuous_latent_embeddings)
            logger.warning(
                "[ContinuousReplay] Rollout trace active: samples=%s traced=%s response_len=%s active_positions=%s saved_latents=%s",
                len(continuous_hidden_states),
                traced_samples,
                int(response.size(1)),
                active_positions,
                saved_latents,
            )
            _ROLLOUT_CERT_LOGGED = True
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def compat_agent_loop_postprocess(self, outputs, input_non_tensor_batch=None):
        global _ROLLOUT_CERT_LOGGED

        batch_output = original_agent_loop_postprocess(self, outputs, input_non_tensor_batch=input_non_tensor_batch)
        packed_traces = _unpack_agent_loop_continuous_traces(outputs)
        if packed_traces is None:
            return batch_output

        batch_output.non_tensor_batch.pop("continuous_trace", None)
        hidden_rows, latent_rows, mask_rows = packed_traces
        batch_output.non_tensor_batch[CONTINUOUS_HIDDEN_KEY] = hidden_rows
        batch_output.non_tensor_batch[CONTINUOUS_LATENT_KEY] = latent_rows
        batch_output.non_tensor_batch[CONTINUOUS_MASK_KEY] = mask_rows

        if not _ROLLOUT_CERT_LOGGED:
            traced_samples = sum(int(getattr(item, "shape", (0,))[0] > 0) for item in hidden_rows.tolist())
            active_positions = sum(int(np.asarray(mask, dtype=np.bool_).sum()) for mask in mask_rows.tolist())
            saved_latents = sum(int(getattr(item, "shape", (0,))[0] > 0) for item in latent_rows.tolist())
            logger.warning(
                "[ContinuousReplay] Rollout trace active: samples=%s traced=%s response_len=%s active_positions=%s saved_latents=%s",
                len(hidden_rows),
                traced_samples,
                int(batch_output.batch["responses"].size(1)),
                active_positions,
                saved_latents,
            )
            _ROLLOUT_CERT_LOGGED = True

        return batch_output

    original_get_gen_batch = ray_trainer_mod.RayPPOTrainer._get_gen_batch

    def compat_get_gen_batch(self, batch):
        if OPSD_TEACHER_PROMPT_IDS_KEY not in batch.non_tensor_batch:
            return original_get_gen_batch(self, batch)

        reward_keys = set({"data_source", "reward_model", "extra_info", "uid", OPSD_TEACHER_PROMPT_IDS_KEY}) & set(
            batch.non_tensor_batch.keys()
        )
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(batch_keys=[], non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop))
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

    dp_actor_mod.DataParallelPPOActor.__init__ = compat_policy_init
    dp_actor_mod.DataParallelPPOActor._forward_micro_batch = compat_forward_micro_batch
    dp_actor_mod.DataParallelPPOActor.compute_log_prob = compat_compute_log_prob
    dp_actor_mod.DataParallelPPOActor.update_policy = compat_update_policy
    async_server_mod.vLLMHttpServer.generate = compat_server_generate
    rollout_mod.ServerAdapter.generate_sequences = compat_rollout_generate_sequences
    agent_loop_mod.AgentLoopWorker._postprocess = compat_agent_loop_postprocess
    ray_trainer_mod.RayPPOTrainer._get_gen_batch = compat_get_gen_batch
    _PATCHED = True

__all__ = [
    "CONTINUOUS_HIDDEN_KEY",
    "CONTINUOUS_LATENT_KEY",
    "CONTINUOUS_LATENT_LOGPROB_KEY",
    "CONTINUOUS_MASK_KEY",
    "OPSD_TEACHER_PROMPT_IDS_KEY",
    "LatentVAE",
    "_restore_policy_latent_vae",
    "apply_continuous_replay_patches",
]
