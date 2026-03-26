"""Repo-local continuous replay patches for VERL GSPO."""

from __future__ import annotations

import importlib
import logging
import os
import functools
import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

CONTINUOUS_HIDDEN_KEY = "continuous_hidden_states"
CONTINUOUS_LATENT_KEY = "continuous_latent_embeddings"
CONTINUOUS_LATENT_LOGPROB_KEY = "continuous_latent_log_probs"
CONTINUOUS_MASK_KEY = "continuous_token_mask"
_PATCHED = False
_REPLAY_DTYPE_CAST_LOGGED = False
_GRAD_FLOW_LOGGED = False
_GRAD_FLOW_DEBUG = os.environ.get("QWEN3VL_CONTINUOUS_REPLAY_DEBUG", "0") == "1"
_POLICY_VAE_LOGGED = False
_ROLLOUT_CERT_LOGGED = False
_REPLAY_CERT_LOGGED = False


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

    hidden_size = _resolve_hidden_size(policy.actor_module)
    vae = LatentVAE(
        hidden_size=hidden_size,
        intermediate_size=int(os.environ.get("QWEN3VL_VAE_INTERMEDIATE_SIZE", "512")),
        deterministic=False,
    )
    vae.load_state_dict(load_file(str(vae_path)), strict=True)
    vae.train(policy.actor_optimizer is not None)
    for param in vae.parameters():
        param.requires_grad = policy.actor_optimizer is not None
    policy.latent_vae = vae
    policy._qwen3vl_latent_vae_path = str(vae_path)

    if policy.actor_optimizer is not None:
        existing_param_ids = {id(param) for group in policy.actor_optimizer.param_groups for param in group["params"]}
        new_params = [param for param in policy.latent_vae.parameters() if id(param) not in existing_param_ids]
        if new_params:
            policy.actor_optimizer.add_param_group({"params": new_params})
    if not _POLICY_VAE_LOGGED:
        logger.warning(
            "[ContinuousReplay] Actor latent VAE loaded: path=%s trainable=%s params=%s",
            vae_path,
            bool(policy.actor_optimizer is not None),
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


def _has_continuous_replay_inputs(micro_batch) -> bool:
    return (
        CONTINUOUS_HIDDEN_KEY in micro_batch
        and CONTINUOUS_LATENT_KEY in micro_batch
        and CONTINUOUS_MASK_KEY in micro_batch
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


def _validate_request_trace(trace: dict, request_id: str, response_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_row = np.asarray(trace[CONTINUOUS_HIDDEN_KEY], dtype=np.float16)
    latent_row = np.asarray(trace[CONTINUOUS_LATENT_KEY], dtype=np.float16)
    mask_row = np.asarray(trace[CONTINUOUS_MASK_KEY], dtype=np.bool_).reshape(-1)

    if hidden_row.ndim == 1 and hidden_row.size > 0:
        hidden_row = hidden_row.reshape(1, -1)
    if latent_row.ndim == 1 and latent_row.size > 0:
        latent_row = latent_row.reshape(1, -1)

    if mask_row.size > response_length:
        mask_row = mask_row[:response_length]
    active_positions = int(mask_row.sum())
    hidden_steps = int(hidden_row.shape[0]) if hidden_row.ndim == 2 else 0
    latent_steps = int(latent_row.shape[0]) if latent_row.ndim == 2 else 0
    if hidden_steps != active_positions or latent_steps != active_positions:
        raise RuntimeError(
            "Continuous replay trace shape mismatch for request_id="
            f"{request_id}: active_positions={active_positions} hidden_steps={hidden_steps} latent_steps={latent_steps}"
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
        mask = input_ids == model.config.image_token_id
        image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds, deepstack_video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == model.config.video_token_id
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
    global _REPLAY_CERT_LOGGED
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

    replay_row_ids = torch.full(input_ids.shape, -1, dtype=torch.long, device=input_ids.device)
    hidden_tensors = []
    latent_tensors = []
    next_row_id = 0
    total_replaced_positions = 0
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

        hidden_tensors.append(torch.as_tensor(hidden_np[:n_steps], device=input_ids.device, dtype=vae_dtype))
        latent_tensors.append(torch.as_tensor(latent_np[:n_steps], device=input_ids.device, dtype=vae_dtype))
        assign_positions = torch.as_tensor(true_positions[:n_steps] + response_start, device=input_ids.device, dtype=torch.long)
        replay_row_ids[batch_idx, assign_positions] = torch.arange(
            next_row_id,
            next_row_id + n_steps,
            device=input_ids.device,
            dtype=torch.long,
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
        "continuous_replay_row_ids": replay_row_ids,
        "continuous_replay_hidden_states": torch.cat(hidden_tensors, dim=0),
        "continuous_replay_latent_embeddings": torch.cat(latent_tensors, dim=0),
        "continuous_replay_latent_vae": vae,
    }
    return {
        "continuous_replay_row_ids": replay_row_ids,
        "continuous_replay_latent_embeddings": replay_state["continuous_replay_latent_embeddings"],
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
        )

    def _apply_continuous_replay_to_inputs_embeds(
        inputs_embeds: torch.Tensor,
        continuous_replay_row_ids: torch.Tensor | None,
        continuous_replay_latent_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        if continuous_replay_row_ids is None or continuous_replay_latent_embeddings is None:
            return inputs_embeds

        row_ids = continuous_replay_row_ids.to(device=inputs_embeds.device)
        replay_mask = row_ids >= 0
        if not torch.any(replay_mask):
            return inputs_embeds

        latent_embeddings = continuous_replay_latent_embeddings.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        selected_row_ids = row_ids[replay_mask].to(dtype=torch.long)
        outputs = inputs_embeds.clone()
        outputs[replay_mask] = latent_embeddings.index_select(0, selected_row_ids)
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
    from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd as rollout_mod

    _patch_verl_qwen3vl_inputs_embeds_support()

    original_policy_init = dp_actor_mod.DataParallelPPOActor.__init__
    original_forward_micro_batch = dp_actor_mod.DataParallelPPOActor._forward_micro_batch

    def compat_policy_init(self, config, actor_module, actor_optimizer=None):
        original_policy_init(self, config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        _load_policy_latent_vae(self)

    def compat_forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        if not _has_continuous_replay_inputs(micro_batch):
            return original_forward_micro_batch(self, micro_batch, temperature, calculate_entropy)

        if self.use_ulysses_sp:
            raise NotImplementedError("Continuous replay does not support ulysses sequence parallelism yet")

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            replay_model_kwargs, replay_state = _prepare_inputs_embeds(
                self,
                micro_batch,
                input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs,
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
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
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
            log_probs, entropy = _merge_continuous_policy_stats(log_probs, entropy, replay_state)
            return entropy, log_probs

    def _continuous_non_tensor_keys(data) -> list[str]:
        keys = []
        if "multi_modal_inputs" in data.non_tensor_batch:
            keys.append("multi_modal_inputs")
        if CONTINUOUS_HIDDEN_KEY in data.non_tensor_batch and CONTINUOUS_MASK_KEY in data.non_tensor_batch:
            keys.extend([CONTINUOUS_HIDDEN_KEY, CONTINUOUS_MASK_KEY])
        if CONTINUOUS_LATENT_KEY in data.non_tensor_batch:
            keys.append(CONTINUOUS_LATENT_KEY)
        return keys

    def compat_compute_log_prob(self, data, calculate_entropy=False):
        self.actor_module.eval()
        if getattr(self, "latent_vae", None) is not None:
            self.latent_vae.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=_continuous_non_tensor_keys(data))

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = dp_actor_mod.prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(dp_actor_mod.get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if calculate_entropy else None
        if use_dynamic_bsz:
            log_probs = dp_actor_mod.restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = dp_actor_mod.restore_dynamic_batch(entropys, batch_idx_list)
        return log_probs, entropys

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

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

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
                    if _GRAD_FLOW_DEBUG:
                        _log_grad_flow_probe(self, model_inputs)

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    dp_actor_mod.append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                dp_actor_mod.append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        self.actor_optimizer.zero_grad()
        return metrics

    def compat_rollout_generate_sequences(self, prompts, **kwargs):
        global _ROLLOUT_CERT_LOGGED
        from verl import DataProto
        from tensordict import TensorDict
        from vllm.lora.request import LoRARequest

        from vllm_thinking.trace_store import pop_request_trace

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [rollout_mod._pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
                strict=True,
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        for input_data in vllm_inputs:
            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            sampling_kwargs = {"best_of": 1, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "temperature": 0, "n": 1}
        elif is_validate:
            sampling_kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        else:
            sampling_kwargs = kwargs

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        continuous_hidden_states = []
        continuous_latent_embeddings = []
        continuous_token_masks = []
        with self.update_sampling_params(**sampling_kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            response = []
            rollout_log_probs = []
            for output in outputs:
                trace = pop_request_trace(output.request_id)
                if len(output.outputs) != 1:
                    raise RuntimeError(
                        "Continuous replay currently requires exactly one sampled output per vLLM request; "
                        f"got {len(output.outputs)} outputs for request_id={output.request_id}"
                    )
                if trace is None:
                    raise RuntimeError(
                        "Continuous replay trace missing for vLLM request_id="
                        f"{output.request_id}. This indicates trace capture did not survive rollout."
                    )
                hidden_row, latent_row, mask_row = _validate_request_trace(
                    trace,
                    request_id=str(output.request_id),
                    response_length=self.config.response_length,
                )
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)

                    if sample_id == 0:
                        continuous_hidden_states.append(hidden_row)
                        continuous_latent_embeddings.append(latent_row)
                        continuous_token_masks.append(mask_row)
                        if self.config.calculate_log_probs:
                            latent_log_probs = np.asarray(
                                trace.get(CONTINUOUS_LATENT_LOGPROB_KEY, np.empty((0,), dtype=np.float32)),
                                dtype=np.float32,
                            ).reshape(-1)
                            true_positions = np.flatnonzero(mask_row)
                            if latent_log_probs.shape[0] != true_positions.size:
                                raise RuntimeError(
                                    "Continuous replay latent log_prob mismatch for request_id="
                                    f"{output.request_id}: latent_log_probs={latent_log_probs.shape[0]} active_positions={true_positions.size}"
                                )
                            for pos_idx in range(true_positions.size):
                                curr_log_prob[int(true_positions[pos_idx])] = float(latent_log_probs[pos_idx])
                    else:
                        continuous_hidden_states.append(np.empty((0, 0), dtype=np.float16))
                        continuous_latent_embeddings.append(np.empty((0, 0), dtype=np.float16))
                        continuous_token_masks.append(np.zeros(len(response_ids), dtype=np.bool_))
                    if self.config.calculate_log_probs:
                        rollout_log_probs.append(curr_log_prob)

            response = rollout_mod.pad_2d_list_to_length(
                response, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
            if self.config.calculate_log_probs:
                rollout_log_probs = rollout_mod.pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = rollout_mod.get_response_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype,
        )
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

    dp_actor_mod.DataParallelPPOActor.__init__ = compat_policy_init
    dp_actor_mod.DataParallelPPOActor._forward_micro_batch = compat_forward_micro_batch
    dp_actor_mod.DataParallelPPOActor.compute_log_prob = compat_compute_log_prob
    dp_actor_mod.DataParallelPPOActor.update_policy = compat_update_policy
    rollout_mod.vLLMRollout.generate_sequences = compat_rollout_generate_sequences
    _PATCHED = True

__all__ = [
    "CONTINUOUS_HIDDEN_KEY",
    "CONTINUOUS_LATENT_KEY",
    "CONTINUOUS_LATENT_LOGPROB_KEY",
    "CONTINUOUS_MASK_KEY",
    "LatentVAE",
    "_restore_policy_latent_vae",
    "apply_continuous_replay_patches",
]
