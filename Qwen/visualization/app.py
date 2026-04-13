#!/usr/bin/env python3
"""
Transformers-based visualization server for Qwen3-VL checkpoints.

This server keeps the existing frontend contract intact while replacing the
backend inference path with Hugging Face generation and tracing.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import errno
import io
import json
import os
import socket
import sys
import time
from types import MethodType
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
import safetensors.torch
from tokenizers import AddedToken
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - environment-specific
    PeftModel = None

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Qwen.inference.vllm_utils import apply_runtime_env_for_thinking
from Qwen.llamafactory.integration import LatentVAE

sys.path.insert(0, str(Path(__file__).parent))
from utils.visualization_utils import compute_tsne


DEFAULT_MODEL_PATH = "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking"
DEFAULT_LORA_PATH = (
    "Qwen/checkpoints/qwen3vl-2b/verl/chimera_gspo/"
    "run_20260401_033750/global_step_20/actor/lora_adapter"
)
DEFAULT_VIS_MAX_TOKENS = int(os.environ.get("QWEN_VIS_DEFAULT_MAX_TOKENS", "8192"))
MAX_VIS_MAX_TOKENS = int(os.environ.get("QWEN_VIS_MAX_TOKENS", "8192"))
ATTENTION_MAX_TOKENS = int(os.environ.get("QWEN_VIS_ATTENTION_MAX_TOKENS", "512"))
TSNE_MAX_FEATURE_POINTS = int(os.environ.get("QWEN_VIS_TSNE_MAX_FEATURE_POINTS", "1500"))
ATTENTION_CACHE_DIR = Path(os.environ.get("QWEN_VIS_ATTENTION_CACHE_DIR", "/tmp/qwen_vis_attention"))
ATTENTION_CACHE_TTL_SECONDS = int(os.environ.get("QWEN_VIS_ATTENTION_CACHE_TTL_SECONDS", str(60 * 60)))
ATTENTION_OVERVIEW_MAX_SIZE = int(os.environ.get("QWEN_VIS_ATTENTION_OVERVIEW_MAX_SIZE", "256"))


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _strip_trailing_think_prompt(prompt_text: str) -> str:
    stripped = prompt_text.rstrip()
    while stripped.endswith("<think>"):
        stripped = stripped[:-len("<think>")].rstrip()
    return stripped


def _has_data(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return value.size > 0
    return len(value) > 0


def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _resolve_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _unwrap_base_model(model: Any) -> Any:
    return model


def _resolve_latent_vae_module(model: Any) -> Any:
    visited: set[int] = set()
    stack = [model]
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        vae = getattr(current, "latent_vae", None)
        if vae is not None:
            return vae
        for attr in ("module", "model", "base_model"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                stack.append(child)
    return None


def _resolve_multimodal_core(model: Any) -> Any:
    current = model
    visited: set[int] = set()
    while True:
        if hasattr(current, "get_placeholder_mask"):
            return current
        current_id = id(current)
        if current_id in visited:
            break
        visited.add(current_id)
        if hasattr(current, "base_model") and current.base_model is not current:
            current = current.base_model
            continue
        if hasattr(current, "model") and current.model is not current:
            current = current.model
            continue
        break
    return model


def _set_attention_implementation(model: Any, implementation: str) -> None:
    visited: set[int] = set()
    stack = [model]
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        setter = getattr(current, "set_attn_implementation", None)
        if callable(setter):
            setter(implementation)
            return
        for attr in ("module", "model", "base_model"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                stack.append(child)


def _align_module_to_model_dtype_device(model: torch.nn.Module, module: torch.nn.Module) -> None:
    ref_tensor = None
    for param in model.parameters():
        if torch.is_floating_point(param):
            ref_tensor = param
            break
    if ref_tensor is None:
        for buffer in model.buffers():
            if torch.is_floating_point(buffer):
                ref_tensor = buffer
                break
    if ref_tensor is not None:
        module.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if value.dtype.is_floating_point:
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _load_image_from_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        if value.startswith("file://"):
            value = value[len("file://"):]
        if value.startswith("data:image/"):
            _, payload = value.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        if os.path.exists(value):
            return Image.open(value).convert("RGB")
        try:
            return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
        except Exception as exc:  # pragma: no cover - bad user input
            raise ValueError("Unsupported image payload") from exc
    raise ValueError(f"Unsupported image value type: {type(value)!r}")


def _normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    normalized: list[dict[str, Any]] = []
    images: list[Image.Image] = []

    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            normalized.append({"role": message["role"], "content": content})
            continue

        if not isinstance(content, list):
            raise ValueError("Message content must be a string or a list")

        normalized_items: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                normalized_items.append({"type": "text", "text": item.get("text", "")})
            elif item_type == "image":
                image_value = item.get("image") or item.get("image_url")
                if image_value is None:
                    raise ValueError("Image item is missing `image` or `image_url`")
                images.append(_load_image_from_value(image_value))
                normalized_items.append({"type": "image"})
            else:
                raise ValueError(f"Unsupported content type: {item_type}")

        normalized.append({"role": message["role"], "content": normalized_items})

    return normalized, images


def _prepare_batch(
    messages: list[dict[str, Any]],
    processor: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], str]:
    normalized_messages, images = _normalize_messages(messages)
    prompt_text = processor.apply_chat_template(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if _env_flag_enabled("VLLM_FORCE_THINK"):
        prompt_text += "<think>"
    else:
        prompt_text = _strip_trailing_think_prompt(prompt_text)

    processor_kwargs: dict[str, Any] = {
        "text": [prompt_text],
        "padding": True,
        "return_tensors": "pt",
    }
    if images:
        processor_kwargs["images"] = images

    batch = processor(**processor_kwargs)
    return _move_batch_to_device(dict(batch), device=device, dtype=dtype), prompt_text


def _ensure_runtime_tokens(tokenizer: Any, model: torch.nn.Module) -> None:
    special_tokens = ["<latent>", "<think_sep>"]
    added_count = 0

    for token in special_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) > 1:
            added_count += tokenizer.add_tokens([token], special_tokens=False)

    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        added = tokenizer.added_tokens_decoder.get(token_id)
        if added is not None and getattr(added, "special", False):
            tokenizer._tokenizer.add_tokens([AddedToken(token, special=False)])

    if added_count > 0:
        model.resize_token_embeddings(len(tokenizer))


def _build_language_inputs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
) -> dict[str, Any]:
    causal_model = _unwrap_base_model(model)
    mm_model = _resolve_multimodal_core(model)
    inputs_embeds = causal_model.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = mm_model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = mm_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = mm_model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = mm_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
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
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        visual_pos_masks = video_mask[..., 0]
        deepstack_visual_embeds = deepstack_video_embeds

    rope_attention_mask = attention_mask
    if rope_attention_mask is not None and rope_attention_mask.ndim == 4:
        rope_attention_mask = torch.diagonal(rope_attention_mask[:, 0], dim1=1, dim2=2)
        if rope_attention_mask.dtype.is_floating_point:
            rope_attention_mask = rope_attention_mask / torch.finfo(rope_attention_mask.dtype).min
            rope_attention_mask = (1.0 - rope_attention_mask).int()

    position_ids, rope_deltas = mm_model.get_rope_index(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        attention_mask=rope_attention_mask,
    )
    mm_model.rope_deltas = rope_deltas

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
    }


def _append_attention_row(matrix: np.ndarray, row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if matrix.size == 0:
        return row.reshape(1, 1)

    prev_len = matrix.shape[0]
    new_len = row.shape[0]
    expanded = np.zeros((new_len, new_len), dtype=np.float32)
    expanded[:prev_len, :prev_len] = matrix
    expanded[new_len - 1, :new_len] = row
    return expanded


class _LastLayerAttentionCollector:
    def __init__(self) -> None:
        self.matrix = np.empty((0, 0), dtype=np.float32)

    def capture(self, attn_weights: torch.Tensor | None) -> None:
        if attn_weights is None or attn_weights.ndim != 4 or attn_weights.shape[0] == 0:
            return
        averaged = attn_weights[0].detach().float().mean(dim=0).cpu().numpy()
        if averaged.ndim != 2:
            return
        if averaged.shape[0] == averaged.shape[1]:
            self.matrix = averaged
            return
        if averaged.shape[0] == 1:
            self.matrix = _append_attention_row(self.matrix, averaged[0])


def _resolve_last_text_attention_module(model: Any) -> Any:
    core_model = _resolve_multimodal_core(model)
    language_model = getattr(core_model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if not layers:
        raise RuntimeError("Unable to resolve last text attention layer for visualization")
    return layers[-1].self_attn


@contextmanager
def _capture_last_layer_attention(model: Any, enabled: bool):
    if not enabled:
        yield None
        return

    attn_module = _resolve_last_text_attention_module(model)
    collector = _LastLayerAttentionCollector()
    original_forward = attn_module.forward

    def wrapped_forward(module_self, *args, **kwargs):
        attn_output, attn_weights = original_forward(*args, **kwargs)
        collector.capture(attn_weights)
        return attn_output, attn_weights

    attn_module.forward = MethodType(wrapped_forward, attn_module)
    try:
        yield collector
    finally:
        attn_module.forward = original_forward


def _align_analysis_lengths(analysis: dict[str, Any]) -> dict[str, Any]:
    token_count = len(analysis["tokens"])

    if _has_data(analysis.get("token_embeddings")):
        analysis["token_embeddings"] = np.asarray(analysis["token_embeddings"], dtype=np.float32)[:token_count]

    if _has_data(analysis.get("hidden_states")):
        analysis["hidden_states"] = np.asarray(analysis["hidden_states"], dtype=np.float32)[:token_count]

    continuous_mask = list(analysis.get("continuous_mask", []))
    if len(continuous_mask) < token_count:
        continuous_mask.extend([False] * (token_count - len(continuous_mask)))
    analysis["continuous_mask"] = continuous_mask[:token_count]

    attention_weights = analysis.get("attention_weights")
    if _has_data(attention_weights):
        attention_array = np.asarray(attention_weights, dtype=np.float32)
        analysis["attention_weights"] = attention_array[:token_count, :token_count]

    vision_embeddings = []
    for item in analysis.get("vision_embeddings", []):
        position = int(item["position"])
        if position < token_count:
            vision_embeddings.append(item)
    analysis["vision_embeddings"] = vision_embeddings

    return analysis


def _extract_image_token_metadata(
    vision_embeddings: list[dict[str, Any]],
    model_config: Any,
    image_grid_thw: torch.Tensor | None = None,
) -> dict[str, Any]:
    image_token_positions = [int(item["position"]) for item in vision_embeddings]
    image_grid = None
    if image_grid_thw is not None and torch.is_tensor(image_grid_thw) and image_grid_thw.numel() >= 3:
        first_grid = image_grid_thw[0].detach().cpu().tolist()
        spatial_merge_size = int(getattr(getattr(model_config, "vision_config", None), "spatial_merge_size", 1) or 1)
        image_grid = [
            int(first_grid[0]),
            max(1, int(first_grid[1]) // spatial_merge_size),
            max(1, int(first_grid[2]) // spatial_merge_size),
        ]
    return {
        "image_token_positions": image_token_positions,
        "image_grid_thw": image_grid,
    }


def _load_vae_checkpoint_if_available(model: Any, model_path: str | None, lora_path: str | None) -> None:
    vae = _resolve_latent_vae_module(model)
    if vae is None:
        return

    candidate_paths: list[Path] = []
    for base in (lora_path, model_path):
        if not base:
            continue
        base_path = Path(base)
        if base_path.is_dir():
            candidate_paths.append(base_path / "vae.safetensors")

    for checkpoint_path in candidate_paths:
        if not checkpoint_path.exists():
            continue
        vae_state_dict = safetensors.torch.load_file(str(checkpoint_path))
        vae.load_state_dict(vae_state_dict, strict=True)
        print(f"[INFO] Loaded latent VAE from {checkpoint_path}", flush=True)
        return


def _ensure_latent_vae_module(model: Any, model_path: str | None, lora_path: str | None) -> None:
    if _resolve_latent_vae_module(model) is not None:
        return

    candidate_paths: list[Path] = []
    for base in (lora_path, model_path):
        if not base:
            continue
        base_path = Path(base)
        if base_path.is_dir():
            candidate_paths.append(base_path / "vae.safetensors")

    if not any(path.exists() for path in candidate_paths):
        return

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        text_config = getattr(model.config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Unable to resolve hidden_size for latent VAE creation")
    hidden_size = int(hidden_size)
    intermediate_size = int(os.environ.get("QWEN3VL_VAE_INTERMEDIATE_SIZE", "512"))
    vae = LatentVAE(hidden_size=hidden_size, intermediate_size=intermediate_size, deterministic=False)
    _align_module_to_model_dtype_device(model, vae)
    model.register_module("latent_vae", vae)
    print(
        f"[INFO] Created latent VAE module for visualization: hidden_size={hidden_size} intermediate_size={intermediate_size}",
        flush=True,
    )


def _prompt_has_unclosed_think(token_ids: list[int]) -> bool:
    think_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    think_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))
    depth = 0
    for token_id in token_ids:
        if token_id == think_start_id:
            depth += 1
        elif token_id == think_end_id and depth > 0:
            depth -= 1
    return depth > 0


def _apply_repetition_penalty_(logits: torch.Tensor, seen_token_ids: list[int], repetition_penalty: float) -> torch.Tensor:
    if repetition_penalty == 1.0 or not seen_token_ids:
        return logits
    penalty = float(repetition_penalty)
    if penalty <= 0.0:
        return logits

    unique_token_ids = torch.tensor(
        sorted(set(int(token_id) for token_id in seen_token_ids)),
        device=logits.device,
        dtype=torch.long,
    )
    selected = logits.index_select(dim=-1, index=unique_token_ids)
    adjusted = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits = logits.clone()
    logits.scatter_(dim=-1, index=unique_token_ids, src=adjusted)
    return logits


def _sample_token_id(
    logits: torch.Tensor,
    *,
    seen_token_ids: list[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> int:
    step_logits = logits.reshape(-1).float()
    step_logits = _apply_repetition_penalty_(step_logits, seen_token_ids, repetition_penalty)

    temp = float(temperature)
    nucleus_p = float(top_p)
    if temp <= 0.0:
        return int(step_logits.argmax(dim=-1).item())

    step_logits = step_logits / max(temp, 1e-5)
    probs = torch.softmax(step_logits, dim=-1)

    if 0.0 < nucleus_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > nucleus_p
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False
        filtered_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        filtered_sum = filtered_probs.sum()
        if torch.isfinite(filtered_sum) and filtered_sum.item() > 0:
            filtered_probs = filtered_probs / filtered_sum
            sampled_offset = torch.multinomial(filtered_probs, num_samples=1)
            return int(sorted_indices[sampled_offset].item())

    sampled_token = torch.multinomial(probs, num_samples=1)
    return int(sampled_token.item())


def _format_response_text_for_display(text: str) -> str:
    if _env_flag_enabled("VLLM_FORCE_THINK"):
        return text
    if text.startswith("<think>") and "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


def _cleanup_attention_cache() -> None:
    try:
        ATTENTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    now = time.time()
    for meta_path in ATTENTION_CACHE_DIR.glob("*.json"):
        try:
            if now - meta_path.stat().st_mtime <= ATTENTION_CACHE_TTL_SECONDS:
                continue
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            array_path = ATTENTION_CACHE_DIR / metadata["array_file"]
            if array_path.exists():
                array_path.unlink()
            meta_path.unlink()
        except Exception:
            continue


def _downsample_attention_matrix(matrix: np.ndarray, max_size: int) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Attention matrix must be 2D")
    rows, cols = matrix.shape
    target = max(1, int(max_size))
    if rows <= target and cols <= target:
        return matrix

    row_indices = np.linspace(0, rows - 1, min(rows, target), dtype=int)
    col_indices = np.linspace(0, cols - 1, min(cols, target), dtype=int)
    return matrix[np.ix_(row_indices, col_indices)]


def _store_attention_matrix(attention_weights: np.ndarray | None, token_count: int) -> dict[str, Any] | None:
    if not _has_data(attention_weights):
        return None

    attention_array = np.asarray(attention_weights, dtype=np.float16)
    if attention_array.ndim != 2:
        return None

    _cleanup_attention_cache()
    ATTENTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    attention_id = f"attn_{uuid.uuid4().hex}"
    array_filename = f"{attention_id}.npy"
    meta_filename = f"{attention_id}.json"
    tmp_array_path = ATTENTION_CACHE_DIR / f"{array_filename}.tmp"
    tmp_meta_path = ATTENTION_CACHE_DIR / f"{meta_filename}.tmp"
    array_path = ATTENTION_CACHE_DIR / array_filename
    meta_path = ATTENTION_CACHE_DIR / meta_filename

    with open(tmp_array_path, "wb") as handle:
        np.save(handle, attention_array, allow_pickle=False)
    os.replace(tmp_array_path, array_path)

    metadata = {
        "id": attention_id,
        "array_file": array_filename,
        "shape": [int(attention_array.shape[0]), int(attention_array.shape[1])],
        "dtype": str(attention_array.dtype),
        "token_count": int(token_count),
        "created_at": int(time.time()),
    }
    with open(tmp_meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    os.replace(tmp_meta_path, meta_path)
    return metadata


def _load_attention_metadata(attention_id: str) -> dict[str, Any]:
    meta_path = ATTENTION_CACHE_DIR / f"{attention_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Attention cache entry not found")
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata


def _load_attention_matrix(attention_id: str) -> tuple[np.memmap, dict[str, Any]]:
    metadata = _load_attention_metadata(attention_id)
    array_path = ATTENTION_CACHE_DIR / metadata["array_file"]
    if not array_path.exists():
        raise HTTPException(status_code=404, detail="Attention cache array not found")
    matrix = np.load(array_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(x) for x in metadata["shape"])
    if matrix.shape != expected_shape:
        raise HTTPException(status_code=500, detail="Attention cache shape mismatch")
    return matrix, metadata


def _generate_with_adaptive_thinking_trace(
    model: Any,
    tokenizer: Any,
    batch: dict[str, Any],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    collect_attention: bool,
) -> tuple[str, dict[str, Any]]:
    causal_model = _unwrap_base_model(model)
    embed_fn = causal_model.get_input_embeddings()
    lm_head = causal_model.lm_head
    core_model = _resolve_multimodal_core(model)
    device = _resolve_device(model)
    latent_vae = _resolve_latent_vae_module(model)

    think_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    think_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))
    max_thinking_steps = int(os.environ.get("QWEN3VL_MAX_THINKING_STEPS", str(max_new_tokens // 2 if max_new_tokens > 1 else 1)))
    min_continuous_steps = max(0, int(os.environ.get("MIN_CONTINUOUS_STEPS", "0")))

    input_ids = batch["input_ids"]
    prompt_len = int(input_ids.shape[1])
    full_attention_mask = torch.ones_like(input_ids, device=device)
    language_inputs = _build_language_inputs(
        causal_model,
        input_ids,
        full_attention_mask,
        pixel_values=batch.get("pixel_values"),
        pixel_values_videos=batch.get("pixel_values_videos"),
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
    )

    with _capture_last_layer_attention(model, collect_attention) as attention_collector:
        with torch.inference_mode():
            prompt_outputs = core_model(
                input_ids=input_ids,
                attention_mask=language_inputs["attention_mask"],
                position_ids=None,
                inputs_embeds=None,
                pixel_values=batch.get("pixel_values"),
                pixel_values_videos=batch.get("pixel_values_videos"),
                image_grid_thw=batch.get("image_grid_thw"),
                video_grid_thw=batch.get("video_grid_thw"),
                output_attentions=False,
                return_dict=True,
                use_cache=True,
            )

    prompt_token_ids = input_ids[0].detach().cpu().tolist()
    token_ids = list(prompt_token_ids)
    token_embedding_matrix = embed_fn.weight.detach()
    token_embeddings_rows = [row for row in language_inputs["inputs_embeds"][0].detach().float().cpu().numpy()]
    hidden_state_rows = [row for row in prompt_outputs.last_hidden_state[0].detach().float().cpu().numpy()]
    continuous_mask = [False] * prompt_len
    generated_token_ids: list[int] = []
    latent_embedding_rows: list[np.ndarray] = []

    vision_embeddings: list[dict[str, Any]] = []
    visual_mask = language_inputs["visual_pos_masks"]
    if visual_mask is not None:
        visual_positions = torch.nonzero(visual_mask[0], as_tuple=False).flatten().tolist()
        for position in visual_positions:
            vision_embeddings.append(
                {
                    "position": int(position),
                    "embedding": np.asarray(token_embeddings_rows[position], dtype=np.float32),
                }
            )

    kv_cache = prompt_outputs.past_key_values
    state = "continuous" if _prompt_has_unclosed_think(prompt_token_ids) else "discrete"
    thinking_steps = 0
    current_hidden = prompt_outputs.last_hidden_state[:, -1:, :]

    def _record_step(
        token_id: int,
        token_embed: torch.Tensor,
        hidden: torch.Tensor,
        *,
        is_continuous: bool,
        latent_embed: torch.Tensor | None,
    ) -> None:
        token_ids.append(int(token_id))
        generated_token_ids.append(int(token_id))
        token_embeddings_rows.append(token_embed[0, 0].detach().float().cpu().numpy())
        hidden_state_rows.append(hidden[0, 0].detach().float().cpu().numpy())
        continuous_mask.append(bool(is_continuous))
        if latent_embed is not None:
            latent_embedding_rows.append(latent_embed[0, 0].detach().float().cpu().numpy())

    with torch.inference_mode():
        while len(generated_token_ids) < max_new_tokens:
            next_token = _sample_token_id(
                lm_head(current_hidden),
                seen_token_ids=token_ids,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            token_embed = token_embedding_matrix[next_token:next_token + 1].unsqueeze(0).to(
                device=current_hidden.device,
                dtype=current_hidden.dtype,
            )

            mode_after_step = state
            if THINKING_MODE_ENABLED:
                if state == "discrete" and next_token == think_start_id:
                    mode_after_step = "continuous"
                    thinking_steps = 0
                elif state == "continuous":
                    forced_exit = thinking_steps >= max_thinking_steps
                    natural_exit = next_token == think_end_id and thinking_steps >= min_continuous_steps
                    if forced_exit or natural_exit:
                        mode_after_step = "discrete"
                        thinking_steps = 0

            next_input_ids = token_tensor
            next_input_embeds = None
            latent_embed = None
            if mode_after_step == "continuous":
                latent_embed = current_hidden
                if latent_vae is not None:
                    if next(latent_vae.parameters()).device != current_hidden.device or next(latent_vae.parameters()).dtype != current_hidden.dtype:
                        latent_vae = latent_vae.to(device=current_hidden.device, dtype=current_hidden.dtype)
                    vae_dist = latent_vae.forward(current_hidden, temperature=1.0)
                    latent_embed = vae_dist.rsample()
                next_input_ids = None
                next_input_embeds = latent_embed
                thinking_steps += 1
            _record_step(
                next_token,
                token_embed,
                current_hidden,
                is_continuous=(mode_after_step == "continuous"),
                latent_embed=latent_embed if mode_after_step == "continuous" else None,
            )

            state = mode_after_step
            if state != "continuous" and next_token == tokenizer.eos_token_id:
                break

            cache_position = torch.tensor([prompt_len + len(generated_token_ids) - 1], dtype=torch.long, device=device)
            decode_attention_mask = torch.ones(
                (1, prompt_len + len(generated_token_ids)),
                dtype=torch.long,
                device=device,
            )
            outputs = core_model(
                input_ids=next_input_ids,
                inputs_embeds=next_input_embeds,
                attention_mask=decode_attention_mask,
                past_key_values=kv_cache,
                cache_position=cache_position,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            kv_cache = outputs.past_key_values
            current_hidden = outputs.last_hidden_state[:, -1:, :]

    last_layer_attention = attention_collector.matrix if attention_collector is not None else np.empty((0, 0), dtype=np.float32)

    tokens = []
    for position, token_id in enumerate(token_ids):
        tokens.append(
            {
                "id": int(token_id),
                "text": _decode_token_text(tokenizer, token_id),
                "position": position,
            }
        )

    response_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    analysis = {
        "tokens": tokens,
        "token_embeddings": np.asarray(token_embeddings_rows, dtype=np.float32),
        "hidden_states": np.asarray(hidden_state_rows, dtype=np.float32),
        "attention_weights": last_layer_attention,
        "vision_embeddings": vision_embeddings,
        "continuous_mask": continuous_mask,
        "prompt_token_count": prompt_len,
        "completion_token_count": len(generated_token_ids),
        "latent_embeddings": np.asarray(latent_embedding_rows, dtype=np.float32) if latent_embedding_rows else None,
        "image_token_metadata": _extract_image_token_metadata(
            vision_embeddings,
            getattr(model, "config", None),
            batch.get("image_grid_thw"),
        ),
    }
    return _format_response_text_for_display(response_text), _align_analysis_lengths(analysis)


def _collect_sequence_analysis(
    model: Any,
    tokenizer: Any,
    generated_sequences: torch.Tensor,
    generation_batch: dict[str, Any],
    *,
    collect_attention: bool,
    generated_input_embeds: torch.Tensor | None = None,
) -> dict[str, Any]:
    full_input_ids = generated_sequences
    full_attention_mask = torch.ones_like(full_input_ids, device=full_input_ids.device)

    language_inputs = _build_language_inputs(
        model,
        full_input_ids,
        full_attention_mask,
        pixel_values=generation_batch.get("pixel_values"),
        pixel_values_videos=generation_batch.get("pixel_values_videos"),
        image_grid_thw=generation_batch.get("image_grid_thw"),
        video_grid_thw=generation_batch.get("video_grid_thw"),
    )

    replay_inputs_embeds = language_inputs["inputs_embeds"]
    if generated_input_embeds is not None and generated_input_embeds.numel() > 0:
        generated_count = int(generated_input_embeds.shape[1])
        prompt_len = int(full_input_ids.shape[1]) - generated_count
        replay_inputs_embeds = replay_inputs_embeds.clone()
        replay_inputs_embeds[:, prompt_len:, :] = generated_input_embeds.to(
            device=replay_inputs_embeds.device,
            dtype=replay_inputs_embeds.dtype,
        )

    core_model = _resolve_multimodal_core(model)
    with torch.inference_mode():
        outputs = core_model(
            input_ids=None,
            inputs_embeds=replay_inputs_embeds,
            attention_mask=language_inputs["attention_mask"],
            position_ids=language_inputs["position_ids"],
            output_attentions=collect_attention,
            return_dict=True,
            use_cache=False,
        )

    token_ids = full_input_ids[0].detach().cpu().tolist()
    token_embeddings = replay_inputs_embeds[0].detach().float().cpu().numpy()
    hidden_states = outputs.last_hidden_state[0].detach().float().cpu().numpy()

    last_layer_attention = None
    if collect_attention and outputs.attentions:
        last_layer_attention = outputs.attentions[-1][0].detach().float().mean(dim=0).cpu().numpy()

    vision_embeddings: list[dict[str, Any]] = []
    visual_mask = language_inputs["visual_pos_masks"]
    if visual_mask is not None:
        visual_positions = torch.nonzero(visual_mask[0], as_tuple=False).flatten().tolist()
        for position in visual_positions:
            vision_embeddings.append(
                {
                    "position": int(position),
                    "embedding": token_embeddings[position].astype(np.float32),
                }
            )

    tokens = []
    for position, token_id in enumerate(token_ids):
        tokens.append(
            {
                "id": int(token_id),
                "text": _decode_token_text(tokenizer, token_id),
                "position": position,
            }
        )

    return _align_analysis_lengths({
        "tokens": tokens,
        "token_embeddings": token_embeddings,
        "hidden_states": hidden_states,
        "attention_weights": last_layer_attention,
        "vision_embeddings": vision_embeddings,
        "continuous_mask": [False] * len(tokens),
        "image_token_metadata": _extract_image_token_metadata(
            vision_embeddings,
            getattr(model, "config", None),
            generation_batch.get("image_grid_thw"),
        ),
    })


def _build_tsne_payload(
    tokens: list[dict[str, Any]],
    token_embeddings: np.ndarray | None,
    hidden_states: np.ndarray | None,
    latent_embeddings: np.ndarray | None,
    vision_embeddings: list[dict[str, Any]],
    continuous_mask: list[bool],
    model_config: Any,
) -> tuple[list[list[float]] | None, list[str], list[int]]:
    if not _has_data(token_embeddings) and not _has_data(hidden_states) and not _has_data(latent_embeddings) and not vision_embeddings:
        return None, [], []

    vision_pos_to_embedding = {item["position"]: np.asarray(item["embedding"]).reshape(-1) for item in vision_embeddings}

    all_features: list[np.ndarray] = []
    feature_types: list[str] = []
    position_indices: list[int] = []

    for idx, token in enumerate(tokens):
        is_true_visual_token = token["position"] in vision_pos_to_embedding

        if is_true_visual_token:
            all_features.append(vision_pos_to_embedding[token["position"]])
            feature_types.append("image_token")
            position_indices.append(idx)

        if _has_data(token_embeddings) and idx < len(token_embeddings):
            emb = np.asarray(token_embeddings[idx]).reshape(-1)
            if emb.size > 0:
                all_features.append(emb)
                feature_types.append("image_token" if is_true_visual_token else "token_emb")
                position_indices.append(idx)

        if _has_data(hidden_states) and idx < len(hidden_states):
            hs = np.asarray(hidden_states[idx]).reshape(-1)
            if hs.size > 0:
                all_features.append(hs)
                feature_types.append("image_token" if is_true_visual_token else "hidden_state")
                position_indices.append(idx)

        if idx < len(continuous_mask) and continuous_mask[idx] and _has_data(latent_embeddings):
            latent_idx = sum(1 for flag in continuous_mask[:idx + 1] if flag) - 1
            if 0 <= latent_idx < len(latent_embeddings):
                latent = np.asarray(latent_embeddings[latent_idx]).reshape(-1)
                if latent.size > 0:
                    all_features.append(latent)
                    feature_types.append("vae_sample")
                    position_indices.append(idx)

    if not all_features:
        return None, [], []

    if len(all_features) > TSNE_MAX_FEATURE_POINTS:
        sample_indices = np.linspace(0, len(all_features) - 1, TSNE_MAX_FEATURE_POINTS, dtype=int)
        all_features = [all_features[idx] for idx in sample_indices]
        feature_types = [feature_types[idx] for idx in sample_indices]
        position_indices = [position_indices[idx] for idx in sample_indices]

    feature_matrix = np.vstack(all_features)
    tsne_coords = compute_tsne(feature_matrix)
    return tsne_coords.tolist(), feature_types, position_indices


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = DEFAULT_VIS_MAX_TOKENS
    top_p: float = 1.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False


app = FastAPI(title="Qwen3-VL Thinking Mode Visualization")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

apply_runtime_env_for_thinking(repo_root=_REPO_ROOT)
THINKING_MODE_ENABLED = _env_flag_enabled("VLLM_THINKING", default="1") or _env_flag_enabled("VLLM_FORCE_THINK")

model = None
processor = None
tokenizer = None
config: dict[str, Any] = {}


def resolve_model_path(model_path: str | None, lora_path: str | None) -> str:
    if not lora_path:
        if not model_path:
            raise ValueError("Either --model-path or --lora-path must be provided")
        return model_path

    adapter_config_path = Path(lora_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        if not model_path:
            raise ValueError(f"No adapter_config.json found in {lora_path}")
        return model_path

    with open(adapter_config_path, "r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    return adapter_config.get("base_model_name_or_path") or model_path


def load_model(
    model_path: str | None,
    *,
    lora_path: str | None,
    gpu_memory_utilization: float,
) -> None:
    global model, processor, tokenizer, config

    resolved_model_path = resolve_model_path(model_path, lora_path)
    print(f"\n{'=' * 80}")
    print("Loading Hugging Face model...")
    print(f"{'=' * 80}")
    print(f"Model: {resolved_model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(resolved_model_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        resolved_model_path,
        trust_remote_code=True,
        torch_dtype=load_dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    _ensure_runtime_tokens(tokenizer, model)

    if lora_path:
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapters")
        model = PeftModel.from_pretrained(model, lora_path)

    _set_attention_implementation(model, "eager")
    model.eval()
    setattr(model, "tokenizer", tokenizer)
    _ensure_latent_vae_module(model, resolved_model_path, lora_path)
    _load_vae_checkpoint_if_available(model, resolved_model_path, lora_path)
    print(f"Loaded model in {time.time() - start_time:.2f}s", flush=True)

    config = {
        "backend": "hf",
        "model_path": resolved_model_path,
        "requested_model_path": model_path,
        "lora_path": lora_path,
        "thinking_enabled": THINKING_MODE_ENABLED,
    }


def run_generation(
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    top_p: float,
    repetition_penalty: float,
) -> tuple[str, dict[str, Any]]:
    if model is None or processor is None or tokenizer is None:
        raise RuntimeError("Model is not loaded")
    effective_max_tokens = max(1, min(int(max_tokens), MAX_VIS_MAX_TOKENS))
    device = _resolve_device(model)
    model_dtype = next(param.dtype for param in model.parameters() if torch.is_floating_point(param))
    batch, _ = _prepare_batch(messages, processor, device=device, dtype=model_dtype)
    collect_attention = True
    response_text, analysis = _generate_with_adaptive_thinking_trace(
        model,
        tokenizer,
        batch,
        max_new_tokens=effective_max_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        collect_attention=collect_attention,
    )
    tsne_coordinates, tsne_feature_types, tsne_position_indices = _build_tsne_payload(
        analysis["tokens"],
        analysis.get("token_embeddings"),
        analysis.get("hidden_states"),
        analysis.get("latent_embeddings"),
        analysis.get("vision_embeddings", []),
        analysis.get("continuous_mask", []),
        getattr(model, "config", None),
    )
    analysis["tsne_coordinates"] = tsne_coordinates
    analysis["tsne_feature_types"] = tsne_feature_types
    analysis["tsne_position_indices"] = tsne_position_indices
    return response_text, analysis


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "checkpoint_path": config.get("lora_path") or config.get("model_path"),
        "backend": config.get("backend"),
        "thinking_enabled": config.get("thinking_enabled", False),
    }


@app.get("/v1/models")
async def list_models():
    model_id = Path(config.get("requested_model_path") or config.get("model_path") or "unknown").name
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.get("/stats")
async def stats():
    return {
        "backend": config.get("backend"),
        "model_path": config.get("model_path"),
        "lora_path": config.get("lora_path"),
        "thinking_enabled": config.get("thinking_enabled", False),
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = static_dir / "index.html"
    with open(index_path, "r", encoding="utf-8") as handle:
        return HTMLResponse(content=handle.read())


@app.get("/api/example")
async def get_example():
    deepvision_path = _REPO_ROOT / "Qwen/data/metadata/deepvision_103k_metadata.jsonl"
    if not deepvision_path.exists():
        raise HTTPException(status_code=404, detail="Deepvision metadata file not found")

    with open(deepvision_path, "r", encoding="utf-8") as handle:
        example = json.loads(handle.readline())

    image_path = example.get("query_image_path")
    question = example.get("question", "")

    image_base64 = None
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as handle:
            image_base64 = base64.b64encode(handle.read()).decode("utf-8")

    return {
        "image_base64": image_base64,
        "question": question,
        "answer": messages[-1].get("content", "") if messages else "",
    }


@app.post("/api/infer")
async def infer_vis(request: dict[str, Any]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        text = request.get("text", "")
        image_base64 = request.get("image_base64")
        requested_max_tokens = int(request.get("max_tokens", DEFAULT_VIS_MAX_TOKENS))
        max_tokens = max(1, min(requested_max_tokens, MAX_VIS_MAX_TOKENS))
        temperature = request.get("temperature", 0.7)
        start_time = time.time()

        messages: list[dict[str, Any]]
        if image_base64:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_base64},
                        {"type": "text", "text": text},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": text}]

        print(
            f"[infer] start prompt_chars={len(text)} image={bool(image_base64)} "
            f"requested_max_tokens={requested_max_tokens} effective_max_tokens={max_tokens}",
            flush=True,
        )

        response_text, analysis = run_generation(
            messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=1.0,
            repetition_penalty=1.0,
        )
        attention_cache = _store_attention_matrix(analysis.get("attention_weights"), len(analysis["tokens"]))

        tsne_coordinates = analysis.get("tsne_coordinates")
        tsne_feature_types = analysis.get("tsne_feature_types", [])
        tsne_position_indices = analysis.get("tsne_position_indices", [])

        elapsed = time.time() - start_time
        print(
            f"[infer] done completion_tokens={analysis['completion_token_count']} "
            f"total_tokens={len(analysis['tokens'])} elapsed_s={elapsed:.2f}",
            flush=True,
        )

        return {
            "answer": response_text,
            "tokens": analysis["tokens"],
            "hidden_states": None,
            "continuous_mask": analysis["continuous_mask"],
            "tsne_coordinates": tsne_coordinates,
            "tsne_feature_types": tsne_feature_types,
            "tsne_position_indices": tsne_position_indices,
            "attention": attention_cache,
            "token_metadata": {
                "total_tokens": len(analysis["tokens"]),
                "continuous_tokens": sum(1 for flag in analysis["continuous_mask"] if flag),
                "prompt_token_count": analysis["prompt_token_count"],
                "image_token_positions": analysis.get("image_token_metadata", {}).get("image_token_positions", []),
                "image_grid_thw": analysis.get("image_token_metadata", {}).get("image_grid_thw"),
            },
        }
    except Exception as exc:
        print(f"[infer] failed: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


@app.get("/api/attention/{attention_id}/overview")
async def get_attention_overview(attention_id: str, max_size: int = ATTENTION_OVERVIEW_MAX_SIZE):
    matrix, metadata = _load_attention_matrix(attention_id)
    overview = _downsample_attention_matrix(matrix, max_size=max_size).astype(np.float32, copy=False)
    row_indices = np.linspace(0, matrix.shape[0] - 1, overview.shape[0], dtype=int).tolist()
    col_indices = np.linspace(0, matrix.shape[1] - 1, overview.shape[1], dtype=int).tolist()
    return {
        "attention_id": attention_id,
        "matrix": overview.tolist(),
        "row_indices": row_indices,
        "col_indices": col_indices,
        "shape": metadata["shape"],
        "dtype": metadata["dtype"],
    }


@app.get("/api/attention/{attention_id}/row")
async def get_attention_row(attention_id: str, index: int):
    matrix, metadata = _load_attention_matrix(attention_id)
    if index < 0 or index >= matrix.shape[0]:
        raise HTTPException(status_code=400, detail="Attention row index out of range")
    row = np.asarray(matrix[index], dtype=np.float32)
    return {
        "attention_id": attention_id,
        "index": int(index),
        "row": row.tolist(),
        "shape": metadata["shape"],
        "dtype": metadata["dtype"],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented for the HF backend")

    try:
        messages = [{"role": message.role, "content": message.content} for message in request.messages]
        response_text, analysis = run_generation(
            messages,
            temperature=float(request.temperature),
            max_tokens=int(request.max_tokens),
            top_p=float(request.top_p),
            repetition_penalty=float(request.repetition_penalty),
        )
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.get("model_path"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": analysis["prompt_token_count"],
                "completion_tokens": analysis["completion_token_count"],
                "total_tokens": analysis["prompt_token_count"] + analysis["completion_token_count"],
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/completions")
async def completions(_: dict[str, Any]):
    raise HTTPException(status_code=501, detail="Use /v1/chat/completions instead")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL visualization server")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to base model")
    parser.add_argument("--port", type=int, default=8501, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--tensor-parallel-size", type=int, default=0, help="Retained for CLI compatibility")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Retained for CLI compatibility")
    parser.add_argument("--lora-path", type=str, default=DEFAULT_LORA_PATH, help="LoRA adapter path")
    parser.add_argument("--lora-name", type=str, default="default", help="Retained for CLI compatibility")
    args = parser.parse_args()

    del args.tensor_parallel_size
    del args.lora_name

    def find_free_port(start_port: int) -> int:
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("", port))
                    sock.listen(1)
                    return port
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    continue
                raise
        return start_port

    original_port = args.port
    args.port = find_free_port(args.port)
    if args.port != original_port:
        print(f"Port {original_port} in use, auto-selected port: {args.port}")

    load_model(
        model_path=args.model_path,
        lora_path=args.lora_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(f"\n{'=' * 80}")
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"{'=' * 80}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
