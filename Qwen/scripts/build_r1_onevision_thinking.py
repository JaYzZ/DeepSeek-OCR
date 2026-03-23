#!/usr/bin/env python3
"""
Build R1-OneVision Thinking Dataset for Qwen3VL Native Training

Uses Qwen3VL Native Vision Encoder for end-to-end feature extraction:
1. Question image → Vision Encoder (ViT + Projector) → LLM space (2048-dim)
2. Thinking text → Adaptive renderer → Images → Vision Encoder → LLM space
3. Latent format:
   - latent_ground_truth: Thinking image features (for injection at <latent>)
   - latent_supervision: Original image features (for OT loss reference)

All features are cached in .feature_cache directory and reused across runs.
Features are already in LLM hidden dimension space - ready for direct injection and OT loss.
"""

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

# Setup path for local imports
script_dir = Path(__file__).parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(repo_root))

# Local imports
from Renderer.skia_renderer import (
    SkiaRenderer,
    measure_finalized_text_canvas,
    prepare_text_for_rendering,
    snap_canvas_to_grid,
)
from Qwen.scripts.utils import chunk_thinking_text, compress_newlines, format_cot_subsequences
from OCRVL.encoder.qwen3vl_encoder import Qwen3VLEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking"
DEFAULT_TRAIN_CONFIG = repo_root / "Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml"


@dataclass(frozen=True)
class SkiaRenderConfig:
    min_size: int = 32
    max_size: int = 4096
    vit_divisor: int = 32
    padding: int = 12
    thinking_padding: int = 8
    short_line_wrap_threshold: int = 20
    short_line_min_lines: int = 8


def _atomic_save_png(image: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=output_path.suffix,
        prefix=f"{output_path.stem}.",
        dir=output_path.parent,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        Image.fromarray(image).save(tmp_path, format="PNG")
        if not tmp_path.exists():
            raise FileNotFoundError(f"Temporary render output missing after save: {tmp_path}")
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class AdaptiveSkiaRenderer:
    """Skia renderer with the same adaptive text preprocessing interface as Vello."""

    def __init__(self, min_font_size: float = 10.0, max_font_size: float = 10.0):
        self._config = SkiaRenderConfig()
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self._renderer_cache: Dict[tuple[int, int, int, bool, bool], SkiaRenderer] = {}

    def _get_renderer(self, width: int, height: int, padding: int, preserve_newlines: bool) -> SkiaRenderer:
        key = (width, height, padding, preserve_newlines, False)
        renderer = self._renderer_cache.get(key)
        if renderer is None:
            renderer = SkiaRenderer(
                width=width,
                height=height,
                padding=padding,
                min_font_size=self.min_font_size,
                max_font_size=self.max_font_size,
                preserve_newlines=preserve_newlines,
            )
            self._renderer_cache[key] = renderer
        return renderer

    def _measure_canvas(self, prepared_text: str, padding: int) -> tuple[int, int, int, int]:
        raw_width, raw_height = measure_finalized_text_canvas(
            prepared_text,
            padding=padding,
            font_size=self.min_font_size,
            min_size=self._config.min_size,
        )
        snapped_width, snapped_height = snap_canvas_to_grid(
            raw_width,
            raw_height,
            divisor=self._config.vit_divisor,
            min_size=self._config.min_size,
            max_size=self._config.max_size,
        )
        return raw_width, raw_height, snapped_width, snapped_height

    def _prepare_render(self, text: str, thinking_mode: bool) -> tuple[str, dict[str, Any], int]:
        padding = self._config.thinking_padding if thinking_mode else self._config.padding
        prepared_text, layout_info = prepare_text_for_rendering(
            text,
            short_line_threshold=self._config.short_line_wrap_threshold,
            min_lines_for_reflow=self._config.short_line_min_lines,
            max_canvas_size=self._config.max_size,
            measurement_padding=padding,
            measurement_font_size=self.min_font_size,
            min_canvas_size=self._config.min_size,
            measurement_divisor=self._config.vit_divisor,
        )
        return prepared_text, layout_info, padding

    def measure_text(self, text: str, thinking_mode: bool) -> tuple[int, int, int, int]:
        prepared_text, _, padding = self._prepare_render(text, thinking_mode)
        return self._measure_canvas(prepared_text, padding)

    def _render_one(self, text: str, thinking_mode: bool) -> Image.Image:
        prepared_text, layout_info, padding = self._prepare_render(text, thinking_mode)
        if not prepared_text or layout_info.get("layout_type") == "failed":
            logger.warning(f"Render failed: layout_type={layout_info.get('layout_type')}, text_len={len(text)}")
            return None
        _, _, width, height = self._measure_canvas(prepared_text, padding)
        return self._get_renderer(width, height, padding, layout_info["preserve_newlines"]).render_batch([prepared_text])[0]

    def render_batch(self, texts: List[str], thinking_mode: bool = False) -> List[Image.Image]:
        prepared_specs = []
        for text in texts:
            prepared_text, layout_info, padding = self._prepare_render(text, thinking_mode)
            prepared_specs.append((prepared_text, padding, layout_info["preserve_newlines"]))

        grouped: Dict[tuple[int, int, int, bool], List[tuple[int, str]]] = {}
        for idx, (prepared_text, padding, preserve_newlines) in enumerate(prepared_specs):
            _, _, width, height = self._measure_canvas(prepared_text, padding)
            grouped.setdefault((width, height, padding, preserve_newlines), []).append((idx, prepared_text))

        results: List[Image.Image] = [None] * len(texts)
        for (width, height, padding, preserve_newlines), items in grouped.items():
            renderer = self._get_renderer(width, height, padding, preserve_newlines)
            images = renderer.render_batch([text for _, text in items])
            for (idx, _), image in zip(items, images):
                results[idx] = image
        return results

    def render(self, text: str, output_path: str, thinking_mode: bool = False) -> bool:
        if not text or not text.strip():
            return False
        image = self._render_one(text, thinking_mode)
        if image is None:
            return False
        output_path = Path(output_path)
        _atomic_save_png(image, output_path)
        return True

    def shutdown(self) -> None:
        for renderer in self._renderer_cache.values():
            renderer.shutdown()
        self._renderer_cache.clear()


def load_cached_latent_seq_len(cache_path: str, seq_len_cache: dict[str, int]) -> int:
    """Load cached latent sequence length from a feature `.pt` file once."""
    if cache_path in seq_len_cache:
        return seq_len_cache[cache_path]

    payload = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(payload, dict):
        tensor = payload.get("latent")
        if tensor is None:
            tensor = payload.get("l_features")
    else:
        tensor = payload

    seq_len = int(tensor.shape[0]) if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1 else 0
    seq_len_cache[cache_path] = seq_len
    return seq_len


def build_cot_chunk_token_ids(tokenizer, thinking_chunks: List[str]) -> List[List[int]]:
    """Tokenize pure CoT chunks only, excluding structural special tokens."""
    chunk_token_ids: List[List[int]] = []
    for chunk in thinking_chunks or []:
        normalized = compress_newlines(chunk).strip()
        if not normalized:
            continue
        chunk_token_ids.append(
            tokenizer.encode(normalized, add_special_tokens=False)
        )
    return chunk_token_ids

def extract_thinking_and_answer(
    content: str,
    max_chars: Optional[int] = 4800,
    return_chunks: bool = True,
) -> Tuple[str, str] | Tuple[List[str], str]:
    """
    Extract thinking section and final answer from R1-Onevision format.

    Format: <think> thinking content </think> final answer

    Args:
        content: Assistant message containing thinking tags
        max_chars: Maximum characters per chunk (default: 4800)
        return_chunks: If True (default), return list of chunks.
                      If False, return only first chunk for backward compatibility.

    Returns:
        If return_chunks=True: (list of thinking chunks, answer)
        If return_chunks=False: (first thinking chunk or full thinking, answer)
    """
    # Find opening and closing tags
    open_tag = '<think>'
    close_tag = '</think>'
    
    open_pos = content.find(open_tag)
    if open_pos < 0:
        return "", content.strip()
    
    close_pos = content.find(close_tag, open_pos + len(open_tag))
    if close_pos < 0:
        return "", content.strip()
    
    # Extract content between tags (thinking to be rendered)
    thinking_start = open_pos + len(open_tag)
    thinking_raw = content[thinking_start:close_pos].strip()

    # Answer is everything after closing tag
    answer_start = close_pos + len(close_tag)
    answer_raw = content[answer_start:].strip()

    # IMPORTANT: Do not fall back to CoT-as-answer when post-</think> answer is empty.
    # Empty final answers should be filtered out by callers.
    answer = answer_raw

    # Potential risk that <image> and </image> tags in the answer, remove them
    answer = re.sub(r'<\/?image>', '', answer).strip()

    # Process thinking based on max_chars
    if max_chars is not None and len(thinking_raw) > max_chars:
        # Use intelligent chunking
        thinking_chunks = chunk_thinking_text(thinking_raw, max_chars)

        if return_chunks:
            # Return all chunks as a list
            return thinking_chunks, answer
        else:
            # Return only first chunk (backward compatibility)
            thinking = thinking_chunks[0]
    else:
        # Compress newlines even if no chunking needed
        thinking = compress_newlines(thinking_raw)
        if return_chunks:
            return [thinking], answer

    return thinking, answer


def _load_cutoff_len_from_yaml(config_path: Path) -> Optional[int]:
    if not config_path.exists():
        logger.warning(f"Training config not found for cutoff_len lookup: {config_path}")
        return None

    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
        cutoff_len = data.get("cutoff_len")
        return int(cutoff_len) if cutoff_len is not None else None

    match = re.search(r"(?m)^\s*cutoff_len\s*:\s*(\d+)\s*$", text)
    return int(match.group(1)) if match else None


def _resolve_max_expanded_len(cli_value: Optional[int], train_config: Path) -> int:
    if cli_value is not None:
        return int(cli_value)

    cutoff_len = _load_cutoff_len_from_yaml(train_config)
    if cutoff_len is not None:
        logger.info(f"Resolved max expanded len from {train_config}: cutoff_len={cutoff_len}")
        return cutoff_len

    logger.warning("Falling back to max expanded len = 8192 because cutoff_len could not be resolved.")
    return 8192


def _get_qwen3vl_image_token_count(processor, image_path: str, image_token_cache: Dict[str, int]) -> int:
    cached = image_token_cache.get(image_path)
    if cached is not None:
        return cached

    image = Image.open(image_path).convert("RGB")
    try:
        mm_inputs = processor.image_processor([image], return_tensors="pt")
    finally:
        image.close()

    image_grid_thw = mm_inputs.get("image_grid_thw")
    if image_grid_thw is None or len(image_grid_thw) == 0:
        raise ValueError(f"Failed to compute image_grid_thw for: {image_path}")

    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    image_token_count = int(image_grid_thw[0].prod().item()) // (merge_size ** 2)
    image_token_cache[image_path] = image_token_count
    return image_token_count


def _compose_qwen3vl_training_text(user_content: str, assistant_content: str, image_token_count: int = 0) -> str:
    if image_token_count > 0:
        vision_tokens = "<|vision_start|>" + ("<|image_pad|>" * image_token_count) + "<|vision_end|>"
        user_content = user_content.replace("<image>", vision_tokens, 1)

    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>\n"
    )


def _get_exact_expanded_length(
    tokenizer,
    processor,
    *,
    user_content: str,
    assistant_content: str,
    image_path: Optional[str],
    latent_seq_lens: List[int],
    image_token_cache: Dict[str, int],
) -> int:
    image_token_count = 0
    if image_path:
        image_token_count = _get_qwen3vl_image_token_count(processor, image_path, image_token_cache)

    composed = _compose_qwen3vl_training_text(
        user_content=user_content,
        assistant_content=assistant_content,
        image_token_count=image_token_count,
    )
    base_len = len(tokenizer.encode(composed, add_special_tokens=False))
    num_latent_steps = assistant_content.count("<latent>")
    return base_len - num_latent_steps + sum(int(x) for x in latent_seq_lens)


def _split_text_for_rendering(text: str) -> tuple[str, str]:
    text = str(text or "").strip()
    if not text:
        return "", ""

    lines = text.splitlines()
    if len(lines) >= 4:
        midpoint = len(lines) // 2
        candidates: List[tuple[int, int]] = []
        for idx in range(1, len(lines)):
            stripped = lines[idx].strip()
            prev_stripped = lines[idx - 1].strip()
            boundary_score = 0
            if not prev_stripped or not stripped:
                boundary_score -= 4
            if re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+|[A-Za-z][\.\)]\s+)", stripped):
                boundary_score -= 2
            if re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+|[A-Za-z][\.\)]\s+)", prev_stripped):
                boundary_score -= 1
            candidates.append((abs(idx - midpoint) + boundary_score, idx))
        if candidates:
            _, split_idx = min(candidates)
            left = "\n".join(lines[:split_idx]).strip()
            right = "\n".join(lines[split_idx:]).strip()
            if left and right:
                return left, right

    sentence_split = re.split(r"(?<=[.!?])\s+", text)
    if len(sentence_split) >= 2:
        midpoint = len(sentence_split) // 2
        left = " ".join(sentence_split[:midpoint]).strip()
        right = " ".join(sentence_split[midpoint:]).strip()
        if left and right:
            return left, right

    words = text.split()
    if len(words) >= 2:
        midpoint = len(words) // 2
        left = " ".join(words[:midpoint]).strip()
        right = " ".join(words[midpoint:]).strip()
        if left and right:
            return left, right

    midpoint = len(text) // 2
    return text[:midpoint].strip(), text[midpoint:].strip()


def _ensure_thinking_chunks_fit_renderer(renderer: AdaptiveSkiaRenderer, thinking_chunks: List[str]) -> List[str]:
    queue_chunks = [str(chunk or "").strip() for chunk in thinking_chunks if str(chunk or "").strip()]
    fitted_chunks: List[str] = []

    while queue_chunks:
        chunk = queue_chunks.pop(0)
        raw_width, raw_height, _, _ = renderer.measure_text(chunk, thinking_mode=True)
        if raw_width <= renderer._config.max_size and raw_height <= renderer._config.max_size:
            fitted_chunks.append(chunk)
            continue

        left, right = _split_text_for_rendering(chunk)
        if not left or not right or left == chunk or right == chunk:
            fitted_chunks.append(chunk)
            continue
        queue_chunks = [left, right, *queue_chunks]

    return fitted_chunks


def get_hash_filename(dataset_name: str, sample_id: str, suffix: str = "") -> str:
    """Generate deterministic filename for images and cache.

    Format: {sanitized_dataset}_{sample_id}[_{suffix}]

    Uses sample_id directly (not hashed) for traceability.
    Single underscore between all components.

    Examples:
        get_hash_filename("Geometry3K(MathV360K)", "abc123", "")
            -> "Geometry3K_MathV360K_abc123"
        get_hash_filename("Geometry3K(MathV360K)", "abc123", "question_text")
            -> "Geometry3K_MathV360K_abc123_question_text"
        get_hash_filename("Geometry3K(MathV360K)", "abc123", "thinking_0")
            -> "Geometry3K_MathV360K_abc123_thinking_0"

    Args:
        dataset_name: Dataset name (e.g., "Geometry3K(MathV360K)")
        sample_id: Sample ID (used directly, not hashed)
        suffix: Optional suffix for original, question_text, or thinking chunks

    Returns:
        Filename without extension
    """
    # Replace parentheses to create consistent naming
    sanitized = dataset_name.replace('(', '_').replace(')', '')

    # Single underscore between all components
    if suffix:
        return f"{sanitized}_{sample_id}_{suffix}"
    else:
        return f"{sanitized}_{sample_id}"


def main_render_only(args, renderer):
    """Phase 1: Pre-render all question and thinking images to disk."""
    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Determine datasets
    if args.all:
        datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    elif args.dataset:
        datasets = args.dataset
    else:
        raise ValueError("Either --dataset or --all must be specified")

    logger.info(f"Processing {len(datasets)} datasets")
    logger.info(f"Render batch size: {args.batch_size}")
    logger.info("")

    total_stats = {
        'total_samples': 0,
        'question_images': 0,
        'question_text_images': 0,
        'thinking_chunks': 0,
        'thinking_images': 0,
        'errors': 0,
    }

    for dataset_name in datasets:
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"Skipping {dataset_name} (directory not found)")
            continue

        parquet_files = sorted(dataset_dir.glob('*.parquet'))
        if not parquet_files:
            logger.warning(f"Skipping {dataset_name} (no parquet files)")
            continue
        logger.info(f"Processing {dataset_name} ({len(parquet_files)} shards)...")

        stats = {'total_samples': 0, 'question_images': 0, 'question_text_images': 0, 'thinking_chunks': 0, 'thinking_images': 0, 'errors': 0}

        for shard_idx, parquet_path in enumerate(parquet_files):
            logger.info(f"  Shard {shard_idx + 1}/{len(parquet_files)}: {parquet_path.name}")

            df = pd.read_parquet(parquet_path)
            if args.max_samples:
                df = df.head(args.max_samples)
            batch_size = max(1, int(args.batch_size))
            question_text_buffer: List[Tuple[str, Path]] = []
            thinking_buffer: List[Tuple[str, Path]] = []

            def flush_render_buffer(
                buffer: List[Tuple[str, Path]],
                thinking_mode: bool,
                stats_key: str,
                force: bool = False,
            ) -> None:
                while len(buffer) >= batch_size or (force and buffer):
                    current = buffer[:batch_size]
                    del buffer[:batch_size]
                    texts = [t for t, _ in current]
                    paths = [p for _, p in current]
                    try:
                        images = renderer.render_batch(texts, thinking_mode=thinking_mode)
                        for image, out_path in zip(images, paths):
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(image).save(out_path)
                            stats[stats_key] += 1
                    except Exception:
                        # Fallback to per-item rendering to avoid losing whole batch.
                        for text, out_path in current:
                            try:
                                if renderer.render(text, str(out_path), thinking_mode=thinking_mode):
                                    stats[stats_key] += 1
                            except Exception:
                                stats['errors'] += 1

            for idx, row in df.iterrows():
                try:
                    sample_id = row['id']
                    stats['total_samples'] += 1

                    # Parse conversations
                    conversations = row.get('conversations', [])
                    if hasattr(conversations, 'tolist'):
                        conversations = conversations.tolist()
                    elif not isinstance(conversations, list):
                        conversations = list(conversations) if conversations is not None else []

                    if len(conversations) < 2:
                        continue

                    # Extract question text and thinking
                    question_text = conversations[0]['value']
                    assistant_msg = conversations[1]['value']
                    thinking_chunks, answer = extract_thinking_and_answer(
                        assistant_msg,
                        max_chars=args.max_chars_per_chunk,
                        return_chunks=True,
                    )
                    thinking_chunks = _ensure_thinking_chunks_fit_renderer(renderer, thinking_chunks)
                    if not answer:
                        stats['errors'] += 1
                        continue

                    # 1. Save visual question image
                    filename = get_hash_filename(dataset_name, sample_id, "main")
                    image_path = images_dir / f"{filename}.png"
                    if not image_path.exists():
                        image_col = row.get('image')
                        if isinstance(image_col, str):
                            if ',' in image_col:
                                _, imgstr = image_col.split(',', 1)
                                image_data = base64.b64decode(imgstr)
                            else:
                                image_data = base64.b64decode(image_col)
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(image_path, 'wb') as f:
                                f.write(image_data)
                            stats['question_images'] += 1

                    # 2. Queue question text render
                    question_text_filename = get_hash_filename(dataset_name, sample_id, "question_text")
                    question_text_path = images_dir / f"{question_text_filename}.png"
                    if not question_text_path.exists():
                        question_text_buffer.append((question_text, question_text_path))

                    # 3. Queue thinking chunks render
                    if thinking_chunks:
                        stats['thinking_chunks'] += len(thinking_chunks)
                        for chunk_idx, chunk in enumerate(thinking_chunks):
                            chunk_filename = get_hash_filename(dataset_name, sample_id, f"thinking_{chunk_idx}")
                            chunk_path = images_dir / f"{chunk_filename}.png"
                            if not chunk_path.exists():
                                thinking_buffer.append((chunk, chunk_path))

                    # Flush batched renders
                    flush_render_buffer(question_text_buffer, thinking_mode=False, stats_key='question_text_images')
                    flush_render_buffer(thinking_buffer, thinking_mode=True, stats_key='thinking_images')

                    # Progress logging
                    if stats['total_samples'] % 100 == 0:
                        logger.info(f"  Progress: {stats['total_samples']}/{len(df)} samples, "
                                  f"{stats['question_text_images']} question text, "
                                  f"{stats['thinking_images']} thinking images")
                except Exception as e:
                    stats['errors'] += 1
                    logger.warning(f"  Error: {e}")

            # Flush tail batches for this shard
            flush_render_buffer(question_text_buffer, thinking_mode=False, stats_key='question_text_images', force=True)
            flush_render_buffer(thinking_buffer, thinking_mode=True, stats_key='thinking_images', force=True)

        # Log dataset completion
        logger.info(f"  ✓ {dataset_name}: {stats['total_samples']} samples, "
                  f"{stats['question_text_images']} question text, "
                  f"{stats['thinking_images']} thinking images rendered")

        for key in total_stats:
            total_stats[key] += stats[key]

    logger.info("")
    logger.info("=" * 70)
    logger.info("Rendering Summary:")
    logger.info(f"  Total samples:        {total_stats['total_samples']}")
    logger.info(f"  Question images:      {total_stats['question_images']}")
    logger.info(f"  Question text imgs:   {total_stats['question_text_images']}")
    logger.info(f"  Thinking chunks:      {total_stats['thinking_chunks']}")
    logger.info(f"  Thinking images:      {total_stats['thinking_images']}")
    logger.info(f"  Errors:               {total_stats['errors']}")
    logger.info("=" * 70)
    logger.info(f"\nImages saved to: {images_dir}")
    logger.info("Next step: Run with --encode-only to extract latents")


def main_encode_only_ddp(args):
    """Multi-GPU encode phase - splits samples across GPUs within each dataset."""
    local_rank = args.local_rank
    world_size = args.world_size
    device = f'cuda:{local_rank}'

    # Initialize encoder (device is handled internally by Qwen3VLEncoder)
    encoder = Qwen3VLEncoder(
        model_name_or_path=DEFAULT_MODEL_PATH,
        device=device,
        dtype=torch.bfloat16,
        use_vllm_kernels=True,  # Enable optimized vLLM kernels for speed
    )

    if local_rank == 0:
        logger.info(f"Sample-level splitting across {world_size} GPUs")
        logger.info("")

    # Run dedicated encode-only pipeline in multi-GPU mode.
    return main_encode_only(args, encoder)


def main_encode_only(args, encoder):
    """Phase 2: Batch encode pre-rendered images with cross-attention (OPTIMIZED).

    Multi-GPU: Splits samples across ranks, each encodes only its share.
    """
    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer so encode-only output contains precomputed
    # per-step CoT chunk token ids for training.
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    image_token_cache: Dict[str, int] = {}
    fit_renderer = AdaptiveSkiaRenderer()

    # Get DDP parameters
    local_rank = getattr(args, 'local_rank', 0)
    world_size = getattr(args, 'world_size', 1)

    # Determine datasets
    if args.all:
        datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    elif args.dataset:
        datasets = args.dataset
    else:
        raise ValueError("Either --dataset or --all must be specified")

    if local_rank == 0:
        logger.info(f"Processing {len(datasets)} datasets")
        logger.info(f"Images dir: {images_dir}")
        if world_size > 1:
            logger.info(f"Multi-GPU: {world_size} ranks, sample-level splitting")
        logger.info("")

    total_stats = {
        'total_samples': 0,
        'features_cached': 0,
        'extracted_features': 0,
        'errors': 0,
        'dropped_expanded_len': 0,
    }

    for dataset_name in datasets:
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"Skipping {dataset_name} (directory not found)")
            continue

        parquet_files = sorted(dataset_dir.glob('*.parquet'))
        if not parquet_files:
            logger.warning(f"Skipping {dataset_name} (no parquet files)")
            continue

        if local_rank == 0:
            logger.info(f"Processing {dataset_name} ({len(parquet_files)} shards)...")

        output_jsonl = output_dir / f"r1ov_{dataset_name}_thinking.jsonl"

        # Use rank-specific output file for multi-GPU
        if world_size > 1:
            output_jsonl = output_dir / f"r1ov_{dataset_name}_thinking_rank{local_rank}.jsonl"

        stats = {'total_samples': 0, 'features_cached': 0, 'extracted_features': 0, 'errors': 0, 'dropped_expanded_len': 0}

        for shard_idx, parquet_path in enumerate(parquet_files):
            if local_rank == 0:
                logger.info(f"  Shard {shard_idx + 1}/{len(parquet_files)}: {parquet_path.name}")

            df = pd.read_parquet(parquet_path)
            if args.max_samples:
                df = df.head(args.max_samples)

            # Split samples across ranks for multi-GPU
            if world_size > 1:
                samples_per_rank = (len(df) + world_size - 1) // world_size
                start_idx = local_rank * samples_per_rank
                end_idx = min(start_idx + samples_per_rank, len(df))
                df = df.iloc[start_idx:end_idx]
                if local_rank == 0:
                    logger.info(f"  Split {len(df)} samples across {world_size} GPUs (this rank: {len(df)})")

            file_mode = 'w' if shard_idx == 0 else 'a'

            # Phase 2a: Collect all samples with thinking images
            samples_to_encode = []

            for idx, row in df.iterrows():
                try:
                    sample_id = row['id']
                    stats['total_samples'] += 1

                    conversations = row.get('conversations', [])
                    if hasattr(conversations, 'tolist'):
                        conversations = conversations.tolist()
                    elif not isinstance(conversations, list):
                        conversations = list(conversations) if conversations is not None else []

                    if len(conversations) < 2:
                        logger.info(f"    Sample {idx}: SKIP (conversations < 2)")
                        continue

                    question_text = conversations[0]['value']
                    assistant_msg = conversations[1]['value']

                    # Extract thinking with same params as render phase for consistency
                    thinking, answer = extract_thinking_and_answer(
                        assistant_msg,
                        max_chars=args.max_chars_per_chunk,
                        return_chunks=True,
                    )
                    thinking = _ensure_thinking_chunks_fit_renderer(fit_renderer, thinking)
                    if not answer:
                        logger.info(f"    Sample {idx}: SKIP (empty answer after </think>)")
                        continue

                    filename = get_hash_filename(dataset_name, sample_id, "main")
                    query_image_path = images_dir / f"{filename}.png"

                    if not query_image_path.exists():
                        logger.info(f"    Sample {idx}: SKIP (query image {query_image_path.name} not found)")
                        continue

                    # Collect thinking image paths (question text + thinking chunks)
                    thinking_image_paths = []

                    # 1. Add question text image first
                    question_text_filename = get_hash_filename(dataset_name, sample_id, "question_text")
                    question_text_path = images_dir / f"{question_text_filename}.png"
                    if local_rank == 0:
                        logger.info(f"    Checking {question_text_path.name}: exists={question_text_path.exists()}")
                    if question_text_path.exists():
                        thinking_image_paths.append(str(question_text_path))

                    # 2. Add thinking chunk images
                    # thinking is already a list of chunks from extract_thinking_and_answer(return_chunks=True)
                    if thinking:
                        if local_rank == 0:
                            logger.info(f"    Thinking has {len(thinking)} chunks")
                        for chunk_idx, chunk in enumerate(thinking):
                            chunk_filename = get_hash_filename(dataset_name, sample_id, f"thinking_{chunk_idx}")
                            chunk_path = images_dir / f"{chunk_filename}.png"
                            if local_rank == 0:
                                logger.info(f"    Checking {chunk_path.name}: exists={chunk_path.exists()}")
                            if chunk_path.exists():
                                thinking_image_paths.append(str(chunk_path))

                    if local_rank == 0:
                        logger.info(f"    Total thinking_image_paths: {len(thinking_image_paths)}")
                    if thinking_image_paths:
                        samples_to_encode.append({
                            'sample_id': sample_id,
                            'question_text': question_text,
                            'answer': answer,
                            'query_image_path': str(query_image_path),
                            'query_cache_key': filename,
                            'question_text_path': str(question_text_path),  # Add question_text.pt path
                            'thinking_chunks': thinking,  # Actual thinking text for 'cot' field
                            'thinking_image_paths': thinking_image_paths,  # For encoding (includes question_text + thinking)
                        })

                except Exception as e:
                    stats['errors'] += 1
                    logger.warning(f"  Error processing sample: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

            if local_rank == 0:
                logger.info(f"  Found {len(samples_to_encode)} samples with thinking images")

            # Phase 2b: Collect ALL unique images (query + question + thinking) together
            if local_rank == 0:
                logger.info(f"  Collecting all unique images...")
            all_image_paths = set()

            for sample in samples_to_encode:
                # Add query image
                all_image_paths.add(sample['query_image_path'])
                # Add all thinking images (question text + thinking chunks)
                for path in sample['thinking_image_paths']:
                    all_image_paths.add(path)

            all_image_paths = sorted(all_image_paths)
            if local_rank == 0:
                logger.info(f"  Found {len(all_image_paths)} total unique images")

            # Phase 2c: Batch encode ALL images together (no grouping by resolution)
            unified_cache_dir = images_dir / ".feature_cache"
            unified_cache_dir.mkdir(exist_ok=True)

            batch_size = int(args.batch_size)

            # Filter: only encode images that aren't already cached
            images_to_encode = []
            for img_path in all_image_paths:
                filename = Path(img_path).stem
                cache_path = unified_cache_dir / f"{filename}.pt"
                if not cache_path.exists():
                    images_to_encode.append(img_path)

            if local_rank == 0:
                logger.info(f"  Encoding {len(images_to_encode)}/{len(all_image_paths)} images (rest cached)...")

            seq_len_cache: dict[str, int] = {}
            if not images_to_encode:
                if local_rank == 0:
                    logger.info(f"  ✓ All features already cached!")
            else:
                for i in range(0, len(images_to_encode), batch_size):
                    batch_paths = images_to_encode[i:i+batch_size]
                    batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

                    # Encode batch through ViT + patch merger
                    # Returns unpadded per-image features for true any-resolution efficiency
                    output = encoder.encode_images(batch_images)
                    l_features = output.features  # List[Tensor], each [actual_tokens, 2048]
                    grid_thw = output.grid_thw    # [batch, 3]

                    # Save each with both features + grid_thw
                    for img_path, l_feat, grid in zip(batch_paths, l_features, grid_thw):
                        filename = Path(img_path).stem
                        cache_path = unified_cache_dir / f"{filename}.pt"
                        torch.save({
                            'latent': l_feat.cpu(),  # [actual_tokens, 2048] - no padding!
                            'grid_thw': grid.cpu(),   # [3]
                        }, cache_path)
                        seq_len_cache[str(cache_path)] = int(l_feat.shape[0]) if l_feat.dim() >= 2 else 1
                        stats['features_cached'] += 1

                    if local_rank == 0 and (i + batch_size) % (batch_size * 10) == 0:
                        logger.info(f"    Cached {min(i + batch_size, len(images_to_encode))}/{len(images_to_encode)} features")

            if local_rank == 0:
                logger.info(f"  ✓ All features cached to {unified_cache_dir}")

            # Phase 2d: Process each sample using cached features (NO encoding!)
            if local_rank == 0:
                logger.info(f"  Combining cached features...")
            with open(output_jsonl, file_mode) as f_out:
                for sample in samples_to_encode:
                    try:
                        # Cache paths
                        query_cache_path = str(unified_cache_dir / f"{Path(sample['query_image_path']).stem}.pt")
                        # thinking_image_paths includes question_text + thinking chunks.
                        # For training, latent_ground_truth should contain ONLY thinking chunks (exclude question_text),
                        # and the number of <latent> placeholders must match len(latent_ground_truth).
                        thinking_cache_paths = []
                        for tp in sample["thinking_image_paths"]:
                            stem = Path(tp).stem
                            if "question_text" in stem:
                                continue
                            thinking_cache_paths.append(str(unified_cache_dir / f"{stem}.pt"))

                        num_thinking_chunks = len(thinking_cache_paths)
                        if num_thinking_chunks == 0:
                            # No thinking chunks -> skip (otherwise we would build <think></think> with empty latent GT).
                            continue

                        latent_seq_lens = [
                            load_cached_latent_seq_len(cache_path, seq_len_cache)
                            for cache_path in thinking_cache_paths
                        ]
                        cot_chunk_token_ids = build_cot_chunk_token_ids(
                            tokenizer,
                            sample.get("thinking_chunks") or [],
                        )
                        if len(cot_chunk_token_ids) != num_thinking_chunks:
                            logger.warning(
                                "  Skipping %s: chunk/token count mismatch (%s vs %s)",
                                sample["sample_id"],
                                len(cot_chunk_token_ids),
                                num_thinking_chunks,
                            )
                            continue

                        # latent_ground_truth: One entry per thinking chunk (for injection at <latent>)
                        latent_ground_truth = thinking_cache_paths

                        # latent_supervision: Main question image feature (used by OT/NCE losses)
                        latent_supervision = [query_cache_path]

                        # Reconstruct assistant message with thinking tags and latent placeholders
                        # Format: <think><latent><think_sep><latent>...</think>{answer}
                        # Note: <latent> and <think_sep> are TEXT tokens with CE loss
                        latent_placeholders = '<think_sep>'.join(['<latent>'] * num_thinking_chunks)
                        thinking_content = f"<think>{latent_placeholders}</think>{sample['answer']}"

                        # Write JSONL entry in llamafactory format
                        # Adaptively add <image> token only if question_text doesn't already have one
                        question_text = sample['question_text']
                        if '<image>' in question_text:
                            user_content = question_text
                        else:
                            user_content = f'<image>\n{question_text}'
                        json_entry = {
                            'messages': [
                                {'role': 'user', 'content': user_content},
                                {'role': 'assistant', 'content': thinking_content}
                            ],
                            'images': [sample['query_image_path']],
                            'latent_ground_truth': latent_ground_truth,
                            'latent_supervision': latent_supervision,
                            'num_latent_steps': num_thinking_chunks,
                            # Mirror CE-loss subsequence boundaries with <think_sep>.
                            # This stays a string for backward compatibility with existing JSONL.
                            'cot': format_cot_subsequences(sample.get('thinking_chunks')),
                            # Precomputed one-time metadata for worker-side latent expansion.
                            'latent_seq_lens': latent_seq_lens,
                            'cot_chunk_token_ids': cot_chunk_token_ids,
                            'task': 'r1_onevision_thinking',
                        }
                        expanded_len = _get_exact_expanded_length(
                            tokenizer,
                            processor,
                            user_content=user_content,
                            assistant_content=thinking_content,
                            image_path=sample['query_image_path'],
                            latent_seq_lens=latent_seq_lens,
                            image_token_cache=image_token_cache,
                        )
                        if args.max_expanded_len > 0 and expanded_len > args.max_expanded_len:
                            stats['dropped_expanded_len'] += 1
                            continue

                        f_out.write(json.dumps(json_entry) + '\n')
                        stats['features_cached'] += 1

                        # Progress
                        if local_rank == 0 and stats['features_cached'] % 100 == 0:
                            logger.info(f"    Progress: {stats['features_cached']}/{len(samples_to_encode)} samples combined")

                    except Exception as e:
                        stats['errors'] += 1
                        logger.warning(f"  Error combining {sample['sample_id']}: {e}")

        if local_rank == 0 or world_size == 1:
            logger.info(f"  ✓ {dataset_name}: {stats['extracted_features']} features extracted")

        for key in total_stats:
            total_stats[key] += stats[key]

    # Merge rank files on rank 0 if using multi-GPU
    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()  # Wait for all ranks to finish

        if local_rank == 0:
            logger.info("")
            logger.info("Merging rank files...")
            for dataset_name in datasets:
                base_jsonl = output_dir / f"r1ov_{dataset_name}_thinking.jsonl"
                rank_files = list(output_dir.glob(f"r1ov_{dataset_name}_thinking_rank*.jsonl"))

                if rank_files:
                    # Concatenate all rank files
                    with open(base_jsonl, 'w') as f_out:
                        for rank_file in sorted(rank_files, key=lambda x: int(x.stem.split('rank')[1])):
                            with open(rank_file, 'r') as f_in:
                                f_out.write(f_in.read())
                            # Delete rank file after merging
                            rank_file.unlink()

                    logger.info(f"  ✓ Merged {len(rank_files)} rank files -> {base_jsonl.name}")

    if local_rank == 0:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Encoding Summary:")
        logger.info(f"  Total samples:          {total_stats['total_samples']}")
        logger.info(f"  Features cached:        {total_stats['features_cached']}")
        logger.info(f"  Extracted features:     {total_stats['extracted_features']}")
        logger.info(f"  Errors:                 {total_stats['errors']}")
        logger.info(f"  Dropped by expanded len > {args.max_expanded_len}: {total_stats['dropped_expanded_len']}")
        logger.info("=" * 70)

    # Merge all dataset JSONL files into one (only on rank 0)
    if local_rank == 0 and len(datasets) > 1:
        logger.info("")
        logger.info("Merging all datasets into single JSONL...")

        merged_jsonl = output_dir / "r1ov_thinking.jsonl"
        total_samples = 0

        with open(merged_jsonl, 'w') as f_out:
            for dataset_name in datasets:
                dataset_jsonl = output_dir / f"r1ov_{dataset_name}_thinking.jsonl"
                if dataset_jsonl.exists():
                    sample_count = 0
                    with open(dataset_jsonl, 'r') as f_in:
                        for line in f_in:
                            f_out.write(line)
                            sample_count += 1
                    total_samples += sample_count
                    logger.info(f"  ✓ {dataset_name}: {sample_count} samples")

        logger.info(f"  ✓ Merged {total_samples} total samples -> {merged_jsonl.name}")

        # Delete per-dataset JSONLs after merge
        logger.info("")
        logger.info("Cleaning up per-dataset JSONLs...")
        for dataset_name in datasets:
            dataset_jsonl = output_dir / f"r1ov_{dataset_name}_thinking.jsonl"
            if dataset_jsonl.exists():
                dataset_jsonl.unlink()
                logger.info(f"  ✓ Deleted {dataset_jsonl.name}")

        logger.info("")

def main():
    parser = argparse.ArgumentParser(
        description='Build R1-OneVision Thinking Dataset with Qwen3VL Native Cross-Attention'
    )
    parser.add_argument('--dataset', nargs='+', help='Dataset name(s) to convert')
    parser.add_argument('--all', action='store_true', help='Convert all datasets')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples per dataset (for testing)')
    parser.add_argument('--base-dir',
                        default='/share/project/xiyan/sources/DeepSeek-OCR',
                        help='Base directory of repository')
    parser.add_argument('--data-dir',
                        default='/share/project/xiyan/huggingface/Fancy-MLLM/R1-Onevision',
                        help='Path to R1-Onevision data')
    parser.add_argument('--output-dir',
                        default='Qwen/data',
                        help='Output directory for JSONL files')
    parser.add_argument('--images-dir',
                        default='Qwen/data/r1ov_images',
                        help='Directory for rendered images and latent files')
    parser.add_argument('--train-config',
                        default=str(DEFAULT_TRAIN_CONFIG),
                        help='Training YAML used to derive cutoff_len when --max-expanded-len is not set')
    parser.add_argument('--max-chars-per-chunk', type=int, default=4800,
                        help='Maximum characters per thinking chunk (default: 4800)')
    parser.add_argument('--max-expanded-len', type=int, default=None,
                        help='Drop rows whose exact post-expansion training length exceeds this value. Defaults to training YAML cutoff_len.')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for cross-attention encoder')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Unified batch size for render and encode phases (default: 64)')
    parser.add_argument('--render-only', action='store_true',
                        help='Phase 1: Only render images (CPU-bound, fast)')
    parser.add_argument('--encode-only', action='store_true',
                        help='Phase 2: Only encode pre-rendered images (GPU-bound, batch processing)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use (will launch torchrun automatically if > 1)')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for DDP (set automatically by torchrun)')
    parser.add_argument('--world-size', type=int, default=1,
                        help='World size for DDP (set automatically by torchrun)')

    args = parser.parse_args()
    args.max_expanded_len = _resolve_max_expanded_len(args.max_expanded_len, Path(args.train_config))

    # Auto-launch with torchrun if --num-gpus > 1 and not already in torchrun
    if args.num_gpus > 1 and 'LOCAL_RANK' not in os.environ:
        # Rebuild command line args without --num-gpus
        torchrun_args = [
            'torchrun',
            f'--nproc_per_node={args.num_gpus}',
            sys.argv[0],  # Current script
        ]

        # Add all args except --num-gpus and its value
        skip_next = False
        for i, arg in enumerate(sys.argv[1:]):
            if skip_next:
                skip_next = False
                continue
            if arg == '--num-gpus':
                skip_next = True  # Skip the next arg (the value)
                continue
            if arg.startswith('--num-gpus='):
                continue  # Skip --num-gpus=N format
            torchrun_args.append(arg)

        print(f"Launching {args.num_gpus} GPUs with torchrun...")
        print(f"Command: {' '.join(torchrun_args)}")
        sys.exit(subprocess.call(torchrun_args))

    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir

    # Get DDP parameters from environment (set by torchrun)
    # IMPORTANT: Update args with environment values so main_ddp gets correct values
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.world_size = int(os.environ.get('WORLD_SIZE', args.world_size))

    local_rank = args.local_rank
    world_size = args.world_size

    logger.info("=" * 70)
    logger.info("R1-OneVision Thinking Dataset Builder (Qwen3VL Native)")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Data directory:   {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Max samples:      {args.max_samples or 'All'}")
    logger.info(f"Max chars/chunk:  {args.max_chars_per_chunk}")
    logger.info(f"Max expanded len: {args.max_expanded_len}")
    logger.info(f"Train config:     {args.train_config}")
    logger.info(f"Device:           {args.device}")
    logger.info(f"World size:       {world_size}")
    logger.info(f"Render only:      {args.render_only}")
    logger.info(f"Encode only:      {args.encode_only}")
    logger.info("")

    # Handle two-phase mode
    if args.render_only:
        logger.info("=" * 70)
        logger.info("PHASE 1: Pre-rendering all images (question + thinking)")
        logger.info("=" * 70)
        logger.info("This will render all images to disk without encoding.")
        logger.info("Run with --encode-only after this completes.")
        logger.info("")

        # Rendering is CPU-bound, no GPU benefit from multi-GPU
        # Use single render path regardless of world_size
        renderer = AdaptiveSkiaRenderer()
        return main_render_only(args, renderer)

    if args.encode_only:
        logger.info("=" * 70)
        logger.info("PHASE 2: Batch encoding pre-rendered images")
        logger.info("=" * 70)
        logger.info("This will batch encode all pre-rendered images.")
        logger.info("")

        if world_size > 1:
            # Multi-GPU encode
            if not dist.is_initialized():
                backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                dist.init_process_group(backend=backend)

            if local_rank == 0:
                logger.info("Multi-GPU encode mode initialized")
                logger.info(f"  World size: {world_size}")
                logger.info("")

            return main_encode_only_ddp(args)
        else:
            # Single-GPU encode
            encoder = Qwen3VLEncoder(
                model_name_or_path=DEFAULT_MODEL_PATH,
                device=args.device,
                dtype=torch.bfloat16,
                use_vllm_kernels=True,  # Enable optimized vLLM kernels for speed
            )
            return main_encode_only(args, encoder)

    parser.error("Specify exactly one phase: --render-only or --encode-only")


if __name__ == '__main__':
    sys.exit(main())
