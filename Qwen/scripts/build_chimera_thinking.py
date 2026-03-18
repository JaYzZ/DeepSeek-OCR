#!/usr/bin/env python3
"""
Build CHIMERA text-only dataset in Qwen3VL latent-training format.

Policy:
- question: unchunked, rendered as query image
- solution: chunked (semantic-aware) and rendered as thinking images
- original_solution: unchunked, rendered as latent supervision image

Two-phase workflow:
1) --render-only: render all images and dump metadata JSONL
2) --encode-only: encode all images, cache to .feature_cache, and write final training JSONL
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoTokenizer

script_dir = Path(__file__).parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(repo_root))

from Renderer.skia_renderer import (
    SkiaRenderer,
    measure_finalized_text_canvas,
    prepare_text_for_rendering,
    snap_canvas_to_grid,
)
from Qwen.scripts.utils import chunk_thinking_text, format_cot_subsequences
from OCRVL.encoder.qwen3vl_encoder import Qwen3VLEncoder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking"


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
    """Write PNG via a unique temp file in the target directory, then rename."""
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

    def _get_renderer(
        self,
        width: int,
        height: int,
        padding: int,
        preserve_newlines: bool,
    ) -> SkiaRenderer:
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
        prepared_text, layout_info, padding = self._prepare_render(text, thinking_mode)
        return self._measure_canvas(
            prepared_text,
            padding,
        )

    def _render_one(self, text: str, thinking_mode: bool) -> Any:
        prepared_text, layout_info, padding = self._prepare_render(text, thinking_mode)
        # Skip rendering if layout failed (text too long to fit)
        if not prepared_text or layout_info.get("layout_type") == "failed":
            logger.warning(f"Render failed: layout_type={layout_info.get('layout_type')}, text_len={len(text)}")
            return None
        _, _, width, height = self._measure_canvas(prepared_text, padding)
        return self._get_renderer(
            width,
            height,
            padding,
            layout_info["preserve_newlines"],
        ).render_batch([prepared_text])[0]

    def render_batch(self, texts: List[str], thinking_mode: bool = False) -> List[Any]:
        prepared_specs = []
        for text in texts:
            prepared_text, layout_info, padding = self._prepare_render(text, thinking_mode)
            prepared_specs.append(
                (
                    prepared_text,
                    padding,
                    layout_info["preserve_newlines"],
                )
            )

        grouped: Dict[tuple[int, int, int, bool], List[tuple[int, str]]] = {}
        for idx, (prepared_text, padding, preserve_newlines) in enumerate(prepared_specs):
            _, _, width, height = self._measure_canvas(prepared_text, padding)
            grouped.setdefault((width, height, padding, preserve_newlines), []).append((idx, prepared_text))

        results: List[Any] = [None] * len(texts)
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
            return False  # Skip failed renders
        output_path = Path(output_path)
        _atomic_save_png(image, output_path)
        return True

    def shutdown(self) -> None:
        for renderer in self._renderer_cache.values():
            renderer.shutdown()
        self._renderer_cache.clear()


def load_cached_latent_seq_len(cache_path: str, seq_len_cache: dict[str, int]) -> int:
    cached = seq_len_cache.get(cache_path)
    if cached is not None:
        return cached
    latent = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=False)
    tensor = None
    if isinstance(latent, dict):
        if "l_features" in latent:
            tensor = latent["l_features"]
        elif "latent" in latent:
            tensor = latent["latent"]
    elif isinstance(latent, torch.Tensor):
        tensor = latent
    if tensor is None:
        raise ValueError(f"Unable to resolve latent tensor from cache: {cache_path}")
    seq_len = int(tensor.shape[0]) if tensor.dim() >= 2 else 1
    seq_len_cache[cache_path] = seq_len
    return seq_len


def build_cot_chunk_token_ids(tokenizer, thinking_chunks: List[str]) -> List[List[int]]:
    token_ids: List[List[int]] = []
    for chunk in thinking_chunks or []:
        text = str(chunk or "")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            ids = tokenizer.encode(" ", add_special_tokens=False)
        token_ids.append(ids)
    return token_ids

IMAGE_PROMPT_POOL = [
    "Solve this {topic} question in the image.",
    "Answer this {topic} problem from the image.",
    "Read the image and solve this {topic} question.",
    "Work out this {topic} problem shown in the image.",
    "Find the answer to this {topic} question in the image.",
    "Please solve this {topic} question from the image.",
    "Solve the {topic} problem in this image.",
    "Read and solve this {topic} problem in the image.",
    "Give the answer to this {topic} question in the image.",
    "Solve this image-based {topic} question.",
    "Please answer this {topic} question in the image.",
    "Read the problem image and solve this {topic} question.",
    "Solve this {topic} problem from the question image.",
    "Work through this {topic} question in the image.",
    "Find the solution to this {topic} problem in the image.",
    "Answer the {topic} question shown in the image.",
    "Solve the question image for this {topic} problem.",
    "Please solve this image {topic} question.",
    "Read this {topic} question in the image and answer it.",
    "Solve this {topic} problem shown here.",
    "Give the solution for this {topic} image question.",
    "Answer this {topic} image-based problem.",
    "Solve this {topic} question from the image prompt.",
    "Read and answer this {topic} question in the image.",
    "Find the correct answer for this {topic} question in the image.",
]


def _load_chimera_rows(data_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(data_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train-*.parquet found under: {data_dir}")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    if "index" in df.columns:
        df = df.sort_values("index").reset_index(drop=True)
    return df


def _sample_id(row: Dict[str, Any], row_idx: int) -> str:
    idx = row.get("index", row_idx)
    return f"chimera_{int(idx):07d}"


def _extract_supervision_thinking(text: str) -> str:
    """Keep only the thinking span for latent supervision rendering.

    If the source contains `<think>...</think>{answer}`, strip the answer and
    unwrap the thinking content. Otherwise, if `</think>` exists without an
    opening tag, keep only the prefix before the closing tag.
    """
    text = str(text or "").strip()
    if not text:
        return text

    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    closing_idx = text.find("</think>")
    if closing_idx != -1:
        return text[:closing_idx].strip()

    return text


def _build_image_user_prompt(topic: str, row_index: int) -> str:
    topic = (topic or "").strip()
    if not topic:
        return "Solve the question in the image."
    tmpl = IMAGE_PROMPT_POOL[int(row_index) % len(IMAGE_PROMPT_POOL)]
    return tmpl.format(topic=topic)


def _image_file_valid(path: Path) -> bool:
    """Return True if an existing image file can be fully decoded."""
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def _save_rendered_results(
    results: List[Any],
    output_paths: List[Path],
) -> None:
    for image, out_path in zip(results, output_paths):
        _atomic_save_png(image, out_path)


def _render_pending_paths(
    renderer: AdaptiveSkiaRenderer,
    texts: List[str],
    output_paths: List[Path],
    thinking_mode: bool,
) -> None:
    if not texts:
        return
    rendered = renderer.render_batch(texts, thinking_mode=thinking_mode)
    _save_rendered_results(rendered, output_paths)


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


def _ensure_thinking_chunks_fit_renderer(
    renderer: AdaptiveSkiaRenderer,
    thinking_chunks: List[str],
) -> List[str]:
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


def _build_render_batch(
    renderer: AdaptiveSkiaRenderer,
    rows: List[Dict[str, Any]],
    start_index: int,
    images_dir: Path,
    max_chars_per_chunk: int,
) -> tuple[List[Dict[str, Any]], List[str], List[Path], List[str], List[Path]]:
    metadata_batch: List[Dict[str, Any]] = []
    plain_texts: List[str] = []
    plain_paths: List[Path] = []
    thinking_texts: List[str] = []
    thinking_paths: List[Path] = []

    for offset, row in enumerate(rows):
        ridx = start_index + offset
        question = str(row.get("question") or "").strip()
        solution = str(row.get("solution") or "").strip()
        answer = str(row.get("answer") or "").strip()
        original_solution = _extract_supervision_thinking(row.get("original_solution") or "")
        topic = str(row.get("topic") or "").strip()

        if not question or not solution or not answer or not original_solution:
            continue

        sid = _sample_id(row, ridx)
        question_img = images_dir / f"{sid}_question.png"
        original_solution_img = images_dir / f"{sid}_supervision.png"

        if not _image_file_valid(question_img):
            plain_texts.append(question)
            plain_paths.append(question_img)

        if not _image_file_valid(original_solution_img):
            plain_texts.append(original_solution)
            plain_paths.append(original_solution_img)

        solution_chunks = chunk_thinking_text(solution, max_chars_per_chunk)
        if not solution_chunks:
            solution_chunks = [solution]
        solution_chunks = _ensure_thinking_chunks_fit_renderer(renderer, solution_chunks)

        solution_chunk_images: List[str] = []
        for cidx, chunk in enumerate(solution_chunks):
            chunk_path = images_dir / f"{sid}_thinking_{cidx}.png"
            if not _image_file_valid(chunk_path):
                thinking_texts.append(chunk)
                thinking_paths.append(chunk_path)
            solution_chunk_images.append(str(chunk_path))

        metadata_batch.append(
            {
                "sample_id": sid,
                "index": int(row.get("index", ridx)),
                "topic": topic,
                "question": question,
                "answer": answer,
                "user_content": f"<image>\n{_build_image_user_prompt(topic, int(row.get('index', ridx)))}".strip(),
                "question_image_path": str(question_img),
                "original_solution_image_path": str(original_solution_img),
                "solution_chunks": solution_chunks,
                "solution_chunk_image_paths": solution_chunk_images,
            }
        )

    return metadata_batch, plain_texts, plain_paths, thinking_texts, thinking_paths


def main_render_only(args):
    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir
    metadata_path = output_dir / args.metadata_file

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    df = _load_chimera_rows(data_dir)
    total_rows = len(df)
    if args.max_samples:
        df = df.head(args.max_samples)
        total_rows = len(df)

    logger.info(f"Total samples to process: {total_rows}")

    renderer = AdaptiveSkiaRenderer()

    kept = 0
    skipped = 0

    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        for ridx, row in enumerate(df.to_dict(orient="records")):
            question = str(row.get("question") or "").strip()
            solution = str(row.get("solution") or "").strip()
            answer = str(row.get("answer") or "").strip()
            original_solution = _extract_supervision_thinking(row.get("original_solution") or "")
            topic = str(row.get("topic") or "").strip()

            if not question or not solution or not answer or not original_solution:
                skipped += 1
                continue

            sid = _sample_id(row, ridx)

            question_img = images_dir / f"{sid}_question.png"
            original_solution_img = images_dir / f"{sid}_supervision.png"

            # Render query/question image (unchunked).
            if not _image_file_valid(question_img):
                renderer.render(question, str(question_img), thinking_mode=False)

            # Render supervision/original_solution image (unchunked).
            # Use non-thinking mode for stronger anti-truncation sizing.
            if not _image_file_valid(original_solution_img):
                renderer.render(original_solution, str(original_solution_img), thinking_mode=False)

            # Chunk solution only (semantic-aware, same utility as r1-onevision)
            solution_chunks = chunk_thinking_text(solution, args.max_chars_per_chunk)
            if not solution_chunks:
                solution_chunks = [solution]
            solution_chunks = _ensure_thinking_chunks_fit_renderer(renderer, solution_chunks)

            solution_chunk_images: List[str] = []
            for cidx, chunk in enumerate(solution_chunks):
                p = images_dir / f"{sid}_thinking_{cidx}.png"
                if not _image_file_valid(p):
                    renderer.render(chunk, str(p), thinking_mode=True)
                solution_chunk_images.append(str(p))

            user_content = f"<image>\n{_build_image_user_prompt(topic, int(row.get('index', ridx)))}".strip()
            item = {
                "sample_id": sid,
                "index": int(row.get("index", ridx)),
                "topic": topic,
                "question": question,
                "answer": answer,
                "user_content": user_content,
                "question_image_path": str(question_img),
                "original_solution_image_path": str(original_solution_img),
                "solution_chunks": solution_chunks,
                "solution_chunk_image_paths": solution_chunk_images,
            }
            f_meta.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 50 == 0:
                pct = (kept + skipped) / total_rows * 100 if total_rows > 0 else 0
                logger.info(f"Render progress: {kept + skipped}/{total_rows} ({pct:.1f}%) kept={kept} skipped={skipped}")

    logger.info("=" * 70)
    logger.info("Render summary")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Kept samples: {kept}")
    logger.info(f"Skipped samples: {skipped}")
    logger.info("=" * 70)


def _load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _encode_missing_features(
    encoder: Qwen3VLEncoder,
    image_paths: List[str],
    cache_dir: Path,
    batch_size: int,
) -> dict[str, int]:
    to_encode: List[str] = []
    for p in image_paths:
        stem = Path(p).stem
        cp = cache_dir / f"{stem}.pt"
        if not cp.exists():
            to_encode.append(p)

    logger.info(f"Encoding missing features: {len(to_encode)}/{len(image_paths)}")
    seq_len_cache: dict[str, int] = {}
    for i in range(0, len(to_encode), batch_size):
        batch_paths = to_encode[i : i + batch_size]
        output = _encode_image_batch(encoder, batch_paths, cache_dir)
        seq_len_cache.update(output)
        if (i // batch_size) % 10 == 0:
            logger.info(f"Encode progress: {min(i + batch_size, len(to_encode))}/{len(to_encode)}")
    return seq_len_cache


def _encode_image_batch(
    encoder: Qwen3VLEncoder,
    image_paths: List[str],
    cache_dir: Path,
) -> dict[str, int]:
    if not image_paths:
        return {}

    seq_len_cache: dict[str, int] = {}
    batch_images = [Image.open(p).convert("RGB") for p in image_paths]
    output = encoder.encode_images(batch_images)
    for p, feat, grid in zip(image_paths, output.features, output.grid_thw):
        cp = cache_dir / f"{Path(p).stem}.pt"
        torch.save({"latent": feat.cpu(), "grid_thw": grid.cpu()}, cp)
        seq_len_cache[str(cp)] = int(feat.shape[0]) if feat.dim() >= 2 else 1
    return seq_len_cache


def _get_distributed_info() -> tuple[int, int]:
    """Return (rank, world_size) from torchrun environment."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _resolve_encode_device(args, local_rank: int, world_size: int) -> str:
    """
    Resolve per-rank device for encode stage.
    For multi-GPU torchrun, force rank-local CUDA device mapping.
    """
    device = str(args.device)
    if world_size > 1 and device.startswith("cuda"):
        return f"cuda:{local_rank}"
    return device


def main_encode_only(args):
    rank, world_size = _get_distributed_info()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    base_dir = Path(args.base_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir
    metadata_path = output_dir / args.metadata_file
    out_jsonl_image = output_dir / args.output_jsonl_image
    out_jsonl_text = output_dir / args.output_jsonl_text

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Run --render-only first.")

    items = _load_metadata(metadata_path)
    if args.max_samples:
        items = items[: args.max_samples]

    logger.info(f"[Rank {rank}/{world_size}] Metadata samples: {len(items)}")

    device = _resolve_encode_device(args, local_rank, world_size)
    use_vllm_kernels = str(device).startswith("cuda")
    encoder = Qwen3VLEncoder(
        model_name_or_path=DEFAULT_MODEL_PATH,
        device=device,
        dtype=torch.bfloat16,
        use_vllm_kernels=use_vllm_kernels,
    )

    cache_dir = images_dir / ".feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_paths = set()
    for s in items:
        all_paths.add(s["question_image_path"])
        all_paths.add(s["original_solution_image_path"])
        for p in s.get("solution_chunk_image_paths", []):
            all_paths.add(p)

    all_paths_sorted = sorted(all_paths)
    if world_size > 1:
        # Rank-level sharding: each process encodes a disjoint subset.
        rank_paths = all_paths_sorted[rank::world_size]
        logger.info(f"[Rank {rank}/{world_size}] Assigned image paths: {len(rank_paths)}/{len(all_paths_sorted)}")
    else:
        rank_paths = all_paths_sorted

    seq_len_cache = _encode_missing_features(
        encoder,
        rank_paths,
        cache_dir,
        args.batch_size,
    )

    # Wait until all ranks finish cache generation before rank 0 writes JSONL.
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    if rank != 0:
        logger.info(f"[Rank {rank}/{world_size}] Encode cache complete; rank 0 will write JSONL.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )

    kept = 0
    with open(out_jsonl_image, "w", encoding="utf-8") as f_img, open(out_jsonl_text, "w", encoding="utf-8") as f_txt:
        for s in items:
            solution_cache_paths = [str(cache_dir / f"{Path(p).stem}.pt") for p in s["solution_chunk_image_paths"]]
            if not solution_cache_paths:
                continue

            supervision_cache_path = str(cache_dir / f"{Path(s['original_solution_image_path']).stem}.pt")
            question_image_path = s["question_image_path"]
            question_text = s["question"]

            num_latent_steps = len(solution_cache_paths)
            latent_seq_lens = [
                load_cached_latent_seq_len(cache_path, seq_len_cache)
                for cache_path in solution_cache_paths
            ]
            cot_chunk_token_ids = build_cot_chunk_token_ids(
                tokenizer,
                s.get("solution_chunks") or [],
            )
            if len(cot_chunk_token_ids) != num_latent_steps:
                logger.warning(
                    "Skipping %s: chunk/token count mismatch (%s vs %s)",
                    s.get("sample_id", "unknown"),
                    len(cot_chunk_token_ids),
                    num_latent_steps,
                )
                continue
            latent_placeholders = "<think_sep>".join(["<latent>"] * num_latent_steps)
            assistant_content = f"<think>{latent_placeholders}</think>{s['answer']}"

            cot = format_cot_subsequences(s.get("solution_chunks"))
            common = {
                "latent_ground_truth": solution_cache_paths,
                "latent_supervision": [supervision_cache_path],
                "num_latent_steps": num_latent_steps,
                "cot": cot,
                "latent_seq_lens": latent_seq_lens,
                "cot_chunk_token_ids": cot_chunk_token_ids,
            }

            # Variant 1: question provided via image input.
            item_image = {
                "messages": [
                    {"role": "user", "content": s["user_content"]},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [question_image_path],
                "task": "chimera_thinking_image_input",
                **common,
            }
            f_img.write(json.dumps(item_image, ensure_ascii=False) + "\n")

            # Variant 2: question provided directly as text input.
            item_text = {
                "messages": [
                    {"role": "user", "content": question_text},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [],
                "task": "chimera_thinking_text_input",
                **common,
            }
            f_txt.write(json.dumps(item_text, ensure_ascii=False) + "\n")

            kept += 1
            if kept % 200 == 0:
                logger.info(f"Write progress: {kept}/{len(items)}")

    logger.info("=" * 70)
    logger.info("Encode summary")
    logger.info(f"Output JSONL (image input): {out_jsonl_image}")
    logger.info(f"Output JSONL (text input):  {out_jsonl_text}")
    logger.info(f"Cached features dir: {cache_dir}")
    logger.info(f"Kept samples: {kept}")
    logger.info("=" * 70)


def main_all(args):
    rank, world_size = _get_distributed_info()
    if world_size > 1 or args.num_gpus > 1:
        raise ValueError("--all currently supports only single-process execution.")

    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir
    metadata_path = output_dir / args.metadata_file
    out_jsonl_image = output_dir / args.output_jsonl_image
    out_jsonl_text = output_dir / args.output_jsonl_text
    cache_dir = images_dir / ".feature_cache"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = _load_chimera_rows(data_dir)
    if args.max_samples:
        df = df.head(args.max_samples)
    rows = df.to_dict(orient="records")

    render_queue: queue.Queue = queue.Queue(maxsize=4)
    metadata_items: List[Dict[str, Any]] = []
    producer_error: List[BaseException] = []
    sentinel = object()

    def producer() -> None:
        renderer = AdaptiveSkiaRenderer()
        try:
            for start in range(0, len(rows), args.batch_size):
                row_batch = rows[start:start + args.batch_size]
                metadata_batch, plain_texts, plain_paths, thinking_texts, thinking_paths = _build_render_batch(
                    renderer,
                    row_batch,
                    start,
                    images_dir,
                    args.max_chars_per_chunk,
                )
                _render_pending_paths(renderer, plain_texts, plain_paths, thinking_mode=False)
                _render_pending_paths(renderer, thinking_texts, thinking_paths, thinking_mode=True)
                ready_paths = [
                    path
                    for path in [*plain_paths, *thinking_paths]
                    if path.exists()
                ]
                render_queue.put((metadata_batch, [str(p) for p in ready_paths]))
            render_queue.put(sentinel)
        except BaseException as exc:
            producer_error.append(exc)
            render_queue.put(sentinel)

    device = _resolve_encode_device(args, local_rank=0, world_size=1)
    encoder = Qwen3VLEncoder(
        model_name_or_path=DEFAULT_MODEL_PATH,
        device=device,
        dtype=torch.bfloat16,
        use_vllm_kernels=str(device).startswith("cuda"),
    )

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    seq_len_cache: dict[str, int] = {}
    pending_encode: List[str] = []
    rendered_count = 0

    while True:
        item = render_queue.get()
        if item is sentinel:
            break
        metadata_batch, ready_paths = item
        metadata_items.extend(metadata_batch)
        rendered_count += len(metadata_batch)
        pending_encode.extend(ready_paths)

        while len(pending_encode) >= args.batch_size:
            batch_paths = pending_encode[:args.batch_size]
            del pending_encode[:args.batch_size]
            seq_len_cache.update(
                _encode_image_batch(
                    encoder,
                    batch_paths,
                    cache_dir,
                )
            )

        if rendered_count % 200 == 0:
            logger.info(f"Pipelined progress: rendered={rendered_count}/{len(rows)}")

    thread.join()
    if producer_error:
        raise producer_error[0]

    while pending_encode:
        batch_paths = pending_encode[:args.batch_size]
        del pending_encode[:args.batch_size]
        seq_len_cache.update(
            _encode_image_batch(
                encoder,
                batch_paths,
                cache_dir,
            )
        )

    metadata_items.sort(key=lambda item: item["index"])
    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        for item in metadata_items:
            f_meta.write(json.dumps(item, ensure_ascii=False) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )

    kept = 0
    with open(out_jsonl_image, "w", encoding="utf-8") as f_img, open(out_jsonl_text, "w", encoding="utf-8") as f_txt:
        for s in metadata_items:
            solution_cache_paths = [str(cache_dir / f"{Path(p).stem}.pt") for p in s["solution_chunk_image_paths"]]
            if not solution_cache_paths:
                continue

            supervision_cache_path = str(cache_dir / f"{Path(s['original_solution_image_path']).stem}.pt")
            question_image_path = s["question_image_path"]
            question_text = s["question"]

            num_latent_steps = len(solution_cache_paths)
            latent_seq_lens = [
                load_cached_latent_seq_len(cache_path, seq_len_cache)
                for cache_path in solution_cache_paths
            ]
            cot_chunk_token_ids = build_cot_chunk_token_ids(tokenizer, s.get("solution_chunks") or [])
            if len(cot_chunk_token_ids) != num_latent_steps:
                logger.warning(
                    "Skipping %s: chunk/token count mismatch (%s vs %s)",
                    s.get("sample_id", "unknown"),
                    len(cot_chunk_token_ids),
                    num_latent_steps,
                )
                continue

            latent_placeholders = "<think_sep>".join(["<latent>"] * num_latent_steps)
            assistant_content = f"<think>{latent_placeholders}</think>{s['answer']}"
            common = {
                "latent_ground_truth": solution_cache_paths,
                "latent_supervision": [supervision_cache_path],
                "num_latent_steps": num_latent_steps,
                "cot": format_cot_subsequences(s.get("solution_chunks")),
                "latent_seq_lens": latent_seq_lens,
                "cot_chunk_token_ids": cot_chunk_token_ids,
            }

            item_image = {
                "messages": [
                    {"role": "user", "content": s["user_content"]},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [question_image_path],
                "task": "chimera_thinking_image_input",
                **common,
            }
            f_img.write(json.dumps(item_image, ensure_ascii=False) + "\n")

            item_text = {
                "messages": [
                    {"role": "user", "content": question_text},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [],
                "task": "chimera_thinking_text_input",
                **common,
            }
            f_txt.write(json.dumps(item_text, ensure_ascii=False) + "\n")
            kept += 1

    logger.info("=" * 70)
    logger.info("All-mode summary")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Output JSONL (image input): {out_jsonl_image}")
    logger.info(f"Output JSONL (text input):  {out_jsonl_text}")
    logger.info(f"Cached features dir: {cache_dir}")
    logger.info(f"Kept samples: {kept}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Build CHIMERA thinking dataset for Qwen3VL")
    parser.add_argument("--base-dir", default="/share/project/xiyan/sources/DeepSeek-OCR")
    parser.add_argument("--data-dir", default="/share/project/xiyan/huggingface/TianHongZXY/CHIMERA/Qwen3.5-397B")
    parser.add_argument("--output-dir", default="Qwen/data")
    parser.add_argument("--images-dir", default="Qwen/data/chimera_images")
    parser.add_argument("--metadata-file", default="chimera_qwen35_metadata.jsonl")
    parser.add_argument("--output-jsonl-image", default="chimera_qwen35_thinking_image_input.jsonl")
    parser.add_argument("--output-jsonl-text", default="chimera_qwen35_thinking_text_input.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-chars-per-chunk", type=int, default=16384,
                        help="Chunk size for solution only (semantic-aware).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    if args.num_gpus > 1 and "LOCAL_RANK" not in os.environ:
        cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}", sys.argv[0]]
        skip_next = False
        for a in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if a == "--num-gpus":
                skip_next = True
                continue
            if a.startswith("--num-gpus="):
                continue
            cmd.append(a)
        logger.info("Auto-launching with torchrun: %s", " ".join(cmd))
        return subprocess.call(cmd)

    selected_modes = sum(bool(x) for x in (args.render_only, args.encode_only, args.all))
    if selected_modes != 1:
        parser.error("Specify exactly one phase: --render-only, --encode-only, or --all")

    logger.info("=" * 70)
    logger.info("CHIMERA Thinking Dataset Builder")
    logger.info("=" * 70)
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Images dir: {args.images_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Solution chunk size: {args.max_chars_per_chunk}")
    logger.info(f"Max samples: {args.max_samples or 'All'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    rank, world_size = _get_distributed_info()
    logger.info(f"Distributed: rank={rank} world_size={world_size}")
    logger.info("")

    if args.render_only:
        main_render_only(args)
    elif args.encode_only:
        main_encode_only(args)
    else:
        main_all(args)

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
