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
import subprocess
import sys
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

from Qwen.scripts.adaptive_vello_renderer import AdaptiveVelloRenderer
from Qwen.scripts.utils import chunk_thinking_text, format_cot_subsequences
from OCRVL.encoder.qwen3vl_encoder import Qwen3VLEncoder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking"

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


def main_render_only(args):
    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir
    metadata_path = output_dir / args.metadata_file

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    df = _load_chimera_rows(data_dir)
    if args.max_samples:
        df = df.head(args.max_samples)

    renderer = AdaptiveVelloRenderer()

    kept = 0
    skipped = 0

    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        for ridx, row in enumerate(df.to_dict(orient="records")):
            question = str(row.get("question") or "").strip()
            solution = str(row.get("solution") or "").strip()
            answer = str(row.get("answer") or "").strip()
            original_solution = str(row.get("original_solution") or "").strip()
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

            if kept % 200 == 0:
                logger.info(f"Render progress: kept={kept} skipped={skipped}")

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


def _encode_missing_features(encoder: Qwen3VLEncoder, image_paths: List[str], cache_dir: Path, batch_size: int):
    to_encode: List[str] = []
    for p in image_paths:
        stem = Path(p).stem
        cp = cache_dir / f"{stem}.pt"
        if not cp.exists():
            to_encode.append(p)

    logger.info(f"Encoding missing features: {len(to_encode)}/{len(image_paths)}")
    for i in range(0, len(to_encode), batch_size):
        batch_paths = to_encode[i : i + batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        output = encoder.encode_images(batch_images)
        l_features = output.features
        grid_thw = output.grid_thw

        for p, feat, grid in zip(batch_paths, l_features, grid_thw):
            stem = Path(p).stem
            cp = cache_dir / f"{stem}.pt"
            torch.save({"latent": feat.cpu(), "grid_thw": grid.cpu()}, cp)

        if (i // batch_size) % 10 == 0:
            logger.info(f"Encode progress: {min(i + batch_size, len(to_encode))}/{len(to_encode)}")


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

    _encode_missing_features(encoder, rank_paths, cache_dir, args.batch_size)

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
            latent_placeholders = "<think_sep>".join(["<latent>"] * num_latent_steps)
            assistant_content = f"<think>{latent_placeholders}</think>{s['answer']}"

            cot = format_cot_subsequences(s.get("solution_chunks"))
            common = {
                "latent_ground_truth": solution_cache_paths,
                "latent_supervision": [supervision_cache_path],
                "num_latent_steps": num_latent_steps,
                "cot": cot,
                "cot_token_ids": tokenizer.encode(cot, add_special_tokens=False),
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
    parser.add_argument("--max-chars-per-chunk", type=int, default=8192,
                        help="Chunk size for solution only (semantic-aware).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
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

    if args.render_only == args.encode_only:
        parser.error("Specify exactly one phase: --render-only or --encode-only")

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
    else:
        main_encode_only(args)

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
