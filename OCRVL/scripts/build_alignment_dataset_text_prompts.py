#!/usr/bin/env python3
"""
Build alignment dataset with text prompts (minimal instruction format).

Format changes:
- Caption/OCR: Text prompt + single image (e.g., "Describe the image: <image>")
- VQA: Text template + 2 images (content image + rendered question image)

This is simpler than the pre-rendered instruction format and reduces storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

# Add repository root to path for Renderer import
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# Instruction Prompt Pools (TEXT, not pre-rendered)
# =============================================================================

IMAGE_CAPTION_INSTRUCTIONS = [
    "Describe the image:",
    "What's in this image?",
    "Describe what you see:",
    "What does this image show?",
    "Provide a description of the image:",
    "What do you see in the image?",
    "Can you describe this image?",
    "Tell me about this image:",
]

# Unified OCR instruction prompts for full image OCR
OCR_INSTRUCTIONS = [
    "Free OCR",
    "Read all text in the image",
    "Transcribe the text",
    "Read and transcribe the document content",
    "Extract all text from the image",
]

# VQA question templates (will be rendered as images)
VQA_QUESTION_TEMPLATES = [
    "How many {objects} are visible in this image?",
    "What objects are in the foreground of this image?",
    "What is happening in this image?",
    "Describe the main object in this image:",
    "What type of setting is this?",
    "What are the main colors in this image?",
    "Where is the main object located?",
]

# VQA text prompt template
VQA_TEXT_PROMPT = "Answer the question shown in the images:"


# =============================================================================
# Vello Renderer (per-process)
# =============================================================================

_vello_renderer = None


def _init_renderer():
    """Initialize Vello renderer in this process."""
    global _vello_renderer
    if _vello_renderer is None:
        from Renderer import VelloRenderer, VELLO_AVAILABLE
        if VELLO_AVAILABLE:
            _vello_renderer = VelloRenderer(width=640, height=640, padding=20)
    return _vello_renderer


def _get_vello_renderer(image_size=(640, 640)):
    """Get cached Vello renderer instance"""
    global _vello_renderer
    if _vello_renderer is None:
        try:
            from Renderer import VelloRenderer, VELLO_AVAILABLE
            if VELLO_AVAILABLE:
                _vello_renderer = VelloRenderer(width=image_size[0], height=image_size[1], padding=20)
                print(f"[Render] VelloRenderer initialized: {image_size}")
        except Exception as e:
            print(f"[Render] Failed to initialize VelloRenderer: {e}")
            raise
    return _vello_renderer


def _render_text_to_file(text: str, output_path: Path, image_size=(640, 640)) -> Path:
    """Render text to image file using Vello."""
    renderer = _get_vello_renderer(image_size)
    img_array = renderer.render_batch([text])[0]
    img = Image.fromarray(img_array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _deterministic_hash(text: str) -> int:
    """Generate deterministic hash for text.

    Uses MD5 instead of Python's hash() to ensure same text always produces
    the same hash across different Python sessions (Python's hash() is
    randomized by default since Python 3.3).

    Args:
        text: The text to hash

    Returns:
        Integer hash value (0 to 9999999)
    """
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 10000000


def _render_and_save(args: Tuple[str, Path]) -> Tuple[str, Path]:
    """Render question and save as PNG (runs in worker process)."""
    question, output_path = args
    renderer = _init_renderer()
    img_array = renderer.render_batch([question])[0]
    img = Image.fromarray(img_array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return (question, str(output_path))


# =============================================================================
# Pre-render VQA Questions Pool
# =============================================================================

def _pre_render_vqa_questions(
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Pre-render VQA question templates to PNG files.

    Returns:
        Dict mapping question text to file path
    """
    question_dir = output_dir / "vqa_questions"
    question_map = {}

    print(f"[Render] Pre-rendering {len(VQA_QUESTION_TEMPLATES)} VQA questions...")

    for i, question in enumerate(VQA_QUESTION_TEMPLATES):
        safe_name = f"question_{i:03d}.png"
        output_path = question_dir / safe_name

        if not output_path.exists():
            _render_text_to_file(question, output_path)

        question_map[question] = output_path

    print(f"[Render] ✓ Pre-rendered {len(question_map)} VQA questions to {question_dir}")
    return question_map


# =============================================================================
# Dataset Iterators
# =============================================================================

def _llava_pretrain_iter(
    *,
    seed: int,
    json_path: Path,
    image_dir: Path,
    max_samples: int,
) -> Iterator[Dict[str, Any]]:
    """
    LLaVA-Pretrain: generates TWO samples per input for alignment.

    Sample 1 (Task 1 - Image Captioning):
        - Text prompt + natural image → caption
        - Format: "Describe the image: <image>"

    Sample 2 (Task 2 - Caption Render OCR):
        - Text prompt + rendered caption → caption
        - Format: "Read the text: <image>" (caption is rendered as image)

    Sample 3 (Task 3 - VQA, optional):
        - Text template + content image + question image → answer
        - Format: "Answer the question: <image><image>"
    """
    rng = random.Random(seed)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("LLaVA-Pretrain JSON must be a list of examples.")

    rng.shuffle(data)

    # ========================================================================
    # Phase 1: Scan existing caption cache for fast lookups
    # ========================================================================
    caption_temp_dir = Path("OCRVL/llamafactory/data/temp_captions")
    existing_caption_hashes = set()

    if caption_temp_dir.exists():
        for f in caption_temp_dir.glob("caption_*.png"):
            # Extract hash number from filename: caption_1234567.png -> 1234567
            hash_num = int(f.stem.split("_")[1])
            existing_caption_hashes.add(hash_num)

    print(f"[LLaVA] Found {len(existing_caption_hashes)} existing rendered captions in cache")
    print("[LLaVA] Generating dataset samples...")
    emitted = 0

    for item in data:
        if max_samples > 0 and emitted >= max_samples:
            break
        if not isinstance(item, dict):
            continue

        image = item.get("image") or item.get("img") or item.get("path")
        if not image:
            continue

        caption = item.get("caption") or item.get("text")
        if not caption and isinstance(item.get("conversations"), list):
            conv = item["conversations"]
            for turn in reversed(conv):
                if isinstance(turn, dict) and str(turn.get("from")) in {"gpt", "assistant"} and "value" in turn:
                    caption = turn["value"]
                    break

        caption = str(caption).strip() if caption is not None else ""
        if not caption:
            continue

        img_path = str(image_dir / image) if not Path(image).is_absolute() else str(image)

        # =====================================================================
        # Sample 1: Task 1 - Image Captioning
        # Text prompt + natural image → caption
        # =====================================================================
        instruction = rng.choice(IMAGE_CAPTION_INSTRUCTIONS)
        prompt = f"{instruction} <image>"

        yield {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": caption},
            ],
            "images": [img_path],  # List of strings (consistent with VQA datasets)
            "task": "image_caption",
        }
        emitted += 1

        # =====================================================================
        # Sample 2: Task 2 - Caption Render OCR
        # Text prompt + rendered caption → caption
        # =====================================================================
        # Render caption as image using deterministic hash for caching
        caption_hash = _deterministic_hash(caption)
        caption_img_path = caption_temp_dir / f"caption_{caption_hash}.png"

        # OPTIMIZATION: Use in-memory cache set lookup instead of filesystem
        if caption_hash not in existing_caption_hashes:
            caption_img_path.parent.mkdir(parents=True, exist_ok=True)
            _render_text_to_file(caption, caption_img_path)
            existing_caption_hashes.add(caption_hash)  # Track newly rendered

        ocr_instruction = rng.choice(OCR_INSTRUCTIONS)
        ocr_prompt = f"{ocr_instruction} <image>"

        yield {
            "messages": [
                {"role": "user", "content": ocr_prompt},
                {"role": "assistant", "content": caption},
            ],
            "images": [str(caption_img_path)],  # List of strings (consistent with VQA datasets)
            "task": "caption_render_ocr",
        }
        emitted += 1


def _doclaynet_iter(
    *,
    seed: int,
    json_path: Path,
    image_dir: Path,
    max_samples: int,
) -> Iterator[Dict[str, Any]]:
    """
    DocLayNet: document OCR with text prompts.

    Format: "Extract all text from this image: <image>" → document text
    """
    rng = random.Random(seed)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rng.shuffle(data)

    print("[DocLayNet] Generating dataset samples...")
    emitted = 0

    for item in data:
        if max_samples > 0 and emitted >= max_samples:
            break
        if not isinstance(item, dict):
            continue

        image = item.get("image") or item.get("path") or item.get("png_filename")
        if not image:
            continue

        # Extract text from various fields
        text = (
            item.get("text") or
            item.get("description") or
            (item.get("segments", [{}])[0].get("text") if item.get("segments") else None)
        )

        if not text:
            continue

        text = str(text).strip()

        img_path = str(image_dir / image) if not Path(image).is_absolute() else str(image)
        if not Path(img_path).exists():
            continue

        instruction = rng.choice(OCR_INSTRUCTIONS)
        prompt = f"{instruction} <image>"

        yield {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
            ],
            "images": [img_path],  # List of strings (consistent with VQA datasets)
            "task": "document_ocr",
        }
        emitted += 1


# =============================================================================
# Main Builder
# =============================================================================

def build_alignment_dataset(
    llava_json_path: Path,
    llava_image_dir: Path,
    doclaynet_json_path: Optional[Path],
    doclaynet_image_dir: Optional[Path],
    output_jsonl: Path,
    max_samples: int = 100000,
    seed: int = 42,
):
    """
    Build alignment dataset with text prompts (minimal instruction format).
    """
    rng = random.Random(seed)

    # Build dataset
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0

    with open(output_jsonl, "w", encoding="utf-8") as f:
        # LLaVA-Pretrain samples
        print("\n" + "="*80)
        print("Building LLaVA-Pretrain alignment dataset...")
        print("="*80)

        for sample in _llava_pretrain_iter(
            seed=seed,
            json_path=llava_json_path,
            image_dir=llava_image_dir,
            max_samples=max_samples,
        ):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            total_samples += 1
            if total_samples % 1000 == 0:
                print(f"[Progress] {total_samples} samples generated...")

        # DocLayNet samples
        if doclaynet_json_path and doclaynet_image_dir:
            print("\n" + "="*80)
            print("Building DocLayNet document OCR dataset...")
            print("="*80)

            for sample in _doclaynet_iter(
                seed=seed,
                json_path=doclaynet_json_path,
                image_dir=doclaynet_image_dir,
                max_samples=max_samples // 10,  # Fewer doc samples
            ):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_samples += 1
                if total_samples % 1000 == 0:
                    print(f"[Progress] {total_samples} samples generated...")

    print(f"\n{'='*80}")
    print(f"✓ Built alignment dataset: {output_jsonl}")
    print(f"  Total samples: {total_samples}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build alignment dataset with text prompts"
    )
    parser.add_argument(
        "--llava-json",
        type=str,
        default="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        help="Path to LLaVA JSON file",
    )
    parser.add_argument(
        "--llava-images",
        type=str,
        default="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Pretrain/images",
        help="Path to LLaVA images directory",
    )
    parser.add_argument(
        "--doclaynet-json",
        type=str,
        default=None,
        help="Path to DocLayNet JSON file",
    )
    parser.add_argument(
        "--doclaynet-images",
        type=str,
        default=None,
        help="Path to DocLayNet images directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="OCRVL/llamafactory/data/ocrvl_alignment_text_prompts.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    build_alignment_dataset(
        llava_json_path=Path(args.llava_json),
        llava_image_dir=Path(args.llava_images),
        doclaynet_json_path=Path(args.doclaynet_json) if args.doclaynet_json else None,
        doclaynet_image_dir=Path(args.doclaynet_images) if args.doclaynet_images else None,
        output_jsonl=Path(args.output),
        max_samples=args.max_samples,
        seed=args.seed,
    )
