#!/usr/bin/env python3
"""
Build alignment dataset with pre-rendered instruction images.

For alignment stage, instructions are from fixed prompt pools and should be
pre-rendered for faster training. Each sample has 2 images with 50% random ordering.

Tasks:
1. Image captioning (LLaVA-Pretrain)
2. Text OCR (BLIP3o text captions)
3. Document OCR (DocLayNet)

Uses multiprocessing for fast PNG saving (bottleneck is I/O, not rendering).
"""
from __future__ import annotations

import argparse
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
# Instruction Prompt Pools (from blip3o_tasks.py)
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

TEXT_OCR_INSTRUCTIONS = [
    "What does the text say?",
    "Read the text:",
    "Transcribe the text:",
    "What text is shown?",
    "Extract the text:",
    "What is written here?",
    "Read what's written:",
    "Transcribe what you see:",
    "Please transcribe all text in the image.",
]

DOC_OCR_INSTRUCTIONS = [
    "Please transcribe all text in the image.",
    "What text is visible in the document?",
    "Read and transcribe the document content.",
    "Extract all text from this image.",
    "What does the document say?",
]


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


def _render_and_save_caption(args: Tuple[str, Path]) -> Tuple[str, Path]:
    """
    Render caption and save as PNG (runs in worker process).

    Args:
        args: (caption_text, output_path)

    Returns:
        (caption_text, output_path)
    """
    caption, output_path = args
    renderer = _init_renderer()

    img_array = renderer.render_batch([caption])[0]
    img = Image.fromarray(img_array)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    return (caption, str(output_path))


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


# =============================================================================
# Pre-render Instruction Pool
# =============================================================================

def _pre_render_instruction_pool(
    instructions: List[str],
    output_dir: Path,
    pool_name: str
) -> Dict[str, Path]:
    """
    Pre-render all instructions in a pool to PNG files.

    Returns:
        Dict mapping instruction text to file path
    """
    pool_dir = output_dir / pool_name
    instruction_map = {}

    print(f"[Render] Pre-rendering {len(instructions)} instructions for {pool_name}...")

    for i, instruction in enumerate(instructions):
        # Sanitize filename
        safe_name = instruction.replace(" ", "_").replace(":", "").replace("?", "")[:50]
        output_path = pool_dir / f"{i:03d}_{safe_name}.png"

        if not output_path.exists():
            _render_text_to_file(instruction, output_path)

        instruction_map[instruction] = output_path

    print(f"[Render] ✓ Pre-rendered {len(instruction_map)} instructions to {pool_dir}")
    return instruction_map


# =============================================================================
# Dataset Iterators
# =============================================================================

def _llava_pretrain_iter(
    *,
    seed: int,
    json_path: Path,
    image_dir: Path,
    caption_instruction_map: Dict[str, Path],
    ocr_instruction_map: Dict[str, Path],
    caption_temp_dir: Path,
    max_samples: int,
    num_workers: int = 8,
) -> Iterator[Dict[str, Any]]:
    """
    LLaVA-Pretrain: generates TWO samples per input for alignment.

    Sample 1 (Task 1 - Image Captioning):
        - Natural image + rendered instruction → caption
        - 50% random ordering

    Sample 2 (Task 2 - Caption Render OCR):
        - Rendered caption + rendered instruction → caption
        - 50% random ordering

    Uses parallel rendering for captions (bottleneck is PNG I/O).
    """
    rng = random.Random(seed)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("LLaVA-Pretrain JSON must be a list of examples.")

    rng.shuffle(data)

    # First pass: collect all unique captions and render them in parallel
    print("[LLaVA] Collecting unique captions for parallel rendering...")
    unique_captions = {}
    for item in data:
        if max_samples > 0 and len(unique_captions) >= max_samples:
            break
        if not isinstance(item, dict):
            continue

        caption = item.get("caption") or item.get("text")
        if not caption and isinstance(item.get("conversations"), list):
            conv = item["conversations"]
            for turn in reversed(conv):
                if isinstance(turn, dict) and str(turn.get("from")) in {"gpt", "assistant"} and "value" in turn:
                    caption = turn["value"]
                    break

        caption = str(caption).strip() if caption is not None else ""
        if caption and caption not in unique_captions:
            caption_hash = hash(caption) % 1000000000
            caption_path = caption_temp_dir / f"caption_{caption_hash}.png"
            unique_captions[caption] = caption_path

    print(f"[LLaVA] Found {len(unique_captions)} unique captions, rendering with {num_workers} workers...")

    # Check which captions need rendering
    captions_to_render = []
    caption_path_map = {}
    for caption, path in unique_captions.items():
        if not path.exists():
            captions_to_render.append((caption, path))
        caption_path_map[caption] = str(path)

    # Render captions in parallel
    if captions_to_render:
        print(f"[LLaVA] Rendering {len(captions_to_render)} captions...")
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_renderer) as executor:
            futures = {executor.submit(_render_and_save_caption, args): args[0] for args in captions_to_render}
            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 1000 == 0:
                    print(f"[LLaVA] Rendered {i+1}/{len(captions_to_render)} captions...")
                future.result()  # Raise exceptions if any
        print(f"[LLaVA] ✓ Finished rendering captions")

    # Second pass: yield samples
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
        # Natural image + rendered instruction → caption
        # =====================================================================
        instruction = rng.choice(IMAGE_CAPTION_INSTRUCTIONS)
        instruction_img_path = str(caption_instruction_map[instruction])

        instruction_first = rng.random() < 0.5
        if instruction_first:
            images = [instruction_img_path, img_path]
        else:
            images = [img_path, instruction_img_path]

        yield {
            "messages": [
                {"role": "user", "content": "<image><image>"},
                {"role": "assistant", "content": caption},
            ],
            "images": [{"path": p, "bytes": None} for p in images],
            "task": "image_caption",
        }
        emitted += 1

        # =====================================================================
        # Sample 2: Task 2 - Caption Render OCR
        # Rendered caption + rendered instruction → caption
        # =====================================================================
        caption_img_path = caption_path_map[caption]

        instruction = rng.choice(TEXT_OCR_INSTRUCTIONS)
        instruction_img_path = str(ocr_instruction_map[instruction])

        instruction_first = rng.random() < 0.5
        if instruction_first:
            images = [instruction_img_path, str(caption_img_path)]
        else:
            images = [str(caption_img_path), instruction_img_path]

        yield {
            "messages": [
                {"role": "user", "content": "<image><image>"},
                {"role": "assistant", "content": caption},
            ],
            "images": [{"path": p, "bytes": None} for p in images],
            "task": "caption_render_ocr",
        }
        emitted += 1


def _doclaynet_iter(
    *,
    doc_root: Path,
    split: str,
    seed: int,
    max_chars: int,
    max_samples: int,
    instruction_map: Dict[str, Path],
) -> Iterator[Dict[str, Any]]:
    """DocLayNet: document OCR with pre-rendered instructions."""
    rng = random.Random(seed)

    coco_path = doc_root / "COCO" / f"{split}.json"
    png_dir = doc_root / "PNG"
    json_dir = doc_root / "JSON"
    has_text = json_dir.exists()

    if not coco_path.exists():
        raise FileNotFoundError(f"DocLayNet COCO file not found: {coco_path}")
    if not png_dir.exists():
        raise FileNotFoundError(f"DocLayNet PNG directory not found: {png_dir}")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    if not isinstance(images, list):
        raise ValueError("DocLayNet COCO JSON missing `images` list.")

    rng.shuffle(images)
    emitted = 0

    for image_info in images:
        if max_samples > 0 and emitted >= max_samples:
            break
        if not isinstance(image_info, dict):
            continue

        file_name = image_info.get("file_name")
        if not file_name:
            continue

        image_path = png_dir / file_name
        if not image_path.exists():
            continue

        image_hash = Path(file_name).stem
        doc_category = str(image_info.get("doc_category", "document")).replace("_", " ").title()
        doc_name = str(image_info.get("doc_name", "unknown"))
        page_no = int(image_info.get("page_no", 0))
        fallback = f"Document: {doc_category} | Source: {doc_name} | Page: {page_no}"

        caption = _load_doclaynet_caption(json_dir, image_hash, fallback, max_chars) if has_text else fallback[:max_chars]

        # Random instruction from pool
        instruction = rng.choice(DOC_OCR_INSTRUCTIONS)
        instruction_img_path = str(instruction_map[instruction])

        # 50% random ordering
        instruction_first = rng.random() < 0.5

        if instruction_first:
            images = [instruction_img_path, str(image_path)]
        else:
            images = [str(image_path), instruction_img_path]

        yield {
            "messages": [
                {"role": "user", "content": "<image><image>"},
                {"role": "assistant", "content": caption},
            ],
            "images": [{"path": p, "bytes": None} for p in images],
            "task": "doc_ocr",
        }
        emitted += 1


def _load_doclaynet_caption(json_dir: Path, image_hash: str, fallback: str, max_chars: int) -> str:
    """Load caption from DocLayNet JSON annotation."""
    json_path = json_dir / f"{image_hash}.json"
    if not json_path.exists():
        return fallback[:max_chars]

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        cells = data.get("cells", [])
        if isinstance(cells, list):
            parts = []
            for cell in cells:
                if isinstance(cell, dict) and "text" in cell:
                    t = str(cell["text"]).strip()
                    if t:
                        parts.append(t)
            text = " ".join(parts).strip()
            if text:
                return text[:max_chars]
    except Exception:
        pass

    return fallback[:max_chars]


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_jsonl", default="OCRVL/llamafactory/data/ocrvl_alignment_prerendered.jsonl")
    ap.add_argument("--instruction_images_dir", default="OCRVL/llamafactory/data/instruction_images")
    ap.add_argument("--caption_images_dir", default="OCRVL/llamafactory/data/caption_images")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--doc_ratio", type=float, default=0.5)

    ap.add_argument("--llava_pretrain_json",
                    default="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json")
    ap.add_argument("--llava_pretrain_image_dir",
                    default="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Pretrain/images")

    ap.add_argument("--doclaynet_root", default="/share/project/xiyan/huggingface/docling-project/DocLayNet")
    ap.add_argument("--doclaynet_split", default="train")
    ap.add_argument("--doclaynet_max_chars", type=int, default=2048)

    args = ap.parse_args()

    out_jsonl = Path(args.output_jsonl)
    instruction_dir = Path(args.instruction_images_dir)
    caption_dir = Path(args.caption_images_dir)

    # Pre-render all instruction pools
    print("=" * 80)
    print("Pre-rendering instruction pools...")
    print("=" * 80)

    caption_instructions = _pre_render_instruction_pool(
        IMAGE_CAPTION_INSTRUCTIONS,
        instruction_dir,
        "image_caption"
    )

    text_ocr_instructions = _pre_render_instruction_pool(
        TEXT_OCR_INSTRUCTIONS,
        instruction_dir,
        "text_ocr"
    )

    doc_instructions = _pre_render_instruction_pool(
        DOC_OCR_INSTRUCTIONS,
        instruction_dir,
        "doc_ocr"
    )

    # Build dataset
    print("=" * 80)
    print("Building alignment dataset with pre-rendered instructions...")
    print("=" * 80)

    llava_iter = _llava_pretrain_iter(
        seed=args.seed,
        json_path=Path(args.llava_pretrain_json),
        image_dir=Path(args.llava_pretrain_image_dir),
        caption_instruction_map=caption_instructions,  # For task 1
        ocr_instruction_map=text_ocr_instructions,    # For task 2
        caption_temp_dir=caption_dir,
        max_samples=args.max_samples,
    )

    doc_iter = _doclaynet_iter(
        doc_root=Path(args.doclaynet_root),
        split=args.doclaynet_split,
        seed=args.seed,
        max_chars=args.doclaynet_max_chars,
        max_samples=args.max_samples,
        instruction_map=doc_instructions,  # For task 3
    )

    # Combine datasets
    # Note: LLaVA generates 2 samples per input (task 1 + task 2), DocLayNet generates 1
    if args.max_samples <= 0:
        def _rows_full():
            yield from llava_iter
            yield from doc_iter
        rows = _rows_full()
    else:
        # Simple interleaving
        rows = []
        rng = random.Random(args.seed)
        for _ in range(args.max_samples):
            if rng.random() < args.doc_ratio:
                try:
                    rows.append(next(doc_iter))
                except StopIteration:
                    try:
                        rows.append(next(llava_iter))
                    except StopIteration:
                        break
            else:
                try:
                    rows.append(next(llava_iter))
                except StopIteration:
                    try:
                        rows.append(next(doc_iter))
                    except StopIteration:
                        break

    # Write JSONL
    n = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print("=" * 80)
    print(f"✓ Wrote {n} examples to {out_jsonl}")
    print(f"  - Task 1 (Image Caption): Generated from LLaVA-Pretrain")
    print(f"  - Task 2 (Caption Render OCR): Generated from LLaVA-Pretrain")
    print(f"  - Task 3 (Doc OCR): Generated from DocLayNet")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
