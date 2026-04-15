#!/usr/bin/env python3
"""
Unified VQA Dataset Builder

Supports two modes:
1. Standard VQA (no rendering): Text questions, content images, text answers
2. Rendered VQA: Conversation history rendered as images

Usage:
    # Standard VQA (no rendering)
    python OCRVL/scripts/build_vqa.py --mode standard

    # Rendered VQA (with conversation history)
    python OCRVL/scripts/build_vqa.py --mode rendered
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from PIL import Image
from Renderer import VELLO_AVAILABLE, VelloRenderer

# Add repository root to path
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from project_paths import hf_path


VQA_INSTRUCTION_TEMPLATES = [
    "Answer the question:",
    "Respond to the question:",
    "Provide an answer:",
    "Answer based on the image:",
]


_vello_renderer = None


def _get_vello_renderer(image_size=(640, 640), preserve_newlines=False):
    """Get cached Vello renderer instance"""
    global _vello_renderer
    if _vello_renderer is None:
        if VELLO_AVAILABLE:
            _vello_renderer = VelloRenderer(width=image_size[0], height=image_size[1], padding=20, preserve_newlines=preserve_newlines)
            print(f"[Vello] VelloRenderer initialized (size={image_size})")
        else:
            print(f"[Vello] VelloRenderer not available")
            return None
    return _vello_renderer


def render_conversation_text_to_image(text: str, image_size=(640, 640)) -> str | None:
    """Render conversation text to PNG image using Vello renderer."""
    renderer = _get_vello_renderer(image_size=image_size, preserve_newlines=True)
    if renderer is None:
        return None

    try:
        images = renderer.render_batch([text])
        if images and len(images) > 0:
            img_array = images[0]
            pil_image = Image.fromarray(img_array) if isinstance(img_array, np.ndarray) else img_array

            # Save to temp file with hash-based name
            hash_hex = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
            temp_dir = Path(tempfile.gettempdir()) / "vqa_rendered"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{hash_hex}.png"
            pil_image.save(temp_path)
            return str(temp_path)
    except Exception as e:
        print(f"[Vello] Failed to render: {e}")
    return None


def build_standard_vqa_dataset(
    llava_json_path: Path,
    llava_image_dir: Path,
    output_jsonl: Path,
    max_samples: int = 0,
    seed: int = 42,
) -> None:
    """Build standard VQA dataset (NO text rendering)."""
    rng = random.Random(seed)

    print("\n" + "=" * 80)
    print("Building Standard VQA Dataset (NO text rendering)")
    print("=" * 80)

    # Load LLaVA-Instruct conversations
    with open(llava_json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("LLaVA JSON must be a list of examples.")

    rng.shuffle(data)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    skipped_items = 0

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item_idx, item in enumerate(data):
            if max_samples > 0 and total_samples >= max_samples:
                break

            if not isinstance(item, dict):
                skipped_items += 1
                continue

            image = item.get("image") or item.get("img") or item.get("path")
            if not image:
                skipped_items += 1
                continue

            conversations = item.get("conversations")
            if not conversations or not isinstance(conversations, list):
                skipped_items += 1
                continue

            img_path = str(llava_image_dir / image) if not Path(image).is_absolute() else str(image)
            if not Path(img_path).exists():
                skipped_items += 1
                continue

            # Extract Q&A pairs
            qa_pairs = []
            for i in range(0, len(conversations) - 1, 2):
                if i + 1 >= len(conversations):
                    break

                q_turn = conversations[i]
                a_turn = conversations[i + 1]

                if not isinstance(q_turn, dict) or not isinstance(a_turn, dict):
                    continue

                if q_turn.get("from") not in ["human", "user"]:
                    continue

                if a_turn.get("from") not in ["gpt", "assistant"]:
                    continue

                question = q_turn.get("value", "").strip()
                answer = a_turn.get("value", "").strip()

                if not question or not answer:
                    continue

                qa_pairs.append((question, answer))

            if not qa_pairs:
                skipped_items += 1
                continue

            # Generate samples
            for qa_idx, (question, answer) in enumerate(qa_pairs):
                # Build conversation history as text
                if qa_idx == 0:
                    conv_text = question
                else:
                    prev_qa_pairs = qa_pairs[:qa_idx]
                    conv_text_lines = []
                    for prev_q, prev_a in prev_qa_pairs:
                        conv_text_lines.append(f"Q: {prev_q}")
                        conv_text_lines.append(f"A: {prev_a}")
                    conv_text_lines.append(f"Q: {question}")
                    conv_text = "\n".join(conv_text_lines)

                instruction = rng.choice(VQA_INSTRUCTION_TEMPLATES)

                sample = {
                    "messages": [
                        {"role": "user", "content": f"{instruction}\n{conv_text}\n<image>"},
                        {"role": "assistant", "content": answer},
                    ],
                    "images": [img_path],
                    "task": "standard_vqa",
                    "category": "vqa",
                    "qa_index": qa_idx,
                    "total_qa_pairs": len(qa_pairs),
                }

                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_samples += 1

                if total_samples % 10000 == 0:
                    print(f"[Progress] {total_samples} samples generated...")

                if max_samples > 0 and total_samples >= max_samples:
                    break

            if item_idx % 10000 == 0:
                print(f"[Progress] Processed {item_idx}/{len(data)} items, generated {total_samples} samples...")

    print(f"\n{'=' * 80}")
    print(f"✓ Built standard VQA dataset: {output_jsonl}")
    print(f"  Total samples: {total_samples}")
    print(f"  Skipped items: {skipped_items}")
    print(f"  Format: Text questions (NO rendering)")
    print(f"{'=' * 80}")


def build_rendered_vqa_dataset(
    llava_json_path: Path,
    llava_image_dir: Path,
    output_jsonl: Path,
    output_images_dir: Path,
    max_samples: int = 0,
    seed: int = 42,
) -> None:
    """Build rendered VQA dataset (WITH conversation history rendering)."""
    print("\n" + "=" * 80)
    print("Building Rendered VQA Dataset (WITH conversation history rendering)")
    print("=" * 80)

    from collections import defaultdict

    # Initialize renderer
    renderer = _get_vello_renderer()
    if renderer is None:
        print("ERROR: VelloRenderer required for rendered mode")
        return

    # Load LLaVA-Instruct conversations
    with open(llava_json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("LLaVA JSON must be a list of examples.")

    rng = random.Random(seed)
    rng.shuffle(data)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Track rendered images to avoid duplication
    rendered_cache: Dict[str, str] = {}

    print(f"[Processing] Generating samples with rendered conversation history...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        total_samples = 0
        skipped_items = 0

        for item_idx, item in enumerate(data):
            if max_samples > 0 and total_samples >= max_samples:
                break

            if not isinstance(item, dict):
                skipped_items += 1
                continue

            image = item.get("image") or item.get("img") or item.get("path")
            if not image:
                skipped_items += 1
                continue

            conversations = item.get("conversations")
            if not conversations or not isinstance(conversations, list):
                skipped_items += 1
                continue

            img_path = str(llava_image_dir / image) if not Path(image).is_absolute() else str(image)
            if not Path(img_path).exists():
                skipped_items += 1
                continue

            # Extract Q&A pairs
            qa_pairs = []
            for i in range(0, len(conversations) - 1, 2):
                if i + 1 >= len(conversations):
                    break

                q_turn = conversations[i]
                a_turn = conversations[i + 1]

                if not isinstance(q_turn, dict) or not isinstance(a_turn, dict):
                    continue

                if q_turn.get("from") not in ["human", "user"]:
                    continue

                if a_turn.get("from") not in ["gpt", "assistant"]:
                    continue

                question = q_turn.get("value", "").strip()
                answer = a_turn.get("value", "").strip()

                if not question or not answer:
                    continue

                qa_pairs.append((question, answer))

            if not qa_pairs:
                skipped_items += 1
                continue

            # Generate rendered samples
            for qa_idx, (question, answer) in enumerate(qa_pairs):
                # Build conversation history text for rendering
                if qa_idx == 0:
                    conv_text = question
                else:
                    prev_qa_pairs = qa_pairs[:qa_idx]
                    conv_text_lines = []
                    for prev_q, prev_a in prev_qa_pairs:
                        conv_text_lines.append(f"Q: {prev_q}")
                        conv_text_lines.append(f"A: {prev_a}")
                    conv_text_lines.append(f"Q: {question}")
                    conv_text = "\n".join(conv_text_lines)

                # Render conversation as image
                if conv_text not in rendered_cache:
                    rendered_img_path = render_conversation_text_to_image(conv_text)
                    if rendered_img_path:
                        # Copy to output directory with hash name
                        hash_hex = hashlib.sha256(conv_text.encode('utf-8')).hexdigest()[:16]
                        final_img_name = f"conv_{hash_hex}.png"
                        final_img_path = output_images_dir / final_img_name
                        shutil.copy(rendered_img_path, final_img_path)
                        rendered_cache[conv_text] = str(final_img_path)
                    else:
                        # Fallback to text mode if rendering fails
                        conv_text = None
                else:
                    rendered_img_path = rendered_cache[conv_text]

                instruction = rng.choice(VQA_INSTRUCTION_TEMPLATES)

                if conv_text and rendered_img_path:
                    # Rendered mode: content image + rendered conversation
                    sample = {
                        "messages": [
                            {"role": "user", "content": f"{instruction}\n<image>\n<image>"},
                            {"role": "assistant", "content": answer},
                        ],
                        "images": [img_path, str(final_img_path)],
                        "task": "rendered_vqa",
                        "category": "vqa",
                        "qa_index": qa_idx,
                        "total_qa_pairs": len(qa_pairs),
                    }
                else:
                    # Fallback to standard mode
                    sample = {
                        "messages": [
                            {"role": "user", "content": f"{instruction}\n{conv_text}\n<image>"},
                            {"role": "assistant", "content": answer},
                        ],
                        "images": [img_path],
                        "task": "standard_vqa",
                        "category": "vqa",
                        "qa_index": qa_idx,
                        "total_qa_pairs": len(qa_pairs),
                    }

                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_samples += 1

                if total_samples % 10000 == 0:
                    print(f"[Progress] {total_samples} samples generated...")

                if max_samples > 0 and total_samples >= max_samples:
                    break

            if item_idx % 10000 == 0:
                print(f"[Progress] Processed {item_idx}/{len(data)} items, generated {total_samples} samples...")

    print(f"\n{'=' * 80}")
    print(f"✓ Built rendered VQA dataset: {output_jsonl}")
    print(f"  Total samples: {total_samples}")
    print(f"  Skipped items: {skipped_items}")
    print(f"  Rendered images: {len(rendered_cache)}")
    print(f"  Images saved to: {output_images_dir}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified VQA Dataset Builder (Standard + Rendered modes)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "rendered"],
        default="standard",
        help="VQA mode: 'standard' (no rendering) or 'rendered' (with conversation history)"
    )
    parser.add_argument(
        "--llava-json",
        type=str,
        default=str(hf_path("liuhaotian", "LLaVA-Instruct-150K", "llava_v1_5_mix665k.json")),
        help="Path to LLaVA-Instruct JSON file",
    )
    parser.add_argument(
        "--llava-images",
        type=str,
        default=str(hf_path("liuhaotian", "LLaVA-Instruct-150K", "images")),
        help="Path to LLaVA images directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="OCRVL/llamafactory/data/ocrvl_llava_vqa.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--rendered-images-dir",
        type=str,
        default="OCRVL/llamafactory/data/ocrvl_rendered_conversations",
        help="Output directory for rendered conversation images (rendered mode only)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples to generate (0 = unlimited)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    if args.mode == "standard":
        build_standard_vqa_dataset(
            llava_json_path=Path(args.llava_json),
            llava_image_dir=Path(args.llava_images),
            output_jsonl=Path(args.output),
            max_samples=args.max_samples,
            seed=args.seed,
        )
    else:  # rendered
        build_rendered_vqa_dataset(
            llava_json_path=Path(args.llava_json),
            llava_image_dir=Path(args.llava_images),
            output_jsonl=Path(args.output),
            output_images_dir=Path(args.rendered_images_dir),
            max_samples=args.max_samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
