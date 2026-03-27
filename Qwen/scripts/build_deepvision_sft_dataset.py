#!/usr/bin/env python3
"""
Build DeepVision-103K SFT Dataset in LlamaFactory format.

This script formats raw DeepVision parquet files into LlamaFactory-compatible JSONL
for supervised fine-tuning, without rendering or encoding steps.

Input:
- DeepVision parquet files (math-77k.parquet, visual_logic-26k.parquet)

Output JSONL schema:
- messages: [user_message, assistant_message]
- images: [image_path]

The answer is extracted from reward_model.ground_truth field.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image
import io

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-.]+", "_", str(x)).strip("_")


def _to_json_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _extract_prompt_text(prompt: Any, role: str) -> str:
    """Extract text from prompt field by role."""
    if not isinstance(prompt, list):
        return ""
    for m in prompt:
        if isinstance(m, dict) and m.get("role") == role:
            c = m.get("content")
            if isinstance(c, str):
                return c
    return ""


def _extract_first_image_bytes(images_field: Any) -> Optional[bytes]:
    """Extract first image bytes from images field."""
    # Pandas may return list-like image structs as numpy.ndarray (object dtype).
    if hasattr(images_field, "tolist") and not isinstance(images_field, list):
        try:
            images_field = images_field.tolist()
        except Exception:
            pass

    if not isinstance(images_field, list) or not images_field:
        return None

    first = images_field[0]
    if hasattr(first, "as_py"):
        try:
            first = first.as_py()
        except Exception:
            pass
    if not isinstance(first, dict):
        return None

    b = first.get("bytes")
    if b is None:
        return None

    if isinstance(b, bytes):
        return b
    if isinstance(b, memoryview):
        return b.tobytes()
    if isinstance(b, bytearray):
        return bytes(b)
    return None


def _iter_deepvision_rows(data_dir: Path, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    """Load and iterate through all parquet files."""
    parquet_files = sorted(data_dir.glob("*.parquet"))
    rows = []
    for parquet_path in parquet_files:
        split_name = parquet_path.stem
        logger.info(f"Loading {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
        if max_samples:
            df = df.head(max_samples)

        for i, r in df.iterrows():
            item = r.to_dict()
            item["_split"] = split_name
            item["_row_idx"] = int(i)
            rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build DeepVision-103K SFT dataset for LlamaFactory")
    parser.add_argument(
        "--data-dir",
        default="/share/project/xiyan/huggingface/skylenage/DeepVision-103K",
        help="Path to DeepVision parquet files"
    )
    parser.add_argument(
        "--output-dir",
        default="Qwen/data",
        help="Output directory for JSONL file"
    )
    parser.add_argument(
        "--output-jsonl",
        default="deepvision_103k_sft.jsonl",
        help="Output JSONL filename"
    )
    parser.add_argument(
        "--images-output-dir",
        default="Qwen/data/deepvision_images",
        help="Directory to save extracted images"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (default: all)"
    )
    # Images are always extracted for LlamaFactory compatibility
    parser.add_argument(
        "--no-extract-images",
        action="store_true",
        help="Skip image extraction (use base64 encoding instead, not recommended)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_jsonl = output_dir / args.output_jsonl
    images_output_dir = Path(args.images_output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DeepVision-103K SFT Dataset Builder")
    logger.info("=" * 70)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output JSONL: {output_jsonl}")
    logger.info(f"Images output dir: {images_output_dir}")
    logger.info(f"Max samples: {args.max_samples or 'All'}")
    logger.info("")

    # Load all rows from parquet files
    all_rows = _iter_deepvision_rows(data_dir, args.max_samples)
    logger.info(f"Total rows loaded: {len(all_rows)}")
    logger.info("")

    kept = 0
    skipped_no_answer = 0
    skipped_no_image = 0

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, row in enumerate(all_rows):
            split_name = _safe_name(row.get("_split", "unknown"))
            row_idx = int(row.get("_row_idx", idx))
            sample_id = f"deepvision_{split_name}_{row_idx:07d}"

            # Extract question
            question = str(row.get("question") or "").strip()
            prompt = row.get("prompt")
            user_prompt = _extract_prompt_text(prompt, "user")
            if not question:
                question = user_prompt.replace("<image>", "").strip()

            # Extract answer from reward_model.ground_truth
            reward = row.get("reward_model")
            if not isinstance(reward, dict):
                reward = {}
            answer = str(reward.get("ground_truth") or "").strip()

            if not answer:
                skipped_no_answer += 1
                if skipped_no_answer <= 10:
                    logger.info(f"Skipping {sample_id}: no answer in reward_model.ground_truth")
                continue

            # Extract image
            img_bytes = _extract_first_image_bytes(row.get("images"))
            if img_bytes is None:
                skipped_no_image += 1
                if skipped_no_image <= 10:
                    logger.info(f"Skipping {sample_id}: no valid image")
                continue

            # Build user content (always extract images to disk)
            image_path = images_output_dir / f"{sample_id}.jpg"
            if not image_path.exists():
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.save(image_path, "JPEG", quality=95)
                except Exception as e:
                    logger.warning(f"Failed to save image {sample_id}: {e}")
                    skipped_no_image += 1
                    continue
            # Use absolute path for images
            image_path_str = str(image_path.resolve())
            user_content = f"<image>\n{question}"

            # Build messages in LlamaFactory format
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer}
            ]

            # Build output item
            item = {
                "messages": messages,
                "images": [image_path_str]
            }

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 1000 == 0:
                logger.info(f"Progress: {kept} samples written, {skipped_no_answer} skipped (no answer), {skipped_no_image} skipped (no image)")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"Output JSONL: {output_jsonl}")
    logger.info(f"Total samples written: {kept}")
    logger.info(f"Skipped (no answer): {skipped_no_answer}")
    logger.info(f"Skipped (no image): {skipped_no_image}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
