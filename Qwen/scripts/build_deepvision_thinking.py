#!/usr/bin/env python3
"""
Build DeepVision-103K Thinking Dataset in Qwen3VL latent-training format.

Input:
- DeepVision parquet files (math-77k.parquet, visual_logic-26k.parquet)

Output JSONL schema (aligned with r1_onevision_thinking.jsonl):
- messages
- images
- latent_ground_truth
- latent_supervision
- num_latent_steps
- cot
- cot_token_ids
- task

Two-phase workflow:
1) --render-only: extract rows, save query images + render thinking chunks to PNG,
   and write metadata JSONL.
2) --encode-only: encode all PNGs with Qwen3VL encoder, cache .pt features,
   and write final training JSONL.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
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
    if not isinstance(prompt, list):
        return ""
    for m in prompt:
        if isinstance(m, dict) and m.get("role") == role:
            c = m.get("content")
            if isinstance(c, str):
                return c
    return ""


def _extract_first_image_bytes(images_field: Any) -> Optional[bytes]:
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


def _build_thinking_text(row: Dict[str, Any], source: str) -> str:
    question = str(row.get("question") or "").strip()
    prompt = row.get("prompt")
    system_text = _extract_prompt_text(prompt, "system")

    if source == "question":
        return question
    if source == "system":
        return system_text
    if source == "annotation":
        parts = []
        for k in ("q32-vision-anno", "gpt5-mini-vision-anno", "domain-anno"):
            if k in row and row.get(k) is not None:
                txt = _to_json_text(row.get(k)).strip()
                if txt:
                    parts.append(txt)
        return "\n".join(parts)
    if source == "auto":
        ann = _build_thinking_text(row, "annotation")
        if ann:
            return ann
        if question:
            return question
        return system_text
    return ""


def _iter_deepvision_rows(data_dir: Path, max_samples: Optional[int]) -> List[Dict[str, Any]]:
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


def main_render_only(args):
    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir
    metadata_path = output_dir / args.metadata_file

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    renderer = AdaptiveVelloRenderer()

    all_rows = _iter_deepvision_rows(data_dir, args.max_samples)
    logger.info(f"Rows loaded: {len(all_rows)}")

    batch_size = max(1, int(args.batch_size))
    logger.info(f"Render batch size: {batch_size}")

    render_buffer: List[tuple[str, Path]] = []

    def flush_render_buffer(force: bool = False) -> None:
        while len(render_buffer) >= batch_size or (force and render_buffer):
            current = render_buffer[:batch_size]
            del render_buffer[:batch_size]
            for text, out_path in current:
                try:
                    renderer.render(text, str(out_path), thinking_mode=True)
                except Exception:
                    continue

    kept = 0
    skipped = 0
    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        for idx, row in enumerate(all_rows):
            split_name = _safe_name(row.get("_split", "unknown"))
            row_idx = int(row.get("_row_idx", idx))
            sample_id = f"deepvision_{split_name}_{row_idx:07d}"

            question = str(row.get("question") or "").strip()
            prompt = row.get("prompt")
            user_prompt = _extract_prompt_text(prompt, "user")
            if not question:
                question = user_prompt.replace("<image>", "").strip()

            reward = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
            answer = str((reward or {}).get("ground_truth") or "").strip()
            if not answer:
                skipped += 1
                continue

            img_bytes = _extract_first_image_bytes(row.get("images"))
            if img_bytes is None:
                skipped += 1
                continue

            query_name = f"{sample_id}_main"
            query_path = images_dir / f"{query_name}.png"
            if not query_path.exists():
                try:
                    import io

                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.save(query_path)
                except Exception:
                    skipped += 1
                    continue

            thinking_text = _build_thinking_text(row, args.thinking_source)
            thinking_chunks = chunk_thinking_text(thinking_text, args.max_chars_per_chunk)
            if not thinking_chunks:
                thinking_chunks = [question[: max(1, args.max_chars_per_chunk)].strip() or "Question context"]

            thinking_image_paths = []
            for cidx, chunk in enumerate(thinking_chunks):
                chunk_name = f"{sample_id}_thinking_{cidx}"
                chunk_path = images_dir / f"{chunk_name}.png"
                if not chunk_path.exists():
                    render_buffer.append((chunk, chunk_path))
                thinking_image_paths.append(str(chunk_path))

            # Flush render queue in batches so generated images exist before metadata write.
            flush_render_buffer(force=False)

            if not thinking_image_paths:
                skipped += 1
                continue

            if "<image>" in user_prompt:
                user_content = user_prompt.strip()
            else:
                user_content = f"<image>\n{question}".strip()

            meta = {
                "sample_id": sample_id,
                "split": split_name,
                "question": question,
                "answer": answer,
                "user_content": user_content,
                "query_image_path": str(query_path),
                "thinking_chunks": thinking_chunks,
                "thinking_image_paths": thinking_image_paths,
            }
            f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 500 == 0:
                logger.info(f"Render progress: kept={kept} skipped={skipped}")

    # Flush remaining render tasks.
    flush_render_buffer(force=True)

    logger.info("=" * 70)
    logger.info("Render summary")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Kept samples: {kept}")
    logger.info(f"Skipped samples: {skipped}")
    logger.info("=" * 70)


def _load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _encode_missing_features(encoder: Qwen3VLEncoder, image_paths: List[str], cache_dir: Path, batch_size: int):
    to_encode = []
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


def main_encode_only(args):
    base_dir = Path(args.base_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir
    metadata_path = output_dir / args.metadata_file
    out_jsonl = output_dir / args.output_jsonl

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Run --render-only first.")

    samples = _load_metadata(metadata_path)
    if args.max_samples:
        samples = samples[: args.max_samples]

    logger.info(f"Metadata samples: {len(samples)}")

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )

    encoder = Qwen3VLEncoder(
        model_name_or_path=DEFAULT_MODEL_PATH,
        device=args.device,
        dtype=torch.bfloat16,
        use_vllm_kernels=True,
    )

    cache_dir = images_dir / ".feature_cache"
    cache_dir.mkdir(exist_ok=True)

    all_paths = set()
    for s in samples:
        all_paths.add(s["query_image_path"])
        for p in s.get("thinking_image_paths", []):
            all_paths.add(p)

    _encode_missing_features(encoder, sorted(all_paths), cache_dir, args.batch_size)

    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for s in samples:
            thinking_cache_paths = []
            for tp in s.get("thinking_image_paths", []):
                thinking_cache_paths.append(str(cache_dir / f"{Path(tp).stem}.pt"))

            query_cache_path = str(cache_dir / f"{Path(s['query_image_path']).stem}.pt")

            num_latent_steps = len(thinking_cache_paths)
            if num_latent_steps == 0:
                continue

            latent_placeholders = "<think_sep>".join(["<latent>"] * num_latent_steps)
            assistant_content = f"<think>{latent_placeholders}</think>{s['answer']}"

            cot = format_cot_subsequences(s.get("thinking_chunks"))
            item = {
                "messages": [
                    {"role": "user", "content": s["user_content"]},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [s["query_image_path"]],
                "latent_ground_truth": thinking_cache_paths,
                "latent_supervision": [query_cache_path],
                "num_latent_steps": num_latent_steps,
                "cot": cot,
                "cot_token_ids": tokenizer.encode(cot, add_special_tokens=False),
                "task": "deepvision_thinking",
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 500 == 0:
                logger.info(f"Write progress: {kept}/{len(samples)}")

    logger.info("=" * 70)
    logger.info("Encode summary")
    logger.info(f"Output JSONL: {out_jsonl}")
    logger.info(f"Kept samples: {kept}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Build DeepVision-103K thinking dataset for Qwen3VL")
    parser.add_argument("--base-dir", default="/share/project/xiyan/sources/DeepSeek-OCR")
    parser.add_argument("--data-dir", default="/share/project/xiyan/huggingface/skylenage/DeepVision-103K")
    parser.add_argument("--output-dir", default="Qwen/data")
    parser.add_argument("--images-dir", default="Qwen/data/deepvision_images")
    parser.add_argument("--metadata-file", default="deepvision_103k_metadata.jsonl")
    parser.add_argument("--output-jsonl", default="deepvision_103k_thinking.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-chars-per-chunk", type=int, default=4800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--thinking-source",
        default="auto",
        choices=["auto", "question", "system", "annotation", "none"],
        help="Source text used to render thinking chunks.",
    )
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="If >1, auto-launch torchrun (script currently uses rank-local single-process work).")

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
    logger.info("DeepVision-103K Thinking Dataset Builder")
    logger.info("=" * 70)
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Images dir: {args.images_dir}")
    logger.info(f"Thinking source: {args.thinking_source}")
    logger.info(f"Max samples: {args.max_samples or 'All'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    if args.render_only:
        main_render_only(args)
    else:
        main_encode_only(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
