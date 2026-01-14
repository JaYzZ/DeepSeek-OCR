#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Try to import renderer for instruction rendering
_VELLO_RENDERER = None
_VELLO_AVAILABLE = False

def _get_vello_renderer(image_size=(640, 640)):
    """Get cached Vello renderer instance"""
    global _VELLO_RENDERER, _VELLO_AVAILABLE

    if _VELLO_RENDERER is None:
        try:
            from Renderer import VelloRenderer, VELLO_AVAILABLE
            _VELLO_AVAILABLE = VELLO_AVAILABLE
            if _VELLO_AVAILABLE:
                _VELLO_RENDERER = VelloRenderer(width=image_size[0], height=image_size[1], padding=20)
        except Exception:
            _VELLO_AVAILABLE = False
            _VELLO_RENDERER = None

    return _VELLO_RENDERER

def _render_instruction_to_pil(instruction: str, image_size=(640, 640)):
    """Render instruction text to PIL image using Vello (or PIL fallback)"""
    vr = _get_vello_renderer(image_size)
    if vr is not None:
        try:
            # Vello returns numpy array, convert to PIL
            img_array = vr.render_batch([instruction])[0]
            from PIL import Image
            return Image.fromarray(img_array)
        except Exception as e:
            pass

    # PIL fallback
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    margin = 16
    max_w = image_size[0] - 2 * margin
    y = margin
    for line in instruction.split('\n'):
        words = line.split()
        cur = ""
        for w in words:
            t = (cur + " " + w) if cur else w
            bbox = draw.textbbox((0, 0), t, font=font)
            if bbox[2] - bbox[0] <= max_w:
                cur = t
            else:
                if cur:
                    draw.text((margin, y), cur, fill='black', font=font)
                    y += 26
                cur = w
        if cur:
            draw.text((margin, y), cur, fill='black', font=font)
            y += 26
        y += 6
    return img


def _decode_txt(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip()
    return str(x).strip()


def _write_jsonl(path: Path, rows: Iterator[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _resolve_image_path(image_dir: Path, image_rel_or_abs: str) -> str:
    p = Path(image_rel_or_abs)
    if p.is_absolute() and p.exists():
        return str(p)
    # Always return an absolute/anchored path under image_dir for relative inputs
    # (even if the file does not exist yet), so manifests are stable.
    return str(image_dir / image_rel_or_abs)


def _llava_pretrain_iter(
    *,
    seed: int,
    json_path: Path,
    image_dir: Path,
    max_samples: int,
) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    if not json_path.exists():
        raise FileNotFoundError(f"LLaVA-Pretrain JSON not found: {json_path}")
    if not image_dir.exists():
        # Allow building the manifest before images are downloaded; paths will resolve once available.
        # LlamaFactory will fail at training time if the images are still missing.
        pass

    emitted = 0
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("LLaVA-Pretrain JSON must be a list of examples.")

    # Shuffle for sampling stability across runs (especially when truncating).
    rng.shuffle(data)

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
            # Try the common LLaVA format: last assistant turn is caption.
            conv = item["conversations"]
            for turn in reversed(conv):
                if isinstance(turn, dict) and str(turn.get("from")) in {"gpt", "assistant"} and "value" in turn:
                    caption = turn["value"]
                    break

        caption = str(caption).strip() if caption is not None else ""
        if not caption:
            continue

        img_path = _resolve_image_path(image_dir, str(image))
        yield {
            "messages": [
                {"role": "user", "content": "<image>Describe the image:"},
                {"role": "assistant", "content": caption},
            ],
            "images": [{"path": img_path}],
        }
        emitted += 1


def _load_doclaynet_caption(json_dir: Path, image_hash: str, fallback: str, max_chars: int) -> str:
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


def _doclaynet_iter(
    *,
    doc_root: Path,
    split: str,
    seed: int,
    max_chars: int,
    max_samples: int,
) -> Iterator[Dict[str, Any]]:
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

    rng = random.Random(seed)
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
        caption = (
            _load_doclaynet_caption(json_dir, image_hash, fallback=fallback, max_chars=max_chars)
            if has_text
            else fallback[:max_chars]
        )

        yield {
            "messages": [
                {"role": "user", "content": "<image>Please transcribe all text in the image."},
                {"role": "assistant", "content": caption},
            ],
            "images": [{"path": str(image_path)}],
        }
        emitted += 1


def _mix_sources(
    *,
    blip_iter: Iterator[Dict[str, Any]],
    doc_iter: Optional[Iterator[Dict[str, Any]]],
    total: int,
    doc_ratio: float,
    seed: int,
) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    doc_ratio = max(0.0, min(1.0, doc_ratio))

    want_doc = [rng.random() < doc_ratio for _ in range(total)]
    for pick_doc in want_doc:
        if pick_doc and doc_iter is not None:
            try:
                yield next(doc_iter)
                continue
            except StopIteration:
                doc_iter = None
        # fallback to blip
        yield next(blip_iter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_jsonl",
        default="OCRVL/llamafactory/data/ocrvl_alignment_llava_pretrain_doclaynet.jsonl",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 = full LLaVA-Pretrain + full DocLayNet (no truncation). Set >0 for sampled debugging.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--doc_ratio", type=float, default=0.5)
    ap.add_argument("--mix_doclaynet", action="store_true", default=True)
    ap.add_argument("--no-mix_doclaynet", dest="mix_doclaynet", action="store_false")

    ap.add_argument(
        "--llava_pretrain_json",
        default="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
    )
    ap.add_argument(
        "--llava_pretrain_image_dir",
        default="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Pretrain/images",
    )

    ap.add_argument("--doclaynet_root", default="/share/project/xiyan/huggingface/docling-project/DocLayNet")
    ap.add_argument("--doclaynet_split", default="train")
    ap.add_argument("--doclaynet_max_chars", type=int, default=2048)

    args = ap.parse_args()

    out_jsonl = Path(args.output_jsonl)

    llava_iter = _llava_pretrain_iter(
        seed=args.seed,
        json_path=Path(args.llava_pretrain_json),
        image_dir=Path(args.llava_pretrain_image_dir),
        max_samples=args.max_samples,
    )

    doc_iter = None
    if args.mix_doclaynet and args.doc_ratio > 0:
        doc_iter = _doclaynet_iter(
            doc_root=Path(args.doclaynet_root),
            split=args.doclaynet_split,
            seed=args.seed,
            max_chars=args.doclaynet_max_chars,
            max_samples=args.max_samples,
        )

    if args.max_samples <= 0:
        def _rows_full():
            yield from llava_iter
            if doc_iter is not None:
                yield from doc_iter
        rows = _rows_full()
    else:
        rows = _mix_sources(
            blip_iter=llava_iter,
            doc_iter=doc_iter,
            total=args.max_samples,
            doc_ratio=args.doc_ratio,
            seed=args.seed,
        )

    n = _write_jsonl(out_jsonl, rows)
    print(f"✓ Wrote {n} examples to {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
