#!/usr/bin/env python3
"""
ODinW-13 Evaluation with OCRVL vLLM Processor (unified interface)

We generate Qwen-style JSONL for reuse of their ODinW evaluator to compute mAP.

Rendering policy (unified):
- Global toggle via env RENDER=1 or 0 across all benchmarks.
- When RENDER=1, we render the instruction/classes text into an image and feed
  it alongside the real image (OCRVL-style, matching training).
- When RENDER=0, we use only the real image and a textual prompt (generic).
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def load_odinw_jobs(data_dir: Path, odinw_dir_env: str | None = None):
    import sys, os
    env_root = os.environ.get('QWEN_EVAL_ROOT')
    if env_root and Path(env_root).exists():
        qwen_eval_root = Path(env_root)
    else:
        qwen_eval_root = Path(__file__).resolve().parents[3] / 'Qwen3-VL' / 'evaluation'
    if not qwen_eval_root.exists():
        raise RuntimeError(f"Qwen3-VL evaluation path not found: {qwen_eval_root}")
    sys.path.insert(0, str(qwen_eval_root / 'ODinW-13'))
    sys.path.insert(0, str(qwen_eval_root))
    from dataset_utils import load_odinw_config, generate_odinw_jobs  # type: ignore
    odinw_dir = odinw_dir_env or os.environ.get('ODINW_DIR') or str(data_dir / 'odinw')
    if not Path(odinw_dir).exists():
        # Fallback: derive from QWEN_EVAL_ROOT's corresponding data directory
        alt = (qwen_eval_root / '..' / 'data' / 'odinw').resolve()
        if alt.exists():
            odinw_dir = str(alt)
    # Load config explicitly from file path expected by Qwen utils
    cfg_path = str(Path(odinw_dir) / 'odinw13_config.py')
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"ODinW config not found: {cfg_path}")
    cfg = load_odinw_config(cfg_path)
    class args:  # minimal namespace for generate_odinw_jobs
        ODINW_MAX_IMAGES_PER_DATASET = None
        ODINW_DATASETS = None
    questions, datasets = generate_odinw_jobs(odinw_dir, args)
    return questions, datasets


def _extract_prompt_from_messages(job: Dict[str, Any]) -> str:
    """Extract the text instruction from Qwen-style messages.

    Follows Qwen3-VL ODinW prompt convention, e.g.:
      "Locate every instance that belongs to the following categories: 'cat1, cat2, ...'.\n"
      "Report bbox coordinates in JSON format."
    """
    try:
        messages = job.get('messages', [])
        if messages and isinstance(messages, list):
            content = messages[0].get('content', [])
            for c in content:
                if isinstance(c, dict) and c.get('type') == 'text':
                    txt = str(c.get('text', '')).strip()
                    if txt:
                        return txt
    except Exception:
        pass
    # Fallback (generic, but still JSON-oriented)
    return "Locate objects from the specified categories. Return bbox_2d in JSON (0-1000 normalized)."


def _render_text_image(text: str, image_size=(640, 640)) -> Image.Image:
    """Render a block of text to an RGB image using Vello when available; fallback to PIL."""
    try:
        from Renderer import VelloRenderer, VELLO_AVAILABLE
    except Exception:
        VELLO_AVAILABLE=False
        VelloRenderer=None
    if VELLO_AVAILABLE and VelloRenderer is not None:
        try:
            vr = VelloRenderer(width=image_size[0], height=image_size[1], padding=16)
            img = vr.render_batch_pil([str(text)])[0]
            try:
                vr.shutdown()
            except Exception:
                pass
            return img
        except Exception:
            pass
    # PIL fallback
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    margin = 16
    max_w = image_size[0] - 2 * margin
    y = margin
    for line in str(text).split('\n'):
        words = line.split()
        cur = ""
        for w in words:
            test = (cur + " " + w) if cur else w
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_w:
                cur = test
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


def evaluate_odinw_vllm(
    checkpoint_path: Path,
    data_dir: Path,
    output_file: Path,
    batch_size: int = 8,
    max_tokens: int = 512,
    device: str = 'cuda:0',
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
    render_text: bool = False,
):
    logger.info("ODinW-13 Evaluation with OCRVL vLLM (unified)")
    logger.info(f"Checkpoint: {checkpoint_path}")

    jobs, datasets = load_odinw_jobs(data_dir)
    if shard_id is not None and num_shards is not None and int(num_shards) > 1:
        total = len(jobs)
        jobs = jobs[int(shard_id)::int(num_shards)]
        logger.info(f"Data Parallelism: Shard {shard_id}/{num_shards} - {len(jobs)} samples (from {total})")
    if not jobs:
        raise RuntimeError("No ODinW jobs generated; check data_dir/odinw and config file")

    from OCRInfer.encoder import DPSKOCREncoder
    from OCRVL.decoder import OCRVLProcessor
    encoder = DPSKOCREncoder(device=device)
    processor = OCRVLProcessor(checkpoint_path=checkpoint_path, device=device, gpu_memory_utilization=0.85)

    import os as _os
    shard_override = _os.environ.get('SHARD_OUTPUT')
    if shard_override:
        output_file = Path(shard_override)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out = open(output_file, 'w')

    num_batches = (len(jobs) + batch_size - 1) // batch_size
    for b in tqdm(range(num_batches), desc='Batches'):
        s, e = b * batch_size, min((b + 1) * batch_size, len(jobs))
        rows = jobs[s:e]

        # Build per-sample images and prompts aligned with Qwen3-VL messages
        imgs: List[Image.Image] = []
        prompts: List[str] = []
        for job in rows:
            # Find first image path from messages
            image_path = None
            try:
                for c in job['messages'][0]['content']:
                    if c.get('type') == 'image':
                        image_path = str(c['image']).replace('file://', '')
                        break
            except Exception:
                image_path = None

            if image_path and Path(image_path).exists():
                try:
                    imgs.append(Image.open(image_path).convert('RGB'))
                except Exception:
                    imgs.append(Image.new('RGB', (640, 640), color='white'))
            else:
                imgs.append(Image.new('RGB', (640, 640), color='white'))

            # Prompt text: reuse Qwen message text (contains dataset classes)
            prompt_text = _extract_prompt_from_messages(job)
            prompts.append(prompt_text)

        # Optionally append rendered prompt text as an image (OCRVL-style)
        # Build prompts based on render mode
        if render_text:
            # RENDER=1: Render prompts as images, use minimal seed prompt
            text_imgs = [_render_text_image(p) for p in prompts]
            all_imgs: List[Image.Image] = []
            for a, bimg in zip(imgs, text_imgs):
                all_imgs.extend([a, bimg])
            # Minimal seed to prevent EOS generation
            import os as _os_instr
            seed = _os_instr.environ.get('EVAL_INSTRUCTION', ' ')
            text_prompts = [seed if seed else ' '] * len(rows)  # At least a space
        else:
            # RENDER=0: Use prompts as text, no rendering
            all_imgs = imgs
            # Add instruction to prompts
            import os as _os_instr
            _min_instr = _os_instr.environ.get('EVAL_INSTRUCTION', 'Answer:')
            text_prompts = [f"{p}\n{_min_instr}" for p in prompts]

        # Encode images
        final_feats, deepstack_feats = encoder.encode_images_with_deepstack(all_imgs)

        # Group features per sample
        per_vis: List[List[Any]] = []
        per_deep: List[List[Any]] = []
        if render_text:
            for i in range(len(rows)):
                per_vis.append([final_feats[2 * i], final_feats[2 * i + 1]])
                per_deep.append([deepstack_feats[2 * i], deepstack_feats[2 * i + 1]])
        else:
            for i in range(len(rows)):
                per_vis.append([final_feats[i]])
                per_deep.append([deepstack_feats[i]])

        # Generate
        preds = processor.generate(
            visual_embeddings=per_vis,
            deepstack_features=per_deep,
            prompts=text_prompts,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        for job, pred in zip(rows, preds):
            pred_text = pred.strip()
            rec = {
                "question_id": job['question_id'],
                "annotation": job['annotation'],
                "extra_info": job['extra_info'],
                "result": {"gen": pred_text, "gen_raw": pred},
                "messages": job['messages'],
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()

    out.close()


def main():
    p = argparse.ArgumentParser(description='ODinW-13 eval with OCRVL vLLM')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=Path('/share/project/xiyan/data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--shard-id', type=int, default=None)
    p.add_argument('--num-shards', type=int, default=None)
    args = p.parse_args()

    try:
        import os as _os
        _render = (_os.environ.get('RENDER', '1') == '1')
        evaluate_odinw_vllm(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_file=Path((__import__('os').environ.get('SHARD_OUTPUT') or str(args.output))),
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            device=args.device,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            render_text=_render,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
