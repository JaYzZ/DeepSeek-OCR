#!/usr/bin/env python3
"""
MathVision Evaluation with OCRVL vLLM Processor (unified interface)

Produces Qwen3-VL JSONL so we can reuse MathVision evaluator.
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


def render_question_to_image(question: str, image_size=(640, 640)) -> Image.Image:
    img = Image.new('RGB', image_size, color='white')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    margin = 20
    max_w = image_size[0] - 2 * margin
    y = 20
    line = f"Question: {question}"
    words = line.split()
    cur = ""
    for w in words:
        t = (cur + " " + w) if cur else w
        bbox = d.textbbox((0, 0), t, font=font)
        if bbox[2] - bbox[0] <= max_w:
            cur = t
        else:
            if cur:
                d.text((margin, y), cur, fill='black', font=font)
                y += 28
            cur = w
    if cur:
        d.text((margin, y), cur, fill='black', font=font)
    return img




# Prefer our repo's GPU Vello renderer when available for text-as-image
def render_question_to_image_vello(question: str, image_size=(640,640)):
    try:
        from Renderer import VelloRenderer, VELLO_AVAILABLE
    except Exception:
        VELLO_AVAILABLE=False
        VelloRenderer=None
    text = 'Question: ' + str(question).strip()
    if VELLO_AVAILABLE and VelloRenderer is not None:
        try:
            vr = VelloRenderer(width=image_size[0], height=image_size[1], padding=20)
            img = vr.render_batch_pil([text])[0]
            try:
                vr.shutdown()
            except Exception:
                pass
            return img
        except Exception:
            pass
    return render_question_to_image(question, image_size=image_size)
def load_mathv_df_and_dump(data_dir: Path, dataset: str):
    import sys, os
    env_root = os.environ.get('QWEN_EVAL_ROOT')
    if env_root and Path(env_root).exists():
        qwen_eval_root = Path(env_root)
    else:
        qwen_eval_root = Path(__file__).resolve().parents[3] / 'Qwen3-VL' / 'evaluation'
    if not qwen_eval_root.exists():
        raise RuntimeError(f"Qwen3-VL evaluation path not found: {qwen_eval_root}")
    sys.path.insert(0, str(qwen_eval_root / 'MathVision'))
    sys.path.insert(0, str(qwen_eval_root))
    from dataset_utils import load_dataset, dump_image  # type: ignore
    os.environ['LMUData'] = str(data_dir)
    df = load_dataset(dataset)
    return df, dump_image


def evaluate_mathvision_vllm(
    checkpoint_path: Path,
    data_dir: Path,
    dataset: str,
    output_file: Path,
    batch_size: int = 8,
    max_tokens: int = 512,
    device: str = 'cuda:0',
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
):
    logger.info("MathVision Evaluation with OCRVL vLLM (unified)")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset:    {dataset}")

    df, dump_image_fn = load_mathv_df_and_dump(data_dir, dataset)
    if shard_id is not None and num_shards is not None and int(num_shards) > 1:
        total = len(df)
        df = df.iloc[int(shard_id)::int(num_shards)].reset_index(drop=True)
        logger.info(f"Data Parallelism: Shard {shard_id}/{num_shards} - {len(df)} samples (from {total})")

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

    img_root = data_dir / 'images' / dataset
    img_root.mkdir(parents=True, exist_ok=True)

    try:
        num_batches = (len(df) + batch_size - 1) // batch_size
        for b in tqdm(range(num_batches), desc='Batches'):
            s, e = b * batch_size, min((b + 1) * batch_size, len(df))
            rows = [df.iloc[i] for i in range(s, e)]

            # Real images (some samples contain multiple images)
            per_sample_paths: List[List[str]] = []
            for row in rows:
                paths = dump_image_fn(row, str(img_root))
                if isinstance(paths, list):
                    per_sample_paths.append(paths)
                else:
                    per_sample_paths.append([paths])

            # Render text image based on RENDER mode
            import os as _os_render
            _render_flag = (_os_render.environ.get('RENDER', '1') == '1')

            # Build question prompts
            prompts_per_sample = []
            q_images: List[Image.Image] = []
            for row in rows:
                question = str(row.get('question', ''))

                if _render_flag:
                    # RENDER=1: Render question as image, use minimal seed prompt
                    q_images.append(render_question_to_image_vello(question))
                    import os as _os_instr
                    seed = _os_instr.environ.get('EVAL_INSTRUCTION', ' ')
                    prompts_per_sample.append(seed if seed else ' ')  # At least a space
                else:
                    # RENDER=0: Use question as text prompt, no rendering
                    question_text = f"Question: {question}\n"
                    import os as _os_instr
                    _min_instr = _os_instr.environ.get('EVAL_INSTRUCTION', 'Answer:')
                    question_text += _min_instr
                    prompts_per_sample.append(question_text)

            # Flatten for encoding
            all_images: List[Image.Image] = []
            sizes: List[int] = []
            for idx, paths in enumerate(per_sample_paths):
                imgs = []
                for p in paths:
                    try:
                        imgs.append(Image.open(p).convert('RGB'))
                    except Exception:
                        imgs.append(Image.new('RGB', (640, 640), color='white'))

                # In RENDER=1, append question image; in RENDER=0, don't
                if _render_flag:
                    imgs.append(q_images[idx])

                sizes.append(len(imgs))
                all_images.extend(imgs)

            final_feats, deepstack_feats = encoder.encode_images_with_deepstack(all_images)

            # Group embeddings
            per_sample_feats: List[List[Any]] = []
            per_sample_deep: List[List[Any]] = []
            cur = 0
            for n in sizes:
                per_sample_feats.append(final_feats[cur:cur+n])
                per_sample_deep.append(deepstack_feats[cur:cur+n])
                cur += n

            # Generate with appropriate prompts
            preds = processor.generate(
                visual_embeddings=per_sample_feats,
                deepstack_features=per_sample_deep,
                prompts=prompts_per_sample,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            for row, pred in zip(rows, preds):
                pred_text = pred.strip()
                line = row.to_dict()
                rec = {
                    "question_id": line.get('index', row.name),
                    "annotation": line,
                    "task": dataset,
                    "result": {"gen": pred_text, "gen_raw": pred},
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
    finally:
        # Ensure vLLM resources are released so shard processes can exit
        try:
            processor.shutdown()
        except Exception:
            pass

    out.close()


def main():
    p = argparse.ArgumentParser(description='MathVision eval with OCRVL vLLM')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=Path('/share/project/xiyan/data'))
    p.add_argument('--dataset', type=str, default='MathVision')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--shard-id', type=int, default=None)
    p.add_argument('--num-shards', type=int, default=None)
    args = p.parse_args()

    try:
        evaluate_mathvision_vllm(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            dataset=args.dataset,
            output_file=Path((__import__('os').environ.get('SHARD_OUTPUT') or str(args.output))),
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            device=args.device,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
