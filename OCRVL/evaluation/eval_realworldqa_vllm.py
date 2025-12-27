#!/usr/bin/env python3
"""
RealWorldQA Evaluation with OCRVL vLLM Processor

This script evaluates OCRVL checkpoints on RealWorldQA using the vLLM backend
with direct connector integration (no HF export needed).

**OCRVL Pure Vision Evaluation Paradigm:**
Following the OCRVL training approach where all inputs are vision-based:
1. Input 1: Original RealWorldQA image (real photo)
2. Input 2: Rendered [question + instruction] (as single image)
3. Both images encoded together
4. Empty text prompt (pure vision)
5. Generate response

This matches how OCRVL was trained on BLIP3o dataset with pure vision format:
- All instructions rendered as images, not passed as text
- Question and instruction packed together in one rendered image
- Model relies purely on vision tokens for understanding

Usage:
    # Default: renders "Question: ...\nA. ...\nB. ...\nAnswer:" as single image
    python OCRVL/evaluation/eval_realworldqa_vllm.py \
        --checkpoint OCRVL/checkpoints/alignment_long_20251223_043933/step_786 \
        --data-dir /share/project/xiyan/data \
        --output results/realworldqa_vllm.jsonl

    # Custom instruction (still rendered together with question)
    EVAL_INSTRUCTION="Choose the correct answer:" python OCRVL/evaluation/eval_realworldqa_vllm.py \
        --checkpoint ... --data-dir ... --output ...

    # No instruction (just question + choices)
    EVAL_INSTRUCTION="" python OCRVL/evaluation/eval_realworldqa_vllm.py \
        --checkpoint ... --data-dir ... --output ...
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import torch
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_realworldqa_dataset_df(data_dir: Path):
    """Load RealWorldQA dataset via Qwen3-VL utilities to ensure consistent schema."""
    import sys, os
    qwen_eval_root = Path(__file__).resolve().parent.parent.parent / "Qwen3-VL" / "evaluation"
    if qwen_eval_root.exists():
        sys.path.insert(0, str(qwen_eval_root / "RealWorldQA"))
        sys.path.insert(0, str(qwen_eval_root))
    else:
        logger.warning(f"Qwen3-VL evaluation not found at {qwen_eval_root}; falling back to TSV loader")

    try:
        from dataset_utils import load_dataset, dump_image  # type: ignore
        os.environ['LMUData'] = str(data_dir)
        df = load_dataset('RealWorldQA')
        return df, dump_image
    except Exception:
        # Fallback: local TSV under data_dir
        import pandas as pd
        tsv_file = data_dir / "RealWorldQA.tsv"
        if not tsv_file.exists():
            raise
        df = pd.read_csv(tsv_file, sep='\t')
        def dump_image_fallback(line, img_root):
            # Decode base64 image if present; otherwise expect image_path
            import base64
            from io import BytesIO
            from PIL import Image
            os.makedirs(img_root, exist_ok=True)
            if 'image' in line:
                out = Path(img_root) / f"{line['index']}.jpg"
                if not out.exists():
                    img = Image.open(BytesIO(base64.b64decode(line['image']))).convert('RGB')
                    img.save(out)
                return [str(out)]
            else:
                return [str(Path(img_root) / line['image_path'])]
        return df, dump_image_fallback


def render_text_to_image(text: str, image_size=(640, 640)) -> Image.Image:
    """Render any text to image using VelloRenderer (fast GPU path) or PIL fallback.

    This is a uniform rendering function that handles any text cleanly without prefixes.
    Use this for instructions, questions, or any other text that needs to be rendered.

    Args:
        text: Text to render (e.g., "Answer:", "What's in the image?", "Question: ...")
        image_size: Output image size (width, height)

    Returns:
        PIL Image with rendered text
    """
    # Defer heavy import; keep local to avoid overhead when not needed
    try:
        from Renderer import VelloRenderer, VELLO_AVAILABLE
    except Exception:
        VELLO_AVAILABLE = False
        VelloRenderer = None  # type: ignore

    text = str(text).strip()

    if VELLO_AVAILABLE and VelloRenderer is not None:
        try:
            vr = VelloRenderer(width=image_size[0], height=image_size[1], padding=20)
            pil_img = vr.render_batch_pil([text])[0]
            try:
                vr.shutdown()
            except Exception:
                pass
            return pil_img
        except Exception:
            pass

    # PIL fallback (kept simple and fast)
    from PIL import ImageDraw, ImageFont
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    margin = 16
    max_w = image_size[0] - 2 * margin
    y = margin
    for line in text.split('\n'):
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


def render_question_to_image(question: str, choices: List[str], image_size=(640, 640)) -> Image.Image:
    """Render question with multiple choice options.

    This formats the question with "Question: " prefix and A/B/C/D choices.
    For rendering plain instructions, use render_text_to_image() instead.
    """
    text = "Question: " + str(question).strip()
    for i, choice in enumerate(choices):
        if choice is None:
            continue
        ch = str(choice).strip()
        if ch:
            text += f"\n{chr(65+i)}. {ch}"

    return render_text_to_image(text, image_size)


def evaluate_realworldqa_vllm(
    checkpoint_path: Path,
    data_dir: Path,
    output_file: Path,
    batch_size: int = 8,
    max_tokens: int = 512,
    device: str = "cuda:0",
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
):
    """
    Evaluate OCRVL on RealWorldQA using vLLM processor.

    Args:
        checkpoint_path: Path to OCRVL checkpoint
        data_dir: Path to data directory containing RealWorldQA
        output_file: Path to save predictions
        batch_size: Batch size for inference
        max_tokens: Maximum tokens to generate
        device: Device for encoder and connectors
    """
    logger.info("=" * 80)
    logger.info("RealWorldQA Evaluation with OCRVL vLLM")
    logger.info("=" * 80)
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info("")

    # Load dataset (DataFrame + dump_image function)
    logger.info("Loading RealWorldQA dataset...")
    df, dump_image_fn = load_realworldqa_dataset_df(data_dir)

    # Apply data-parallel sharding if requested
    if shard_id is not None and num_shards is not None and int(num_shards) > 1:
        total = len(df)
        df = df.iloc[int(shard_id)::int(num_shards)].reset_index(drop=True)
        logger.info(f"Data Parallelism: Shard {shard_id}/{num_shards} - {len(df)} samples (from {total})")

    # Initialize encoder
    logger.info("\nInitializing DeepSeek-OCR encoder...")
    from OCRInfer.encoder import DPSKOCREncoder
    encoder = DPSKOCREncoder(device=device)

    # Initialize OCRVL vLLM processor
    logger.info("\nInitializing OCRVL vLLM processor...")
    from OCRVL.decoder import OCRVLProcessor
    processor = OCRVLProcessor(
        checkpoint_path=checkpoint_path,
        device=device,
        gpu_memory_utilization=0.85,
    )

    # Prepare output file
    # Override output with SHARD_OUTPUT if provided (for data-parallel merges)
    import os as _os
    shard_override = _os.environ.get('SHARD_OUTPUT')
    if shard_override:
        output_file = Path(shard_override)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_handle = open(output_file, 'w')

    logger.info("\n" + "=" * 80)
    logger.info("Running Inference (OCRVL)")
    logger.info("=" * 80)
    logger.info("  RENDER=1: Pure vision mode (real image + rendered question, empty prompt)")
    logger.info("  RENDER=0: Hybrid mode (real image + text prompt, no rendering)")
    logger.info("  Instruction: env EVAL_INSTRUCTION (default 'Answer:')")
    logger.info("")

    # Process in batches
    num_batches = (len(df) + batch_size - 1) // batch_size

    img_root = data_dir / 'images' / 'RealWorldQA'
    img_root.mkdir(parents=True, exist_ok=True)

    try:
        for batch_idx in tqdm(range(num_batches), desc="Batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_rows = [df.iloc[i] for i in range(start_idx, end_idx)]

            # Build images per sample and text prompts based on RENDER mode
            # Get instruction text from environment (default: "Answer:")
            import os as _os_instr
            _instr_text = _os_instr.environ.get('EVAL_INSTRUCTION', 'Answer:')
            _render_flag = (_os_instr.environ.get('RENDER', '1') == '1')

            images_per_sample = []
            prompts_per_sample = []
            for row in batch_rows:
                # Dump real image
                paths = dump_image_fn(row, str(img_root))
                p = paths[0] if isinstance(paths, list) and paths else paths
                try:
                    real_img = Image.open(p).convert('RGB')
                except Exception:
                    real_img = Image.new('RGB', (640, 640), color='white')

                # Build question + choices + instruction text
                q = str(row.get('question', '')).strip()
                choices = []
                for k in ['A', 'B', 'C', 'D']:
                    v = row.get(k)
                    if v is not None:
                        choices.append(str(v))

                # Pack question + choices + instruction into one text
                text = "Question: " + q
                for i, choice in enumerate(choices):
                    if choice:
                        text += f"\n{chr(65+i)}. {choice}"
                # Add instruction at the end
                if _instr_text:
                    text += f"\n{_instr_text}"

                # RENDER=1: Pure vision mode (render text as image, use chat tokens)
                # RENDER=0: Hybrid mode (use text as prompt, no rendering)
                if _render_flag:
                    # Render question as image
                    question_instruction_img = render_text_to_image(text)
                    imgs = [real_img, question_instruction_img]
                    # Use proper Qwen3-VL assistant start token (matches training format)
                    prompt_text = '<|im_start|>assistant\n'
                else:
                    # Use text directly as prompt
                    imgs = [real_img]  # Only real image
                    prompt_text = text  # Question as text prompt

                images_per_sample.append(imgs)
                prompts_per_sample.append(prompt_text)

            # Flatten for encoding
            all_images = []
            sizes = []
            for imgs in images_per_sample:
                sizes.append(len(imgs))
                all_images.extend(imgs)

            # Encode
            final_features, deepstack_features = encoder.encode_images_with_deepstack(all_images)

            # Group embeddings per sample (variable number of images)
            batch_feature_lists = []
            batch_deepstack_lists = []
            cur = 0
            for n in sizes:
                batch_feature_lists.append(final_features[cur:cur+n])
                batch_deepstack_lists.append(deepstack_features[cur:cur+n])
                cur += n

            # Generate
            # RENDER=1: Pure vision (rendered images, empty prompts)
            # RENDER=0: Hybrid mode (real image, text prompts)
            predictions = processor.generate(
                visual_embeddings=batch_feature_lists,
                deepstack_features=batch_deepstack_lists,
                prompts=prompts_per_sample,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            # Save in Qwen-compatible JSONL format
            for row, prediction in zip(batch_rows, predictions):
                prediction_clean = prediction.strip()
                # Build annotation dict
                line_dict = {k: (int(v) if isinstance(v, bool) is False and isinstance(v, (int,)) else v) for k, v in row.to_dict().items()}
                result_obj = {
                    "question_id": line_dict.get('index', row.name),
                    "annotation": line_dict,
                    "task": 'RealWorldQA',
                    "result": {"gen": prediction_clean, "gen_raw": prediction},
                }
                output_handle.write(json.dumps(result_obj) + "\n")
                output_handle.flush()
    finally:
        # Ensure vLLM resources are released
        try:
            processor.shutdown()
        except Exception:
            pass

    output_handle.close()

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Predictions saved to: {output_file}")
    logger.info("=" * 80)

    # Local quick accuracy (A/B/C/D) if the dataset has answers
    try:
        logger.info("\nComputing quick accuracy (rule-based)...")
        correct = 0
        total = 0
        with open(output_file) as f:
            for line in f:
                job = json.loads(line)
                ann = job.get('annotation', {})
                gt = str(ann.get('answer', '')).strip().upper()
                pred = str(job.get('result', {}).get('gen', '')).strip().upper()
                # Extract first choice letter if present
                pred_letter = None
                if pred and pred[0] in 'ABCD':
                    pred_letter = pred[0]
                else:
                    for ch in pred:
                        if ch in 'ABCD':
                            pred_letter = ch
                            break
                if pred_letter is None:
                    pred_letter = ''
                total += 1
                if pred_letter == gt:
                    correct += 1
        acc = (correct / total) if total else 0.0
        logger.info(f"  Correct: {correct}/{total}")
        logger.info(f"  Accuracy: {acc:.2%}")
        with open(output_file.parent / "accuracy.json", 'w') as f:
            json.dump({'correct': correct, 'total': total, 'accuracy': acc}, f, indent=2)
    except Exception as e:
        logger.info(f"Quick accuracy skipped: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCRVL on RealWorldQA with vLLM"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to OCRVL checkpoint directory"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="/share/project/xiyan/data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="OCRVL/evaluation/results/realworldqa_vllm/predictions.jsonl",
        help="Output file for predictions"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for encoder and connectors"
    )
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard ID for data parallelism (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total shards for data parallelism")

    args = parser.parse_args()

    try:
        evaluate_realworldqa_vllm(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_file=Path((__import__('os').environ.get('SHARD_OUTPUT') or str(args.output))),
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            device=args.device,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        # Best-effort cleanup: if an OCRVLProcessor was created in this module
        # in future edits, call its shutdown to release vLLM. Current script
        # constructs processor inside evaluate_realworldqa_vllm and it cleans up
        # on return, so this is just a guard for future changes.
        try:
            from OCRVL.decoder import OCRVLProcessor  # noqa: F401
        except Exception:
            pass
    exit(code)
