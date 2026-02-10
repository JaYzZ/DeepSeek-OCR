#!/usr/bin/env python3
"""
Backfill transparent evaluation for existing training checkpoints.

This script loads a checkpoint WITHOUT FSDP and runs transparent eval,
then saves results back to the checkpoint directory. Use this to recover
eval_results for checkpoints that didn't have transparent eval enabled
during training.

Usage:
    # Backfill latest checkpoint
    python Qwen/scripts/backfill_transparent_eval.py --checkpoint_dir Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking

    # Backfill specific checkpoint
    python Qwen/scripts/backfill_transparent_eval.py --checkpoint_dir ... --checkpoint run_20260208_192913

    # Quick test on 5 samples
    python Qwen/scripts/backfill_transparent_eval.py --checkpoint_dir ... --max_samples 5
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add repo root to path
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the latest checkpoint subdirectory."""
    # First, check if this is a valid model directory directly
    config_files = list(checkpoint_dir.glob("config.json")) + list(checkpoint_dir.glob("model_config.json"))
    if config_files:
        return checkpoint_dir

    # Look for run_* subdirectories
    checkpoints = list(checkpoint_dir.glob("run_*/"))
    if not checkpoints:
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))

    if not checkpoints:
        # If still no checkpoints, assume checkpoint_dir is the model path
        return checkpoint_dir

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def load_model(checkpoint_path: Path, device: str = "cuda"):
    """Load model checkpoint WITHOUT FSDP wrapping."""
    logger.info(f"Loading model from {checkpoint_path}...")

    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    logger.info(f"Model loaded: {type(model).__name__}")
    return model, processor


def run_evaluation(model, processor, eval_metadata: Path, max_samples: int = None):
    """Run transparent evaluation on loaded model."""
    # Load metadata
    with open(eval_metadata, 'r') as f:
        metadata = json.load(f)

    samples = metadata['samples'][:max_samples] if max_samples else metadata['samples']

    logger.info(f"Running transparent eval on {len(samples)} samples...")

    # Load full samples from JSONL to get ground_truth
    full_samples = {}
    jsonl_path = eval_metadata.parent / "ocrvl_transparent_eval.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                sample_id = sample.get('id')
                if sample_id:
                    if 'ground_truth' not in sample:
                        for msg in sample.get('messages', []):
                            if msg.get('role') == 'assistant':
                                sample['ground_truth'] = msg.get('content', '')
                                break
                    full_samples[sample_id] = sample

    # Merge ground_truth
    for sample in samples:
        sample_id = sample.get('id')
        if sample_id in full_samples:
            if 'ground_truth' not in sample or not sample['ground_truth']:
                sample['ground_truth'] = full_samples[sample_id].get('ground_truth', '')

    # Run generation
    results = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        sample_start = time.time()

        try:
            # Load images
            images = []
            for img_path in sample['images']:
                full_path = _REPO_ROOT / img_path
                if full_path.exists():
                    images.append(Image.open(full_path).convert('RGB'))
                else:
                    logger.warning(f"Image not found: {full_path}")
                    continue

            if len(images) not in (1, 2):
                logger.warning(f"Expected 1 or 2 images, got {len(images)}")
                continue

            # Process images
            image_processor = processor.image_processor
            mm_inputs = image_processor(images, return_tensors="pt")

            # Calculate vision placeholders
            image_grid_thw = mm_inputs.get("image_grid_thw")
            merge_length = getattr(image_processor, "merge_size", 2) ** 2

            vision_placeholders = []
            for j in range(len(images)):
                if image_grid_thw is not None:
                    image_seqlen = image_grid_thw[j].prod().item() // merge_length
                else:
                    image_seqlen = 100
                placeholder = "<|vision_start|>" + "<|image_pad|>" * image_seqlen + "<|vision_end|>"
                vision_placeholders.append(placeholder)

            # Build conversation
            instruction_text = sample.get('instruction', 'Describe the image:').replace('<image>', '').strip()

            if len(images) == 1:
                user_content = instruction_text + " " + vision_placeholders[0]
            else:
                user_content = instruction_text + " " + vision_placeholders[0] + vision_placeholders[1]

            conversation = [{"role": "user", "content": user_content}]

            # Tokenize
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            text_inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False
            )

            # Move to device
            device = next(model.parameters()).device
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            pixel_values = mm_inputs['pixel_values'].to(device=device, dtype=torch.bfloat16)
            image_grid_thw = mm_inputs['image_grid_thw'].to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=False,
                )

            # Decode
            input_len = input_ids.shape[1]
            generated_ids = outputs[0][input_len:].tolist()
            full_output = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()

            # Extract display output (strip thinking tags)
            display_output = full_output
            if "</think>" in display_output:
                display_output = display_output.split("</think>", 1)[-1].strip()
            display_output = display_output.replace("<think>", "").replace("</think>", "").strip()

            sample_elapsed = time.time() - sample_start
            logger.info(f"[{i+1}/{len(samples)}] {sample['id']} ({sample_elapsed:.1f}s): {display_output[:60]}...")

            results.append({
                'id': sample['id'],
                'task': sample['task'],
                'images': sample['images'],
                'instruction': sample.get('instruction', ''),
                'ground_truth': sample.get('ground_truth', ''),
                'generated_answer': full_output if full_output else "[EMPTY]",
                'generated_answer_display': display_output if display_output else "[EMPTY]",
            })

        except Exception as e:
            logger.warning(f"Failed to process sample {sample.get('id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue

    elapsed = time.time() - start_time
    logger.info(f"Generated {len(results)}/{len(samples)} samples in {elapsed:.1f}s ({elapsed/len(results):.1f}s/sample)")

    return results


def save_results(results: list, checkpoint_dir: Path):
    """Save results to checkpoint directory in the same format as training callback."""
    eval_dir = checkpoint_dir / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON in same format as TransparentEvalCallback
    json_path = eval_dir / f"backfill_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'num_samples': len(results),
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(results)} results to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Backfill transparent eval for existing checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to training output directory (e.g., Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint subdirectory (default: latest)")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path to eval metadata JSON (default: OCRVL/data/ocrvl_transparent_eval.metadata.json)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load model on (default: cuda)")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = checkpoint_dir / args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)

    logger.info(f"Using checkpoint: {checkpoint_path}")

    # Find metadata
    if args.metadata:
        eval_metadata = Path(args.metadata)
    else:
        eval_metadata = _REPO_ROOT / "OCRVL/data/ocrvl_transparent_eval.metadata.json"

    if not eval_metadata.exists():
        raise FileNotFoundError(f"Metadata not found: {eval_metadata}")

    # Load model WITHOUT FSDP (for backfilling)
    model, processor = load_model(checkpoint_path, args.device)

    # Run evaluation
    results = run_evaluation(model, processor, eval_metadata, args.max_samples)

    # Save results
    save_results(results, checkpoint_path)

    logger.info("Backfill complete!")


if __name__ == "__main__":
    main()
