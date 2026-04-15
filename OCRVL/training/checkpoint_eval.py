#!/usr/bin/env python3
"""
Checkpoint Evaluation for Training Transparency

Runs inference on a fixed set of VQA samples during checkpoint saves
to provide qualitative monitoring of training progress.
"""

import json
import logging
from pathlib import Path
from project_paths import hf_path
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def run_checkpoint_evaluation(
    model,
    tokenizer,
    ocr_adapter,
    checkpoint_dir: Path,
    eval_samples_path: str,
    image_base_dir: str,
    global_step: int,
    args,
    max_samples: int = 10,
) -> None:
    """
    Run inference on fixed evaluation samples and save results to checkpoint directory.

    Args:
        model: Trained model (DDP-wrapped or regular)
        tokenizer: Tokenizer for input/output formatting
        ocr_adapter: OCR adapter for encoding images
        checkpoint_dir: Directory where checkpoint is saved
        eval_samples_path: Path to eval_samples.json
        image_base_dir: Base directory for loading images
        global_step: Current training step
        args: Training arguments
        max_samples: Maximum number of samples to evaluate
    """
    from torch.amp import autocast

    # Only run on rank 0
    if not _is_main_process():
        return

    eval_samples_file = Path(eval_samples_path)
    if not eval_samples_file.exists():
        logger.warning(f"Evaluation samples file not found: {eval_samples_path}")
        return

    logger.info(f"Running TEXT-FREE checkpoint evaluation on {max_samples} samples (10 VQA + 2 DocLayNet OCR, questions rendered as images)...")

    # Load evaluation samples
    with open(eval_samples_file, 'r') as f:
        eval_samples = json.load(f)[:max_samples]

    # Unwrap DDP model if needed
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()

    results = []
    image_loader = None

    # Initialize image loader for LLaVA datasets
    try:
        from OCRVL.data.parquet_image_loader import Mix665kImageLoader
        image_loader = Mix665kImageLoader(image_base_dir)
    except Exception as e:
        logger.warning(f"Failed to initialize Mix665k image loader: {e}")

    # Process each sample
    for sample in eval_samples:
        try:
            # Load image
            image_path = sample['image_path']
            image = None

            # Handle DocLayNet images (absolute paths from huggingface root)
            if 'docling-project/DocLayNet' in image_path:
                # DocLayNet: path is relative to the shared Hugging Face mirror root.
                doclaynet_base = str(hf_path())
                full_path = Path(doclaynet_base) / image_path
                if full_path.exists():
                    image = Image.open(full_path).convert('RGB')
                else:
                    logger.warning(f"DocLayNet image not found: {full_path}")
                    continue
            else:
                # LLaVA images: Try Mix665k loader first
                if image_loader is not None:
                    try:
                        image = image_loader.load_image(image_path)
                    except Exception:
                        pass

                # Fallback to direct file loading
                if image is None:
                    full_path = Path(image_base_dir) / image_path
                    if full_path.exists():
                        image = Image.open(full_path).convert('RGB')
                    else:
                        logger.warning(f"Image not found: {full_path}")
                        continue

            # Encode image - handle different OCR adapter types
            if hasattr(ocr_adapter, 'encode_images_with_deepstack'):
                # DPSKOCREncoder-style adapter (inference mode)
                ocr_features, deepstack_features = ocr_adapter.encode_images_with_deepstack([image])
            elif hasattr(ocr_adapter, 'images_to_ocr_features'):
                # Qwen3VLOCRTextAdapter-style adapter (training mode)
                encoded = ocr_adapter.images_to_ocr_features([image])
                if isinstance(encoded, tuple):
                    ocr_features, deepstack_features = encoded
                else:
                    ocr_features = encoded
                    deepstack_features = None
            else:
                logger.warning(f"Unknown OCR adapter type: {type(ocr_adapter)}")
                continue

            # Render question to image (TEXT-FREE evaluation, matching training)
            question = sample['question']
            question_image = _render_question_to_image(question, ocr_adapter)

            # Encode question image
            if hasattr(ocr_adapter, 'encode_images_with_deepstack'):
                question_ocr_features, question_deepstack_features = ocr_adapter.encode_images_with_deepstack([question_image])
            elif hasattr(ocr_adapter, 'images_to_ocr_features'):
                encoded = ocr_adapter.images_to_ocr_features([question_image])
                if isinstance(encoded, tuple):
                    question_ocr_features, question_deepstack_features = encoded
                else:
                    question_ocr_features = encoded
                    question_deepstack_features = None
            else:
                logger.warning(f"Cannot encode question image")
                continue

            # Generate answer with TEXT-FREE input (question image + real image)
            with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                generated_answer = _generate_answer(
                    eval_model,
                    tokenizer,
                    ocr_features,  # Image features
                    deepstack_features,
                    question_ocr_features,  # Question features (rendered)
                    question_deepstack_features,
                    args.device,
                    max_new_tokens=128,
                    temperature=0.0,
                )

            # Store result
            results.append({
                'id': sample['id'],
                'image_path': image_path,
                'question': question,
                'ground_truth': sample.get('answer', 'N/A'),
                'generated_answer': generated_answer,
                'step': global_step,
            })

        except Exception as e:
            logger.warning(f"Failed to evaluate sample {sample.get('id', 'unknown')}: {e}")
            continue

    # Save results to checkpoint directory
    _save_evaluation_results(checkpoint_dir, results, global_step)

    # Restore training mode
    eval_model.train()
    logger.info(f"✓ TEXT-FREE checkpoint evaluation complete: {len(results)}/{len(eval_samples)} samples")


def _render_question_to_image(question: str, ocr_adapter):
    """Render question text to image using the OCR adapter's renderer."""
    from PIL import Image, ImageDraw, ImageFont

    # Use the same renderer as training (via ocr_adapter)
    if hasattr(ocr_adapter, '_render_texts'):
        # Use built-in renderer
        rendered_images = ocr_adapter._render_texts([question])
        return rendered_images[0]
    else:
        # Fallback: simple PIL rendering (640x640, matching training)
        width, height = 640, 640
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Try to load a proper font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()

        # Simple text wrapping
        margin = 30
        y_offset = margin
        max_width = width - 2 * margin

        words = question.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        # Draw lines
        for line in lines:
            draw.text((margin, y_offset), line, fill='black', font=font)
            y_offset += 25

        return img


def _generate_answer(
    model,
    tokenizer,
    image_ocr_features: List[torch.Tensor],
    image_deepstack_features: Optional[List[List[torch.Tensor]]],
    question_ocr_features: List[torch.Tensor],
    question_deepstack_features: Optional[List[List[torch.Tensor]]],
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """
    Generate answer using TEXT-FREE inputs (rendered question + real image).

    Matches training format: two vision inputs concatenated.
    """
    # Qwen3-VL chat tokens (same as training collate_fn)
    user_start_ids = torch.tensor([151644, 872, 198], dtype=torch.long)  # <|im_start|>user\n
    user_end_ids = torch.tensor([151645, 198], dtype=torch.long)  # <|im_end|>\n
    assistant_start_ids = torch.tensor([151644, 77091, 198], dtype=torch.long)  # <|im_start|>assistant\n

    # Vision token placeholders for BOTH images (question + real image)
    vision_placeholder_question = "<|vision_start|>" + "<|image_pad|>" * 100 + "<|vision_end|>"
    vision_placeholder_image = "<|vision_start|>" + "<|image_pad|>" * 100 + "<|vision_end|>"

    # Tokenize both vision placeholders
    question_vision_ids = tokenizer(vision_placeholder_question, return_tensors="pt", add_special_tokens=False).input_ids[0]
    image_vision_ids = tokenizer(vision_placeholder_image, return_tensors="pt", add_special_tokens=False).input_ids[0]

    # Build input sequence: <|im_start|>user\n<question_vision><image_vision><|im_end|>\n<|im_start|>assistant\n
    # This matches training where we have TWO vision inputs
    input_ids = torch.cat([
        user_start_ids,
        question_vision_ids,
        image_vision_ids,
        user_end_ids,
        assistant_start_ids,
    ]).unsqueeze(0).to(device)

    # Prepare OCR features - concatenate question + image features (matching training)
    # Training uses: [question_features, image_features] as a list
    all_ocr_features = question_ocr_features + image_ocr_features

    if image_deepstack_features is not None and question_deepstack_features is not None:
        all_deepstack_features = question_deepstack_features + image_deepstack_features
        ocr_image_features = (all_ocr_features, all_deepstack_features)
    else:
        ocr_image_features = all_ocr_features

    # Generate using model.generate() with no_grad
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                ocr_image_features=ocr_image_features,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode (remove prompt)
        generated_ids = outputs[0][input_ids.shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer if answer else "[EMPTY]"

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "[GENERATION_FAILED]"


def _save_evaluation_results(checkpoint_dir: Path, results: List[Dict], global_step: int):
    """Save evaluation results in both JSON and human-readable formats"""
    eval_dir = checkpoint_dir / "eval_results"
    eval_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON (for programmatic analysis)
    json_path = eval_dir / f"step_{global_step}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'step': global_step,
            'timestamp': timestamp,
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    # Save human-readable text
    txt_path = eval_dir / f"step_{global_step}_results.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TEXT-FREE Checkpoint Evaluation - Step {global_step}\n")
        f.write(f"Evaluation Mode: Questions rendered as images (matching training)\n")
        f.write(f"Input Format: <Rendered Question Image> + <Real Image> → Answer\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"Sample {i}: {result['id']}\n")
            f.write(f"Image: {result['image_path']}\n")
            f.write(f"Question (rendered to image): {result['question']}\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"Generated: {result['generated_answer']}\n")
            f.write("-" * 80 + "\n\n")

    logger.info(f"  Saved evaluation results:")
    logger.info(f"    - {json_path}")
    logger.info(f"    - {txt_path}")


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0)"""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True
