#!/usr/bin/env python3
"""
Standalone TEXT-FREE Checkpoint Evaluation

Evaluates a checkpoint with rendered question images (matching training format).
Saves results in same format as checkpoint_eval.py.

Usage:
    python OCRVL/evaluate_checkpoint_textfree.py OCRVL/checkpoints/llava_20260104_201113/instruction/step_10000
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.amp import autocast
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def render_question_to_image(question: str, width=640, height=640, font_size=18) -> Image.Image:
    """Render question text to image."""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
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


def load_checkpoint_and_adapter(checkpoint_dir: Path, device: str = 'cuda'):
    """Load model, tokenizer, and OCR adapter from checkpoint."""
    from OCRVL.model.language_model.ocr_qwen3_vl import (
        OCRQwen3VLForConditionalGeneration,
        Qwen3VLOCRTextAdapter
    )
    from transformers import AutoTokenizer

    logger.info(f"Loading checkpoint from {checkpoint_dir}...")

    # Base model path (Qwen3-VL)
    base_model_path = "Qwen/Qwen3-VL-2B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Load base model
    logger.info(f"Loading base model from {base_model_path}...")
    model = OCRQwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    # Load connectors from checkpoint
    connector_path = checkpoint_dir / "connectors.pt"
    if connector_path.exists():
        logger.info(f"Loading connectors from {connector_path}...")
        state = torch.load(connector_path, map_location='cpu')

        # Load connector weights
        target_model = model.model  # OCRQwen3VLForConditionalGeneration → .model → OCRQwen3VLModel

        if 'ocr_connector' in state and hasattr(target_model, 'ocr_connector'):
            target_model.ocr_connector.load_state_dict(state['ocr_connector'])
            logger.info(f"  ✓ Loaded ocr_connector")

        if 'deepstack_connectors' in state and hasattr(target_model, '_ocr_deepstack_connectors'):
            for k, v in state['deepstack_connectors'].items():
                if k in target_model._ocr_deepstack_connectors:
                    target_model._ocr_deepstack_connectors[k].load_state_dict(v)
            logger.info(f"  ✓ Loaded deepstack_connectors")
    else:
        logger.warning(f"No connectors.pt found in {checkpoint_dir}")

    # Check for LoRA adapters
    lora_dir = checkpoint_dir / "lora_adapters"
    if lora_dir.exists():
        try:
            from peft import PeftModel
            logger.info(f"Loading LoRA adapters from {lora_dir}...")
            model = PeftModel.from_pretrained(model, str(lora_dir))
            logger.info(f"  ✓ Loaded LoRA adapters")
        except Exception as e:
            logger.warning(f"Failed to load LoRA adapters: {e}")

    model.eval()

    # Initialize OCR adapter (for encoding images)
    ocr_adapter = Qwen3VLOCRTextAdapter(
        encoder_model_path="deepseek-ai/DeepSeek-OCR",
        device=device,
        dtype=torch.bfloat16,
        use_deepstack=True,
    )

    logger.info("✓ Model and adapter loaded")
    return model, tokenizer, ocr_adapter


def test_ocr_transcription(
    model,
    tokenizer,
    ocr_adapter,
    test_texts: List[str],
    device: str = 'cuda',
    max_new_tokens: int = 128,
    debug: bool = False,
) -> List[Dict]:
    """
    Test if model can transcribe rendered text (basic OCR capability).

    This tests whether the connectors have learned to decode visual OCR tokens
    back into semantic text understanding.
    """
    results = []

    for i, text in enumerate(test_texts, 1):
        logger.info(f"Testing OCR transcription {i}/{len(test_texts)}: '{text[:50]}...'")

        # Render text to image
        rendered_image = render_question_to_image(text)

        # Encode with OCR encoder
        encoded = ocr_adapter.images_to_ocr_features([rendered_image])
        if isinstance(encoded, tuple):
            ocr_features, deepstack_features = encoded
        else:
            ocr_features = encoded
            deepstack_features = None

        # Generate transcription
        transcription = generate_ocr_transcription(
            model=model,
            tokenizer=tokenizer,
            ocr_features=ocr_features,
            deepstack_features=deepstack_features,
            device=device,
            max_new_tokens=max_new_tokens,
            debug=debug,
        )

        results.append({
            'input_text': text,
            'transcription': transcription,
            'correct': text.lower().strip() in transcription.lower() or transcription.lower().strip() in text.lower(),
        })

        logger.info(f"  Input:  '{text}'")
        logger.info(f"  Output: '{transcription}'")
        logger.info(f"  Match:  {'✓' if results[-1]['correct'] else '✗'}")

    return results


def generate_ocr_transcription(
    model,
    tokenizer,
    ocr_features: List[torch.Tensor],
    deepstack_features: Optional[List[List[torch.Tensor]]],
    device: str = 'cuda',
    max_new_tokens: int = 128,
    debug: bool = False,
) -> str:
    """Generate OCR transcription with explicit instruction."""
    # Qwen3-VL chat tokens
    user_start_ids = torch.tensor([151644, 872, 198], dtype=torch.long)
    user_end_ids = torch.tensor([151645, 198], dtype=torch.long)
    assistant_start_ids = torch.tensor([151644, 77091, 198], dtype=torch.long)

    # Vision placeholder
    vision_placeholder = "<|vision_start|>" + "<|image_pad|>" * 100 + "<|vision_end|>"
    vision_ids = tokenizer(vision_placeholder, return_tensors="pt", add_special_tokens=False).input_ids[0]

    # Transcription prompt
    prompt = "Read and transcribe the text shown in the image:"
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]

    # Build input: <|im_start|>user\n<vision><prompt><|im_end|>\n<|im_start|>assistant\n
    input_ids = torch.cat([
        user_start_ids,
        vision_ids,
        prompt_ids,
        user_end_ids,
        assistant_start_ids,
    ]).unsqueeze(0).to(device)

    if debug:
        logger.info(f"[OCR DEBUG] Input shape: {input_ids.shape}")
        logger.info(f"[OCR DEBUG] Features shape: {ocr_features[0].shape}")

    # Prepare features
    if deepstack_features is not None:
        ocr_image_features = (ocr_features, deepstack_features)
    else:
        ocr_image_features = ocr_features

    # Generate
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                ocr_image_features=ocr_image_features,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_ids.shape[1]:]
        transcription = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if debug:
            logger.info(f"[OCR DEBUG] Generated {len(generated_ids)} tokens")
            logger.info(f"[OCR DEBUG] Transcription: '{transcription}'")

        return transcription if transcription else "[EMPTY]"

    except Exception as e:
        logger.error(f"OCR transcription failed: {e}")
        return "[FAILED]"


def evaluate_sample(
    model,
    tokenizer,
    ocr_adapter,
    sample: Dict,
    image_base_dir: str,
    device: str = 'cuda',
    prompt: str = '',
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    debug: bool = False,
) -> Dict:
    """Evaluate a single sample with TEXT-FREE input."""
    # Load real image
    image_path = sample['image_path']
    full_path = Path(image_base_dir) / image_path

    # Try parquet loader first
    try:
        from OCRVL.data.parquet_image_loader import Mix665kImageLoader
        loader = Mix665kImageLoader(image_base_dir)
        real_image = loader.load_image(image_path)
    except:
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        real_image = Image.open(full_path).convert('RGB')

    # Render question to image
    question = sample['question']
    question_image = render_question_to_image(question)

    # Encode both images using the adapter's method
    encoded_question = ocr_adapter.images_to_ocr_features([question_image])
    encoded_image = ocr_adapter.images_to_ocr_features([real_image])

    # Handle the return format (can be tuple with deepstack or just features)
    if isinstance(encoded_question, tuple):
        question_ocr_features, question_deepstack_features = encoded_question
    else:
        question_ocr_features = encoded_question
        question_deepstack_features = None

    if isinstance(encoded_image, tuple):
        image_ocr_features, image_deepstack_features = encoded_image
    else:
        image_ocr_features = encoded_image
        image_deepstack_features = None

    # Generate answer (TEXT-FREE)
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        generated_answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question_ocr_features=question_ocr_features,
            question_deepstack_features=question_deepstack_features,
            image_ocr_features=image_ocr_features,
            image_deepstack_features=image_deepstack_features,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            debug=debug,
        )

    return {
        'id': sample['id'],
        'image_path': image_path,
        'question': question,
        'ground_truth': sample.get('answer', 'N/A'),
        'generated_answer': generated_answer,
    }


def generate_answer(
    model,
    tokenizer,
    question_ocr_features: List[torch.Tensor],
    question_deepstack_features: List[List[torch.Tensor]],
    image_ocr_features: List[torch.Tensor],
    image_deepstack_features: List[List[torch.Tensor]],
    device: str = 'cuda',
    max_new_tokens: int = 128,
    prompt: str = '',
    temperature: float = 0.0,
    debug: bool = False,
) -> str:
    """Generate answer with TEXT-FREE inputs (question image + real image) + optional text prompt."""
    # Qwen3-VL chat tokens
    user_start_ids = torch.tensor([151644, 872, 198], dtype=torch.long)  # <|im_start|>user\n
    user_end_ids = torch.tensor([151645, 198], dtype=torch.long)  # <|im_end|>\n
    assistant_start_ids = torch.tensor([151644, 77091, 198], dtype=torch.long)  # <|im_start|>assistant\n

    # Two vision placeholders (question + image)
    # OCR encoder outputs 100 tokens per image (10x10 grid, no newlines in this version)
    vision_placeholder = "<|vision_start|>" + "<|image_pad|>" * 100 + "<|vision_end|>"
    question_vision_ids = tokenizer(vision_placeholder, return_tensors="pt", add_special_tokens=False).input_ids[0]
    image_vision_ids = tokenizer(vision_placeholder, return_tensors="pt", add_special_tokens=False).input_ids[0]

    # Tokenize prompt if provided
    if prompt:
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    else:
        prompt_ids = torch.tensor([], dtype=torch.long)

    # Build input: <|im_start|>user\n<question_vision><image_vision><prompt><|im_end|>\n<|im_start|>assistant\n
    # This allows: pure vision-only (no prompt) OR vision + guiding text
    input_ids = torch.cat([
        user_start_ids,
        question_vision_ids,
        image_vision_ids,
        prompt_ids,
        user_end_ids,
        assistant_start_ids,
    ]).unsqueeze(0).to(device)

    if debug:
        logger.info(f"[DEBUG] Input shape: {input_ids.shape}")
        logger.info(f"[DEBUG] Prompt tokens: {len(prompt_ids)}")
        logger.info(f"[DEBUG] Question features shape: {question_ocr_features[0].shape}")
        logger.info(f"[DEBUG] Image features shape: {image_ocr_features[0].shape}")
        logger.info(f"[DEBUG] Num OCR features: {len(question_ocr_features) + len(image_ocr_features)}")

    # Concatenate features: [question, image]
    all_ocr_features = question_ocr_features + image_ocr_features
    all_deepstack_features = question_deepstack_features + image_deepstack_features
    ocr_image_features = (all_ocr_features, all_deepstack_features)

    # Generate
    try:
        gen_kwargs = {
            'input_ids': input_ids,
            'ocr_image_features': ocr_image_features,
            'max_new_tokens': max_new_tokens,
            'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }

        if temperature > 0:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = temperature
        else:
            gen_kwargs['do_sample'] = False

        outputs = model.generate(**gen_kwargs)

        # Decode
        generated_ids = outputs[0][input_ids.shape[1]:]

        if debug:
            logger.info(f"[DEBUG] Generated {len(generated_ids)} tokens")
            logger.info(f"[DEBUG] Generated IDs: {generated_ids.tolist()[:20]}")  # First 20 tokens

        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if debug:
            logger.info(f"[DEBUG] Decoded answer: '{answer}'")

        return answer if answer else "[EMPTY]"

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "[GENERATION_FAILED]"


def save_results(results: List[Dict], checkpoint_dir: Path, global_step: int, prompt: str = ''):
    """Save results in same format as checkpoint_eval.py."""
    eval_dir = checkpoint_dir / "eval_results"
    eval_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON
    json_path = eval_dir / f"step_{global_step}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'step': global_step,
            'timestamp': timestamp,
            'prompt': prompt,
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    # Save human-readable text
    txt_path = eval_dir / f"step_{global_step}_results.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TEXT-FREE Checkpoint Evaluation - Step {global_step}\n")
        f.write(f"Evaluation Mode: Questions rendered as images (matching training)\n")
        f.write(f"Input Format: <Rendered Question Image> + <Real Image>")
        if prompt:
            f.write(f" + Text Prompt")
        f.write(f" → Answer\n")
        if prompt:
            f.write(f"Guiding Prompt: \"{prompt}\"\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"Sample {i}: {result['id']}\n")
            f.write(f"Image: {result['image_path']}\n")
            f.write(f"Question (rendered to image): {result['question']}\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"Generated: {result['generated_answer']}\n")
            f.write("-" * 80 + "\n\n")

    logger.info(f"✓ Results saved:")
    logger.info(f"  - {json_path}")
    logger.info(f"  - {txt_path}")


def save_ocr_results(results: List[Dict], checkpoint_dir: Path, global_step: int):
    """Save OCR transcription test results."""
    eval_dir = checkpoint_dir / "eval_results"
    eval_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON
    json_path = eval_dir / f"step_{global_step}_ocr_test.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'step': global_step,
            'timestamp': timestamp,
            'test_type': 'ocr_transcription',
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    # Save human-readable text
    txt_path = eval_dir / f"step_{global_step}_ocr_test.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"OCR Transcription Test - Step {global_step}\n")
        f.write(f"Test: Can the model read rendered text?\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        correct_count = sum(1 for r in results if r['correct'])
        f.write(f"Accuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"Test {i}:\n")
            f.write(f"  Input text:     '{result['input_text']}'\n")
            f.write(f"  Transcription:  '{result['transcription']}'\n")
            f.write(f"  Match:          {'✓ CORRECT' if result['correct'] else '✗ WRONG'}\n")
            f.write("-" * 80 + "\n")

    logger.info(f"✓ OCR test results saved:")
    logger.info(f"  - {json_path}")
    logger.info(f"  - {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="TEXT-FREE checkpoint evaluation")
    parser.add_argument('checkpoint_dir', type=str, help='Path to checkpoint directory (e.g., step_10000)')
    parser.add_argument('--eval-samples', type=str, default='OCRVL/data/eval_samples.json',
                        help='Path to eval samples JSON')
    parser.add_argument('--image-base-dir', type=str,
                        default='/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/images',
                        help='Base directory for images')
    parser.add_argument('--max-samples', type=int, default=10, help='Maximum samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--prompt', type=str, default='',
                        help='Text prompt to guide the model (e.g., "Answer the question in the first image based on the second image.")')
    parser.add_argument('--max-new-tokens', type=int, default=128,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0 = greedy)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--ocr-test', action='store_true',
                        help='Test basic OCR transcription instead of VQA (renders text and asks model to read it)')

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Extract step number from checkpoint directory name
    checkpoint_name = checkpoint_dir.name
    step_str = checkpoint_name.replace('step_', '')

    # Try to parse step number, fallback to reading from trainer_state.json
    if step_str.isdigit():
        global_step = int(step_str)
    else:
        # Handle "step_latest" or other non-numeric names by reading trainer_state.json
        trainer_state_path = checkpoint_dir / 'trainer_state.json'
        if trainer_state_path.exists():
            import json as json_mod
            with open(trainer_state_path, 'r') as f:
                trainer_state = json_mod.load(f)
                global_step = trainer_state.get('global_step', 0)
            logger.info(f"Extracted global_step={global_step} from trainer_state.json")
        else:
            global_step = 0
            logger.warning(f"Could not extract step number from '{checkpoint_name}', using global_step=0")

    # Load eval samples
    eval_samples_path = Path(args.eval_samples)
    if not eval_samples_path.exists():
        raise FileNotFoundError(f"Eval samples not found: {eval_samples_path}")

    with open(eval_samples_path, 'r') as f:
        eval_samples = json.load(f)[:args.max_samples]

    logger.info(f"Evaluating checkpoint: {checkpoint_dir}")
    logger.info(f"Eval samples: {len(eval_samples)}")

    if args.ocr_test:
        logger.info(f"Mode: OCR TRANSCRIPTION TEST (testing basic OCR capability)")
    else:
        logger.info(f"Mode: TEXT-FREE VQA (questions rendered as images)")

    if args.prompt and not args.ocr_test:
        logger.info(f"Guiding prompt: '{args.prompt}'")

    # Load model and adapter
    model, tokenizer, ocr_adapter = load_checkpoint_and_adapter(checkpoint_dir, args.device)

    # Run OCR test or VQA evaluation
    if args.ocr_test:
        # OCR transcription test - use questions from eval samples as test texts
        test_texts = [sample['question'] for sample in eval_samples]

        logger.info("Running OCR transcription test...")
        ocr_results = test_ocr_transcription(
            model=model,
            tokenizer=tokenizer,
            ocr_adapter=ocr_adapter,
            test_texts=test_texts,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            debug=args.debug,
        )

        # Save OCR test results
        save_ocr_results(ocr_results, checkpoint_dir, global_step)

        # Print summary
        correct = sum(1 for r in ocr_results if r['correct'])
        logger.info(f"✓ OCR test complete: {correct}/{len(ocr_results)} correct transcriptions")

    else:
        # Standard TEXT-FREE VQA evaluation
        results = []
        for i, sample in enumerate(eval_samples, 1):
            try:
                logger.info(f"Evaluating sample {i}/{len(eval_samples)}: {sample['id']}")
                result = evaluate_sample(
                    model=model,
                    tokenizer=tokenizer,
                    ocr_adapter=ocr_adapter,
                    sample=sample,
                    image_base_dir=args.image_base_dir,
                    device=args.device,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    debug=args.debug,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate sample {sample.get('id', 'unknown')}: {e}")
                continue

        # Save results
        save_results(results, checkpoint_dir, global_step, prompt=args.prompt)

        logger.info(f"✓ TEXT-FREE evaluation complete: {len(results)}/{len(eval_samples)} samples")


if __name__ == '__main__':
    main()
