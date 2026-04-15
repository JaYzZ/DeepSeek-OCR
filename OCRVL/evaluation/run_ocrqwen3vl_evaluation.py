#!/usr/bin/env python3
"""
Unified OCRQwen3VL & Qwen3VL Evaluation Script

This script evaluates models on various benchmarks using vLLM with native multimodal support.

Supported Models:
1. OCRQwen3VL - End-to-end model with LoRA support
2. Qwen3VL (official) - Baseline Qwen3-VL model

Supported Benchmarks:
- M3CoT: Multimodal Chain-of-Thought reasoning
- ScienceQA: Science question answering
- MathVision: Math problems with images
- RealWorldQA: Real-world visual questions
- MMMU: Multimodal massive understanding
- ODinW: Object detection in the wild

Usage:
    # OCRQwen3VL with LoRA
    python -m OCRVL.evaluation.run_ocrqwen3vl_evaluation \\
        --benchmark scienceqa \\
        --model-type ocrqwen3vl \\
        --lora-path checkpoints/lora/... \\
        --output results/scienceqa.jsonl

    # Official Qwen3VL baseline
    python -m OCRVL.evaluation.run_ocrqwen3vl_evaluation \\
        --benchmark scienceqa \\
        --model-type qwen3vl \\
        --checkpoint Qwen/Qwen3-VL-2B-Instruct \\
        --output results/scienceqa.jsonl
"""

import argparse
import json
import logging
import os
from pathlib import Path
import re
import traceback
from typing import Any, Dict, List, Optional

from OCRVL.decoder import OCRQwen3VLProcessor
from datasets import load_dataset
from PIL import Image
from Renderer import VelloRenderer
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Default paths
DEFAULT_CHECKPOINT = "OCRVL/checkpoints/OCR-Qwen3-VL-2B"
DEFAULT_LORA_PATH = "OCRVL/checkpoints/llamafactory/qwen3vl-2b/lora/run_20260120_045134/checkpoint-8607"


def load_dataset(benchmark: str, split: str = "test"):
    """Load benchmark dataset from HuggingFace."""
    # Set shared cache
    repo_root = Path(__file__).resolve().parents[2]
    shared_cache = repo_root.parent / 'huggingface' / 'cache'
    if shared_cache.exists():
        os.environ['HF_DATASETS_CACHE'] = str(shared_cache)

    datasets = {
        "m3cot": ("LightChen2333/M3CoT", split),
        "scienceqa": ("derek-thomas/ScienceQA", split),
        "mathvision": ("AI4Math/MathVista", "testmini"),  # Use testmini for faster eval
        "realworldqa": ("ariusyyy/RealWorldQA", split),
        "mmmu": ("currken/MMMU", "validation"),  # MMMU only has validation
        "odinw": ("processors-community/OdinW", "test"),
    }

    if benchmark not in datasets:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(datasets.keys())}")

    dataset_name, dataset_split = datasets[benchmark]
    logger.info(f"Loading {benchmark} dataset: {dataset_name} ({dataset_split})")
    dataset = load_dataset(dataset_name, split=dataset_split)
    logger.info(f"  Loaded {len(dataset)} samples")
    return dataset


def format_prompt(benchmark: str, item: Dict) -> str:
    """Format prompt for specific benchmark."""
    if benchmark == "m3cot":
        question = item['question']
        choices = item['choices']
        text = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            text += f"{chr(65+i)}. {choice}\n"
        text += "Answer:"
        return text

    elif benchmark == "scienceqa":
        question = item['question']
        choices = item.get('choices', [])
        text = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            text += f"{chr(65+i)}. {choice}\n"
        text += "Answer:"
        return text

    elif benchmark == "mathvision":
        question = item['question']
        return f"Question: {question}\nAnswer:"

    elif benchmark == "realworldqa":
        question = item['question']
        return f"Question: {question}\nAnswer:"

    elif benchmark == "mmmu":
        question = item['question']
        choices = item.get('choices', [])
        text = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            text += f"{chr(65+i)}. {choice}\n"
        text += "Answer:"
        return text

    elif benchmark == "odinw":
        # ODinW is object detection, format differently
        question = item.get('question', 'What objects are in this image?')
        return f"Question: {question}\nAnswer:"

    else:
        return "What do you see?\nAnswer:"


def format_question_for_rendering(benchmark: str, item: Dict) -> str:
    """Format question text for rendering as image."""
    if benchmark in ["m3cot", "scienceqa", "mmmu"]:
        # Multiple choice format
        question = item['question']
        choices = item.get('choices', [])
        text = f"{question}\n"
        for i, choice in enumerate(choices):
            text += f"({chr(65+i)}) {choice}"
        return text

    elif benchmark == "mathvision":
        return item['question']

    elif benchmark == "realworldqa":
        return item['question']

    elif benchmark == "odinw":
        return item.get('question', 'What objects are in this image?')

    else:
        return item.get('question', 'What do you see?')


def extract_answer(benchmark: str, prediction: str, item: Dict) -> Optional[str]:
    """Extract answer from prediction based on benchmark format."""
    prediction = str(prediction).strip().upper()

    if benchmark in ["m3cot", "scienceqa", "mmmu"]:
        # Multiple choice: extract letter
        num_choices = len(item.get('choices', []))
        valid_choices = [chr(65 + i) for i in range(num_choices)]

        # Look for last occurrence of valid choice
        for char in reversed(prediction):
            if char in valid_choices:
                return char

        # Fallback: check if prediction ends with choice
        if prediction and prediction[-1] in valid_choices:
            return prediction[-1]

    elif benchmark == "mathvision":
        # Extract final answer from math solution
        match = re.search(r'(?:The answer is|Therefore|Thus|Answer)[:\s]*([A-Z]?\d*\.?\d*)', prediction, re.IGNORECASE)
        if match:
            return match.group(1)

    elif benchmark == "realworldqa":
        # Short answer, just return prediction
        return prediction[:100]

    return prediction[:50]  # Truncate long answers


def evaluate_benchmark(
    benchmark: str,
    checkpoint_path: str,
    model_type: str,
    lora_path: Optional[str],
    output_file: Path,
    max_samples: Optional[int],
    max_tokens: int,
    gpu_memory_utilization: float,
    batch_size: int,
    render_questions: bool = False,
):
    """Evaluate model on a benchmark.

    Supports:
    - OCRQwen3VL: Custom model with optional LoRA
    - Qwen3VL: Official Qwen3-VL model (baseline)

    Args:
        render_questions: If True, render questions as images (OCRQwen3VL only).
                        This enables the training pattern: vision + rendered_q + instruction → answer
    """
    logger.info("=" * 80)
    logger.info(f"Evaluation: {benchmark.upper()}")
    logger.info(f"Model Type: {model_type.upper()}")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"LoRA: {lora_path if lora_path else 'None'}")
    logger.info(f"Render Questions: {render_questions}")
    logger.info(f"Output: {output_file}")
    logger.info("")

    # Load dataset
    dataset = load_dataset(benchmark)

    # Limit samples for testing
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {max_samples} samples for testing")

    # Initialize processor based on model type
    if model_type == "ocrqwen3vl":
        logger.info("Initializing OCRQwen3VL processor...")
        processor = OCRQwen3VLProcessor(
            checkpoint_path=checkpoint_path,
            lora_path=lora_path,
            enable_lora=lora_path is not None,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )

        # Wrapper function for generate API
        def generate_func(images, prompts, max_t):
            return processor.generate(images=images, prompts=prompts, max_tokens=max_t)

        logger.info("")

        # Initialize Vello renderer for question rendering if enabled
        renderer = None
        if render_questions:
            logger.info("Initializing Vello renderer for question rendering...")
            renderer = VelloRenderer(
                width=640,
                height=640,
                padding=20,
                min_font_size=10,
                max_font_size=18,
                preserve_newlines=True,
            )
            logger.info(f"  ✓ Vello renderer ready: {renderer}")

    elif model_type == "qwen3vl":
        # Use official Qwen3VL with vLLM
        logger.info("Initializing official Qwen3VL with vLLM...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

        # LoRA not supported for official Qwen3VL baseline
        if lora_path:
            logger.warning("  Note: LoRA adapters not supported for official Qwen3VL baseline")
            logger.warning("  --lora-path will be ignored")

        llm = LLM(
            model=checkpoint_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
            disable_log_stats=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 20},
        )

        # Wrapper to match OCRQwen3VLProcessor API
        class Qwen3VLProcessor:
            def __init__(self, llm, tokenizer):
                self.llm = llm
                self.tokenizer = tokenizer

            def generate(self, images, prompts, max_tokens, temperature=0.0):
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=["<|im_end|>", ""],
                )

                # Build vLLM inputs
                vllm_inputs = []
                for img, prompt_text in zip(images, prompts):
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text}
                        ]
                    }]

                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    vllm_inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": img},
                    })

                outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params)
                return [output.outputs[0].text for output in outputs]

            def shutdown(self):
                try:
                    if hasattr(self.llm, "llm_engine"):
                        self.llm.llm_engine.shutdown()
                except:
                    pass

        processor = Qwen3VLProcessor(llm, tokenizer)

        def generate_func(images, prompts, max_t):
            return processor.generate(images, prompts, max_t)

        logger.info("")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose: ocrqwen3vl, qwen3vl")

    # Process in batches
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results = []

    logger.info(f"Running inference on {len(dataset)} samples...")

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"{benchmark.upper()}"):
        end_idx = min(i + batch_size, len(dataset))
        batch_items = [dataset[j] for j in range(i, end_idx)]

        # Prepare inputs
        images = []
        prompts = []
        metadatas = []

        # Render questions in batch if enabled
        if renderer is not None:
            question_texts = [format_question_for_rendering(benchmark, item) for item in batch_items]
            rendered_images = renderer.render_batch_pil(question_texts)

        for idx, item in enumerate(batch_items):
            # Get image
            img = item.get('image')
            if img is None:
                # Create blank image if none provided
                img = Image.new('RGB', (640, 640), color='white')

            # Format prompt
            if renderer is not None:
                # Question rendering mode: prompt is simple instruction
                prompt_text = "Answer:"
                # Pass both original image and rendered question image
                img_list = [img, rendered_images[idx]]
            else:
                # Standard mode: prompt contains the full question
                prompt_text = format_prompt(benchmark, item)
                img_list = [img]

            images.append(img_list)
            prompts.append(prompt_text)
            metadatas.append({
                'id': item.get('id', f"{i}"),
                'question': item.get('question', ''),
                'choices': item.get('choices', []),
                'answer': item.get('answer', ''),
            })

        # Generate
        try:
            outputs = generate_func(
                images=images,
                prompts=prompts,
                max_t=max_tokens,
            )

            # Collect results
            for metadata, output in zip(metadatas, outputs):
                prediction = output.strip()
                extracted_answer = extract_answer(benchmark, prediction, metadata)

                results.append({
                    **metadata,
                    'prediction': prediction,
                    'extracted_answer': extracted_answer,
                })

        except Exception as e:
            logger.error(f"Error processing batch {i}-{end_idx}: {e}")
            # Add placeholder results for failed batch
            for metadata in metadatas:
                results.append({
                    **metadata,
                    'prediction': f"ERROR: {str(e)}",
                    'extracted_answer': None,
                })

        # Save incremental results
        if (i + batch_size) % (batch_size * 10) == 0:
            with open(output_file, 'w') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')

    # Save final results
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"\nResults saved to: {output_file}")

    # Compute accuracy if ground truth available
    if 'answer' in dataset.column_names or any('answer' in r for r in results):
        correct = 0
        total = 0

        for r in results:
            gt = r.get('answer', '').strip().upper()
            pred = r.get('extracted_answer', '').strip().upper()

            if gt and pred:
                total += 1
                if gt == pred:
                    correct += 1

        if total > 0:
            accuracy = correct / total
            logger.info(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")

            # Save accuracy
            with open(output_file.parent / 'accuracy.json', 'w') as f:
                json.dump({
                    'benchmark': benchmark,
                    'checkpoint': checkpoint_path,
                    'lora': lora_path,
                    'correct': correct,
                    'total': total,
                    'accuracy': accuracy,
                }, f, indent=2)

    # Cleanup
    processor.shutdown()
    if renderer is not None:
        renderer.shutdown()

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCRQwen3VL on various benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on ScienceQA without LoRA
  python -m OCRVL.evaluation.run_ocrqwen3vl_evaluation --benchmark scienceqa --output results/scienceqa.jsonl

  # Evaluate on M3CoT with LoRA
  python -m OCRVL.evaluation.run_ocrqwen3vl_evaluation --benchmark m3cot --lora-path {default_lora} --output results/m3cot.jsonl

  # Quick test (10 samples)
  python -m OCRVL.evaluation.run_ocrqwen3vl_evaluation --benchmark scienceqa --max-samples 10 --output results/test.jsonl
        """.format(default_lora=DEFAULT_LORA_PATH)
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["m3cot", "scienceqa", "mathvision", "realworldqa", "mmmu", "odinw"],
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ocrqwen3vl",
        choices=["ocrqwen3vl", "qwen3vl"],
        help="Model type: ocrqwen3vl (default) or qwen3vl (official baseline)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help=f"Path to LoRA adapters for ocrqwen3vl (default: {DEFAULT_LORA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for predictions",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit to N samples for testing",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.85,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--render-questions",
        action="store_true",
        help="Render questions as images (OCRQwen3VL only). Enables training pattern: vision + rendered_q + instruction → answer",
    )

    args = parser.parse_args()

    # Validate: LoRA only supported for ocrqwen3vl
    if args.model_type == "qwen3vl" and args.lora_path is not None:
        logger.warning("Note: --lora-path is not supported for qwen3vl baseline")
        logger.warning("  Ignoring --lora-path argument")
        args.lora_path = None

    # Validate: render_questions only supported for ocrqwen3vl
    if args.render_questions and args.model_type != "ocrqwen3vl":
        logger.warning("Note: --render-questions is only supported for ocrqwen3vl model")
        logger.warning("  Ignoring --render-questions argument")
        args.render_questions = False

    # Set default LoRA path for ocrqwen3vl if not specified
    if args.model_type == "ocrqwen3vl" and args.lora_path is None:
        lora_env = os.environ.get("OCRQWEN3VL_LORA_PATH")
        if lora_env:
            args.lora_path = lora_env
        elif Path(DEFAULT_LORA_PATH).exists():
            logger.info(f"Found default LoRA at {DEFAULT_LORA_PATH}")
            args.lora_path = DEFAULT_LORA_PATH

    try:
        evaluate_benchmark(
            benchmark=args.benchmark,
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            lora_path=args.lora_path,
            output_file=args.output,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory,
            batch_size=args.batch_size,
            render_questions=args.render_questions,
        )
        return 0
    except Exception as e:
        logger.error(f"\nEvaluation failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
