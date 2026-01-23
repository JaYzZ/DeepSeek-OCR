#!/usr/bin/env python3
"""
One-stop script for R1-Onevision dataset setup and conversion.

This script handles:
1. Verification of model checkpoint and data
2. Conversion of R1-Onevision parquet files to OCRVL JSONL format
3. Generates dual-format samples (text-based + rendered-based)
4. V↔Q image randomization for robustness

Usage:
    # Full setup for all datasets
    python OCRVL/scripts/build_r1_onevision.py --all

    # Setup specific datasets
    python OCRVL/scripts/build_r1_onevision.py --dataset ai2d doc_vqa

    # Quick test with limited samples
    python OCRVL/scripts/build_r1_onevision.py --all --max-samples 100
"""

import argparse
import hashlib
import json
import base64
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd

# Add parent directory to path for imports
script_dir = Path(__file__).parent.parent
# Also add repo root for Renderer module
repo_root = script_dir.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(repo_root))

try:
    from Renderer import VelloRenderer
    VELLO_AVAILABLE = True
except ImportError:
    VELLO_AVAILABLE = None

# Instruction templates - aligned with unified SFT (build_vqa.py)
VQA_INSTRUCTION_TEMPLATES = [
    "Answer the question:",
    "Respond to the question:",
    "Provide an answer:",
    "Answer based on the image:",
]


def verify_setup(base_dir: Path) -> bool:
    """Verify model checkpoint and dependencies."""
    print("=" * 70)
    print("STEP 0: Verifying Setup")
    print("=" * 70)
    print()

    # Check model checkpoint
    checkpoint_path = base_dir / "OCRVL/checkpoints/OCR-Qwen3-VL-2B-Thinking"
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    print(f"✓ Checkpoint exists: {checkpoint_path}")

    # Check think tokens
    tokenizer_config_path = checkpoint_path / "tokenizer_config.json"
    with open(tokenizer_config_path) as f:
        tokenizer_config = json.load(f)

    added_tokens = tokenizer_config.get("added_tokens_decoder", {})
    has_think_tokens = '151667' in added_tokens and '151668' in added_tokens

    if has_think_tokens:
        print(f"✓ Think tokens present (151667, 151668)")
    else:
        print(f"✗ Think tokens missing")
        return False

    # Check dataset_info.json
    dataset_info_path = base_dir / "OCRVL/llamafactory/data/dataset_info.json"
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    if "ocrvl_r1_onevision" in dataset_info:
        print(f"✓ Dataset info entry exists")
    else:
        print(f"✗ Dataset info entry missing")
        return False

    print()
    return True


def get_hash_filename(dataset_name: str, sample_id: str, img_type: str) -> str:
    """Generate deterministic hash-based filename with dataset prefix."""
    # Use SHA256 for deterministic, collision-resistant filenames
    hash_input = f"{sample_id}_{img_type}".encode('utf-8')
    hash_hex = hashlib.sha256(hash_input).hexdigest()[:12]
    # Format: {dataset}_{hash}.png
    return f"{dataset_name}_{hash_hex}.png"


def decode_and_save_image(base64_string: str, output_dir: Path, dataset_name: str, sample_id: str, img_type: str) -> str:
    """Decode base64 image and save to disk with hash-based filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]

    image_data = base64.b64decode(base64_string)

    # Always use PNG for consistency
    filename = get_hash_filename(dataset_name, sample_id, img_type)
    image_path = output_dir / filename

    with open(image_path, 'wb') as f:
        f.write(image_data)

    return str(image_path)


def extract_thinking_and_answer(content: str) -> Tuple[str, str]:
    """Extract thinking section and final answer from R1-Onevision format."""
    content = content.lstrip('\n')
    parts = content.split('\n\n', 2)

    if len(parts) >= 3:
        thinking = parts[0].strip()
        answer = parts[2].strip()
    elif len(parts) == 2:
        thinking = parts[0].strip()
        answer = parts[1].strip()
    else:
        thinking = ""
        answer = content.strip()

    return thinking, answer


def render_text_to_image(
    text: str,
    renderer: 'VelloRenderer',
    output_dir: Path,
    dataset_name: str,
    sample_id: str,
    img_type: str
) -> Optional[str]:
    """Render text as an image using VelloRenderer with hash-based filename."""
    if not text or len(text.strip()) < 5:
        return None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        images = renderer.render_batch([text])

        if images and len(images) > 0:
            from PIL import Image
            import numpy as np

            img_array = images[0]
            if isinstance(img_array, np.ndarray):
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = img_array

            filename = get_hash_filename(dataset_name, sample_id, img_type)
            image_path = output_dir / filename
            pil_image.save(image_path)

            return str(image_path)

    except Exception as e:
        pass  # Silent fail for rendering

    return None


def create_text_based_sample(
    dataset_name: str,
    sample_id: str,
    question_text: str,
    thinking: str,
    answer: str,
    main_image_path: str,
    images_dir: Path,
    renderer: Optional['VelloRenderer']
) -> Optional[dict]:
    """Create text-based training sample."""
    images = [main_image_path]
    user_content = f"<image>\n{question_text}"

    # Render thinking (single image for both sample types)
    if thinking and renderer:
        thinking_img = render_text_to_image(
            thinking,
            renderer,
            images_dir,
            dataset_name,
            sample_id,
            "thinking"
        )
        if thinking_img:
            images.append(thinking_img)
            think_start = ''.join(chr(c) for c in [0x15, 0x01, 0x1d, 0x00, 0x00, 0x00, 0x3e])
            think_end = ''.join(chr(c) for c in [0x15, 0x00, 0x00, 0x00, 0x01, 0x1d, 0x00, 0x3e])
            assistant_content = think_start + "\n<image>\n" + think_end + "\n" + answer
        else:
            assistant_content = answer
    else:
        assistant_content = answer

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": images,
        "task": "text_vqa",
        "sample_type": "text_based"
    }


def create_rendered_based_sample(
    dataset_name: str,
    sample_id: str,
    question_text: str,
    thinking: str,
    answer: str,
    main_image_path: str,
    images_dir: Path,
    renderer: Optional['VelloRenderer']
) -> Optional[dict]:
    """Create rendered-based training sample with V↔Q randomization."""
    # Render question as image
    question_img = render_text_to_image(
        question_text,
        renderer,
        images_dir,
        dataset_name,
        sample_id,
        "question"
    )

    if not question_img:
        return None

    # Randomize V and Q image order (50% chance to swap)
    if random.random() < 0.5:
        images = [question_img, main_image_path]
        image_order = "QV"
    else:
        images = [main_image_path, question_img]
        image_order = "VQ"

    user_content = "<image>\n<image>\nAnswer the question shown in the images."

    # Render thinking (same image as text-based sample)
    if thinking and renderer:
        thinking_img = render_text_to_image(
            thinking,
            renderer,
            images_dir,
            dataset_name,
            sample_id,
            "thinking"  # Same key for both sample types
        )
        if thinking_img:
            images.append(thinking_img)
            think_start = ''.join(chr(c) for c in [0x15, 0x01, 0x1d, 0x00, 0x00, 0x00, 0x3e])
            think_end = ''.join(chr(c) for c in [0x15, 0x00, 0x00, 0x00, 0x01, 0x1d, 0x00, 0x3e])
            assistant_content = think_start + "\n<image>\n" + think_end + "\n" + answer
        else:
            assistant_content = answer
    else:
        assistant_content = answer

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": images,
        "task": "rendered_vqa",
        "sample_type": "rendered_based",
        "image_order": image_order
    }


def batch_render_texts(
    texts: List[str],
    renderer: 'VelloRenderer',
    images_dir: Path,
    dataset_name: str,
    sample_ids: List[str],
    img_types: List[str]
) -> dict:
    """Batch render multiple texts to images efficiently."""
    if not texts or not renderer:
        return {}

    # Render all texts in one batch
    images = renderer.render_batch(texts)

    result = {}
    for text, img_array, sample_id, img_type in zip(texts, images, sample_ids, img_types):
        if img_array is None:
            continue
        try:
            from PIL import Image
            import numpy as np

            if isinstance(img_array, np.ndarray):
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = img_array

            filename = get_hash_filename(dataset_name, sample_id, img_type)
            image_path = images_dir / filename
            pil_image.save(image_path)

            # Store by text for lookup
            result[text] = str(image_path)
        except Exception as e:
            pass

    return result


def process_parquet_file(
    parquet_path: Path,
    output_jsonl: Path,
    images_dir: Path,
    dataset_name: str,
    render_thinking: bool = True,
    renderer: Optional['VelloRenderer'] = None,
    max_samples: Optional[int] = None
) -> dict:
    """Process a single parquet file with optimized batch rendering."""
    df = pd.read_parquet(parquet_path)

    if max_samples:
        df = df.head(max_samples)

    stats = {
        'total_samples': len(df),
        'text_based': 0,
        'rendered_based': 0,
        'with_thinking': 0,
        'errors': 0
    }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # First pass: collect all data and decode main images
    samples_data = []
    for idx, row in df.iterrows():
        try:
            sample_id = row['id']

            # Save main image (V)
            main_image_path = decode_and_save_image(
                row['image'],
                images_dir,
                dataset_name,
                sample_id,
                "main"
            )

            # Parse conversations
            conversations = row['conversations']
            if len(conversations) < 2:
                stats['errors'] += 1
                continue

            human_msg = conversations[0]['value']
            assistant_msg = conversations[1]['value']

            # Extract thinking and answer
            thinking, answer = extract_thinking_and_answer(assistant_msg)

            if thinking:
                stats['with_thinking'] += 1

            samples_data.append({
                'sample_id': sample_id,
                'main_image_path': main_image_path,
                'human_msg': human_msg,
                'thinking': thinking,
                'answer': answer
            })
        except Exception as e:
            stats['errors'] += 1

    # Second pass: batch render all unique texts
    if renderer:
        print(f"  Batch rendering {len(samples_data)} samples...")

        # Collect unique thinking texts
        thinking_texts = []
        thinking_sample_ids = []
        for s in samples_data:
            if s['thinking']:
                thinking_texts.append(s['thinking'])
                thinking_sample_ids.append(s['sample_id'])

        # Batch render thinking images
        thinking_cache = {}
        if thinking_texts:
            thinking_cache = batch_render_texts(
                thinking_texts,
                renderer,
                images_dir,
                dataset_name,
                thinking_sample_ids,
                ["thinking"] * len(thinking_texts)
            )

        # Collect unique question texts (for rendered-based samples)
        question_texts = [s['human_msg'] for s in samples_data]
        question_sample_ids = [s['sample_id'] for s in samples_data]

        # Batch render question images
        question_cache = {}
        if question_texts:
            question_cache = batch_render_texts(
                question_texts,
                renderer,
                images_dir,
                dataset_name,
                question_sample_ids,
                ["question"] * len(question_texts)
            )

        print(f"  Rendering complete, generating samples...")

    # Third pass: generate samples with pre-rendered images
    with open(output_jsonl, 'w') as f_out:
        for s in samples_data:
            try:
                # Generate 1: Text-based sample (always create)
                images = [s['main_image_path']]
                instruction = random.choice(VQA_INSTRUCTION_TEMPLATES)
                user_content = f"{instruction}\n{s['human_msg']}\n<image>"

                # Add thinking image if exists
                if s['thinking'] and renderer:
                    thinking_img = thinking_cache.get(s['thinking'])
                    if thinking_img:
                        images.append(thinking_img)
                        think_start = ''.join(chr(c) for c in [0x15, 0x01, 0x1d, 0x00, 0x00, 0x00, 0x3e])
                        think_end = ''.join(chr(c) for c in [0x15, 0x00, 0x00, 0x00, 0x01, 0x1d, 0x00, 0x3e])
                        assistant_content = think_start + "\n<image>\n" + think_end + "\n" + s['answer']
                    else:
                        assistant_content = s['answer']
                else:
                    assistant_content = s['answer']

                text_sample = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ],
                    "images": images,
                    "task": "text_vqa",
                    "sample_type": "text_based"
                }
                f_out.write(json.dumps(text_sample) + '\n')
                stats['text_based'] += 1

                # Generate 2: Rendered-based sample
                if renderer:
                    question_img = question_cache.get(s['human_msg'])
                    if not question_img:
                        continue

                    # Randomize V and Q image order (50% chance to swap)
                    if random.random() < 0.5:
                        user_images = [question_img, s['main_image_path']]
                        image_order = "QV"
                    else:
                        user_images = [s['main_image_path'], question_img]
                        image_order = "VQ"

                    instruction = random.choice(VQA_INSTRUCTION_TEMPLATES)
                    # User: V + Q only (2 images)
                    user_content = f"{instruction}\n<image>\n<image>"

                    # Assistant: thinking image (if exists)
                    if s['thinking'] and renderer:
                        thinking_img = thinking_cache.get(s['thinking'])
                        if thinking_img:
                            # All images: V, Q, T
                            images = user_images + [thinking_img]
                            think_start = ''.join(chr(c) for c in [0x15, 0x01, 0x1d, 0x00, 0x00, 0x00, 0x3e])
                            think_end = ''.join(chr(c) for c in [0x15, 0x00, 0x00, 0x00, 0x01, 0x1d, 0x00, 0x3e])
                            assistant_content = think_start + "\n<image>\n" + think_end + "\n" + s['answer']
                        else:
                            # V, Q only (no thinking image)
                            images = user_images
                            assistant_content = s['answer']
                    else:
                        # V, Q only (no thinking)
                        images = user_images
                        assistant_content = s['answer']

                    rendered_sample = {
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ],
                        "images": images,
                        "task": "rendered_vqa",
                        "sample_type": "rendered_based",
                        "image_order": image_order
                    }
                    f_out.write(json.dumps(rendered_sample) + '\n')
                    stats['rendered_based'] += 1

                # Progress
                if (stats['text_based'] + stats['rendered_based']) % 100 == 0:
                    print(f"  Progress: {stats['text_based'] + stats['rendered_based']} samples", end='\r')

            except Exception as e:
                stats['errors'] += 1

    print(f"  Progress: {stats['text_based'] + stats['rendered_based']} samples")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='R1-Onevision dataset setup and conversion (all-in-one)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full setup for all datasets
    python OCRVL/scripts/build_r1_onevision.py --all

    # Setup specific datasets
    python OCRVL/scripts/build_r1_onevision.py --dataset ai2d doc_vqa

    # Quick test with limited samples
    python OCRVL/scripts/build_r1_onevision.py --all --max-samples 100
        """
    )

    parser.add_argument('--dataset', nargs='+', help='Dataset name(s) to convert')
    parser.add_argument('--all', action='store_true', help='Convert all datasets')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples per dataset (for testing)')
    parser.add_argument('--base-dir',
                        default='/share/project/xiyan/sources/DeepSeek-OCR',
                        help='Base directory of OCRVL repository')
    parser.add_argument('--data-dir',
                        default='/share/project/xiyan/huggingface/Fancy-MLLM/R1-Onevision',
                        help='Path to R1-Onevision data')
    parser.add_argument('--output-dir',
                        default='OCRVL/llamafactory/data',
                        help='Output directory for JSONL files')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess even if output exists')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    # Single images directory for all V, Q, thinking images
    images_dir = base_dir / "OCRVL/llamafactory/data/r1_onevision_images"

    # Verify setup
    if not verify_setup(base_dir):
        print("\n✗ Setup verification failed. Please fix the issues above.")
        return 1

    # Determine datasets
    if args.all:
        datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    elif args.dataset:
        datasets = args.dataset
    else:
        parser.error("Either --dataset or --all must be specified")

    print("=" * 70)
    print("R1-Onevision Dataset Conversion")
    print("=" * 70)
    print()
    print(f"Data directory:     {data_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Max samples:        {args.max_samples or 'All'}")
    print(f"Vello available:    {VELLO_AVAILABLE is not None}")
    print()

    # Initialize renderer
    renderer = None
    if VELLO_AVAILABLE:
        print("Initializing VelloRenderer...")
        try:
            renderer = VelloRenderer(
                width=640,
                height=640,
                padding=20,
                min_font_size=8.0,
                max_font_size=16.0,
                preserve_newlines=True
            )
            print(f"✓ Renderer initialized")
        except Exception as e:
            print(f"✗ Failed to initialize VelloRenderer: {e}")
            return 1
    else:
        print("✗ VelloRenderer not available. Install with:")
        print("  cd Renderer && maturin develop --release")
        return 1

    print()

    total_stats = {
        'total_samples': 0,
        'text_based': 0,
        'rendered_based': 0,
        'with_thinking': 0,
        'errors': 0
    }

    for dataset_name in sorted(datasets):
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            continue

        parquet_files = list(dataset_dir.glob('*.parquet'))
        if not parquet_files:
            continue

        parquet_path = parquet_files[0]
        output_jsonl = output_dir / f'r1_onevision_{dataset_name}.jsonl'

        # Check if output exists
        if output_jsonl.exists() and output_jsonl.stat().st_size > 0 and not args.force:
            # Check if incomplete by comparing against source parquet
            try:
                # Read parquet to get source sample count
                df_skip = pd.read_parquet(parquet_path)
                source_count = len(df_skip)
                total_stats['total_samples'] += source_count

                # Count existing samples in JSONL
                text_count = 0
                rendered_count = 0
                with open(output_jsonl, 'r') as f:
                    for line in f:
                        sample = json.loads(line)
                        if sample.get('sample_type') == 'text_based':
                            text_count += 1
                        elif sample.get('sample_type') == 'rendered_based':
                            rendered_count += 1

                # Calculate gaps
                missing_text = source_count - text_count
                missing_rendered = source_count - rendered_count

                # Auto-detect incomplete: any gaps means backfill needed
                if missing_text > 0 or missing_rendered > 0:
                    print(f"Backfilling {dataset_name}:")
                    print(f"  Source: {source_count} samples")
                    print(f"  Have:   {text_count} text-based, {rendered_count} rendered")
                    print(f"  Missing: {missing_text} text-based, {missing_rendered} rendered")
                    # Proceed to process this dataset
                else:
                    print(f"Skipping {dataset_name} (complete: {text_count} text-based, {rendered_count} rendered)")
                    total_stats['text_based'] += text_count
                    total_stats['rendered_based'] += rendered_count
                    continue
            except Exception as e:
                print(f"Error checking {dataset_name}, reprocessing: {e}")
                # Proceed to process on error

        # Sanitize dataset name for filename prefix
        dataset_prefix = dataset_name.replace('(', '_').replace(')', '_').replace('-', '_')

        print(f"Processing {dataset_name}...")
        stats = process_parquet_file(
            parquet_path,
            output_jsonl,
            images_dir,
            dataset_prefix,  # Use sanitized name as prefix
            render_thinking=True,
            renderer=renderer,
            max_samples=args.max_samples
        )

        print(f"  Text-based: {stats['text_based']}, Rendered: {stats['rendered_based']}, "
              f"With thinking: {stats['with_thinking']}, Errors: {stats['errors']}")
        print()

        for key in total_stats:
            total_stats[key] += stats[key]

    # Cleanup
    if renderer:
        try:
            renderer.shutdown()
        except:
            pass

    print("=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"Original samples:      {total_stats['total_samples']}")
    print(f"Text-based samples:    {total_stats['text_based']}")
    print(f"Rendered samples:      {total_stats['rendered_based']}")
    print(f"Total generated:       {total_stats['text_based'] + total_stats['rendered_based']}")
    print()
    print(f"JSONL files:           {output_dir}")
    print(f"All images (V+Q+T):    {images_dir}")
    print(f"  - Filenames:         SHA256 hash-based (deterministic)")
    print()
    print("✓ Ready to train!")
    print()
    print("Train with:")
    print(f"  bash OCRVL/scripts/train_llamafactory.sh \\")
    print(f"    {base_dir}/OCRVL/examples/llamafactory/qwen3vl_dpskocr_lora_r1onevision.yaml")

    return 0


if __name__ == '__main__':
    sys.exit(main())
