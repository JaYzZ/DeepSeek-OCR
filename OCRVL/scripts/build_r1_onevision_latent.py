#!/usr/bin/env python3
"""
R1-OneVision dataset builder with new thinking token format.

This script converts R1-Onevision parquet files to OCRVL JSONL format using:
    <|thinking_start|><|latent_step|><|thinking_sep|><|latent_step|><|thinking_sep|>...<|thinking_end|>

The thinking text is adaptively chunked and each chunk becomes a latent step.
Unlike the original builder, this DOES NOT render thinking to images - instead
it stores the raw thinking text for on-the-fly OCR encoding during training.

Usage:
    # Add thinking tokens to tokenizer first
    python -m OCRVL.utils.latent_tokens \
        --tokenizer /path/to/checkpoint \
        --output /path/to/checkpoint

    # Build dataset
    python OCRVL/scripts/build_r1_onevision_latent.py --all
"""

import argparse
import hashlib
import json
import base64
import os
import random
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

# Add parent directory to path for imports
script_dir = Path(__file__).parent.parent
repo_root = script_dir.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(repo_root))

# Instruction templates - aligned with unified SFT (build_vqa.py)
VQA_INSTRUCTION_TEMPLATES = [
    "Answer the question:",
    "Respond to the question:",
    "Provide an answer:",
    "Answer based on the image:",
]

# New thinking tokens
THINKING_START = "<|thinking_start|>"
LATENT_STEP = "<|latent_step|>"
THINKING_SEP = "<|thinking_sep|>"
THINKING_END = "<|thinking_end|>"


def get_hash_filename(dataset_name: str, sample_id: str, img_type: str) -> str:
    """Generate deterministic hash-based filename with dataset prefix."""
    hash_input = f"{sample_id}_{img_type}".encode('utf-8')
    hash_hex = hashlib.sha256(hash_input).hexdigest()[:12]
    return f"{dataset_name}_{hash_hex}.png"


def decode_and_save_image(base64_string: str, output_dir: Path, dataset_name: str, sample_id: str, img_type: str) -> str:
    """Decode base64 image and save to disk with hash-based filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]

    image_data = base64.b64decode(base64_string)
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


def create_latent_training_sample(
    dataset_name: str,
    sample_id: str,
    question_text: str,
    thinking: str,
    answer: str,
    main_image_path: str,
) -> Optional[dict]:
    """Create a latent training sample.

    This format stores raw thinking text for on-the-fly OCR encoding during training.
    The number of latent steps is determined adaptively by the OCR adapter.

    JSONL format:
    {
        "question": "What is 2+2?",
        "thinking": "To solve 2+2, I need to add...",
        "answer": "4",
        "image": "path/to/image.png"
    }
    """
    # No thinking field needed for simple VQA
    if not thinking:
        return {
            "question": question_text,
            "answer": answer,
            "image": main_image_path,
        }

    # With thinking: let dataset determine num_steps adaptively
    return {
        "question": question_text,
        "thinking": thinking,
        "answer": answer,
        "image": main_image_path,
    }


def process_parquet_file(
    parquet_path: Path,
    output_jsonl: Path,
    images_dir: Path,
    dataset_name: str,
    max_samples: Optional[int] = None
) -> dict:
    """Process a single parquet file for latent training."""
    df = pd.read_parquet(parquet_path)

    if max_samples:
        df = df.head(max_samples)

    stats = {
        'total_samples': len(df),
        'with_thinking': 0,
        'with_image': 0,
        'errors': 0
    }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, 'w') as f_out:
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

                question_text = conversations[0]['value']
                assistant_msg = conversations[1]['value']

                # Extract thinking and answer
                thinking, answer = extract_thinking_and_answer(assistant_msg)

                if thinking:
                    stats['with_thinking'] += 1

                stats['with_image'] += 1

                # Create latent training sample
                sample = create_latent_training_sample(
                    dataset_name=dataset_name,
                    sample_id=sample_id,
                    question_text=question_text,
                    thinking=thinking,
                    answer=answer,
                    main_image_path=main_image_path,
                )

                if sample:
                    f_out.write(json.dumps(sample) + '\n')

                # Progress
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{len(df)} samples", end='\r')

            except Exception as e:
                stats['errors'] += 1
                print(f"\n  Error processing sample {idx}: {e}")

    print(f"  Progress: {len(df)}/{len(df)} samples")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='R1-Onevision dataset builder with new thinking token format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build all datasets
    python OCRVL/scripts/build_r1_onevision_latent.py --all

    # Build specific datasets
    python OCRVL/scripts/build_r1_onevision_latent.py --dataset ai2d doc_vqa

    # Quick test with limited samples
    python OCRVL/scripts/build_r1_onevision_latent.py --all --max-samples 100

Note: Before running, add thinking tokens to your tokenizer:
    python -m OCRVL.utils.latent_tokens \\
        --tokenizer /path/to/OCR-Qwen3-VL-2B \\
        --output /path/to/OCR-Qwen3-VL-2B
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
    images_dir = base_dir / "OCRVL/llamafactory/data/r1_onevision_latent_images"

    # Determine datasets
    if args.all:
        datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    elif args.dataset:
        datasets = args.dataset
    else:
        parser.error("Either --dataset or --all must be specified")

    print("=" * 70)
    print("R1-OneVision Latent Training Dataset Builder")
    print("=" * 70)
    print()
    print(f"Data directory:     {data_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Images directory:   {images_dir}")
    print(f"Max samples:        {args.max_samples or 'All'}")
    print()

    print("Format:")
    print(f"  Question + {THINKING_START}<steps>{THINKING_END} + Answer")
    print(f"  Steps: {LATENT_STEP} (adaptive per sample)")
    print(f"  Separators: {THINKING_SEP} (between steps)")
    print()

    total_stats = {
        'total_samples': 0,
        'with_thinking': 0,
        'with_image': 0,
        'errors': 0
    }

    for dataset_name in sorted(datasets):
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            print(f"Skipping {dataset_name} (directory not found)")
            continue

        parquet_files = list(dataset_dir.glob('*.parquet'))
        if not parquet_files:
            print(f"Skipping {dataset_name} (no parquet files)")
            continue

        parquet_path = parquet_files[0]
        output_jsonl = output_dir / f'r1_onevision_{dataset_name}_latent.jsonl'

        # Check if output exists
        if output_jsonl.exists() and output_jsonl.stat().st_size > 0 and not args.force:
            # Check if complete
            with open(output_jsonl, 'r') as f:
                line_count = sum(1 for _ in f)

            source_count = len(pd.read_parquet(parquet_path))
            if line_count >= source_count:
                print(f"Skipping {dataset_name} (complete: {line_count} samples)")
                total_stats['total_samples'] += source_count
                total_stats['with_thinking'] += line_count  # Approximate
                continue

        # Sanitize dataset name for filename prefix
        dataset_prefix = dataset_name.replace('(', '_').replace(')', '_').replace('-', '_')

        print(f"Processing {dataset_name}...")
        stats = process_parquet_file(
            parquet_path,
            output_jsonl,
            images_dir,
            dataset_prefix,
            max_samples=args.max_samples
        )

        print(f"  Samples: {stats['total_samples']}, With thinking: {stats['with_thinking']}, "
              f"Errors: {stats['errors']}")
        print()

        for key in total_stats:
            total_stats[key] += stats[key]

    print("=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"Total samples:       {total_stats['total_samples']}")
    print(f"With thinking:       {total_stats['with_thinking']}")
    print(f"Errors:              {total_stats['errors']}")
    print()
    print(f"JSONL files:          {output_dir}")
    print(f"Images directory:    {images_dir}")
    print()
    print("✓ Ready to train!")
    print()
    print("Train with latent dataset:")
    print("  Use OCRVL/data/latent_dataset.py with the generated JSONL files")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
