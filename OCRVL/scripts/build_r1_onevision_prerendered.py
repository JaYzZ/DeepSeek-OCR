#!/usr/bin/env python3
"""
R1-OneVision Prerendered Dataset Builder

This script creates JSONL files from pre-rendered R1-OneVision images.
It references existing rendered images instead of creating new ones.

Prerendered data structure expected:
    <base_dir>/
        <dataset_name>/
            <sample_id>/
                question_0.png, question_1.png, ...
                thinking_0.png, thinking_1.png, ...
                main.png

Output JSONL format:
    {
        "question_images": ["question_0.png", "question_1.png"],
        "thinking_images": ["thinking_0.png", "thinking_1.png", ...],
        "answer": "4",
        "image": "main.png"
    }

Usage:
    python OCRVL/scripts/build_r1_onevision_prerendered.py \
        --base-dir /path/to/prerendered \
        --output-dir OCRVL/llamafactory/data \
        --dataset ai2d doc_vqa
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add parent directory to path for imports
script_dir = Path(__file__).parent.parent
repo_root = script_dir.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(repo_root))


def scan_prerendered_sample(
    sample_dir: Path,
    dataset_name: str
) -> Optional[dict]:
    """Scan a single sample directory for pre-rendered images.

    Args:
        sample_dir: Path to sample directory
        dataset_name: Dataset name for logging

    Returns:
        Dict with question_images, thinking_images, answer, main_image paths
        Or None if sample is invalid
    """
    if not sample_dir.is_dir():
        return None

    # Find question images
    question_images = sorted(sample_dir.glob("question_*.png"))

    # Find thinking images
    thinking_images = sorted(sample_dir.glob("thinking_*.png"))

    # Find main image (V)
    main_images = list(sample_dir.glob("main.png")) + list(sample_dir.glob("image.png"))

    if not main_images:
        return None

    # Try to find answer text file
    answer_file = sample_dir / "answer.txt"
    if answer_file.exists():
        with open(answer_file, 'r') as f:
            answer = f.read().strip()
    else:
        # Fallback: try to find in metadata
        answer = ""

    return {
        "question_images": [str(img.relative_to(sample_dir.parent)) for img in question_images],
        "thinking_images": [str(img.relative_to(sample_dir.parent)) for img in thinking_images],
        "answer": answer,
        "image": str(main_images[0].relative_to(sample_dir.parent)),
    }


def build_prerendered_dataset(
    base_dir: Path,
    dataset_name: str,
    output_jsonl: Path,
) -> dict:
    """Build prerendered dataset from existing rendered images.

    Args:
        base_dir: Base directory containing rendered datasets
        dataset_name: Name of dataset to process
        output_jsonl: Path to output JSONL file

    Returns:
        Statistics dict
    """
    dataset_dir = base_dir / dataset_name

    if not dataset_dir.exists():
        print(f"Skipping {dataset_name} (directory not found)")
        return {'total_samples': 0, 'valid_samples': 0, 'errors': 0}

    # Find all sample directories
    sample_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not sample_dirs:
        print(f"Skipping {dataset_name} (no sample directories found)")
        return {'total_samples': 0, 'valid_samples': 0, 'errors': 0}

    stats = {
        'total_samples': len(sample_dirs),
        'valid_samples': 0,
        'with_question': 0,
        'with_thinking': 0,
        'errors': 0
    }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, 'w') as f_out:
        for sample_dir in sorted(sample_dirs):
            try:
                sample = scan_prerendered_sample(sample_dir, dataset_name)

                if sample is None:
                    stats['errors'] += 1
                    continue

                stats['valid_samples'] += 1

                if sample['question_images']:
                    stats['with_question'] += 1

                if sample['thinking_images']:
                    stats['with_thinking'] += 1

                f_out.write(json.dumps(sample) + '\n')

                # Progress
                if stats['valid_samples'] % 100 == 0:
                    print(f"  Progress: {stats['valid_samples']} samples", end='\r')

            except Exception as e:
                stats['errors'] += 1
                print(f"\n  Error processing {sample_dir}: {e}")

    print(f"  Progress: {stats['valid_samples']} samples")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='R1-OneVision Prerendered Dataset Builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build all datasets
    python OCRVL/scripts/build_r1_onevision_prerendered.py --all

    # Build specific datasets
    python OCRVL/scripts/build_r1_onevision_prerendered.py --dataset ai2d doc_vqa

Note:
    This script creates JSONL files that reference pre-rendered images.
    The images should already exist in the expected directory structure:

    <base_dir>/
        <dataset_name>/
            <sample_id>/
                question_0.png, question_1.png, ...
                thinking_0.png, thinking_1.png, ...
                main.png
                answer.txt
        """
    )

    parser.add_argument('--dataset', nargs='+', help='Dataset name(s) to convert')
    parser.add_argument('--all', action='store_true', help='Convert all datasets')
    parser.add_argument('--base-dir',
                        default='/share/project/xiyan/huggingface/Fancy-MLLM/R1-Onevision-prerendered',
                        help='Base directory of pre-rendered images')
    parser.add_argument('--output-dir',
                        default='OCRVL/llamafactory/data',
                        help='Output directory for JSONL files')
    parser.add_argument('--force', action='store_true',
                        help='Force rebuild even if output exists')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    # Determine datasets
    if args.all:
        datasets = [d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    elif args.dataset:
        datasets = args.dataset
    else:
        parser.error("Either --dataset or --all must be specified")

    print("=" * 70)
    print("R1-OneVision Prerendered Dataset Builder")
    print("=" * 70)
    print()
    print(f"Base directory:    {base_dir}")
    print(f"Output directory:  {output_dir}")
    print()

    total_stats = {
        'total_samples': 0,
        'valid_samples': 0,
        'with_question': 0,
        'with_thinking': 0,
        'errors': 0
    }

    for dataset_name in sorted(datasets):
        dataset_dir = base_dir / dataset_name
        if not dataset_dir.exists():
            print(f"Skipping {dataset_name} (directory not found)")
            continue

        output_jsonl = output_dir / f'r1_onevision_{dataset_name}_prerendered.jsonl'

        # Check if output exists
        if output_jsonl.exists() and output_jsonl.stat().st_size > 0 and not args.force:
            line_count = sum(1 for _ in open(output_jsonl))
            print(f"Skipping {dataset_name} (exists: {line_count} samples)")
            total_stats['total_samples'] += line_count
            total_stats['valid_samples'] += line_count
            continue

        print(f"Processing {dataset_name}...")
        stats = build_prerendered_dataset(
            base_dir=base_dir,
            dataset_name=dataset_name,
            output_jsonl=output_jsonl,
        )

        print(f"  Valid: {stats['valid_samples']}, "
              f"Question: {stats['with_question']}, "
              f"Thinking: {stats['with_thinking']}, "
              f"Errors: {stats['errors']}")
        print()

        for key in total_stats:
            total_stats[key] += stats[key]

    print("=" * 70)
    print("Build Complete!")
    print("=" * 70)
    print(f"Total valid samples: {total_stats['valid_samples']}")
    print(f"Errors:              {total_stats['errors']}")
    print()
    print(f"JSONL files:          {output_dir}")
    print()
    print("Usage with OCRThinkingDataset:")
    print("  dataset = OCRThinkingDataset(")
    print("      jsonl_path='OCRVL/llamafactory/data/r1_onevision_<dataset>_prerendered.jsonl',")
    print("      tokenizer=tokenizer,")
    print("      ocr_adapter=ocr_adapter,")
    print("      use_prerendered=True,")
    print("      prerendered_base_dir='/path/to/prerendered',")
    print("  )")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
