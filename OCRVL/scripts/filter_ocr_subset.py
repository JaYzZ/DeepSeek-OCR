#!/usr/bin/env/python3
"""
Create OCR-centric subset from existing unified SFT data.

This script filters the existing unified SFT dataset to include only OCR tasks,
avoiding a full dataset rebuild.

Can be used for both DPSK and Qwen3VL native training.
"""

import json
import sys
from pathlib import Path

# OCR tasks to include
OCR_TASKS = {
    'bbox_ocr',
    'full_document_ocr',
    'markdown_conversion',
    'caption_render_ocr',
}


def filter_ocr_subset(
    input_path: str,
    output_path: str,
):
    """
    Filter unified SFT data to include only OCR tasks.

    Args:
        input_path: Path to ocrvl_unified_sft.jsonl
        output_path: Path to write filtered subset
    """
    print(f"Reading from: {input_path}")

    total_samples = 0
    ocr_samples = 0

    with open(output_path, 'w') as f_out:
        with open(input_path) as f_in:
            for line_num, line in enumerate(f_in, 1):
                try:
                    data = json.loads(line)
                    task = data.get('task', 'unknown')

                    if task in OCR_TASKS:
                        # Keep this sample
                        f_out.write(line)
                        ocr_samples += 1

                    total_samples += 1

                    # Progress every 100K samples
                    if total_samples % 100000 == 0:
                        print(f"  Processed {total_samples:,} samples, kept {ocr_samples:,} OCR samples")

                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse line {line_num}: {e}")
                    continue

    print(f"\nComplete!")
    print(f"  Total samples processed: {total_samples:,}")
    print(f"  OCR samples written: {ocr_samples:,}")
    print(f"  Filtered out: {total_samples - ocr_samples:,}")
    print(f"\nOutput: {output_path}")


def main():
    # DPSK version - outputs to OCRVL/data
    base_dir = Path("/share/project/xiyan/sources/DeepSeek-OCR")

    input_path = base_dir / "OCRVL/data/ocrvl_unified_sft.jsonl"
    output_path = base_dir / "OCRVL/data/ocrvl_ocr_supplement.jsonl"

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output already exists
    if output_path.exists():
        # Count lines
        with open(output_path) as f:
            existing_lines = sum(1 for _ in f)

        with open(input_path) as f:
            total_lines = sum(1 for _ in f)

        if existing_lines > 0:
            print(f"Output file already exists: {output_path}")
            print(f"  Existing: {existing_lines:,} samples")
            print(f"  Source:   {total_lines:,} samples")

            if existing_lines >= total_lines * 0.95:  # Allow 5% tolerance
                print("\nDataset appears complete. Skipping generation.")
                print("To force regeneration, delete the output file first:")
                print(f"  rm {output_path}")
                return 0

    filter_ocr_subset(str(input_path), str(output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
