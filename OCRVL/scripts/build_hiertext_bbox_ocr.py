#!/usr/bin/env python3
"""
HierText BBox OCR Dataset Builder - Centralized Script

Processes original HierText data and creates a unified bbox-ocr dataset.

Usage:
    python build_hiertext_bbox_ocr.py --output hiertext_unified.jsonl

Output format (LlamaFactory):
{
  "messages": [
    {"role": "user", "content": "Transcribe the text in [x1, y1, x2, y2]:\n<image>"},
    {"role": "assistant", "content": "text content"}
  ],
  "images": ["/absolute/path/to/image.jpg"],
  "task": "bbox_ocr",
  "split": "train|validation|test",
  "metadata": {"bbox": [x1, y1, x2, y2], "source": "hiertext"}
}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional
from project_paths import sources_path

# Add parent directory to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))


def compute_bounding_box(vertices):
    """Given vertices [[x1, y1], [x2, y2], ...], compute [xmin, ymin, xmax, ymax]."""
    xs = [point[0] for point in vertices]
    ys = [point[1] for point in vertices]
    return [min(xs), min(ys), max(xs), max(ys)]


def process_hiertext_split(
    json_file: Path,
    split_name: str,
    max_samples: Optional[int] = None
) -> Iterator[Dict]:
    """Process a HierText split (train/validation/test).

    Args:
        json_file: Path to HierText JSON file
        split_name: Name of the split (train/validation/test)
        max_samples: Maximum samples to generate

    Yields:
        Training samples with bbox-specific questions
    """
    # Load dataset
    with open(json_file, 'r') as f:
        data = json.load(f)

    base_dir = json_file.parent
    image_dir = base_dir / split_name

    total_emitted = 0

    # Process each image annotation
    for annotation in data.get("annotations", []):
        if max_samples and total_emitted >= max_samples:
            break

        image_id = annotation["image_id"]
        image_path = image_dir / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        # Get image dimensions
        if "image_size" in annotation:
            height, width = annotation["image_size"]
        elif "image_height" in annotation and "image_width" in annotation:
            height = annotation["image_height"]
            width = annotation["image_width"]
        else:
            continue

        original_hw = [height, width]
        original_area = width * height

        # Process each paragraph
        for paragraph in annotation.get("paragraphs", []):
            vertices = paragraph.get("vertices", [])
            if not vertices:
                continue

            bbox = compute_bounding_box(vertices)

            # Extract text from legible lines
            lines_text = []
            for line in paragraph.get("lines", []):
                if line.get("legible", False):
                    lines_text.append(line.get("text", ""))
            paragraph_text = "\n".join(lines_text).strip()

            if not paragraph_text:
                continue

            # Calculate bbox area for filtering
            width_bbox = bbox[2] - bbox[0]
            height_bbox = bbox[3] - bbox[1]
            bbox_area = width_bbox * height_bbox

            # Apply filters: must be 1/20 to 1/4 of image area, at least 400px, and >=8 chars
            if bbox_area < (original_area / 20) or bbox_area > (original_area / 4) or bbox_area < 400 or len(paragraph_text) < 8:
                continue

            # Convert absolute bbox to normalized [0, 1000] for Qwen3VL
            xmin, ymin, xmax, ymax = bbox
            x1_norm = round(xmin / width * 1000)
            y1_norm = round(ymin / height * 1000)
            x2_norm = round(xmax / width * 1000)
            y2_norm = round(ymax / height * 1000)

            bbox_normalized = [x1_norm, y1_norm, x2_norm, y2_norm]
            bbox_str = f"[{bbox_normalized[0]}, {bbox_normalized[1]}, {bbox_normalized[2]}, {bbox_normalized[3]}]"

            # Yield in LlamaFactory format
            yield {
                "messages": [
                    {"role": "user", "content": f"Transcribe the text in {bbox_str}:\n<image>"},
                    {"role": "assistant", "content": paragraph_text}
                ],
                "images": [str(image_path.resolve())],
                "task": "bbox_ocr",
                "split": split_name,
                "metadata": {
                    "bbox": bbox_normalized,
                    "source": "hiertext",
                }
            }

            total_emitted += 1


def main():
    parser = argparse.ArgumentParser(
        description="HierText BBox OCR Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build unified dataset from all splits
    python OCRVL/scripts/build_hiertext_bbox_ocr.py \\
        --hiertext-dir $ROOT_DIR/sources/hiertext \\
        --output OCRVL/llamafactory/data/hiertext_unified.jsonl

    # Build with limited samples for testing
    python OCRVL/scripts/build_hiertext_bbox_ocr.py \\
        --hiertext-dir $ROOT_DIR/sources/hiertext \\
        --output OCRVL/llamafactory/data/hiertext_unified.jsonl \\
        --max-samples 100
        """
    )

    parser.add_argument('--hiertext-dir', default=str(sources_path('hiertext')),
                        help='Path to HierText directory containing train.jsonl, validation.jsonl, test.jsonl')
    parser.add_argument('--output', required=True,
                        help='Output unified JSONL file path')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per split (for testing)')

    args = parser.parse_args()

    hiertext_dir = Path(args.hiertext_dir)
    output_path = Path(args.output)

    print("=" * 70)
    print("HierText BBox OCR Dataset Builder")
    print("=" * 70)
    print()
    print(f"HiText directory: {hiertext_dir}")
    print(f"Output: {output_path}")
    print(f"Max samples per split: {args.max_samples or 'All'}")
    print()

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process all splits and write to unified file
    total = 0
    split_counts = {}

    with open(output_path, 'w') as f:
        for split_name in ['train', 'validation', 'test']:
            json_file = hiertext_dir / f"{split_name}.jsonl"

            if not json_file.exists():
                print(f"Warning: {json_file} not found, skipping...")
                continue

            print(f"Processing {split_name}...")

            split_count = 0
            for sample in process_hiertext_split(
                json_file=json_file,
                split_name=split_name,
                max_samples=args.max_samples
            ):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                split_count += 1
                total += 1

            split_counts[split_name] = split_count
            print(f"  Added {split_count} {split_name} samples")

    print()
    print("=" * 70)
    print(f"✓ Built HierText unified dataset: {output_path}")
    print(f"  Total samples: {total}")
    for split, count in split_counts.items():
        print(f"    - {split}: {count}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
