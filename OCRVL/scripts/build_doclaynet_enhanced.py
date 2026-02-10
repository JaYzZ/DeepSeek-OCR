#!/usr/bin/env python3
"""
Enhanced DocLayNet Dataset Builder

Creates multiple types of training samples from DocLayNet bbox annotations:

1. Per-bbox OCR: Extract text from specific bbox regions
2. Full-document OCR: Extract all text from entire document
3. Markdown conversion: Convert document to markdown structure

All bbox coordinates are normalized to [0, 1] based on image dimensions.
"""

import argparse
import base64
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import threading

import requests

# Add parent directory to path
script_dir = Path(__file__).parent.parent
repo_root = script_dir.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(repo_root))


# =============================================================================
# Qwen2.5-VL API Client
# =============================================================================

class QwenVLClient:
    """Thread-safe client for Qwen2.5-VL vLLM server."""

    # Class-level lock for thread safety
    _lock = threading.Lock()

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{base_url}/v1/chat/completions"

        # Get actual model ID from server
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        self.model_id = data['data'][0]['id']
        print(f"  Using model: {self.model_id.split('/')[-1]}")

    def generate_markdown(
        self,
        image_path: str,
        full_text: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        """Generate markdown from image with full text hint.

        Args:
            image_path: Path to document image
            full_text: Full document text (OCR output in reading order)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated markdown string
        """
        # Build prompt with full text as reference
        prompt = f"""You are a document layout analysis expert. Convert this document image to properly formatted markdown.

The following OCR text contains all the content but has LOST its original layout structure (no formatting, no table boundaries, no list hierarchy, broken captions/labels). Your task is to RECONSTRUCT the proper layout by examining the image.

Reference text (for transcription accuracy only, layout is broken):
---
{full_text}
---

IMPORTANT - Focus on LAYOUT RECONSTRUCTION from the image:
1. **Tables**: Detect column/row boundaries from image, format as markdown tables with | separators
2. **Figures/Images**: Identify by captions, labels, or visual boundaries - use ![caption](image) syntax
3. **Lists**: Detect indentation levels and bullet/number styles from image - use - for bullets, 1. for numbered
4. **Headers**: Detect font size/weight hierarchy from image - use # ## ### accordingly
5. **Sections**: Identify text blocks, columns, and spatial grouping
6. **Equations**: Look for math notation - use $inline$ or $$block$$ LaTeX

Guidelines:
- Get text CONTENT from reference (ensure accuracy)
- Get STRUCTURE from the image (restore layout)
- Preserve spatial relationships visible in image
- Group related items based on visual proximity
- Add proper spacing between sections

Convert this image to markdown:"""

        # Encode image to base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Build request payload
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        # Send request
        response = requests.post(self.api_url, json=payload, timeout=300)
        response.raise_for_status()

        # Extract markdown
        result = response.json()
        markdown = result["choices"][0]["message"]["content"]

        # Clean up any markdown code blocks
        markdown = markdown.strip()
        if markdown.startswith("```markdown"):
            markdown = markdown[11:].strip()
        elif markdown.startswith("```"):
            markdown = markdown[3:].strip()
        if markdown.endswith("```"):
            markdown = markdown[:-3].strip()

        return markdown


# =============================================================================
# OCR Instruction Prompts
# =============================================================================

FULL_IMAGE_OCR_INSTRUCTIONS = [
    "Free OCR",
    "Read all text in the image",
    "Transcribe the text",
    "Read and transcribe the document content",
    "Extract all text from the image",
]


def sort_segments_by_reading_order(
    segments: List[Dict],
    image_width: int = 640
) -> List[Dict]:
    """Sort segments by reading order (top-to-bottom, left-to-right).

    Args:
        segments: List of segment dicts with 'bbox' and 'text' fields
        image_width: Image width for centering detection

    Returns:
        Sorted list of segments in reading order
    """
    if not segments:
        return []

    # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
    sorted_segments = sorted(segments, key=lambda s: (s['bbox'][1], s['bbox'][0]))

    # Group into lines (similar y-coordinate, ±5px tolerance)
    y_tolerance = 5
    lines = []
    current_line = []

    for seg in sorted_segments:
        if not current_line:
            current_line = [seg]
        elif abs(seg['bbox'][1] - current_line[0]['bbox'][1]) <= y_tolerance:
            current_line.append(seg)
        else:
            lines.append(current_line)
            current_line = [seg]

    if current_line:
        lines.append(current_line)

    # Sort each line left-to-right
    for line in lines:
        line.sort(key=lambda s: s['bbox'][0])

    # Flatten lines
    return [seg for line in lines for seg in line]


def iter_bbox_ocr_samples(
    json_path: Path,
    image_dir: Path,
    image_base_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Generate per-bbox OCR training samples.

    Args:
        json_path: Path to DocLayNet JSON file or directory
        image_dir: Directory containing DocLayNet images
        image_base_dir: Base directory for image paths (for relative paths)
        max_samples: Maximum number of samples to generate

    Yields:
        Training samples with bbox-specific questions
    """
    # Check if json_path is a directory (DocLayNet JSON folder)
    if json_path.is_dir():
        json_files = list(json_path.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files in {json_path}")
    else:
        json_files = [json_path]

    total_emitted = 0

    for json_file in json_files:
        if max_samples and total_emitted >= max_samples:
            break

        with json_file.open('r', encoding='utf-8') as f:
            doc = json.load(f)

        # Handle DocLayNet format
        # Get image filename from json filename
        image_filename = json_file.stem + ".png"
        img_path = str(image_dir / image_filename)

        if not Path(img_path).exists():
            continue

        # Get image dimensions for normalization
        metadata = doc.get("metadata", {})
        img_width = metadata.get("coco_width", metadata.get("original_width", 1025))
        img_height = metadata.get("coco_height", metadata.get("original_height", 1025))

        # Get segments/cells from DocLayNet format
        segments = doc.get("cells", [])

        if not segments:
            continue

        # Convert DocLayNet bbox format [x, y, w, h] to normalized [x1, y1, x2, y2]
        # and create standard segment format
        standard_segments = []
        for cell in segments:
            bbox_xywh = cell.get('bbox', [])
            if len(bbox_xywh) >= 4:
                x, y, w, h = bbox_xywh[:4]
                # Convert to normalized [x1, y1, x2, y2] in [0, 1000] for Qwen3VL
                x1_norm = round(x / img_width * 1000)
                y1_norm = round(y / img_height * 1000)
                x2_norm = round((x + w) / img_width * 1000)
                y2_norm = round((y + h) / img_height * 1000)
                standard_segments.append({
                    "bbox": [x1_norm, y1_norm, x2_norm, y2_norm],
                    "text": cell.get('text', '')
                })

        if not standard_segments:
            continue

        # Sort segments by reading order
        sorted_segments = sort_segments_by_reading_order(standard_segments)

        # Sample 10 bboxes: 5 largest + 5 random
        # Filter to only segments with text
        valid_segments = [s for s in sorted_segments if s['text'].strip()]

        if not valid_segments:
            continue

        # Calculate bbox areas
        segments_with_area = []
        for seg in valid_segments:
            x1, y1, x2, y2 = seg['bbox']
            area = (x2 - x1) * (y2 - y1)
            segments_with_area.append((area, seg))

        # Sort by area (descending) and pick 5 largest
        segments_with_area.sort(key=lambda x: x[0], reverse=True)
        largest_segments = [s for _, s in segments_with_area[:5]]

        # Get remaining segments (excluding the 5 largest)
        remaining_segments = [s for _, s in segments_with_area[5:]]

        # Pick 5 random from remaining
        if len(remaining_segments) >= 5:
            random_segments = random.sample(remaining_segments, 5)
        else:
            # If fewer than 5 remaining, take all of them
            random_segments = remaining_segments

        # Combine: 5 largest + 5 random (max 10 total)
        selected_segments = largest_segments + random_segments

        # Generate per-bbox samples
        for seg in selected_segments:
            if max_samples and total_emitted >= max_samples:
                break

            bbox = seg['bbox']
            text = seg['text'].strip()

            if not text:
                continue

            # Format bbox as [x1, y1, x2, y2]
            bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"

            # Build image path (make relative if base_dir provided)
            final_img_path = img_path
            if image_base_dir:
                try:
                    rel_path = Path(img_path).relative_to(image_base_dir)
                    final_img_path = str(rel_path)
                except ValueError:
                    # Keep absolute if not relative
                    pass

            yield {
                "messages": [
                    {"role": "user", "content": f"Transcribe the text in {bbox_str}:\n<image>"},
                    {"role": "assistant", "content": text}
                ],
                "images": [final_img_path],
                "task": "bbox_ocr",
                "metadata": {
                    "bbox": bbox,
                    "doc_id": doc.get("id", ""),
                }
            }

            total_emitted += 1


def iter_full_ocr_samples(
    json_path: Path,
    image_dir: Path,
    image_base_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Generate full-document OCR training samples.

    Extracts all text from the entire document in reading order.

    Args:
        json_path: Path to DocLayNet JSON file or directory
        image_dir: Directory containing DocLayNet images
        image_base_dir: Base directory for image paths (for relative paths)
        max_samples: Maximum number of samples to generate

    Yields:
        Training samples with full-document OCR questions
    """
    # Check if json_path is a directory (DocLayNet JSON folder)
    if json_path.is_dir():
        json_files = list(json_path.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files in {json_path}")
    else:
        json_files = [json_path]

    total_emitted = 0

    for json_file in json_files:
        if max_samples and total_emitted >= max_samples:
            break

        with json_file.open('r', encoding='utf-8') as f:
            doc = json.load(f)

        # Handle DocLayNet format
        # Get image filename from json filename
        image_filename = json_file.stem + ".png"
        img_path = str(image_dir / image_filename)

        if not Path(img_path).exists():
            continue

        # Get image dimensions for normalization
        metadata = doc.get("metadata", {})
        img_width = metadata.get("coco_width", metadata.get("original_width", 1025))
        img_height = metadata.get("coco_height", metadata.get("original_height", 1025))

        # Get segments/cells from DocLayNet format
        segments = doc.get("cells", [])

        if not segments:
            continue

        # Convert DocLayNet bbox format [x, y, w, h] to normalized [x1, y1, x2, y2]
        # and create standard segment format
        standard_segments = []
        for cell in segments:
            bbox_xywh = cell.get('bbox', [])
            if len(bbox_xywh) >= 4:
                x, y, w, h = bbox_xywh[:4]
                # Convert to normalized [x1, y1, x2, y2] in [0, 1000] for Qwen3VL
                x1_norm = round(x / img_width * 1000)
                y1_norm = round(y / img_height * 1000)
                x2_norm = round((x + w) / img_width * 1000)
                y2_norm = round((y + h) / img_height * 1000)
                text = cell.get('text', '').strip()
                if text:
                    standard_segments.append({
                        "bbox": [x1_norm, y1_norm, x2_norm, y2_norm],
                        "text": text
                    })

        if not standard_segments:
            continue

        # Sort segments by reading order
        sorted_segments = sort_segments_by_reading_order(standard_segments)

        # Collect text in reading order with line breaks
        text_lines = []
        current_line = []
        y_tolerance_px = 5  # 5 pixel tolerance for same line

        for seg in sorted_segments:
            text = seg['text']
            bbox = seg['bbox']
            y1 = bbox[1] * img_height  # Convert normalized y to pixels

            if not current_line:
                current_line = [(y1, text)]
            elif abs(y1 - current_line[0][0]) <= y_tolerance_px:
                # Same line - add to current line
                current_line.append((y1, text))
            else:
                # New line - save current line and start new one
                text_lines.append(' '.join([t for _, t in current_line]))
                current_line = [(y1, text)]

        # Don't forget the last line
        if current_line:
            text_lines.append(' '.join([t for _, t in current_line]))

        full_text = '\n'.join(text_lines).strip()

        if not full_text:
            continue

        # Build image path (make relative if base_dir provided)
        final_img_path = img_path
        if image_base_dir:
            try:
                rel_path = Path(img_path).relative_to(image_base_dir)
                final_img_path = str(rel_path)
            except ValueError:
                # Keep absolute if not relative
                pass

        # Randomly select OCR instruction
        instruction = random.choice(FULL_IMAGE_OCR_INSTRUCTIONS)

        # Return LlamaFactory format
        yield {
            "messages": [
                {"role": "user", "content": f"{instruction}:\n<image>"},
                {"role": "assistant", "content": full_text}
            ],
            "images": [final_img_path],
            "task": "full_document_ocr",
            "metadata": {
                "source": "doclaynet",
                "num_segments": len(sorted_segments),
            }
        }

        total_emitted += 1


def iter_markdown_model_samples(
    json_path: Path,
    image_dir: Path,
    client: QwenVLClient,
    image_base_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Generate markdown conversion samples using Qwen2.5-VL model.

    This uses the running Qwen2.5-VL-72B server to generate proper markdown
    annotations with full OCR text as a hint for transcription accuracy.

    Args:
        json_path: Path to DocLayNet JSON file or directory
        image_dir: Directory containing DocLayNet images
        client: QwenVL client instance
        image_base_dir: Base directory for image paths
        max_samples: Maximum number of samples to generate

    Yields:
        Training samples with model-generated markdown
    """
    # Check if json_path is a directory (DocLayNet JSON folder)
    if json_path.is_dir():
        json_files = list(json_path.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files in {json_path}")
    else:
        json_files = [json_path]

    total_emitted = 0

    for json_file in json_files:
        if max_samples and total_emitted >= max_samples:
            break

        try:
            with json_file.open('r', encoding='utf-8') as f:
                doc = json.load(f)

            # Handle DocLayNet format
            # Get image filename from json filename
            image_filename = json_file.stem + ".png"
            img_path = str(image_dir / image_filename)

            if not Path(img_path).exists():
                continue

            # Get image dimensions for normalization
            metadata = doc.get("metadata", {})
            img_width = metadata.get("coco_width", metadata.get("original_width", 1025))
            img_height = metadata.get("coco_height", metadata.get("original_height", 1025))

            # Get segments/cells from DocLayNet format
            segments = doc.get("cells", [])

            if not segments:
                continue

            # Build full text in reading order from all segments
            standard_segments = []
            for cell in segments:
                bbox_xywh = cell.get('bbox', [])
                if len(bbox_xywh) >= 4:
                    x, y, w, h = bbox_xywh[:4]
                    # Convert to [0, 1000] for Qwen3VL
                    x1_norm = round(x / img_width * 1000)
                    y1_norm = round(y / img_height * 1000)
                    x2_norm = round((x + w) / img_width * 1000)
                    y2_norm = round((y + h) / img_height * 1000)
                    text = cell.get('text', '').strip()
                    if text:
                        standard_segments.append({
                            "bbox": [x1_norm, y1_norm, x2_norm, y2_norm],
                            "text": text
                        })

            if not standard_segments:
                continue

            # Sort by reading order and build full text
            sorted_segments = sort_segments_by_reading_order(standard_segments)
            text_lines = []
            current_line = []
            y_tolerance_px = 5

            for seg in sorted_segments:
                text = seg['text']
                bbox = seg['bbox']
                y1 = bbox[1] * img_height

                if not current_line:
                    current_line = [(y1, text)]
                elif abs(y1 - current_line[0][0]) <= y_tolerance_px:
                    current_line.append((y1, text))
                else:
                    text_lines.append(' '.join([t for _, t in current_line]))
                    current_line = [(y1, text)]

            if current_line:
                text_lines.append(' '.join([t for _, t in current_line]))

            full_text = '\n'.join(text_lines).strip()

            # Generate markdown using Qwen2.5-VL with full text as hint
            print(f"  [{total_emitted + 1}] Generating markdown for {image_filename}...", end='\r')
            markdown_answer = client.generate_markdown(img_path, full_text)

            # Build image path (make relative if base_dir provided)
            final_img_path = img_path
            if image_base_dir:
                try:
                    rel_path = Path(img_path).relative_to(image_base_dir)
                    final_img_path = str(rel_path)
                except ValueError:
                    pass

            # Filter markdown samples: only include if >= 100 chars
            if len(markdown_answer) < 100:
                continue

            # Return LlamaFactory format
            yield {
                "messages": [
                    {"role": "user", "content": "Convert this image to markdown:\n<image>"},
                    {"role": "assistant", "content": markdown_answer}
                ],
                "images": [final_img_path],
                "task": "markdown_conversion",
                "metadata": {
                    "source": "doclaynet_qwen2.5vl",
                    "num_segments": len(sorted_segments),
                }
            }

            total_emitted += 1

        except Exception as e:
            print(f"\n  Error processing {json_file.name}: {e}")
            continue


def process_markdown_task(args: tuple) -> Optional[Dict[str, Any]]:
    """Worker function for parallel markdown generation.

    Args:
        args: Tuple of (json_file, image_dir, image_base_dir, server_url)

    Returns:
        Training sample with markdown, or None if failed
    """
    json_file, image_dir, image_base_dir, server_url = args

    try:
        # Create client per worker (thread-safe)
        client = QwenVLClient(server_url)

        with json_file.open('r', encoding='utf-8') as f:
            doc = json.load(f)

        # Get image filename and path
        image_filename = json_file.stem + ".png"
        img_path = str(image_dir / image_filename)

        if not Path(img_path).exists():
            return None

        # Get image dimensions
        metadata = doc.get("metadata", {})
        img_width = metadata.get("coco_width", metadata.get("original_width", 1025))
        img_height = metadata.get("coco_height", metadata.get("original_height", 1025))

        # Get segments/cells
        segments = doc.get("cells", [])
        if not segments:
            return None

        # Build full text in reading order from all segments
        standard_segments = []
        for cell in segments:
            bbox_xywh = cell.get('bbox', [])
            if len(bbox_xywh) >= 4:
                x, y, w, h = bbox_xywh[:4]
                x1_norm = round(x / img_width * 1000)
                y1_norm = round(y / img_height * 1000)
                x2_norm = round((x + w) / img_width * 1000)
                y2_norm = round((y + h) / img_height * 1000)
                text = cell.get('text', '').strip()
                if text:
                    standard_segments.append({
                        "bbox": [x1_norm, y1_norm, x2_norm, y2_norm],
                        "text": text
                    })

        if not standard_segments:
            return None

        # Sort by reading order and build full text
        sorted_segments = sort_segments_by_reading_order(standard_segments)
        text_lines = []
        current_line = []
        y_tolerance_px = 5

        for seg in sorted_segments:
            text = seg['text']
            bbox = seg['bbox']
            y1 = bbox[1] * img_height

            if not current_line:
                current_line = [(y1, text)]
            elif abs(y1 - current_line[0][0]) <= y_tolerance_px:
                current_line.append((y1, text))
            else:
                text_lines.append(' '.join([t for _, t in current_line]))
                current_line = [(y1, text)]

        if current_line:
            text_lines.append(' '.join([t for _, t in current_line]))

        full_text = '\n'.join(text_lines).strip()

        # Generate markdown
        markdown_answer = client.generate_markdown(img_path, full_text)

        # Build image path (make relative if base_dir provided)
        final_img_path = img_path
        if image_base_dir:
            try:
                rel_path = Path(img_path).relative_to(image_base_dir)
                final_img_path = str(rel_path)
            except ValueError:
                pass

        # Filter markdown samples: only include if >= 100 chars
        if len(markdown_answer) < 100:
            return None

        # Return LlamaFactory format
        return {
            "messages": [
                {"role": "user", "content": "Convert this image to markdown:\n<image>"},
                {"role": "assistant", "content": markdown_answer}
            ],
            "images": [final_img_path],
            "task": "markdown_conversion",
            "metadata": {
                "source": "doclaynet_qwen2.5vl",
                "num_segments": len(sorted_segments),
            }
        }

    except Exception as e:
        print(f"\n  Error in worker processing {json_file.name}: {e}")
        return None


def iter_markdown_model_samples_parallel(
    json_path: Path,
    image_dir: Path,
    server_url: str,
    image_base_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
) -> Iterator[Dict[str, Any]]:
    """Generate markdown conversion samples using parallel Qwen2.5-VL processing.

    Uses multiple threads to process documents in parallel for faster throughput.

    Args:
        json_path: Path to DocLayNet JSON file or directory
        image_dir: Directory containing DocLayNet images
        server_url: QwenVL server URL
        image_base_dir: Base directory for image paths
        max_samples: Maximum number of samples to generate
        num_workers: Number of parallel workers

    Yields:
        Training samples with model-generated markdown
    """
    # Check if json_path is a directory
    if json_path.is_dir():
        json_files = list(json_path.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files in {json_path}")
    else:
        json_files = [json_path]

    if max_samples:
        json_files = json_files[:max_samples]

    total_emitted = 0
    total_files = len(json_files)

    # Prepare tasks
    tasks = [(jf, image_dir, image_base_dir, server_url) for jf in json_files]

    # Process in parallel
    print(f"  Using {num_workers} parallel workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_markdown_task, task): task[0] for task in tasks}

        # Process completed tasks
        for future in as_completed(futures):
            json_file = futures[future]
            try:
                result = future.result()
                if result:
                    total_emitted += 1
                    yield result

                    if total_emitted % 10 == 0:
                        progress = (total_emitted / total_files) * 100
                        print(f"  Progress: {total_emitted}/{total_files} ({progress:.1f}%)")

            except Exception as e:
                print(f"\n  Error processing {json_file.name}: {e}")
                continue

    print(f"  ✓ Completed {total_emitted}/{total_files} documents")


def presample_bboxes(
    segments: List[Dict],
    samples_per_set: int = 3,
    num_sets: int = 5,
    seed: int = 42
) -> List[Dict]:
    """Presample bboxes by size, stratified across 5 size buckets.

    Args:
        segments: List of segment dicts with 'bbox' [x1, y1, x2, y2] and 'text'
        samples_per_set: Number of bboxes to sample from each size set
        num_sets: Number of size sets (splits by percentile)
        seed: Random seed

    Returns:
        Presampled list of segments
    """
    import random

    if len(segments) <= 15:
        # If <=15 bboxes, keep all
        return segments

    rng = random.Random(seed)

    # Calculate bbox areas and sort
    segments_with_area = []
    for seg in segments:
        bbox = seg['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        segments_with_area.append((area, seg))

    # Sort by area
    segments_with_area.sort(key=lambda x: x[0])

    # Split into 5 sets by percentile
    n = len(segments_with_area)
    set_size = n // num_sets

    sets = []
    for i in range(num_sets):
        start_idx = i * set_size
        # Last set gets remainder
        end_idx = n if i == num_sets - 1 else (i + 1) * set_size
        sets.append(segments_with_area[start_idx:end_idx])

    # Sample from each set
    presampled = []
    for seg_set in sets:
        if len(seg_set) <= samples_per_set:
            # Take all if set is small enough
            presampled.extend([s[1] for s in seg_set])
        else:
            # Random sample from this set
            rng.shuffle(seg_set)
            presampled.extend([s[1] for s in seg_set[:samples_per_set]])

    return presampled


def iter_all_samples(
    json_path: Path,
    image_dir: Path,
    client: Optional[QwenVLClient] = None,
    image_base_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    use_markdown_model: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Generate all sample types (bbox-ocr, full-ocr, markdown) from DocLayNet.

    Processes each document once and yields all sample types efficiently.

    Args:
        json_path: Path to DocLayNet JSON file or directory
        image_dir: Directory containing DocLayNet images
        client: QwenVL client (optional, for markdown generation)
        image_base_dir: Base directory for image paths
        max_samples: Maximum number of documents to process
        use_markdown_model: If True and client provided, use model for markdown

    Yields:
        Training samples (bbox-ocr, full-ocr, markdown)
    """
    # Check if json_path is a directory
    if json_path.is_dir():
        json_files = list(json_path.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files in {json_path}")
    else:
        json_files = [json_path]

    docs_processed = 0
    total_samples = 0

    for json_file in json_files:
        if max_samples and docs_processed >= max_samples:
            break

        try:
            with json_file.open('r', encoding='utf-8') as f:
                doc = json.load(f)

            # Get image filename and path
            image_filename = json_file.stem + ".png"
            img_path = str(image_dir / image_filename)

            if not Path(img_path).exists():
                continue

            # Get image dimensions
            metadata = doc.get("metadata", {})
            img_width = metadata.get("coco_width", metadata.get("original_width", 1025))
            img_height = metadata.get("coco_height", metadata.get("original_height", 1025))

            # Get segments/cells
            segments = doc.get("cells", [])
            if not segments:
                continue

            # Convert to standard format with normalized bboxes
            standard_segments = []
            for cell in segments:
                bbox_xywh = cell.get('bbox', [])
                if len(bbox_xywh) >= 4:
                    x, y, w, h = bbox_xywh[:4]
                    x1_norm = round(x / img_width * 1000)
                    y1_norm = round(y / img_height * 1000)
                    x2_norm = round((x + w) / img_width * 1000)
                    y2_norm = round((y + h) / img_height * 1000)
                    text = cell.get('text', '').strip()
                    if text:
                        standard_segments.append({
                            "bbox": [x1_norm, y1_norm, x2_norm, y2_norm],
                            "text": text
                        })

            if not standard_segments:
                continue

            # Sort by reading order
            sorted_segments = sort_segments_by_reading_order(standard_segments)

            # Presample bboxes (max 15 per doc, stratified by size)
            presampled_segments = presample_bboxes(sorted_segments)

            # Build image path (make relative if base_dir provided)
            final_img_path = img_path
            if image_base_dir:
                try:
                    rel_path = Path(img_path).relative_to(image_base_dir)
                    final_img_path = str(rel_path)
                except ValueError:
                    pass

            # 1. Yield bbox-ocr samples (one per presampled segment)
            for seg in presampled_segments:
                bbox = seg['bbox']
                text = seg['text']
                bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"

                yield {
                    "messages": [
                        {"role": "user", "content": f"Transcribe the text in {bbox_str}:\n<image>"},
                        {"role": "assistant", "content": text}
                    ],
                    "images": [final_img_path],
                    "task": "bbox_ocr",
                    "metadata": {"bbox": bbox}
                }
                total_samples += 1

            # 2. Yield full-ocr sample (one per document)
            # Use ALL segments for complete document text
            text_lines = []
            current_line = []
            y_tolerance_px = 5

            for seg in sorted_segments:
                text = seg['text']
                bbox = seg['bbox']
                y1 = bbox[1] * img_height

                if not current_line:
                    current_line = [(y1, text)]
                elif abs(y1 - current_line[0][0]) <= y_tolerance_px:
                    current_line.append((y1, text))
                else:
                    text_lines.append(' '.join([t for _, t in current_line]))
                    current_line = [(y1, text)]

            if current_line:
                text_lines.append(' '.join([t for _, t in current_line]))

            full_text = '\n'.join(text_lines).strip()

            if full_text:
                # Randomly select OCR instruction
                instruction = random.choice(FULL_IMAGE_OCR_INSTRUCTIONS)

                yield {
                    "messages": [
                        {"role": "user", "content": f"{instruction}:\n<image>"},
                        {"role": "assistant", "content": full_text}
                    ],
                    "images": [final_img_path],
                    "task": "full_document_ocr",
                    "metadata": {
                        "num_segments": len(sorted_segments)
                    }
                }
                total_samples += 1

            # 3. Yield markdown sample (one per document) - only if client available
            markdown = None
            if client and use_markdown_model and full_text:
                # Use model to generate markdown with full OCR text as hint
                print(f"  [{docs_processed + 1}] Generating markdown for {image_filename}...", end='\r')
                markdown = client.generate_markdown(img_path, full_text)

            if markdown:
                # Filter markdown samples: only include if >= 100 chars
                if len(markdown) < 100:
                    continue

                yield {
                    "messages": [
                        {"role": "user", "content": "Convert this image to markdown:\n<image>"},
                        {"role": "assistant", "content": markdown}
                    ],
                    "images": [final_img_path],
                    "task": "markdown_conversion",
                    "metadata": {
                        "source": "doclaynet_qwen2.5vl",
                        "num_segments": len(sorted_segments),
                    }
                }
                total_samples += 1

            docs_processed += 1

            if docs_processed % 100 == 0:
                print(f"  Progress: {docs_processed} documents, {total_samples} samples")

        except Exception as e:
            print(f"\n  Error processing {json_file.name}: {e}")
            continue

    print(f"  ✓ Processed {docs_processed} documents, generated {total_samples} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced DocLayNet Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build per-bbox OCR dataset from DocLayNet JSON directory
    python OCRVL/scripts/build_doclaynet_enhanced.py \\
        --mode bbox-ocr \\
        --doclaynet-json /share/project/xiyan/huggingface/docling-project/DocLayNet/JSON \\
        --images-dir /share/project/xiyan/huggingface/docling-project/DocLayNet/PNG \\
        --output OCRVL/llamafactory/data/doclaynet_bbox_ocr.jsonl

    # Build with limited samples for testing
    python OCRVL/scripts/build_doclaynet_enhanced.py \\
        --mode bbox-ocr \\
        --doclaynet-json /share/project/xiyan/huggingface/docling-project/DocLayNet/JSON \\
        --images-dir /share/project/xiyan/huggingface/docling-project/DocLayNet/PNG \\
        --output OCRVL/llamafactory/data/doclaynet_bbox_ocr_test.jsonl \\
        --max-samples 100

    # Build full-document OCR dataset
    python OCRVL/scripts/build_doclaynet_enhanced.py \\
        --mode full-ocr \\
        --doclaynet-json /share/project/xiyan/huggingface/docling-project/DocLayNet/JSON \\
        --images-dir /share/project/xiyan/huggingface/docling-project/DocLayNet/PNG \\
        --output OCRVL/llamafactory/data/doclaynet_full_ocr.jsonl

    # Build markdown placeholder dataset
    python OCRVL/scripts/build_doclaynet_enhanced.py \\
        --mode markdown-placeholder \\
        --doclaynet-json /share/project/xiyan/huggingface/docling-project/DocLayNet/JSON \\
        --images-dir /share/project/xiyan/huggingface/docling-project/DocLayNet/PNG \\
        --output OCRVL/llamafactory/data/doclaynet_markdown_placeholder.jsonl

    # Build markdown dataset using Qwen2.5-VL model (requires running server)
    python OCRVL/scripts/build_doclaynet_enhanced.py \\
        --mode markdown-model \\
        --doclaynet-json /share/project/xiyan/huggingface/docling-project/DocLayNet/JSON \\
        --images-dir /share/project/xiyan/huggingface/docling-project/DocLayNet/PNG \\
        --output OCRVL/llamafactory/data/doclaynet_markdown.jsonl \\
        --server-url http://localhost:8000 \\
        --workers 8 \\
        --max-samples 1000

    # Build ALL DocLayNet samples (checks for existing intermediates, builds only missing)
    python OCRVL/scripts/build_doclaynet_enhanced.py \\
        --mode all \\
        --doclaynet-json /share/project/xiyan/huggingface/docling-project/DocLayNet/JSON \\
        --images-dir /share/project/xiyan/huggingface/docling-project/DocLayNet/PNG \\
        --output OCRVL/llamafactory/data/doclaynet_all.jsonl \\
        --server-url http://localhost:8000

    # Force complete rebuild (ignore all existing intermediate files)
    python OCRVL/scripts/build_doclaynet_enhanced.py \\
        --mode all \\
        --rebuild \\
        --doclaynet-json /share/project/xiyan/huggingface/docling-project/DocLayNet/JSON \\
        --images-dir /share/project/xiyan/huggingface/docling-project/DocLayNet/PNG \\
        --output OCRVL/llamafactory/data/doclaynet_all.jsonl \\
        --server-url http://localhost:8000

Note:
    - 'all' mode checks for existing intermediate files (bbox/full/markdown) and reuses them
    - Only generates missing components - saves time on incremental builds
    - Use --rebuild to force complete regeneration of all components
    - Intermediate files: doclaynet_bbox_ocr.jsonl, doclaynet_full_ocr.jsonl, doclaynet_markdown.jsonl
    - Requires --server-url for markdown generation with Qwen2.5-VL model
    - Use --workers N for parallel markdown generation (default: 4 workers)
        """
    )

    parser.add_argument('--mode', choices=['bbox-ocr', 'full-ocr', 'markdown-model', 'all'],
                        required=True, help='Dataset mode to generate')
    parser.add_argument('--rebuild', action='store_true',
                        help='Force complete rebuild (ignore existing intermediate files)')
    parser.add_argument('--doclaynet-json', required=True,
                        help='Path to DocLayNet JSON file or directory')
    parser.add_argument('--images-dir', required=True,
                        help='Path to DocLayNet images directory')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file path')
    parser.add_argument('--image-base-dir',
                        default=None,
                        help='Base directory for relative image paths')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to generate (for testing)')
    parser.add_argument('--server-url', default=None,
                        help='Qwen2.5-VL server URL (optional, for model-based markdown)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for markdown generation (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for bbox sampling (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    json_path = Path(args.doclaynet_json)
    image_dir = Path(args.images_dir)
    output_path = Path(args.output)
    image_base_dir = Path(args.image_base_dir) if args.image_base_dir else None

    print("=" * 70)
    print(f"Enhanced DocLayNet Dataset Builder: {args.mode}")
    print("=" * 70)
    print()
    print(f"DocLayNet JSON: {json_path}")
    print(f"Images directory: {image_dir}")
    print(f"Output: {output_path}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"Random seed: {args.seed}")
    print()

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate samples based on mode
    total = 0

    with open(output_path, 'w') as f:
        if args.mode == 'bbox-ocr':
            print("Generating per-bbox OCR samples...")
            print("  Sampling strategy: 5 largest bboxes + 5 random bboxes per document")
            for sample in iter_bbox_ocr_samples(
                json_path=json_path,
                image_dir=image_dir,
                image_base_dir=image_base_dir,
                max_samples=args.max_samples
            ):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                total += 1
                if total % 1000 == 0:
                    print(f"  Progress: {total} samples", end='\r')

        elif args.mode == 'full-ocr':
            print("Generating full-document OCR samples...")
            for sample in iter_full_ocr_samples(
                json_path=json_path,
                image_dir=image_dir,
                image_base_dir=image_base_dir,
                max_samples=args.max_samples
            ):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                total += 1
                if total % 100 == 0:
                    print(f"  Progress: {total} samples", end='\r')

        elif args.mode == 'markdown-model':
            print("Generating markdown with Qwen2.5-VL model...")
            print(f"  Server: {args.server_url}")
            print(f"  Workers: {args.workers}")

            # Check server is running
            try:
                response = requests.get(f"{args.server_url}/v1/models", timeout=5)
                response.raise_for_status()
                print("  ✓ Server is running")
            except Exception as e:
                print(f"  ✗ Server not accessible: {e}")
                return 1

            # Use parallel processing
            for sample in iter_markdown_model_samples_parallel(
                json_path=json_path,
                image_dir=image_dir,
                server_url=args.server_url,
                image_base_dir=image_base_dir,
                max_samples=args.max_samples,
                num_workers=args.workers
            ):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                total += 1

        elif args.mode == 'all':
            print("Generating ALL DocLayNet samples (bbox-ocr + full-ocr + markdown)...")

            # Define intermediate file paths
            bbox_file = output_path.parent / "doclaynet_bbox_ocr.jsonl"
            full_file = output_path.parent / "doclaynet_full_ocr.jsonl"
            markdown_file = output_path.parent / "doclaynet_markdown.jsonl"

            # Check for --rebuild flag
            if args.rebuild:
                print("  --rebuild flag: Ignoring all existing files, rebuilding from scratch...")
                reuse_bbox = False
                reuse_full = False
                reuse_markdown = False
            else:
                print("  Checking for existing intermediate files...")
                reuse_bbox = False
                reuse_full = False
                reuse_markdown = False

                # Check bbox-ocr
                if bbox_file.exists() and bbox_file.stat().st_size > 1000:
                    with open(bbox_file, 'r') as bf:
                        bbox_count = sum(1 for line in bf if line.strip())
                    print(f"  Found existing bbox-ocr: {bbox_file} ({bbox_count} samples)")
                    reuse_bbox = True

                # Check full-ocr
                if full_file.exists() and full_file.stat().st_size > 1000:
                    with open(full_file, 'r') as ff:
                        full_count = sum(1 for line in ff if line.strip())
                    print(f"  Found existing full-ocr: {full_file} ({full_count} samples)")
                    reuse_full = True

                # Check markdown
                if markdown_file.exists() and markdown_file.stat().st_size > 1000:
                    with open(markdown_file, 'r') as mf:
                        markdown_count = sum(1 for line in mf if line.strip())
                    print(f"  Found existing markdown: {markdown_file} ({markdown_count} samples)")
                    reuse_markdown = True

                # Report what will be generated
                missing = []
                if not reuse_bbox:
                    missing.append("bbox-ocr")
                if not reuse_full:
                    missing.append("full-ocr")
                if not reuse_markdown:
                    missing.append("markdown")

                if not missing:
                    print("  All intermediate files exist - combining into final output")
                    print("  (Use --rebuild to force regeneration)")
                else:
                    print(f"  Generating missing components: {', '.join(missing)}")
                    print("  (Use --rebuild to ignore existing files and rebuild all)")

            # Initialize client for markdown generation if needed
            client = None
            if not reuse_markdown and args.server_url:
                try:
                    response = requests.get(f"{args.server_url}/v1/models", timeout=5)
                    response.raise_for_status()
                    print(f"  Server: {args.server_url}")
                    print("  ✓ Server is running - using model for markdown")
                    client = QwenVLClient(args.server_url)
                except Exception as e:
                    print(f"  Server not accessible ({e}) - markdown generation will fail")
                    print("  (Run without --server-url to skip markdown, or ensure server is running)")

            # Generate/reuse bbox-ocr
            if reuse_bbox:
                print(f"  Reusing bbox-ocr from {bbox_file}...")
                with open(bbox_file, 'r') as bf:
                    for line in bf:
                        if line.strip():
                            f.write(line)
                            total += 1
            else:
                print(f"  Generating bbox-ocr samples...")
                bbox_count = 0
                for sample in iter_bbox_ocr_samples(
                    json_path=json_path,
                    image_dir=image_dir,
                    image_base_dir=image_base_dir,
                    max_samples=args.max_samples
                ):
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    total += 1
                    bbox_count += 1
                    if bbox_count % 1000 == 0:
                        print(f"    Progress: {bbox_count} bbox samples", end='\r')
                print(f"    Generated {bbox_count} bbox samples")

            # Generate/reuse full-ocr
            if reuse_full:
                print(f"  Reusing full-ocr from {full_file}...")
                with open(full_file, 'r') as ff:
                    for line in ff:
                        if line.strip():
                            f.write(line)
                            total += 1
            else:
                print(f"  Generating full-ocr samples...")
                full_count = 0
                for sample in iter_full_ocr_samples(
                    json_path=json_path,
                    image_dir=image_dir,
                    image_base_dir=image_base_dir,
                    max_samples=args.max_samples
                ):
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    total += 1
                    full_count += 1
                    if full_count % 100 == 0:
                        print(f"    Progress: {full_count} full samples", end='\r')
                print(f"    Generated {full_count} full samples")

            # Generate/reuse markdown
            if reuse_markdown:
                print(f"  Reusing markdown from {markdown_file}...")
                with open(markdown_file, 'r') as mf:
                    for line in mf:
                        if line.strip():
                            f.write(line)
                            total += 1
            elif client:
                print(f"  Generating markdown samples with model...")
                markdown_count = 0
                for sample in iter_markdown_model_samples(
                    json_path=json_path,
                    image_dir=image_dir,
                    client=client,
                    image_base_dir=image_base_dir,
                    max_samples=args.max_samples
                ):
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    total += 1
                    markdown_count += 1
                print(f"    Generated {markdown_count} markdown samples")
            else:
                print(f"  Skipping markdown (no client available)")

    print(f"\n  Progress: {total} samples")
    print()

    print("=" * 70)
    print(f"✓ Built enhanced DocLayNet dataset: {output_path}")
    print(f"  Total samples: {total}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
