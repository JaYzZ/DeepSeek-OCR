#!/usr/bin/env python3
"""
DocLayNet Dataset for OCRVL Alignment Stage

DocLayNet provides 80K+ document pages with:
- Human-annotated layout segmentation (11 categories)
- 1025×1025 PNG images
- Ground truth text (from DocLayNet_extra.zip)
- Diverse document types (finance, science, patents, law, manuals, tenders)

Dataset structure after extraction:
    DocLayNet/
    ├── COCO/
    │   ├── train.json (69,375 samples)
    │   ├── val.json (6,489 samples)
    │   └── test.json (4,999 samples)
    ├── PNG/
    │   └── <hash>.png (80,863 images)
    └── JSON/  (from DocLayNet_extra.zip)
        └── <hash>.json (text cells with coordinates)

Usage:
    from OCRVL.data.doclaynet_dataset import DocLayNetOCRDataset

    dataset = DocLayNetOCRDataset(
        split='train',
        data_dir='/share/project/xiyan/huggingface/docling-project/DocLayNet',
        max_text_length=3072
    )
"""

import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import torch
from torch.utils.data import Dataset


class DocLayNetOCRDataset(Dataset):
    """
    DocLayNet dataset for document OCR alignment training.

    Each sample contains a document image and extracted text content.
    """

    # DocLayNet layout categories
    CATEGORIES = [
        "Caption", "Footnote", "Formula", "List-item",
        "Page-footer", "Page-header", "Picture", "Section-header",
        "Table", "Text", "Title"
    ]

    def __init__(
        self,
        split: str = 'train',
        data_dir: str = '/share/project/xiyan/huggingface/docling-project/DocLayNet',
        max_text_length: int = 2048,
        seed: int = 42,
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            data_dir: Root directory of extracted DocLayNet (contains COCO/, PNG/, JSON/)
            max_text_length: Maximum text length in characters
            seed: Random seed for reproducibility
        """
        self.split = split
        self.data_dir = Path(data_dir)
        self.max_text_length = max_text_length
        self.rng = random.Random(seed)

        # Paths
        self.coco_dir = self.data_dir / 'COCO'
        self.png_dir = self.data_dir / 'PNG'
        self.json_dir = self.data_dir / 'JSON'

        # Load COCO annotations
        coco_file = self.coco_dir / f'{split}.json'
        if not coco_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {coco_file}\n"
                f"Please extract DocLayNet_core.zip to {self.data_dir}"
            )

        print(f"Loading DocLayNet {split} split from {coco_file}...")
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        self.images = coco_data['images']
        self.annotations = coco_data.get('annotations', [])

        # Build image_id -> annotations mapping
        self.image_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)

        # Check if JSON directory exists
        self.has_text_jsons = self.json_dir.exists()

        print(f"✓ Loaded {len(self.images)} samples from DocLayNet {split}")
        print(f"  PNG directory: {self.png_dir}")
        print(f"  JSON directory: {self.json_dir} ({'Found' if self.has_text_jsons else 'Not found - extract DocLayNet_extra.zip'})")
        print(f"  Total annotations: {len(self.annotations)}")
        print(f"  Image resolution: 1025×1025 PNG (will be encoded to 640×640 by OCR encoder)")

    def __len__(self):
        return len(self.images)

    def _extract_text_from_json(self, image_hash: str) -> str:
        """
        Extract ground truth OCR text from DocLayNet JSON file.

        Raw JSON format:
        {
            "metadata": {...},
            "cells": [
                {
                    "text": "CONTENTS",
                    "bbox": [x, y, w, h],
                    "font": {"color": [...], "name": "...", "size": ...}
                },
                ...
            ]
        }

        Extraction process:
        1. Load JSON file: DocLayNet/JSON/<image_hash>.json
        2. Get all cells with text content
        3. Sort cells by y-coordinate (top to bottom reading order)
        4. Join all text with spaces: ' '.join(texts)

        Returns:
            Ground truth OCR text as a single string
        """
        json_path = self.json_dir / f"{image_hash}.json"

        if not json_path.exists():
            return ""

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract text cells sorted by y-coordinate (top to bottom)
            cells = data.get('cells', [])
            if not cells:
                return ""

            # Sort by y-coordinate for reading order
            sorted_cells = sorted(cells, key=lambda c: c.get('bbox', [0, 0, 0, 0])[1])

            # Concatenate text with spaces
            texts = [cell.get('text', '').strip() for cell in sorted_cells if cell.get('text')]

            return ' '.join(texts)

        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}")
            return ""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a sample formatted for document OCR training.

        Image processing:
        - Loads 1025×1025 PNG image as PIL.Image
        - Image will be resized to 640×640 by OCR encoder during training
        - OCR encoder outputs 100 visual tokens (grid only, no newline tokens)

        Text processing:
        - Extracts ground truth text from JSON (cells joined with spaces)
        - Text is used as-is for OCR supervision
        - Character-based truncation applied if needed (legacy)

        Returns:
            {
                'image': PIL.Image (1025×1025),
                'caption': str (ground truth OCR text),
                'doc_category': str,
                'doc_name': str,
                'page_no': int,
                'image_id': int,
            }
        """
        image_info = self.images[idx]

        # Load image (1025×1025 PNG, will be resized to 640×640 by encoder)
        image_path = self.png_dir / image_info['file_name']
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')

        # Extract ground truth OCR text from JSON file
        image_hash = Path(image_info['file_name']).stem

        if self.has_text_jsons:
            caption = self._extract_text_from_json(image_hash)

            # Fallback to layout description if no text found
            if not caption:
                caption = self._generate_layout_fallback(image_info)
        else:
            # Use layout description as fallback
            caption = self._generate_layout_fallback(image_info)

        # Truncate if too long (character-based, legacy)
        if len(caption) > self.max_text_length:
            caption = caption[:self.max_text_length]

        return {
            'image': image,
            'caption': caption,
            'doc_category': image_info.get('doc_category', ''),
            'doc_name': image_info.get('doc_name', ''),
            'page_no': image_info.get('page_no', 0),
            'image_id': image_info['id'],
        }

    def _generate_layout_fallback(self, image_info: Dict) -> str:
        """Generate layout description as fallback when no text available."""
        doc_category = image_info.get('doc_category', 'document')
        doc_name = image_info.get('doc_name', 'unknown')
        page_no = image_info.get('page_no', 0)

        # Get annotations for this image
        image_id = image_info['id']
        annotations = self.image_annotations.get(image_id, [])

        # Count layout elements by category
        category_counts = {}
        for ann in annotations:
            cat_id = ann.get('category_id', -1)
            if 0 <= cat_id < len(self.CATEGORIES):
                cat_name = self.CATEGORIES[cat_id]
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        # Build caption
        caption_parts = [
            f"Document: {doc_category.replace('_', ' ').title()}",
            f"Source: {doc_name}",
            f"Page: {page_no}",
        ]

        if category_counts:
            layout_desc = ", ".join([
                f"{count} {cat.lower()}{'s' if count > 1 else ''}"
                for cat, count in sorted(category_counts.items(),
                                        key=lambda x: -x[1])[:5]
            ])
            caption_parts.append(f"Layout: {layout_desc}")

        return " | ".join(caption_parts)

    def get_sample_for_visualization(self, idx: int) -> Dict:
        """Get a sample with additional metadata for visualization."""
        sample = self[idx]
        image_info = self.images[idx]
        annotations = self.image_annotations.get(image_info['id'], [])

        return {
            **sample,
            'annotations': annotations,
            'width': image_info.get('width', 1025),
            'height': image_info.get('height', 1025),
        }


def test_doclaynet_loading():
    """Test DocLayNet dataset loading"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    print("Testing DocLayNet dataset loading...")

    # Load dataset
    dataset = DocLayNetOCRDataset(split='train')

    # Get first sample
    sample = dataset[0]

    print("\nSample structure:")
    print(f"  Image shape: {sample['image'].size}")
    print(f"  Caption: {sample['caption'][:200]}...")
    print(f"  Document category: {sample['doc_category']}")
    print(f"  Document name: {sample['doc_name']}")
    print(f"  Page number: {sample['page_no']}")
    print(f"  Image ID: {sample['image_id']}")

    # Visualize with bounding boxes
    vis_sample = dataset.get_sample_for_visualization(0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(vis_sample['image'])

    # Draw bounding boxes with category colors
    colors = plt.cm.tab10(range(len(DocLayNetOCRDataset.CATEGORIES)))

    for ann in vis_sample.get('annotations', []):
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            cat_id = ann.get('category_id', 0)
            color = colors[cat_id % len(colors)]

            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    ax.set_title(f"{vis_sample['doc_category']} - {vis_sample['doc_name']} (Page {vis_sample['page_no']})")
    ax.axis('off')

    output_path = '/tmp/doclaynet_sample.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\n✓ Visualization saved to: {output_path}")


if __name__ == '__main__':
    test_doclaynet_loading()
