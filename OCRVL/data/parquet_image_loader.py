"""
Parquet-based image dataset for LLaVA training with mix665k.

Handles images stored in parquet files (GQA, OCR-VQA, TextVQA) and 
regular image files (COCO, Visual Genome).
"""

import os
import io
import json
from typing import Dict, Tuple, Optional
from PIL import Image
import pyarrow.parquet as pq
from datasets import load_dataset


class ParquetImageIndex:
    """Index for quickly loading images from parquet files with caching."""

    def __init__(self, parquet_dir: str, id_column: str = 'id'):
        """
        Args:
            parquet_dir: Directory containing parquet files
            id_column: Column name containing image IDs
        """
        self.parquet_dir = parquet_dir
        self.id_column = id_column
        self.index = {}  # id -> (parquet_file, row_index)
        self.table_cache = {}  # parquet_file -> pandas DataFrame (CACHED!)
        self._build_index()

    def _build_index(self):
        """Build index mapping image ID to parquet file and row, and cache tables."""
        parquet_files = sorted([
            f for f in os.listdir(self.parquet_dir)
            if f.endswith('.parquet')
        ])

        print(f"Building index for {len(parquet_files)} parquet files in {self.parquet_dir}...")
        total_images = 0

        for pf in parquet_files:
            parquet_path = os.path.join(self.parquet_dir, pf)
            # Read the full table once and cache it
            table = pq.read_table(parquet_path, columns=[self.id_column, 'image'])
            df = table.to_pandas()
            self.table_cache[pf] = df  # Cache the entire table!

            for idx, row in df.iterrows():
                image_id = str(row[self.id_column])
                self.index[image_id] = (pf, idx)
                total_images += 1

        print(f"✓ Indexed {total_images:,} images (all parquet tables cached in memory)")

    def get_image(self, image_id: str) -> Optional[Image.Image]:
        """Load image by ID from cached parquet table and convert to RGB."""
        if image_id not in self.index:
            return None

        parquet_file, row_idx = self.index[image_id]

        # Use cached table instead of re-reading from disk!
        df = self.table_cache[parquet_file]
        image_data = df.iloc[row_idx]['image']

        if isinstance(image_data, dict) and 'bytes' in image_data:
            img_bytes = image_data['bytes']
            img = Image.open(io.BytesIO(img_bytes))
            # Convert to RGB to ensure 3 channels (handles RGBA, L, etc.)
            return img.convert('RGB')

        return None


class Mix665kImageLoader:
    """Image loader for LLaVA mix665k dataset with multiple backends."""
    
    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: Base directory containing image subdirectories
                     (e.g., /path/to/LLaVA-Instruct-150K/images/)
        """
        self.base_dir = base_dir
        self.parquet_indices = {}
        self._setup_indices()
    
    def _setup_indices(self):
        """Setup parquet indices for datasets stored in parquet format."""
        # GQA - images in parquet with 'id' column
        gqa_dir = os.path.join(self.base_dir, "gqa")
        if os.path.exists(gqa_dir):
            print("Setting up GQA parquet index...")
            self.parquet_indices['gqa'] = ParquetImageIndex(gqa_dir, id_column='id')

        # OCR-VQA - images in parquet with 'image_id' column
        ocr_vqa_dir = os.path.join(self.base_dir, "ocr_vqa")
        if os.path.exists(ocr_vqa_dir):
            print("Setting up OCR-VQA parquet index...")
            self.parquet_indices['ocr_vqa'] = ParquetImageIndex(ocr_vqa_dir, id_column='image_id')

        # TextVQA - images in parquet with 'image_id' column
        textvqa_dir = os.path.join(self.base_dir, "textvqa")
        if os.path.exists(textvqa_dir):
            print("Setting up TextVQA parquet index...")
            self.parquet_indices['textvqa'] = ParquetImageIndex(textvqa_dir, id_column='image_id')
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load image from path specified in mix665k.json.
        
        Args:
            image_path: Relative path like "gqa/images/2354786.jpg"
                                        or "coco/train2017/000000033471.jpg"
        
        Returns:
            PIL Image or None if not found
        """
        # Parse the path
        parts = image_path.split('/')
        if len(parts) < 2:
            return None
        
        dataset = parts[0]
        
        # Handle parquet-based datasets
        if dataset == 'gqa':
            # Extract ID from "gqa/images/2354786.jpg" -> "2354786"
            image_id = parts[-1].replace('.jpg', '')
            if 'gqa' in self.parquet_indices:
                return self.parquet_indices['gqa'].get_image(image_id)
        
        elif dataset == 'ocr_vqa':
            # Extract ID from "ocr_vqa/images/140031996X.jpg" -> "140031996X"
            image_id = parts[-1].replace('.jpg', '')
            if 'ocr_vqa' in self.parquet_indices:
                return self.parquet_indices['ocr_vqa'].get_image(image_id)
        
        elif dataset == 'textvqa':
            # Extract ID from "textvqa/train_images/011e7e629fb9ae7b.jpg"
            image_id = parts[-1].replace('.jpg', '')
            if 'textvqa' in self.parquet_indices:
                return self.parquet_indices['textvqa'].get_image(image_id)

        # Handle file-based datasets (COCO, VG)
        else:
            # Try loading from filesystem
            full_path = os.path.join(self.base_dir, image_path)
            if os.path.exists(full_path):
                img = Image.open(full_path)
                # Convert to RGB to ensure 3 channels
                return img.convert('RGB')

        return None


if __name__ == "__main__":
    # Test the loader
    print("Testing Mix665k Image Loader...")
    loader = Mix665kImageLoader("/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/images")
    
    # Test GQA
    print("\nTesting GQA image load...")
    img = loader.load_image("gqa/images/2375429.jpg")
    if img:
        print(f"✓ Loaded GQA image: {img.size}")
    else:
        print("✗ Failed to load GQA image")
    
    # Test OCR-VQA  
    print("\nTesting OCR-VQA image load...")
    img = loader.load_image("ocr_vqa/images/195338561.jpg")
    if img:
        print(f"✓ Loaded OCR-VQA image: {img.size}")
    else:
        print("✗ Failed to load OCR-VQA image")
