#!/usr/bin/env python3
"""
BLIP3o Dataset Loader - Simple version for raw image-caption pairs

Provides:
- WebDataset loading with HuggingFace caching
- Percentage sampling
- Returns raw {"image": PIL.Image, "caption": str}

Task-specific formatting, rendering, and tokenization handled in training step.
"""

import glob
import io
import os
import logging
import random
from typing import Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)


class WebDatasetWrapper:
    """
    Wrapper for HuggingFace datasets that provides list-like interface.

    Provides efficient access to WebDataset with deterministic sampling.
    """

    def __init__(self, hf_dataset, meta_type, sample_percentage=None, seed=42):
        self.hf_dataset = hf_dataset
        self.meta_type = meta_type
        self._full_length = len(hf_dataset)

        # Apply sampling if specified (deterministic with fixed seed)
        if sample_percentage is not None and 0 < sample_percentage < 1:
            self._length = max(1, int(self._full_length * sample_percentage))
            # Create deterministic sample indices using fixed seed
            rng = random.Random(seed)
            self._sample_indices = sorted(rng.sample(range(self._full_length), self._length))
            logger.info(f"Sampling {self._length} items ({sample_percentage*100:.2f}%) from {self._full_length} total items (deterministic, seed={seed})")
        else:
            self._length = self._full_length
            self._sample_indices = None  # Use all indices

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """Get item from HF dataset and parse on-the-fly."""
        # Map to actual index if sampling
        if self._sample_indices is not None:
            if idx >= len(self._sample_indices):
                raise IndexError(f"Index {idx} out of range for sampled dataset of length {len(self._sample_indices)}")
            actual_idx = self._sample_indices[idx]
        else:
            actual_idx = idx

        item = self.hf_dataset[actual_idx]

        # Parse BLIP3o format: {' __key__': '...', 'jpg': bytes, 'txt': bytes}
        result = {}

        # Extract image
        if 'jpg' in item:
            jpg_data = item['jpg']
            if isinstance(jpg_data, bytes):
                result['image'] = Image.open(io.BytesIO(jpg_data)).convert('RGB')
            elif isinstance(jpg_data, Image.Image):
                result['image'] = jpg_data.convert('RGB')
            else:
                logger.warning(f"Unexpected jpg type: {type(jpg_data)}")
                result['image'] = None

        # Extract caption
        if 'txt' in item:
            txt_content = item['txt']
            if isinstance(txt_content, bytes):
                txt_content = txt_content.decode('utf-8', errors='ignore')
            elif isinstance(txt_content, torch.Tensor):
                try:
                    txt_content = bytes(txt_content.detach().cpu().numpy().tobytes()).decode('utf-8', errors='ignore')
                except Exception:
                    txt_content = str(txt_content)
            result['caption'] = txt_content.strip()

        return result


def load_blip3o_webdataset(
    data_path: str,
    sample_percentage: Optional[float] = None,
    cache_dir: Optional[str] = None,
    num_proc: int = 8,
    seed: int = 42,
) -> WebDatasetWrapper:
    """
    Load BLIP3o WebDataset from tar archives using HuggingFace's infrastructure.

    Args:
        data_path: Directory containing *.tar files
        sample_percentage: Fraction of dataset to sample (e.g., 0.01 for 1%)
        cache_dir: Cache directory for HuggingFace datasets
        num_proc: Number of processes for parallel loading
        seed: Random seed for deterministic sampling

    Returns:
        WebDatasetWrapper with efficient access
    """
    # Find all tar files
    if os.path.isdir(data_path):
        tar_shards = sorted(glob.glob(os.path.join(data_path, '*.tar')))
    else:
        tar_shards = sorted(glob.glob(data_path))

    if not tar_shards:
        raise ValueError(f"No .tar files found in {data_path}")

    # If sample_percentage is specified, only load a subset of shards
    # This dramatically speeds up dataset loading
    if sample_percentage is not None and sample_percentage < 1.0:
        num_shards_to_load = max(1, int(len(tar_shards) * sample_percentage))
        tar_shards = tar_shards[:num_shards_to_load]
        logger.info(f"Found {len(glob.glob(os.path.join(data_path, '*.tar')))} total shards, loading {num_shards_to_load} ({sample_percentage*100:.1f}%)")
    else:
        logger.info(f"Found {len(tar_shards)} WebDataset tar shards in {data_path}")

    logger.info(f"Loading WebDataset shards (first load may take time, cached thereafter)")

    # Use Lumina-DiMOO's cache directory
    if cache_dir is None:
        cache_dir = os.environ.get('HF_DATASETS_CACHE', '/share/project/xiyan/huggingface/cache/datasets')

    # Simple loading - let HuggingFace handle caching
    # All ranks can load simultaneously, HF cache handles conflicts
    num_proc_arg = None if num_proc == 0 else num_proc
    hf_dataset = load_dataset(
        'webdataset',
        data_files=tar_shards,
        cache_dir=cache_dir,
        split='train',
        num_proc=num_proc_arg,
        keep_in_memory=False,  # Use disk cache
    )

    logger.info(f"✓ Dataset loaded with {len(hf_dataset)} items (cached at {cache_dir})")

    # Wrap for efficient sampling
    # NOTE: We already sampled shards above, so pass None to avoid double-sampling
    return WebDatasetWrapper(hf_dataset, "blip3o", sample_percentage=None, seed=seed)


class BLIP3oDataset(Dataset):
    """
    Simple BLIP3o dataset - returns raw image-caption pairs.

    Task-specific formatting handled in training step.

    Supports three dataset variants:
    - Short captions: ~10-20 tokens, concise descriptions (772GB, 27M images)
    - Long captions: ~120 tokens, detailed descriptions (27M images)
    - 60k: Curated 60k high-quality samples from multiple sources
    """

    def __init__(
        self,
        dataset_type: str = "mixed",  # "short", "long", "60k", or "mixed"
        base_path: str = "/share/project/xiyan/huggingface/BLIP3o",
        mix_ratio: float = 0.5,  # For "mixed": 0.5 = 50% short, 50% long
        sample_percentage: float = 0.01,  # Use 1% of dataset by default
        num_proc: int = 8,  # HuggingFace dataset loading processes
        cache_dir: Optional[str] = None,  # HuggingFace cache dir
        seed: int = 42,  # Random seed for sampling
    ):
        """
        Args:
            dataset_type: Which BLIP3o dataset to use
                - "short": Short captions only (~10-20 tokens)
                - "long": Long captions only (~120 tokens)
                - "60k": Curated 60k high-quality dataset
                - "mixed": Mix of short and long (uses mix_ratio)
            base_path: Base path to BLIP3o datasets
            mix_ratio: For "mixed" type, ratio of short to long (0=all long, 1=all short)
            sample_percentage: Fraction of dataset to use (e.g., 0.01 = 1%)
            num_proc: Number of processes for HuggingFace dataset loading
            cache_dir: Cache directory (defaults to Lumina-DiMOO's cache)
            seed: Random seed for deterministic sampling
        """
        self.dataset_type = dataset_type
        self.mix_ratio = mix_ratio
        self.seed = seed

        # Determine paths based on dataset type
        if dataset_type == "short":
            short_path = os.path.join(base_path, "BLIP3o-Pretrain-Short-Caption")
            long_path = None
            num_short_pct = sample_percentage
            num_long_pct = None
        elif dataset_type == "long":
            short_path = None
            long_path = os.path.join(base_path, "BLIP3o-Pretrain-Long-Caption")
            num_short_pct = None
            num_long_pct = sample_percentage
        elif dataset_type == "60k":
            short_path = os.path.join(base_path, "BLIP3o-60k")
            long_path = None
            num_short_pct = sample_percentage  # For 60k, use "short" loader
            num_long_pct = None
        elif dataset_type == "mixed":
            short_path = os.path.join(base_path, "BLIP3o-Pretrain-Short-Caption")
            long_path = os.path.join(base_path, "BLIP3o-Pretrain-Long-Caption")
            num_short_pct = sample_percentage * mix_ratio if sample_percentage else None
            num_long_pct = sample_percentage * (1 - mix_ratio) if sample_percentage else None
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose from: short, long, 60k, mixed")

        # Load short captions (or 60k)
        self.short_sampler = None
        if short_path and os.path.exists(short_path):
            try:
                self.short_sampler = load_blip3o_webdataset(
                    short_path,
                    sample_percentage=num_short_pct,
                    cache_dir=cache_dir,
                    num_proc=num_proc,
                    seed=seed
                )
            except Exception as e:
                logger.warning(f"Failed to load from {short_path}: {e}")

        # Load long captions
        self.long_sampler = None
        if long_path and os.path.exists(long_path):
            try:
                self.long_sampler = load_blip3o_webdataset(
                    long_path,
                    sample_percentage=num_long_pct,
                    cache_dir=cache_dir,
                    num_proc=num_proc,
                    seed=seed + 1  # Different seed for long captions
                )
            except Exception as e:
                logger.warning(f"Failed to load from {long_path}: {e}")

        # Combine indices
        self.indices = []
        if self.short_sampler:
            label = "60k" if dataset_type == "60k" else "short"
            self.indices.extend([(label, i) for i in range(len(self.short_sampler))])
        if self.long_sampler:
            self.indices.extend([('long', i) for i in range(len(self.long_sampler))])

        # Shuffle with fixed seed for reproducibility
        rng = random.Random(seed + 2)
        rng.shuffle(self.indices)

        logger.info(f"✓ BLIP3o Dataset loaded:")
        logger.info(f"  Dataset type: {dataset_type}")
        if dataset_type == "60k":
            logger.info(f"  60k samples: {len(self.short_sampler) if self.short_sampler else 0}")
        else:
            logger.info(f"  Short captions: {len(self.short_sampler) if self.short_sampler else 0}")
            logger.info(f"  Long captions: {len(self.long_sampler) if self.long_sampler else 0}")
        logger.info(f"  Total samples: {len(self.indices)}")
        logger.info(f"  Sample percentage: {sample_percentage*100 if sample_percentage else 100:.2f}%")
        logger.info(f"  Random seed: {seed}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get a raw sample (no task-specific formatting).

        Returns:
            dict with:
                - image: PIL.Image
                - caption: str
                - source: str ("short", "long", or "60k")
        """
        source, base_idx = self.indices[idx]

        # Get image and caption
        if source in ['short', '60k']:
            item = self.short_sampler[base_idx]
        else:  # 'long'
            item = self.long_sampler[base_idx]

        return {
            "image": item.get('image'),
            "caption": item.get('caption', ''),
            "source": source,
        }


if __name__ == "__main__":
    # Quick test
    print("Testing BLIP3oDataset (simple version)...")

    dataset = BLIP3oDataset(
        dataset_type="long",
        sample_percentage=0.0001,  # Use 0.01% for quick test
        mix_ratio=0.5,
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        # Load a sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image: {sample['image']}")
        print(f"  Caption: {sample['caption'][:100]}...")
        print(f"  Source: {sample['source']}")

    print("\n✓ Test passed!")
    print(f"Cache location: /share/project/xiyan/huggingface/cache/datasets")
