"""
Cached Visual Token Dataset for Fast Markovian Training

Loads pre-computed visual tokens from disk for maximum training speed.
Eliminates the CPU bottleneck of text rendering and encoding.

Usage:
    1. Pre-compute visual tokens:
       python OCRFlow/scripts/precompute_vistok.py \
           --dataset_type fineweb \
           --fineweb_subset 10BT \
           --output_dir ./cache/vistok_fineweb_10bt \
           --max_chunks 1000000

    2. Train with cached data:
       python OCRFlow/examples/train_markovian.py \
           --use_cached_vistok \
           --cache_dir ./cache/vistok_fineweb_10bt \
           --batch_size 256
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class CachedMarkovianDataset(Dataset):
    """
    Fast-loading dataset from pre-computed visual tokens.

    Expected structure:
        cache_dir/
            chunks/
                00000000.pt  # Contains: {'tokens': [111, 1280], 'text_hash': str, 'word_count': int}
                00000001.pt
                ...
            metadata.json   # Contains: total_chunks, config, etc.
            pairs.json      # Contains: list of (chunk_i, chunk_i+1) pair indices

    Args:
        cache_dir: Directory with pre-computed visual tokens
        shuffle: Shuffle pairs each epoch
        max_samples: Maximum number of pairs to use (None = all)
        load_to_memory: If True, load all tokens to RAM for faster access
    """

    def __init__(
        self,
        cache_dir: str,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        load_to_memory: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.chunks_dir = self.cache_dir / "chunks"
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.load_to_memory = load_to_memory

        # Load metadata
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata not found at {metadata_path}. Run precompute_vistok.py first.")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded metadata: {self.metadata.get('total_chunks', 0)} chunks, "
                   f"{self.metadata.get('total_pairs', 0)} pairs")

        # Load pairs
        pairs_path = self.cache_dir / "pairs.json"
        if not pairs_path.exists():
            raise ValueError(f"Pairs not found at {pairs_path}. Run precompute_vistok.py first.")

        with open(pairs_path) as f:
            self.pairs = json.load(f)

        # Limit samples if specified
        if self.max_samples and len(self.pairs) > self.max_samples:
            self.pairs = self.pairs[:self.max_samples]

        logger.info(f"Using {len(self.pairs)} training pairs")

        # Shuffle pairs
        if self.shuffle:
            random.shuffle(self.pairs)

        # Memory cache for loaded chunks
        self.memory_cache = {}

        # Optionally load all chunks to memory
        if self.load_to_memory:
            self._load_all_to_memory()

    def _load_all_to_memory(self):
        """Load all chunks to RAM"""
        logger.info("Loading all chunks to memory...")

        # Find all unique chunk indices
        unique_indices = set()
        for i, j in self.pairs:
            unique_indices.add(i)
            unique_indices.add(j)

        for idx in unique_indices:
            chunk_path = self.chunks_dir / f"{idx:08d}.pt"
            if chunk_path.exists():
                data = torch.load(chunk_path, map_location="cpu")
                self.memory_cache[idx] = data['tokens'].float()

        logger.info(f"Loaded {len(self.memory_cache)} chunks to memory")

    def _load_chunk(self, idx: int) -> torch.Tensor:
        """Load a single chunk by index"""
        # Check memory cache
        if idx in self.memory_cache:
            return self.memory_cache[idx]

        chunk_path = self.chunks_dir / f"{idx:08d}.pt"
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk not found: {chunk_path}")

        data = torch.load(chunk_path, map_location="cpu")
        tokens = data['tokens'].float()  # Convert from float16 to float32

        return tokens

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (input_chunk, target_chunk)
            Each tensor has shape [111, 1280]
        """
        input_idx, target_idx = self.pairs[idx]

        input_chunk = self._load_chunk(input_idx)
        target_chunk = self._load_chunk(target_idx)

        return input_chunk, target_chunk


def create_cached_dataloader(
    cache_dir: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 8,
    max_samples: Optional[int] = None,
    load_to_memory: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create dataloader for cached visual tokens.

    This is MUCH faster than on-the-fly encoding (~10-100x speedup).

    Args:
        cache_dir: Directory with pre-computed visual tokens
        batch_size: Batch size
        shuffle: Shuffle pairs
        num_workers: DataLoader workers (set high, e.g. 8-16)
        max_samples: Max training pairs
        load_to_memory: Load all chunks to RAM for fastest access
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader yielding (input_batch, target_batch) tuples
        Each tensor has shape [batch, 111, 1280]
    """
    dataset = CachedMarkovianDataset(
        cache_dir=cache_dir,
        shuffle=shuffle,
        max_samples=max_samples,
        load_to_memory=load_to_memory,
    )

    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        inputs = torch.stack([pair[0] for pair in batch], dim=0)
        targets = torch.stack([pair[1] for pair in batch], dim=0)
        return inputs, targets

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Additional shuffle per epoch
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return dataloader


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing CachedMarkovianDataset...")

    # Test with default cache location
    cache_dir = "./cache/vistok_fineweb_10bt"

    if not Path(cache_dir).exists():
        print(f"Cache directory not found: {cache_dir}")
        print("Run precompute_vistok.py first to create the cache.")
        exit(1)

    # Create dataloader
    dataloader = create_cached_dataloader(
        cache_dir=cache_dir,
        batch_size=64,
        num_workers=4,
        max_samples=1000,
    )

    print(f"\nDataset size: {len(dataloader.dataset)}")
    print(f"Batch count: {len(dataloader)}")

    # Test iteration
    import time
    start = time.time()

    for i, (inputs, targets) in enumerate(dataloader):
        if i == 0:
            print(f"\nInputs shape: {inputs.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Dtype: {inputs.dtype}")

        if i >= 9:  # Test 10 batches
            break

    elapsed = time.time() - start
    print(f"\n10 batches loaded in {elapsed:.2f}s")
    print(f"Throughput: {10 * 64 / elapsed:.1f} samples/sec")

    print("\n✓ Test passed!")
