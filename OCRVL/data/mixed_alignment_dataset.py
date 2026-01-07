#!/usr/bin/env python3
"""
Mixed Alignment Dataset - Combines BLIP3o and DocLayNet

Interleaves samples from both datasets for diverse alignment training.
"""

import random
from torch.utils.data import Dataset
from typing import List, Dict, Any


class MixedAlignmentDataset(Dataset):
    """
    Mixed dataset combining BLIP3o and DocLayNet for alignment training.

    Interleaves samples from both datasets based on their sizes.
    """

    def __init__(
        self,
        blip3o_dataset,
        doclaynet_dataset,
        seed=42,
    ):
        """
        Args:
            blip3o_dataset: BLIP3o dataset instance
            doclaynet_dataset: DocLayNet dataset instance
            seed: Random seed for shuffling
        """
        self.blip3o_dataset = blip3o_dataset
        self.doclaynet_dataset = doclaynet_dataset

        self.blip3o_len = len(blip3o_dataset)
        self.doclaynet_len = len(doclaynet_dataset)
        self.total_len = self.blip3o_len + self.doclaynet_len

        # Create shuffled index mapping
        # Each index maps to (dataset_id, dataset_index)
        # dataset_id: 0 = BLIP3o, 1 = DocLayNet
        self.index_mapping = self._create_index_mapping(seed)

        print(f"Mixed Alignment Dataset:")
        print(f"  BLIP3o: {self.blip3o_len:,} samples")
        print(f"  DocLayNet: {self.doclaynet_len:,} samples")
        print(f"  Total: {self.total_len:,} samples")

    def _create_index_mapping(self, seed):
        """Create shuffled index mapping for random interleaving."""
        rng = random.Random(seed)

        # Create pairs of (dataset_id, dataset_index)
        mapping = []

        # Add all BLIP3o indices
        for i in range(self.blip3o_len):
            mapping.append((0, i))  # 0 = BLIP3o

        # Add all DocLayNet indices
        for i in range(self.doclaynet_len):
            mapping.append((1, i))  # 1 = DocLayNet

        # Shuffle the mapping for random interleaving
        rng.shuffle(mapping)

        return mapping

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item from mixed dataset.

        Returns sample from either BLIP3o or DocLayNet based on index mapping.
        """
        dataset_id, dataset_idx = self.index_mapping[idx]

        if dataset_id == 0:
            # BLIP3o sample
            sample = self.blip3o_dataset[dataset_idx]
            sample['dataset_source'] = 'blip3o'
        else:
            # DocLayNet sample
            sample = self.doclaynet_dataset[dataset_idx]
            sample['dataset_source'] = 'doclaynet'

        return sample
