#!/usr/bin/env python3
"""
Pre-encoded BLIP3o Dataset - Loads pre-computed OCR features from disk.

This eliminates encoding bottleneck during training.
"""

import os
import h5py
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class PreEncodedBLIP3oDataset(Dataset):
    """
    Loads pre-encoded BLIP3o features from HDF5 files.

    Much faster than encoding on-the-fly:
    - No OCR encoding overhead (~300 img/s → instant load)
    - Features loaded directly from disk
    - Enables DataLoader with num_workers > 0
    """

    def __init__(
        self,
        encoded_dir: str = "/share/project/xiyan/huggingface/BLIP3o/BLIP3o-60k-encoded",
        tokenizer=None,
        max_caption_length: int = 512,
    ):
        self.encoded_dir = encoded_dir
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length

        # Load all shards
        shard_files = sorted([
            os.path.join(encoded_dir, f)
            for f in os.listdir(encoded_dir)
            if f.endswith('.h5')
        ])

        if not shard_files:
            raise ValueError(f"No .h5 files found in {encoded_dir}")

        # Open all shards (keep handles open for fast access)
        self.shards = []
        self.shard_offsets = [0]
        total_samples = 0

        for shard_file in shard_files:
            h5_file = h5py.File(shard_file, 'r')
            num_samples = h5_file.attrs.get('num_samples', len(h5_file['final_features']))
            self.shards.append(h5_file)
            total_samples += num_samples
            self.shard_offsets.append(total_samples)

        self.total_samples = total_samples

        logger.info(f"✓ Loaded pre-encoded dataset:")
        logger.info(f"  Shards: {len(self.shards)}")
        logger.info(f"  Total samples: {self.total_samples}")
        logger.info(f"  Storage: {encoded_dir}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which shard this index belongs to
        shard_idx = 0
        for i, offset in enumerate(self.shard_offsets[1:]):
            if idx < offset:
                shard_idx = i
                break

        local_idx = idx - self.shard_offsets[shard_idx]
        shard = self.shards[shard_idx]

        # Load pre-encoded features
        final_feat = torch.from_numpy(shard['final_features'][local_idx]).float()

        deepstack_feats = [
            torch.from_numpy(shard[f'deepstack_level_{i}'][local_idx]).float()
            for i in range(3)
        ]

        caption = shard['captions'][local_idx].decode('utf-8')

        # Tokenize caption
        if self.tokenizer:
            caption_ids = self.tokenizer(
                caption,
                max_length=self.max_caption_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).input_ids.squeeze(0)
        else:
            caption_ids = torch.zeros(self.max_caption_length, dtype=torch.long)

        # Create labels (for training)
        labels = caption_ids.clone()
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": caption_ids,
            "labels": labels,
            "ocr_image_features": (final_feat, deepstack_feats),
        }

    def __del__(self):
        # Close all HDF5 files
        for shard in self.shards:
            shard.close()
