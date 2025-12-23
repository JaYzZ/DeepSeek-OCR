#!/usr/bin/env python3
"""
BLIP3o Pipeline Training - Dedicated Encoder/Trainer Architecture

Optimized for 8-GPU systems with pipeline parallelism:
- 6 GPUs (1-6): Dedicated OCR encoding (6 × 300 = 1,800 img/s)
- 2 GPUs (0,7): Dedicated training with DDP (consumes encoded features)

Architecture:
    GPUs 1-6: DPSK OCR encoders → shared memory cache
    GPUs 0,7: Qwen3-VL connector training ← shared memory cache

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python OCRVL/examples/train_pipeline.py
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List
import multiprocessing as mp

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from OCRVL.model.language_model.ocr_qwen3_vl import (
    OCRQwen3VLForConditionalGeneration,
    Qwen3VLOCRTextAdapter
)
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(processName)s] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SharedMemoryCache:
    """Zero-copy shared memory cache for encoded features."""

    def __init__(self, max_size=10000, token_seq_len=111, token_dim=1280):
        self.max_size = max_size

        # Shared tensors for features
        self.final_buffer = torch.zeros(
            max_size, token_seq_len, token_dim,
            dtype=torch.float32
        ).share_memory_()

        # Shared tensors for deepstack (3 levels, 1024-dim)
        self.deepstack_buffers = [
            torch.zeros(max_size, token_seq_len, 1024, dtype=torch.float32).share_memory_()
            for _ in range(3)
        ]

        # Shared tensors for captions (tokenized, max 512 tokens)
        self.caption_ids = torch.zeros(max_size, 512, dtype=torch.long).share_memory_()
        self.caption_lengths = torch.zeros(max_size, dtype=torch.long).share_memory_()

        # Circular buffer indices
        self.write_idx = mp.Value('i', 0)
        self.read_idx = mp.Value('i', 0)
        self.size = mp.Value('i', 0)

    def write_batch(self, final_feats, deepstack_feats, caption_ids_batch, lengths):
        """Write a batch of encoded features."""
        batch_size = len(final_feats)

        with self.write_idx.get_lock():
            start_idx = self.write_idx.value
            end_idx = (start_idx + batch_size) % self.max_size

            if end_idx > start_idx:
                self.final_buffer[start_idx:end_idx] = torch.stack(final_feats).cpu()
                for i, ds_buf in enumerate(self.deepstack_buffers):
                    ds_buf[start_idx:end_idx] = torch.stack([d[i] for d in deepstack_feats]).cpu()
                self.caption_ids[start_idx:end_idx, :lengths.max()] = caption_ids_batch
                self.caption_lengths[start_idx:end_idx] = lengths
            else:
                # Wrap around
                first_part = self.max_size - start_idx
                self.final_buffer[start_idx:] = torch.stack(final_feats[:first_part]).cpu()
                self.final_buffer[:end_idx] = torch.stack(final_feats[first_part:]).cpu()
                # Similar for deepstack and captions...

            self.write_idx.value = end_idx
            with self.size.get_lock():
                self.size.value = min(self.size.value + batch_size, self.max_size)

    def read_batch(self, batch_size, device):
        """Read a batch for training."""
        with self.read_idx.get_lock():
            if self.size.value < batch_size:
                return None

            start_idx = self.read_idx.value
            end_idx = (start_idx + batch_size) % self.max_size

            if end_idx > start_idx:
                final = self.final_buffer[start_idx:end_idx].to(device, non_blocking=True)
                deepstack = [buf[start_idx:end_idx].to(device, non_blocking=True)
                           for buf in self.deepstack_buffers]
                caption_ids = self.caption_ids[start_idx:end_idx].to(device)
                lengths = self.caption_lengths[start_idx:end_idx]
            else:
                # Handle wrap-around
                first_part = self.max_size - start_idx
                final = torch.cat([
                    self.final_buffer[start_idx:],
                    self.final_buffer[:end_idx]
                ]).to(device, non_blocking=True)
                # Similar for others...
                deepstack = None  # Simplified
                caption_ids = None
                lengths = None

            self.read_idx.value = end_idx
            with self.size.get_lock():
                self.size.value -= batch_size

            return final, deepstack, caption_ids, lengths


def encoder_worker(rank, dataset_path, cache, num_workers=6):
    """Encoder worker process."""
    gpu_id = rank + 1  # GPUs 1-6
    torch.cuda.set_device(gpu_id)

    logger.info(f"Encoder {rank} starting on GPU {gpu_id}")

    # Load OCR encoder
    from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
    encoder = DPSKOCREncoder(
        model_path="deepseek-ai/DeepSeek-OCR",
        device=f"cuda:{gpu_id}",
        dtype=torch.bfloat16
    )

    # Load BLIP3o dataset subset (shard for this worker)
    from OCRVL.data.blip3o_dataset import BLIP3oAlignmentDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True
    )

    # Each encoder gets a shard
    dataset = BLIP3oAlignmentDataset(
        dataset_type="60k",
        sample_percentage=1.0,
        use_images=True,
        tokenizer=tokenizer,
        ocr_adapter=None,  # We'll encode manually
        seed=42 + rank
    )

    # Encode loop
    batch_size = 8
    idx = rank

    while True:
        batch_images = []
        batch_captions = []

        for _ in range(batch_size):
            if idx >= len(dataset):
                idx = rank  # Reset to shard start

            sample = dataset[idx]
            if 'image' in sample and sample['image'] is not None:
                batch_images.append(sample['image'])
                batch_captions.append(sample.get('caption', ''))

            idx += num_workers

        if not batch_images:
            time.sleep(0.1)
            continue

        # Encode batch
        final_feats, deepstack_feats = encoder.encode_images_with_deepstack(batch_images)

        # Tokenize captions
        caption_tokens = tokenizer(
            batch_captions,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )

        # Write to cache
        cache.write_batch(
            final_feats,
            deepstack_feats,
            caption_tokens['input_ids'],
            caption_tokens['attention_mask'].sum(dim=1)
        )

        if idx % 1000 == 0:
            logger.info(f"Encoder {rank} processed {idx} samples, cache size: {cache.size.value}")


def trainer_worker(rank, cache, world_size=2):
    """Trainer worker process (DDP across GPUs 0,7)."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend='nccl')

    gpu_id = 0 if rank == 0 else 7
    torch.cuda.set_device(gpu_id)

    logger.info(f"Trainer {rank} starting on GPU {gpu_id}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True
    )

    model = OCRQwen3VLForConditionalGeneration.from_pretrained(
        "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}",
        trust_remote_code=True
    )

    # Freeze and create connectors
    for param in model.parameters():
        param.requires_grad = False

    target_dim = model.config.text_config.hidden_size
    model.model.ocr_connector = model.model._init_ocr_connector(
        1280, device=f"cuda:{gpu_id}", dtype=torch.bfloat16
    )
    model.model._ocr_deepstack_connectors = {
        1024: model.model._init_ocr_connector(
            1024, device=f"cuda:{gpu_id}", dtype=torch.bfloat16
        )
    }

    for param in model.model.ocr_connector.parameters():
        param.requires_grad = True
    for param in model.model._ocr_deepstack_connectors[1024].parameters():
        param.requires_grad = True

    # Wrap with DDP
    model = DDP(model, device_ids=[gpu_id])

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    # Training loop
    batch_size = 64
    global_step = 0

    while global_step < 10000:
        batch = cache.read_batch(batch_size, f"cuda:{gpu_id}")

        if batch is None:
            time.sleep(0.1)
            continue

        # TODO: Construct batch and train
        # For now, just log
        global_step += 1

        if global_step % 50 == 0 and rank == 0:
            logger.info(f"Step {global_step}, cache size: {cache.size.value}")

    dist.destroy_process_group()


def main():
    mp.set_start_method('spawn', force=True)

    # Create shared cache
    cache = SharedMemoryCache(max_size=10000)

    # Start encoder workers (GPUs 1-6)
    num_encoders = 6
    encoder_processes = []
    for rank in range(num_encoders):
        p = mp.Process(
            target=encoder_worker,
            args=(rank, "/share/project/xiyan/huggingface/BLIP3o/BLIP3o-60k", cache, num_encoders)
        )
        p.start()
        encoder_processes.append(p)

    # Wait for cache to fill
    time.sleep(30)

    # Start trainer workers (GPUs 0,7)
    trainer_processes = []
    for rank in range(2):
        p = mp.Process(target=trainer_worker, args=(rank, cache))
        p.start()
        trainer_processes.append(p)

    # Wait for training to complete
    for p in trainer_processes:
        p.join()

    # Stop encoders
    for p in encoder_processes:
        p.terminate()


if __name__ == "__main__":
    main()
