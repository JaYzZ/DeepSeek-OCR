#!/usr/bin/env python3
"""
Pre-encode BLIP3o dataset with DPSK OCR encoder.

This eliminates the encoding bottleneck during training by pre-computing
all visual features offline.

Storage: ~17GB for 60k samples (284KB per sample)
Speed: ~4-5 hours to encode 60k images on 8 GPUs

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python OCRVL/scripts/precompute_blip3o_features.py
"""

import os
import sys
from pathlib import Path
import argparse
import torch
from tqdm import tqdm
import h5py
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from OCRVL.data.blip3o_dataset import BLIP3oAlignmentDataset
from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
from transformers import AutoTokenizer


def encode_and_save_shard(dataset, encoder, start_idx, end_idx, output_file, gpu_id):
    """Encode a shard of the dataset and save to HDF5."""

    # Prepare HDF5 file
    max_samples = end_idx - start_idx

    with h5py.File(output_file, 'w') as f:
        # Create datasets for features
        final_feats = f.create_dataset(
            'final_features',
            shape=(max_samples, 111, 1280),
            dtype='float16',
            compression='gzip',
            compression_opts=4
        )

        # Deepstack: 3 levels of 111x1024
        deepstack_feats = [
            f.create_dataset(
                f'deepstack_level_{i}',
                shape=(max_samples, 111, 1024),
                dtype='float16',
                compression='gzip',
                compression_opts=4
            )
            for i in range(3)
        ]

        # Captions (tokenized)
        captions = f.create_dataset(
            'captions',
            shape=(max_samples,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )

        # Process in batches
        batch_size = 8
        write_idx = 0

        for i in tqdm(range(start_idx, end_idx, batch_size), desc=f"GPU {gpu_id}"):
            batch_end = min(i + batch_size, end_idx)
            batch_images = []
            batch_captions = []

            for idx in range(i, batch_end):
                try:
                    sample = dataset[idx]
                    if 'image' in sample and sample['image'] is not None:
                        batch_images.append(sample['image'])
                        batch_captions.append(sample.get('caption', ''))
                except Exception as e:
                    print(f"Error loading sample {idx}: {e}")
                    continue

            if not batch_images:
                continue

            # Encode batch
            try:
                final, deepstack = encoder.encode_images_with_deepstack(batch_images)

                # Save to HDF5
                for j, (final_feat, ds_feats, caption) in enumerate(zip(final, deepstack, batch_captions)):
                    final_feats[write_idx] = final_feat.cpu().half().numpy()
                    for level_idx, ds_feat in enumerate(ds_feats):
                        deepstack_feats[level_idx][write_idx] = ds_feat.cpu().half().numpy()
                    captions[write_idx] = caption
                    write_idx += 1

            except Exception as e:
                print(f"Error encoding batch {i}: {e}")
                continue

        # Store metadata
        f.attrs['num_samples'] = write_idx
        f.attrs['dataset_type'] = '60k'

    print(f"GPU {gpu_id}: Saved {write_idx} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                       default="/share/project/xiyan/huggingface/BLIP3o/BLIP3o-60k-encoded")
    parser.add_argument("--dataset_type", type=str, default="60k")
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer (for dataset initialization)
    tokenizer = AutoTokenizer.from_pretrained(
        "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True
    )

    # Load dataset (full, no sampling)
    print("Loading BLIP3o dataset...")
    dataset = BLIP3oAlignmentDataset(
        dataset_type=args.dataset_type,
        sample_percentage=1.0,
        use_images=True,
        tokenizer=tokenizer,
        ocr_adapter=None  # We'll encode manually
    )

    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")

    # Divide work among GPUs
    samples_per_gpu = total_samples // args.num_gpus

    import multiprocessing as mp
    processes = []

    for gpu_id in range(args.num_gpus):
        start_idx = gpu_id * samples_per_gpu
        end_idx = (gpu_id + 1) * samples_per_gpu if gpu_id < args.num_gpus - 1 else total_samples
        output_file = os.path.join(args.output_dir, f"shard_{gpu_id:02d}.h5")

        # Create encoder for this GPU
        def worker(gpu_id, start_idx, end_idx, output_file):
            torch.cuda.set_device(gpu_id)
            encoder = DPSKOCREncoder(
                model_path="deepseek-ai/DeepSeek-OCR",
                device=f"cuda:{gpu_id}",
                dtype=torch.bfloat16
            )
            encode_and_save_shard(dataset, encoder, start_idx, end_idx, output_file, gpu_id)

        p = mp.Process(target=worker, args=(gpu_id, start_idx, end_idx, output_file))
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()

    print(f"\n✓ Pre-encoding complete! Features saved to {args.output_dir}")
    print(f"  Total samples: {total_samples}")
    print(f"  Storage: ~{total_samples * 284 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
