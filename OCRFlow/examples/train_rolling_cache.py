#!/usr/bin/env python3
"""
Distributed Rolling Cache Training

Scalable multi-GPU training where each GPU runs BOTH encoder and training:
- EncoderWorker: Continuously fills local rolling cache
- Training: Samples from local cache, DDP syncs gradients across all GPUs

Architecture (per GPU):
┌─────────────────────────────────────────────────────────────────┐
│                       GPU N (any rank)                          │
│  EncoderWorker (enc_bs=64)  ──▶ LocalCache ──▶ Training (bs=512)│
│         ~265 pairs/s                              ~1900/s       │
└─────────────────────────────────────────────────────────────────┘
                              │ DDP gradient sync (all GPUs)

Scaling: N GPUs = Nx encoder throughput, Nx training throughput
Effective batch size = train_batch_size × num_gpus

Usage:
    # Single GPU
    python OCRFlow/examples/train_distributed.py --gpu-ids 0 --epochs 3

    # 2 GPUs
    python OCRFlow/examples/train_distributed.py --gpu-ids 0,1 --epochs 5

    # 4 GPUs
    python OCRFlow/examples/train_distributed.py --gpu-ids 0,1,2,3 --epochs 10

    # 8 GPUs (full node)
    python OCRFlow/examples/train_distributed.py --gpu-ids 0,1,2,3,4,5,6,7 --epochs 10
"""

import os
import sys
import subprocess

# Parse --gpu-ids early, before torch import
def parse_gpu_ids():
    """Parse --gpu-ids before torch import to set CUDA_VISIBLE_DEVICES"""
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu-ids' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith('--gpu-ids='):
            return arg.split('=')[1]
    return None

# Check if we're being launched by torchrun (has LOCAL_RANK env var - more reliable than RANK)
_is_torchrun_worker = 'LOCAL_RANK' in os.environ

# If not a torchrun worker and --gpu-ids specified, set up environment
if not _is_torchrun_worker:
    gpu_ids = parse_gpu_ids()
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from project_paths import hf_path

from OCRFlow.models.markovian_chunk_decoder import create_chunk_decoder
from OCRFlow.training.rolling_cache_dataset import (
    RollingTokenCache,
    EncoderWorker,
    chunk_text_variable,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def estimate_dataset_size(data_path: Path) -> int:
    """Estimate dataset size from parquet files metadata (fast, no data loading)"""
    import pyarrow.parquet as pq

    parquet_files = list(data_path.glob("**/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_path}")

    total_rows = 0
    for pf in parquet_files:
        metadata = pq.read_metadata(pf)
        total_rows += metadata.num_rows

    return total_rows


def train(args):
    """Main training function"""
    rank, local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}"

    is_main = rank == 0

    # Resolve dataset path first (needed for auto-estimation)
    if args.fineweb_subset:
        data_path = Path(args.fineweb_path) / "sample" / args.fineweb_subset
    else:
        data_path = Path(args.fineweb_path) / "data"

    # Calculate max_steps from epochs if provided
    if args.epochs is not None:
        # Auto-estimate dataset size if not provided
        if args.dataset_size is None:
            if is_main:
                logger.info(f"Estimating dataset size from {data_path}...")
            dataset_size = estimate_dataset_size(data_path)
            if is_main:
                logger.info(f"Found {dataset_size:,} samples in dataset")
        else:
            dataset_size = args.dataset_size

        effective_batch_size = args.train_batch_size * world_size
        steps_per_epoch = dataset_size // effective_batch_size
        max_steps = args.epochs * steps_per_epoch
        if is_main:
            logger.info(f"Epoch mode: {args.epochs} epochs")
            logger.info(f"Dataset size: {dataset_size:,}")
            logger.info(f"Steps per epoch: {steps_per_epoch:,}")
            logger.info(f"Total steps: {max_steps:,}")
    elif args.max_steps is not None:
        max_steps = args.max_steps
        steps_per_epoch = None
    else:
        max_steps = 50000  # Default
        steps_per_epoch = None
        if is_main:
            logger.info(f"Using default max_steps: {max_steps}")

    if is_main:
        logger.info(f"Starting distributed training")
        logger.info(f"World size: {world_size} GPUs")
        logger.info(f"Effective batch size: {args.train_batch_size * world_size}")
        logger.info(f"Encoder batch size per GPU: {args.encode_batch_size}")
        logger.info(f"Training batch size per GPU: {args.train_batch_size}")
        logger.info(f"Cache size per GPU: {args.cache_size}")

    data_sources = {"fineweb": {"path": str(data_path), "weight": 1.0}}

    # Create LOCAL rolling cache on this GPU
    cache = RollingTokenCache(
        cache_size=args.cache_size,
        device=device,
    )
    logger.info(f"[Rank {rank}] Created local cache on {device}")

    # Create encoder worker for this GPU
    encoder = EncoderWorker(
        cache=cache,
        data_sources=data_sources,
        encoder_device=device,  # Same GPU as training
        encode_batch_size=args.encode_batch_size,
        num_render_workers=args.num_render_workers,
        min_words=50,
        max_words=900,
    )

    # Create model
    model = create_chunk_decoder(model_size=args.model_size).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler('cuda')

    # Learning rate scheduler
    def lr_lambda(step):
        warmup_steps = args.warmup_steps
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Start encoder
    encoder.start()
    logger.info(f"[Rank {rank}] Encoder started on {device}")

    # Wait for cache to fill
    min_fill = int(args.cache_size * args.min_cache_fill)
    logger.info(f"[Rank {rank}] Waiting for cache to fill to {min_fill} pairs...")
    while cache.valid_count < min_fill:
        time.sleep(0.5)
    logger.info(f"[Rank {rank}] Cache ready: {cache.valid_count} pairs")

    # Training loop
    model.train()
    step = 0
    total_loss = 0.0
    log_interval = 50
    save_interval = args.save_interval
    start_time = time.time()
    step_times = []

    if is_main:
        logger.info("Starting training loop...")

    try:
        while step < max_steps:
            step_start = time.time()

            # Sample from local cache
            batch = cache.sample_batch(args.train_batch_size)
            if batch is None:
                time.sleep(0.01)
                continue

            inputs, targets = batch

            # Stack into sequence [batch, 2, 111, 1280]
            chunk_sequences = torch.stack([inputs, targets], dim=1)

            # Forward pass
            optimizer.zero_grad()
            with autocast('cuda'):
                if isinstance(model, DDP):
                    loss, metrics = model.module.compute_loss(chunk_sequences)
                else:
                    loss, metrics = model.compute_loss(chunk_sequences)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            total_loss += loss.item()
            step_time = time.time() - step_start
            step_times.append(step_time)

            # Logging
            if is_main and step % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                avg_step_time = sum(step_times[-log_interval:]) / len(step_times[-log_interval:])
                pairs_per_sec = args.train_batch_size * world_size / avg_step_time
                cache_stats = cache.get_stats()

                # Build progress string
                progress = f"Step {step}/{max_steps}"
                if steps_per_epoch:
                    epoch = step // steps_per_epoch
                    step_in_epoch = step % steps_per_epoch
                    progress = f"Epoch {epoch+1}/{args.epochs} Step {step_in_epoch}/{steps_per_epoch} (total: {step})"

                logger.info(
                    f"{progress} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Train: {pairs_per_sec:.0f} pairs/s | "
                    f"Cache refresh: {cache_stats['refresh_rate']:.1f}x | "
                    f"Time: {elapsed/60:.1f}min"
                )
                total_loss = 0.0

            # Save checkpoint
            if is_main and step % save_interval == 0:
                save_path = Path(args.output_dir) / f"checkpoint_{step}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                state = {
                    'step': step,
                    'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(state, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        encoder.stop()
        cleanup_distributed()

    if is_main:
        logger.info("Training complete!")
        total_time = time.time() - start_time
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Total steps: {step}")


def main():
    parser = argparse.ArgumentParser(description="Distributed Rolling Cache Training")

    # GPU selection
    parser.add_argument("--gpu-ids", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g., '0,1' for 2 GPUs)")

    # Dataset
    parser.add_argument("--fineweb_path", type=str,
                        default=str(hf_path("HuggingFaceFW", "fineweb-edu")))
    parser.add_argument("--fineweb_subset", type=str, default="10BT")

    # Batch sizes (optimized for 80GB H100 based on benchmarks)
    # Encoder optimal at bs=24 (~329 img/s), larger batches see diminishing returns
    parser.add_argument("--encode_batch_size", type=int, default=24,
                        help="Encoder batch size per GPU (24 optimal @ 329 img/s)")
    parser.add_argument("--train_batch_size", type=int, default=512,
                        help="Training batch size per GPU (maximized for VRAM)")
    parser.add_argument("--num_render_workers", type=int, default=24,
                        help="Number of CPU workers for text rendering per GPU")

    # Cache
    parser.add_argument("--cache_size", type=int, default=10000,
                        help="Rolling cache size per GPU (pairs)")
    parser.add_argument("--min_cache_fill", type=float, default=0.1,
                        help="Minimum cache fill before training starts")

    # Model
    parser.add_argument("--model_size", type=str, default="large")

    # Training duration (use either max_steps OR epochs, not both)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps (takes priority over epochs)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (auto-estimates dataset size from parquet)")
    parser.add_argument("--dataset_size", type=int, default=None,
                        help="Override dataset size (optional, auto-estimated if not provided)")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=5000)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/distributed_training")

    args = parser.parse_args()

    # Count GPUs
    gpu_list = [g.strip() for g in args.gpu_ids.split(',')]
    num_gpus = len(gpu_list)

    # If multiple GPUs and not already in torchrun, relaunch with torchrun
    if num_gpus > 1 and not _is_torchrun_worker:
        # Build command for torchrun
        script_path = os.path.abspath(__file__)

        # Remove --gpu-ids and its value from args for the relaunched command
        new_argv = []
        skip_next = False
        for arg in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if arg == '--gpu-ids':
                skip_next = True  # Skip the next arg (the value)
                continue
            if arg.startswith('--gpu-ids='):
                continue
            new_argv.append(arg)

        cmd = [
            sys.executable, '-m', 'torch.distributed.run',
            '--nproc_per_node', str(num_gpus),
            script_path,
        ] + new_argv

        # Set environment and run
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

        print(f"Launching distributed training on {num_gpus} GPUs: {args.gpu_ids}")
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    else:
        # Single GPU or already in torchrun worker
        train(args)


if __name__ == "__main__":
    main()
