"""
Markovian Chunk Decoder Training (Single-Step Chunk-to-Chunk Prediction)

Train a decoder to predict the next visual chunk given current chunk.
Uses variable-length word-based chunking for training data.

Chunking Strategy:
- Target: ~500 words per chunk (with variance for robustness)
- Range: 10-900 words per chunk
- Each document is split into 2+ variable-length chunks
- Training pairs: consecutive (C_i, C_{i+1}) pairs

Training:
- Input: C_i (current chunk visual tokens)
- Target: C_{i+1} (next chunk visual tokens)
- Loss: MSE on single-step prediction

Features:
- Variable-length word-based chunking (not fixed token count)
- Direct encoder integration (22x faster than server)
- DDP support for multi-GPU training
- SwanLab logging for experiment tracking
- Mixed precision training

Usage:
    # Single GPU
    python examples/train_markovian.py \
        --dataset_type fineweb \
        --fineweb_subset 10BT \
        --batch_size 8 \
        --max_steps 50000

    # Multi-GPU with torchrun
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 OCRFlow/examples/train_markovian.py \
        --dataset_type fineweb \
        --fineweb_subset 10BT \
        --batch_size 8 \
        --max_steps 50000
"""

import argparse
import math
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# SwanLab for experiment tracking
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from OCRFlow.models.markovian_chunk_decoder import create_chunk_decoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Markovian chunk decoder")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="fineweb", choices=["fineweb", "openwebmath", "multi"])
    parser.add_argument("--fineweb_path", type=str, default="/share/project/xiyan/huggingface/HuggingFaceFW/fineweb-edu")
    parser.add_argument("--fineweb_subset", type=str, default="10BT", choices=["10BT", "100BT", "350BT"])
    parser.add_argument("--openwebmath_path", type=str, default="/share/project/xiyan/huggingface/open-web-math/open-web-math")
    parser.add_argument("--fineweb_weight", type=float, default=0.7)
    parser.add_argument("--openwebmath_weight", type=float, default=0.3)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    # Cached visual tokens (pre-computed for fast training)
    parser.add_argument("--use_cached_vistok", action="store_true", default=False,
                       help="Use pre-computed visual tokens from disk (faster training)")
    parser.add_argument("--vistok_cache_dir", type=str, default=None,
                       help="Directory with pre-computed visual tokens")
    parser.add_argument("--load_vistok_to_memory", action="store_true", default=False,
                       help="Load all visual tokens to RAM (fastest, but high memory)")
    parser.add_argument("--dataloader_workers", type=int, default=8,
                       help="Number of dataloader workers for cached dataset")

    # Chunking parameters (word-based)
    parser.add_argument("--target_words", type=int, default=500,
                       help="Target words per chunk")
    parser.add_argument("--min_words", type=int, default=10,
                       help="Minimum words per chunk")
    parser.add_argument("--max_words", type=int, default=900,
                       help="Maximum words per chunk")
    parser.add_argument("--chunk_variance", type=float, default=0.5,
                       help="Variance factor for chunk size")
    parser.add_argument("--min_doc_words", type=int, default=100,
                       help="Minimum words for a document to be used")
    parser.add_argument("--mixed_length", action="store_true", default=False,
                       help="Enable mixed-length chunking for short answer capability")

    # Encoder options
    parser.add_argument("--encoder_model", type=str, default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--num_render_workers", type=int, default=16,
                       help="Number of parallel text rendering workers")

    # Model
    parser.add_argument("--model_size", type=str, default="large", choices=["base", "large", "xl"])

    # Training
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Number of (input, target) chunk pairs per batch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # System
    parser.add_argument("--output_dir", type=str, default="./checkpoints/markovian_chunks")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)

    # SwanLab logging
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--swanlab_project", type=str, default="ocrflow")
    parser.add_argument("--swanlab_experiment", type=str, default=None)

    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine LR schedule"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_training(args, rank=0, world_size=1):
    """Setup training environment"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # TensorBoard (rank 0 only, optional)
    tb_writer = None
    if rank == 0 and TENSORBOARD_AVAILABLE:
        tb_writer = SummaryWriter(str(output_dir / "tensorboard" / timestamp))
    if rank == 0:
        with open(output_dir / "config.json", 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Saved config to {output_dir / 'config.json'}")

    # SwanLab initialization (rank 0 only)
    swanlab_run = None
    if args.use_swanlab and SWANLAB_AVAILABLE and rank == 0:
        # Load API key from .env file
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, _, value = line.strip().partition('=')
                        if key == "SWANLAB_API_KEY":
                            os.environ["SWANLAB_API_KEY"] = value

        experiment_name = args.swanlab_experiment or f"markovian-{args.fineweb_subset or 'full'}-{timestamp}"

        swanlab.init(
            project=args.swanlab_project,
            experiment_name=experiment_name,
            config={
                "model": "markovian_chunk_decoder",
                "dataset_type": args.dataset_type,
                "fineweb_subset": args.fineweb_subset,
                "model_size": args.model_size,
                "target_words": args.target_words,
                "min_words": args.min_words,
                "max_words": args.max_words,
                "chunk_variance": args.chunk_variance,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                "learning_rate": args.learning_rate,
                "max_steps": args.max_steps,
                "world_size": world_size,
            }
        )
        swanlab_run = swanlab
        logger.info(f"SwanLab initialized: {args.swanlab_project}/{experiment_name}")

    return output_dir, checkpoint_dir, tb_writer, swanlab_run


def create_dataloader(args, device):
    """Create dataloader with Markovian chunk dataset"""

    # Use cached visual tokens if specified (MUCH faster)
    if args.use_cached_vistok:
        if not args.vistok_cache_dir:
            raise ValueError("--vistok_cache_dir required when using --use_cached_vistok")

        from OCRFlow.training.cached_markovian_dataset import create_cached_dataloader
        logger.info(f"Using cached visual tokens from: {args.vistok_cache_dir}")

        dataloader = create_cached_dataloader(
            cache_dir=args.vistok_cache_dir,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloader_workers,
            max_samples=args.max_samples,
            load_to_memory=args.load_vistok_to_memory,
            pin_memory=True,
        )
        return dataloader

    # On-the-fly encoding (slower, but no pre-computation needed)
    from OCRFlow.training.markovian_dataset import create_markovian_dataloader

    dataloader = create_markovian_dataloader(
        dataset_type=args.dataset_type,
        model_path=args.encoder_model,
        fineweb_path=args.fineweb_path,
        fineweb_subset=args.fineweb_subset,
        openwebmath_path=args.openwebmath_path,
        fineweb_weight=args.fineweb_weight,
        openwebmath_weight=args.openwebmath_weight,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        target_words=args.target_words,
        min_words=args.min_words,
        max_words=args.max_words,
        variance=args.chunk_variance,
        min_doc_words=args.min_doc_words,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        device=str(device),
        num_render_workers=args.num_render_workers,
        mixed_length=args.mixed_length,
    )

    return dataloader


def train_step(model, input_chunks, target_chunks, optimizer, scaler, device, args, use_amp=False):
    """
    Single training step (single-step chunk prediction)

    Args:
        model: Markovian chunk decoder
        input_chunks: [batch, 111, 1280] input visual tokens
        target_chunks: [batch, 111, 1280] target visual tokens
        optimizer: Optimizer
        scaler: AMP scaler
        device: Device
        args: Training arguments
        use_amp: Whether to use automatic mixed precision

    Returns:
        loss: Scalar loss value
        metrics: Dict of metrics
    """
    input_chunks = input_chunks.to(device)
    target_chunks = target_chunks.to(device)

    # Stack into sequences of 2 for model input: [batch, 2, 111, 1280]
    chunk_sequences = torch.stack([input_chunks, target_chunks], dim=1)

    # Forward pass
    with autocast(enabled=use_amp):
        # Handle DDP wrapped model
        if hasattr(model, 'module'):
            loss, metrics = model.module.compute_loss(chunk_sequences)
        else:
            loss, metrics = model.compute_loss(chunk_sequences)
        loss = loss / args.gradient_accumulation_steps

    # Backward pass
    if use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item() * args.gradient_accumulation_steps, metrics


def train(model, dataloader, optimizer, scheduler, scaler, args, tb_writer, swanlab_run, device, rank=0, start_step=0):
    """Main training loop for single-step Markovian prediction"""
    model.train()
    global_step = start_step
    optimizer.zero_grad()

    running_loss = 0.0
    running_metrics = {}
    accum_count = 0

    max_steps = args.max_steps if args.max_steps else 100000
    if rank == 0:
        logger.info(f"Training from step {start_step} to {max_steps}")

    pbar = tqdm(total=max_steps - start_step, initial=0, desc="Single-Step Markovian Training", disable=(rank != 0))

    for epoch in range(args.num_epochs):
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, (input_chunks, target_chunks) in enumerate(dataloader):
            # Training step with (input, target) pairs
            loss, metrics = train_step(
                model, input_chunks, target_chunks,
                optimizer, scaler, device, args, use_amp=args.use_amp
            )

            running_loss += loss
            for k, v in metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v
            accum_count += 1

            # Update weights every accumulation_steps
            if accum_count % args.gradient_accumulation_steps == 0:
                if args.use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{current_lr:.2e}",
                })

                # Logging
                if global_step % args.log_every == 0 and rank == 0:
                    avg_loss = running_loss / args.log_every

                    # TensorBoard logging
                    if tb_writer:
                        tb_writer.add_scalar('train/loss', avg_loss, global_step)
                        tb_writer.add_scalar('train/lr', current_lr, global_step)
                        for k in running_metrics:
                            avg_metric = running_metrics[k] / args.log_every
                            tb_writer.add_scalar(f'train/{k}', avg_metric, global_step)

                    # SwanLab logging
                    if swanlab_run:
                        swanlab_metrics = {
                            'train/loss': avg_loss,
                            'train/lr': current_lr,
                        }
                        for k in running_metrics:
                            swanlab_metrics[f'train/{k}'] = running_metrics[k] / args.log_every
                        swanlab_run.log(swanlab_metrics, step=global_step)

                    logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                    running_loss = 0.0
                    running_metrics = {}

                # Save checkpoint
                if global_step % args.save_every == 0 and rank == 0:
                    checkpoint_path = args.checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if args.use_amp else None,
                        'args': vars(args),
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Check if reached max steps
                if global_step >= max_steps:
                    logger.info(f"Reached max_steps: {max_steps}")
                    pbar.close()
                    return global_step

    pbar.close()
    return global_step


def main():
    args = parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Set device based on distributed or single GPU mode
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Using device: {device}, World size: {world_size}")

    # Setup directories and logging
    output_dir, checkpoint_dir, tb_writer, swanlab_run = setup_training(args, rank, world_size)
    args.checkpoint_dir = checkpoint_dir

    # Log config
    effective_batch = args.batch_size * args.gradient_accumulation_steps * world_size
    if rank == 0:
        logger.info(f"Effective batch size: {effective_batch} chunk pairs")
        logger.info(f"Chunking: target={args.target_words} words, range=[{args.min_words}, {args.max_words}], variance={args.chunk_variance}")

    # Create dataloader
    if rank == 0:
        logger.info("Creating Markovian chunk dataloader with variable word-based chunking...")
    dataloader = create_dataloader(args, device)

    # Create model (Markovian chunk decoder)
    if rank == 0:
        logger.info(f"Creating Markovian Chunk Decoder ({args.model_size})...")
    model = create_chunk_decoder(model_size=args.model_size)
    model = model.to(device)

    # Wrap with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if rank == 0:
        base_model = model.module if hasattr(model, 'module') else model
        total_params = sum(p.numel() for p in base_model.parameters())
        logger.info(f"Model parameters: {total_params / 1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    num_training_steps = args.max_steps if args.max_steps else 100000
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    if rank == 0:
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        base_model = model.module if hasattr(model, 'module') else model
        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if args.use_amp and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint.get('step', 0)
        if rank == 0:
            logger.info(f"Resumed from step {start_step}")

    # Training
    if rank == 0:
        logger.info("Starting single-step Markovian training...")
        logger.info("Architecture: Chunk Encoder → Sequence Decoder → Chunk Decoder")
        logger.info("Training: C_i → C_(i+1) (single-step prediction)")
        logger.info("Loss: MSE on predicted next chunk")

    try:
        final_step = train(
            model, dataloader, optimizer, scheduler, scaler,
            args, tb_writer, swanlab_run, device, rank, start_step
        )
    except KeyboardInterrupt:
        if rank == 0:
            logger.info("Training interrupted")
        final_step = start_step
    finally:
        # Save final model (rank 0 only)
        if rank == 0:
            final_path = output_dir / "final_model.pt"
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(model_state, final_path)
            logger.info(f"Saved final model to {final_path}")

            if tb_writer:
                tb_writer.close()
            if swanlab_run:
                swanlab.finish()

        cleanup_distributed()

    if rank == 0:
        logger.info(f"Training complete! Final step: {final_step}")


if __name__ == "__main__":
    main()
