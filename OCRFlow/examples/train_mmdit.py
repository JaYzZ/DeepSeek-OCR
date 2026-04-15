"""
MMDiT Flow Matching Training (Visual Token Generation)

Train MMDiT from scratch using rectified flow matching to generate visual tokens.
Uses direct encoder integration for 22x faster training.

Architecture:
- MMDiT (Multi-Modal Diffusion Transformer) with joint attention
- Rectified flow matching: learns velocity v to transform noise → visual tokens
- Text conditioning from T5-like encoder (text_seq) + pooled embedding

Training:
- Sample x_0 (visual tokens from encoder), x_1 (noise)
- Interpolate: x_t = (1-t)*x_0 + t*x_1
- Target: v = x_1 - x_0 (velocity)
- Loss: MSE(v_pred, v)

Features:
- Direct encoder integration (22x faster than server)
- DDP support for multi-GPU training
- SwanLab logging for experiment tracking
- Mixed precision training (bf16)
- Classifier-free guidance dropout

Usage:
    # Single GPU
    python examples/train_mmdit.py \
        --dataset_type fineweb \
        --fineweb_subset 10BT \
        --batch_size 8 \
        --max_steps 50000

    # Multi-GPU with torchrun
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 OCRFlow/examples/train_mmdit.py \
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
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import swanlab

SWANLAB_AVAILABLE = True

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from OCRFlow.models.mmdit_scratch import create_mmdit_scratch
from project_paths import hf_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MMDiT flow matching model")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="fineweb", choices=["fineweb", "openwebmath", "multi"])
    parser.add_argument("--fineweb_path", type=str, default=str(hf_path("HuggingFaceFW", "fineweb-edu")))
    parser.add_argument("--fineweb_subset", type=str, default="10BT", choices=["10BT", "100BT", "350BT"])
    parser.add_argument("--openwebmath_path", type=str, default=str(hf_path("open-web-math", "open-web-math")))
    parser.add_argument("--fineweb_weight", type=float, default=0.7)
    parser.add_argument("--openwebmath_weight", type=float, default=0.3)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--min_tokens", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--max_samples", type=int, default=None)

    # Encoder options
    parser.add_argument("--encoder_model", type=str, default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--encode_batch_size", type=int, default=8)

    # Model
    parser.add_argument("--model_size", type=str, default="large", choices=["small", "medium", "large", "xl"])

    # Text encoder (dummy for now, will use real T5 later)
    parser.add_argument("--text_encoder", type=str, default="dummy", choices=["dummy", "t5"])
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--text_dim", type=int, default=768)

    # Flow matching
    parser.add_argument("--cfg_dropout", type=float, default=0.1, help="CFG dropout rate")

    # Training (GPT-style)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # System
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mmdit_flow")
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

    # TensorBoard (rank 0 only)
    tb_writer = None
    if rank == 0:
        tb_writer = SummaryWriter(str(output_dir / "tensorboard" / timestamp))
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

        experiment_name = args.swanlab_experiment or f"mmdit-{args.model_size}-{args.fineweb_subset or 'full'}-{timestamp}"

        swanlab.init(
            project=args.swanlab_project,
            experiment_name=experiment_name,
            config={
                "model": "mmdit_scratch",
                "model_size": args.model_size,
                "dataset_type": args.dataset_type,
                "fineweb_subset": args.fineweb_subset,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                "learning_rate": args.learning_rate,
                "max_steps": args.max_steps,
                "cfg_dropout": args.cfg_dropout,
                "world_size": world_size,
            }
        )
        swanlab_run = swanlab
        logger.info(f"SwanLab initialized: {args.swanlab_project}/{experiment_name}")

    return output_dir, checkpoint_dir, tb_writer, swanlab_run


def create_dataloader(args, device):
    """Create dataloader with direct encoder"""
    from OCRFlow.training.direct_encoder_dataset import create_direct_dataloader

    dataloader = create_direct_dataloader(
        dataset_type=args.dataset_type,
        model_path=args.encoder_model,
        fineweb_path=args.fineweb_path,
        fineweb_subset=args.fineweb_subset,
        openwebmath_path=args.openwebmath_path,
        fineweb_weight=args.fineweb_weight,
        openwebmath_weight=args.openwebmath_weight,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        device=str(device),
    )

    return dataloader


def compute_flow_matching_loss(model, x_0, text_seq, text_pooled, cfg_dropout=0.1, use_amp=False):
    """
    Compute rectified flow matching loss.

    Args:
        model: MMDiT model
        x_0: Clean visual tokens [B, N, D] (111 or 100 tokens)
        text_seq: Text sequence embeddings [B, L, D]
        text_pooled: Pooled text embeddings [B, D]
        cfg_dropout: Probability of dropping text conditioning
        use_amp: Whether to use automatic mixed precision

    Returns:
        loss: Scalar loss
        metrics: Dict of metrics
    """
    B = x_0.shape[0]
    device = x_0.device
    dtype = x_0.dtype

    # Sample timesteps uniformly in [0, 1]
    t = torch.rand(B, device=device, dtype=dtype)

    # Sample noise (x_1)
    x_1 = torch.randn_like(x_0)

    # Interpolate: x_t = (1-t)*x_0 + t*x_1
    t_expand = t.view(B, 1, 1)
    x_t = (1 - t_expand) * x_0 + t_expand * x_1

    # Target velocity: v = x_1 - x_0
    v_target = x_1 - x_0

    # CFG dropout: randomly drop conditioning
    cfg_mask = (torch.rand(B, device=device) > cfg_dropout).float()

    # Forward pass
    with autocast(enabled=use_amp):
        # Handle DDP wrapped model
        if hasattr(model, 'module'):
            v_pred = model.module(x_t, t, text_seq, text_pooled, cfg_mask=cfg_mask)
        else:
            v_pred = model(x_t, t, text_seq, text_pooled, cfg_mask=cfg_mask)

        # MSE loss on velocity
        loss = F.mse_loss(v_pred, v_target, reduction='mean')

    metrics = {
        'loss': loss.item(),
        'v_pred_norm': v_pred.norm(dim=-1).mean().item(),
        'v_target_norm': v_target.norm(dim=-1).mean().item(),
        't_mean': t.mean().item(),
    }

    return loss, metrics


def create_dummy_text_embeddings(batch_size, max_len, text_dim, device, dtype):
    """Create dummy text embeddings (placeholder until T5 integration)"""
    # Random embeddings that will be learned to match visual tokens
    text_seq = torch.randn(batch_size, max_len, text_dim, device=device, dtype=dtype) * 0.1
    text_pooled = torch.randn(batch_size, text_dim, device=device, dtype=dtype) * 0.1
    return text_seq, text_pooled


def train_step(model, batch, args, device, scaler, use_amp=False):
    """Single training step"""
    if isinstance(batch, tuple):
        vistok, _, _ = batch
    else:
        vistok = batch

    vistok = vistok.to(device)
    B = vistok.shape[0]

    # Use only visual tokens (100) for training, not structural (11)
    if vistok.shape[1] == 111:
        x_0 = vistok[:, :100, :]  # Only visual tokens
    else:
        x_0 = vistok

    # Create dummy text embeddings (will be replaced with real T5 later)
    text_seq, text_pooled = create_dummy_text_embeddings(
        B, args.max_text_len, args.text_dim, device, vistok.dtype
    )

    # Compute flow matching loss
    loss, metrics = compute_flow_matching_loss(
        model, x_0, text_seq, text_pooled,
        cfg_dropout=args.cfg_dropout,
        use_amp=use_amp
    )

    loss = loss / args.gradient_accumulation_steps

    # Backward pass
    if use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item() * args.gradient_accumulation_steps, metrics


def train(model, dataloader, optimizer, scheduler, scaler, args, tb_writer, swanlab_run, device, rank=0, start_step=0):
    """Main training loop"""
    model.train()
    global_step = start_step
    optimizer.zero_grad()

    running_loss = 0.0
    running_metrics = {}
    accum_count = 0

    max_steps = args.max_steps if args.max_steps else 100000
    if rank == 0:
        logger.info(f"Training from step {start_step} to {max_steps}")

    pbar = tqdm(total=max_steps - start_step, initial=0, desc="MMDiT Flow Training", disable=(rank != 0))

    for epoch in range(args.num_epochs):
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            # Training step
            loss, metrics = train_step(model, batch, args, device, scaler, use_amp=args.use_amp)

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
        logger.info(f"Effective batch size: {effective_batch}")
        logger.info(f"Dataset: {args.dataset_type}, Subset: {args.fineweb_subset}")
        logger.info(f"CFG dropout: {args.cfg_dropout}")

    # Create dataloader
    if rank == 0:
        logger.info("Creating dataloader with direct encoder...")
    dataloader = create_dataloader(args, device)

    # Create model
    if rank == 0:
        logger.info(f"Creating MMDiT from scratch ({args.model_size})...")
    model = create_mmdit_scratch(model_size=args.model_size)
    model = model.to(device).to(torch.bfloat16)

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
        logger.info("Starting MMDiT flow matching training...")
        logger.info(f"Architecture: MMDiT with joint attention + AdaLN")
        logger.info(f"Loss: Rectified flow matching (velocity prediction)")

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
