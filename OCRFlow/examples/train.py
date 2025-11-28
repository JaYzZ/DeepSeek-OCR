#!/usr/bin/env python3
"""
Example Training Script for OCRFlow

This script demonstrates a complete training pipeline for OCRFlow.
"""

import torch
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import os

# Add OCRFlow to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from OCRFlow.models import create_mmdit_ocrflow, create_image_token_encoder
from OCRFlow.models.text_encoders import load_text_encoders, encode_text_dual
from OCRFlow.training.loss import RectifiedFlowLoss, compute_flow_matching_loss
from OCRFlow.training.dataset import create_dataloaders
from OCRFlow.utils.helpers import (
    set_seed, get_device, save_checkpoint, load_checkpoint,
    AverageMeter, count_parameters, format_time
)
import time


def main():
    parser = argparse.ArgumentParser(description="OCRFlow Training")
    parser.add_argument("--data-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-data-path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--output-dir", type=str, default="./outputs/ocrflow", help="Output directory")
    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "base", "large", "xl"])
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--cfg-dropout", type=float, default=0.1, help="CFG dropout probability")
    parser.add_argument("--loss-type", type=str, default="huber", choices=["mse", "l1", "huber"])
    parser.add_argument("--image-size", type=int, default=640, help="Image size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    dtype = torch.bfloat16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OCRFlow Training")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Model size: {args.model_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Load models
    print("\n[1/4] Loading models...")

    # MMDiT model (trainable)
    model = create_mmdit_ocrflow(args.model_size)
    model = model.to(device).to(dtype).train()
    print(f"  MMDiT parameters: {count_parameters(model)/1e6:.1f}M")

    # Image token encoder (frozen)
    image_encoder = create_image_token_encoder(
        token_dim=1280,
        image_size=args.image_size,
        freeze=True,
        device=device,
        dtype=dtype
    )
    print(f"  Image encoder loaded (frozen)")

    # Text encoders (frozen)
    text_encoders = load_text_encoders(
        t5_model_path="google/t5-v1_1-base",
        clip_model_path="openai/clip-vit-large-patch14",
        device=device,
        dtype=dtype,
        freeze=True
    )
    print(f"  Text encoders loaded (frozen)")

    # Setup training
    print("\n[2/4] Setting up training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    loss_fn = RectifiedFlowLoss(loss_type=args.loss_type, huber_delta=1.0)

    # Load data
    print("\n[3/4] Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_data_path=args.data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        metadata = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch = metadata['epoch']
        global_step = metadata['global_step']

    # Training loop
    print("\n[4/4] Starting training...")
    print("=" * 80)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        loss_meter = AverageMeter("loss")
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, (images, captions) in enumerate(pbar):
            images = images.to(device).to(dtype)

            # Encode images to tokens (target x_1)
            with torch.no_grad():
                x_1 = image_encoder(images)

            # Encode text
            with torch.no_grad():
                t5_emb, clip_emb, _ = encode_text_dual(
                    captions,
                    t5_tokenizer=text_encoders['t5_tokenizer'],
                    t5_model=text_encoders['t5_model'],
                    clip_tokenizer=text_encoders['clip_tokenizer'],
                    clip_model=text_encoders['clip_model'],
                    device=device,
                    dtype=dtype
                )

            # Sample noise (x_0)
            x_0 = torch.randn_like(x_1)

            # Compute loss
            loss = compute_flow_matching_loss(
                model=model,
                x_0=x_0,
                x_1=x_1,
                text_seq_embeds=t5_emb,
                text_pooled_embeds=clip_emb,
                loss_fn=loss_fn,
                cfg_dropout_prob=args.cfg_dropout
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Update metrics
            loss_meter.update(loss.item())
            global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})

            # Logging
            if global_step % args.log_every == 0:
                tqdm.write(f"Step {global_step}: {loss_meter}")

            # Save checkpoint
            if global_step % args.save_every == 0:
                ckpt_path = output_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    loss_meter.avg, str(ckpt_path)
                )

        # End of epoch
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} complete!")
        print(f"  Time: {format_time(epoch_time)}")
        print(f"  Avg loss: {loss_meter.avg:.4f}")

        # Save epoch checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(
            model, optimizer, epoch+1, global_step,
            loss_meter.avg, str(ckpt_path)
        )

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
