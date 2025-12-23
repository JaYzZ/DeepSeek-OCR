#!/usr/bin/env python3
"""
OCRVL Training Script - Multi-Stage Training for DPSK-Qwen Alignment

Three-stage training pipeline inspired by LLaVA and InstructBLIP:

Stage 1: Connector Alignment (Feature Alignment)
    - Freeze: DPSK encoder + Qwen LLM
    - Train: Connectors only (4.7M params)
    - Objective: Align DPSK features to Qwen's embedding space
    - Data: Image-caption pairs or rendered text
    - Duration: ~1-2 hours, 5K-10K steps

Stage 2: Visual Instruction Tuning (VIT)
    - Freeze: DPSK encoder
    - Train: Connectors + Qwen LLM
    - Objective: Instruction following with visual inputs
    - Data: Instruction-following dataset (OCR, VQA, captioning)
    - Duration: ~4-8 hours, 20K-50K steps

Stage 3: Reinforcement Learning (RL) - Optional
    - Freeze: DPSK encoder
    - Train: Connectors + Qwen LLM
    - Objective: Policy optimization for specific tasks
    - Data: Reward-based feedback
    - Duration: Task-dependent

Usage:
    # Stage 1: Connector alignment
    python OCRVL/train.py \\
        --stage alignment \\
        --max_steps 5000 \\
        --batch_size 8 \\
        --lr 1e-3 \\
        --output_dir checkpoints/stage1_alignment

    # Stage 2: Visual instruction tuning
    python OCRVL/train.py \\
        --stage vit \\
        --max_steps 20000 \\
        --batch_size 4 \\
        --lr 2e-5 \\
        --load_connectors checkpoints/stage1_alignment/connectors_final.pt \\
        --output_dir checkpoints/stage2_vit

    # Stage 3: RL finetuning (optional)
    python OCRVL/train.py \\
        --stage rl \\
        --max_steps 10000 \\
        --batch_size 2 \\
        --lr 1e-6 \\
        --load_checkpoint checkpoints/stage2_vit/checkpoint_final.pt \\
        --output_dir checkpoints/stage3_rl
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image

# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import swanlab

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from OCRVL.model.language_model.ocr_qwen3_vl import (
    OCRQwen3VLForConditionalGeneration,
    Qwen3VLOCRTextAdapter
)
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Distributed Training Setup
# ============================================================================

def setup_distributed():
    """Initialize distributed training process group."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        logger.info(f"Initialized process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        logger.info("Single GPU training (no distributed)")

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


# ============================================================================
# Dataset Classes
# ============================================================================

class AlignmentDataset(Dataset):
    """
    Stage 1: Connector Alignment Dataset

    Minimal dataset for aligning DPSK features to Qwen embedding space.
    Uses simple text rendering + next token prediction.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        ocr_adapter: Qwen3VLOCRTextAdapter,
        max_length: int = 512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.ocr_adapter = ocr_adapter
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Render text and encode with DPSK OCR
        input_ids, ocr_features = self.ocr_adapter.prepare_qwen_inputs(
            instruction="",  # No instruction for alignment
            dense_text=text,
            tokenizer=self.tokenizer,
            return_deepstack=True
        )

        # Create labels for next token prediction
        labels = input_ids.clone()
        # Shift labels: predict next token
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.tokenizer.eos_token_id

        return {
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "ocr_image_features": ocr_features,
        }


class VITDataset(Dataset):
    """
    Stage 2: Visual Instruction Tuning Dataset

    Instruction-following dataset with visual inputs.
    Format: (instruction, image/text, response)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        ocr_adapter: Qwen3VLOCRTextAdapter,
        max_length: int = 2048,
    ):
        """
        Args:
            data: List of dicts with keys:
                - 'instruction': str
                - 'input': str (text to render) or PIL.Image
                - 'output': str (target response)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.ocr_adapter = ocr_adapter
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        input_data = item['input']
        output = item['output']

        # Prepare inputs based on type
        if isinstance(input_data, str):
            # Text input: render and encode
            input_ids, ocr_features = self.ocr_adapter.prepare_qwen_inputs(
                instruction=instruction,
                dense_text=input_data,
                tokenizer=self.tokenizer,
                return_deepstack=True
            )
        elif isinstance(input_data, Image.Image):
            # Image input: encode directly
            input_ids, ocr_features = self.ocr_adapter.prepare_qwen_inputs_from_images(
                instruction=instruction,
                images=[input_data],
                tokenizer=self.tokenizer,
                return_deepstack=True
            )
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Append response and create labels
        response_ids = self.tokenizer(
            output,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Concatenate: [image placeholders] [instruction] [response]
        full_input_ids = torch.cat([input_ids.squeeze(0), response_ids], dim=0)

        # Create labels: mask instruction part, predict response only
        labels = full_input_ids.clone()
        labels[:len(input_ids.squeeze(0))] = -100  # Ignore loss on instruction

        # Truncate if needed
        if len(full_input_ids) > self.max_length:
            full_input_ids = full_input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": full_input_ids,
            "labels": labels,
            "ocr_image_features": ocr_features,
        }


class RLDataset(Dataset):
    """
    Stage 3: Reinforcement Learning Dataset

    Dataset with reward signals for policy optimization.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        ocr_adapter: Qwen3VLOCRTextAdapter,
        max_length: int = 2048,
    ):
        """
        Args:
            data: List of dicts with keys:
                - 'instruction': str
                - 'input': str or PIL.Image
                - 'output': str (target response)
                - 'reward': float (optional, will be computed if not provided)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.ocr_adapter = ocr_adapter
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Similar to VIT dataset but with reward
        item = self.data[idx]
        # ... (implementation similar to VITDataset)
        raise NotImplementedError("RL dataset not fully implemented yet")


# ============================================================================
# Collate Functions
# ============================================================================

def collate_fn(batch):
    """Generic collate function for all stages"""
    # Find max length for padding
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        ids = item["input_ids"]
        lbls = item["labels"]

        # Pad to max length
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), 0, dtype=ids.dtype)])
            lbls = torch.cat([lbls, torch.full((pad_len,), -100, dtype=lbls.dtype)])

        input_ids.append(ids)
        labels.append(lbls)
        attention_mask.append(torch.ones_like(ids))

    # Stack
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask)

    # OCR features: combine from all samples
    ocr_features_list = [item["ocr_image_features"] for item in batch]
    final_feats = []
    deepstack_feats = []

    for ocr_feat in ocr_features_list:
        if isinstance(ocr_feat, tuple):
            final, ds = ocr_feat
            final_feats.extend(final)
            deepstack_feats.extend(ds)
        else:
            final_feats.extend(ocr_feat)

    ocr_features = (final_feats, deepstack_feats) if deepstack_feats else final_feats

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ocr_image_features": ocr_features,
    }


# ============================================================================
# Training Functions
# ============================================================================

def freeze_parameters(model, freeze_encoder=True, freeze_llm=True):
    """
    Freeze model parameters based on training stage

    Args:
        freeze_encoder: Freeze DPSK encoder (always True, it's pretrained)
        freeze_llm: Freeze Qwen LLM (True for stage 1, False for stage 2/3)
    """
    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False

    # LLM is always accessible via model.model.language_model or model.lm_head
    if not freeze_llm:
        # Unfreeze LLM parameters
        if hasattr(model, 'lm_head'):
            for param in model.lm_head.parameters():
                param.requires_grad = True

        if hasattr(model.model, 'language_model'):
            for param in model.model.language_model.parameters():
                param.requires_grad = True

    # Connectors will be unfrozen after first forward pass
    logger.info(f"Freeze config: encoder={freeze_encoder}, llm={freeze_llm}")


def unfreeze_connectors(model):
    """Unfreeze connector parameters after they're created"""
    unfrozen_params = 0

    if hasattr(model.model, 'ocr_connector'):
        for param in model.model.ocr_connector.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()

    if hasattr(model.model, '_ocr_deepstack_connectors'):
        for connector in model.model._ocr_deepstack_connectors.values():
            for param in connector.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()

    logger.info(f"✓ Unfroze {unfrozen_params:,} connector parameters")
    return unfrozen_params


def count_trainable_params(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scaler, global_step, output_dir, args):
    """Save training checkpoint (only from rank 0)"""
    if not is_main_process():
        return

    checkpoint_dir = Path(output_dir) / f"step_{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if needed
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save connector weights
    connector_state = {}
    if hasattr(model_to_save.model, 'ocr_connector'):
        connector_state['ocr_connector'] = model_to_save.model.ocr_connector.state_dict()
    if hasattr(model_to_save.model, '_ocr_deepstack_connectors'):
        connector_state['deepstack_connectors'] = {
            k: v.state_dict()
            for k, v in model_to_save.model._ocr_deepstack_connectors.items()
        }

    torch.save(connector_state, checkpoint_dir / "connectors.pt")

    # Save full model if LLM was trained
    if args.stage in ['vit', 'rl']:
        model_to_save.save_pretrained(checkpoint_dir / "model")

    # Save optimizer and scaler
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'global_step': global_step,
        'args': vars(args),
    }, checkpoint_dir / "training_state.pt")

    logger.info(f"✓ Saved checkpoint to {checkpoint_dir}")


def load_connectors(model, connector_path):
    """Load pretrained connector weights"""
    logger.info(f"Loading connectors from {connector_path}...")
    state = torch.load(connector_path, map_location='cpu')

    if 'ocr_connector' in state and hasattr(model.model, 'ocr_connector'):
        model.model.ocr_connector.load_state_dict(state['ocr_connector'])

    if 'deepstack_connectors' in state and hasattr(model.model, '_ocr_deepstack_connectors'):
        for k, v in state['deepstack_connectors'].items():
            if k in model.model._ocr_deepstack_connectors:
                model.model._ocr_deepstack_connectors[k].load_state_dict(v)

    logger.info("✓ Loaded connector weights")


def train_one_epoch(model, dataloader, optimizer, scaler, args, global_step, max_steps, epoch_num, expected_total_steps):
    """Train for one epoch with comprehensive logging"""
    import time

    model.train()
    running_loss = 0.0
    accumulation_counter = 0
    epoch_start_time = time.time()
    step_start_time = time.time()

    # Track metrics
    total_samples = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0

    # Track task-specific losses (for BLIP3o multi-task training)
    task_losses = {1: [], 2: [], 3: []}  # Task 1: Image cap, Task 2: Text OCR, Task 3: Contrastive

    # Only show progress bar on rank 0
    if is_main_process():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} Step {global_step}")
    else:
        progress_bar = dataloader

    for batch_idx, batch in enumerate(progress_bar):
        if global_step >= max_steps:
            break

        # Move to device
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        ocr_features = batch["ocr_image_features"]

        # Extract task_id if present (for BLIP3o multi-task tracking)
        task_id = batch.get("task_id", None)

        batch_size = input_ids.size(0)
        total_samples += batch_size

        with autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                ocr_image_features=ocr_features,
            )
            loss = outputs.loss / args.gradient_accumulation_steps  # Scale loss

        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulation_counter += 1
        running_loss += loss.item() * args.gradient_accumulation_steps  # Unscale for logging

        # Track task-specific loss
        if task_id is not None:
            task_losses[task_id].append(loss.item() * args.gradient_accumulation_steps)

        # Update weights after accumulation_steps
        if accumulation_counter >= args.gradient_accumulation_steps:
            grad_norm = None
            if args.use_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm
                )
                optimizer.step()

            # Track gradient norm
            if grad_norm is not None:
                grad_norm_sum += grad_norm.item()
                grad_norm_count += 1

            optimizer.zero_grad()
            accumulation_counter = 0
            global_step += 1

        # Enhanced logging (only on rank 0)
        if global_step % args.log_interval == 0 and is_main_process() and accumulation_counter == 0:
            avg_loss = running_loss / args.log_interval
            step_time = time.time() - step_start_time
            samples_per_sec = (args.log_interval * args.batch_size * args.gradient_accumulation_steps * (dist.get_world_size() if dist.is_initialized() else 1)) / step_time

            # GPU memory
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3

            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Average gradient norm
            avg_grad_norm = grad_norm_sum / max(grad_norm_count, 1)

            # Calculate average task-specific losses (for BLIP3o multi-task tracking)
            avg_task1_loss = sum(task_losses[1]) / max(len(task_losses[1]), 1) if task_losses[1] else 0.0
            avg_task2_loss = sum(task_losses[2]) / max(len(task_losses[2]), 1) if task_losses[2] else 0.0
            avg_task3_loss = sum(task_losses[3]) / max(len(task_losses[3]), 1) if task_losses[3] else 0.0

            # ETA
            elapsed_time = time.time() - epoch_start_time
            steps_done = batch_idx + 1
            steps_remaining_epoch = len(dataloader) - steps_done
            eta_epoch = (elapsed_time / steps_done) * steps_remaining_epoch if steps_done > 0 else 0

            # Progress bar
            if hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "samples/s": f"{samples_per_sec:.1f}",
                    "mem": f"{gpu_mem_allocated:.1f}GB"
                })

            # Detailed log - use expected_total_steps for display
            logger.info(
                f"Step {global_step}/{expected_total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"GradNorm: {avg_grad_norm:.3f} | "
                f"Speed: {samples_per_sec:.1f} samples/s | "
                f"GPU: {gpu_mem_allocated:.2f}GB/{gpu_mem_reserved:.2f}GB | "
                f"ETA: {eta_epoch/60:.1f}min"
            )

            if args.use_wandb and HAS_WANDB:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    "train/grad_norm": avg_grad_norm,
                    "train/samples_per_sec": samples_per_sec,
                    "train/gpu_mem_gb": gpu_mem_allocated,
                    "step": global_step
                })

            if args.use_swanlab:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    "train/grad_norm": avg_grad_norm,
                    "train/samples_per_sec": samples_per_sec,
                    "train/gpu_mem_gb": gpu_mem_allocated,
                }

                # Add task-specific losses if available (BLIP3o multi-task training)
                if task_losses[1]:
                    log_dict["train/task1_image_caption_loss"] = avg_task1_loss
                if task_losses[2]:
                    log_dict["train/task2_text_ocr_loss"] = avg_task2_loss
                if task_losses[3]:
                    log_dict["train/task3_contrastive_loss"] = avg_task3_loss

                swanlab.log(log_dict, step=global_step)

            running_loss = 0.0
            grad_norm_sum = 0.0
            grad_norm_count = 0
            task_losses = {1: [], 2: [], 3: []}  # Reset task-specific losses
            step_start_time = time.time()

        # Checkpointing (only on rank 0, with barrier)
        if global_step % args.save_interval == 0 and accumulation_counter == 0:
            save_checkpoint(model, optimizer, scaler, global_step, args.output_dir, args)
            # Synchronize all processes after checkpoint
            if dist.is_initialized():
                dist.barrier()

    # Epoch summary
    epoch_time = time.time() - epoch_start_time
    if is_main_process():
        logger.info(
            f"✓ Epoch {epoch_num} complete | "
            f"Time: {epoch_time/60:.2f}min | "
            f"Samples: {total_samples} | "
            f"Throughput: {total_samples/epoch_time:.1f} samples/s"
        )

    return global_step


# ============================================================================
# Main Training Logic
# ============================================================================

def train_stage_alignment(args):
    """Stage 1: Connector Alignment"""
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    if is_main_process():
        logger.info("=" * 70)
        logger.info("Stage 1: Connector Alignment")
        logger.info("=" * 70)
        logger.info(f"Distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    device = torch.device(f'cuda:{local_rank}')
    args.device = device  # Update args.device to use local_rank

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path, trust_remote_code=True)
    # Qwen3-VL already has vision tokens: <|vision_start|>, <|vision_end|>, <|image_pad|>

    # Load model
    model = OCRQwen3VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    # No need to resize embeddings - using Qwen3-VL's native vision tokens

    # Freeze encoder and LLM
    freeze_parameters(model, freeze_encoder=True, freeze_llm=True)

    # Create OCR connectors upfront (no dummy forward needed)
    # DPSK OCR encoder outputs:
    #   - Final features: 1280-dim (100 tokens)
    #   - Deepstack features: 1024-dim (3 levels)
    logger.info("Creating OCR connectors...")
    target_dim = model.config.text_config.hidden_size  # 2048 for Qwen3-VL-2B

    # Final feature connector: 1280 → 2048
    model.model.ocr_connector = model.model._init_ocr_connector(
        in_dim=1280,
        device=device,
        dtype=torch.bfloat16
    )
    logger.info(f"  ✓ Created final connector: 1280 → {target_dim}")

    # Deepstack connectors: 1024 → 2048 (for 3 intermediate layers)
    model.model._ocr_deepstack_connectors = {}
    model.model._ocr_deepstack_connectors[1024] = model.model._init_ocr_connector(
        in_dim=1024,
        device=device,
        dtype=torch.bfloat16
    )
    logger.info(f"  ✓ Created deepstack connector: 1024 → {target_dim}")

    # Unfreeze connectors immediately
    unfrozen_params = unfreeze_connectors(model)
    trainable_params = count_trainable_params(model)
    if is_main_process():
        logger.info(f"Total trainable parameters: {trainable_params:,}")

    # Wrap model with DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main_process():
            logger.info(f"✓ Wrapped model with DistributedDataParallel")

    # Create dataset FIRST (before loading CUDA models)
    # This avoids conflicts between CUDA model loading and HuggingFace dataset caching
    if args.use_blip3o:
        if is_main_process():
            logger.info("Loading BLIP3o dataset...")

        from OCRVL.data.blip3o_dataset import BLIP3oDataset

        dataset = BLIP3oDataset(
            dataset_type=args.blip3o_dataset,
            base_path=args.blip3o_base_path,
            mix_ratio=args.blip3o_mix_ratio,
            sample_percentage=args.blip3o_sample_percentage,
            num_proc=args.num_workers,
            seed=args.seed,
        )

        if is_main_process():
            logger.info("✓ Dataset loaded. Now initializing OCR adapter...")

        # Load OCR adapter (needed on all ranks for collate function)
        # Each rank loads its own copy (~800MB per GPU)
        ocr_adapter = Qwen3VLOCRTextAdapter(
            encoder_model_path=args.dpsk_model_path,
            device=args.device,
            use_deepstack=True
        )

        if is_main_process():
            logger.info("✓ OCR adapter initialized")

        # Create custom collate function for BLIP3o (handles all task logic)
        from OCRVL.data.blip3o_collate import create_blip3o_collate_fn
        blip3o_collate_fn = create_blip3o_collate_fn(
            tokenizer=tokenizer,
            ocr_adapter=ocr_adapter,
            task_ratios=(0.4, 0.4, 0.2),  # Task 1: 40%, Task 2: 40%, Task 3: 20%
            seed=args.seed,
        )
        if is_main_process():
            logger.info("✓ Created BLIP3o collate function (handles Task 1/2/3 formatting)")

    else:
        logger.info(f"Loading dataset from {args.data_path}...")

        # Load OCR adapter first for non-BLIP3o datasets
        ocr_adapter = Qwen3VLOCRTextAdapter(
            encoder_model_path=args.dpsk_model_path,
            device=args.device,
            use_deepstack=True
        )

        if args.data_path and Path(args.data_path).exists():
            with open(args.data_path) as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            logger.warning("No dataset provided, using dummy data")
            texts = ["Sample text " * 50] * 1000

        dataset = AlignmentDataset(texts, tokenizer, ocr_adapter)

    # DataLoader
    # BLIP3o uses encoding in collate_fn (CUDA), so num_workers must be 0
    if args.num_workers > 0 and args.use_blip3o:
        if is_main_process():
            logger.warning(f"  Setting num_workers=0 (was {args.num_workers}) because BLIP3o uses CUDA in collate_fn")
        dataloader_num_workers = 0
    else:
        dataloader_num_workers = args.num_workers

    # Use DistributedSampler for multi-GPU training
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    shuffle = (sampler is None)  # Only shuffle if not using sampler

    # Select appropriate collate function
    if args.use_blip3o:
        active_collate_fn = blip3o_collate_fn  # Custom collate with task logic
    else:
        active_collate_fn = collate_fn  # Default collate

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=dataloader_num_workers,
        collate_fn=active_collate_fn,
        pin_memory=False,  # Disabled: OCR features are already on CUDA
    )

    # Calculate dataset statistics (needed by all processes)
    dataset_size = len(dataset)
    effective_batch_size = args.batch_size * world_size * args.gradient_accumulation_steps
    steps_per_epoch = (dataset_size + effective_batch_size - 1) // effective_batch_size

    # Determine actual training limit (needed by all processes)
    if hasattr(args, '_training_by_epochs') and args._training_by_epochs:
        # Training by epochs: set max_steps to expected steps or use provided max_steps as safety limit
        expected_steps = steps_per_epoch * args.num_epochs
        if args.max_steps > expected_steps:
            # max_steps is just a safety limit, actual training will be epoch-based
            actual_max_steps = expected_steps
            max_steps_note = f"{args.max_steps} (safety limit, will train for {expected_steps} steps = {args.num_epochs} epoch(s))"
        else:
            # max_steps is lower than expected, will stop early
            actual_max_steps = args.max_steps
            max_steps_note = f"{args.max_steps} (will stop early, before completing {args.num_epochs} epoch(s))"
    else:
        # Training by steps: use max_steps directly
        expected_steps = min(args.max_steps, steps_per_epoch * args.num_epochs)
        actual_max_steps = args.max_steps
        max_steps_note = f"{args.max_steps}"

    total_steps = expected_steps

    if is_main_process():
        logger.info(f"DataLoader: batch_size={args.batch_size} (per GPU), world_size={world_size}, "
                   f"accumulation_steps={args.gradient_accumulation_steps}, "
                   f"effective_batch_size={args.batch_size * world_size * args.gradient_accumulation_steps}")

        logger.info(f"=" * 70)
        logger.info(f"Training Configuration Summary")
        logger.info(f"=" * 70)
        logger.info(f"  Dataset size: {dataset_size:,} samples")
        logger.info(f"  Batch size per GPU: {args.batch_size}")
        logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {effective_batch_size:,}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total epochs: {args.num_epochs}")
        logger.info(f"  Max steps: {max_steps_note}")
        logger.info(f"  Expected total steps: {total_steps}")
        logger.info(f"  Learning rate: {args.lr:.2e}")
        logger.info(f"  Weight decay: {args.weight_decay}")
        if args.use_blip3o:
            sample_pct_str = f"{args.blip3o_sample_percentage*100:.2f}%" if args.blip3o_sample_percentage else "100% (full dataset, using cached)"
            logger.info(f"  Dataset: BLIP3o {args.blip3o_dataset} - {sample_pct_str}")
        logger.info(f"=" * 70)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # GradScaler - not needed for bfloat16, only for float16
    # Since model is in bfloat16, disable GradScaler
    scaler = GradScaler('cuda', enabled=False)
    logger.info("  GradScaler disabled (model uses bfloat16, not float16)")

    # Training loop
    import time
    training_start_time = time.time()
    global_step = 0

    for epoch in range(args.num_epochs):
        if is_main_process():
            logger.info(f"=" * 70)
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
            logger.info(f"=" * 70)

        # Set epoch for sampler (important for reproducibility with DistributedSampler)
        if sampler is not None:
            sampler.set_epoch(epoch)

        global_step = train_one_epoch(
            model, dataloader, optimizer, scaler, args, global_step,
            actual_max_steps, epoch + 1, total_steps
        )

        if global_step >= actual_max_steps:
            break

    # Final save
    save_checkpoint(model, optimizer, scaler, global_step, args.output_dir, args)

    # Training completion summary
    total_training_time = time.time() - training_start_time
    if is_main_process():
        logger.info(f"=" * 70)
        logger.info(f"Training Complete!")
        logger.info(f"=" * 70)
        logger.info(f"  Total steps: {global_step}")
        logger.info(f"  Total time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
        logger.info(f"  Average time per step: {total_training_time/max(global_step,1):.2f} seconds")
        logger.info(f"  Final checkpoint: {args.output_dir}/step_{global_step}")
        logger.info(f"=" * 70)

    # Cleanup distributed training
    cleanup_distributed()


def train_stage_vit(args):
    """Stage 2: Visual Instruction Tuning"""
    logger.info("=" * 70)
    logger.info("Stage 2: Visual Instruction Tuning")
    logger.info("=" * 70)

    device = torch.device(args.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path, trust_remote_code=True)
    # Qwen3-VL already has vision tokens: <|vision_start|>, <|vision_end|>, <|image_pad|>

    # Load model
    model = OCRQwen3VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    # No need to resize embeddings - using Qwen3-VL's native vision tokens

    # Freeze encoder, unfreeze LLM
    freeze_parameters(model, freeze_encoder=True, freeze_llm=False)

    # Load pretrained connectors if provided
    if args.load_connectors:
        # Need dummy forward first
        ocr_adapter = Qwen3VLOCRTextAdapter(
            encoder_model_path=args.dpsk_model_path,
            device=args.device,
            use_deepstack=True
        )
        # Create dummy batch
        dummy_text = ["test"] * args.batch_size
        dummy_dataset = AlignmentDataset(dummy_text, tokenizer, ocr_adapter)
        dummy_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        dummy_batch = next(iter(dummy_loader))

        model.eval()
        with torch.no_grad():
            # Slice ocr_image_features to match input_ids[:1]
            ocr_feats_batch = dummy_batch["ocr_image_features"]
            if isinstance(ocr_feats_batch, tuple):
                final_feats, deepstack_feats = ocr_feats_batch
                ocr_feats_slice = ([final_feats[0]], [deepstack_feats[0]])
            elif isinstance(ocr_feats_batch, list):
                ocr_feats_slice = [ocr_feats_batch[0]]
            else:
                raise ValueError(f"Unexpected ocr_image_features type: {type(ocr_feats_batch)}")

            _ = model(
                input_ids=dummy_batch["input_ids"][:1].to(device),
                ocr_image_features=ocr_feats_slice
            )

        load_connectors(model, args.load_connectors)

    # Unfreeze connectors
    unfreeze_connectors(model)
    trainable_params = count_trainable_params(model)
    logger.info(f"Total trainable parameters: {trainable_params:,}")

    # TODO: Load VIT dataset and implement training loop
    logger.error("VIT stage not fully implemented yet")
    raise NotImplementedError


def train_stage_rl(args):
    """Stage 3: Reinforcement Learning"""
    logger.info("=" * 70)
    logger.info("Stage 3: Reinforcement Learning")
    logger.info("=" * 70)
    logger.error("RL stage not implemented yet")
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()

    # Stage selection
    parser.add_argument("--stage", type=str, required=True,
                       choices=["alignment", "vit", "rl"])

    # Model paths
    parser.add_argument("--dpsk_model_path", type=str,
                       default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--qwen_model_path", type=str,
                       default="Qwen/Qwen3-VL-2B-Instruct")

    # Data
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data (text file or jsonl)")

    # BLIP3o dataset (alternative to --data_path)
    parser.add_argument("--use_blip3o", action="store_true",
                       help="Use BLIP3o dataset instead of custom data")
    parser.add_argument("--blip3o_dataset", type=str, default="mixed",
                       choices=["short", "long", "60k", "mixed"],
                       help="BLIP3o dataset variant: short (concise), long (detailed), 60k (curated), or mixed")
    parser.add_argument("--blip3o_base_path", type=str,
                       default="/share/project/xiyan/huggingface/BLIP3o",
                       help="Base path to BLIP3o datasets")
    parser.add_argument("--blip3o_mix_ratio", type=float, default=0.5,
                       help="For 'mixed': ratio of short to long captions (0=all long, 1=all short)")
    parser.add_argument("--blip3o_sample_percentage", type=float, default=None,
                       help="Percentage of dataset to sample (e.g., 0.01 = 1%, None = 100% full dataset). "
                            "Default: None (uses Lumina-DiMOO's cached full datasets)")
    parser.add_argument("--blip3o_use_images", action="store_true", default=True,
                       help="Use real images from BLIP3o (vs rendering captions)")
    parser.add_argument("--blip3o_image_caption_ratio", type=float, default=0.5,
                       help="Ratio of real images to rendered captions (0=all rendered, 1=all real)")
    parser.add_argument("--enable_feature_cache", action="store_true", default=False,
                       help="Enable in-memory caching of encoded features (uses ~3GB RAM per GPU, speeds up epoch 2+)")

    # Training config
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=100000,
                       help="Maximum training steps (default: 100000). When num_epochs is set, this acts as a safety limit.")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size per GPU (effective batch = batch_size × num_gpus × accumulation_steps)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # Logging
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpsk-qwen-alignment")
    parser.add_argument("--use_swanlab", action="store_true", default=False,
                       help="Enable SwanLab logging (requires swanlab[dashboard])")
    parser.add_argument("--swanlab_project", type=str, default="dpsk-qwen-alignment")
    parser.add_argument("--swanlab_experiment", type=str, default="blip3o-alignment-{timestamp}")

    # Paths
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    parser.add_argument("--load_connectors", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # System
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device (will be overridden by local_rank in distributed training)")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    # Get rank early for logging control
    rank = int(os.environ.get('RANK', 0))
    is_main = (rank == 0)

    # Logic: if epochs is set (not default), calculate expected steps and ignore max_steps
    # This ensures 1 epoch completes the entire dataset, not stopped by max_steps
    if args.num_epochs != 5:
        # Mark that we're using epoch-based training
        args._training_by_epochs = True
        if is_main:
            print(f"[INFO] num_epochs={args.num_epochs} is set → training by epochs (max_steps used as safety limit only)")
    else:
        args._training_by_epochs = False


    # Set default output directory if not provided (only on rank 0)
    if is_main and (not args.output_dir or args.output_dir in ["checkpoints/blip3o_alignment", "checkpoints/stage1_alignment"]):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_type = args.blip3o_dataset if args.use_blip3o else "custom"
        experiment_setting = f"{args.stage}_{dataset_type}"
        args.output_dir = f"OCRVL/checkpoints/{experiment_setting}_{timestamp}"

    # Create output directory (only on rank 0)
    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup file logging
        file_handler = logging.FileHandler(Path(args.output_dir) / "training.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Logging to: {Path(args.output_dir) / 'training.log'}")

    # Initialize wandb (only on rank 0)
    if args.use_wandb and is_main:
        if HAS_WANDB:
            wandb.init(project=args.wandb_project, config=vars(args))
        else:
            logger.warning("wandb not installed, skipping wandb logging")

    # Initialize swanlab (only on rank 0)
    if args.use_swanlab and is_main:
        from datetime import datetime
        experiment_name = args.swanlab_experiment.replace(
            '{timestamp}', datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=experiment_name,
            config=vars(args),
            mode="local"  # Run in local mode (no cloud upload)
        )
        logger.info(f"SwanLab initialized: {args.swanlab_project}/{experiment_name} (local mode)")

    # Save config (only on rank 0)
    if is_main:
        with open(Path(args.output_dir) / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Run training
    if args.stage == "alignment":
        train_stage_alignment(args)
    elif args.stage == "vit":
        train_stage_vit(args)
    elif args.stage == "rl":
        train_stage_rl(args)


if __name__ == "__main__":
    main()
