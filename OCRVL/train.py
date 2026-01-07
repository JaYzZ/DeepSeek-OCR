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
import math
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

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False
    swanlab = None

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# CUDA Configuration for Better Error Detection
# ============================================================================
# Enable verbose CUDA error reporting
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')  # Keep async for speed
# Memory allocation strategy to reduce fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,expandable_segments:True')
# Enable NCCL debug on failure
os.environ.setdefault('NCCL_DEBUG', 'WARN')
# ============================================================================

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
    """Unfreeze connector and thinking projection parameters after they're created (works for both regular and LoRA-wrapped models)"""
    unfrozen_params = 0
    unfrozen_names = []

    logger.info("Unfreezing connector and thinking projection parameters...")

    # Step 1: Unfreeze via named_parameters (for properly registered connectors and thinking projection)
    for name, param in model.named_parameters():
        # Match connector parameters (ocr_connector or deepstack connectors) OR thinking_projection
        if ('connector' in name.lower() and ('ocr_connector' in name or 'deepstack' in name)) or 'thinking_projection' in name:
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_params += param.numel()
                unfrozen_names.append(name)
                logger.info(f"  ✓ Unfroze {name}: {param.numel():,} params")
            else:
                logger.info(f"  ⚠️  Already trainable: {name}: {param.numel():,} params")

    # Step 2: Handle dictionary-stored deepstack connectors (not registered via named_parameters)
    # Access them through the model structure
    if hasattr(model, 'peft_config'):
        # LoRA-wrapped model
        base_model = model.base_model.model.model
    else:
        # Regular model
        base_model = model.model

    if hasattr(base_model, '_ocr_deepstack_connectors'):
        logger.info(f"  Found _ocr_deepstack_connectors dictionary with {len(base_model._ocr_deepstack_connectors)} connectors")
        for key, connector in base_model._ocr_deepstack_connectors.items():
            for name, param in connector.named_parameters():
                full_name = f"_ocr_deepstack_connectors[{key}].{name}"
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                    unfrozen_names.append(full_name)
                    logger.info(f"  ✓ Unfroze {full_name}: {param.numel():,} params")
                else:
                    logger.info(f"  ⚠️  Already trainable: {full_name}: {param.numel():,} params")

    # Step 3: Handle thinking_projection if not caught by Step 1
    if hasattr(model, 'peft_config'):
        base_model = model.base_model.model.model
    else:
        base_model = model.model

    if hasattr(base_model, 'thinking_projection') and base_model.thinking_projection is not None:
        logger.info(f"  Found thinking_projection module")
        for name, param in base_model.thinking_projection.named_parameters():
            full_name = f"thinking_projection.{name}"
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_params += param.numel()
                unfrozen_names.append(full_name)
                logger.info(f"  ✓ Unfroze {full_name}: {param.numel():,} params")
            else:
                logger.info(f"  ⚠️  Already trainable: {full_name}: {param.numel():,} params")

    logger.info(f"✓ Unfroze {unfrozen_params:,} parameters ({len(unfrozen_names)} tensors)")

    # Verify by recounting trainable params
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_trainable = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)
    connector_trainable = sum(p.numel() for n, p in model.named_parameters() if 'connector' in n.lower() and p.requires_grad)
    thinking_trainable = sum(p.numel() for n, p in model.named_parameters() if 'thinking_projection' in n and p.requires_grad)

    logger.info(f"  Verification after unfreezing:")
    logger.info(f"    Total trainable: {total_trainable:,}")
    logger.info(f"    LoRA trainable: {lora_trainable:,}")
    logger.info(f"    Connector trainable: {connector_trainable:,}")
    logger.info(f"    Thinking projection trainable: {thinking_trainable:,}")
    logger.info(f"    Expected total: {lora_trainable + connector_trainable + thinking_trainable:,}")

    if total_trainable != lora_trainable + connector_trainable + thinking_trainable:
        other = total_trainable - lora_trainable - connector_trainable - thinking_trainable
        logger.warning(f"    ⚠️  Unexpected trainable params: {other:,} (should be 0!)")
        logger.warning("    Listing first 10 unexpected trainable params:")
        count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora_' not in name and 'connector' not in name.lower():
                logger.warning(f"      - {name}: {param.numel():,}")
                count += 1
                if count >= 10:
                    break

    return unfrozen_params


def count_trainable_params(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_lora_to_llm(model, args):
    """
    Apply LoRA adapters to Qwen3-VL LLM only (not DPSK encoder or connectors).

    Args:
        model: OCRQwen3VLForConditionalGeneration model
        args: Training arguments with LoRA config

    Returns:
        model: Model with LoRA adapters applied
    """
    if not args.use_lora:
        return model

    if not HAS_PEFT:
        raise ImportError("peft library not installed. Install with: pip install peft")

    logger.info("=" * 70)
    logger.info("Applying LoRA to Qwen3-VL LLM")
    logger.info("=" * 70)

    # LoRA configuration - targeting Qwen3-VL's attention layers only
    # IMPORTANT: We use modules_to_save=["model.ocr_connector", "model._ocr_deepstack_connectors"]
    # to keep connectors trainable (PEFT won't freeze them)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj"  # MLP layers
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=[]  # Connectors will be manually unfrozen after model creation
    )

    logger.info(f"  LoRA rank (r): {lora_config.r}")
    logger.info(f"  LoRA alpha: {lora_config.lora_alpha}")
    logger.info(f"  LoRA dropout: {lora_config.lora_dropout}")
    logger.info(f"  Target modules: {lora_config.target_modules}")

    # Apply LoRA using get_peft_model
    # This wraps the model and adds LoRA adapters to target modules
    model = get_peft_model(model, lora_config)

    # Count LoRA parameters
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  LoRA parameters: {lora_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info("=" * 70)

    return model


def save_checkpoint(model, optimizer, scaler, global_step, output_dir, args, checkpoint_name=None, scheduler=None, ocr_adapter=None, tokenizer=None):
    """Save training checkpoint (only from rank 0)

    Args:
        checkpoint_name: Optional override for checkpoint directory name (e.g., "step_latest").
                        If None, uses f"step_{global_step}"
        scheduler: Optional learning rate scheduler to save
        ocr_adapter: Optional OCR adapter for checkpoint evaluation
        tokenizer: Optional tokenizer for checkpoint evaluation
    """
    if not is_main_process():
        return

    if checkpoint_name is None:
        checkpoint_name = f"step_{global_step}"
    checkpoint_dir = Path(output_dir) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Unwrap DDP if needed
        model_to_save = model.module if hasattr(model, 'module') else model

        # Save connector weights (always, regardless of LoRA)
        # For LoRA models, we need to access the base model's connectors
        # Regular: model_to_save.model.ocr_connector (OCRQwen3VLForConditionalGeneration → .model → OCRQwen3VLModel)
        # LoRA:    model_to_save.base_model.model.model.ocr_connector (PeftModel → .base_model.model → OCRQwen3VLForConditionalGeneration → .model → OCRQwen3VLModel)
        # Check if this is a LoRA-wrapped model by checking for peft_config attribute
        if hasattr(model_to_save, 'peft_config'):
            # LoRA wrapped model: PeftModel → base_model.model → OCRQwen3VLForConditionalGeneration → .model → OCRQwen3VLModel
            target_model = model_to_save.base_model.model.model
        else:
            # Regular model: OCRQwen3VLForConditionalGeneration → .model → OCRQwen3VLModel
            target_model = model_to_save.model

        connector_state = {}
        if hasattr(target_model, 'ocr_connector'):
            connector_state['ocr_connector'] = target_model.ocr_connector.state_dict()
            logger.info(f"  ✓ Saving ocr_connector with {sum(p.numel() for p in target_model.ocr_connector.parameters())} params")
        if hasattr(target_model, '_ocr_deepstack_connectors'):
            connector_state['deepstack_connectors'] = {
                k: v.state_dict()
                for k, v in target_model._ocr_deepstack_connectors.items()
            }
            logger.info(f"  ✓ Saving {len(target_model._ocr_deepstack_connectors)} deepstack connectors")

        # Save thinking projection if it exists
        if hasattr(target_model, 'thinking_projection') and target_model.thinking_projection is not None:
            connector_state['thinking_projection'] = target_model.thinking_projection.state_dict()
            logger.info(f"  ✓ Saving thinking_projection with {sum(p.numel() for p in target_model.thinking_projection.parameters())} params")

        if not connector_state:
            logger.warning(f"  ⚠️  No connectors found to save! target_model type: {type(target_model)}")
            logger.warning(f"  ⚠️  target_model attributes: {dir(target_model)[:10]}...")

        torch.save(connector_state, checkpoint_dir / "connectors.pt")
        logger.info(f"  ✓ Saved connectors.pt ({(checkpoint_dir / 'connectors.pt').stat().st_size / 1024 / 1024:.2f} MB)")

        # Save LoRA adapters if using LoRA
        if args.use_lora and hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(checkpoint_dir / "lora_adapters")
            logger.info(f"  ✓ Saved LoRA adapters to {checkpoint_dir / 'lora_adapters'}")

        # Save full model if LLM was trained (stage 2/3 without LoRA)
        if args.stage in ['vit', 'rl'] and not args.use_lora:
            model_to_save.save_pretrained(checkpoint_dir / "model")
            logger.info(f"  ✓ Saved full model to {checkpoint_dir / 'model'}")

        # Save optimizer, scaler, and scheduler
        training_state = {
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'global_step': global_step,
            'args': vars(args),
        }

        # Add scheduler state if available
        if scheduler is not None:
            training_state['scheduler'] = scheduler.state_dict()
            logger.info(f"  ✓ Saving scheduler state")

        torch.save(training_state, checkpoint_dir / "training_state.pt")
        logger.info(f"  ✓ Saved training_state.pt ({(checkpoint_dir / 'training_state.pt').stat().st_size / 1024 / 1024:.2f} MB)")

        # Run checkpoint evaluation if enabled
        if hasattr(args, 'enable_checkpoint_eval') and args.enable_checkpoint_eval:
            if ocr_adapter is not None and tokenizer is not None:
                logger.info("")
                logger.info("Running checkpoint evaluation...")
                try:
                    from OCRVL.training.checkpoint_eval import run_checkpoint_evaluation

                    # Determine image base directory for evaluation samples
                    # Use LLaVA image directory if available, otherwise try to construct correct path
                    if hasattr(args, 'llava_image_dir') and args.llava_image_dir:
                        eval_image_base = args.llava_image_dir
                        # If it's just the huggingface root, append the LLaVA-Instruct-150K/images path
                        if eval_image_base.endswith('/huggingface'):
                            eval_image_base = f"{eval_image_base}/liuhaotian/LLaVA-Instruct-150K/images"
                    elif hasattr(args, 'blip3o_base_path'):
                        # For BLIP3o training, eval samples should still use LLaVA images
                        eval_image_base = "/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/images"
                    else:
                        # Fallback to default LLaVA images path
                        eval_image_base = "/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/images"

                    run_checkpoint_evaluation(
                        model=model,
                        tokenizer=tokenizer,
                        ocr_adapter=ocr_adapter,
                        checkpoint_dir=checkpoint_dir,
                        eval_samples_path=args.checkpoint_eval_samples,
                        image_base_dir=eval_image_base,
                        global_step=global_step,
                        args=args,
                        max_samples=args.checkpoint_eval_max_samples,
                    )
                except Exception as e:
                    logger.warning(f"  ⚠️  Checkpoint evaluation failed: {e}")
                    logger.warning(f"  Training will continue without evaluation")

        logger.info(f"✓ Checkpoint saved successfully to {checkpoint_dir}")

    except Exception as e:
        logger.error(f"❌ FAILED to save checkpoint to {checkpoint_dir}: {e}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Traceback:\n{traceback.format_exc()}")
        raise  # Re-raise to stop training if checkpoint fails


def load_connectors(model, connector_path):
    """Load pretrained connector weights (works for both regular and LoRA-wrapped models)"""
    if is_main_process():
        logger.info(f"Loading connectors from {connector_path}...")

    state = torch.load(connector_path, map_location='cpu')

    # Handle both regular models and LoRA-wrapped models
    # Regular: model.model (OCRQwen3VLForConditionalGeneration → .model → OCRQwen3VLModel)
    # LoRA:    model.base_model.model.model (PeftModel → .base_model.model → OCRQwen3VLForConditionalGeneration → .model → OCRQwen3VLModel)
    # Check if this is a LoRA-wrapped model by checking for peft_config attribute
    if hasattr(model, 'peft_config'):
        # LoRA-wrapped model: PeftModel → base_model.model → OCRQwen3VLForConditionalGeneration → .model → OCRQwen3VLModel
        target_model = model.base_model.model.model
    else:
        # Regular model: OCRQwen3VLForConditionalGeneration → .model → OCRQwen3VLModel
        target_model = model.model

    if 'ocr_connector' in state and hasattr(target_model, 'ocr_connector'):
        target_model.ocr_connector.load_state_dict(state['ocr_connector'])

    if 'deepstack_connectors' in state and hasattr(target_model, '_ocr_deepstack_connectors'):
        for k, v in state['deepstack_connectors'].items():
            if k in target_model._ocr_deepstack_connectors:
                target_model._ocr_deepstack_connectors[k].load_state_dict(v)

    # Load thinking projection if it exists in checkpoint
    if 'thinking_projection' in state:
        # Initialize projection first if not already initialized
        if not hasattr(target_model, 'thinking_projection') or target_model.thinking_projection is None:
            target_model._maybe_get_thinking_projection(
                device=next(target_model.parameters()).device,
                dtype=next(target_model.parameters()).dtype
            )
        target_model.thinking_projection.load_state_dict(state['thinking_projection'])
        if is_main_process():
            logger.info("  ✓ Loaded thinking_projection")

    if is_main_process():
        logger.info("✓ Loaded connector weights")


def load_checkpoint_with_lora(model, checkpoint_path, args):
    """
    Load checkpoint and resume training with LoRA support.

    Args:
        model: Base model (before LoRA is applied)
        checkpoint_path: Path to checkpoint directory (e.g., step_786)
        args: Training arguments

    Returns:
        model: Model with loaded connectors and LoRA adapters (if applicable)
        optimizer_state: Optimizer state dict to restore
        scaler_state: GradScaler state dict to restore
        scheduler_state: Scheduler state dict to restore
        global_step: Global step to resume from
    """
    checkpoint_path = Path(checkpoint_path)
    if is_main_process():
        logger.info(f"Loading checkpoint from {checkpoint_path}...")

    # Load connectors first
    connector_path = checkpoint_path / "connectors.pt"
    if connector_path.exists():
        load_connectors(model, connector_path)
    else:
        if is_main_process():
            logger.warning(f"  Connector checkpoint not found: {connector_path}")

    # Synchronize all ranks after loading connectors
    if dist.is_initialized():
        dist.barrier()

    # Load LoRA adapters if they exist in checkpoint (always load if available)
    lora_adapter_path = checkpoint_path / "lora_adapters"
    if lora_adapter_path.exists():
        if is_main_process():
            logger.info(f"Loading LoRA adapters from {lora_adapter_path}...")
        if not HAS_PEFT:
            raise ImportError("peft library required for loading LoRA checkpoints")

        # Load LoRA model (always load if checkpoint has LoRA, regardless of current use_lora setting)
        model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
        if is_main_process():
            logger.info("  ✓ Loaded LoRA adapters from checkpoint")
            if not args.use_lora:
                logger.warning("  ⚠️  Checkpoint has LoRA but use_lora=False. Loaded LoRA anyway (checkpoint takes priority).")
    elif args.use_lora:
        # LoRA enabled but not in checkpoint - will be initialized by caller
        if is_main_process():
            logger.info(f"  ℹ️  LoRA adapters not found at {lora_adapter_path}, will initialize new LoRA layers")

    # Load training state only if explicitly requested
    training_state_path = checkpoint_path / "training_state.pt"
    optimizer_state = None
    scaler_state = None
    scheduler_state = None
    global_step = 0

    if args.resume_training_state and training_state_path.exists():
        training_state = torch.load(training_state_path, map_location='cpu')
        optimizer_state = training_state.get('optimizer', None)
        scaler_state = training_state.get('scaler', None)
        scheduler_state = training_state.get('scheduler', None)
        global_step = training_state.get('global_step', 0)
        if is_main_process():
            logger.info(f"  ✓ Loaded training state (resuming from step {global_step})")
            if scheduler_state is not None:
                logger.info(f"  ✓ Found scheduler state in checkpoint")
    elif args.resume_training_state and is_main_process():
        logger.warning(f"  Training state not found: {training_state_path}")
    elif is_main_process():
        logger.info(f"  ℹ️  Starting fresh from step 0 (only loaded weights from {checkpoint_path})")

    # Synchronize all ranks after loading checkpoint
    if dist.is_initialized():
        dist.barrier()

    return model, optimizer_state, scaler_state, scheduler_state, global_step


def train_one_epoch(model, dataloader, optimizer, scaler, args, global_step, max_steps, epoch_num, expected_total_steps, tokenizer, ocr_adapter, scheduler=None):
    """Train for one epoch with comprehensive logging and task formatting

    Args:
        scheduler: Optional learning rate scheduler
    """
    import time
    # Note: BLIP3o task formatters are imported conditionally below when dataset_type == "blip3o"

    model.train()
    running_loss = 0.0
    accumulation_counter = 0
    epoch_start_time = time.time()
    step_start_time = time.time()

    # Track metrics
    total_samples = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    steps_in_epoch = 0  # Track steps completed in current epoch (for ETA calculation)

    # Track task-specific losses (for BLIP3o multi-task training)
    task_losses = {1: [], 2: []}  # Task 1: Image captioning, Task 2: OCR transcription

    # Task selection RNG (seeded for reproducibility)
    import random
    task_rng = random.Random(args.seed)
    sample_rng = random.Random(args.seed + 1)  # Separate RNG for task formatting

    # Only show progress bar on rank 0
    if is_main_process():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} Step {global_step}")
    else:
        progress_bar = dataloader

    # Batch encoding optimization: prefetch and encode in large batches
    dataloader_iter = iter(progress_bar)

    while global_step < max_steps:
        # Prefetch gradient_accumulation_steps batches
        prefetch_batches = []
        try:
            for _ in range(args.gradient_accumulation_steps):
                prefetch_batches.append(next(dataloader_iter))
        except StopIteration:
            if not prefetch_batches:
                break  # End of epoch

        # For LLaVA: batch-encode all images at once
        if args.dataset_type == "llava":
            # Collect all images from all micro-batches
            all_images = []
            batch_metadata = []
            
            for raw_batch in prefetch_batches:
                if "images_to_encode" in raw_batch:
                    start_idx = len(all_images)
                    batch_images = []
                    for sample_images in raw_batch["images_to_encode"]:
                        batch_images.extend(sample_images)
                    all_images.extend(batch_images)
                    batch_metadata.append({
                        'batch': raw_batch,
                        'start': start_idx,
                        'count': len(batch_images)
                    })
                else:
                    batch_metadata.append({
                        'batch': raw_batch,
                        'start': -1,
                        'count': 0
                    })
            
            # SINGLE BATCH ENCODING (32x more efficient!)
            if all_images:
                print(f"[Batch Encoding] Encoding {len(all_images)} images in one batch (grad_accum={args.gradient_accumulation_steps})")
                all_encoded = ocr_adapter.images_to_ocr_features(all_images)
                
                # Distribute features back to batches
                for meta in batch_metadata:
                    if meta['count'] > 0:
                        start, end = meta['start'], meta['start'] + meta['count']
                        if isinstance(all_encoded, tuple):
                            meta['features'] = (all_encoded[0][start:end], all_encoded[1][start:end])
                        else:
                            meta['features'] = all_encoded[start:end]
                    else:
                        meta['features'] = None
            
            # Process each micro-batch with pre-encoded features
            for meta in batch_metadata:
                sub_batch = meta['batch']
                
                if meta['features'] is not None:
                    ocr_features = meta['features']
                elif "ocr_image_features" in sub_batch:
                    ocr_features = sub_batch["ocr_image_features"]
                else:
                    continue
                
                input_ids = sub_batch["input_ids"].to(args.device)
                attention_mask = sub_batch["attention_mask"].to(args.device)
                labels = sub_batch["labels"].to(args.device)
                
                batch_size = input_ids.size(0)
                total_samples += batch_size
                
                # Forward + Backward
                with autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                  labels=labels, ocr_image_features=ocr_features)
                    loss = outputs.loss / args.gradient_accumulation_steps

                # Numerical stability check - detect NaN/Inf before backward pass
                # CRITICAL: Must always do backward() to maintain NCCL sync across DDP ranks
                batch_is_valid = True
                if not torch.isfinite(loss):
                    logger.warning(f"⚠️  Step {global_step} (LLaVA): Non-finite loss detected: {loss.item()}")
                    logger.warning(f"   Input IDs shape: {input_ids.shape}, Labels shape: {labels.shape}")
                    loss = loss * 0.0  # Zero out loss but still do backward for DDP sync
                    batch_is_valid = False

                # Backward pass - MUST ALWAYS EXECUTE to maintain DDP gradient synchronization
                # If we skip backward(), this rank won't participate in NCCL ALLREDUCE and other ranks will hang
                if args.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulation_counter += 1

                # Track loss only if batch was valid
                if batch_is_valid:
                    loss_value = loss.detach().item() * args.gradient_accumulation_steps
                    if math.isfinite(loss_value):
                        running_loss += loss_value
                        task_losses[1].append(loss_value)
                    else:
                        logger.warning(f"⚠️  Step {global_step} (LLaVA): Non-finite loss value after backward")
                        batch_is_valid = False

            # Clean up LLaVA batch processing variables to prevent memory accumulation
            del all_images, all_encoded, batch_metadata

        # For BLIP3o and DocLayNet: process through BLIP3o task formatters (both are image-caption pairs)
        elif args.dataset_type in ["blip3o", "doclaynet"]:
            for raw_batch in prefetch_batches:
                # Check if this is a mixed batch (has dataset_source markers)
                is_mixed = 'dataset_source' in raw_batch[0] if raw_batch else False

                if is_mixed:
                    # Mixed batch: separate by source
                    blip3o_samples = [s for s in raw_batch if s.get('dataset_source') == 'blip3o']
                    doclaynet_samples = [s for s in raw_batch if s.get('dataset_source') == 'doclaynet']

                    # Remove source markers
                    for s in blip3o_samples:
                        s.pop('dataset_source', None)
                    for s in doclaynet_samples:
                        s.pop('dataset_source', None)

                    formatted_batches = []

                    # BLIP3o: Process BOTH tasks (caption + OCR)
                    if blip3o_samples:
                        from OCRVL.data.blip3o_tasks import format_caption_batch, format_ocr_batch
                        caption_batches = format_caption_batch(blip3o_samples, tokenizer, sample_rng)
                        ocr_batches = format_ocr_batch(blip3o_samples, tokenizer, sample_rng)
                        formatted_batches.extend(caption_batches + ocr_batches)

                    # DocLayNet: Process ONLY OCR task (document image + OCR instruction -> text)
                    if doclaynet_samples:
                        from OCRVL.data.blip3o_tasks import format_ocr_batch
                        doclaynet_ocr_batches = format_ocr_batch(doclaynet_samples, tokenizer, sample_rng)
                        formatted_batches.extend(doclaynet_ocr_batches)

                else:
                    # Pure BLIP3o or DocLayNet batch
                    from OCRVL.data.blip3o_tasks import format_caption_batch, format_ocr_batch

                    if args.dataset_type == "blip3o":
                        # BLIP3o: Process BOTH tasks
                        caption_batches = format_caption_batch(raw_batch, tokenizer, sample_rng)
                        ocr_batches = format_ocr_batch(raw_batch, tokenizer, sample_rng)
                        formatted_batches = caption_batches + ocr_batches
                    else:
                        # DocLayNet: Process ONLY OCR task
                        formatted_batches = format_ocr_batch(raw_batch, tokenizer, sample_rng)

            # PROCESS SUB-BATCHES: Loop through formatted batches
            for sub_batch_idx, sub_batch in enumerate(formatted_batches):
                # Determine task ID for logging
                # BLIP3o alternates: caption (1), OCR (2), caption (1), OCR (2), ...
                # DocLayNet: all OCR (2)
                task = (sub_batch_idx % 2) + 1 if args.dataset_type == "blip3o" else 2

                # BLIP3o/DocLayNet: Encode images on GPU (moved from task format functions)
                if args.dataset_type in ["blip3o", "doclaynet"] and "images" in sub_batch:
                    # Batch-encode all images at once
                    all_images = sub_batch["images"]  # PIL images
                    image_counts = sub_batch["image_counts"]  # [2, 2, 2, ...] (each sample has 2 images)
                    response_texts = sub_batch["response_texts"]
                    orderings = sub_batch["orderings"]
                    tokenizer = sub_batch["tokenizer"]
    
                    # Encode all images in one batch call
                    all_encoded_features = ocr_adapter.images_to_ocr_features(all_images)
    
                    # Assemble final batches
                    batch_input_ids = []
                    batch_labels = []
                    batch_ocr_features = []
    
                    # Qwen chat tokens
                    user_start_ids = torch.tensor([151644, 872, 198], dtype=torch.long)  # <|im_start|>user\n
                    user_end_ids = torch.tensor([151645, 198], dtype=torch.long)  # <|im_end|>\n
                    assistant_start_ids = torch.tensor([151644, 77091, 198], dtype=torch.long)  # <|im_start|>assistant\n
                    assistant_end_ids = torch.tensor([151645, 198], dtype=torch.long)  # <|im_end|>\n
    
                    img_idx = 0
                    for sample_idx, num_images in enumerate(image_counts):
                        # Get encoded features for this sample's images
                        sample_features = []
                        for _ in range(num_images):
                            if isinstance(all_encoded_features, tuple):
                                feat = all_encoded_features[0][img_idx]
                                deepstack = [all_encoded_features[1][img_idx]] if len(all_encoded_features) > 1 else []
                                sample_features.append((feat, deepstack))
                            else:
                                sample_features.append(all_encoded_features[img_idx])
                            img_idx += 1
    
                        # Qwen3-VL official format: All images in ONE user block
                        # <|im_start|>user<img1><img2><|im_end|><|im_start|>assistant<response><|im_end|>
    
                        # Start ONE user block
                        sequence_parts = [user_start_ids]
                        label_parts = [torch.full_like(user_start_ids, -100)]
    
                        combined_feats = []
                        combined_deepstack = []
    
                        # Add all vision tokens to ONE user block
                        for feat_data in sample_features:
                            if isinstance(feat_data, tuple):
                                feat, deepstack = feat_data
                            else:
                                feat = feat_data
                                deepstack = []
    
                            # Create vision token block
                            num_tokens = feat.shape[0]
                            vision_block = "<|vision_start|>" + "<|image_pad|>" * num_tokens + "<|vision_end|>"
                            vision_ids = tokenizer(vision_block, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                            sequence_parts.append(vision_ids)
                            label_parts.append(torch.full_like(vision_ids, -100))
    
                            combined_feats.append(feat)
                            combined_deepstack.extend(deepstack)
    
                        # Close user block
                        sequence_parts.append(user_end_ids)
                        label_parts.append(torch.full_like(user_end_ids, -100))

                        # Response text + EOS token
                        response = response_texts[sample_idx]
                        response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
                        eos_token = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)  # <|im_end|> (151645)
                        response_ids = torch.cat([response_ids, eos_token], dim=0)  # Add EOS so model learns when to stop

                        # Assistant block
                        sequence_parts.extend([assistant_start_ids, response_ids])
                        label_parts.extend([
                            torch.full_like(assistant_start_ids, -100),
                            response_ids.clone()  # Compute loss on response + EOS
                        ])

                        # Concatenate
                        full_input_ids = torch.cat(sequence_parts, dim=0)
                        labels = torch.cat(label_parts, dim=0)

                        batch_input_ids.append(full_input_ids)
                        batch_labels.append(labels)
    
                        # Combine OCR features for this sample
                        if combined_deepstack:
                            ocr_features = (combined_feats, combined_deepstack)
                        else:
                            ocr_features = combined_feats
                        batch_ocr_features.append(ocr_features)
    
                    # Pad sequences
                    max_len = max(ids.size(0) for ids in batch_input_ids)
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
                    padded_input_ids = []
                    padded_labels = []
                    attention_masks = []
    
                    for input_ids, labels in zip(batch_input_ids, batch_labels):
                        seq_len = input_ids.size(0)
                        padding_len = max_len - seq_len
    
                        padded_input_ids.append(
                            torch.cat([input_ids, torch.full((padding_len,), pad_token_id, dtype=input_ids.dtype)])
                        )
                        padded_labels.append(
                            torch.cat([labels, torch.full((padding_len,), -100, dtype=labels.dtype)])
                        )
                        attention_masks.append(
                            torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)])
                        )
    
                    # Combine all OCR features
                    final_feats_combined = []
                    deepstack_feats_combined = []
                    for ocr_feat in batch_ocr_features:
                        if isinstance(ocr_feat, tuple):
                            final, ds = ocr_feat
                            final_feats_combined.extend(final)
                            deepstack_feats_combined.extend(ds)
                        else:
                            final_feats_combined.extend(ocr_feat)
    
                    combined_ocr_features = (final_feats_combined, deepstack_feats_combined) if deepstack_feats_combined else final_feats_combined
    
                    # Prepare final batch
                    input_ids = torch.stack(padded_input_ids)
                    attention_mask = torch.stack(attention_masks)
                    labels = torch.stack(padded_labels)
                    ocr_features = combined_ocr_features
                    latent_supervision = None
                    latent_positions = None

                # LLaVA: Encode images and immediately forward (no reassembly delay)
                elif args.dataset_type == "llava" and "images_to_encode" in sub_batch:
                    # Flatten all images for batch encoding
                    all_images = []
                    for sample_images in sub_batch["images_to_encode"]:
                        all_images.extend(sample_images)

                    # Batch-encode all images - returns list of features
                    all_encoded = ocr_adapter.images_to_ocr_features(all_images)

                    # Prepare OCR features (model accepts list of features directly)
                    if isinstance(all_encoded, tuple):
                        final_feats, deepstack_feats = all_encoded
                        ocr_features = (final_feats, deepstack_feats)
                    else:
                        ocr_features = all_encoded

                    # Get sequences from collate_fn
                    input_ids = sub_batch["input_ids"]
                    attention_mask = sub_batch["attention_mask"]
                    labels = sub_batch["labels"]
                    latent_supervision = None
                    latent_positions = None

                # DocLayNet: Encode document images
                elif args.dataset_type == "doclaynet" and "images_to_encode" in sub_batch:
                    # Batch-encode all document images
                    all_encoded = ocr_adapter.images_to_ocr_features(sub_batch["images_to_encode"])

                    # Prepare OCR features
                    if isinstance(all_encoded, tuple):
                        final_feats, deepstack_feats = all_encoded
                        ocr_features = (final_feats, deepstack_feats)
                    else:
                        ocr_features = all_encoded

                    # Get sequences from collate_fn
                    input_ids = sub_batch["input_ids"]
                    attention_mask = sub_batch["attention_mask"]
                    labels = sub_batch["labels"]
                    latent_supervision = None
                    latent_positions = None

                # Thinking: Encode real images for context + use pre-encoded thinking text
                elif args.dataset_type == "thinking" and "images" in sub_batch:
                    # Encode real images from dataset (e.g., COCO, GQA, etc.)
                    all_images = sub_batch["images"]
                    all_encoded = ocr_adapter.images_to_ocr_features(all_images)

                    # Prepare OCR features for real images
                    if isinstance(all_encoded, tuple):
                        final_feats, deepstack_feats = all_encoded
                        ocr_features = (final_feats, deepstack_feats)
                    else:
                        ocr_features = all_encoded

                    # Get sequences and thinking-specific data from collate_fn
                    input_ids = sub_batch["input_ids"]
                    attention_mask = sub_batch["attention_mask"]
                    labels = sub_batch["labels"]
                    latent_supervision = sub_batch["latent_supervision"]
                    latent_positions = sub_batch["latent_positions"]

                else:
                    # Other datasets: use pre-encoded features from batch
                    ocr_features = sub_batch["ocr_image_features"]
                    input_ids = sub_batch["input_ids"]
                    attention_mask = sub_batch["attention_mask"]
                    labels = sub_batch["labels"]
                    latent_supervision = None
                    latent_positions = None
    
                # Move to device
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)

                batch_size = input_ids.size(0)
                total_samples += batch_size

                # Forward pass
                with autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                    # Check if thinking data is available
                    has_thinking = latent_supervision is not None and latent_positions is not None

                    if has_thinking:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            ocr_image_features=ocr_features,
                            latent_supervision=latent_supervision,
                            latent_positions=latent_positions.to(args.device),
                            thinking_loss_weight=args.thinking_loss_weight,
                        )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            ocr_image_features=ocr_features,
                        )
                    loss = outputs.loss / args.gradient_accumulation_steps

                # Numerical stability check - detect NaN/Inf before backward pass
                # CRITICAL: Must always do backward() to maintain NCCL sync across DDP ranks
                batch_is_valid = True
                if not torch.isfinite(loss):
                    logger.warning(f"⚠️  Step {global_step}: Non-finite loss detected: {loss.item()}")
                    logger.warning(f"   Input IDs shape: {input_ids.shape}, Labels shape: {labels.shape}")
                    loss = loss * 0.0  # Zero out loss but still do backward for DDP sync
                    batch_is_valid = False

                # Backward pass - MUST ALWAYS EXECUTE to maintain DDP gradient synchronization
                # If we skip backward(), this rank won't participate in NCCL ALLREDUCE and other ranks will hang
                if args.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulation_counter += 1

                # Track loss only if batch was valid
                if batch_is_valid:
                    loss_value = loss.detach().item() * args.gradient_accumulation_steps
                    if math.isfinite(loss_value):
                        running_loss += loss_value
                        task_losses[task].append(loss_value)
                    else:
                        logger.warning(f"⚠️  Step {global_step}: Non-finite loss value after backward")
                        batch_is_valid = False
    
            # Update weights after accumulation_steps (after processing all sub-batches)
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

            # Learning rate scheduling (after optimizer step)
            if scheduler is not None:
                # Warmup: linearly increase LR from 0 to args.lr over warmup_steps
                if args.lr_warmup_steps > 0 and global_step < args.lr_warmup_steps:
                    warmup_lr = args.lr * (global_step / args.lr_warmup_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                else:
                    # After warmup, use cosine annealing
                    scheduler.step()

            accumulation_counter = 0
            global_step += 1
            steps_in_epoch += 1  # Increment epoch step counter

            # Memory cleanup: Clear prefetch batches and intermediate tensors after gradient update
            # This prevents memory accumulation across training steps
            del prefetch_batches
            # Periodic GPU cache cleanup (every 10 steps to avoid overhead)
            if global_step % 10 == 0:
                torch.cuda.empty_cache()

        # Enhanced logging (only on rank 0)
        if global_step % args.log_interval == 0 and is_main_process() and accumulation_counter == 0:
            avg_loss = running_loss / (args.log_interval * args.gradient_accumulation_steps)
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

            # ETA
            elapsed_time = time.time() - epoch_start_time
            steps_remaining_total = expected_total_steps - global_step
            # Use steps_in_epoch (not global_step) for time-per-step calculation
            eta = (elapsed_time / steps_in_epoch) * steps_remaining_total if steps_in_epoch > 0 else 0

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
                f"ETA: {eta/60:.1f}min"
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

            if args.use_swanlab and HAS_SWANLAB:
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
                    log_dict["train/task2_ocr_transcription_loss"] = avg_task2_loss

                swanlab.log(log_dict, step=global_step)

            running_loss = 0.0
            grad_norm_sum = 0.0
            grad_norm_count = 0
            task_losses = {1: [], 2: []}  # Reset task-specific losses
            step_start_time = time.time()

        # Checkpointing (only on rank 0, with barrier)
        if global_step % args.save_interval == 0 and accumulation_counter == 0:
            save_checkpoint(model, optimizer, scaler, global_step, args.output_dir, args, scheduler=scheduler, ocr_adapter=ocr_adapter, tokenizer=tokenizer)
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
        # More accurate description based on what's actually being trained
        if args.use_lora:
            logger.info("Stage: Instruction Tuning (LoRA + Connectors)")
        else:
            logger.info("Stage: Connector Alignment")
        logger.info("=" * 70)
        logger.info(f"Distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    device = torch.device(f'cuda:{local_rank}')
    args.device = device  # Update args.device to use local_rank

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path, trust_remote_code=True)
    # Qwen3-VL already has vision tokens: <|vision_start|>, <|vision_end|>, <|image_pad|>

    # Add special tokens for thinking-with-latent-tokens training
    special_tokens = {"additional_special_tokens": ["<think>", "</think>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    if is_main_process() and num_added > 0:
        logger.info(f"✓ Added {num_added} special tokens: <think>, </think>")

    # Load model
    model = OCRQwen3VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Enable Flash Attention for 1.5-2x speedup
    )

    # Resize embeddings if special tokens were added
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        if is_main_process():
            logger.info(f"✓ Resized token embeddings to {len(tokenizer)}")

    # Store special token IDs for dataset use
    args.think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    args.think_end_id = tokenizer.convert_tokens_to_ids("</think>")

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
    # IMPORTANT: Use nn.ModuleDict so connectors are registered in model.parameters()
    model.model._ocr_deepstack_connectors = nn.ModuleDict({
        '1024': model.model._init_ocr_connector(
            in_dim=1024,
            device=device,
            dtype=torch.bfloat16
        )
    })
    logger.info(f"  ✓ Created deepstack connector: 1024 → {target_dim}")

    # Load checkpoint if resuming
    resume_global_step = 0
    optimizer_state_to_load = None
    scaler_state_to_load = None
    scheduler_state_to_load = None

    if args.load_checkpoint:
        # Load connectors and LoRA adapters if present
        model, optimizer_state_to_load, scaler_state_to_load, scheduler_state_to_load, resume_global_step = load_checkpoint_with_lora(
            model, args.load_checkpoint, args
        )
        if is_main_process():
            logger.info(f"  ✓ Resuming from step {resume_global_step}")

    # Apply LoRA to LLM if enabled (and not already loaded from checkpoint)
    if args.use_lora and not (args.load_checkpoint and (Path(args.load_checkpoint) / "lora_adapters").exists()):
        model = apply_lora_to_llm(model, args)

    # Unfreeze connectors immediately (works for both regular and LoRA models)
    unfrozen_params = unfreeze_connectors(model)
    trainable_params = count_trainable_params(model)
    if is_main_process():
        logger.info(f"Total trainable parameters: {trainable_params:,}")

    # Enable gradient checkpointing if requested (before DDP wrapping)
    if args.use_gradient_checkpointing:
        # Get base model (unwrap LoRA if needed)
        model_to_checkpoint = model.base_model.model if hasattr(model, 'peft_config') else model

        # Enable gradient checkpointing on the LLM
        if hasattr(model_to_checkpoint, 'gradient_checkpointing_enable'):
            model_to_checkpoint.gradient_checkpointing_enable()
            if is_main_process():
                logger.info("✓ Enabled gradient checkpointing (saves ~30-40% memory, ~20% slower)")
        else:
            if is_main_process():
                logger.warning("⚠️  Model does not support gradient_checkpointing_enable, skipping")

    # Wrap model with DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main_process():
            logger.info(f"✓ Wrapped model with DistributedDataParallel")

    # Create dataset FIRST (before loading CUDA models)
    # This avoids conflicts between CUDA model loading and HuggingFace dataset caching
    if args.dataset_type == "blip3o":
        if is_main_process():
            logger.info("Loading BLIP3o dataset...")

        from OCRVL.data.blip3o_dataset import BLIP3oDataset

        blip3o_dataset = BLIP3oDataset(
            dataset_type=args.blip3o_dataset,
            base_path=args.blip3o_base_path,
            mix_ratio=args.blip3o_mix_ratio,
            sample_percentage=args.blip3o_sample_percentage,
            num_proc=args.num_workers,
            seed=args.seed,
        )

        if is_main_process():
            logger.info(f"✓ BLIP3o dataset loaded: {len(blip3o_dataset):,} samples")

        # Check if mixing with DocLayNet
        if args.blip3o_mix_doclaynet:
            if is_main_process():
                logger.info("Loading DocLayNet for mixed training...")

            from OCRVL.data.doclaynet_dataset import DocLayNetOCRDataset
            from OCRVL.data.mixed_alignment_dataset import MixedAlignmentDataset

            doclaynet_dataset = DocLayNetOCRDataset(
                split=args.doclaynet_split,
                data_dir=args.doclaynet_data_dir,
                max_text_length=3072,
                seed=args.seed,
            )

            if is_main_process():
                logger.info(f"✓ DocLayNet dataset loaded: {len(doclaynet_dataset):,} samples")
                logger.info("Creating mixed alignment dataset...")

            # Create mixed dataset
            dataset = MixedAlignmentDataset(
                blip3o_dataset=blip3o_dataset,
                doclaynet_dataset=doclaynet_dataset,
                seed=args.seed,
            )

            if is_main_process():
                logger.info(f"✓ Mixed dataset created: {len(dataset):,} total samples")
        else:
            # Use BLIP3o only
            dataset = blip3o_dataset

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

        # Create custom collate function for BLIP3o (SIMPLE - task logic in training loop)
        from OCRVL.data.blip3o_collate import create_blip3o_collate_fn

        # Configure task info for logging
        if args.enable_match_task:
            task_info = "Task 1/2/3 (with image-text matching)"
        else:
            task_info = "Task 1/2 only (matching disabled)"

        blip3o_collate_fn = create_blip3o_collate_fn(
            tokenizer=tokenizer,
            ocr_adapter=ocr_adapter,
        )
        if is_main_process():
            logger.info(f"✓ Created BLIP3o collate function (simple, task logic in training loop)")
            logger.info(f"  Task configuration: {task_info}")

    elif args.dataset_type == "llava":
        # Detect dataset name from JSON path
        json_filename = Path(args.llava_json_path).stem
        if "665k" in json_filename.lower() or "mix665k" in json_filename.lower():
            dataset_name = "LLaVA v1.5 Mix-665K"
        elif "150k" in json_filename.lower():
            dataset_name = "LLaVA-Instruct-150K"
        else:
            dataset_name = f"LLaVA ({json_filename})"

        if is_main_process():
            logger.info(f"Loading {dataset_name} dataset...")

        from OCRVL.data.llava_instruct_dataset import LLaVAInstructDataset, create_llava_collate_fn

        dataset = LLaVAInstructDataset(
            json_path=args.llava_json_path,
            image_dir=args.llava_image_dir,
            render_questions=args.llava_render_questions,
            seed=args.seed,
        )

        if is_main_process():
            render_mode = "RENDER=1 (questions as images)" if args.llava_render_questions else "RENDER=0 (text prompts)"
            logger.info(f"✓ Dataset loaded: {len(dataset)} samples ({render_mode})")

        # Load OCR adapter on ALL ranks (needed for batched encoding in collate_fn)
        # Each rank needs its own encoder for parallel processing
        if is_main_process():
            logger.info("Initializing OCR adapter for batched encoding in collate_fn...")

        ocr_adapter = Qwen3VLOCRTextAdapter(
            encoder_model_path=args.dpsk_model_path,
            device=args.device,
            use_deepstack=True
        )

        if is_main_process():
            logger.info("✓ OCR adapter initialized")
            logger.info("✓ Batched encoding: 8 GPUs × batch_size images encoded in parallel")

        # Create LLaVA collate function (CPU-only, no encoding)
        llava_collate_fn = create_llava_collate_fn(
            tokenizer=tokenizer,
        )
        if is_main_process():
            logger.info(f"✓ Created LLaVA collate function (CPU-only, encoding moved to training loop)")
            logger.info(f"  Render mode: {'RENDER=1 (questions as images)' if args.llava_render_questions else 'RENDER=0 (text prompts)'}")

    elif args.dataset_type == "thinking":
        # LLaVA-CoT dataset for thinking-with-latent-tokens training
        if is_main_process():
            logger.info(f"Loading LLaVA-CoT dataset for thinking training...")
            logger.info(f"  JSONL: {args.thinking_jsonl_path}")
            logger.info(f"  Image dir: {args.thinking_image_dir}")

        # Load OCR adapter for encoding thinking text
        ocr_adapter = Qwen3VLOCRTextAdapter(
            encoder_model_path=args.dpsk_model_path,
            device=args.device,
            use_deepstack=True
        )

        if is_main_process():
            logger.info("✓ OCR adapter initialized for thinking text encoding")

        from OCRVL.data.thinking_dataset import LLaVACoTDataset, llava_cot_collate_fn

        dataset = LLaVACoTDataset(
            jsonl_path=args.thinking_jsonl_path,
            image_base_dir=args.thinking_image_dir,
            tokenizer=tokenizer,
            ocr_adapter=ocr_adapter,
            max_samples=args.thinking_max_samples,
        )

        if is_main_process():
            logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
            logger.info(f"  Thinking loss weight: {args.thinking_loss_weight}")

        # Create collate function
        thinking_collate_fn = lambda batch: llava_cot_collate_fn(batch, ocr_adapter=ocr_adapter)

        if is_main_process():
            logger.info(f"✓ Created thinking collate function")

    elif args.dataset_type == "doclaynet":
        # DocLayNet dataset for document OCR alignment training
        if is_main_process():
            logger.info(f"Loading DocLayNet dataset for document OCR training...")
            logger.info(f"  Data dir: {args.doclaynet_data_dir}")
            logger.info(f"  Split: {args.doclaynet_split}")

        from OCRVL.data.doclaynet_dataset import DocLayNetOCRDataset

        dataset = DocLayNetOCRDataset(
            split=args.doclaynet_split,
            data_dir=args.doclaynet_data_dir,
            max_text_length=3072,
            seed=args.seed,
        )

        if is_main_process():
            logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
            logger.info("  Note: DocLayNet will be processed through BLIP3o task formatters")
            logger.info("  (Document image → layout description, same as image captioning)")

        # Load OCR adapter for BLIP3o-style processing
        ocr_adapter = Qwen3VLOCRTextAdapter(
            encoder_model_path=args.dpsk_model_path,
            device=args.device,
            use_deepstack=True
        )

        if is_main_process():
            logger.info("✓ OCR adapter initialized")

        # Use BLIP3o collate function (DocLayNet is just image-caption pairs)
        from OCRVL.data.blip3o_collate import create_blip3o_collate_fn

        blip3o_collate_fn = create_blip3o_collate_fn(
            tokenizer=tokenizer,
            ocr_adapter=ocr_adapter,
        )

        if is_main_process():
            logger.info(f"✓ Using BLIP3o collate function for DocLayNet")

    else:  # custom
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
    # All datasets now support num_workers > 0 (encoding moved to training loop)
    dataloader_num_workers = args.num_workers

    # Use DistributedSampler for multi-GPU training
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    shuffle = (sampler is None)  # Only shuffle if not using sampler

    # Select appropriate collate function
    if args.dataset_type == "blip3o":
        active_collate_fn = blip3o_collate_fn  # Custom collate with task logic
    elif args.dataset_type == "llava":
        active_collate_fn = llava_collate_fn  # LLaVA collate with random ordering
    elif args.dataset_type == "thinking":
        active_collate_fn = thinking_collate_fn  # Thinking collate with CoT processing
    elif args.dataset_type == "doclaynet":
        active_collate_fn = blip3o_collate_fn  # Use BLIP3o collate (DocLayNet is image-caption pairs)
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
        if args.dataset_type == "blip3o":
            sample_pct_str = f"{args.blip3o_sample_percentage*100:.2f}%" if args.blip3o_sample_percentage else "100% (full dataset, using cached)"
            logger.info(f"  Dataset: BLIP3o {args.blip3o_dataset} - {sample_pct_str}")
        elif args.dataset_type == "llava":
            logger.info(f"  Dataset: LLaVA-Instruct-150K")
            logger.info(f"  Render mode: {'RENDER=1 (questions as images)' if args.llava_render_questions else 'RENDER=0 (text prompts)'}")
        elif args.dataset_type == "doclaynet":
            logger.info(f"  Dataset: DocLayNet {args.doclaynet_split} split ({dataset_size:,} samples)")
            logger.info(f"  Data dir: {args.doclaynet_data_dir}")
        logger.info(f"=" * 70)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True  # Use fused kernel for 5-10% speedup
    )

    # Restore optimizer state if resuming
    if optimizer_state_to_load is not None:
        optimizer.load_state_dict(optimizer_state_to_load)
        if is_main_process():
            logger.info("  ✓ Restored optimizer state")

    # GradScaler - not needed for bfloat16, only for float16
    # Since model is in bfloat16, disable GradScaler
    scaler = GradScaler('cuda', enabled=False)

    # Restore scaler state if resuming
    if scaler_state_to_load is not None:
        scaler.load_state_dict(scaler_state_to_load)
        if is_main_process():
            logger.info("  ✓ Restored scaler state")

    if is_main_process():
        logger.info("  GradScaler disabled (model uses bfloat16, not float16)")

    # Learning rate scheduler
    scheduler = None
    scheduler_state_to_load = None
    if args.use_lr_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR

        # Calculate total steps for scheduler
        total_steps = actual_max_steps - resume_global_step

        # Calculate warmup steps from ratio
        warmup_steps = int(total_steps * args.warmup_ratio)

        if is_main_process():
            logger.info("=" * 70)
            logger.info("Learning Rate Scheduler")
            logger.info("=" * 70)
            logger.info(f"  Type: CosineAnnealingLR")
            logger.info(f"  Initial LR: {args.lr}")
            logger.info(f"  Min LR (eta_min): {args.lr_scheduler_min_lr}")
            logger.info(f"  Total steps: {total_steps}")
            if warmup_steps > 0:
                logger.info(f"  Warmup ratio: {args.warmup_ratio} ({warmup_steps} steps)")
                logger.info(f"  Warmup will linearly increase LR from 0 to {args.lr}")
            logger.info("=" * 70)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.lr_scheduler_min_lr
        )

        # Store warmup_steps in args for use in training loop
        args.lr_warmup_steps = warmup_steps

        # Load scheduler state if available from checkpoint
        if scheduler_state_to_load is not None:
            scheduler.load_state_dict(scheduler_state_to_load)
            if is_main_process():
                logger.info("  ✓ Restored scheduler state from checkpoint")

    else:
        args.lr_warmup_steps = 0  # No warmup if scheduler disabled
        if is_main_process():
            logger.info("  Learning rate scheduler: Disabled (constant LR)")


    # Training loop
    import time
    training_start_time = time.time()
    global_step = resume_global_step  # Resume from checkpoint if applicable

    # Save initial checkpoint at step 0 to validate checkpoint saving works
    # This prevents wasting hours of training if checkpoint saving is broken
    if is_main_process():
        logger.info("=" * 70)
        logger.info("Saving initial checkpoint for validation...")
        logger.info("=" * 70)
    save_checkpoint(model, optimizer, scaler, global_step, args.output_dir, args, scheduler=scheduler, ocr_adapter=ocr_adapter, tokenizer=tokenizer)
    if is_main_process():
        checkpoint_dir = Path(args.output_dir) / f"step_{global_step}"
        logger.info(f"✓ Initial checkpoint saved: {checkpoint_dir}")
        logger.info(f"  Please verify checkpoint files exist before training continues:")
        logger.info(f"    - {checkpoint_dir}/connectors.pt")
        logger.info(f"    - {checkpoint_dir}/training_state.pt")
        logger.info("=" * 70)
    # Synchronize all processes after initial checkpoint
    if dist.is_initialized():
        dist.barrier()

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
            actual_max_steps, epoch + 1, total_steps, tokenizer, ocr_adapter,
            scheduler=scheduler
        )

        if global_step >= actual_max_steps:
            break

    # Final save
    save_checkpoint(model, optimizer, scaler, global_step, args.output_dir, args, checkpoint_name="step_latest", scheduler=scheduler, ocr_adapter=ocr_adapter, tokenizer=tokenizer)

    # Training completion summary
    total_training_time = time.time() - training_start_time
    if is_main_process():
        logger.info(f"=" * 70)
        logger.info(f"Training Complete!")
        logger.info(f"=" * 70)
        logger.info(f"  Total steps: {global_step}")
        logger.info(f"  Total time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
        logger.info(f"  Average time per step: {total_training_time/max(global_step,1):.2f} seconds")
        logger.info(f"  Final checkpoint: {args.output_dir}/step_latest (step {global_step})")
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
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Enable Flash Attention for 1.5-2x speedup
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
    parser.add_argument("--dataset-type", type=str, default=None,
                       choices=["blip3o", "llava", "thinking", "doclaynet", "custom"],
                       help="Dataset type: blip3o, llava, thinking, doclaynet, or custom (default: auto-detect from other args)")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data (text file or jsonl) - used when dataset-type=custom")

    # BLIP3o dataset configuration
    parser.add_argument("--blip3o_dataset", type=str, default="long",
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
    parser.add_argument("--blip3o_mix_doclaynet", action="store_true", default=False,
                       help="Mix DocLayNet dataset with BLIP3o for alignment training (trains on both in 1 epoch)")
    parser.add_argument("--enable_feature_cache", action="store_true", default=False,
                       help="Enable in-memory caching of encoded features (uses ~3GB RAM per GPU, speeds up epoch 2+)")
    parser.add_argument("--enable_match_task", action="store_true", default=False,
                       help="Enable image-text matching task. Disabled by default to prevent overfitting.")

    # LLaVA-Instruct-150K dataset configuration
    parser.add_argument("--llava_json_path", type=str,
                       default="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_instruct_150k.json",
                       help="Path to llava_instruct_150k.json")
    parser.add_argument("--llava_image_dir", type=str,
                       default="/share/project/xiyan/huggingface",
                       help="Base directory containing datasets (e.g., huggingface root). For 665K, images span coco/vg/gqa/textvqa/ocr_vqa.")
    parser.add_argument("--llava_image_index", type=str, default=None,
                       help="Optional JSON mapping from dataset-relative image path (e.g. 'coco/train2017/xxxx.jpg') to absolute local file path.")
    parser.add_argument("--llava_render_questions", action="store_true", default=True,
                       help="Render questions as images (RENDER=1). Use --no-llava_render_questions for RENDER=0.")
    parser.add_argument("--no-llava_render_questions", dest="llava_render_questions", action="store_false",
                       help="Use text prompts instead of rendering (RENDER=0)")

    # Thinking (LLaVA-CoT) dataset configuration
    parser.add_argument("--thinking_jsonl_path", type=str,
                       default="/share/project/xiyan/huggingface/Xkev/LLaVA-CoT-100k/train.jsonl",
                       help="Path to LLaVA-CoT JSONL file")
    parser.add_argument("--thinking_image_dir", type=str,
                       default="/share/project/xiyan/huggingface",
                       help="Base directory containing LLaVA-CoT images")
    parser.add_argument("--thinking_loss_weight", type=float, default=1.0,
                       help="Weight for thinking alignment loss (MSE between projected hidden states and OCR-encoded reasoning)")
    parser.add_argument("--thinking_max_samples", type=int, default=None,
                       help="Maximum number of samples to load from thinking dataset (None = all)")

    # DocLayNet dataset configuration
    parser.add_argument("--doclaynet_data_dir", type=str,
                       default="/share/project/xiyan/huggingface/docling-project/DocLayNet",
                       help="Root directory of extracted DocLayNet_core.zip (contains COCO/ and PNG/ subdirectories)")
    parser.add_argument("--doclaynet_split", type=str, default="train",
                       choices=["train", "val", "test"],
                       help="DocLayNet split: train (69,375), val (6,489), or test (4,999)")

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

    # Learning rate scheduler
    parser.add_argument("--use_lr_scheduler", action="store_true", default=False,
                       help="Enable cosine annealing learning rate scheduler")
    parser.add_argument("--lr_scheduler_min_lr", type=float, default=0.0,
                       help="Minimum learning rate for cosine annealing (eta_min)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="Ratio of total steps for linear warmup (e.g., 0.03 = 3%% of training)")

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
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Path to checkpoint directory to resume training from (e.g., OCRVL/checkpoints/.../step_786)")
    parser.add_argument("--resume_training_state", action="store_true", default=False,
                       help="Resume training state (optimizer, scaler, step counter) from checkpoint. "
                            "By default (False), only loads model weights (connectors/LoRA). "
                            "Enable this to continue training from exact same point.")

    # Checkpoint evaluation (optional, for training transparency)
    parser.add_argument("--enable_checkpoint_eval", action="store_true", default=False,
                       help="Enable checkpoint evaluation: run inference on fixed VQA samples during checkpoint saves")
    parser.add_argument("--checkpoint_eval_samples", type=str,
                       default="OCRVL/data/eval_samples.json",
                       help="Path to JSON file containing evaluation samples")
    parser.add_argument("--checkpoint_eval_max_samples", type=int, default=12,
                       help="Maximum number of evaluation samples to run per checkpoint (default: 12, includes 10 VQA + 2 DocLayNet OCR)")

    # LoRA configuration (optional, disabled by default)
    parser.add_argument("--use_lora", action="store_true", default=False,
                       help="Enable LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. "
                            "Only applies to Qwen3-VL LLM, not DPSK encoder or connectors.")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank (default: 8). Higher rank = more parameters but better expressiveness.")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha scaling factor (default: 16). Typically 2x lora_r.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout rate (default: 0.05)")
    parser.add_argument("--lora_target_modules", type=str, default=None,
                       help="Comma-separated list of modules to apply LoRA to (default: all attention+MLP layers). "
                            "Example: 'q_proj,k_proj,v_proj,o_proj'")

    # Gradient checkpointing
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False,
                       help="Enable gradient checkpointing to reduce memory usage (~30-40% less memory, ~20% slower training). "
                            "Recommended for large batch sizes or limited GPU memory.")

    # System
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device (will be overridden by local_rank in distributed training)")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="Number of data loading workers (default: 16 for parallel loading)")

    args = parser.parse_args()

    # Auto-detect dataset type if not specified
    if args.dataset_type is None:
        # Auto-detect based on which arguments are provided
        # Priority: explicit paths > defaults
        if args.data_path is not None:
            args.dataset_type = "custom"
        elif Path(args.llava_json_path).exists():
            args.dataset_type = "llava"
        else:
            args.dataset_type = "blip3o"  # Default

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
        if args.dataset_type == "llava":
            dataset_name = "llava"
        elif args.dataset_type == "blip3o":
            dataset_name = args.blip3o_dataset
        else:
            dataset_name = "custom"
        experiment_setting = f"{args.stage}_{dataset_name}"
        args.output_dir = f"OCRVL/checkpoints/{experiment_setting}_{timestamp}"

    # Create output directory (only on rank 0)
    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup file logging (mode='w' to truncate existing log)
        file_handler = logging.FileHandler(Path(args.output_dir) / "training.log", mode='w')
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
        if HAS_SWANLAB:
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
        else:
            logger.warning("swanlab not installed, skipping swanlab logging")

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
