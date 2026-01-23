#!/usr/bin/env python3
"""
Standalone Transparent Evaluation Script

Run transparent evaluation on an existing checkpoint without resuming training.
This uses the same logic as TransparentEvalCallback but as a standalone script.

Usage:
    python run_transparent_eval_standalone.py \
        --checkpoint checkpoints/path/run_xyz/checkpoint-311 \
        --output checkpoints/path/run_xyz/eval_results
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

# Add repository root to path
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Import template registration before loading model
from OCRVL.llamafactory.qwen3_vl_ocrvl_template import register_ocrvl_qwen3_vl_template
register_ocrvl_qwen3_vl_template()

from OCRVL.llamafactory.transparent_eval_callback import TransparentEvalCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run transparent evaluation on checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., checkpoints/.../checkpoint-311)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for eval results (default: checkpoint_dir/eval_results)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default="",
        help="Limit number of samples (e.g., '5' for first 5 samples, empty for all)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Output directory
    output_dir = Path(args.output) if args.output else checkpoint_path / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables for callback
    os.environ["OCRVL_ENABLE_TRANSPARENT_EVAL"] = "1"
    os.environ["OCRVL_TRANSPARENT_EVAL_LIMIT"] = args.limit
    os.environ["OCRVL_TRANSPARENT_EVAL_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["OCRVL_TRANSPARENT_EVAL_TEMPERATURE"] = str(args.temperature)
    os.environ["OCRVL_REPO_ROOT"] = str(_REPO_ROOT)

    logger.info("="*80)
    logger.info("Standalone Transparent Evaluation")
    logger.info("="*80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Limit: {args.limit or 'all'}")
    logger.info("")

    # Load model, tokenizer, processor
    logger.info("Loading model and tokenizer...")
    try:
        # For PEFT/Lora checkpoints, load base model first then load adapters
        # Check if this is a PEFT checkpoint (has adapter_config.json)
        adapter_config = checkpoint_path / "adapter_config.json"
        if adapter_config.exists():
            logger.info("Detected PEFT/LoRA checkpoint, loading base model...")
            # Load base model (need to determine from adapter_config)
            with open(adapter_config, 'r') as f:
                adapter_cfg = json.load(f)
            base_model = adapter_cfg.get("base_model_name_or_path", "")
            if not base_model:
                # Try parent directories
                for parent in checkpoint_path.parents:
                    if (parent / "adapter_config.json").exists():
                        continue
                    if "OCR-Qwen3-VL" in str(parent) or "qwen3" in str(parent).lower():
                        base_model = str(parent)
                        break

            if not base_model:
                # Default path
                base_model = str(_REPO_ROOT / "OCRVL/checkpoints/OCR-Qwen3-VL-2B")

            logger.info(f"Loading base model from: {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            # Load LoRA adapters
            logger.info(f"Loading LoRA adapters from: {checkpoint_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint_path)
            model.merge_and_unload()  # Merge LoRA weights for inference
        else:
            # Direct checkpoint
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        logger.info("✓ Model loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Create callback
    callback = TransparentEvalCallback(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
    )

    if not callback.enabled:
        logger.error("Transparent eval callback not enabled. Check environment variables.")
        sys.exit(1)

    # Create fake TrainerState for callback
    from transformers import TrainerState

    # Extract step number from checkpoint path (e.g., "checkpoint-311")
    checkpoint_name = checkpoint_path.name
    if checkpoint_name.startswith("checkpoint-"):
        step = int(checkpoint_name.replace("checkpoint-", ""))
    else:
        step = 0
        logger.warning(f"Could not extract step from checkpoint name: {checkpoint_name}")

    state = TrainerState()
    state.global_step = step

    # Create fake TrainingArguments for callback
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        report_to="none",
    )

    # Run evaluation
    logger.info("Running transparent evaluation...")
    try:
        callback.on_evaluate(
            args=training_args,
            state=state,
            control=None,  # Not needed for eval
        )
        logger.info("")
        logger.info("="*80)
        logger.info(f"✓ Evaluation complete")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*80)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
