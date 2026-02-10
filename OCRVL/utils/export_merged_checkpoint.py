#!/usr/bin/env python3
"""
Export Merged OCRQwen3VL + LoRA Checkpoint

This script merges the LoRA adapter weights with the base OCRQwen3VL model
and exports it as a complete checkpoint that can be loaded directly in vLLM.

The merged checkpoint can then be used with:
    python eval_ocrqwen3vl.py --model-type ocrqwen3vl --checkpoint <merged_path> --lora-path None

Usage:
    python export_merged_checkpoint.py \\
        --base-model OCRVL/checkpoints/OCR-Qwen3-VL-2B \\
        --lora-adapter OCRVL/checkpoints/llamafactory/qwen3vl-2b/lora/run_20260120_045134/checkpoint-8607 \\
        --output OCRVL/checkpoints/OCRQwen3VL-2B-merged-lora
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


def export_merged_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
):
    """
    Merge LoRA adapter with base model and export as complete checkpoint.

    This creates a standalone checkpoint that doesn't require vLLM's LoRA loading mechanism,
    working around the "modules_to_save not supported" limitation.
    """
    logger.info("=" * 80)
    logger.info("Exporting Merged OCRQwen3VL + LoRA Checkpoint")
    logger.info("=" * 80)
    logger.info(f"Base model:    {base_model_path}")
    logger.info(f"LoRA adapter:  {lora_adapter_path}")
    logger.info(f"Output:        {output_path}")
    logger.info("")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base model - use Qwen3VLForConditionalGeneration for vision-language model
    logger.info("Loading base OCRQwen3VL model...")
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU to avoid GPU conflicts
        trust_remote_code=True,
    )
    logger.info(f"  ✓ Model loaded: {model.config.model_type}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    logger.info("  ✓ Tokenizer loaded")

    # Load LoRA and merge
    logger.info("Loading LoRA adapter...")
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        model,
        lora_adapter_path,
        is_trainable=False,
    )

    logger.info("  ✓ LoRA loaded")

    # Merge LoRA weights into base model
    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    logger.info("  ✓ LoRA weights merged")

    # Save merged model
    logger.info(f"\nSaving merged model to {output_path}...")
    model.save_pretrained(output_path, max_shard_size="5GB")
    tokenizer.save_pretrained(output_path)
    logger.info("  ✓ Model and tokenizer saved")

    # Copy processor config files from base model (required for vLLM)
    base_model_dir = Path(base_model_path)
    processor_files = [
        "preprocessor_config.json",
        "processor_config.json",
    ]
    for fname in processor_files:
        src_file = base_model_dir / fname
        if src_file.exists():
            shutil.copy(src_file, output_path / fname)
            logger.info(f"  ✓ Copied {fname}")

    # Copy LoRA config for reference
    src_config = Path(lora_adapter_path) / "adapter_config.json"
    if src_config.exists():
        shutil.copy(src_config, output_path / "lora_adapter_config.json")
        logger.info("  ✓ LoRA config copied for reference")

    # Save metadata
    metadata = {
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "merge_type": "peft_merge",
        "model_type": model.config.model_type,
    }
    with open(output_path / "merge_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ Export complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Merged checkpoint saved to: {output_path}")
    logger.info("")
    logger.info("Usage:")
    logger.info(f"  python eval_ocrqwen3vl.py \\")
    logger.info(f"      --benchmark m3cot \\")
    logger.info(f"      --model-type ocrqwen3vl \\")
    logger.info(f"      --checkpoint {output_path} \\")
    logger.info(f"      --output results/m3cot.jsonl")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Export merged OCRQwen3VL + LoRA checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python export_merged_checkpoint.py \\
      --base-model OCRVL/checkpoints/OCR-Qwen3-VL-2B \\
      --lora-adapter OCRVL/checkpoints/llamafactory/qwen3vl-2b/lora/run_20260120_045134/checkpoint-8607 \\
      --output OCRVL/checkpoints/OCRQwen3VL-2B-merged
        """
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base OCRQwen3VL model",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged checkpoint",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.base_model):
        print(f"❌ Base model not found: {args.base_model}")
        return 1

    if not os.path.exists(args.lora_adapter):
        print(f"❌ LoRA adapter not found: {args.lora_adapter}")
        return 1

    adapter_config = Path(args.lora_adapter) / "adapter_config.json"
    if not adapter_config.exists():
        print(f"❌ LoRA config not found: {adapter_config}")
        return 1

    try:
        export_merged_model(
            base_model_path=args.base_model,
            lora_adapter_path=args.lora_adapter,
            output_path=args.output,
        )
        return 0
    except Exception as e:
        logger.error(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
