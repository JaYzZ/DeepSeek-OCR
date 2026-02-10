#!/usr/bin/env python3
"""
Export OCRVL checkpoint to HuggingFace-compatible format.

This script merges OCRVL-trained connectors into the base Qwen3-VL model
and saves it as a standalone HuggingFace model that can be used directly
with the Qwen3-VL evaluation scripts.

Usage:
    python OCRVL/evaluation/export_hf_checkpoint.py \\
        --checkpoint OCRVL/checkpoints/alignment_60k_*/step_1000 \\
        --output OCRVL/hf_models/ocrvl_step1000

This creates a fully self-contained model directory that can be used as:
    export MODEL_PATH=OCRVL/hf_models/ocrvl_step1000
    cd ../Qwen3-VL/evaluation
    bash run_all_benchmarks.sh
"""

import argparse
import json
import shutil
import torch
from pathlib import Path
from typing import Optional


def export_ocrvl_to_hf(
    checkpoint_path: Path,
    output_path: Path,
    base_model_path: Optional[Path] = None,
):
    """
    Export OCRVL checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to OCRVL checkpoint directory (with connectors.pt)
        output_path: Path to output HuggingFace model directory
        base_model_path: Path to base Qwen3-VL model (auto-detected if None)
    """
    print("=" * 80)
    print("OCRVL to HuggingFace Checkpoint Exporter")
    print("=" * 80)

    # Validate checkpoint
    connectors_path = checkpoint_path / "connectors.pt"
    if not connectors_path.exists():
        raise FileNotFoundError(f"connectors.pt not found in {checkpoint_path}")

    # Auto-detect base model if not provided
    if base_model_path is None:
        config_path = checkpoint_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found and base_model_path not provided. "
                f"Please specify --base-model-path"
            )

        with open(config_path) as f:
            config = json.load(f)
            base_model_path = Path(config['qwen_model_path'])

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base model: {base_model_path}")
    print(f"Output: {output_path}")
    print()

    # Load OCRVL model
    print("Loading OCRVL model...")
    import sys
    sys.path.insert(0, str(checkpoint_path.parent.parent.parent))
    from OCRVL.model.language_model.ocr_qwen3_vl import OCRQwen3VLForConditionalGeneration
    from transformers import AutoProcessor

    model = OCRQwen3VLForConditionalGeneration.from_pretrained(
        str(base_model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Load processor (includes tokenizer with chat_template)
    print("Loading processor from base model...")
    processor = AutoProcessor.from_pretrained(
        str(base_model_path),
        trust_remote_code=True
    )

    # Load connectors
    print(f"Loading connectors from {connectors_path}...")
    state = torch.load(connectors_path, map_location='cpu')

    if 'ocr_connector' in state and hasattr(model.model, 'ocr_connector'):
        model.model.ocr_connector.load_state_dict(state['ocr_connector'])
        print("  ✓ Loaded ocr_connector")

    if 'deepstack_connectors' in state and hasattr(model.model, '_ocr_deepstack_connectors'):
        for k, v in state['deepstack_connectors'].items():
            if k in model.model._ocr_deepstack_connectors:
                model.model._ocr_deepstack_connectors[k].load_state_dict(v)
        print(f"  ✓ Loaded {len(state['deepstack_connectors'])} deepstack connectors")

    # Save merged model
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving merged model to {output_path}...")

    model.save_pretrained(
        str(output_path),
        safe_serialization=True,  # Use safetensors format
    )

    # Save processor (this includes tokenizer with chat_template)
    print("Saving processor...")
    processor.save_pretrained(str(output_path))
    print("  ✓ Saved processor with chat_template")

    # Save metadata
    metadata = {
        "model_type": "ocrvl-qwen3-vl",
        "base_model": str(base_model_path),
        "checkpoint_path": str(checkpoint_path),
        "exported_from": "OCRVL training checkpoint",
    }

    # Try to load training config
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            training_config = json.load(f)
            metadata["training_config"] = training_config

    metadata_path = output_path / "ocrvl_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved metadata to {metadata_path.name}")

    print()
    print("=" * 80)
    print("✓ Export complete!")
    print("=" * 80)
    print(f"HuggingFace-compatible model saved to: {output_path}")
    print()
    print("Usage:")
    print(f"  export MODEL_PATH={output_path}")
    print("  cd ../Qwen3-VL/evaluation")
    print("  bash run_all_benchmarks.sh")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Export OCRVL checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        required=True,
        help="Path to OCRVL checkpoint directory (e.g., OCRVL/checkpoints/.../step_1000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path for HuggingFace model (e.g., OCRVL/hf_models/ocrvl_step1000)"
    )
    parser.add_argument(
        "--base-model-path", "-b",
        type=Path,
        default=None,
        help="Path to base Qwen3-VL model (auto-detected from config.json if not provided)"
    )

    args = parser.parse_args()

    try:
        export_ocrvl_to_hf(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            base_model_path=args.base_model_path
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
