#!/usr/bin/env python3
"""
Setup OCR-Qwen3-VL-2B-Thinking checkpoint by combining:
- Qwen3-VL-2B-Thinking language model from huggingface
- DPSK OCR encoder (kept the same)
- OCRVL components (connectors, etc.)
"""

import os
import shutil
import subprocess
from pathlib import Path


def setup_ocr_qwen3_vl_thinking(
    source_instruct: Path,
    source_thinking: Path,
    target_dir: Path,
):
    """
    Create OCR-Qwen3-VL-2B-Thinking checkpoint.

    Args:
        source_instruct: Path to OCR-Qwen3-VL-2B-Instruct (existing)
        source_thinking: Path to Qwen3-VL-2B-Thinking (huggingface)
        target_dir: Path to OCR-Qwen3-VL-2B-Thinking (to create)
    """
    print("=" * 70)
    print("Setting up OCR-Qwen3-VL-2B-Thinking")
    print("=" * 70)
    print()

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Files to copy from Qwen3-VL-2B-Thinking (LLM weights)
    thinking_files = [
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",  # New in Thinking version
    ]

    # Files to keep from existing OCR-Qwen3-VL-2B-Instruct
    ocrvl_files = [
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "dpsk_ocr_encoder",  # Directory
    ]

    print("Step 1: Copying LLM files from Qwen3-VL-2B-Thinking...")
    for fname in thinking_files:
        src = source_thinking / fname
        dst = target_dir / fname

        if not src.exists():
            print(f"  Warning: {fname} not found in {source_thinking}")
            continue

        if fname == "model.safetensors":
            # Use symlink to save space
            dst_link = target_dir / "qwen3_model.safetensors"
            if dst_link.exists():
                dst_link.unlink()
            os.symlink(src, dst_link)
            # Also create model.safetensors symlink
            model_link = target_dir / "model.safetensors"
            if model_link.exists():
                model_link.unlink()
            os.symlink("qwen3_model.safetensors", model_link)
            print(f"  Created symlink: qwen3_model.safetensors -> {src}")
        else:
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")

    print()
    print("Step 2: Copying OCRVL components from existing checkpoint...")

    for fname in ocrvl_files:
        src = source_instruct / fname
        dst = target_dir / fname

        if not src.exists():
            print(f"  Warning: {fname} not found in {source_instruct}")
            continue

        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied directory: {fname}/")
        else:
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")

    print()
    print("Step 3: Verifying tokenizer has think tokens...")

    # Verify tokenizer has think tokens
    tokenizer_config_path = target_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        import json
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)

        added_tokens = tokenizer_config.get("added_tokens_decoder", {})
        think_tokens_found = False
        for token_id, token_obj in added_tokens.items():
            # token_obj is a dict with 'content' key
            token_content = token_obj.get('content', '') if isinstance(token_obj, dict) else str(token_obj)
            if "think" in token_content.lower():
                print(f"  Found think token: {token_content} (ID: {token_id})")
                think_tokens_found = True

        if think_tokens_found:
            print("  ✓ Think tokens verified")
        else:
            print("  Warning: Think tokens not found in tokenizer")

    print()
    print("=" * 70)
    print("Setup complete!")
    print("=" * 70)
    print()
    print(f"Target directory: {target_dir}")
    print()
    print("Key files:")
    print(f"  LLM: qwen3_model.safetensors -> {source_thinking}/model.safetensors")
    print(f"  OCR: dpsk_ocr_encoder/ -> {source_instruct}/dpsk_ocr_encoder/")
    print()
    print("You can now use this checkpoint for training:")
    print("  model_name_or_path: {target_dir}")


def main():
    # Default paths
    script_dir = Path(__file__).parent.parent
    source_instruct = script_dir / "checkpoints" / "OCR-Qwen3-VL-2B"
    source_thinking = Path("/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking")
    target_dir = script_dir / "checkpoints" / "OCR-Qwen3-VL-2B-Thinking"

    # Verify sources exist
    if not source_instruct.exists():
        print(f"Error: Source instruct not found: {source_instruct}")
        return 1

    if not source_thinking.exists():
        print(f"Error: Source thinking not found: {source_thinking}")
        return 1

    try:
        setup_ocr_qwen3_vl_thinking(source_instruct, source_thinking, target_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
