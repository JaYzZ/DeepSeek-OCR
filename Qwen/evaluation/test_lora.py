#!/usr/bin/env python
"""
Test script to verify LoRA loading works with vLLM and Qwen3-VL.

Usage:
    # Test with base model (no LoRA)
    python test_lora.py --model-path /path/to/base/model

    # Test with LoRA
    python test_lora.py --model-path /path/to/base/model --enable-lora --lora-path /path/to/lora
"""

import os
import sys
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Note: Image preprocessing now handled by vLLM internally

def test_inference(model_path, lora_path=None, lora_name="default"):
    """Test basic inference with or without LoRA."""

    print("\n" + "="*80)
    print("Testing vLLM Inference with LoRA")
    print("="*80)
    print(f"Model: {model_path}")
    if lora_path:
        print(f"LoRA: {lora_path} (name: {lora_name})")
    print("="*80 + "\n")

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("✓ Processor loaded\n")

    # Initialize vLLM
    print("Initializing vLLM...")
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.3,
        "max_model_len": 8192,
        "trust_remote_code": True,
        "disable_log_stats": True,
        "limit_mm_per_prompt": {"image": 1},
    }

    lora_request = None
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        llm_kwargs["max_loras"] = 1
        # Note: lora_modules not passed in initialization, LoRA loaded dynamically via LoRARequest
        print(f"   LoRA enabled: {lora_name} from {lora_path}")

    try:
        llm = LLM(**llm_kwargs)
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Prepare a simple test prompt
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is 2+2? Reply with just the number."}
        ]
    }]

    # Prepare inputs for vLLM - let vLLM handle everything (official approach)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = text + "<|im_start|>assistant\n"

    # Extract raw images from messages - let vLLM handle preprocessing
    raw_images = []
    raw_videos = []

    for item in messages[0].get('content', []):
        if isinstance(item, dict):
            if item.get('type') == 'image':
                raw_images.append(item['image'])  # Can be path, PIL image, or base64
            elif item.get('type') == 'video':
                raw_videos.append(item['video'])

    # Get min/max pixels from processor
    min_pixels = getattr(processor.image_processor, 'min_pixels', 28 * 28 * 256)
    max_pixels = getattr(processor.image_processor, 'max_pixels', 28 * 28 * 2048)

    mm_data = {}
    if raw_images:
        mm_data['image'] = raw_images
    if raw_videos:
        mm_data['video'] = raw_videos

    inputs = [{
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': {
            'min_pixels': min_pixels,
            'max_pixels': max_pixels,
        }
    }]

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=100,
    )

    # Prepare LoRA request
    if lora_path:
        from vllm.v1.engine import LoRARequest
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=1,
            lora_path=lora_path,
        )
        print(f"Testing inference with LoRA: {lora_name}\n")
    else:
        print("Testing inference with base model\n")

    # Run inference
    print("Running inference...")
    try:
        if lora_request:
            outputs = llm.generate(inputs, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate(inputs, sampling_params)

        result = outputs[0].outputs[0].text
        print(f"✓ Inference successful!")
        print(f"\nOutput: {result}\n")

        print("="*80)
        print("✅ LoRA test PASSED")
        print("="*80 + "\n")
        return True

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LoRA with vLLM")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--lora-path", type=str)
    parser.add_argument("--lora-name", type=str, default="test_lora")

    args = parser.parse_args()

    if args.enable_lora and not args.lora_path:
        print("Error: --enable-lora requires --lora-path")
        sys.exit(1)

    success = test_inference(args.model_path, args.lora_path, args.lora_name)
    sys.exit(0 if success else 1)
