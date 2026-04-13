#!/usr/bin/env python3
"""Benchmark QPS: Continuous AR vs Discrete AR for Qwen3VL-2B."""
import os
import time
import argparse

os.environ["VLLM_THINKING"] = "1"

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.v1.engine import LoRARequest

BASE_MODEL = "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking"


def run_benchmark(llm, tokenizer, processor, image, question, num_runs=20, use_continuous_ar=True, lora_request=None):
    """Run benchmark and return QPS."""
    if use_continuous_ar:
        # Continuous AR: WITH <think> in prompt
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": "dummy"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Discrete AR: WITHOUT <think> in prompt
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text = text + "<|im_start|>assistant\n"

    image_inputs, _, video_kwargs = process_vision_info(
        messages if use_continuous_ar else [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}],
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True
    )

    # Warmup
    for _ in range(2):
        _ = llm.generate([{
            'prompt': text,
            'multi_modal_data': {'image': image_inputs} if image_inputs else {},
            'mm_processor_kwargs': video_kwargs,
        }], SamplingParams(max_tokens=256, temperature=0.0), lora_request=lora_request)

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        outputs = llm.generate([{
            'prompt': text,
            'multi_modal_data': {'image': image_inputs} if image_inputs else {},
            'mm_processor_kwargs': video_kwargs,
        }], SamplingParams(max_tokens=256, temperature=0.0), lora_request=lora_request)
    elapsed = time.time() - start

    qps = num_runs / elapsed
    return qps, outputs[0].outputs[0].text[:100]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--lora-path", type=str, default=None)
    args = parser.parse_args()

    print("Loading model...")
    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
        enable_lora=True if args.lora_path else False,
        max_lora_rank=64
    )

    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest(lora_name="test", lora_int_id=1, lora_path=args.lora_path)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Test image
    image = Image.open("/share/project/xiyan/huggingface/liuhaotian/LLaVA-Pretrain/images/00293/002933218.jpg").convert('RGB')
    question = "What's in this image?"

    print(f"\n{'='*60}")
    print("Benchmark: Continuous AR vs Discrete AR")
    print(f"Model: {BASE_MODEL}")
    if args.lora_path:
        print(f"LoRA: {args.lora_path}")
    print(f"Num runs: {args.num_runs}")
    print(f"{'='*60}\n")

    # Discrete AR (no <think>)
    print("Running Discrete AR (no <think>)...")
    qps_discrete, output_discrete = run_benchmark(
        llm, tokenizer, processor, image, question,
        num_runs=args.num_runs, use_continuous_ar=False, lora_request=lora_request
    )
    print(f"Discrete AR QPS: {qps_discrete:.2f}")
    print(f"Output: {output_discrete}...\n")

    # Continuous AR (with <think>)
    print("Running Continuous AR (with <think>)...")
    qps_continuous, output_continuous = run_benchmark(
        llm, tokenizer, processor, image, question,
        num_runs=args.num_runs, use_continuous_ar=True, lora_request=lora_request
    )
    print(f"Continuous AR QPS: {qps_continuous:.2f}")
    print(f"Output: {output_continuous}...\n")

    # Summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Discrete AR QPS:  {qps_discrete:.2f}")
    print(f"Continuous AR QPS: {qps_continuous:.2f}")
    print(f"Ratio: {qps_continuous/qps_discrete:.2f}x")


if __name__ == "__main__":
    main()
