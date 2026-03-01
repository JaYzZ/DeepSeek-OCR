#!/usr/bin/env python3
"""
Test transparent eval with FSDP+LoRA wrapping with regular eval first.
This replicates: regular eval → transparent eval
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
else:
    print("ERROR: Run with: torchrun --nproc_per_node=4 Qwen/scripts/test_transparent_eval_fsdp.py")
    sys.exit(1)

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.integrations import is_fsdp_managed_module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from accelerate.utils import FullyShardedDataParallelPlugin
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType


def log(msg):
    print(f"[Rank {rank}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=18,
                        help="Max samples to test (default: 18)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens to generate (default: 512)")
    parser.add_argument("--model_path", type=str,
                        default="/share/project/xiyan/sources/DeepSeek-OCR/Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking")
    args = parser.parse_args()

    log("="*80)
    log("TEST: Regular Eval → Transparent Eval (replicates training flow)")
    log("="*80)

    # 1. Load model
    log(f"Loading model from {args.model_path}...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 2. Add LoRA
    log("Adding LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)

    # 3. Wrap with FSDP
    log("Wrapping with FSDP (use_orig_params=True)...")
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch='BACKWARD_POST', forward_prefetch=True,
        use_orig_params=True, sync_module_states=True,
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin, mixed_precision='bf16', gradient_accumulation_steps=1)
    model = accelerator.prepare(model)
    log("Model wrapped with FSDP")

    is_fsdp = is_fsdp_managed_module(model)
    log(f"FSDP managed: {is_fsdp}")

    # 4. Load samples
    repo_root = "/share/project/xiyan/sources/DeepSeek-OCR"
    metadata_path = f"{repo_root}/Qwen/evaluation/data/qwen3vl_transparent_eval.metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    samples = metadata['samples'][:args.max_samples]

    # 5. Run REGULAR eval first (4 samples, like training does)
    log("")
    log("="*80)
    log("STEP 1: REGULAR EVAL (4 samples, like llamafactory)")
    log("="*80)

    regular_samples = samples[:4]
    model.eval()

    with torch.no_grad():
        with FSDP.summon_full_params(model, writeback=False, recurse=True):
            log("Inside summon_full_params for regular eval")

            for i, sample in enumerate(regular_samples):
                log(f"Regular eval sample {i+1}/4: {sample['id']}")

                img_path = sample['images'][0]
                if not os.path.exists(img_path):
                    img_path = f"{repo_root}/{img_path}"
                image = Image.open(img_path).convert('RGB')

                image_processor = processor.image_processor
                mm_inputs = image_processor([image], return_tensors="pt")
                image_grid_thw = mm_inputs.get("image_grid_thw")
                merge_length = getattr(image_processor, "merge_size", 2) ** 2
                image_seqlen = image_grid_thw[0].prod().item() // merge_length
                placeholder = "<|vision_start|>" + "<|image_pad|>" * image_seqlen + "<|vision_end|>"

                instruction_text = sample.get('instruction', 'Describe the image:').replace('<image>', '').strip()
                user_content = instruction_text + " " + placeholder
                conversation = [{"role": "user", "content": user_content}]

                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                
                text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                text = text + "<|im_start|>assistant\n"
                text_inputs = tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False)

                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                pixel_values = mm_inputs['pixel_values'].to(device=device, dtype=torch.bfloat16)
                image_grid_thw = mm_inputs['image_grid_thw'].to(device)

                reg_outputs = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    synced_gpus=is_fsdp,
                )
                log(f"  Regular eval sample {i+1} generate() returned")

    log(f"Model mode after regular eval: {model.training}")
    log("✓ Regular eval complete")
    log("")

    # 6. Run TRANSPARENT eval (18 samples with padding)
    log("="*80)
    log("STEP 2: TRANSPARENT EVAL (18 samples with padding)")
    log("="*80)

    # Pad for even distribution
    original_len = len(samples)
    samples_per_rank = (len(samples) + world_size - 1) // world_size
    total_padded = samples_per_rank * world_size
    if len(samples) < total_padded:
        samples = samples + [None] * (total_padded - len(samples))
        log(f"Padded {original_len} samples to {total_padded}")

    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    local_samples = samples[start_idx:end_idx]
    log(f"Processing {len(local_samples)} samples (indices {start_idx}-{end_idx})")

    # Force mode reset
    was_training = model.training
    log(f"Model mode before reset: {was_training}, forcing train->eval transition")
    model.train()
    model.eval()
    log(f"Model mode after reset: {model.training}")

    start_time = time.time()
    results = []

    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # KEY FIX: Call summon_full_params PER SAMPLE (like llamafactory)
            # NOT holding it open across all samples (old slow approach)
            log("Starting per-sample generation with summon_full_params")
            first_real_inputs = None

            for i, sample in enumerate(local_samples):
                if sample is None:
                    log(f"Sample {i+1}/{len(local_samples)} is dummy padding")
                    if first_real_inputs is not None:
                        log(f"  Calling generate() for dummy sample {i+1}")
                        # Summon for dummy sample
                        with FSDP.summon_full_params(model, writeback=False, recurse=True):
                            dummy_outputs = model.generate(
                                input_ids=first_real_inputs['input_ids'],
                                pixel_values=first_real_inputs['pixel_values'],
                                image_grid_thw=first_real_inputs['image_grid_thw'],
                                attention_mask=first_real_inputs['attention_mask'],
                                max_new_tokens=5,  # Short generation for dummies
                                do_sample=False,
                                synced_gpus=is_fsdp,
                            )
                        log(f"  Dummy sample {i+1} generate() returned")
                    continue

                img_path = sample['images'][0]
                if not os.path.exists(img_path):
                    img_path = f"{repo_root}/{img_path}"
                image = Image.open(img_path).convert('RGB')

                image_processor = processor.image_processor
                mm_inputs = image_processor([image], return_tensors="pt")
                image_grid_thw = mm_inputs.get("image_grid_thw")
                merge_length = getattr(image_processor, "merge_size", 2) ** 2
                image_seqlen = image_grid_thw[0].prod().item() // merge_length
                placeholder = "<|vision_start|>" + "<|image_pad|>" * image_seqlen + "<|vision_end|>"

                instruction_text = sample.get('instruction', 'Describe the image:').replace('<image>', '').strip()
                user_content = instruction_text + " " + placeholder
                conversation = [{"role": "user", "content": user_content}]

                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                
                text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                text = text + "<|im_start|>assistant\n"
                text_inputs = tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False)

                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                pixel_values = mm_inputs['pixel_values'].to(device=device, dtype=torch.bfloat16)
                image_grid_thw = mm_inputs['image_grid_thw'].to(device)

                # Cache first sample for dummy padding
                if first_real_inputs is None:
                    first_real_inputs = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'pixel_values': pixel_values,
                        'image_grid_thw': image_grid_thw,
                    }
                    log(f"Cached first sample inputs")

                log(f"Calling generate() for sample {i+1}/{len(local_samples)} (id={sample['id']})")
                # Summon for this sample only (like llamafactory)
                with FSDP.summon_full_params(model, writeback=False, recurse=True):
                    outputs = model.generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        synced_gpus=is_fsdp,
                    )
                log(f"generate() returned for sample {i+1}/{len(local_samples)} (id={sample['id']})")

                generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                results.append({'id': sample['id'], 'generated': generated_text})

        elapsed = time.time() - start_time
        log(f"✓ Transparent eval completed: {len(results)} samples in {elapsed:.1f}s")

        if dist.is_initialized():
            gathered_results = [None] * world_size
            dist.gather_object(results, gathered_results if rank == 0 else None, dst=0)
            if rank == 0:
                all_results = [r for sublist in gathered_results if sublist for r in sublist]
                print(f"\n{'='*80}")
                print(f"[Rank 0] ✓ TEST PASSED - Full training flow works!")
                print(f"Total samples: {len(all_results)}")
                print(f"{'='*80}\n")

    except Exception as e:
        log(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    finally:
        dist.barrier()
        dist.destroy_process_group()
