#!/usr/bin/env python3
"""
Transparent Evaluation Callback for LlamaFactory Training (Qwen3VL-specific)

Monitors environment variable QWEN3VL_TRANSPARENT_EVAL and runs
inference on transparent eval samples during training, saving results
in designated format for qualitative monitoring.
"""

import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from PIL import Image
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.integrations import is_fsdp_managed_module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

logger = logging.getLogger(__name__)


class QwenTransparentEvalCallback(TrainerCallback):
    """
    Callback for transparent evaluation during LlamaFactory training.

    Reads evaluation samples from QWEN3VL_TRANSPARENT_EVAL_SAMPLES environment variable
    and runs inference at each evaluation step, saving results in format:

    checkpoint-N/
      └── eval_results/
          ├── step_N_results.txt
          └── step_N_results.json
    """

    def __init__(self, model, tokenizer=None, processor=None, synced_gpus=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        # synced_gpus is accepted for compatibility with sitecustomize.py but ignored
        # (FSDP detection is automatic via is_fsdp_managed_module())
        # Auto-enable - callback will check eval_dataset in on_evaluate
        self.samples_path = os.environ.get("QWEN3VL_TRANSPARENT_EVAL_SAMPLES", "")
        self.max_new_tokens = int(os.environ.get("QWEN3VL_TRANSPARENT_EVAL_MAX_NEW_TOKENS", "2048"))
        self.temperature = float(os.environ.get("QWEN3VL_TRANSPARENT_EVAL_TEMPERATURE", "0.0"))
        self.limit = os.environ.get("QWEN3VL_TRANSPARENT_EVAL_LIMIT", "")
        self.repo_root = os.environ.get("REPO_ROOT", "/share/project/xiyan/sources/DeepSeek-OCR")

        logger.info("[QwenTransparentEval] Initialized - will auto-enable when eval_dataset='qwen3vl_transparent_eval'")
        logger.info(f"[QwenTransparentEval] Config: max_tokens={self.max_new_tokens}, temp={self.temperature}, limit={self.limit}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Run transparent evaluation after standard evaluation completes."""
        # Auto-enable: Check if eval_dataset is configured to qwen3vl_transparent_eval
        should_run = False

        eval_dataloader = kwargs.get('eval_dataloader')
        if eval_dataloader is not None:
            dataset = getattr(eval_dataloader, 'dataset', None)
            if dataset is not None:
                dataset_name = getattr(dataset, 'dataset_name', None)
                if dataset_name == 'qwen3vl_transparent_eval':
                    should_run = True

        # Also check if output_dir contains 'thinking' (for r1_onevision_thinking runs)
        if not should_run and 'thinking' in getattr(args, 'output_dir', ''):
            should_run = True

        if not should_run:
            return

        # Start timing
        is_main = self._is_main_process()
        eval_start_time = time.time()

        if is_main:
            msg = f"[QwenTransparentEval] ========== STARTING =========="
            logger.info(msg)
            print(msg, flush=True)

        # Load samples from metadata
        metadata_path = Path(self.repo_root) / "Qwen/evaluation/data/qwen3vl_transparent_eval.metadata.json"
        if not metadata_path.exists():
            if is_main:
                logger.warning(f"[QwenTransparentEval] No metadata found at {metadata_path}")
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        samples = metadata['samples']
        if self.limit:
            samples = samples[:int(self.limit)]

        # Split samples across GPUs for parallel evaluation
        num_ranks = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        original_len = len(samples)

        # Ensure each GPU gets at least one real sample when possible
        # Strategy: Give each rank its first sample by rank order, then distribute rest
        local_samples = []

        # First pass: ensure each rank gets at least one sample (if we have enough)
        if len(samples) >= num_ranks:
            # Give rank 'i' the sample at index 'i'
            if rank < len(samples):
                local_samples.append(samples[rank])
        else:
            # Not enough samples for all GPUs - only some ranks get one
            # Ranks 0, 1, 2 get samples if we have 3 samples
            if rank < len(samples):
                local_samples.append(samples[rank])

        # Second pass: distribute remaining samples in round-robin fashion
        # Start from num_ranks to skip the ones we already assigned
        start_idx = num_ranks
        for sample_idx in range(start_idx, len(samples)):
            if (sample_idx - start_idx) % num_ranks == rank:
                local_samples.append(samples[sample_idx])

        # Pad with dummy samples to ensure all ranks have same count (for synced_gpus=True)
        max_local_size = (len(samples) + num_ranks - 1) // num_ranks
        while len(local_samples) < max_local_size:
            local_samples.append(None)

        if is_main:
            real_count = sum(1 for s in local_samples if s is not None)
            msg = f"[QwenTransparentEval] Distributed {original_len} samples across {num_ranks} GPUs (~{real_count} real samples on this GPU)"
            logger.info(msg)
            print(msg, flush=True)

        # Each rank runs inference on its subset
        local_results = self._run_inference(local_samples, state.global_step, kwargs.get('model'))

        # Gather results from all ranks to rank 0
        if dist.is_initialized():
            gathered_results = [None] * num_ranks
            dist.gather_object(local_results, gathered_results if rank == 0 else None, dst=0)

            if rank == 0:
                results = [r for sublist in gathered_results if sublist for r in sublist]
            else:
                results = []
        else:
            results = local_results

        # Only rank 0 saves results
        if is_main:
            total_time = time.time() - eval_start_time
            self._save_results(Path(args.output_dir), results, state.global_step, total_time)

    def _run_inference(self, samples: List[Dict], global_step: int, model=None) -> List[Dict]:
        """Run inference on evaluation samples.

        Note: All ranks must call this method for distributed inference.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        inference_model = model if model is not None else self.model
        is_main = self._is_main_process()

        # Track timing
        prep_start = time.time()

        # Set model to eval mode
        was_training = inference_model.training
        inference_model.eval()

        results = []

        # Prepare all inputs
        prepared_inputs = []
        # Track if we have any valid real inputs for dummy creation
        has_real_input = False
        first_real_inputs = None

        for i, sample in enumerate(samples):
            if sample is None:
                # Only create dummy if we have a valid real input
                if first_real_inputs is not None and has_real_input:
                    dummy_input = {
                        'is_dummy': True,
                        'sample': {'id': f'dummy_{i}'},
                        'input_ids': first_real_inputs['input_ids'],
                        'attention_mask': first_real_inputs['attention_mask'],
                        'pixel_values': first_real_inputs['pixel_values'],
                        'image_grid_thw': first_real_inputs['image_grid_thw'],
                        'input_len': first_real_inputs['input_len'],
                    }
                    if is_main:
                        logger.info(f"[QwenTransparentEval] Rank {dist.get_rank() if dist.is_initialized() else 0}: Creating dummy input {i} with input_len={dummy_input.get('input_len')}")
                    prepared_inputs.append(dummy_input)
                else:
                    if is_main:
                        logger.warning(f"[QwenTransparentEval] Cannot create dummy {i}: no valid real input available yet")
                continue

            try:
                images = []
                instruction_text = ""

                # Qwen3VL format: extract from messages array
                if 'messages' in sample and isinstance(sample['messages'], list):
                    user_msg = next((m for m in sample['messages'] if m.get('role') == 'user'), None)
                    if user_msg and isinstance(user_msg.get('content'), str):
                        # Extract <image> placeholders and text
                        content = user_msg.get('content', '')
                        # Count <image> tokens
                        image_count = content.count('<image>')
                        # Remove <image> markers for instruction text
                        instruction_text = content.replace('<image>', '').strip()

                        # Load images from the images field
                        for img_path in sample.get('images', []):
                            # Handle both absolute and relative paths
                            if Path(img_path).is_absolute():
                                full_path = Path(img_path)
                            else:
                                full_path = Path(self.repo_root) / img_path

                            if full_path.exists():
                                images.append(Image.open(full_path).convert('RGB'))
                            else:
                                if is_main:
                                    logger.warning(f"[QwenTransparentEval] Image not found: {full_path}")

                        # Verify image count matches
                        if len(images) != image_count:
                            if is_main:
                                logger.warning(f"[QwenTransparentEval] Image count mismatch for {sample.get('id')}: {image_count} <image> tokens but {len(images)} images loaded")
                            # Skip this sample
                            continue

                if len(images) == 0:
                    if is_main:
                        logger.warning(f"[QwenTransparentEval] No images loaded for sample {sample.get('id')}")
                    continue

                image_processor = getattr(self.processor, "image_processor", None)
                if image_processor is None:
                    if is_main:
                        logger.warning(f"[QwenTransparentEval] No image processor available")
                    continue

                mm_inputs = image_processor(images, return_tensors="pt")

                # Validate that image processing succeeded
                if mm_inputs is None or 'pixel_values' not in mm_inputs or mm_inputs['pixel_values'] is None:
                    if is_main:
                        logger.warning(f"[QwenTransparentEval] Image processor returned None for sample {sample.get('id')}")
                    continue

                image_grid_thw = mm_inputs.get("image_grid_thw")
                merge_length = getattr(image_processor, "merge_size", 2) ** 2

                # Build vision placeholders
                vision_placeholders = []
                for i in range(len(images)):
                    if image_grid_thw is not None:
                        image_seqlen = image_grid_thw[i].prod().item() // merge_length
                    else:
                        image_seqlen = 256  # fallback for older models
                    placeholder = "<|vision_start|>" + "<|image_pad|>" * image_seqlen + "<|vision_end|>"
                    vision_placeholders.append(placeholder)

                # Clean up instruction text
                user_content = instruction_text.strip() + " " + vision_placeholders[0]

                # Build conversation for tokenizer (Option 2: no <think> in prompt)
                conversation = [{"role": "user", "content": user_content}]
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                text = text + "<|im_start|>assistant\n"
                text_inputs = self.tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False)

                # Validate tokenizer output before converting to device
                if text_inputs.get('input_ids') is None or text_inputs.get('attention_mask') is None:
                    if is_main:
                        logger.warning(f"[QwenTransparentEval] Tokenizer returned None for sample {sample.get('id')}")
                    continue

                device = next(inference_model.parameters()).device
                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                # Get pixel_values from processor - convert to bfloat16 explicitly (like OCRVL does)
                pixel_values = mm_inputs['pixel_values'].to(device=device, dtype=torch.bfloat16)
                # image_grid_thw should remain as long/int (grid dimensions)
                image_grid_thw = mm_inputs['image_grid_thw'].to(device)

                prepared_inputs.append({
                    'sample': sample,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'pixel_values': pixel_values,
                    'image_grid_thw': image_grid_thw,
                    'input_len': input_ids.shape[1],
                })

                # Capture first valid real input for dummy creation
                if not has_real_input:
                    first_real_inputs = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'pixel_values': pixel_values,
                        'image_grid_thw': image_grid_thw,
                        'input_len': input_ids.shape[1],
                    }
                    has_real_input = True

            except Exception as e:
                if is_main:
                    logger.warning(f"[QwenTransparentEval] Failed to prepare sample {sample.get('id')}: {e}")
                    logger.debug(f"[QwenTransparentEval] Preparation traceback:\n{traceback.format_exc()}")
                continue

        prep_time = time.time() - prep_start
        gen_start = time.time()

        if is_main:
            msg = f"[QwenTransparentEval] Prepared {len(prepared_inputs)} samples ({prep_time:.2f}s), starting generation..."
            logger.info(msg)
            print(msg, flush=True)

        # Check if model is FSDP-wrapped
        is_fsdp = is_fsdp_managed_module(inference_model)

        # Phase 2: Generate all outputs
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if is_fsdp:
                # FSDP: summon full params for generation
                with FSDP.summon_full_params(inference_model, writeback=False, recurse=True):
                    for inputs in prepared_inputs:
                        if inputs.get('is_dummy', False):
                            continue

                        try:
                            # Validate inputs before generation
                            if inputs.get('pixel_values') is None:
                                if is_main:
                                    logger.warning(f"[QwenTransparentEval] pixel_values is None for sample {inputs['sample'].get('id')}, skipping")
                                continue

                            outputs = inference_model.generate(
                                input_ids=inputs['input_ids'],
                                pixel_values=inputs['pixel_values'],
                                image_grid_thw=inputs['image_grid_thw'],
                                attention_mask=inputs['attention_mask'],
                                max_new_tokens=self.max_new_tokens,
                                do_sample=self.temperature > 0,
                                temperature=self.temperature if self.temperature > 0 else None,
                                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                synced_gpus=True,  # Required for FSDP multi-GPU
                            )

                            # Guard against None outputs
                            if outputs is None or outputs[0] is None:
                                if is_main:
                                    logger.warning(f"[QwenTransparentEval] generate() returned None for sample {inputs['sample'].get('id')}")
                                continue

                            # Additional check: verify outputs[0] is valid tensor
                            if outputs is not None and not torch.is_tensor(outputs[0]):
                                if is_main:
                                    logger.warning(f"[QwenTransparentEval] outputs[0] is not a tensor (type: {type(outputs[0])}, value: {outputs[0]})")
                                continue

                            generated_ids = outputs[0][inputs['input_len']:]
                            generated_ids_list = generated_ids.tolist()
                            full_output = self._decode_full_output(generated_ids_list)
                            display_output = self._extract_display_output(generated_ids_list, full_output)

                            sample = inputs['sample']
                            results.append({
                                'id': sample['id'],
                                'task': sample['task'],
                                'images': sample.get('images', []),
                                'instruction': '',
                                'ground_truth': sample.get('ground_truth', ''),
                                'generated_answer': full_output if full_output else "[EMPTY]",
                                'generated_answer_display': display_output if display_output else "[EMPTY]",
                                'step': global_step,
                            })

                            if is_main:
                                logger.info(f"[QwenTransparentEval] Generated {len(results)}/{len(prepared_inputs)} samples")

                        except Exception as e:
                            if is_main:
                                logger.warning(f"[QwenTransparentEval] Failed to generate for sample {inputs['sample'].get('id')}: {e}")
                                logger.debug(f"[QwenTransparentEval] Full traceback:\n{traceback.format_exc()}")
                            continue
            else:
                # Non-FSDP: direct generation
                for inputs in prepared_inputs:
                    if inputs.get('is_dummy', False):
                        continue

                    try:
                        # Validate inputs before generation
                        if inputs.get('pixel_values') is None:
                            if is_main:
                                logger.warning(f"[QwenTransparentEval] pixel_values is None for sample {inputs['sample'].get('id')}, skipping")
                            continue

                        outputs = inference_model.generate(
                            input_ids=inputs['input_ids'],
                            pixel_values=inputs['pixel_values'],
                            image_grid_thw=inputs['image_grid_thw'],
                            attention_mask=inputs['attention_mask'],
                            max_new_tokens=self.max_new_tokens,
                            do_sample=self.temperature > 0,
                            temperature=self.temperature if self.temperature > 0 else None,
                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                        # Guard against None outputs (consistent with FSDP path)
                        if outputs is None or outputs[0] is None:
                            if is_main:
                                logger.warning(f"[QwenTransparentEval] generate() returned None for sample {inputs['sample'].get('id')}")
                            continue

                        # Additional check: verify outputs[0] is valid tensor
                        if not torch.is_tensor(outputs[0]):
                            if is_main:
                                logger.warning(f"[QwenTransparentEval] outputs[0] is not a tensor (type: {type(outputs[0])})")
                            continue

                        generated_ids = outputs[0][inputs['input_len']:]
                        generated_ids_list = generated_ids.tolist()
                        full_output = self._decode_full_output(generated_ids_list)
                        display_output = self._extract_display_output(generated_ids_list, full_output)

                        sample = inputs['sample']
                        results.append({
                            'id': sample['id'],
                            'task': sample['task'],
                            'images': sample.get('images', []),
                            'instruction': '',
                            'ground_truth': sample.get('ground_truth', ''),
                            'generated_answer': full_output if full_output else "[EMPTY]",
                            'generated_answer_display': display_output if display_output else "[EMPTY]",
                            'step': global_step,
                        })

                        if is_main:
                            logger.info(f"[QwenTransparentEval] Generated {len(results)}/{len(prepared_inputs)} samples")

                    except Exception as e:
                        if is_main:
                            logger.warning(f"[QwenTransparentEval] Failed to generate for sample {inputs['sample'].get('id')}: {e}")
                            logger.debug(f"[QwenTransparentEval] Full traceback:\n{traceback.format_exc()}")
                        continue

        # Restore training mode
        if was_training:
            inference_model.train()

        gen_time = time.time() - gen_start if gen_start else 0
        if is_main:
            msg = f"[QwenTransparentEval] Generated {len(results)} samples (prep={prep_time:.2f}s, gen={gen_time:.2f}s)"
            logger.info(msg)
            print(msg, flush=True)

        return results

    def _decode_full_output(self, generated_ids: List[int]) -> str:
        """Decode full model output, preserving <think> tags for .txt logging."""
        if not generated_ids:
            return ""
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if eos_token and eos_token in text:
            text = text.split(eos_token)[0].strip()
        return text

    def _extract_display_output(self, generated_ids: List[int], full_output: str) -> str:
        """Extract answer content after </think> for composite display."""
        # Qwen3VL thinking format uses <think>...</think> or equivalent
        # For display purposes, we show everything after thinking ends
        think_end = os.environ.get("QWEN3VL_THINKING_END_ID", "")

        if think_end == "":
            # No thinking end configured, return full output
            return full_output

        try:
            end_token_id = int(think_end)
        except ValueError:
            return full_output

        # Try to find the thinking end token in generated ids
        try:
            last_idx = len(generated_ids) - 1 - generated_ids[::-1].index(end_token_id)
        except ValueError:
            return full_output

        answer_ids = generated_ids[last_idx + 1:]
        if not answer_ids:
            return ""
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def _save_results(self, checkpoint_dir: Path, results: List[Dict], global_step: int, total_time: float = 0):
        """Save evaluation results in both JSON and human-readable formats."""
        eval_dir = checkpoint_dir / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON
        json_path = eval_dir / f"step_{global_step}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'step': global_step,
                'timestamp': timestamp,
                'num_samples': len(results),
                'total_time_seconds': round(total_time, 2),
                'results': results,
            }, f, indent=2, ensure_ascii=False)

        # Save human-readable text
        txt_path = eval_dir / f"step_{global_step}_results.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Qwen Transparent Evaluation - Step {global_step}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Samples: {len(results)}\n")
            if total_time > 0 and len(results) > 0:
                f.write(f"Total Time: {total_time:.2f}s ({total_time/len(results):.2f}s per sample)\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"Sample {i}: {result['id']}\n")
                f.write(f"Task: {result['task']}\n")

                # Display ground truth
                if result['ground_truth']:
                    gt = result['ground_truth']
                    f.write(f"Ground Truth ({len(gt)} chars):\n{gt}\n")

                # Display generated answer
                gen = result['generated_answer']
                f.write(f"Generated ({len(gen)} chars):\n{gen}\n")

                f.write("-" * 80 + "\n\n")

        # Generate composite images
        composite_dir = self._generate_composite_images(eval_dir, results, global_step)

        logger.info(f"[QwenTransparentEval] Saved results to:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {txt_path}")
        logger.info(f"  - {composite_dir}/ (composite images)")
        if total_time > 0 and len(results) > 0:
            msg = f"[QwenTransparentEval] Total time: {total_time:.2f}s ({total_time/len(results):.2f}s per sample)"
            logger.info(msg)
            print(msg, flush=True)
        msg = f"[QwenTransparentEval] ========== COMPLETED =========="
        logger.info(msg)
        print(msg, flush=True)

    def _generate_composite_images(self, eval_dir: Path, results: List[Dict], global_step: int) -> Path:
        """Generate composite images for each sample."""
        composite_dir = eval_dir / f"step_{global_step}_composite"
        composite_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(self.repo_root)

        # Try to load fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            text_font = ImageFont.load_default()

        img_width = 800
        padding = 20

        for idx, result in enumerate(results, 1):
            sample_id = result.get('id', f'sample_{idx}')
            output_path = composite_dir / f"{idx:02d}_{sample_id}.png"

            # Load images
            images = []
            for img_path in result.get('images', []):
                full_path = repo_root / img_path
                if full_path.exists():
                    img = Image.open(full_path).convert('RGB')
                    images.append(img)

            # For bbox_ocr tasks, draw bbox
            if result.get('task') == 'bbox_ocr' and images:
                instruction = result.get('instruction', '')
                bbox_match = re.search(r'\[([^\]]+)\]', instruction)
                if bbox_match:
                    try:
                        bbox_coords = [float(x.strip()) for x in bbox_match.group(1).split(',')]
                        if len(bbox_coords) == 4:
                            img = images[0]
                            draw = ImageDraw.Draw(img)
                            x1, y1, x2, y2 = bbox_coords
                            abs_x1 = int(x1 * img.width / 1000)
                            abs_y1 = int(y1 * img.height / 1000)
                            abs_x2 = int(x2 * img.width / 1000)
                            abs_y2 = int(y2 * img.height / 1000)
                            draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline='red', width=5)
                            overlay = Image.new('RGBA', img.size, (255, 0, 0, 0))
                            overlay_draw = ImageDraw.Draw(overlay)
                            overlay_draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], fill=(255, 0, 0, 30))
                            images[0] = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                    except (ValueError, IndexError):
                        pass

            if not images:
                continue

            # For VQA, keep only first image
            if result.get('task') == 'visual_question_answering' and len(images) == 2:
                images = [images[0]]

            # Resize images
            max_img_width = img_width - 2 * padding
            resized_images = []
            for img in images:
                ratio = max_img_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_img_width, new_height), Image.Resampling.LANCZOS)
                resized_images.append(img)

            # Get text content
            instruction = result.get('instruction', '')
            ground_truth = result.get('ground_truth', '')
            generated = result.get('generated_answer_display') or result.get('generated_answer', '[EMPTY]')

            # Estimate height (simplified)
            text_area_height = 400  # Conservative estimate
            total_img_height = sum(img.height for img in resized_images)
            spacing = 15
            total_height = total_img_height + text_area_height + (len(resized_images) - 1) * spacing + 3 * padding

            # Create composite
            composite = Image.new('RGB', (img_width, max(total_height, 400)), color='white')
            draw = ImageDraw.Draw(composite)

            # Draw title
            task = result.get('task', 'unknown').replace('_', ' ').title()
            title = f"Sample: {sample_id} | Task: {task}"
            draw.rectangle([padding - 5, padding - 5, img_width - padding + 5, padding + 30], fill='#2196F3')
            draw.text((padding, padding), title, fill='white', font=title_font)

            y_offset = padding + 40

            # Draw images
            for img in resized_images:
                composite.paste(img, (padding, y_offset))
                y_offset += img.height + spacing

            y_offset += 10

            # Draw instruction
            if instruction:
                draw.text((padding, y_offset), "Instruction:", fill='#9C27B0', font=label_font)
                y_offset += 20

            # Draw ground truth
            draw.text((padding, y_offset), "Ground Truth:", fill='#4CAF50', font=label_font)
            y_offset += 20

            # Draw generated answer - separate thinking from answer if different
            full_output = result.get('generated_answer', '')
            display_output = result.get('generated_answer_display', '')

            # Check if there's a difference (thinking vs answer)
            if display_output and display_output != full_output:
                # Extract thinking (the part that gets stripped)
                thinking = full_output.replace(display_output, '', 1).strip()

                # Draw thinking section
                if thinking:
                    draw.text((padding, y_offset), "Thinking:", fill='#9E9E9E', font=label_font)
                    y_offset += 20

                # Draw final answer
                draw.text((padding, y_offset), "Answer:", fill='#FF9800', font=label_font)
            else:
                # No thinking detected, show full output
                draw.text((padding, y_offset), "Model Output:", fill='#FF9800', font=label_font)

            composite.save(output_path, optimize=True, quality=95)

        logger.info(f"[QwenTransparentEval] Generated {len(results)} composite images")
        return composite_dir

    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
