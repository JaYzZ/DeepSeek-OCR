#!/usr/bin/env python3
"""
Transparent Evaluation Callback for LlamaFactory Training

Monitors environment variable ENABLE_TRANSPARENT_EVAL and runs
inference on transparent eval samples during training, saving results
in the designated format for qualitative monitoring.
"""

import json
import logging
import os
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.integrations import is_fsdp_managed_module
from project_paths import get_deepseek_ocr_dir, resolve_project_path

logger = logging.getLogger(__name__)

from OCRVL.scripts.generate_composite_images import generate_composite_images


class TransparentEvalCallback(TrainerCallback):
    """
    Callback for transparent evaluation during LlamaFactory training.

    Reads evaluation samples from OCRVL_TRANSPARENT_EVAL_SAMPLES environment variable
    and runs inference at each evaluation step, saving results in the format:

    checkpoint-N/
      └── eval_results/
          ├── step_N_results.txt
          └── step_N_results.json
    """

    def __init__(self, model, tokenizer=None, processor=None, processing_class=None, synced_gpus=None):
        self.model = model
        # Support processing_class (newer Transformers) or tokenizer (older)
        self.tokenizer = processing_class if processing_class is not None else tokenizer
        self.processor = processor
        # Auto-enable - callback will check eval_dataset in on_evaluate
        self.samples_path = os.environ.get("TRANSPARENT_EVAL_SAMPLES", "")
        self.max_new_tokens = int(os.environ.get("TRANSPARENT_EVAL_MAX_NEW_TOKENS", "128"))
        self.temperature = float(os.environ.get("TRANSPARENT_EVAL_TEMPERATURE", "0.0"))
        self.limit = os.environ.get("TRANSPARENT_EVAL_LIMIT", "")
        self.project_dir = str(get_deepseek_ocr_dir())

        logger.info(f"[TransparentEval] Initialized - will auto-enable when eval_dataset='ocrvl_transparent_eval'")
        logger.info(f"[TransparentEval] Config: max_tokens={self.max_new_tokens}, temp={self.temperature}, limit={self.limit}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Compile vision tower AFTER FSDP preparation is complete.

        This is called after accelerator.prepare() has wrapped the model with FSDP.
        At this point, we can safely compile the vision tower without interfering
        with FSDP's unwrap logic.
        """
        # Only compile if explicitly enabled
        if os.environ.get("QWEN3VL_COMPILE_VISION_ONLY", "0") != "1":
            return

        model = kwargs.get("model")
        if model is None:
            logger.warning("[TransparentEval] No model in kwargs for vision compilation")
            return

        # Only compile on rank 0 to avoid redundant compilation
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        if not is_rank0:
            return

        pass

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Run transparent evaluation after standard evaluation completes."""
        # Auto-enable: Check if eval_dataset is configured to ocrvl_transparent_eval
        # Method 1: Check eval_dataloader for the dataset name
        should_run = False

        eval_dataloader = kwargs.get('eval_dataloader')
        if eval_dataloader is not None:
            dataset = getattr(eval_dataloader, 'dataset', None)
            if dataset is not None:
                dataset_name = getattr(dataset, 'dataset_name', None)
                if dataset_name == 'ocrvl_transparent_eval':
                    should_run = True

        # Method 2: Check if output_dir contains 'thinking' (for r1_onevision_thinking runs)
        if not should_run and 'thinking' in getattr(args, 'output_dir', ''):
            should_run = True

        if not should_run:
            return

        # Start timing
        is_main = self._is_main_process()
        eval_start_time = time.time()

        if is_main:
            msg = f"[TransparentEval] ========== STARTING =========="
            logger.info(msg)
            print(msg, flush=True)

        # Load samples (all ranks need this for FSDP)
        metadata_path = Path(self.project_dir) / "OCRVL/data/ocrvl_transparent_eval.metadata.json"
        if not metadata_path.exists():
            if is_main:
                logger.warning(f"[TransparentEval] Metadata not found: {metadata_path}")
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        samples = metadata['samples']
        if self.limit:
            samples = samples[:int(self.limit)]

        # Load full samples from JSONL to get ground_truth (from assistant message)
        # This is needed because unified SFT format doesn't have ground_truth field
        full_samples = {}
        jsonl_path = Path(self.project_dir) / "OCRVL/data/ocrvl_transparent_eval.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, 'r') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    sample_id = sample.get('id')
                    if sample_id:
                        # Extract ground_truth from assistant message if not present
                        if 'ground_truth' not in sample:
                            for msg in sample.get('messages', []):
                                if msg.get('role') == 'assistant':
                                    sample['ground_truth'] = msg.get('content', '')
                                    break
                        full_samples[sample_id] = sample

        # Merge ground_truth into samples
        for sample in samples:
            sample_id = sample.get('id')
            if sample_id in full_samples:
                if 'ground_truth' not in sample or not sample['ground_truth']:
                    sample['ground_truth'] = full_samples[sample_id].get('ground_truth', '')

        # Split samples across GPUs for parallel evaluation
        num_ranks = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        original_len = len(samples)
        samples_per_rank = (len(samples) + num_ranks - 1) // num_ranks

        # Build local_samples by round-robin distribution
        local_samples = []
        for i in range(samples_per_rank):
            sample_idx = rank + (i * num_ranks)
            if sample_idx < len(samples):
                local_samples.append(samples[sample_idx])
            else:
                local_samples.append(None)  # Dummy sample for synced_gpus=True

        if is_main:
            msg = f"[TransparentEval] Distributed {original_len} samples across {num_ranks} GPUs (~{samples_per_rank} per GPU)"
            logger.info(msg)
            print(msg, flush=True)

        # Each rank runs inference on its subset
        local_results = self._run_inference(local_samples, state.global_step, kwargs.get('model'))

        # Gather results from all ranks to rank 0
        if dist.is_initialized():
            # Use gather_object to collect results from all ranks
            gathered_results = [None] * num_ranks
            dist.gather_object(local_results, gathered_results if rank == 0 else None, dst=0)

            # Flatten results on rank 0
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
        """Run inference on evaluation samples using LlamaFactory data collator approach.

        Note: All ranks must call this method for FSDP to work properly.

        Args:
            samples: List of evaluation samples
            global_step: Current training step
            model: Optional model to use (from kwargs). If None, uses self.model
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        inference_model = model if model is not None else self.model
        is_fsdp = is_fsdp_managed_module(inference_model)
        is_main = self._is_main_process()

        # Track timing
        prep_start = time.time()
        gen_start = None

        # Set model to eval mode and disable gradient checkpointing
        was_training = inference_model.training
        gc_was_enabled = False

        if hasattr(inference_model, 'gradient_checkpointing_disable'):
            try:
                is_gc_enabled = getattr(inference_model, 'is_gradient_checkpointing', False)
                if is_gc_enabled:
                    inference_model.gradient_checkpointing_disable()
                    gc_was_enabled = True
            except Exception:
                pass
        else:
            if hasattr(inference_model, 'base_model'):
                base_model = inference_model.base_model
                if hasattr(base_model, 'gradient_checkpointing_disable'):
                    try:
                        is_gc_enabled = getattr(base_model, 'is_gradient_checkpointing', False)
                        if is_gc_enabled:
                            base_model.gradient_checkpointing_disable()
                            gc_was_enabled = True
                    except Exception:
                        pass

        inference_model.eval()
        results = []

        # PHASE 1: Prepare all inputs (outside FSDP summon to avoid overhead)
        prepared_inputs = []
        first_real_inputs = None

        for i, sample in enumerate(samples):
            if sample is None:
                if first_real_inputs is not None:
                    prepared_inputs.append({
                        'is_dummy': True,
                        'sample': {'id': f'dummy_{i}'},
                        'input_ids': first_real_inputs['input_ids'],
                        'attention_mask': first_real_inputs['attention_mask'],
                        'pixel_values': first_real_inputs['pixel_values'],
                        'image_grid_thw': first_real_inputs['image_grid_thw'],
                        'input_len': first_real_inputs['input_len'],
                    })
                continue

            try:
                images = []
                for img_path in sample['images']:
                    full_path = resolve_project_path(img_path, repo_root=self.project_dir)
                    if full_path.exists():
                        images.append(Image.open(full_path).convert('RGB'))

                if len(images) not in (1, 2):
                    continue

                image_processor = getattr(self.processor, "image_processor", None)
                if image_processor is None:
                    continue

                mm_inputs = image_processor(images, return_tensors="pt")

                # Validate that image processing succeeded
                if mm_inputs is None or 'pixel_values' not in mm_inputs or mm_inputs['pixel_values'] is None:
                    if is_main:
                        logger.warning(f"[TransparentEval] Image processor returned None for sample {sample.get('id', 'unknown')}")
                    continue

                image_grid_thw = mm_inputs.get("image_grid_thw")
                merge_length = getattr(image_processor, "merge_size", 2) ** 2

                vision_placeholders = []
                for i in range(len(images)):
                    if image_grid_thw is not None:
                        image_seqlen = image_grid_thw[i].prod().item() // merge_length
                    else:
                        image_seqlen = 100
                    placeholder = "<|vision_start|>" + "<|image_pad|>" * image_seqlen + "<|vision_end|>"
                    vision_placeholders.append(placeholder)

                instruction_text = sample.get('instruction', 'Describe the image:').replace('<image>', '').strip()
                if len(images) == 1:
                    user_content = instruction_text + " " + vision_placeholders[0]
                else:
                    user_content = instruction_text + " " + vision_placeholders[0] + vision_placeholders[1]

                conversation = [{"role": "user", "content": user_content}]
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                text_inputs = self.tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False)

                device = next(inference_model.parameters()).device
                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                pixel_values = mm_inputs['pixel_values'].to(device=device, dtype=torch.bfloat16)
                image_grid_thw = mm_inputs['image_grid_thw'].to(device)

                prepared_inputs.append({
                    'sample': sample,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'pixel_values': pixel_values,
                    'image_grid_thw': image_grid_thw,
                    'input_len': input_ids.shape[1],
                })

                if first_real_inputs is None:
                    first_real_inputs = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'pixel_values': pixel_values,
                        'image_grid_thw': image_grid_thw,
                        'input_len': input_ids.shape[1],
                    }

            except Exception as e:
                if is_main:
                    logger.warning(f"[TransparentEval] Failed to prepare sample {sample.get('id', 'unknown')}: {e}")
                    logger.debug(f"[TransparentEval] Preparation traceback:\n{traceback.format_exc()}")
                continue

        prep_time = time.time() - prep_start
        gen_start = time.time()

        if is_main:
            msg = f"[TransparentEval] Prepared {len(prepared_inputs)} samples ({prep_time:.2f}s), starting generation..."
            logger.info(msg)
            print(msg, flush=True)

        # PHASE 2: Generate all outputs
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if is_fsdp:
                with FSDP.summon_full_params(inference_model, writeback=False, recurse=True):
                    for i, inputs in enumerate(prepared_inputs):
                        is_dummy = inputs.get('is_dummy', False)

                        try:
                            # Validate inputs before generation
                            if inputs.get('pixel_values') is None:
                                if is_main:
                                    logger.warning(f"[TransparentEval] pixel_values is None for sample {inputs['sample'].get('id', 'unknown')}, skipping")
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
                                synced_gpus=True,
                            )

                            if is_dummy:
                                continue

                            generated_ids = outputs[0][inputs['input_len']:]
                            generated_ids_list = generated_ids.tolist()
                            full_output = self._decode_full_output(generated_ids_list)
                            display_output = self._extract_display_output(generated_ids_list, full_output)

                            sample = inputs['sample']
                            results.append({
                                'id': sample['id'],
                                'task': sample['task'],
                                'images': sample['images'],
                                'instruction': sample.get('instruction', ''),
                                'ground_truth': sample.get('ground_truth', ''),
                                'ground_truth_path': sample.get('ground_truth_path', ''),
                                'generated_answer': full_output if full_output else "[EMPTY]",
                                'generated_answer_display': display_output if display_output else "[EMPTY]",
                                'step': global_step,
                            })
                            if sample.get('question_text'):
                                results[-1]['question_text'] = sample['question_text']

                            if is_main:
                                logger.info(f"[TransparentEval] Generated {len(results)}/{len(prepared_inputs)} samples")

                        except Exception as e:
                            if is_main:
                                logger.warning(f"[TransparentEval] Failed to generate for sample {inputs['sample'].get('id', 'unknown')}: {e}")
                                logger.debug(f"[TransparentEval] Full traceback:\n{traceback.format_exc()}")
                            continue
            else:
                for i, inputs in enumerate(prepared_inputs):
                    is_dummy = inputs.get('is_dummy', False)
                    try:
                        # Validate inputs before generation
                        if inputs.get('pixel_values') is None:
                            if is_main:
                                logger.warning(f"[TransparentEval] pixel_values is None for sample {inputs['sample'].get('id', 'unknown')}, skipping")
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
                            synced_gpus=False,
                        )

                        if is_dummy:
                            continue

                        generated_ids = outputs[0][inputs['input_len']:]
                        generated_ids_list = generated_ids.tolist()
                        full_output = self._decode_full_output(generated_ids_list)
                        display_output = self._extract_display_output(generated_ids_list, full_output)

                        sample = inputs['sample']
                        results.append({
                            'id': sample['id'],
                            'task': sample['task'],
                            'images': sample['images'],
                            'instruction': sample.get('instruction', ''),
                            'ground_truth': sample.get('ground_truth', ''),
                            'ground_truth_path': sample.get('ground_truth_path', ''),
                            'generated_answer': full_output if full_output else "[EMPTY]",
                            'generated_answer_display': display_output if display_output else "[EMPTY]",
                            'step': global_step,
                        })
                        if sample.get('question_text'):
                            results[-1]['question_text'] = sample['question_text']

                        if is_main:
                            logger.info(f"[TransparentEval] Generated {len(results)}/{len(prepared_inputs)} samples")

                    except Exception as e:
                        if is_main:
                            logger.warning(f"[TransparentEval] Failed to generate for sample {inputs['sample'].get('id', 'unknown')}: {e}")
                            logger.debug(f"[TransparentEval] Full traceback:\n{traceback.format_exc()}")
                        continue

        # Restore training mode and gradient checkpointing
        if was_training:
            inference_model.train()

        if gc_was_enabled:
            if hasattr(inference_model, 'gradient_checkpointing_enable'):
                try:
                    inference_model.gradient_checkpointing_enable()
                except Exception:
                    pass
            else:
                if hasattr(inference_model, 'base_model'):
                    base_model = inference_model.base_model
                    if hasattr(base_model, 'gradient_checkpointing_enable'):
                        try:
                            base_model.gradient_checkpointing_enable()
                        except Exception:
                            pass

        gen_time = time.time() - gen_start if gen_start else 0
        if is_main:
            msg = f"[TransparentEval] Generated {len(results)} samples (prep={prep_time:.2f}s, gen={gen_time:.2f}s)"
            logger.info(msg)
            print(msg, flush=True)

        return results

    def _save_results(self, checkpoint_dir: Path, results: List[Dict], global_step: int, total_time: float = 0):
        """Save evaluation results in both JSON and human-readable formats.

        Also creates an images/ subdirectory with symlinks to all evaluation images
        for easy cross-referencing, and generates composite images for visual inspection.
        """
        eval_dir = checkpoint_dir / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Create images/ subdirectory and symlink all evaluation images
        images_dir = eval_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Create symlinks for each sample's images, named by sample ID
        for result in results:
            sample_id = result['id']
            images = result.get('images', [])

            for img_idx, img_path in enumerate(images):
                src = resolve_project_path(img_path, repo_root=self.project_dir)
                if src.exists():
                    # Name: {sample_id}_img{idx}{ext}
                    ext = src.suffix
                    link_name = images_dir / f"{sample_id}_img{img_idx}{ext}"

                    try:
                        if not link_name.exists():
                            link_name.symlink_to(src.resolve())
                    except (OSError, NotImplementedError):
                        # Symlink not supported (Windows), copy instead
                        shutil.copy2(src, link_name)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON (for programmatic analysis)
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
            f.write(f"Transparent Evaluation - Step {global_step}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Samples: {len(results)}\n")
            if total_time > 0:
                f.write(f"Total Time: {total_time:.2f}s ({total_time/len(results):.2f}s per sample)\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"Sample {i}: {result['id']}\n")
                f.write(f"Task: {result['task']}\n")
                f.write(f"Images:\n")
                for img_idx, img in enumerate(result['images']):
                    f.write(f"  - {result['id']}_img{img_idx}{Path(img).suffix}\n")

                # Display instruction/question (important for VQA)
                if result.get('instruction'):
                    f.write(f"Question/Instruction: {result['instruction']}\n")

                # Display ground truth (no truncation)
                if result['ground_truth']:
                    gt = result['ground_truth']
                    f.write(f"Ground Truth ({len(gt)} chars):\n{gt}\n")

                if result['ground_truth_path']:
                    f.write(f"Ground Truth Path: {result['ground_truth_path']}\n")

                # Display generated answer (NO truncation)
                gen = result['generated_answer']
                f.write(f"Generated ({len(gen)} chars):\n{gen}\n")

                f.write("-" * 80 + "\n\n")

        logger.info(f"[TransparentEval] Saved results to:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {txt_path}")
        logger.info(f"  - {images_dir}/ (symlinks to evaluation images)")
        if total_time > 0:
            msg = f"[TransparentEval] Total time: {total_time:.2f}s ({total_time/len(results):.2f}s per sample)"
            logger.info(msg)
            print(msg, flush=True)
        msg = f"[TransparentEval] ========== COMPLETED =========="
        logger.info(msg)
        print(msg, flush=True)

        # Generate composite images for visual inspection
        self._generate_composite_images(json_path, eval_dir)

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
        end_id = self._get_think_end_id()
        if end_id is None:
            if "</think>" in full_output:
                return full_output.rsplit("</think>", 1)[-1].strip()
            return full_output
        try:
            last_idx = len(generated_ids) - 1 - generated_ids[::-1].index(end_id)
        except ValueError:
            if "</think>" in full_output:
                return full_output.rsplit("</think>", 1)[-1].strip()
            return full_output
        answer_ids = generated_ids[last_idx + 1:]
        if not answer_ids:
            return ""
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def _get_think_end_id(self) -> Optional[int]:
        """Return </think> token id if available via env or tokenizer."""
        env_val = os.environ.get("QWEN3VL_THINKING_END_ID", "").strip()
        if env_val.isdigit():
            return int(env_val)
        if self.tokenizer is None:
            return None
        try:
            token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        except Exception:
            return None
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is None or (unk_id is not None and token_id == unk_id):
            return None
        return token_id

    def _generate_composite_images(self, results_json: Path, output_dir: Path):
        """Generate composite images for all evaluation samples.

        Creates visual summaries showing input images with ground truth and
        generated text overlaid for easy qualitative assessment.

        Args:
            results_json: Path to step_N_results.json file
            output_dir: Directory to save composite images (will create step_N_composite/ subdir)
        """
        if generate_composite_images is None:
            logger.warning("[TransparentEval] Composite image generation not available (import failed)")
            return

        try:
            logger.info(f"[TransparentEval] Generating composite images...")
            generate_composite_images(
                results_json=results_json,
                output_dir=output_dir,
                repo_root=Path(self.project_dir),
            )

        except Exception as e:
            logger.warning(f"[TransparentEval] Failed to generate composite images: {e}")
            logger.exception("Composite image generation failed")

    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
