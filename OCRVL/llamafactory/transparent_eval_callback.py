#!/usr/bin/env python3
"""
Transparent Evaluation Callback for LlamaFactory Training

Monitors environment variable OCRVL_ENABLE_TRANSPARENT_EVAL and runs
inference on transparent eval samples during training, saving results
in the designated format for qualitative monitoring.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


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

    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.enabled = os.environ.get("OCRVL_ENABLE_TRANSPARENT_EVAL", "0") == "1"
        self.samples_path = os.environ.get("OCRVL_TRANSPARENT_EVAL_SAMPLES", "")
        self.max_new_tokens = int(os.environ.get("OCRVL_TRANSPARENT_EVAL_MAX_NEW_TOKENS", "128"))
        self.temperature = float(os.environ.get("OCRVL_TRANSPARENT_EVAL_TEMPERATURE", "0.0"))
        self.limit = os.environ.get("OCRVL_TRANSPARENT_EVAL_LIMIT", "")
        self.repo_root = os.environ.get("OCRVL_REPO_ROOT", "/share/project/xiyan/sources/DeepSeek-OCR")

        if self.enabled:
            logger.info(f"[TransparentEval] Enabled - samples: {self.samples_path}")
            logger.info(f"[TransparentEval] Config: max_tokens={self.max_new_tokens}, temp={self.temperature}, limit={self.limit}")
        else:
            logger.info("[TransparentEval] Disabled (OCRVL_ENABLE_TRANSPARENT_EVAL != 1)")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Run transparent evaluation after standard evaluation completes."""
        if not self.enabled:
            return

        # CRITICAL: All ranks must participate in inference for FSDP
        # Only rank 0 loads metadata and saves results, but all ranks must run model.generate()

        is_main = self._is_main_process()

        # DEBUG: Check what's in kwargs
        if is_main:
            print(f"[DEBUG] kwargs keys: {kwargs.keys()}", flush=True)
            if 'model' in kwargs:
                print(f"[DEBUG] model in kwargs: {type(kwargs['model'])}", flush=True)

        # Load samples (all ranks need this for FSDP)
        metadata_path = Path(self.repo_root) / "OCRVL/llamafactory/data/ocrvl_transparent_eval.metadata.json"
        if not metadata_path.exists():
            if is_main:
                logger.warning(f"[TransparentEval] Metadata not found: {metadata_path}")
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        samples = metadata['samples']
        if self.limit:
            samples = samples[:int(self.limit)]

        if is_main:
            logger.info(f"[TransparentEval] Running inference on {len(samples)} samples at step {state.global_step}")

        # All ranks run inference (required for FSDP)
        results = self._run_inference(samples, state.global_step, kwargs.get('model'))

        # Only rank 0 saves results
        if is_main:
            eval_results_dir = Path(args.output_dir) / "eval_results"
            self._save_results(eval_results_dir, results, state.global_step)
            logger.info(f"[TransparentEval] Completed {len(results)}/{len(samples)} samples")

    def _run_inference(self, samples: List[Dict], global_step: int, model=None) -> List[Dict]:
        """Run inference on evaluation samples using LlamaFactory data collator approach.

        Note: All ranks must call this method for FSDP to work properly.

        Args:
            samples: List of evaluation samples
            global_step: Current training step
            model: Optional model to use (from kwargs). If None, uses self.model
        """
        # FSDP handling: Detect FSDP and set synced_gpus=True for generation
        # This is how transformers Seq2SeqTrainer handles FSDP evaluation
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers.integrations import is_fsdp_managed_module
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Use provided model or fall back to self.model
        inference_model = model if model is not None else self.model

        # Detect if model is FSDP-managed (works even with use_orig_params=True)
        is_fsdp = is_fsdp_managed_module(inference_model)

        if rank == 0:
            print(f"[DEBUG] FSDP managed: {is_fsdp}, using model from kwargs: {model is not None}", flush=True)

        # DON'T put model in eval mode with FSDP - might interfere with parameter gathering
        # Standard evaluation in transformers doesn't explicitly call .eval() before generate
        # inference_model.eval()

        results = []
        is_main = self._is_main_process()

        for sample in samples:
            try:
                # Load images
                images = []
                for img_path in sample['images']:
                    full_path = Path(self.repo_root) / img_path
                    if full_path.exists():
                        from PIL import Image as PILImage
                        images.append(PILImage.open(full_path).convert('RGB'))
                    else:
                        if is_main:
                            logger.warning(f"[TransparentEval] Image not found: {full_path}")
                        continue

                if len(images) != 2:
                    if is_main:
                        logger.warning(f"[TransparentEval] Expected 2 images, got {len(images)} for {sample['id']}")
                    continue

                # Use processor.image_processor to get pixel_values and image_grid_thw
                # This is the LlamaFactory way (see Qwen3VLPlugin._get_mm_inputs)
                image_processor = getattr(self.processor, "image_processor", None)
                if image_processor is None:
                    if is_main:
                        logger.warning(f"[TransparentEval] No image_processor found in processor")
                    continue

                # Process images to get pixel_values and image_grid_thw
                mm_inputs = image_processor(images, return_tensors="pt")

                # Build text with vision placeholders
                # Calculate sequence length per image based on image_grid_thw
                image_grid_thw = mm_inputs.get("image_grid_thw")
                merge_length = getattr(image_processor, "merge_size", 2) ** 2

                vision_placeholders = []
                for i in range(len(images)):
                    if image_grid_thw is not None:
                        image_seqlen = image_grid_thw[i].prod().item() // merge_length
                    else:
                        image_seqlen = 100  # Default for OCRVL
                    # Format: <|vision_start|><|image_pad|>*N<|vision_end|>
                    placeholder = "<|vision_start|>" + "<|image_pad|>" * image_seqlen + "<|vision_end|>"
                    vision_placeholders.append(placeholder)

                # Build conversation with vision placeholders
                conversation = [{"role": "user", "content": vision_placeholders[0] + vision_placeholders[1]}]

                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Tokenize
                text_inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=False,
                    add_special_tokens=False
                )

                # Move to device
                device = next(inference_model.parameters()).device
                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                pixel_values = mm_inputs['pixel_values'].to(device)
                image_grid_thw = mm_inputs['image_grid_thw'].to(device)

                # DEBUG: Log shapes (ALL RANKS - critical for FSDP debugging)
                print(f"[DEBUG rank={rank}] input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}", flush=True)
                print(f"[DEBUG rank={rank}] pixel_values shape: {pixel_values.shape}", flush=True)
                print(f"[DEBUG rank={rank}] image_grid_thw shape: {image_grid_thw.shape}, values: {image_grid_thw}", flush=True)

                # Generate using standard pixel_values format
                # CRITICAL: Set synced_gpus=True for FSDP to ensure proper weight gathering
                # This matches how transformers Seq2SeqTrainer handles FSDP

                # FSDP FIX: Use summon_full_params to unshard embedding weights during generation
                # Without this, embedding lookup fails with "'weight' must be 2-D" error
                # because FSDP shards the embedding weight into FlatParameter
                with torch.no_grad():
                    if is_fsdp:
                        # Temporarily unshard all parameters for generation
                        # recurse=True: handle nested FSDP modules
                        # writeback=False: don't update sharded weights (inference only)
                        with FSDP.summon_full_params(inference_model, writeback=False, recurse=True):
                            outputs = inference_model.generate(
                                input_ids=input_ids,
                                pixel_values=pixel_values,
                                image_grid_thw=image_grid_thw,
                                attention_mask=attention_mask,
                                max_new_tokens=self.max_new_tokens,
                                do_sample=self.temperature > 0,
                                temperature=self.temperature if self.temperature > 0 else None,
                                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                synced_gpus=is_fsdp,  # KEY: Enable synced generation for FSDP
                            )
                    else:
                        outputs = inference_model.generate(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            attention_mask=attention_mask,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=self.temperature > 0,
                            temperature=self.temperature if self.temperature > 0 else None,
                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            synced_gpus=is_fsdp,  # KEY: Enable synced generation for FSDP
                        )

                # Decode (remove prompt)
                input_len = input_ids.shape[1]
                generated_ids = outputs[0][input_len:]
                answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                # Get ground truth (already embedded in metadata)
                ground_truth = sample.get('ground_truth', '')
                ground_truth_path = sample.get('ground_truth_path', '')  # For reference only

                # Store result
                results.append({
                    'id': sample['id'],
                    'task': sample['task'],
                    'images': sample['images'],
                    'ground_truth': ground_truth,
                    'ground_truth_path': ground_truth_path,
                    'generated_answer': answer if answer else "[EMPTY]",
                    'step': global_step,
                })

            except Exception as e:
                if is_main:
                    logger.warning(f"[TransparentEval] Failed on sample {sample.get('id', 'unknown')}: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                continue

        # Restore training mode
        inference_model.train()

        return results

    def _save_results(self, checkpoint_dir: Path, results: List[Dict], global_step: int):
        """Save evaluation results in both JSON and human-readable formats."""
        eval_dir = checkpoint_dir / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON (for programmatic analysis)
        json_path = eval_dir / f"step_{global_step}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'step': global_step,
                'timestamp': timestamp,
                'num_samples': len(results),
                'results': results,
            }, f, indent=2, ensure_ascii=False)

        # Save human-readable text
        txt_path = eval_dir / f"step_{global_step}_results.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Transparent Evaluation - Step {global_step}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Samples: {len(results)}\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"Sample {i}: {result['id']}\n")
                f.write(f"Task: {result['task']}\n")
                f.write(f"Images:\n")
                for img in result['images']:
                    f.write(f"  - {img}\n")

                # Display ground truth (no truncation)
                if result['ground_truth']:
                    gt = result['ground_truth']
                    f.write(f"Ground Truth ({len(gt)} chars):\n{gt}\n")

                if result['ground_truth_path']:
                    f.write(f"Ground Truth Path: {result['ground_truth_path']}\n")

                # Display generated answer (truncate if very long)
                gen = result['generated_answer']
                if len(gen) > 500:
                    preview = gen[:250] + f"\n... [truncated {len(gen)-500} chars] ...\n" + gen[-250:]
                    f.write(f"Generated ({len(gen)} chars, truncated for display):\n{preview}\n")
                else:
                    f.write(f"Generated: {gen}\n")

                f.write("-" * 80 + "\n\n")

        logger.info(f"[TransparentEval] Saved results to:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {txt_path}")

    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
