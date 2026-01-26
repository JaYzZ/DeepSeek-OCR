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
import shutil
import sys
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

logger = logging.getLogger(__name__)

# Add OCRVL scripts to path for composite image generation
_REPO_ROOT = Path(__file__).parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "OCRVL" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Import composite image generation function
try:
    from generate_composite_images import generate_composite_images
except ImportError as e:
    logger.warning(f"Could not import composite image generation: {e}")
    generate_composite_images = None


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
        metadata_path = Path(self.repo_root) / "OCRVL/data/ocrvl_transparent_eval.metadata.json"
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
        jsonl_path = Path(self.repo_root) / "OCRVL/data/ocrvl_transparent_eval.jsonl"
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

        if is_main:
            logger.info(f"[TransparentEval] Running inference on {len(samples)} samples at step {state.global_step}")

        # All ranks run inference (required for FSDP)
        results = self._run_inference(samples, state.global_step, kwargs.get('model'))

        # Only rank 0 saves results
        if is_main:
            # Use output_dir directly - _save_results will create eval_results/ subdir
            self._save_results(Path(args.output_dir), results, state.global_step)
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
                        images.append(Image.open(full_path).convert('RGB'))
                    else:
                        if is_main:
                            logger.warning(f"[TransparentEval] Image not found: {full_path}")
                        continue

                # Support 1 or 2 images (1 for caption/ocr, 2 for VQA)
                if len(images) not in (1, 2):
                    if is_main:
                        logger.warning(f"[TransparentEval] Expected 1 or 2 images, got {len(images)} for {sample['id']}")
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

                # Build conversation with instruction text + vision placeholders
                # For 1-image tasks (caption/ocr): instruction + <image>
                # For 2-image tasks (VQA): instruction + <image><image>
                instruction_text = sample.get('instruction', 'Describe the image:')
                # Remove <image> placeholders if present and strip
                instruction_text = instruction_text.replace('<image>', '').strip()

                if len(images) == 1:
                    # Single image task: instruction + vision placeholder
                    user_content = instruction_text + " " + vision_placeholders[0]
                else:
                    # Two images (VQA): instruction + both vision placeholders
                    user_content = instruction_text + " " + vision_placeholders[0] + vision_placeholders[1]

                conversation = [{"role": "user", "content": user_content}]

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
                result = {
                    'id': sample['id'],
                    'task': sample['task'],
                    'images': sample['images'],
                    'instruction': sample.get('instruction', ''),
                    'ground_truth': ground_truth,
                    'ground_truth_path': ground_truth_path,
                    'generated_answer': answer if answer else "[EMPTY]",
                    'step': global_step,
                }
                # Add question_text for VQA tasks
                if sample.get('question_text'):
                    result['question_text'] = sample['question_text']
                results.append(result)

            except Exception as e:
                if is_main:
                    logger.warning(f"[TransparentEval] Failed on sample {sample.get('id', 'unknown')}: {e}")
                    logger.exception(f"Sample {sample.get('id', 'unknown')} failed")
                continue

        # Restore training mode
        inference_model.train()

        return results

    def _save_results(self, checkpoint_dir: Path, results: List[Dict], global_step: int):
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
                src = Path(self.repo_root) / img_path
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

        # Generate composite images for visual inspection
        self._generate_composite_images(json_path, eval_dir)

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
                repo_root=Path(self.repo_root),
            )

        except Exception as e:
            logger.warning(f"[TransparentEval] Failed to generate composite images: {e}")
            logger.exception("Composite image generation failed")

    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
