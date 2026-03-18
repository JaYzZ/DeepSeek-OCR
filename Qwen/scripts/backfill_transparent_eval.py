#!/usr/bin/env python3
"""
Backfill transparent evaluation for existing training checkpoints using vLLM with LoRA.

This script loads the base model with vLLM and applies LoRA dynamically via LoRARequest,
matching the approach used in Qwen/evaluation/run_all_benchmarks.py.
"""
import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Qwen.scripts.vllm_utils import (
    apply_runtime_env_for_thinking,
    infer_tensor_parallel_size,
    normalize_checkpoint_name,
    parse_cuda_visible_devices,
)

# Load runtime env before enabling plugins so YAML/shell control VLLM_THINKING.
apply_runtime_env_for_thinking(repo_root=_REPO_ROOT)
existing_plugins = [p.strip() for p in os.environ.get("VLLM_PLUGINS", "").split(",") if p.strip()]
if "vllm_thinking" not in existing_plugins:
    existing_plugins.append("vllm_thinking")
os.environ["VLLM_PLUGINS"] = ",".join(existing_plugins)

from PIL import Image, ImageFont, ImageDraw
from tokenizers import AddedToken
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.engine import LoRARequest
from vllm_thinking.runner_patch import apply_thinking_mode_patch


if os.environ.get("VLLM_THINKING", "0").strip().lower() in {"1", "true", "yes", "on"}:
    apply_thinking_mode_patch()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
_WARNED_KEYS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    logger.warning(message)


def _safe_float(value):
    if value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _serialize_logprob_item(token_id, item):
    """Serialize a vLLM logprob item across vLLM versions."""
    # vLLM objects can be plain floats or objects with logprob/rank/decoded_token.
    logprob = _safe_float(getattr(item, "logprob", item))
    rank = getattr(item, "rank", None)
    decoded_token = getattr(item, "decoded_token", None)
    return {
        "token_id": int(token_id),
        "logprob": logprob,
        "rank": int(rank) if isinstance(rank, int) else rank,
        "decoded_token": decoded_token,
    }


def _serialize_logprobs_for_steps(token_logprobs, max_steps: int, topk: int):
    """Convert vLLM token logprobs into JSON-serializable summaries."""
    if not token_logprobs:
        return []

    serialized = []
    steps = min(len(token_logprobs), max_steps)
    for idx in range(steps):
        step = token_logprobs[idx]
        if not isinstance(step, dict):
            serialized.append({"step": idx, "topk": []})
            continue

        entries = []
        for tid, item in step.items():
            try:
                token_id = int(tid)
            except Exception:
                continue
            entries.append(_serialize_logprob_item(token_id, item))

        entries.sort(key=lambda x: (x["logprob"] is None, -(x["logprob"] or -1e9)))
        serialized.append({"step": idx, "topk": entries[:topk]})
    return serialized


def _extract_post_think_candidates(
    generated_token_ids,
    token_logprobs,
    think_start_id: int,
    probe_token_ids: list[int],
):
    think_positions = [i for i, t in enumerate(generated_token_ids) if t == think_start_id]
    if not think_positions:
        return None
    probe_idx = think_positions[0] + 1
    if probe_idx < 0 or probe_idx >= len(token_logprobs) or not isinstance(token_logprobs[probe_idx], dict):
        return None
    lp_map = token_logprobs[probe_idx]
    candidates = []
    for tok_id in probe_token_ids:
        item = lp_map.get(tok_id)
        if item is None:
            candidates.append({"token_id": tok_id, "logprob": None, "rank": None, "decoded_token": None})
        else:
            candidates.append(_serialize_logprob_item(tok_id, item))
    return {
        "probe_step": probe_idx,
        "chosen_token_id": generated_token_ids[probe_idx] if probe_idx < len(generated_token_ids) else None,
        "candidates": candidates,
    }


def _extract_display_output(
    full_output: str,
) -> str:
    """Return the visible answer span after the final </think> marker."""
    return full_output.rsplit("</think>", 1)[-1].strip()


def _load_target_prob_trace(path: str) -> dict[str, list[dict]]:
    traces: dict[str, list[dict]] = {}
    if not path or not os.path.exists(path):
        return traces
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            req_id = str(rec.get("req_id", ""))
            if not req_id:
                continue
            traces.setdefault(req_id, []).append(rec)
    for req_id in traces:
        traces[req_id].sort(key=lambda x: int(x.get("step", 0)))
    return traces


def _load_target_prob_trace_records(path: str) -> list[dict]:
    records: list[dict] = []
    if not path or not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    run_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {checkpoint_dir}")
    latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
    checkpoint_dirs = [d for d in latest_run.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {latest_run}")
    return max(checkpoint_dirs, key=lambda d: int(d.name.split('-')[1]))


def load_model_vllm(checkpoint_path: Path, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.8):
    """Load model with vLLM - handles LoRA detection and setup."""

    # Fallback base model path if adapter metadata is missing.
    OFFICIAL_BASE_MODEL = "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking"

    # Check for LoRA adapter
    adapter_config_path = checkpoint_path / "adapter_config.json"
    base_model_path = None
    lora_path = None

    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path") or OFFICIAL_BASE_MODEL
        lora_path = str(checkpoint_path)
        logger.info(f"Detected LoRA adapter. Using base model: {base_model_path}")
        logger.info(f"LoRA path: {lora_path}")
    else:
        base_model_path = str(checkpoint_path)
        logger.info(f"Loading base model from {base_model_path}")

    # ============================================================================
    # CRITICAL: Add special tokens for latent thinking BEFORE loading vLLM
    # ============================================================================

    # Load tokenizer and add special tokens
    tokenizer_with_special_tokens = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True
    )

    # Ensure <latent>/<think_sep> are single tokens for inference.
    # Use regular added tokens (not "special") so they can be generated/displayed
    # like <think> and </think>.
    special_tokens = ["<latent>", "<think_sep>"]
    added_count = 0
    for token in special_tokens:
        encoded = tokenizer_with_special_tokens.encode(token, add_special_tokens=False)
        if len(encoded) > 1:
            num_added = tokenizer_with_special_tokens.add_tokens([token], special_tokens=False)
            added_count += num_added

    # If checkpoint tokenizer marks these as special, demote them at runtime.
    # vLLM generation then treats them like regular tokens.
    for token in special_tokens:
        token_id = tokenizer_with_special_tokens.convert_tokens_to_ids(token)
        added = tokenizer_with_special_tokens.added_tokens_decoder.get(token_id)
        if added is not None and getattr(added, "special", False):
            tokenizer_with_special_tokens._tokenizer.add_tokens([AddedToken(token, special=False)])
            logger.info(f"Demoted special token to regular token at runtime: {token} (ID {token_id})")

    if added_count > 0:
        logger.info(f"Added {added_count} regular thinking tokens to tokenizer")

    # Always export unified token IDs used by both training and vLLM plugin.
    for token in special_tokens:
        token_id = tokenizer_with_special_tokens.convert_tokens_to_ids(token)
        logger.info(f"  {token} -> ID {token_id}")
        if token == "<latent>":
            os.environ["QWEN3VL_LATENT_TOKEN_ID"] = str(token_id)
        elif token == "<think_sep>":
            os.environ["QWEN3VL_THINKING_SEP_ID"] = str(token_id)

    # Initialize vLLM
    llm_kwargs = {
        "model": base_model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
        "disable_custom_all_reduce": True, # Key to the distributed inference with mode change
        "disable_log_stats": True,
    }

    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        llm_kwargs["max_loras"] = 1

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Load processor separately (tokenizer doesn't have image_processor)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    logger.info(f"Model loaded with vLLM (thinking mode enabled)")
    return llm, tokenizer, processor, lora_path


def run_evaluation(
    llm,
    tokenizer,
    processor,
    eval_metadata: Path,
    lora_path: str = None,
    max_samples: int = None,
    repo_root: Path = None,
    logprobs_k: int = 10,
    logprobs_max_steps: int = 8,
    target_prob_trace_path: str | None = None,
):
    if repo_root is None:
        repo_root = _REPO_ROOT

    # Load metadata - support both JSON (with samples array) and JSONL format
    with open(eval_metadata, 'r') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            # JSON format with samples array directly
            samples = json.load(f)
            samples = samples[:max_samples] if max_samples else samples
        else:
            # JSON format with samples key OR JSONL format
            try:
                metadata = json.load(f)
                if isinstance(metadata, dict) and 'samples' in metadata:
                    samples = metadata['samples'][:max_samples] if max_samples else metadata['samples']
                else:
                    raise ValueError("Unknown JSON format")
            except json.JSONDecodeError:
                # JSONL format - each line is a sample
                f.seek(0)
                samples = []
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line.strip()))
                samples = samples[:max_samples] if max_samples else samples

    logger.info(f"Running transparent eval on {len(samples)} samples...")

    # Build full_samples dict from samples for ground_truth lookup
    full_samples = {}
    for sample in samples:
        sample_id = sample.get('id')
        if sample_id:
            # Extract ground_truth from assistant message
            if 'ground_truth' not in sample or not sample.get('ground_truth'):
                for msg in sample.get('messages', []):
                    if msg.get('role') == 'assistant':
                        gt_content = msg.get('content', '')
                        # Filter out thinking tags
                        if "<think>" in gt_content and "</think>" in gt_content:
                            gt_content = gt_content.split("</think>", 1)[-1].strip()
                        if gt_content and gt_content != "Let me analyze this math problem step by step.":
                            sample['ground_truth'] = gt_content
                        break
            full_samples[sample_id] = sample

    results = []
    debug_samples = []
    start_time = time.time()

    # Prepare LoRA request if LoRA is enabled
    lora_request = None
    lora_name = "backfill_lora"
    if lora_path:
        lora_name = f"backfill_{normalize_checkpoint_name(lora_path)}"
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=1,
            lora_path=lora_path,
        )
        logger.info(f"Using LoRA: {lora_name} from {lora_path}")

    # ============ BATCH MODE: Pre-process all samples first ============
    logger.info(f"Pre-processing {len(samples)} samples for batch inference...")
    batch_inputs = []
    valid_indices = []

    for i, sample in enumerate(samples):
        try:
            # Load images
            images = []
            for img_path in sample['images']:
                full_path = repo_root / img_path
                if full_path.exists():
                    images.append(Image.open(full_path).convert('RGB'))
                else:
                    logger.warning(f"Image not found: {full_path}")
                    continue

            if len(images) not in (1, 2):
                logger.warning(f"Expected 1 or 2 images, got {len(images)}")
                continue

            # Extract instruction from messages (user message content)
            instruction_text = ""
            if 'messages' in sample and isinstance(sample['messages'], list):
                user_msg = next((m for m in sample['messages'] if m.get('role') == 'user'), None)
                if user_msg:
                    content = user_msg.get('content', '')
                    if isinstance(content, str):
                        # Extract text, removing <image> markers
                        instruction_text = content.replace('<image>', '').strip()
                    elif isinstance(content, list):
                        # Multi-modal format: list of {"type": "text/image", ...}
                        for item in content:
                            if item.get('type') == 'text':
                                instruction_text = item.get('text', '').replace('<image>', '').strip()
                                break
            if not instruction_text:
                instruction_text = "Describe the image."

            # Extract ground_truth from assistant message
            ground_truth = ""
            if 'messages' in sample and isinstance(sample['messages'], list):
                assistant_msg = next((m for m in sample['messages'] if m.get('role') == 'assistant'), None)
                if assistant_msg:
                    gt_content = assistant_msg.get('content', '')
                    if isinstance(gt_content, str):
                        # Filter out thinking tags for ground truth
                        if "<think>" in gt_content and "</think>" in gt_content:
                            gt_content = gt_content.split("</think>", 1)[-1].strip()
                        if gt_content and gt_content != "Let me analyze this math problem step by step.":
                            ground_truth = gt_content

            # Build conversation for vLLM - let vLLM handle everything
            # Use official vLLM format: prompt with image placeholders, raw images passed directly
            messages = [{"role": "user", "content": []}]

            # Add images with proper placeholder
            for idx, img in enumerate(images):
                messages[0]["content"].append({"type": "image", "image": img})

            # Add text instruction
            messages[0]["content"].append({"type": "text", "text": instruction_text})

            # Apply chat template to get prompt with proper placeholders
            # This will insert <|vision_start|><|image_pad|><|vision_end|> for each image
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text = text + "<|im_start|>assistant\n"
            if os.environ.get("VLLM_FORCE_THINK", "0") == "1":
                text = text + "<think>"

            # Get min/max pixels from processor for vLLM image processing
            min_pixels = getattr(processor.image_processor, 'min_pixels', 28 * 28 * 256)
            max_pixels = getattr(processor.image_processor, 'max_pixels', 28 * 28 * 2048)

            # Pass RAW images directly - let vLLM handle everything
            mm_data = {}
            if images:
                mm_data['image'] = images  # Raw PIL images

            batch_inputs.append({
                'prompt': text,
                'multi_modal_data': mm_data,
                'mm_processor_kwargs': {
                    'min_pixels': min_pixels,
                    'max_pixels': max_pixels,
                },
                'sample': sample,
                'instruction_text': instruction_text,
                'ground_truth': ground_truth,
            })
            valid_indices.append(i)

        except Exception as e:
            logger.warning(f"Failed to pre-process sample {sample.get('id', 'unknown')}: {e}")
            continue

    logger.info(f"Pre-processed {len(batch_inputs)}/{len(samples)} samples")

    # ============ BATCH INFERENCE: All at once ============
    if not batch_inputs:
        logger.warning("No valid samples to process")
        return results

    # Prepare batch inputs for vLLM
    llm_inputs = [
        {
            'prompt': inp['prompt'],
            'multi_modal_data': inp['multi_modal_data'],
            'mm_processor_kwargs': inp['mm_processor_kwargs']
        }
        for inp in batch_inputs
    ]

    # Sampling params - match TransparentEvalCallback
    logger.info(
        f"Logprob capture: topk={logprobs_k}, max_steps={logprobs_max_steps} "
        "(target token probs come from runner trace)"
    )

    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=0.0,  # Greedy for reproducibility
        stop_token_ids=[tokenizer.eos_token_id],
        skip_special_tokens=False,
        logprobs=logprobs_k,
    )

    logger.info(f"Running batch inference on {len(llm_inputs)} samples...")
    if lora_request:
        outputs = llm.generate(llm_inputs, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(llm_inputs, sampling_params)

    # ============ Post-process results ============
    target_traces = _load_target_prob_trace(target_prob_trace_path or "")
    if target_prob_trace_path:
        trace_records = sum(len(v) for v in target_traces.values())
        logger.info(
            "Target prob trace loaded: path=%s, requests=%d, records=%d",
            target_prob_trace_path,
            len(target_traces),
            trace_records,
        )
        if trace_records == 0:
            logger.warning(
                "Target prob trace is empty. step_target_token_logprobs_runner will be empty; "
                "fallback step_target_token_logprobs only reflects top-k visibility."
            )
    think_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    think_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))
    latent_id = int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669"))
    think_sep_id = int(os.environ.get("QWEN3VL_THINKING_SEP_ID", "151670"))

    for i, (inp, output) in enumerate(zip(batch_inputs, outputs)):
        sample = inp['sample']
        sample_elapsed = time.time() - start_time

        generated_text = output.outputs[0].text
        generated_token_ids = list(output.outputs[0].token_ids or [])
        token_logprobs = output.outputs[0].logprobs or []
        full_output = generated_text.strip()
        req_id = str(getattr(output, "request_id", ""))

        think_start_positions = [idx for idx, tid in enumerate(generated_token_ids) if tid == think_start_id]
        think_end_positions = [idx for idx, tid in enumerate(generated_token_ids) if tid == think_end_id]
        latent_positions = [idx for idx, tid in enumerate(generated_token_ids) if tid == latent_id]
        think_sep_positions = [idx for idx, tid in enumerate(generated_token_ids) if tid == think_sep_id]

        display_output = _extract_display_output(full_output)

        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] [{i+1}/{len(batch_inputs)}] {sample['id']}: {display_output[:60]}...")

        think_span_token_count = 0
        if think_start_positions and think_end_positions:
            start = think_start_positions[0]
            end = think_end_positions[-1]
            if end > start:
                think_span_token_count = max(0, end - start - 1)

        # Focus diagnostics: immediately after first <think>, compare </think> vs <latent>.
        probe_token_ids = [think_start_id, think_end_id, latent_id, think_sep_id]
        probe_name_by_id = {
            think_start_id: "think_start",
            think_end_id: "think_end",
            latent_id: "latent",
            think_sep_id: "think_sep",
        }
        probe_text_by_id = {
            think_start_id: "<think>",
            think_end_id: "</think>",
            latent_id: "<latent>",
            think_sep_id: "<think_sep>",
        }
        post_think_token_logprobs = _extract_post_think_candidates(
            generated_token_ids=generated_token_ids,
            token_logprobs=token_logprobs,
            think_start_id=think_start_id,
            probe_token_ids=probe_token_ids,
        )

        runner_by_step: dict[int, dict] = {}
        for rec in target_traces.get(req_id, []):
            try:
                runner_by_step[int(rec.get("step", 0))] = rec.get("target_logprobs", {}) or {}
            except Exception:
                continue

        step_target_logprobs = []
        max_steps = min(max(len(token_logprobs), len(runner_by_step)), logprobs_max_steps)
        for step_idx in range(max_steps):
            if step_idx < len(token_logprobs) and isinstance(token_logprobs[step_idx], dict):
                step_map = token_logprobs[step_idx]
            else:
                step_map = {}
            step_entry = {"step": step_idx, "tokens": []}
            for tok_id in probe_token_ids:
                token_name = probe_name_by_id[tok_id]
                runner_item = runner_by_step.get(step_idx, {}).get(token_name)
                if isinstance(runner_item, dict):
                    step_entry["tokens"].append(
                        {
                            "token_id": int(tok_id),
                            "logprob": _safe_float(runner_item.get("logprob")),
                            "rank": runner_item.get("rank"),
                            "decoded_token": probe_text_by_id.get(int(tok_id), tokenizer.decode([int(tok_id)], skip_special_tokens=False)),
                        }
                    )
                    continue

                item = step_map.get(tok_id)
                if item is None:
                    step_entry["tokens"].append(
                        {
                            "token_id": int(tok_id),
                            "logprob": None,
                            "rank": None,
                            "decoded_token": probe_text_by_id.get(int(tok_id), tokenizer.decode([int(tok_id)], skip_special_tokens=False)),
                        }
                    )
                else:
                    serialized = _serialize_logprob_item(tok_id, item)
                    if serialized.get("decoded_token") is None:
                        serialized["decoded_token"] = probe_text_by_id.get(int(tok_id), tokenizer.decode([int(tok_id)], skip_special_tokens=False))
                    step_entry["tokens"].append(serialized)
            step_target_logprobs.append(step_entry)

        token_logprobs_topk = _serialize_logprobs_for_steps(
            token_logprobs, max_steps=max_steps, topk=logprobs_k
        )
        existing_steps = {int(x.get("step", -1)) for x in token_logprobs_topk}
        for step_idx in range(max_steps):
            if step_idx not in existing_steps:
                token_logprobs_topk.append({"step": step_idx, "topk": []})
        token_logprobs_topk.sort(key=lambda x: int(x.get("step", 0)))
        for step_entry in token_logprobs_topk:
            step_idx = int(step_entry.get("step", 0))
            existing_ids = {int(x.get("token_id")) for x in step_entry.get("topk", [])}
            for entry in step_entry.get("topk", []):
                tok_id = int(entry.get("token_id"))
                if tok_id in probe_text_by_id and not entry.get("decoded_token"):
                    entry["decoded_token"] = probe_text_by_id[tok_id]
            for special in step_target_logprobs[step_idx]["tokens"]:
                tok_id = int(special["token_id"])
                if tok_id not in existing_ids:
                    step_entry["topk"].append(special)

        results.append({
            'id': sample['id'],
            'task': sample['task'],
            'images': sample['images'],
            'instruction': inp['instruction_text'],
            'ground_truth': inp['ground_truth'],
            'generated_answer': full_output if full_output else "[EMPTY]",
            'generated_answer_display': display_output if display_output else "[EMPTY]",
        })
        debug_samples.append({
            'id': sample['id'],
            'task': sample['task'],
            'request_id': getattr(output, "request_id", None),
            'debug': {
                'request_id': getattr(output, "request_id", None),
                'generated_token_ids': generated_token_ids,
                'generated_num_tokens': len(generated_token_ids),
                'prompt_num_tokens': len(getattr(output, "prompt_token_ids", []) or []),
                'think_start_positions': think_start_positions,
                'think_end_positions': think_end_positions,
                'latent_positions': latent_positions,
                'think_sep_positions': think_sep_positions,
                'think_span_token_count': think_span_token_count,
                'token_logprobs_topk': token_logprobs_topk,
                'post_think_token_logprobs': post_think_token_logprobs,
                'step_target_token_logprobs': step_target_logprobs,
                'step_target_token_logprobs_runner': target_traces.get(req_id, []),
            },
        })

    elapsed = time.time() - start_time
    logger.info(f"Generated {len(results)}/{len(samples)} samples in {elapsed:.1f}s ({elapsed/len(results):.1f}s/sample)")

    return results, debug_samples


def _text_width(text: str, font) -> int:
    bbox = font.getbbox(text or " ")
    return bbox[2] - bbox[0]


def _line_height(font, extra_spacing: int = 4) -> int:
    bbox = font.getbbox("Ag")
    return (bbox[3] - bbox[1]) + extra_spacing


def _wrap_paragraph(paragraph: str, font, max_width: int) -> list[str]:
    if not paragraph:
        return [""]

    words = paragraph.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_line = ""

    for word in words:
        candidate = f"{current_line} {word}".strip()
        if current_line and _text_width(candidate, font) <= max_width:
            current_line = candidate
            continue
        if not current_line and _text_width(word, font) <= max_width:
            current_line = word
            continue
        if current_line:
            lines.append(current_line)
            current_line = ""

        if _text_width(word, font) <= max_width:
            current_line = word
            continue

        chunk = ""
        for ch in word:
            candidate = f"{chunk}{ch}"
            if chunk and _text_width(candidate, font) > max_width:
                lines.append(chunk)
                chunk = ch
            else:
                chunk = candidate
        current_line = chunk

    if current_line:
        lines.append(current_line)

    return lines or [""]


def wrap_text(text: str, font, max_width: int) -> list[str]:
    """Wrap text to fit within max_width while preserving explicit newlines."""
    if not text:
        return []

    wrapped_lines: list[str] = []
    for paragraph in str(text).splitlines():
        wrapped_lines.extend(_wrap_paragraph(paragraph, font, max_width))
    return wrapped_lines or [""]


def save_results(
    results: list,
    debug_samples: list,
    checkpoint_dir: Path,
    repo_root: Path = None,
    target_prob_trace_path: str | None = None,
):
    eval_dir = checkpoint_dir / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON
    json_path = eval_dir / f"backfill_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': timestamp, 'num_samples': len(results), 'results': results}, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(results)} results to {json_path}")

    # Save compact debug bundle and clean up temporary trace file.
    trace_records = _load_target_prob_trace_records(target_prob_trace_path or "")
    trace_by_req: dict[str, list[dict]] = {}
    for rec in trace_records:
        req_id = str(rec.get("req_id", ""))
        if not req_id:
            continue
        trace_by_req.setdefault(req_id, []).append(rec)
    for req_id in trace_by_req:
        trace_by_req[req_id].sort(key=lambda x: int(x.get("step", 0)))

    debug_path = eval_dir / f"debug_{timestamp}.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "num_samples": len(results),
                "source_backfill": str(json_path),
                "source_debug_trace_file": str(target_prob_trace_path) if target_prob_trace_path else None,
                "num_target_prob_records": len(trace_records),
                "sample_debug": debug_samples,
                "target_prob_trace_records": trace_records,
                "target_prob_trace_by_req": trace_by_req,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved debug bundle to %s (records=%d)", debug_path, len(trace_records))

    if target_prob_trace_path and os.path.exists(target_prob_trace_path):
        os.remove(target_prob_trace_path)
        logger.info("Removed temporary debug trace file: %s", target_prob_trace_path)

    # Generate composite images
    composite_dir = eval_dir / f"backfill_{timestamp}_composite"
    composite_dir.mkdir(parents=True, exist_ok=True)

    if repo_root is None:
        repo_root = Path.cwd()

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
            # Handle both absolute and relative paths
            if os.path.isabs(img_path):
                full_path = Path(img_path)
            else:
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
                except (ValueError, IndexError) as e:
                    _warn_once(
                        "bbox_parse_failure",
                        f"[Backfill] Failed to parse bbox instruction; skipping overlay. Error: {e}",
                    )

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
        full_output = str(result.get('generated_answer', '') or '')
        generated = str(
            result.get('generated_answer_display')
            or _extract_display_output(full_output)
            or full_output
            or '[EMPTY]'
        )

        text_width = img_width - 2 * padding
        body_line_height = _line_height(text_font, extra_spacing=2)
        section_gap = 10
        label_gap = 20

        instruction_lines = wrap_text(instruction, text_font, text_width) if instruction else []
        gt_lines = wrap_text(ground_truth, text_font, text_width)
        gen_lines = wrap_text(generated, text_font, text_width)

        text_area_height = 20
        if instruction_lines:
            text_area_height += 30
            text_area_height += len(instruction_lines) * body_line_height
            text_area_height += section_gap
        text_area_height += 30
        text_area_height += len(gt_lines) * body_line_height
        text_area_height += section_gap
        text_area_height += 30
        text_area_height += len(gen_lines) * body_line_height
        text_area_height += 20

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
            y_offset += label_gap
            for line in instruction_lines:
                draw.text((padding, y_offset), line, fill='black', font=text_font)
                y_offset += body_line_height

        y_offset += section_gap

        # Draw ground truth
        draw.text((padding, y_offset), "Ground Truth:", fill='#4CAF50', font=label_font)
        y_offset += label_gap
        for line in gt_lines:
            draw.text((padding, y_offset), line, fill='black', font=text_font)
            y_offset += body_line_height

        y_offset += section_gap

        # Draw answer-only model output
        draw.text((padding, y_offset), "Answer:", fill='#FF9800', font=label_font)
        y_offset += label_gap
        for line in gen_lines:
            draw.text((padding, y_offset), line, fill='black', font=text_font)
            y_offset += body_line_height

        composite.save(output_path, optimize=True, quality=95)
        logger.info(f"Generated composite: {output_path.name}")

    logger.info(f"Generated {len(results)} composite images to {composite_dir}/")


def main():
    start_time = datetime.now()
    logger.info(f"=== BACKFILL START: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    parser = argparse.ArgumentParser(description="Backfill transparent eval using vLLM with LoRA")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=0,
        help="Tensor parallel size. Use 0 to auto-infer from CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--logprobs_k", type=int, default=10)
    parser.add_argument("--logprobs_max_steps", type=int, default=8)
    args = parser.parse_args()
    apply_runtime_env_for_thinking(repo_root=_REPO_ROOT, logger=logger)

    # Log GPU info
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_gpus = parse_cuda_visible_devices(cuda_visible)
    resolved_tp = (
        infer_tensor_parallel_size(cuda_visible, fallback=1)
        if args.tensor_parallel_size <= 0
        else args.tensor_parallel_size
    )
    logger.info(f"CUDA_VISIBLE_DEVICES: {','.join(visible_gpus) if visible_gpus else 'not set'}")
    logger.info(f"Tensor parallel (resolved): {resolved_tp}")

    checkpoint_dir = Path(args.checkpoint_dir)
    if args.checkpoint:
        checkpoint_path = checkpoint_dir / args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    checkpoint_path = checkpoint_path.resolve()

    logger.info(f"Using checkpoint: {checkpoint_path}")

    if args.metadata:
        eval_metadata = Path(args.metadata)
    else:
        eval_metadata = _REPO_ROOT / "Qwen/data/transparent_eval.jsonl"

    if not eval_metadata.exists():
        raise FileNotFoundError(f"Metadata not found: {eval_metadata}")

    # Set checkpoint path for VAE loading in vLLM plugin
    os.environ["VLLM_LORA_CHECKPOINT_PATH"] = str(checkpoint_path)
    trace_dir = checkpoint_path / "eval_results"
    trace_dir.mkdir(parents=True, exist_ok=True)
    target_prob_trace_path = str(
        (trace_dir / f"_debug_trace_{int(time.time())}_{os.getpid()}.jsonl").resolve()
    )
    os.environ["VLLM_THINKING_TARGET_PROB_PATH"] = target_prob_trace_path
    os.environ["VLLM_THINKING_TARGET_PROB_MAX_STEPS"] = str(args.logprobs_max_steps)
    if os.path.exists(target_prob_trace_path):
        os.remove(target_prob_trace_path)

    llm, tokenizer, processor, lora_path = load_model_vllm(
        checkpoint_path,
        tensor_parallel_size=resolved_tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    results, debug_samples = run_evaluation(
        llm=llm,
        tokenizer=tokenizer,
        processor=processor,
        eval_metadata=eval_metadata,
        lora_path=lora_path,
        max_samples=args.max_samples,
        repo_root=_REPO_ROOT,
        logprobs_k=args.logprobs_k,
        logprobs_max_steps=args.logprobs_max_steps,
        target_prob_trace_path=target_prob_trace_path,
    )
    save_results(
        results=results,
        debug_samples=debug_samples,
        checkpoint_dir=checkpoint_path,
        repo_root=_REPO_ROOT,
        target_prob_trace_path=target_prob_trace_path,
    )
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"=== BACKFILL COMPLETE: {end_time.strftime('%Y-%m-%d %H:%M:%S')} | Duration: {duration} ===")


if __name__ == "__main__":
    main()
