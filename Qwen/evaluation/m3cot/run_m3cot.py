import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Qwen.evaluation.m3cot.dataset_utils import deterministic_limit, dump_image, load_dataset
from Qwen.evaluation.m3cot.eval_utils import evaluate_records, save_eval_summary
from Qwen.evaluation.config import QWEN3_VL_2B_THINKING, resolve_path
from Qwen.evaluation.utils import normalize_chat_api_url
from Qwen.inference.vllm_utils import normalize_media_path, resolve_lora_artifacts
from vllm.v1.engine import LoRARequest

HAS_LORA_REQUEST = True

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _prompt_style_from_env() -> str:
    return os.environ.get("M3COT_PROMPT_STYLE", "cot").strip().lower()


def build_m3cot_prompt(line, dump_image_func, dataset_name):
    del dataset_name
    min_pixels = 768 * 28 * 28
    max_pixels = 5120 * 28 * 28

    image_paths = dump_image_func(line)
    context = str(line.get("context", "") or "").strip()
    question = str(line["question"]).strip()
    choices = list(line["choices"])
    prompt_style = _prompt_style_from_env()

    prompt = ""
    if context:
        prompt += f"[Context]\n{context}\n"
    prompt += f"[Question]\n{question}\n[Choices]\n"
    for i, choice in enumerate(choices):
        prompt += f"({chr(65 + i)}) {choice}\n"
    if prompt_style == "cot":
        prompt += "\nLet's think step-by-step!"

    content = []
    for image_path in image_paths:
        content.append(
            {
                "type": "image",
                "image": image_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )
    content.append({"type": "text", "text": prompt.rstrip()})
    return [{"role": "user", "content": content}]


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = text + "<|im_start|>assistant\n"

    raw_images = []
    for item in messages[0].get("content", []):
        if isinstance(item, dict) and item.get("type") == "image":
            raw_images.append(normalize_media_path(item["image"]))

    min_pixels = getattr(processor.image_processor, "min_pixels", 28 * 28 * 256)
    max_pixels = getattr(processor.image_processor, "max_pixels", 28 * 28 * 2048)

    return {
        "prompt": text,
        "multi_modal_data": {"image": raw_images} if raw_images else {},
        "mm_processor_kwargs": {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    }


def _sanitize_annotation(line_dict: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for key, value in line_dict.items():
        if isinstance(value, np.integer):
            sanitized[key] = int(value)
        elif isinstance(value, np.floating):
            sanitized[key] = float(value)
        elif isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif key == "image" and isinstance(value, dict):
            image_path = value.get("path")
            sanitized[key] = {"path": image_path} if image_path else None
        else:
            sanitized[key] = value
    return sanitized


def _load_rows(args):
    if args.data_dir:
        os.environ["M3COT_DATA_ROOT"] = args.data_dir
    data = load_dataset(args.dataset)
    return deterministic_limit(data, args.num_samples)


def _build_messages(data: pd.DataFrame, dataset_name: str):
    img_root = os.path.join(os.environ.get("LMUData", str(Path(__file__).parent.parent / "data")), "images", "M3CoT")
    os.makedirs(img_root, exist_ok=True)

    def dump_image_func(line):
        return dump_image(line, img_root)

    all_messages = []
    all_annotations = []
    for _, line in tqdm(data.iterrows(), total=len(data), desc="Building prompts"):
        line_dict = line.to_dict()
        all_messages.append(build_m3cot_prompt(line_dict, dump_image_func, dataset_name))
        all_annotations.append(_sanitize_annotation(line_dict))
    return all_messages, all_annotations


def _infer_via_api(args, all_messages, all_annotations):
    records = []
    url = normalize_chat_api_url(args.api_url)
    for idx, (messages, annotation) in enumerate(tqdm(zip(all_messages, all_annotations), total=len(all_messages), desc="API infer")):
        payload = {
            "messages": messages,
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        raw = choice["message"]["content"]
        final = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
        record = {
            "question_id": idx,
            "annotation": annotation,
            "task": "M3CoT",
            "result": {"gen": final, "gen_raw": raw},
            "messages": messages,
        }
        if "usage" in body:
            record["usage"] = body["usage"]
        records.append(record)
    return records


def _infer_via_vllm(args, all_messages, all_annotations):
    processor = AutoProcessor.from_pretrained(args.model_path)
    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "limit_mm_per_prompt": {"image": args.max_images_per_prompt},
        "seed": 42,
    }
    if args.enable_lora and args.lora_path:
        adapter_meta = resolve_lora_artifacts(args.model_path, args.lora_path)
        resolved_lora_rank = adapter_meta["lora_rank"] if adapter_meta["lora_rank"] is not None else 64
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = resolved_lora_rank
        llm_kwargs["max_loras"] = 1

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stop_token_ids=[],
    )
    all_inputs = [prepare_inputs_for_vllm(messages, processor) for messages in all_messages]
    lora_request = None
    if HAS_LORA_REQUEST and args.enable_lora and args.lora_path:
        lora_request = LoRARequest(lora_name=args.lora_name, lora_int_id=1, lora_path=args.lora_path)
    outputs = llm.generate(all_inputs, sampling_params=sampling_params, lora_request=lora_request)

    records = []
    for idx, (annotation, messages, output) in enumerate(zip(all_annotations, all_messages, outputs)):
        raw = output.outputs[0].text
        final = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
        records.append(
            {
                "question_id": idx,
                "annotation": annotation,
                "task": "M3CoT",
                "result": {"gen": final, "gen_raw": raw},
                "messages": messages,
            }
        )
    return records


def run_inference(args):
    args.output_file = resolve_path(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    data = _load_rows(args)
    all_messages, all_annotations = _build_messages(data, args.dataset)

    start = time.time()
    if args.api_url:
        records = _infer_via_api(args, all_messages, all_annotations)
    else:
        records = _infer_via_vllm(args, all_messages, all_annotations)
    elapsed = time.time() - start

    with open(args.output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"✓ M3CoT inference completed in {elapsed:.2f}s")
    print(f"✓ Results saved to {args.output_file}")


def run_evaluation(args):
    args.input_file = resolve_path(args.input_file)
    args.output_file = resolve_path(args.output_file)
    with open(args.input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if args.limit is not None and args.limit > 0 and args.limit < len(records):
        records = records[:args.limit]
    summary = evaluate_records(records)
    save_eval_summary(args.output_file, summary)
    print(json.dumps(summary, indent=2))
    print(f"✓ Saved evaluation results to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="M3CoT benchmark runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, default=str(QWEN3_VL_2B_THINKING))
    infer_parser.add_argument("--dataset", type=str, default="M3CoT")
    infer_parser.add_argument("--data-dir", type=str, default=None)
    infer_parser.add_argument("--output-file", type=str, required=True)
    infer_parser.add_argument("--num-samples", type=int, default=None)
    infer_parser.add_argument("--prompt-style", type=str, default=_prompt_style_from_env(), choices=["direct", "cot"])
    infer_parser.add_argument("--api-url", type=str, default=None)
    infer_parser.add_argument("--tensor-parallel-size", type=int, default=None)
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    infer_parser.add_argument("--max-model-len", type=int, default=128000)
    infer_parser.add_argument("--max-images-per-prompt", type=int, default=4)
    infer_parser.add_argument("--enable-lora", action="store_true")
    infer_parser.add_argument("--lora-path", type=str, default=None)
    infer_parser.add_argument("--lora-name", type=str, default="default")
    infer_parser.add_argument("--max-new-tokens", type=int, default=32768)
    infer_parser.add_argument("--temperature", type=float, default=0.7)
    infer_parser.add_argument("--top-p", type=float, default=0.8)
    infer_parser.add_argument("--top-k", type=int, default=20)
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    infer_parser.add_argument("--presence-penalty", type=float, default=1.5)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--input-file", type=str, required=True)
    eval_parser.add_argument("--output-file", type=str, required=True)
    eval_parser.add_argument("--dataset", type=str, default="M3CoT")
    eval_parser.add_argument("--data-dir", type=str, default=None)
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.add_argument("--eval-model", type=str, default=None)
    eval_parser.add_argument("--api-type", type=str, default="custom")
    eval_parser.add_argument("--api-url", type=str, default=None)
    eval_parser.add_argument("--api-key", type=str, default=None)
    eval_parser.add_argument("--nproc", type=int, default=1)

    args = parser.parse_args()
    if getattr(args, "prompt_style", None):
        os.environ["M3COT_PROMPT_STYLE"] = args.prompt_style
    if args.command == "infer" and args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count() or 1
    if args.command == "infer":
        run_inference(args)
    elif args.command == "eval":
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
