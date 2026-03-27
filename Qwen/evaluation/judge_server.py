#!/usr/bin/env python3
"""Local judge server compatible with the evaluation suite."""

import argparse
import errno
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, AutoTokenizer
from utils import (
    infer_tensor_parallel_size,
    parse_cuda_visible_devices,
    pick_compatible_tensor_parallel_size,
)
from vllm import LLM, SamplingParams

app = FastAPI(title="Local Judge Server")

llm = None
chat_formatter = None
config: dict[str, Any] = {}


class JudgeRequest(BaseModel):
    question: str
    reference: str
    prediction: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.0
    stream: bool = False
    n: int = 1

def find_free_port(start_port: int) -> int:
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", port))
                sock.listen(1)
                return port
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                continue
            raise
    return start_port


def build_judge_messages(question: str, reference: str, prediction: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are a strict evaluation judge.\n"
        "Given a question, a reference answer, and a model prediction, decide if the prediction is correct.\n"
        "Judge semantic equivalence rather than exact wording.\n"
        "If the prediction contains multiple candidate answers, judge by the final answer.\n"
        "Respond with JSON only using keys verdict and reason.\n"
        'The verdict must be exactly "CORRECT" or "INCORRECT".'
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{reference}\n\n"
        f"Model Prediction:\n{prediction}\n\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    return chat_formatter.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(part for part in parts if part)
    return str(content)


def normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": normalize_message_content(message.get("content", "")),
            }
        )
    return normalized


def run_chat_completion(
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> tuple[str, dict[str, int]]:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = format_chat_prompt(normalize_chat_messages(messages))
    outputs = llm.generate(
        [prompt],
        sampling_params=SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
        ),
    )
    text = outputs[0].outputs[0].text
    usage = {
        "prompt_tokens": count_tokens(prompt),
        "completion_tokens": count_tokens(text),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return text, usage


def parse_judge_output(text: str) -> tuple[str, Optional[str]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        payload = json.loads(stripped)
        verdict = str(payload.get("verdict", "")).strip().upper()
        reason = payload.get("reason")
        if verdict in {"CORRECT", "INCORRECT"}:
            return verdict, None if reason is None else str(reason).strip()
    except Exception:
        pass

    match = re.search(r"\b(CORRECT|INCORRECT)\b", stripped.upper())
    if match:
        return match.group(1), stripped
    return "UNKNOWN", stripped


def load_chat_formatter(model_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer
    except Exception:
        pass
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def get_counting_tokenizer():
    if chat_formatter is None:
        return None
    if hasattr(chat_formatter, "encode"):
        return chat_formatter
    tokenizer = getattr(chat_formatter, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return tokenizer
    return None


def count_tokens(text: str) -> int:
    tokenizer = get_counting_tokenizer()
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


def load_model(
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> None:
    global llm, chat_formatter, config

    print(f"\n{'=' * 80}")
    print("Loading local judge model")
    print(f"Model: {model_path}")
    print(f"Tensor parallel: {tensor_parallel_size}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Max model len: {max_model_len}")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    chat_formatter = load_chat_formatter(model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
        disable_custom_all_reduce=True,
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    config = {
        "model_path": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": config.get("model_path"),
    }


@app.get("/v1/models")
async def list_models():
    model_id = Path(config.get("model_path") or "unknown").name
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/judge")
async def judge(request: JudgeRequest):
    try:
        raw_response, usage = run_chat_completion(
            build_judge_messages(
                question=request.question,
                reference=request.reference,
                prediction=request.prediction,
            ),
            temperature=0.0,
            max_tokens=256,
        )
        verdict, reason = parse_judge_output(raw_response)
        if verdict not in {"CORRECT", "INCORRECT"}:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "verdict": "UNKNOWN",
                    "correct": False,
                    "error": "Failed to parse judge verdict",
                    "raw_response": raw_response,
                    "usage": usage,
                },
            )

        return {
            "success": True,
            "verdict": verdict,
            "correct": verdict == "CORRECT",
            "reason": reason,
            "raw_response": raw_response,
            "usage": usage,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        raw_response, usage = run_chat_completion(
            request.messages,
            temperature=float(request.temperature or 0.0),
            max_tokens=int(request.max_completion_tokens or request.max_tokens or 4096),
        )
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                }
            },
        )

    model_id = request.model or Path(config.get("model_path") or "unknown").name
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": raw_response},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Local judge server backed by vLLM")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the judge model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8600, help="Server port")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="Tensor parallel size. Use 0 to infer from CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Target GPU memory utilization",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max model length for the judge server",
    )
    args = parser.parse_args()

    resolved_tp = (
        infer_tensor_parallel_size(os.environ.get("CUDA_VISIBLE_DEVICES"), fallback=1)
        if args.tensor_parallel_size <= 0
        else args.tensor_parallel_size
    )
    visible_gpus = parse_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    compatible_tp = pick_compatible_tensor_parallel_size(
        args.model_path,
        resolved_tp,
        capped=True,
    )
    if compatible_tp != resolved_tp:
        print(
            f"Adjusting tensor parallel size from {resolved_tp} to {compatible_tp} "
            f"to match model attention-head divisibility."
        )
        resolved_tp = compatible_tp
        if visible_gpus:
            visible_gpus = visible_gpus[:resolved_tp]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)
    print(f"CUDA_VISIBLE_DEVICES: {','.join(visible_gpus) if visible_gpus else 'not set'}")
    print(f"Tensor parallel (resolved): {resolved_tp}")

    original_port = args.port
    args.port = find_free_port(args.port)
    if args.port != original_port:
        print(f"Port {original_port} in use, auto-selected port: {args.port}")

    load_model(
        model_path=args.model_path,
        tensor_parallel_size=resolved_tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    print(f"\n{'=' * 80}")
    print(f"Starting judge server at http://{args.host}:{args.port}")
    print(f"{'=' * 80}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
