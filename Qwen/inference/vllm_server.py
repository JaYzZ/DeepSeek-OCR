#!/usr/bin/env python3
"""
vLLM Server with Thinking Mode Enabled

A simple HTTP server that loads vLLM once and serves inference requests.
Supports LoRA and continuous latent AR mode for thinking.

Usage:
    # Start server
    python vllm_server.py --model /path/to/model --port 8016

    # Query
    curl -X POST http://localhost:8016/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": [{"type": "image", "image": "https://example.com/img.jpg"}, {"type": "text", "text": "What is in this image?"}]}]}'
"""

import argparse
import errno
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from Qwen.inference.vllm_utils import (
    apply_runtime_env_for_thinking,
    append_forced_think_prompt,
    infer_tensor_parallel_size,
    normalize_media_path,
    normalize_checkpoint_name,
    parse_cuda_visible_devices,
    prepare_inference_tokenizer,
    resolve_lora_artifacts,
    vllm_thinking_enabled,
)


# Set vLLM multiprocessing method BEFORE importing vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Load runtime env before enabling plugins so YAML can control VLLM_THINKING.
apply_runtime_env_for_thinking(repo_root=_REPO_ROOT)
THINKING_MODE_ENABLED = vllm_thinking_enabled(default="0")
existing_plugins = [p.strip() for p in os.environ.get("VLLM_PLUGINS", "").split(",") if p.strip()]
if "vllm_thinking" not in existing_plugins:
    existing_plugins.append("vllm_thinking")
os.environ["VLLM_PLUGINS"] = ",".join(existing_plugins)

# Import decode patch BEFORE vLLM to support both continuous AR and discrete
# latent carry.
from vllm_thinking.runner_patch import apply_thinking_mode_patch

from vllm import LLM, SamplingParams
from vllm.multimodal.hasher import MultiModalHasher
from vllm.v1.engine import LoRARequest
from transformers import AutoProcessor, AutoTokenizer

HAS_LORA_REQUEST = True

app = FastAPI(title="vLLM Thinking Server")

# Global state
llm = None
processor = None
lora_request = None
config = {}


def _patch_multimodal_none_hashing() -> None:
    original = MultiModalHasher.serialize_item.__func__
    if getattr(MultiModalHasher, "_qwen_none_patch_applied", False):
        return

    def _serialize_item(cls, obj: object):
        if obj is None:
            return (b"<none>",)
        return original(cls, obj)

    MultiModalHasher.serialize_item = classmethod(_serialize_item)
    MultiModalHasher._qwen_none_patch_applied = True


_patch_multimodal_none_hashing()


class ChatMessage(BaseModel):
    role: str
    content: Any  # Can be string or list of dicts


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 8192
    n: int = 1
    top_p: float = 1.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False


class ChatCompletionBatchRequest(BaseModel):
    requests: List[ChatCompletionRequest]


def _make_sampling_params(
    *,
    temperature: float,
    max_tokens: int,
    n: int = 1,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> SamplingParams:
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        top_p=top_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        stop_token_ids=[151643, 151645],
        skip_special_tokens=False,
    )


def _serialize_chat_output(output) -> Dict[str, Any]:
    choices = []
    total_completion_tokens = 0
    for idx, candidate in enumerate(output.outputs):
        total_completion_tokens += len(candidate.token_ids)
        choices.append(
            {
                "index": idx,
                "message": {
                    "role": "assistant",
                    "content": candidate.text,
                },
                "finish_reason": candidate.finish_reason if hasattr(candidate, "finish_reason") else "stop",
            }
        )

    return {
        "id": "chatcmpl-" + str(int(time.time())),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config.get("model_path"),
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": total_completion_tokens,
            "total_tokens": len(output.prompt_token_ids) + total_completion_tokens,
        },
    }


def _request_sampling_signature(request: ChatCompletionRequest) -> Tuple[float, int, int, float, float, float]:
    return (
        request.temperature,
        request.max_tokens,
        request.n,
        request.top_p,
        request.presence_penalty,
        request.repetition_penalty,
    )


def run_llm_generation_batch(requests: List[ChatCompletionRequest]):
    global llm, processor, lora_request

    if not requests:
        return []

    signature = _request_sampling_signature(requests[0])
    for request in requests[1:]:
        if _request_sampling_signature(request) != signature:
            raise ValueError("Batched chat completions require identical sampling parameters")

    vllm_inputs = []
    for request in requests:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        vllm_inputs.append(prepare_inputs_for_vllm(messages, processor))

    sampling_params = _make_sampling_params(
        temperature=requests[0].temperature,
        max_tokens=requests[0].max_tokens,
        n=requests[0].n,
        top_p=requests[0].top_p,
        presence_penalty=requests[0].presence_penalty,
        repetition_penalty=requests[0].repetition_penalty,
    )
    return llm.generate(
        vllm_inputs,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": config.get("model_path"),
        "lora": config.get("lora_path"),
        "thinking_enabled": config.get("thinking_enabled", False),
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing."""
    model_id = Path(config.get("requested_model_path") or config.get("model_path") or "unknown").name
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


@app.get("/stats")
async def stats():
    """Get server statistics."""
    return {
        "model_path": config.get("model_path"),
        "lora_path": config.get("lora_path"),
        "tensor_parallel_size": config.get("tensor_parallel_size"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization"),
        "thinking_enabled": config.get("thinking_enabled", False),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completion endpoint."""
    global llm, processor, lora_request

    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        output = run_llm_generation_batch([request])[0]

        # DEBUG: Print raw output to see if thinking tokens exist (only if VLLM_DEBUG=1)
        response_text = output.outputs[0].text
        if os.environ.get("VLLM_DEBUG", "0") == "1":
            print(f"[DEBUG] Raw response_text: {repr(response_text[:500])}")

        return _serialize_chat_output(output)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions_batch")
async def chat_completions_batch(request: ChatCompletionBatchRequest):
    """Batch chat completion endpoint for benchmark inference."""
    global llm

    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.requests:
        raise HTTPException(status_code=400, detail="No requests provided")

    try:
        outputs = run_llm_generation_batch(request.requests)
        return {
            "object": "list",
            "data": [_serialize_chat_output(output) for output in outputs],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: Dict):
    """Legacy completion endpoint."""
    raise HTTPException(status_code=501, detail="Use /v1/chat/completions instead")


def prepare_inputs_for_vllm(messages, processor):
    """Prepare messages for vLLM input."""
    # Use processor to convert messages to tokenizer format
    # This handles image URLs, base64, etc.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Manually add generation prompt as per repo's common practice
    text = text + "<|im_start|>assistant\n"
    text = append_forced_think_prompt(text, default="0")

    # Extract media from messages and, if provided, preserve benchmark-specific
    # min/max pixel constraints (MathVision/RealWorldQA/etc).
    images = []
    videos = []
    min_pixels = None
    max_pixels = None
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    img = item.get("image")
                    if img:
                        # ODinW jobs may emit "file:///abs/path". vLLM/HF processors generally
                        # expect a plain filesystem path for local files.
                        images.append(normalize_media_path(img))
                    # Preserve any per-image resolution constraints if present.
                    if min_pixels is None and "min_pixels" in item:
                        min_pixels = item.get("min_pixels")
                    if max_pixels is None and "max_pixels" in item:
                        max_pixels = item.get("max_pixels")
                elif item.get("type") == "video":
                    vid = item.get("video")
                    if vid:
                        videos.append(normalize_media_path(vid))

    if images:
        # Multi-modal input
        if min_pixels is None:
            min_pixels = getattr(processor.image_processor, "min_pixels", 28 * 28 * 256)
        if max_pixels is None:
            max_pixels = getattr(processor.image_processor, "max_pixels", 28 * 28 * 2048)

        mm_data = {}
        if images:
            mm_data["image"] = images
        if videos:
            mm_data["video"] = videos

        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": {
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
        }
    else:
        # Text only
        return text


def load_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 12288,
    lora_path: str = None,
    lora_name: str = "default",
):
    """Load vLLM model."""
    global llm, processor, lora_request, config
    adapter_meta = resolve_lora_artifacts(model_path, lora_path)
    resolved_model_path = adapter_meta["model_path"] or model_path
    resolved_lora_path = adapter_meta["lora_path"]
    resolved_tokenizer_path = os.environ.get("VLLM_TOKENIZER_PATH", "").strip() or resolved_model_path
    resolved_lora_rank = adapter_meta["lora_rank"] if adapter_meta["lora_rank"] is not None else 64

    print(f"\n{'='*80}")
    print(f"Loading vLLM model...")
    print(f"{'='*80}")
    print(f"Model: {resolved_model_path}")
    print(f"Tokenizer: {resolved_tokenizer_path}")
    print(f"Tensor parallel: {tensor_parallel_size}")
    print(f"GPU memory: {gpu_memory_utilization}")
    if resolved_lora_path:
        print(f"LoRA: {resolved_lora_path}")
        print(f"LoRA rank: {resolved_lora_rank}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # ============================================================================
    # CRITICAL: Add special tokens for latent thinking BEFORE loading vLLM
    # ============================================================================
    prepare_inference_tokenizer(resolved_tokenizer_path)

    # Apply thinking mode patch only when explicitly enabled.
    apply_thinking_mode_patch()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        resolved_model_path,
        trust_remote_code=True
    )

    # Build kwargs
    llm_kwargs = {
        "model": resolved_model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "max_model_len": int(max_model_len),
        "limit_mm_per_prompt": {"image": 10},
        "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
        "disable_custom_all_reduce": True, # Key to the distributed inference with mode change
    }

    # Add LoRA config if path provided
    if resolved_lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = resolved_lora_rank
        llm_kwargs["max_loras"] = 1

    # Load model
    llm = LLM(**llm_kwargs)

    # Create LoRA request
    effective_lora_name = lora_name
    if resolved_lora_path and lora_name == "default":
        effective_lora_name = normalize_checkpoint_name(resolved_lora_path)

    if HAS_LORA_REQUEST and resolved_lora_path:
        lora_request = LoRARequest(
            lora_name=effective_lora_name,
            lora_int_id=1,
            lora_path=resolved_lora_path,
        )

    load_time = time.time() - start_time
    print(f"\n✓ Model loaded in {load_time:.2f}s")

    # Store config
    config = {
        "model_path": resolved_model_path,
        "requested_model_path": model_path,
        "lora_path": resolved_lora_path,
        "resolved_lora_rank": resolved_lora_rank,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "thinking_enabled": THINKING_MODE_ENABLED,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM Server with Thinking Mode")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--port", type=int, default=8016, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="Tensor parallel size. Use 0 to auto-infer from CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=12288, help="Max model length for vLLM")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--lora-name", type=str, default="default", help="LoRA adapter name")
    args = parser.parse_args()

    visible_gpus = parse_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    resolved_tp = (
        infer_tensor_parallel_size(os.environ.get("CUDA_VISIBLE_DEVICES"), fallback=1)
        if args.tensor_parallel_size <= 0
        else args.tensor_parallel_size
    )
    print(f"CUDA_VISIBLE_DEVICES: {','.join(visible_gpus) if visible_gpus else 'not set'}")
    print(f"Tensor parallel (resolved): {resolved_tp}")

    # Auto-find free port if default is taken
    def find_free_port(start_port):
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(('', port))
                    s.listen(1)
                    return port
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    continue
                raise
        return start_port

    # Check if port is available, auto-select if not
    original_port = args.port
    args.port = find_free_port(args.port)
    if args.port != original_port:
        print(f"Port {original_port} in use, auto-selected port: {args.port}")

    # Set LoRA checkpoint path (for VAE loading via thinking plugin)
    if args.lora_path:
        os.environ["VLLM_LORA_CHECKPOINT_PATH"] = args.lora_path
        print(f"Set VLLM_LORA_CHECKPOINT_PATH={args.lora_path}")

    # Check VAE file exists
    vae_file = None
    if args.lora_path:
        vae_file = os.path.join(args.lora_path, "vae.safetensors")
        if not os.path.exists(vae_file):
            vae_file = os.path.join(args.lora_path, "vae.pt")
        if not os.path.exists(vae_file):
            # Try in model path too
            vae_file = os.path.join(args.model_path, "vae.safetensors")
            if not os.path.exists(vae_file):
                vae_file = os.path.join(args.model_path, "vae.pt")
        if vae_file and os.path.exists(vae_file):
            print(f"Found VAE weights: {vae_file}")
        else:
            print(f"WARNING: No VAE weights found in {args.lora_path} or {args.model_path}")

    # Load model
    load_model(
        model_path=args.model_path,
        tensor_parallel_size=resolved_tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        lora_path=args.lora_path,
        lora_name=args.lora_name,
    )

    # Start server
    print(f"\n{'='*80}")
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"{'='*80}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
