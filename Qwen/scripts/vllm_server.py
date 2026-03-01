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
import os
import socket
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Set vLLM multiprocessing method BEFORE importing vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Enable thinking mode with VAE
os.environ.setdefault("VLLM_THINKING", "1")

# Import thinking mode plugin BEFORE vLLM to apply patches
from vllm_thinking.runner_patch import apply_thinking_mode_patch

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Try to import LoRARequest
try:
    from vllm.v1.engine import LoRARequest
    HAS_LORA_REQUEST = True
except ImportError:
    HAS_LORA_REQUEST = False
    LoRARequest = None

app = FastAPI(title="vLLM Thinking Server")

# Global state
llm = None
processor = None
lora_request = None
config = {}


class ChatMessage(BaseModel):
    role: str
    content: Any  # Can be string or list of dicts


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 8192
    top_p: float = 1.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": config.get("model_path"),
        "lora": config.get("lora_path"),
    }


@app.get("/stats")
async def stats():
    """Get server statistics."""
    return {
        "model_path": config.get("model_path"),
        "lora_path": config.get("lora_path"),
        "tensor_parallel_size": config.get("tensor_parallel_size"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization"),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completion endpoint."""
    global llm, processor, lora_request

    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert messages to vLLM format
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Prepare inputs
        vllm_input = prepare_inputs_for_vllm(messages, processor)

        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            presence_penalty=request.presence_penalty,
            repetition_penalty=request.repetition_penalty,
            stop_token_ids=[151643, 151645],
        )

        # Run inference
        outputs = llm.generate(
            [vllm_input],
            sampling_params=sampling_params,
            lora_request=lora_request
        )

        # Extract response (raw; do not strip thinking tokens).
        output = outputs[0]
        response_text = output.outputs[0].text

        # DEBUG: Print raw output to see if thinking tokens exist (only if VLLM_DEBUG=1)
        if os.environ.get("VLLM_DEBUG", "0") == "1":
            print(f"[DEBUG] Raw response_text: {repr(response_text[:500])}")

        return {
            "id": "chatcmpl-" + str(int(time.time())),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.get("model_path"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": output.prompt_token_ids,
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            }
        }

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
                        if isinstance(img, str) and img.startswith("file://"):
                            img = img[len("file://"):]
                        images.append(img)
                    # Preserve any per-image resolution constraints if present.
                    if min_pixels is None and "min_pixels" in item:
                        min_pixels = item.get("min_pixels")
                    if max_pixels is None and "max_pixels" in item:
                        max_pixels = item.get("max_pixels")
                elif item.get("type") == "video":
                    vid = item.get("video")
                    if vid:
                        if isinstance(vid, str) and vid.startswith("file://"):
                            vid = vid[len("file://"):]
                        videos.append(vid)

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
    lora_path: str = None,
    lora_name: str = "default",
):
    """Load vLLM model."""
    global llm, processor, lora_request, config

    print(f"\n{'='*80}")
    print(f"Loading vLLM model...")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Tensor parallel: {tensor_parallel_size}")
    print(f"GPU memory: {gpu_memory_utilization}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # ============================================================================
    # CRITICAL: Add special tokens for latent thinking BEFORE loading vLLM
    # ============================================================================
    from transformers import AutoTokenizer

    # Load tokenizer and add special tokens
    tokenizer_with_special_tokens = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    # Add <latent> and <think_sep> as special tokens
    special_tokens = ["<latent>", "<think_sep>"]
    added_count = 0
    for token in special_tokens:
        encoded = tokenizer_with_special_tokens.encode(token, add_special_tokens=False)
        if len(encoded) > 1:
            num_added = tokenizer_with_special_tokens.add_special_tokens(
                {"additional_special_tokens": [token]},
                replace_additional_special_tokens=False
            )
            added_count += num_added

    if added_count > 0:
        print(f"Added {added_count} special thinking tokens to tokenizer:")
        for token in special_tokens:
            token_id = tokenizer_with_special_tokens.convert_tokens_to_ids(token)
            print(f"  {token} -> ID {token_id}")
            # Set environment variables for vLLM thinking mode
            if token == "<latent>":
                os.environ["QWEN3VL_LATENT_TOKEN_ID"] = str(token_id)
            elif token == "<think_sep>":
                os.environ["QWEN3VL_THINKING_SEP_ID"] = str(token_id)

    # Apply thinking mode patch BEFORE loading vLLM (patches GPUModelRunner class)
    apply_thinking_mode_patch()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Build kwargs
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "max_model_len": 128000,
        "limit_mm_per_prompt": {"image": 10},
    }

    # Add LoRA config if path provided
    if lora_path:
        llm_kwargs["enable_lora"] = True
        # Note: vLLM LoRA config needs proper setup

    # Load model
    llm = LLM(**llm_kwargs)

    # Create LoRA request
    if HAS_LORA_REQUEST and lora_path:
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=1,
            lora_path=lora_path,
        )

    load_time = time.time() - start_time
    print(f"\n✓ Model loaded in {load_time:.2f}s")

    # Store config
    config = {
        "model_path": model_path,
        "lora_path": lora_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM Server with Thinking Mode")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--port", type=int, default=8016, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--lora-name", type=str, default="default", help="LoRA adapter name")

    args = parser.parse_args()

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
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
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
