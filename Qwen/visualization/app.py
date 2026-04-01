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
import base64
import errno
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from tokenizers import AddedToken

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Qwen.scripts.vllm_utils import (
    apply_runtime_env_for_thinking,
    infer_tensor_parallel_size,
    normalize_checkpoint_name,
    parse_cuda_visible_devices,
)


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _has_data(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return value.size > 0
    return len(value) > 0


def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


# Set vLLM multiprocessing method BEFORE importing vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# Enable visualization data collection for the visualization server
os.environ['VLLM_STORE_VISUALIZATION_DATA'] = '1'

# Load runtime env before enabling plugins so YAML can control VLLM_THINKING.
apply_runtime_env_for_thinking(repo_root=_REPO_ROOT)
# Default to thinking mode enabled for this visualization app (can be disabled with VLLM_THINKING=0)
THINKING_MODE_ENABLED = _env_flag_enabled("VLLM_THINKING", default="1") or _env_flag_enabled("VLLM_FORCE_THINK")
if THINKING_MODE_ENABLED:
    existing_plugins = [p.strip() for p in os.environ.get("VLLM_PLUGINS", "").split(",") if p.strip()]
    if "vllm_thinking" not in existing_plugins:
        existing_plugins.append("vllm_thinking")
    os.environ["VLLM_PLUGINS"] = ",".join(existing_plugins)

    # Import thinking mode plugin BEFORE vLLM to apply patches.
    from vllm_thinking.runner_patch import apply_thinking_mode_patch
else:
    apply_thinking_mode_patch = None

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Try to import LoRARequest
try:
    from vllm.v1.engine import LoRARequest
    HAS_LORA_REQUEST = True
except ImportError:
    HAS_LORA_REQUEST = False
    LoRARequest = None

app = FastAPI(title="Qwen3-VL Thinking Mode Visualization")

# Mount static files directory
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Import visualization utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.visualization_utils import compute_tsne, aggregate_attention, create_attention_heatmap_data, encode_image_to_base64

# Import trace store utilities if thinking mode is enabled
if THINKING_MODE_ENABLED:
    from vllm_thinking.trace_store import get_latest_trace, get_request_trace

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


def run_llm_generation(
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
):
    global llm, processor, lora_request

    vllm_input = prepare_inputs_for_vllm(messages, processor)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        stop_token_ids=[151643, 151645],
        skip_special_tokens=False,
    )
    outputs = llm.generate(
        [vllm_input],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    output = outputs[0]
    response_text = output.outputs[0].text
    return output, response_text


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "checkpoint_path": config.get("lora_path") or config.get("model_path"),
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


# ============================================================================
# Visualization-specific endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main visualization interface."""
    static_dir = Path(__file__).parent / "static"
    index_path = static_dir / "index.html"
    with open(index_path, 'r') as f:
        return HTMLResponse(content=f.read())


@app.get("/api/example")
async def get_example():
    """Get default deepvision example."""
    # Load first example from deepvision data
    deepvision_path = _REPO_ROOT / "Qwen/data/deepvision_103k_sft.jsonl"
    if deepvision_path.exists():
        with open(deepvision_path, 'r') as f:
            first_line = f.readline()
            example = json.loads(first_line)

        # Extract image and question
        image_path = example.get("images", [None])[0]
        messages = example.get("messages", [])
        question = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Remove <image> tag if present
                question = content.replace("<image>\n", "")
                break

        # Encode image to base64
        image_base64 = None
        if image_path and Path(image_path).exists():
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        return {
            "image_base64": image_base64,
            "question": question,
            "answer": messages[-1].get("content", "") if messages else "",
        }
    else:
        raise HTTPException(status_code=404, detail="Deepvision data file not found")


@app.post("/api/infer")
async def infer_vis(request: Dict[str, Any]):
    """Run inference with thinking mode and return visualization data."""
    global llm, processor, lora_request

    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare messages from request
        text = request.get("text", "")
        image_base64 = request.get("image_base64")
        max_tokens = request.get("max_tokens", 8192)
        temperature = request.get("temperature", 0.7)

        messages = []
        if image_base64:
            # For vLLM, use the image directly as bytes
            import io
            from PIL import Image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": text,
            })

        # Run inference
        output, response_text = run_llm_generation(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Collect trace data for visualization
        trace_data = {
            "tokens": [],
            "hidden_states": None,
            "attention_weights": None,
            "continuous_mask": [],
            "tsne_coordinates": None,
        }

        if THINKING_MODE_ENABLED:
            try:
                request_id = getattr(output, "request_id", None)
                trace = get_request_trace(request_id) if request_id is not None else None
                if trace is None:
                    trace = get_latest_trace()

                output_token_ids = list(output.outputs[0].token_ids)
                token_ids = output_token_ids
                if trace and _has_data(trace.get("all_token_ids", [])):
                    trace_token_ids = list(trace.get("all_token_ids", []))
                    # Only trust trace token ids when they are at least as complete
                    # as the RequestOutput sequence for this request.
                    if len(trace_token_ids) >= len(output_token_ids):
                        token_ids = trace_token_ids

                for i, token_id in enumerate(token_ids):
                    token_text = _decode_token_text(processor.tokenizer, token_id)
                    trace_data["tokens"].append({
                        'id': int(token_id),
                        'text': token_text,
                        'position': i,
                    })

                if trace:
                    # Extract hidden states, token embeddings, and latent embeddings
                    token_hidden_states = trace.get('all_hidden_states', [])
                    token_embeddings = trace.get('all_token_embeddings', [])
                    latent_embeddings = trace.get('continuous_latent_embeddings', [])
                    continuous_mask = trace.get('continuous_token_mask', [])
                    attention_weights = trace.get('attention_weights', [])

                    # Debug logging
                    print(f"[DEBUG] token_hidden_states: {len(token_hidden_states) if _has_data(token_hidden_states) else 0} items")
                    print(f"[DEBUG] token_embeddings: {len(token_embeddings) if _has_data(token_embeddings) else 0} items")
                    print(f"[DEBUG] latent_embeddings: {len(latent_embeddings) if _has_data(latent_embeddings) else 0} items")

                    trace_data["continuous_mask"] = np.asarray(continuous_mask, dtype=np.bool_).tolist()

                    # Compute t-SNE if we have any feature type available
                    if _has_data(token_hidden_states) or _has_data(token_embeddings) or _has_data(latent_embeddings):
                        # Prepare features for t-SNE: combine all three types
                        # For each position, we'll have up to 3 feature vectors
                        all_features = []
                        feature_types = []  # 'token_emb', 'hidden_state', 'vae_sample'
                        position_indices = []

                        for i in range(len(trace_data["tokens"])):
                            is_continuous = i < len(continuous_mask) and continuous_mask[i]

                            # 1. Token embedding (if available)
                            if _has_data(token_embeddings) and i < len(token_embeddings):
                                te = np.asarray(token_embeddings[i])
                                if te.ndim > 1:
                                    te = te.reshape(-1)
                                if te.size > 0:
                                    all_features.append(te)
                                    feature_types.append('token_emb')
                                    position_indices.append(i)

                            # 2. Hidden state (if available)
                            if _has_data(token_hidden_states) and i < len(token_hidden_states):
                                hs = np.asarray(token_hidden_states[i])
                                if hs.ndim > 1:
                                    hs = hs.reshape(-1)
                                if hs.size > 0:
                                    all_features.append(hs)
                                    feature_types.append('hidden_state')
                                    position_indices.append(i)

                            # 3. VAE sample (only for continuous tokens)
                            if is_continuous and _has_data(latent_embeddings):
                                # Find the corresponding latent embedding
                                latent_idx = sum(continuous_mask[:i])  # count continuous tokens before this position
                                if latent_idx < len(latent_embeddings):
                                    le = np.asarray(latent_embeddings[latent_idx])
                                    if le.ndim > 1:
                                        le = le.reshape(-1)
                                    if le.size > 0:
                                        all_features.append(le)
                                        feature_types.append('vae_sample')
                                        position_indices.append(i)

                        if all_features:
                            # Stack all features and compute t-SNE
                            feature_matrix = np.vstack(all_features)
                            tsne_coords = compute_tsne(feature_matrix)
                            trace_data["tsne_coordinates"] = tsne_coords.tolist()
                            trace_data["tsne_feature_types"] = feature_types
                            trace_data["tsne_position_indices"] = position_indices
                            print(f"[DEBUG] Computed t-SNE with {len(feature_types)} features: {feature_types[:10]}...")  # Show first 10
                        else:
                            print(f"[DEBUG] No features collected for t-SNE")

                    if _has_data(attention_weights):
                        trace_data["attention_weights"] = np.asarray(attention_weights).tolist()

            except Exception as e:
                print(f"Warning: Failed to collect trace data: {e}")

        return {
            "answer": response_text,
            "tokens": trace_data["tokens"],
            "hidden_states": trace_data["hidden_states"],
            "attention_weights": trace_data["attention_weights"],
            "continuous_mask": trace_data["continuous_mask"],
            "tsne_coordinates": trace_data["tsne_coordinates"],
            "tsne_feature_types": trace_data.get("tsne_feature_types", []),
            "tsne_position_indices": trace_data.get("tsne_position_indices", []),
            "token_metadata": {
                "total_tokens": len(trace_data["tokens"]),
                "continuous_tokens": sum(trace_data["continuous_mask"]),
            },
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")



@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completion endpoint."""
    global llm, processor, lora_request

    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert messages to vLLM format
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        output, response_text = run_llm_generation(
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            presence_penalty=request.presence_penalty,
            repetition_penalty=request.repetition_penalty,
        )

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
    if _env_flag_enabled("VLLM_FORCE_THINK"):
        text = text + "<think>"

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


def resolve_model_path(model_path: str | None, lora_path: str | None) -> str:
    """Mirror backfill behavior: prefer adapter-declared base model for LoRA checkpoints."""
    if not lora_path:
        if not model_path:
            raise ValueError("Either --model-path or --lora-path must be provided")
        return model_path

    adapter_config_path = Path(lora_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        if not model_path:
            raise ValueError(f"No adapter_config.json found in {lora_path}, and no --model-path provided")
        return model_path

    try:
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        resolved = adapter_config.get("base_model_name_or_path") or model_path
        print(f"Detected LoRA adapter. Using base model: {resolved}")
        return resolved
    except Exception as exc:
        print(f"Warning: failed to read {adapter_config_path}: {exc}")
        if not model_path:
            raise ValueError(f"Failed to read adapter_config.json from {lora_path}, and no --model-path provided")
        return model_path


def load_model(
    model_path: str | None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    lora_path: str = None,
    lora_name: str = "default",
):
    """Load vLLM model."""
    global llm, processor, lora_request, config
    resolved_model_path = resolve_model_path(model_path, lora_path)

    print(f"\n{'='*80}")
    print(f"Loading vLLM model...")
    print(f"{'='*80}")
    print(f"Model: {resolved_model_path}")
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
        resolved_model_path, trust_remote_code=True
    )

    # Ensure <latent>/<think_sep> are single tokens for inference.
    # Use regular added tokens (not "special") so they are generated/displayed
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
            print(f"Demoted special token to regular token at runtime: {token} (ID {token_id})")

    if added_count > 0:
        print(f"Added {added_count} regular thinking tokens to tokenizer")

    # Always export unified token IDs used by both training and vLLM plugin.
    for token in special_tokens:
        token_id = tokenizer_with_special_tokens.convert_tokens_to_ids(token)
        print(f"  {token} -> ID {token_id}")
        if token == "<latent>":
            os.environ["QWEN3VL_LATENT_TOKEN_ID"] = str(token_id)
        elif token == "<think_sep>":
            os.environ["QWEN3VL_THINKING_SEP_ID"] = str(token_id)

    # Apply thinking mode patch only when explicitly enabled.
    if THINKING_MODE_ENABLED and apply_thinking_mode_patch is not None:
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
        "max_model_len": 128000,
        "limit_mm_per_prompt": {"image": 10},
        "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
        "disable_custom_all_reduce": True, # Key to the distributed inference with mode change
    }

    # Add LoRA config if path provided
    if lora_path:
        llm_kwargs["enable_lora"] = True
        # Note: vLLM LoRA config needs proper setup

    # Load model
    llm = LLM(**llm_kwargs)

    # Create LoRA request
    effective_lora_name = lora_name
    if lora_path and lora_name == "default":
        effective_lora_name = normalize_checkpoint_name(lora_path)

    if HAS_LORA_REQUEST and lora_path:
        lora_request = LoRARequest(
            lora_name=effective_lora_name,
            lora_int_id=1,
            lora_path=lora_path,
        )

    load_time = time.time() - start_time
    print(f"\n✓ Model loaded in {load_time:.2f}s")

    # Store config
    config = {
        "model_path": resolved_model_path,
        "requested_model_path": model_path,
        "lora_path": lora_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "thinking_enabled": THINKING_MODE_ENABLED,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Thinking Mode Visualization Server")
    parser.add_argument("--model-path", type=str, default="Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking", help="Path to model (auto-detected from --lora-path if not provided)")
    parser.add_argument("--port", type=int, default=8501, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="Tensor parallel size. Use 0 to auto-infer from CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
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
