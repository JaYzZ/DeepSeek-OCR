#!/usr/bin/env python3
"""
Example Inference Script for OCRFlow

This script demonstrates how to use a trained OCRFlow model for generating
text via latent flow in DeepSeek OCR's frozen image token space.

Pipeline:
  Text Prompt → DiT (Flow Model) → Visual Tokens → vLLM Decode API → Text Output
"""

import torch
import argparse
from pathlib import Path
import sys
import base64
import requests
import json

# Add OCRFlow to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from OCRFlow.models import create_mmdit_ocrflow
from OCRFlow.models.text_encoders import load_text_encoders, encode_text_dual
from OCRFlow.utils.sampling import euler_sampling
from OCRFlow.utils.helpers import set_seed, get_device


def decode_tokens_to_text(tokens, vllm_server_url, temperature=0.0, max_tokens=2048, verbose=True):
    """
    Decode visual tokens to text using vLLM server

    Args:
        tokens: Tensor [B, N, D] of visual tokens
        vllm_server_url: URL of vLLM server (e.g., http://localhost:8009)
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        verbose: Print progress

    Returns:
        List of decoded text strings (one per batch item)
    """
    endpoint = f"{vllm_server_url}/visual-tokens-to-text"

    # Convert tokens to numpy and prepare chunks
    tokens_np = tokens.cpu().float().numpy()  # [B, N, D]
    batch_size = tokens_np.shape[0]

    chunks = []
    for idx in range(batch_size):
        token_chunk = tokens_np[idx]  # [N, D]

        # Encode to base64
        token_bytes = token_chunk.tobytes()
        token_b64 = base64.b64encode(token_bytes).decode('utf-8')

        chunks.append({
            "chunk_index": idx,
            "visual_tokens_base64": token_b64,
            "embedding_shape": list(token_chunk.shape)
        })

    # Prepare request
    payload = {
        "chunks": chunks,
        "prompt_prefix": "",
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if verbose:
        print(f"\nDecoding {batch_size} batch items via vLLM...")
        print(f"  Server: {vllm_server_url}")
        print(f"  Token shape per item: {list(token_chunk.shape)}")

    try:
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()

        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            print(f"Error from vLLM server: {error_msg}")
            return None

        if verbose:
            print(f"✓ Decoded {result['total_chunks']} chunks")
            print(f"✓ Combined text length: {len(result['combined_text'])} characters")

        # Return list of decoded texts
        decoded_texts = [chunk['text'] for chunk in result['chunks']]
        return decoded_texts

    except requests.exceptions.ConnectionError:
        print(f"\n✗ Could not connect to vLLM server at {vllm_server_url}")
        print("  Make sure the server is running (see server/README.md)")
        return None
    except requests.exceptions.Timeout:
        print(f"\n✗ Request timed out after 300s")
        return None
    except Exception as e:
        print(f"\n✗ Error during decoding: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="OCRFlow Inference - Text Generation via Latent Flow")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "base", "large", "xl"])
    parser.add_argument("--prompts", type=str, nargs="+", default=["<image>\n<|grounding|>Convert the document to markdown."])
    parser.add_argument("--num-steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default="generated_tokens.pt", help="Output file path for tokens")
    parser.add_argument("--vllm-server", type=str, default="http://localhost:8009", help="vLLM server URL for decoding")
    parser.add_argument("--decode", action="store_true", default=True, help="Decode tokens to text via vLLM (default: True)")
    parser.add_argument("--no-decode", dest="decode", action="store_false", help="Skip decoding, only save tokens")
    parser.add_argument("--decode-temperature", type=float, default=0.0, help="Temperature for decoding")
    parser.add_argument("--decode-max-tokens", type=int, default=2048, help="Max tokens for decoding")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device and dtype
    device = get_device(args.device)
    dtype = torch.bfloat16

    print(f"Using device: {device}")
    print(f"Model size: {args.model_size}")
    print(f"Prompts: {args.prompts}")

    # Load MMDiT model
    print("\nLoading MMDiT model...")
    model = create_mmdit_ocrflow(args.model_size)
    model = model.to(device).to(dtype).eval()

    # Load checkpoint
    if Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Checkpoint step: {checkpoint.get('global_step', 'N/A')}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Using randomly initialized model (for testing only)")

    # Load text encoders
    print("\nLoading text encoders...")
    text_encoders = load_text_encoders(
        t5_model_path="google/t5-v1_1-base",
        clip_model_path="openai/clip-vit-large-patch14",
        device=device,
        dtype=dtype,
        freeze=True
    )
    print(f"  T5 dim: {text_encoders['t5_dim']}")
    print(f"  CLIP dim: {text_encoders['clip_dim']}")

    # Encode text
    print("\nEncoding text prompts...")
    with torch.no_grad():
        t5_embeds, clip_embeds, t5_mask = encode_text_dual(
            args.prompts,
            t5_tokenizer=text_encoders['t5_tokenizer'],
            t5_model=text_encoders['t5_model'],
            clip_tokenizer=text_encoders['clip_tokenizer'],
            clip_model=text_encoders['clip_model'],
            device=device,
            dtype=dtype
        )

    print(f"  T5 embeddings shape: {t5_embeds.shape}")
    print(f"  CLIP embeddings shape: {clip_embeds.shape}")

    # Generate image tokens
    print(f"\nGenerating image tokens...")
    print(f"  Sampling steps: {args.num_steps}")
    print(f"  CFG scale: {args.cfg_scale}")

    with torch.no_grad():
        generated_tokens = euler_sampling(
            model=model,
            text_seq_embeds=t5_embeds,
            text_pooled_embeds=clip_embeds,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            num_tokens=model.num_image_tokens,
            token_dim=model.token_dim,
            device=device,
            dtype=dtype,
            verbose=True
        )

    print(f"\nGenerated tokens shape: {generated_tokens.shape}")

    # Save tokens to disk
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'tokens': generated_tokens.cpu(),
        'prompts': args.prompts,
        'num_steps': args.num_steps,
        'cfg_scale': args.cfg_scale,
        'seed': args.seed,
    }, output_path)

    print(f"\nSaved generated tokens to: {output_path}")

    # Decode tokens to text via vLLM
    if args.decode:
        print("\n" + "="*80)
        print("DECODING TOKENS TO TEXT")
        print("="*80)

        decoded_texts = decode_tokens_to_text(
            tokens=generated_tokens,
            vllm_server_url=args.vllm_server,
            temperature=args.decode_temperature,
            max_tokens=args.decode_max_tokens,
            verbose=True
        )

        if decoded_texts:
            print("\n" + "="*80)
            print("GENERATED TEXT OUTPUT")
            print("="*80)

            for i, (prompt, text) in enumerate(zip(args.prompts, decoded_texts)):
                print(f"\n[Prompt {i+1}]: {prompt}")
                print(f"[Generated Text {i+1}]:\n{'-'*60}")
                print(text)
                print("-"*60)

            # Optionally save text output
            text_output_path = output_path.with_suffix('.txt')
            with open(text_output_path, 'w', encoding='utf-8') as f:
                for i, (prompt, text) in enumerate(zip(args.prompts, decoded_texts)):
                    f.write(f"=== Prompt {i+1} ===\n")
                    f.write(f"{prompt}\n\n")
                    f.write(f"=== Generated Text {i+1} ===\n")
                    f.write(f"{text}\n\n")
                    f.write("="*60 + "\n\n")

            print(f"\n✓ Saved decoded text to: {text_output_path}")
        else:
            print("\n✗ Decoding failed. Tokens saved but no text output.")
            print("  Check that vLLM server is running and accessible.")

    print("\n" + "="*80)
    print("Inference complete!")
    print("="*80)


if __name__ == "__main__":
    main()
