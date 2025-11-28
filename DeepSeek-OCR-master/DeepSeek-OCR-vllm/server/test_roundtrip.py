#!/usr/bin/env python3
"""
Round-trip accuracy test for text-to-vistok and vistok-to-text endpoints

Tests the full pipeline:
1. Input text → text-to-vistok (render + encode) → binary visual tokens
2. Binary visual tokens → vistok-to-text (decode) → output text
3. Compare input vs output for accuracy
"""

import requests
import json
import base64
import sys
import io
from pathlib import Path
from PIL import Image
import numpy as np
from difflib import SequenceMatcher

SERVER_URL = "http://localhost:8009"

# Test texts with varying complexity
TEST_CASES = [
    {
        "name": "Simple Text",
        "text": "This is a simple test with basic characters and numbers 123.",
        "expected_tokens": ~100  # Approximate
    },
    {
        "name": "Medium Complexity",
        "text": """DeepSeek-OCR is a vision-language model for document understanding.
It can extract text from images and convert text to visual tokens.
Key features:
- High accuracy OCR
- Bidirectional text↔visual conversion
- Efficient binary format (33% smaller)""",
        "expected_tokens": ~500
    },
    {
        "name": "Dense Text (from test image)",
        "text": """# Vision-Language Models for Document Understanding

## Introduction
Vision-language models combine computer vision and natural language processing to understand documents. These models can process images containing text, tables, and figures to extract structured information.

## Architecture
The typical architecture consists of:
1. Vision Encoder (CLIP + SAM)
2. Projector (maps visual features to language space)
3. Language Model (generates text from visual tokens)

## Applications
- Document OCR and digitization
- Form understanding and extraction
- Scientific paper analysis
- Receipt and invoice processing""",
        "expected_tokens": ~1000
    }
]


def text_to_visual_tokens(text: str, server_url: str = SERVER_URL):
    """
    Convert text to visual tokens using /text-to-vistok endpoint

    Returns: (binary_data, metadata_dict)
    """
    url = f"{server_url}/text-to-vistok"

    payload = {
        "texts": [text],
        "output_format": "binary",
        "include_rendered_images": True,  # Get rendered image for verification
    }

    print(f"  → Sending text to {url} ({len(text)} chars)...")
    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"text-to-vistok failed: {response.status_code} - {response.text}")

    # Extract metadata from headers
    metadata = {
        "shape": json.loads(response.headers.get("X-Tensor-Shape", "[0,0,0]")),
        "dtype": response.headers.get("X-Tensor-Dtype", "float32"),
        "total_chunks": int(response.headers.get("X-Total-Chunks", "0")),
        "total_visual_tokens": int(response.headers.get("X-Total-Visual-Tokens", "0")),
        "render_time": float(response.headers.get("X-Render-Time", "0")),
        "encode_time": float(response.headers.get("X-Encode-Time", "0")),
        "total_time": float(response.headers.get("X-Total-Time", "0")),
    }

    binary_data = response.content

    print(f"  ✓ Received visual tokens: {len(binary_data):,} bytes")
    print(f"    Shape: {metadata['shape']}, Chunks: {metadata['total_chunks']}")
    print(f"    Timing: render={metadata['render_time']:.3f}s, encode={metadata['encode_time']:.3f}s, total={metadata['total_time']:.3f}s")

    return binary_data, metadata


def visual_tokens_to_text(binary_data: bytes, shape: list, dtype: str = "float32", server_url: str = SERVER_URL):
    """
    Convert visual tokens to text using /vistok-to-text endpoint

    Returns: decoded_text
    """
    url = f"{server_url}/vistok-to-text"

    # Prepare multipart form data
    files = {
        'visual_tokens': ('tokens.bin', io.BytesIO(binary_data), 'application/octet-stream')
    }

    data = {
        'shape': json.dumps(shape),
        'dtype': dtype,
        'prompt_prefix': '',  # Empty prefix for clean extraction
        'temperature': 0.0,
        'max_tokens': 2048
    }

    print(f"  → Sending visual tokens to {url} (shape={shape})...")
    response = requests.post(url, files=files, data=data, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"vistok-to-text failed: {response.status_code} - {response.text}")

    result = response.json()

    if not result.get("success"):
        raise RuntimeError(f"Decoding failed: {result.get('error')}")

    decoded_text = result.get("text", "")
    num_chunks = result.get("num_chunks", 0)

    print(f"  ✓ Decoded text: {len(decoded_text)} chars from {num_chunks} chunks")

    return decoded_text


def compare_texts(original: str, decoded: str):
    """
    Compare original and decoded texts, return metrics
    """
    # Normalize whitespace
    orig_normalized = " ".join(original.split())
    decoded_normalized = " ".join(decoded.split())

    # Calculate similarity
    similarity = SequenceMatcher(None, orig_normalized, decoded_normalized).ratio()

    # Character-level accuracy
    char_accuracy = similarity * 100

    # Word-level accuracy
    orig_words = orig_normalized.split()
    decoded_words = decoded_normalized.split()

    word_similarity = SequenceMatcher(None, orig_words, decoded_words).ratio()
    word_accuracy = word_similarity * 100

    # Length comparison
    orig_len = len(orig_normalized)
    decoded_len = len(decoded_normalized)
    length_ratio = decoded_len / orig_len if orig_len > 0 else 0

    return {
        "char_accuracy": char_accuracy,
        "word_accuracy": word_accuracy,
        "original_chars": orig_len,
        "decoded_chars": decoded_len,
        "length_ratio": length_ratio,
        "exact_match": orig_normalized == decoded_normalized
    }


def run_roundtrip_test(test_case: dict, server_url: str = SERVER_URL):
    """Run a single round-trip test"""
    print(f"\n{'='*80}")
    print(f"Test Case: {test_case['name']}")
    print(f"{'='*80}")
    print(f"Input text ({len(test_case['text'])} chars):")
    print("-" * 80)
    print(test_case['text'][:200] + ("..." if len(test_case['text']) > 200 else ""))
    print("-" * 80)

    try:
        # Step 1: Text → Visual Tokens
        print("\n[1/2] Text → Visual Tokens")
        binary_data, metadata = text_to_visual_tokens(test_case['text'], server_url)

        # Step 2: Visual Tokens → Text
        print("\n[2/2] Visual Tokens → Text")
        decoded_text = visual_tokens_to_text(binary_data, metadata['shape'], metadata['dtype'], server_url)

        # Compare results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        print(f"\nDecoded text ({len(decoded_text)} chars):")
        print("-" * 80)
        print(decoded_text[:200] + ("..." if len(decoded_text) > 200 else ""))
        print("-" * 80)

        metrics = compare_texts(test_case['text'], decoded_text)

        print(f"\n📊 Accuracy Metrics:")
        print(f"  Character-level accuracy: {metrics['char_accuracy']:.2f}%")
        print(f"  Word-level accuracy:      {metrics['word_accuracy']:.2f}%")
        print(f"  Original length:          {metrics['original_chars']} chars")
        print(f"  Decoded length:           {metrics['decoded_chars']} chars")
        print(f"  Length ratio:             {metrics['length_ratio']:.2f}x")
        print(f"  Exact match:              {'✓ YES' if metrics['exact_match'] else '✗ NO'}")

        return {
            "name": test_case['name'],
            "success": True,
            "metrics": metrics,
            "original_text": test_case['text'],
            "decoded_text": decoded_text
        }

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "name": test_case['name'],
            "success": False,
            "error": str(e)
        }


def main():
    """Run all round-trip tests"""
    print("\n" + "="*80)
    print("DeepSeek-OCR Round-Trip Accuracy Test")
    print("="*80)
    print(f"Server: {SERVER_URL}")
    print(f"Tests: {len(TEST_CASES)}")

    # Check server health
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        health = response.json()
        print(f"\n✓ Server Status: {health['status']}")
        print(f"  Model loaded: {health['model_loaded']}")
        print(f"  GPU available: {health['gpu_available']}")
    except Exception as e:
        print(f"\n❌ Server health check failed: {e}")
        print(f"   Make sure the server is running on {SERVER_URL}")
        sys.exit(1)

    # Run tests
    results = []
    for test_case in TEST_CASES:
        result = run_roundtrip_test(test_case, SERVER_URL)
        results.append(result)

    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\nTotal tests: {len(results)}")
    print(f"Passed: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\n📊 Overall Accuracy (successful tests):")
        avg_char_acc = sum(r['metrics']['char_accuracy'] for r in successful) / len(successful)
        avg_word_acc = sum(r['metrics']['word_accuracy'] for r in successful) / len(successful)
        exact_matches = sum(1 for r in successful if r['metrics']['exact_match'])

        print(f"  Average character-level accuracy: {avg_char_acc:.2f}%")
        print(f"  Average word-level accuracy:      {avg_word_acc:.2f}%")
        print(f"  Exact matches:                    {exact_matches}/{len(successful)}")

    if failed:
        print(f"\n❌ Failed tests:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")

    # Exit code
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
