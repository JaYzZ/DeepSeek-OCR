"""
Test OCR Server Latency for Visual Token Generation

Measures the throughput of the DeepSeek OCR server for text-to-vistok conversion.
"""

import time
import requests
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = "http://localhost:8010"

# Sample texts of different lengths
SAMPLE_TEXTS = {
    "short": "Hello world. This is a short test.",  # ~10 tokens
    "medium": """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn from data.
The key concepts include supervised learning, unsupervised learning, and reinforcement learning.

## Supervised Learning
In supervised learning, we train models on labeled data. Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

## Mathematical Foundation
The loss function is defined as: L(θ) = Σ(y - f(x; θ))²
""",  # ~100 tokens
    "long": """
# Comprehensive Guide to Deep Learning

## Chapter 1: Neural Network Fundamentals

Neural networks are computational models inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process information.

### 1.1 Perceptron

The perceptron is the simplest form of a neural network:

y = σ(Σ wᵢxᵢ + b)

Where:
- x is the input vector
- w is the weight vector
- b is the bias term
- σ is the activation function

### 1.2 Multi-Layer Perceptron

A multi-layer perceptron (MLP) consists of:
1. Input layer
2. One or more hidden layers
3. Output layer

The forward propagation is:
h₁ = σ(W₁x + b₁)
h₂ = σ(W₂h₁ + b₂)
y = softmax(W₃h₂ + b₃)

### 1.3 Backpropagation

The backpropagation algorithm computes gradients using the chain rule:

∂L/∂W = ∂L/∂y · ∂y/∂h · ∂h/∂W

## Chapter 2: Optimization

### 2.1 Gradient Descent

θ_{t+1} = θ_t - η∇L(θ_t)

### 2.2 Adam Optimizer

Adam combines momentum and RMSprop:
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - η·m_t/(√v_t + ε)

## Chapter 3: Convolutional Neural Networks

CNNs are designed for processing grid-like data such as images.

### 3.1 Convolution Operation

(f * g)(t) = ∫ f(τ)g(t-τ)dτ

In discrete form for 2D images:
S(i,j) = Σ_m Σ_n I(i+m, j+n)K(m,n)

### 3.2 Pooling Layers

Max pooling: y = max(x₁, x₂, ..., xₙ)
Average pooling: y = (1/n)Σxᵢ

## Chapter 4: Recurrent Neural Networks

RNNs process sequential data with hidden states.

h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b_h)
y_t = W_hy·h_t + b_y

### 4.1 LSTM

LSTM addresses vanishing gradients with gates:
- Forget gate: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
- Input gate: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
- Output gate: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)

## Chapter 5: Transformers

The transformer architecture revolutionized NLP.

### 5.1 Self-Attention

Attention(Q,K,V) = softmax(QK^T/√d_k)V

### 5.2 Multi-Head Attention

MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
""",  # ~500-600 tokens
}


def test_single_request(text: str, chunk_size: int = 1000) -> dict:
    """Test single request latency"""
    start = time.time()

    response = requests.post(
        f"{SERVER_URL}/text-to-vistok",
        json={
            "texts": [text],
            "chunk_size": chunk_size,
            "render_width": 640,
            "render_height": 640,
            "font_size": 18,
            "include_rendered_images": False,
            "skip_embeddings": False,
            "output_format": "binary",
        },
        timeout=60,
    )

    latency = time.time() - start

    if response.status_code == 200:
        shape_str = response.headers.get('X-Tensor-Shape')
        shape = json.loads(shape_str) if shape_str else None
        return {
            "success": True,
            "latency": latency,
            "shape": shape,
            "text_len": len(text),
        }
    else:
        return {
            "success": False,
            "latency": latency,
            "error": response.status_code,
        }


def test_batch_request(texts: list, chunk_size: int = 1000) -> dict:
    """Test batch request latency"""
    start = time.time()

    response = requests.post(
        f"{SERVER_URL}/text-to-vistok",
        json={
            "texts": texts,
            "chunk_size": chunk_size,
            "render_width": 640,
            "render_height": 640,
            "font_size": 18,
            "include_rendered_images": False,
            "skip_embeddings": False,
            "output_format": "binary",
        },
        timeout=120,
    )

    latency = time.time() - start

    if response.status_code == 200:
        return {
            "success": True,
            "latency": latency,
            "num_texts": len(texts),
            "per_text_latency": latency / len(texts),
        }
    else:
        return {
            "success": False,
            "latency": latency,
            "error": response.status_code,
        }


def test_concurrent_requests(text: str, num_requests: int = 10) -> dict:
    """Test concurrent request throughput"""
    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(test_single_request, text) for _ in range(num_requests)]
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.time() - start
    successful = [r for r in results if r.get("success")]

    return {
        "total_time": total_time,
        "num_requests": num_requests,
        "successful": len(successful),
        "throughput": len(successful) / total_time,
        "avg_latency": np.mean([r["latency"] for r in successful]) if successful else 0,
    }


def main():
    print("=" * 70)
    print("OCR Server Latency Test")
    print("=" * 70)

    # Test health
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"\n✓ Server is healthy: {SERVER_URL}")
    except Exception as e:
        print(f"\n✗ Server not available: {e}")
        return

    # Test 1: Single request latency for different text lengths
    print("\n" + "-" * 70)
    print("Test 1: Single Request Latency (by text length)")
    print("-" * 70)

    for name, text in SAMPLE_TEXTS.items():
        # Warm up
        test_single_request(text)

        # Measure
        latencies = []
        for _ in range(5):
            result = test_single_request(text)
            if result["success"]:
                latencies.append(result["latency"])

        if latencies:
            avg = np.mean(latencies)
            std = np.std(latencies)
            print(f"  {name:10s} ({len(text):5d} chars): {avg*1000:7.1f}ms ± {std*1000:.1f}ms  shape={result.get('shape')}")

    # Test 2: Batch request
    print("\n" + "-" * 70)
    print("Test 2: Batch Request Latency")
    print("-" * 70)

    medium_text = SAMPLE_TEXTS["medium"]
    for batch_size in [1, 2, 4, 8]:
        texts = [medium_text] * batch_size

        # Warm up
        test_batch_request(texts)

        # Measure
        latencies = []
        for _ in range(3):
            result = test_batch_request(texts)
            if result["success"]:
                latencies.append(result["latency"])

        if latencies:
            avg = np.mean(latencies)
            per_text = avg / batch_size
            print(f"  Batch size {batch_size}: {avg*1000:7.1f}ms total, {per_text*1000:7.1f}ms per text")

    # Test 3: Concurrent requests (throughput)
    print("\n" + "-" * 70)
    print("Test 3: Concurrent Request Throughput")
    print("-" * 70)

    for num_concurrent in [1, 2, 4, 8]:
        result = test_concurrent_requests(SAMPLE_TEXTS["medium"], num_concurrent)
        print(f"  {num_concurrent} concurrent: {result['throughput']:.2f} req/sec, avg latency {result['avg_latency']*1000:.1f}ms")

    # Test 4: Sustained throughput
    print("\n" + "-" * 70)
    print("Test 4: Sustained Throughput (30 seconds)")
    print("-" * 70)

    text = SAMPLE_TEXTS["medium"]
    start = time.time()
    count = 0
    total_latency = 0

    while time.time() - start < 30:
        result = test_single_request(text)
        if result["success"]:
            count += 1
            total_latency += result["latency"]

    elapsed = time.time() - start
    throughput = count / elapsed
    avg_latency = total_latency / count if count > 0 else 0

    print(f"  Processed {count} requests in {elapsed:.1f}s")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Avg latency: {avg_latency*1000:.1f}ms")

    # Summary and projections
    print("\n" + "=" * 70)
    print("SUMMARY & TRAINING TIME PROJECTIONS")
    print("=" * 70)

    samples_per_sec = throughput
    samples_per_hour = samples_per_sec * 3600
    samples_per_day = samples_per_hour * 24

    print(f"\nServer Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"                   {samples_per_hour:,.0f} samples/hour")
    print(f"                   {samples_per_day:,.0f} samples/day")

    print("\n" + "-" * 70)
    print("Training Time Estimates (WITHOUT cache, on-the-fly encoding)")
    print("-" * 70)

    datasets = {
        "OpenWebMath (15M samples)": 15_000_000,
        "FineWeb subset (10M samples)": 10_000_000,
        "FineWeb subset (100M samples)": 100_000_000,
        "Quick test (100K samples)": 100_000,
    }

    for name, num_samples in datasets.items():
        # Time to process all samples once (1 epoch)
        time_1_epoch_sec = num_samples / samples_per_sec
        time_1_epoch_hours = time_1_epoch_sec / 3600
        time_1_epoch_days = time_1_epoch_hours / 24

        # For training: assume each sample seen ~10 times (10 epochs effective)
        time_10_epochs_days = time_1_epoch_days * 10

        if time_1_epoch_days < 1:
            time_str = f"{time_1_epoch_hours:.1f} hours"
        else:
            time_str = f"{time_1_epoch_days:.1f} days"

        print(f"  {name}:")
        print(f"    1 epoch:  {time_str}")
        print(f"    10 epochs: {time_10_epochs_days:.1f} days")
        print()


if __name__ == "__main__":
    main()
