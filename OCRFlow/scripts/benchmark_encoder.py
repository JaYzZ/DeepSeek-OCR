"""
Benchmark: Direct Encoder vs vLLM Server

Compare visual token generation throughput between:
1. vLLM HTTP server (current approach)
2. Direct encoder integration (proposed)

The direct approach should be significantly faster due to:
- No HTTP overhead
- Native batching
- No serialization/deserialization
"""

import torch
import time
import numpy as np
from pathlib import Path
import sys

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"))

from PIL import Image, ImageDraw, ImageFont


def render_text_to_image(text: str, width: int = 640, height: int = 640, font_size: int = 18) -> Image.Image:
    """Render text to image (same as server does)"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Simple text wrapping
    margin = 20
    max_width = width - 2 * margin
    lines = []
    current_line = ""

    for word in text.split():
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # Draw text
    y = margin
    line_height = font_size + 4
    for line in lines:
        if y + line_height > height - margin:
            break
        draw.text((margin, y), line, fill='black', font=font)
        y += line_height

    return img


# Sample texts for benchmarking
SAMPLE_TEXTS = [
    """
# Machine Learning Fundamentals

Machine learning is a branch of artificial intelligence that focuses on building systems
that learn from data. The key types include:

1. Supervised Learning: Training with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through interaction

## Mathematical Foundation

The loss function L(θ) measures how well our model fits the data:
L(θ) = (1/n) Σᵢ (yᵢ - f(xᵢ; θ))²

Gradient descent updates: θ ← θ - η∇L(θ)
""",
] * 8  # Repeat for batching


def benchmark_direct_encoder(model_path: str, num_iterations: int = 20, batch_sizes: list = [1, 2, 4, 8]):
    """Benchmark direct encoder integration"""
    print("\n" + "=" * 70)
    print("Benchmarking DIRECT ENCODER Integration")
    print("=" * 70)

    from server.standalone_vision_encoder import StandaloneVisionEncoder

    # Load encoder
    print("\nLoading encoder model...")
    start = time.time()
    encoder = StandaloneVisionEncoder(
        model_path=model_path,
        device="cuda",
        dtype=torch.bfloat16
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Pre-render images
    print("\nPre-rendering images...")
    images = [render_text_to_image(text) for text in SAMPLE_TEXTS]

    results = {}

    for batch_size in batch_sizes:
        batch_images = images[:batch_size]

        # Warmup
        print(f"\nBatch size {batch_size}: Warming up...")
        for _ in range(3):
            _ = encoder.encode_images(batch_images, return_global=False, return_local=True)

        torch.cuda.synchronize()

        # Benchmark
        print(f"Batch size {batch_size}: Running {num_iterations} iterations...")
        latencies = []

        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()

            embeddings = encoder.encode_images(batch_images, return_global=False, return_local=True)

            torch.cuda.synchronize()
            latency = time.time() - start
            latencies.append(latency)

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        per_sample = avg_latency / batch_size
        throughput = batch_size / avg_latency

        results[batch_size] = {
            "avg_latency": avg_latency,
            "std_latency": std_latency,
            "per_sample": per_sample,
            "throughput": throughput,
            "shape": embeddings[0].shape if embeddings else None,
        }

        print(f"  Batch {batch_size}: {avg_latency*1000:.1f}ms ± {std_latency*1000:.1f}ms")
        print(f"           Per sample: {per_sample*1000:.1f}ms")
        print(f"           Throughput: {throughput:.2f} samples/sec")

    return results


def benchmark_vllm_server(server_url: str = "http://localhost:8010", num_iterations: int = 20, batch_sizes: list = [1, 2, 4, 8]):
    """Benchmark vLLM server approach"""
    import requests
    import json

    print("\n" + "=" * 70)
    print("Benchmarking vLLM HTTP SERVER")
    print("=" * 70)

    # Check server health
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if not response.ok:
            print(f"Server not healthy: {response.status_code}")
            return None
        print(f"\n✓ Server connected: {server_url}")
    except Exception as e:
        print(f"\n✗ Server not available: {e}")
        return None

    results = {}

    for batch_size in batch_sizes:
        texts = SAMPLE_TEXTS[:batch_size]

        # Warmup
        print(f"\nBatch size {batch_size}: Warming up...")
        for _ in range(2):
            response = requests.post(
                f"{server_url}/text-to-vistok",
                json={
                    "texts": texts,
                    "chunk_size": 1000,
                    "render_width": 640,
                    "render_height": 640,
                    "font_size": 18,
                    "include_rendered_images": False,
                    "output_format": "binary",
                },
                timeout=60,
            )

        # Benchmark
        print(f"Batch size {batch_size}: Running {num_iterations} iterations...")
        latencies = []

        for i in range(num_iterations):
            start = time.time()

            response = requests.post(
                f"{server_url}/text-to-vistok",
                json={
                    "texts": texts,
                    "chunk_size": 1000,
                    "render_width": 640,
                    "render_height": 640,
                    "font_size": 18,
                    "include_rendered_images": False,
                    "output_format": "binary",
                },
                timeout=60,
            )

            latency = time.time() - start
            latencies.append(latency)

            if response.status_code != 200:
                print(f"  Warning: Request failed with status {response.status_code}")

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        per_sample = avg_latency / batch_size
        throughput = batch_size / avg_latency

        results[batch_size] = {
            "avg_latency": avg_latency,
            "std_latency": std_latency,
            "per_sample": per_sample,
            "throughput": throughput,
        }

        print(f"  Batch {batch_size}: {avg_latency*1000:.1f}ms ± {std_latency*1000:.1f}ms")
        print(f"           Per sample: {per_sample*1000:.1f}ms")
        print(f"           Throughput: {throughput:.2f} samples/sec")

    return results


def print_comparison(direct_results: dict, server_results: dict):
    """Print comparison summary"""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print("\n{:^10} | {:^20} | {:^20} | {:^10}".format(
        "Batch", "Direct Encoder", "vLLM Server", "Speedup"
    ))
    print("-" * 70)

    for batch_size in direct_results.keys():
        direct = direct_results[batch_size]
        server = server_results.get(batch_size, {}) if server_results else {}

        direct_tp = direct["throughput"]
        server_tp = server.get("throughput", 0)
        speedup = direct_tp / server_tp if server_tp > 0 else float('inf')

        print("{:^10} | {:>8.2f} samples/s    | {:>8.2f} samples/s    | {:>6.1f}x".format(
            batch_size, direct_tp, server_tp, speedup
        ))

    # Best throughput
    best_direct = max(direct_results.values(), key=lambda x: x["throughput"])
    best_direct_batch = [k for k, v in direct_results.items() if v == best_direct][0]

    print("\n" + "-" * 70)
    print(f"Best Direct Encoder: {best_direct['throughput']:.2f} samples/sec (batch {best_direct_batch})")

    if server_results:
        best_server = max(server_results.values(), key=lambda x: x["throughput"])
        best_server_batch = [k for k, v in server_results.items() if v == best_server][0]
        print(f"Best vLLM Server:    {best_server['throughput']:.2f} samples/sec (batch {best_server_batch})")

        overall_speedup = best_direct["throughput"] / best_server["throughput"]
        print(f"\nOverall Speedup: {overall_speedup:.1f}x faster with direct encoder")

    # Training time projections
    print("\n" + "=" * 70)
    print("TRAINING TIME PROJECTIONS (Direct Encoder)")
    print("=" * 70)

    samples_per_sec = best_direct["throughput"]
    samples_per_hour = samples_per_sec * 3600
    samples_per_day = samples_per_hour * 24

    print(f"\nDirect Encoder Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"                           {samples_per_hour:,.0f} samples/hour")
    print(f"                           {samples_per_day:,.0f} samples/day")

    datasets = {
        "Quick test (100K)": 100_000,
        "OpenWebMath (15M)": 15_000_000,
        "FineWeb (10M)": 10_000_000,
    }

    print("\n{:<25} | {:>15} | {:>15}".format("Dataset", "1 Epoch", "10 Epochs"))
    print("-" * 60)

    for name, num_samples in datasets.items():
        time_1_epoch = num_samples / samples_per_sec / 3600  # hours
        time_10_epochs = time_1_epoch * 10

        if time_1_epoch < 24:
            t1_str = f"{time_1_epoch:.1f} hours"
        else:
            t1_str = f"{time_1_epoch/24:.1f} days"

        if time_10_epochs < 24:
            t10_str = f"{time_10_epochs:.1f} hours"
        else:
            t10_str = f"{time_10_epochs/24:.1f} days"

        print(f"{name:<25} | {t1_str:>15} | {t10_str:>15}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--server_url", type=str, default="http://localhost:8010")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--skip_server", action="store_true", help="Skip vLLM server benchmark")
    parser.add_argument("--skip_direct", action="store_true", help="Skip direct encoder benchmark")
    args = parser.parse_args()

    batch_sizes = [1, 2, 4, 8]

    direct_results = None
    server_results = None

    # Benchmark direct encoder
    if not args.skip_direct:
        try:
            direct_results = benchmark_direct_encoder(
                args.model_path,
                num_iterations=args.iterations,
                batch_sizes=batch_sizes
            )
        except Exception as e:
            print(f"\n✗ Direct encoder benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # Benchmark vLLM server
    if not args.skip_server:
        server_results = benchmark_vllm_server(
            args.server_url,
            num_iterations=args.iterations,
            batch_sizes=batch_sizes
        )

    # Print comparison
    if direct_results:
        print_comparison(direct_results, server_results)


if __name__ == "__main__":
    main()
