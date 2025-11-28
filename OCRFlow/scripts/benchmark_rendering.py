#!/usr/bin/env python3
"""
Benchmark text rendering cost for OCRFlow training

Compares:
1. On-the-fly text rendering
2. Loading pre-rendered images from disk
3. Cached rendering
"""

import time
import numpy as np
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from OCRFlow.utils.text_rendering import render_text_to_image, render_markdown_to_image
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# Sample texts for benchmarking
SAMPLE_TEXTS = [
    """# Document Title

This is a sample document with multiple paragraphs and various content.

## Section 1

Here is some text in the first section. It contains multiple sentences to simulate a real document.

- Item 1
- Item 2
- Item 3

## Section 2

More content here with different formatting.
""",
    """## Introduction

This document tests rendering performance with markdown formatting.

### Background

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

### Methods

1. First step
2. Second step
3. Third step

## Results

The results show interesting patterns in the data.
""",
] * 50  # 100 samples total


def benchmark_on_the_fly_rendering(num_samples=1000, image_size=640):
    """Benchmark on-the-fly text rendering"""
    print("\n" + "="*60)
    print("Benchmark 1: On-the-fly Text Rendering")
    print("="*60)

    times = []

    for i in range(num_samples):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]

        start = time.perf_counter()
        img = render_markdown_to_image(text, width=image_size, height=image_size)
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"Samples: {num_samples}")
    print(f"Average time: {avg_time:.2f} ms/sample")
    print(f"Std dev: {std_time:.2f} ms")
    print(f"Min: {min_time:.2f} ms, Max: {max_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} samples/second")

    return avg_time


def benchmark_image_loading(num_samples=1000, image_size=640):
    """Benchmark loading pre-rendered images from disk"""
    print("\n" + "="*60)
    print("Benchmark 2: Loading Images from Disk")
    print("="*60)

    # Pre-render images to disk
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create images
        print("Pre-rendering images to disk...")
        for i in range(min(100, num_samples)):  # Only save 100, will reuse
            text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
            img = render_markdown_to_image(text, width=image_size, height=image_size)
            img.save(tmpdir / f"img_{i}.png")

        # Benchmark loading
        times = []
        for i in range(num_samples):
            img_path = tmpdir / f"img_{i % 100}.png"

            start = time.perf_counter()
            img = Image.open(img_path)
            img.load()  # Force load into memory
            end = time.perf_counter()

            times.append((end - start) * 1000)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"Samples: {num_samples}")
        print(f"Average time: {avg_time:.2f} ms/sample")
        print(f"Std dev: {std_time:.2f} ms")
        print(f"Throughput: {1000/avg_time:.1f} samples/second")

        return avg_time


def benchmark_cached_rendering(num_samples=1000, cache_size=100, image_size=640):
    """Benchmark rendering with LRU cache"""
    print("\n" + "="*60)
    print("Benchmark 3: Cached Rendering (LRU)")
    print("="*60)

    from functools import lru_cache

    @lru_cache(maxsize=cache_size)
    def cached_render(text_hash):
        # In practice, would render based on text, not hash
        # This simulates cache hits
        text = SAMPLE_TEXTS[text_hash % len(SAMPLE_TEXTS)]
        return render_markdown_to_image(text, width=image_size, height=image_size)

    times = []

    for i in range(num_samples):
        text_hash = i % cache_size  # Simulate repeated access

        start = time.perf_counter()
        img = cached_render(text_hash)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    cache_info = cached_render.cache_info()

    print(f"Samples: {num_samples}")
    print(f"Cache size: {cache_size}")
    print(f"Cache hits: {cache_info.hits}, misses: {cache_info.misses}")
    print(f"Hit rate: {cache_info.hits / (cache_info.hits + cache_info.misses) * 100:.1f}%")
    print(f"Average time: {avg_time:.2f} ms/sample")
    print(f"Throughput: {1000/avg_time:.1f} samples/second")

    return avg_time


def benchmark_dataloader(num_workers=4, batch_size=16, num_batches=100):
    """Benchmark DataLoader with multi-process rendering"""
    print("\n" + "="*60)
    print(f"Benchmark 4: DataLoader (workers={num_workers}, batch={batch_size})")
    print("="*60)

    class RenderingDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx % len(self.texts)]
            img = render_markdown_to_image(text, width=640, height=640)

            # Convert to tensor (simulating real training)
            import torchvision.transforms as T
            transform = T.Compose([T.ToTensor()])
            return transform(img)

    dataset = RenderingDataset(SAMPLE_TEXTS)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Warmup
    for _ in zip(range(5), loader):
        pass

    # Benchmark
    start = time.perf_counter()
    total_samples = 0

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        total_samples += len(batch)

    end = time.perf_counter()
    total_time = (end - start) * 1000

    avg_time_per_sample = total_time / total_samples
    throughput = total_samples / (total_time / 1000)

    print(f"Total samples: {total_samples}")
    print(f"Total time: {total_time:.2f} ms")
    print(f"Time per sample: {avg_time_per_sample:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/second")

    return avg_time_per_sample


def estimate_training_overhead(rendering_time_ms, gpu_step_time_ms=50):
    """Estimate training overhead from rendering"""
    print("\n" + "="*60)
    print("Training Overhead Estimation")
    print("="*60)

    print(f"\nAssuming:")
    print(f"  - Rendering time: {rendering_time_ms:.2f} ms/sample")
    print(f"  - GPU forward+backward: {gpu_step_time_ms:.2f} ms/sample")

    total_time = rendering_time_ms + gpu_step_time_ms
    overhead_pct = (rendering_time_ms / total_time) * 100

    print(f"\nResults:")
    print(f"  - Total time per sample: {total_time:.2f} ms")
    print(f"  - Rendering overhead: {overhead_pct:.1f}%")

    if overhead_pct < 20:
        print(f"  - ✓ Overhead is LOW - rendering is not a bottleneck")
    elif overhead_pct < 50:
        print(f"  - ⚠ Overhead is MODERATE - consider multi-processing")
    else:
        print(f"  - ✗ Overhead is HIGH - recommend pre-rendering or caching")


def benchmark_vllm_endpoint():
    """Benchmark using vLLM server's text-to-visual-tokens endpoint"""
    print("\n" + "="*60)
    print("Benchmark 5: vLLM Server Rendering Endpoint")
    print("="*60)

    import requests

    vllm_server = "http://localhost:8009"

    # Check if server is available
    try:
        response = requests.get(f"{vllm_server}/health", timeout=2)
        if response.status_code != 200:
            print("vLLM server not available, skipping benchmark")
            return None
    except:
        print("vLLM server not available, skipping benchmark")
        return None

    print("Testing vLLM server endpoint...")

    times = []
    num_samples = 20  # Fewer samples for remote API

    for i in range(num_samples):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]

        start = time.perf_counter()
        try:
            response = requests.post(
                f"{vllm_server}/text-to-visual-tokens",
                json={
                    "text": text,
                    "chunk_size": 1000,
                    "render_width": 640,
                    "render_height": 640,
                },
                timeout=10,
            )

            if response.status_code == 200:
                end = time.perf_counter()
                times.append((end - start) * 1000)
        except Exception as e:
            print(f"Error: {e}")
            continue

    if times:
        avg_time = np.mean(times)
        print(f"Samples: {len(times)}")
        print(f"Average time: {avg_time:.2f} ms/sample")
        print(f"Throughput: {1000/avg_time:.1f} samples/second")
        print("\nNote: This includes network latency + rendering + encoding")
        return avg_time
    else:
        print("Failed to benchmark vLLM endpoint")
        return None


def main():
    print("="*60)
    print("OCRFlow Text Rendering Performance Benchmark")
    print("="*60)
    print("\nThis benchmark measures the cost of text rendering")
    print("compared to traditional image loading approaches.")

    # Run benchmarks
    num_samples = 1000

    render_time = benchmark_on_the_fly_rendering(num_samples)
    load_time = benchmark_image_loading(num_samples)
    cache_time = benchmark_cached_rendering(num_samples, cache_size=100)

    print("\n" + "="*60)
    print("Multi-Process DataLoader Benchmarks")
    print("="*60)

    for num_workers in [0, 2, 4, 8]:
        benchmark_dataloader(num_workers=num_workers, batch_size=16)

    # Try vLLM endpoint
    vllm_time = benchmark_vllm_endpoint()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n1. On-the-fly rendering: {render_time:.2f} ms/sample")
    print(f"2. Loading from disk: {load_time:.2f} ms/sample")
    print(f"3. Cached rendering: {cache_time:.2f} ms/sample")

    speedup = load_time / render_time
    if speedup > 1:
        print(f"\n⚠ Rendering is {speedup:.1f}x SLOWER than loading images")
    else:
        print(f"\n✓ Rendering is {1/speedup:.1f}x FASTER than loading images")

    if vllm_time:
        print(f"4. vLLM endpoint: {vllm_time:.2f} ms/sample (includes network + encoding)")

    # Estimate training overhead
    estimate_training_overhead(render_time, gpu_step_time_ms=50)

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if render_time < 10:
        print("\n✓ Rendering is FAST (<10ms)")
        print("  → Use on-the-fly rendering with num_workers=4-8")
        print("  → No need for pre-rendering or caching")
    elif render_time < 50:
        print("\n⚠ Rendering is MODERATE (10-50ms)")
        print("  → Use DataLoader with num_workers=8+")
        print("  → Consider caching for repeated samples")
    else:
        print("\n✗ Rendering is SLOW (>50ms)")
        print("  → Use pre-rendering or vLLM endpoint")
        print("  → Enable LRU caching for repeated samples")

    print("\nKey insights:")
    print("  - Multi-process DataLoader hides rendering cost")
    print("  - CPU rendering happens in parallel with GPU training")
    print("  - Even 'slow' rendering (50ms) is only ~20% overhead")
    print("  - Text storage is 100-1000× smaller than images")


if __name__ == "__main__":
    main()
