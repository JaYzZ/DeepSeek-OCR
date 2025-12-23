#!/usr/bin/env python3
"""
Benchmark comparison: Vello (GPU) vs Skia (CPU) vs PIL

Compares:
1. Rendering throughput (images/second)
2. Worker scaling (for CPU renderers)
3. Memory usage
4. Image quality (visual inspection)
"""

import sys
import time
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Renderer.pil_renderer import PILRenderer
import numpy as np

# Try to import Vello (local to this package)
try:
    from vello_renderer_wrapper import VelloRenderer
    VELLO_AVAILABLE = True
except ImportError:
    VELLO_AVAILABLE = False
    print("⚠️  Vello renderer not available - GPU benchmark will be skipped")

# Try to import Skia (local to this package)
try:
    from skia_renderer import SkiaRenderer
    SKIA_AVAILABLE = True
except ImportError:
    SKIA_AVAILABLE = False
    SkiaRenderer = None
    print("⚠️  Skia renderer not available")


def generate_test_texts(num_texts: int = 64, vary_length: bool = True):
    """Generate test texts with varying lengths"""
    base_short = "Machine learning. " * 10
    base_medium = "Machine learning is a branch of AI. " * 20
    base_long = "Machine learning is a branch of AI that focuses on algorithms. " * 30

    if vary_length:
        # Mix of different lengths
        texts = []
        for i in range(num_texts):
            if i % 3 == 0:
                texts.append(base_short)
            elif i % 3 == 1:
                texts.append(base_medium)
            else:
                texts.append(base_long)
        return texts
    else:
        # All same length
        return [base_medium] * num_texts


def benchmark_renderer(renderer_class, name, texts, num_workers=16, iterations=5):
    """Benchmark a renderer"""
    print(f"\n{'=' * 70}")
    print(f"{name} Benchmark")
    print(f"{'=' * 70}")

    # Create renderer
    renderer = renderer_class(num_workers=num_workers)

    # Warmup
    print("Warming up...")
    renderer.render_batch(texts[:4])

    # Benchmark
    times = []
    for i in range(iterations):
        start = time.time()
        images = renderer.render_batch(texts)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.1f} ms ({len(texts)/elapsed:.1f} img/s)")

    min_time = min(times)
    avg_time = sum(times) / len(times)
    rate_best = len(texts) / min_time
    rate_avg = len(texts) / avg_time

    print(f"\nResults:")
    print(f"  Batch size: {len(texts)}")
    print(f"  Best time: {min_time*1000:.1f} ms")
    print(f"  Avg time: {avg_time*1000:.1f} ms")
    print(f"  Best rate: {rate_best:.1f} img/s")
    print(f"  Avg rate: {rate_avg:.1f} img/s")
    print(f"  Time per image: {avg_time/len(texts)*1000:.2f} ms")

    renderer.shutdown()
    return rate_avg


def benchmark_worker_scaling(renderer_class, name, texts):
    """Test worker scaling"""
    print(f"\n{'=' * 70}")
    print(f"{name} - Worker Scaling")
    print(f"{'=' * 70}")

    worker_counts = [4, 8, 12, 16, 20, 24]
    results = []

    for num_workers in worker_counts:
        renderer = renderer_class(num_workers=num_workers)
        renderer.render_batch(texts[:4])  # warmup

        start = time.time()
        for _ in range(3):
            renderer.render_batch(texts)
        elapsed = (time.time() - start) / 3

        rate = len(texts) / elapsed
        results.append((num_workers, rate))
        print(f"  {num_workers:2d} workers: {rate:6.1f} img/s")
        renderer.shutdown()

    return results


def compare_quality(texts, output_dir="./comparison_samples"):
    """Generate sample images for visual comparison"""
    print(f"\n{'=' * 70}")
    print("Generating Quality Comparison Samples")
    print(f"{'=' * 70}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Take first 3 texts
    sample_texts = texts[:3]

    # Render with PIL
    print("Rendering with PIL...")
    pil_renderer = PILRenderer(num_workers=1)
    pil_images = pil_renderer.render_batch_pil(sample_texts)
    for i, img in enumerate(pil_images):
        img.save(output_path / f"sample_{i}_pil.png")
    pil_renderer.shutdown()

    # Render with Skia (if available)
    if SKIA_AVAILABLE and SkiaRenderer is not None:
        print("Rendering with Skia...")
        skia_renderer = SkiaRenderer(num_workers=1)
        skia_images = skia_renderer.render_batch_pil(sample_texts)
        for i, img in enumerate(skia_images):
            img.save(output_path / f"sample_{i}_skia.png")
        skia_renderer.shutdown()
    else:
        print("Skipping Skia rendering (not available)")

    print(f"Saved comparison samples to {output_path}")
    print("Compare visually for:")
    print("  - Font rendering quality (anti-aliasing)")
    print("  - Kerning and spacing")
    print("  - Text layout accuracy")


def benchmark_vello_renderer(texts, iterations=5):
    """Benchmark Vello GPU renderer"""
    print(f"\n{'=' * 70}")
    print("VelloRenderer (GPU) Benchmark")
    print(f"{'=' * 70}")

    # Create renderer (no num_workers parameter)
    renderer = VelloRenderer(width=640, height=640, padding=20)

    # Warmup
    print("Warming up...")
    renderer.render_batch_pil(texts[:4])

    # Benchmark
    times = []
    for i in range(iterations):
        start = time.time()
        images = renderer.render_batch_pil(texts)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.1f} ms ({len(texts)/elapsed:.1f} img/s)")

    min_time = min(times)
    avg_time = sum(times) / len(times)
    rate_best = len(texts) / min_time
    rate_avg = len(texts) / avg_time

    print(f"\nResults:")
    print(f"  Batch size: {len(texts)}")
    print(f"  Best time: {min_time*1000:.1f} ms")
    print(f"  Avg time: {avg_time*1000:.1f} ms")
    print(f"  Best rate: {rate_best:.1f} img/s")
    print(f"  Avg rate: {rate_avg:.1f} img/s")
    print(f"  Time per image: {avg_time/len(texts)*1000:.2f} ms")

    renderer.shutdown()
    return rate_avg


def main():
    print("=" * 70)
    print("Renderer Performance Comparison")
    print("=" * 70)
    print()

    # Generate test data
    texts_small = generate_test_texts(64, vary_length=True)
    texts_large = generate_test_texts(256, vary_length=True)

    # 0. Benchmark Vello GPU renderer (if available)
    vello_rate_small = None
    vello_rate_large = None
    if VELLO_AVAILABLE:
        vello_rate_small = benchmark_vello_renderer(texts_small)
        vello_rate_large = benchmark_vello_renderer(texts_large)
    else:
        print("\n⚠️  Skipping Vello GPU benchmark (not available)")

    # 1. Benchmark PIL renderer
    pil_rate_small = benchmark_renderer(
        PILRenderer, "PILRenderer (PIL)", texts_small
    )
    pil_rate_large = benchmark_renderer(
        PILRenderer, "PILRenderer (PIL)", texts_large
    )

    # 2. Benchmark Skia renderer (if available)
    skia_rate_small = None
    skia_rate_large = None
    if SKIA_AVAILABLE and SkiaRenderer is not None:
        skia_rate_small = benchmark_renderer(
            SkiaRenderer, "SkiaRenderer (Skia+Binary Search)", texts_small
        )
        skia_rate_large = benchmark_renderer(
            SkiaRenderer, "SkiaRenderer (Skia+Binary Search)", texts_large
        )
    else:
        print("\n⚠️  Skipping Skia CPU benchmark (not available)")

    # 3. Worker scaling comparison
    print("\n\n" + "=" * 70)
    print("WORKER SCALING COMPARISON")
    print("=" * 70)
    pil_scaling = benchmark_worker_scaling(PILRenderer, "PIL", texts_small)
    if SKIA_AVAILABLE and SkiaRenderer is not None:
        skia_scaling = benchmark_worker_scaling(SkiaRenderer, "Skia", texts_small)
    else:
        skia_scaling = []

    # 4. Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBatch Size: 64")
    if VELLO_AVAILABLE and vello_rate_small:
        print(f"  Vello (GPU): {vello_rate_small:6.1f} img/s")
    if SKIA_AVAILABLE and skia_rate_small:
        print(f"  Skia  (CPU): {skia_rate_small:6.1f} img/s")
    print(f"  PIL   (CPU): {pil_rate_small:6.1f} img/s")
    if VELLO_AVAILABLE and vello_rate_small and SKIA_AVAILABLE and skia_rate_small:
        print(f"  Vello vs Skia: {vello_rate_small/skia_rate_small:.2f}x speedup")
    if VELLO_AVAILABLE and vello_rate_small:
        print(f"  Vello vs PIL:  {vello_rate_small/pil_rate_small:.2f}x speedup")
    if SKIA_AVAILABLE and skia_rate_small:
        print(f"  Skia vs PIL:   {skia_rate_small/pil_rate_small:.2f}x speedup")

    print(f"\nBatch Size: 256")
    if VELLO_AVAILABLE and vello_rate_large:
        print(f"  Vello (GPU): {vello_rate_large:6.1f} img/s")
    if SKIA_AVAILABLE and skia_rate_large:
        print(f"  Skia  (CPU): {skia_rate_large:6.1f} img/s")
    print(f"  PIL   (CPU): {pil_rate_large:6.1f} img/s")
    if VELLO_AVAILABLE and vello_rate_large and SKIA_AVAILABLE and skia_rate_large:
        print(f"  Vello vs Skia: {vello_rate_large/skia_rate_large:.2f}x speedup")
    if VELLO_AVAILABLE and vello_rate_large:
        print(f"  Vello vs PIL:  {vello_rate_large/pil_rate_large:.2f}x speedup")
    if SKIA_AVAILABLE and skia_rate_large:
        print(f"  Skia vs PIL:   {skia_rate_large/pil_rate_large:.2f}x speedup")

    print(f"\nBest Worker Configuration (CPU renderers):")
    pil_best = max(pil_scaling, key=lambda x: x[1])
    print(f"  PIL:   {pil_best[0]} workers → {pil_best[1]:.1f} img/s")
    if SKIA_AVAILABLE and skia_scaling:
        skia_best = max(skia_scaling, key=lambda x: x[1])
        print(f"  Skia:  {skia_best[0]} workers → {skia_best[1]:.1f} img/s")
    if VELLO_AVAILABLE and vello_rate_small:
        print(f"  Vello: GPU (no workers) → {vello_rate_small:.1f} img/s")

    # 5. Generate quality comparison samples
    compare_quality(texts_small[:3])

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('fork', force=True)
    main()
