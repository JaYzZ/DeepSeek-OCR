#!/usr/bin/env python3
"""Benchmark Qwen Vision Encoders.

Compares Qwen3-VL (406M) and Qwen2.5-VL (668M) vision encoders against
DPSKOCREncoder baseline (~294 img/s).

Usage:
    python Qwen/tests/benchmark_qwen_encoders.py
    CUDA_VISIBLE_DEVICES=0 python Qwen/tests/benchmark_qwen_encoders.py
"""

import logging
import time
import traceback

import torch
from PIL import Image

from OCRInfer.encoder import DPSKOCREncoder
from Qwen.encoder import Qwen25VLEncoder, Qwen3VLEncoder
from Renderer.pil_renderer import render_to_pil

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_test_images(num_images: int = 64, width: int = 640, height: int = 640) -> list:
    """Generate test images by rendering text."""
    logger.info(f"Generating {num_images} test images ({width}x{height})...")

    texts = [
        "Machine learning is transforming industries worldwide. " * 15
        for _ in range(num_images)
    ]

    images = []
    for text in texts:
        img = render_to_pil(text, width=width, height=height, padding=20)
        images.append(img)

    return images


def benchmark_encoder(
    encoder,
    images: list,
    batch_size: int = 8,
    num_iterations: int = 3,
    warmup_iterations: int = 1,
    name: str = "Encoder",
) -> dict:
    """Benchmark encoder performance."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Benchmarking: {name}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Total images: {len(images)}")
    logger.info(f"{'='*70}")

    # Warmup
    logger.info("Warming up...")
    for i in range(warmup_iterations):
        batch = images[:min(batch_size, len(images))]
        try:
            _ = encoder.encode_images(batch)
            logger.info(f"  Warmup {i+1}/{warmup_iterations} complete")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            traceback.print_exc()
            return None

    # Benchmark
    times = []
    logger.info("Benchmarking...")

    for iteration in range(num_iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        try:
            # Process all images in batches
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                _ = encoder.encode_images(batch)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            times.append(elapsed)

            throughput = len(images) / elapsed
            logger.info(f"  Iteration {iteration+1}: {elapsed*1000:.1f} ms ({throughput:.1f} img/s)")

        except Exception as e:
            logger.error(f"Iteration {iteration+1} failed: {e}")
            traceback.print_exc()
            return None

    # Calculate statistics
    min_time = min(times)
    avg_time = sum(times) / len(times)
    throughput_best = len(images) / min_time
    throughput_avg = len(images) / avg_time

    logger.info(f"\nResults:")
    logger.info(f"  Best time: {min_time*1000:.1f} ms")
    logger.info(f"  Avg time: {avg_time*1000:.1f} ms")
    logger.info(f"  Best throughput: {throughput_best:.1f} img/s")
    logger.info(f"  Avg throughput: {throughput_avg:.1f} img/s")
    logger.info(f"  Per image: {avg_time/len(images)*1000:.2f} ms")

    return {
        'name': name,
        'batch_size': batch_size,
        'min_time': min_time,
        'avg_time': avg_time,
        'throughput_best': throughput_best,
        'throughput_avg': throughput_avg,
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        logger.error("CUDA not available! Benchmarks require GPU.")
        return

    logger.info("="*70)
    logger.info("Qwen Vision Encoder Benchmark")
    logger.info("="*70)
    logger.info(f"Device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info("="*70)

    # Generate test images
    images = generate_test_images(64, width=640, height=640)

    all_results = []

    # Benchmark DPSKOCREncoder (baseline)
    logger.info("\n" + "="*70)
    logger.info("BASELINE: DPSKOCREncoder")
    logger.info("="*70)

    try:
        dpsk_encoder = DPSKOCREncoder(device=device, dtype=torch.bfloat16)
        for batch_size in [1, 4, 8, 16]:
            result = benchmark_encoder(
                dpsk_encoder,
                images,
                batch_size=batch_size,
                num_iterations=2,
                warmup_iterations=1,
                name=f"DPSKOCR (batch={batch_size})",
            )
            if result:
                all_results.append(result)
    except Exception as e:
        logger.error(f"DPSKOCR benchmark failed: {e}")
        traceback.print_exc()

    # Benchmark Qwen3-VL
    logger.info("\n" + "="*70)
    logger.info("Qwen3-VL Vision Encoder (406M params)")
    logger.info("="*70)

    try:
        qwen3_encoder = Qwen3VLEncoder(device=device, dtype=torch.bfloat16)
        for batch_size in [1, 2, 4, 8]:
            result = benchmark_encoder(
                qwen3_encoder,
                images,
                batch_size=batch_size,
                num_iterations=2,
                warmup_iterations=1,
                name=f"Qwen3-VL (batch={batch_size})",
            )
            if result:
                all_results.append(result)
    except Exception as e:
        logger.error(f"Qwen3-VL benchmark failed: {e}")
        traceback.print_exc()

    # Benchmark Qwen2.5-VL
    logger.info("\n" + "="*70)
    logger.info("Qwen2.5-VL Vision Encoder (668M params)")
    logger.info("="*70)

    try:
        qwen25_encoder = Qwen25VLEncoder(device=device, dtype=torch.bfloat16)
        for batch_size in [1, 2, 4, 8]:
            result = benchmark_encoder(
                qwen25_encoder,
                images,
                batch_size=batch_size,
                num_iterations=2,
                warmup_iterations=1,
                name=f"Qwen2.5-VL (batch={batch_size})",
            )
            if result:
                all_results.append(result)
    except Exception as e:
        logger.error(f"Qwen2.5-VL benchmark failed: {e}")
        traceback.print_exc()

    # Print summary
    if all_results:
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*70)

        sorted_results = sorted(all_results, key=lambda x: x['throughput_avg'], reverse=True)

        logger.info(f"\nAll results sorted by avg throughput:")
        for i, result in enumerate(sorted_results, 1):
            logger.info(
                f"  {i}. {result['throughput_avg']:6.1f} img/s | "
                f"batch={result['batch_size']:2d} | {result['name']}"
            )

        # Find best for each encoder
        dpsk_results = [r for r in all_results if 'DPSKOCR' in r['name']]
        qwen3_results = [r for r in all_results if 'Qwen3-VL' in r['name']]
        qwen25_results = [r for r in all_results if 'Qwen2.5-VL' in r['name']]

        logger.info(f"\n" + "="*70)
        logger.info("BEST CONFIGURATION PER ENCODER")
        logger.info("="*70)

        if dpsk_results:
            best_dpsk = max(dpsk_results, key=lambda x: x['throughput_avg'])
            logger.info(f"\nDPSKOCREncoder: {best_dpsk['throughput_avg']:.1f} img/s (batch={best_dpsk['batch_size']})")

        if qwen3_results:
            best_qwen3 = max(qwen3_results, key=lambda x: x['throughput_avg'])
            logger.info(f"Qwen3-VL:       {best_qwen3['throughput_avg']:.1f} img/s (batch={best_qwen3['batch_size']})")

        if qwen25_results:
            best_qwen25 = max(qwen25_results, key=lambda x: x['throughput_avg'])
            logger.info(f"Qwen2.5-VL:     {best_qwen25['throughput_avg']:.1f} img/s (batch={best_qwen25['batch_size']})")

        logger.info(f"\n" + "="*70)
        logger.info("MEASURED RESULTS COMPLETE")
        logger.info("="*70)
    else:
        logger.error("All benchmarks failed.")


if __name__ == "__main__":
    main()
