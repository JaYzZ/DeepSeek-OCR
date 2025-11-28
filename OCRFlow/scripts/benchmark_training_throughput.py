#!/usr/bin/env python3
"""
Benchmark Training Throughput for Markovian Chunk Decoder

This script finds the optimal batch_size and num_render_workers configuration
by measuring:
1. Text rendering throughput (CPU bound)
2. Vision encoding throughput (GPU bound)
3. End-to-end training throughput

Usage:
    CUDA_VISIBLE_DEVICES=0 python OCRFlow/scripts/benchmark_training_throughput.py
    CUDA_VISIBLE_DEVICES=0,1 python OCRFlow/scripts/benchmark_training_throughput.py --multi_gpu
"""

import torch
import time
import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_sample_texts(num_texts: int, words_per_text: int = 500) -> list:
    """Generate sample texts for benchmarking"""
    base_words = [
        "machine", "learning", "artificial", "intelligence", "neural",
        "network", "deep", "transformer", "attention", "model",
        "training", "data", "optimization", "gradient", "loss",
        "batch", "epoch", "learning", "rate", "weight",
        "parameter", "layer", "activation", "function", "output",
    ]

    texts = []
    for i in range(num_texts):
        words = [base_words[(i + j) % len(base_words)] for j in range(words_per_text)]
        texts.append(" ".join(words))
    return texts


def benchmark_renderer(num_workers_list: list, batch_sizes: list, num_warmup: int = 2, num_iterations: int = 5):
    """Benchmark UltraFastRenderer at various worker counts"""
    from OCRFlow.utils.ultra_fast_renderer import UltraFastRenderer

    print("=" * 70)
    print("BENCHMARK: UltraFast Text Renderer")
    print("=" * 70)

    results = {}

    for batch_size in batch_sizes:
        texts = generate_sample_texts(batch_size, words_per_text=500)
        print(f"\nBatch size: {batch_size}")
        print("-" * 40)

        for num_workers in num_workers_list:
            renderer = UltraFastRenderer(num_workers=num_workers)

            # Warmup
            for _ in range(num_warmup):
                renderer.render_batch(texts)

            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.time()
                renderer.render_batch(texts)
                elapsed = time.time() - start
                times.append(elapsed)

            renderer.shutdown()

            min_time = min(times)
            rate = batch_size / min_time
            results[(batch_size, num_workers)] = rate

            print(f"  {num_workers:2d} workers: {rate:6.1f} img/s (best: {min_time*1000:.1f}ms)")

    return results


def benchmark_encoder(batch_sizes: list, num_render_workers: int = 16, num_iterations: int = 5, device: str = "cuda"):
    """Benchmark Vision Encoder at various batch sizes"""
    from OCRFlow.utils.vision_encoder import create_vision_encoder

    print("\n" + "=" * 70)
    print("BENCHMARK: Vision Encoder (CLIP + SAM + Projector)")
    print("=" * 70)

    encoder = create_vision_encoder(device=device, num_render_workers=num_render_workers)

    # Warmup
    warmup_texts = generate_sample_texts(8, 500)
    encoder.encode_texts(warmup_texts)
    torch.cuda.synchronize()

    results = {}

    for batch_size in batch_sizes:
        texts = generate_sample_texts(batch_size, words_per_text=500)

        # Warmup this batch size
        encoder.encode_texts(texts)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            tokens = encoder.encode_texts(texts)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

        min_time = min(times)
        rate = batch_size / min_time
        results[batch_size] = rate

        # Memory usage
        mem_gb = torch.cuda.max_memory_allocated() / 1e9

        print(f"Batch {batch_size:3d}: {rate:6.1f} samples/s (best: {min_time*1000:.0f}ms, peak mem: {mem_gb:.1f}GB)")

    return results, encoder


def benchmark_training_step(batch_sizes: list, model_size: str = "large", num_iterations: int = 10, device: str = "cuda"):
    """Benchmark training step at various batch sizes"""
    from OCRFlow.models.markovian_chunk_decoder import create_chunk_decoder
    from torch.cuda.amp import autocast, GradScaler

    print("\n" + "=" * 70)
    print(f"BENCHMARK: Training Step ({model_size} model)")
    print("=" * 70)

    model = create_chunk_decoder(model_size=model_size).to(device)
    model.train()

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    results = {}

    for batch_size in batch_sizes:
        # Create fake visual tokens [batch, 111, 1280]
        input_chunks = torch.randn(batch_size, 111, 1280, device=device, dtype=torch.float32)
        target_chunks = torch.randn(batch_size, 111, 1280, device=device, dtype=torch.float32)
        chunk_sequences = torch.stack([input_chunks, target_chunks], dim=1)

        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            with autocast(enabled=True):
                loss, _ = model.compute_loss(chunk_sequences)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            optimizer.zero_grad()
            torch.cuda.synchronize()
            start = time.time()

            with autocast(enabled=True):
                loss, _ = model.compute_loss(chunk_sequences)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

        min_time = min(times)
        avg_time = np.mean(times)
        rate = batch_size / avg_time
        results[batch_size] = rate

        mem_gb = torch.cuda.max_memory_allocated() / 1e9

        print(f"Batch {batch_size:3d}: {rate:6.1f} samples/s (avg: {avg_time*1000:.0f}ms, peak mem: {mem_gb:.1f}GB)")

        # Clear memory for next batch size
        torch.cuda.reset_peak_memory_stats()

    return results


def benchmark_end_to_end(
    batch_sizes: list,
    num_render_workers_list: list,
    num_iterations: int = 5,
    device: str = "cuda",
):
    """
    End-to-end benchmark: render -> encode -> train

    Measures the entire pipeline throughput.
    """
    from OCRFlow.utils.vision_encoder import create_vision_encoder
    from OCRFlow.models.markovian_chunk_decoder import create_chunk_decoder
    from OCRFlow.utils.ultra_fast_renderer import UltraFastRenderer
    from torch.cuda.amp import autocast, GradScaler

    print("\n" + "=" * 70)
    print("BENCHMARK: End-to-End Pipeline (Render → Encode → Train)")
    print("=" * 70)

    # Create model
    model = create_chunk_decoder(model_size="large").to(device)
    model.train()
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    results = {}

    for num_workers in num_render_workers_list:
        print(f"\n--- Render Workers: {num_workers} ---")

        # Create encoder with this worker count
        encoder = create_vision_encoder(device=device, num_render_workers=num_workers)

        for batch_size in batch_sizes:
            # Generate texts for 2 chunks per sample (input + target)
            num_texts = batch_size * 2
            texts = generate_sample_texts(num_texts, words_per_text=500)

            # Warmup
            tokens = encoder.encode_texts(texts[:16])
            torch.cuda.synchronize()

            # Benchmark full pipeline
            times = []
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.time()

                # 1. Encode texts (includes rendering)
                tokens = encoder.encode_texts(texts)

                # 2. Prepare batch
                input_tokens = torch.stack([t for t in tokens[::2]], dim=0).float().to(device)
                target_tokens = torch.stack([t for t in tokens[1::2]], dim=0).float().to(device)
                chunk_sequences = torch.stack([input_tokens, target_tokens], dim=1)

                # 3. Training step
                optimizer.zero_grad()
                with autocast(enabled=True):
                    loss, _ = model.compute_loss(chunk_sequences)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                torch.cuda.synchronize()
                elapsed = time.time() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            rate = batch_size / avg_time
            results[(batch_size, num_workers)] = rate

            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Batch {batch_size:3d}: {rate:5.1f} pairs/s (avg: {avg_time*1000:.0f}ms, mem: {mem_gb:.1f}GB)")

        # Reset for next worker count
        torch.cuda.reset_peak_memory_stats()

    return results


def find_optimal_config(e2e_results: dict):
    """Find the optimal batch_size and num_workers configuration"""
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)

    # Find best configuration
    best_config = max(e2e_results, key=e2e_results.get)
    best_rate = e2e_results[best_config]

    batch_size, num_workers = best_config

    print(f"\nBest configuration:")
    print(f"  batch_size = {batch_size}")
    print(f"  num_render_workers = {num_workers}")
    print(f"  Throughput = {best_rate:.1f} pairs/sec")

    # Show top 5 configurations
    print("\nTop 5 configurations:")
    sorted_configs = sorted(e2e_results.items(), key=lambda x: x[1], reverse=True)[:5]
    for (bs, nw), rate in sorted_configs:
        print(f"  batch_size={bs:2d}, workers={nw:2d}: {rate:.1f} pairs/s")

    return best_config


def main():
    parser = argparse.ArgumentParser(description="Benchmark training throughput")
    parser.add_argument("--device", type=str, default="cuda", help="Device to benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with fewer configurations")
    parser.add_argument("--multi_gpu", action="store_true", help="Test with multi-GPU setup")
    args = parser.parse_args()

    print("Training Throughput Benchmark")
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Configuration ranges to test
    if args.quick:
        batch_sizes = [8, 16, 32]
        num_workers_list = [8, 16, 24]
    else:
        # Full benchmark for 80GB H100
        batch_sizes = [4, 8, 12, 16, 20, 24, 28, 32, 48, 64]
        num_workers_list = [4, 8, 12, 16, 20, 24, 32]

    # 1. Benchmark renderer alone
    renderer_results = benchmark_renderer(num_workers_list, batch_sizes[:5])

    # 2. Benchmark encoder alone
    encoder_results, encoder = benchmark_encoder(batch_sizes, num_render_workers=16)

    # 3. Benchmark training step alone
    training_results = benchmark_training_step(batch_sizes)

    # 4. End-to-end benchmark
    e2e_results = benchmark_end_to_end(batch_sizes, num_workers_list)

    # 5. Find optimal configuration
    optimal_config = find_optimal_config(e2e_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nRenderer throughput (img/s):")
    for (bs, nw), rate in sorted(renderer_results.items()):
        print(f"  batch={bs:2d}, workers={nw:2d}: {rate:.1f}")

    print("\nEncoder throughput (samples/s):")
    for bs, rate in sorted(encoder_results.items()):
        print(f"  batch={bs:2d}: {rate:.1f}")

    print("\nTraining throughput (samples/s):")
    for bs, rate in sorted(training_results.items()):
        print(f"  batch={bs:2d}: {rate:.1f}")

    print("\n" + "=" * 70)
    print("RECOMMENDED COMMAND")
    print("=" * 70)

    batch_size, num_workers = optimal_config
    print(f"""
# Single GPU
CUDA_VISIBLE_DEVICES=0 python OCRFlow/examples/train_markovian.py \\
    --dataset_type fineweb \\
    --fineweb_subset 10BT \\
    --batch_size {batch_size} \\
    --num_render_workers {num_workers} \\
    --max_steps 50000

# Multi-GPU (CUDA 0,1)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 OCRFlow/examples/train_markovian.py \\
    --dataset_type fineweb \\
    --fineweb_subset 10BT \\
    --batch_size {batch_size} \\
    --num_render_workers {num_workers} \\
    --max_steps 50000
""")


if __name__ == "__main__":
    main()
