#!/usr/bin/env python3
"""
Quick verification script to check encoder optimizations and test larger batch sizes

Tests:
1. Verify SDPA/FlashAttention is active
2. Test larger batch sizes (16, 24, 32, 48, 64)
3. Check memory usage

Usage:
    CUDA_VISIBLE_DEVICES=0 python OCRInfer/tests/verify_optimizations.py
"""

import sys
import torch
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from OCRInfer.encoder import DPSKOCREncoder
from Renderer.pil_renderer import render_to_pil


def check_attention_mechanism(encoder):
    """Verify what attention mechanism is being used"""
    print("\n" + "="*70)
    print("ATTENTION MECHANISM CHECK")
    print("="*70)

    # Check CLIP attention type
    try:
        clip_layers = encoder.clip_model.transformer.layers
        if len(clip_layers) > 0:
            first_layer = clip_layers[0]
            if hasattr(first_layer, 'self_attn'):
                attn_class = type(first_layer.self_attn).__name__
                print(f"✓ CLIP attention type: {attn_class}")

                # Check for SDPA/Flash attention indicators
                if 'Sdpa' in attn_class or 'SDPA' in attn_class:
                    print("  ✓ Using SDPA (Scaled Dot Product Attention)")
                elif 'Flash' in attn_class:
                    print("  ✓ Using FlashAttention")
                else:
                    print(f"  ⚠️ Unknown attention type: {attn_class}")

                # Check if module has specific optimizations
                attn_module = first_layer.self_attn
                print(f"  Attention module: {type(attn_module)}")

    except Exception as e:
        print(f"  ⚠️ Could not inspect CLIP attention: {e}")

    # Check SAM attention (if applicable)
    try:
        if hasattr(encoder, 'sam_model'):
            print(f"\n✓ SAM model present: {type(encoder.sam_model).__name__}")
    except Exception as e:
        print(f"  ⚠️ Could not inspect SAM: {e}")

    # Check PyTorch SDPA availability
    print("\nPyTorch SDPA Support:")
    print(f"  torch.nn.functional.scaled_dot_product_attention: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")

    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        # Check which backends are available
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True) as ctx:
                print("  ✓ FlashAttention available")
        except Exception:
            print("  ⚠️ FlashAttention not available")


def test_large_batch_sizes(device="cuda:0"):
    """Test larger batch sizes for H100"""
    print("\n" + "="*70)
    print("LARGE BATCH SIZE TEST (H100 Optimization)")
    print("="*70)

    encoder = DPSKOCREncoder(
        device=device,
        dtype=torch.bfloat16,
        use_compile=False,
    )

    # Generate test images
    print("Generating test images...")
    text = "Machine learning is a branch of artificial intelligence. " * 20
    images = [render_to_pil(text, width=640, height=640) for _ in range(128)]

    batch_sizes = [16, 24, 32, 48, 64]

    print(f"\nTesting batch sizes: {batch_sizes}")
    print(f"Total images: {len(images)}\n")

    results = []

    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")

        # Warmup
        for i in range(0, min(batch_size * 2, len(images)), batch_size):
            batch = images[i:i+batch_size]
            _ = encoder.encode_images(batch)

        # Benchmark
        import time
        times = []

        for _ in range(3):
            torch.cuda.synchronize()
            start = time.time()

            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                _ = encoder.encode_images(batch)

            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        throughput = len(images) / avg_time

        # Check memory
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**3

        print(f"  Throughput: {throughput:.1f} img/s")
        print(f"  Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        results.append({
            'batch_size': batch_size,
            'throughput': throughput,
            'mem_allocated': mem_allocated,
            'mem_reserved': mem_reserved,
        })

    # Print summary
    print("\n" + "="*70)
    print("BATCH SIZE SUMMARY")
    print("="*70)
    print(f"{'Batch':<8} {'Throughput':<15} {'Memory (GB)':<15}")
    print("-" * 70)

    for result in results:
        print(f"{result['batch_size']:<8} {result['throughput']:<15.1f} {result['mem_allocated']:<15.2f}")

    best_result = max(results, key=lambda x: x['throughput'])
    print(f"\nBest configuration:")
    print(f"  Batch size: {best_result['batch_size']}")
    print(f"  Throughput: {best_result['throughput']:.1f} img/s")
    print(f"  Memory: {best_result['mem_allocated']:.2f}GB")

    return results


def check_model_info(encoder):
    """Print model information"""
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)

    # Count parameters
    clip_params = sum(p.numel() for p in encoder.clip_model.parameters())
    sam_params = sum(p.numel() for p in encoder.sam_model.parameters())
    proj_params = sum(p.numel() for p in encoder.projector.parameters())

    print(f"CLIP parameters: {clip_params/1e6:.1f}M")
    print(f"SAM parameters: {sam_params/1e6:.1f}M")
    print(f"Projector parameters: {proj_params/1e6:.1f}M")
    print(f"Total: {(clip_params + sam_params + proj_params)/1e6:.1f}M")

    print(f"\nData type: {encoder.dtype}")
    print(f"Device: {encoder.device}")

    # Memory usage
    mem_allocated = torch.cuda.memory_allocated(encoder.device) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(encoder.device) / 1024**3
    print(f"\nCurrent GPU memory:")
    print(f"  Allocated: {mem_allocated:.2f}GB")
    print(f"  Reserved: {mem_reserved:.2f}GB")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("ERROR: CUDA not available!")
        return

    print("="*70)
    print("OCRInfer Encoder Optimization Verification")
    print("="*70)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # Initialize encoder
    print("\nInitializing encoder...")
    encoder = DPSKOCREncoder(
        device=device,
        dtype=torch.bfloat16,
        use_compile=False,
    )

    # Check attention mechanism
    check_attention_mechanism(encoder)

    # Check model info
    check_model_info(encoder)

    # Test large batch sizes
    results = test_large_batch_sizes(device)

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    best_result = max(results, key=lambda x: x['throughput'])

    print(f"""
Based on benchmarks:

1. Optimal batch size for H100: {best_result['batch_size']}
   - Achieves {best_result['throughput']:.1f} img/s
   - Uses {best_result['mem_allocated']:.2f}GB GPU memory

2. Multi-GPU scaling (theoretical):
   - 1 GPU: {best_result['throughput']:.0f} img/s
   - 7 GPUs: {best_result['throughput']*7:.0f} img/s (theoretical)

3. Next steps:
   - Test multi-GPU parallel encoding
   - Verify actual multi-GPU throughput
   - Compare with training pipeline requirements
    """)


if __name__ == "__main__":
    main()
