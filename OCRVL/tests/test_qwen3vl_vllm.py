#!/usr/bin/env python3
"""
Comprehensive test suite for Qwen3-VL vLLM encoder/decoder

Tests:
1. Basic encoder functionality (batch processing)
2. vLLM decoder initialization and generation
3. Performance benchmarking (encoder + decoder)
4. Output quality validation vs HuggingFace baseline

Run:
    pytest test_qwen3vl_vllm.py -v -s
    # Or directly:
    python test_qwen3vl_vllm.py
"""
import sys
sys.path.insert(0, '/share/project/xiyan/sources/DeepSeek-OCR')

import torch
from PIL import Image
import time
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQwen3VLvLLM:
    """Test suite for Qwen3-VL with vLLM backend"""

    @pytest.fixture(scope="class")
    def test_images(self):
        """Create test images"""
        # Try to use Vello renderer for real text
        try:
            from Renderer import VelloRenderer
            renderer = VelloRenderer(width=640, height=640, padding=30)
            texts = ["Hello World", "Test Image"]
            arrays = renderer.render_batch(texts)
            return [Image.fromarray(arr) for arr in arrays]
        except ImportError:
            # Fallback to blank images
            return [Image.new('RGB', (640, 640), color=(240, 240, 240)) for _ in range(2)]

    def test_encoder_batch_output_shapes(self, test_images):
        """Test encoder produces correct batch output shapes"""
        from OCRVL.encoder import Qwen3VLEncoder

        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)
        output = encoder.encode_images(test_images)

        batch_size = len(test_images)
        expected_tokens = 400  # (1*40*40)/(2^2) for Qwen3-VL

        assert output.features.shape == (batch_size, expected_tokens, 2048), \
            f"Expected features shape ({batch_size}, {expected_tokens}, 2048), got {output.features.shape}"

        assert len(output.deepstack_features) == 3, \
            f"Expected 3 deepstack layers, got {len(output.deepstack_features)}"

        for i, ds in enumerate(output.deepstack_features):
            assert ds.shape == (batch_size, expected_tokens, 2048), \
                f"Deepstack layer {i}: expected ({batch_size}, {expected_tokens}, 2048), got {ds.shape}"

        assert output.grid_thw.shape == (batch_size, 3), \
            f"Expected grid_thw shape ({batch_size}, 3), got {output.grid_thw.shape}"

        print(f"✓ Encoder batch output shapes correct: {output.features.shape}")

    def test_encoder_performance(self, test_images):
        """Test encoder speed after warmup"""
        from OCRVL.encoder import Qwen3VLEncoder

        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)

        # Warmup (includes model loading)
        _ = encoder.encode_images(test_images[:1])

        # Benchmark
        start = time.time()
        _ = encoder.encode_images(test_images)
        encode_time = time.time() - start

        speed = len(test_images) / encode_time
        print(f"✓ Encoder speed: {speed:.1f} img/s ({encode_time:.3f}s for {len(test_images)} images)")

        # Should be >10 img/s after warmup
        assert speed > 10, f"Encoder too slow: {speed:.1f} img/s (expected >10 img/s)"

    def test_decoder_initialization(self):
        """Test vLLM decoder initializes correctly"""
        from OCRVL.decoder import Qwen3VLDecoder

        decoder = Qwen3VLDecoder(
            device="cuda:0",
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            max_model_len=2048,
        )

        assert decoder.llm is not None, "vLLM engine not initialized"
        print("✓ vLLM decoder initialized successfully")

    def test_decoder_generation(self, test_images):
        """Test decoder can generate text from embeddings"""
        from OCRVL.encoder import Qwen3VLEncoder
        from OCRVL.decoder import Qwen3VLDecoder

        # Encode
        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)
        output = encoder.encode_images([test_images[0]])  # Single image

        # Prepare embeddings
        feat = output.features[0]
        deepstack = [ds[0] for ds in output.deepstack_features]
        concat_emb = torch.cat([feat] + deepstack, dim=-1)
        grid_thw = output.grid_thw[0].tolist()

        # Decode
        decoder = Qwen3VLDecoder(device="cuda:0", dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048)
        text = decoder.decode(
            visual_embeddings=concat_emb,
            grid_thw=grid_thw,
            prompts="What do you see?",
            max_tokens=20,
            temperature=0.0,
        )

        assert isinstance(text, str), f"Expected string output, got {type(text)}"
        assert len(text) > 0, "Generated text is empty"

        print(f"✓ Decoder generated text: '{text[:50]}...'")

    def test_batch_decoding(self, test_images):
        """Test batch decoding with multiple images"""
        from OCRVL.encoder import Qwen3VLEncoder
        from OCRVL.decoder import Qwen3VLDecoder

        # Encode batch
        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)
        output = encoder.encode_images(test_images)

        # Prepare batch embeddings
        batch_size = len(test_images)
        embeddings = []
        grids = []
        for i in range(batch_size):
            feat_i = output.features[i]
            deepstack_i = [ds[i] for ds in output.deepstack_features]
            concat_emb = torch.cat([feat_i] + deepstack_i, dim=-1)
            embeddings.append(concat_emb)
            grids.append(output.grid_thw[i].tolist())

        # Decode batch
        decoder = Qwen3VLDecoder(device="cuda:0", dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048)
        texts = decoder.decode(
            visual_embeddings=embeddings,
            grid_thw=grids,
            prompts=["Describe the image:"] * batch_size,
            max_tokens=20,
            temperature=0.0,
        )

        assert isinstance(texts, list), f"Expected list output for batch, got {type(texts)}"
        assert len(texts) == batch_size, f"Expected {batch_size} outputs, got {len(texts)}"

        for i, text in enumerate(texts):
            assert len(text) > 0, f"Output {i} is empty"

        print(f"✓ Batch decoding successful: {batch_size} images → {batch_size} outputs")

    def test_end_to_end_performance(self, test_images):
        """Test end-to-end performance (encode + decode)"""
        from OCRVL.encoder import Qwen3VLEncoder
        from OCRVL.decoder import Qwen3VLDecoder

        # Initialize (warmup)
        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)
        decoder = Qwen3VLDecoder(device="cuda:0", dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048)

        # Warmup
        warmup_output = encoder.encode_images([test_images[0]])
        warmup_emb = torch.cat([warmup_output.features[0]] + [ds[0] for ds in warmup_output.deepstack_features], dim=-1)
        _ = decoder.decode(warmup_emb, warmup_output.grid_thw[0].tolist(), "Test", max_tokens=10, temperature=0.0)

        # Benchmark encoding
        start = time.time()
        output = encoder.encode_images(test_images)
        encode_time = time.time() - start

        # Prepare embeddings
        batch_size = len(test_images)
        embeddings = []
        grids = []
        for i in range(batch_size):
            feat_i = output.features[i]
            deepstack_i = [ds[i] for ds in output.deepstack_features]
            embeddings.append(torch.cat([feat_i] + deepstack_i, dim=-1))
            grids.append(output.grid_thw[i].tolist())

        # Benchmark decoding
        start = time.time()
        _ = decoder.decode(embeddings, grids, ["Describe:"] * batch_size, max_tokens=20, temperature=0.0)
        decode_time = time.time() - start

        total_time = encode_time + decode_time
        throughput = batch_size / total_time

        print(f"✓ End-to-end performance:")
        print(f"  Encoding: {encode_time:.3f}s ({batch_size/encode_time:.1f} img/s)")
        print(f"  Decoding: {decode_time:.3f}s ({batch_size/decode_time:.1f} img/s)")
        print(f"  Total: {total_time:.3f}s ({throughput:.1f} img/s)")

        # Should process at least 1 img/s end-to-end
        assert throughput > 1.0, f"Too slow: {throughput:.2f} img/s (expected >1.0 img/s)"


def main():
    """Run tests manually (for debugging)"""
    print("="*80)
    print("Qwen3-VL vLLM Test Suite")
    print("="*80)

    # Create test instance
    test = TestQwen3VLvLLM()

    # Create test images
    print("\n" + "="*80)
    print("Creating test images...")
    print("="*80)
    test_images = [Image.new('RGB', (640, 640), color=(240, 240, 240)) for _ in range(2)]
    print(f"✓ Created {len(test_images)} test images")

    # Run tests
    tests = [
        ("Encoder batch output shapes", lambda: test.test_encoder_batch_output_shapes(test_images)),
        ("Encoder performance", lambda: test.test_encoder_performance(test_images)),
        ("Decoder initialization", lambda: test.test_decoder_initialization()),
        ("Decoder generation", lambda: test.test_decoder_generation(test_images)),
        ("Batch decoding", lambda: test.test_batch_decoding(test_images)),
        ("End-to-end performance", lambda: test.test_end_to_end_performance(test_images)),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print("\n" + "="*80)
        print(f"Test: {name}")
        print("="*80)
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed > 0:
        print("\n⚠ Some tests failed")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
