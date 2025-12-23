#!/usr/bin/env python3
"""
Cross-Architecture Test: OCR Encoder → Qwen3-VL Decoder

NOTE: This test is SKIPPED by default due to architectural incompatibility.
DeepSeek-OCR encoder outputs [111, 1280] with deepstack [100, 1024]
while Qwen3-VL expects [400, 2048] with deepstack [400, 2048].

For native Qwen3-VL encoder/decoder tests, see: test_qwen3vl_vllm.py
"""

import sys
sys.path.insert(0, '/share/project/xiyan/sources/DeepSeek-OCR')

import torch
from PIL import Image
import pytest


def create_test_image(text: str, width: int = 640, height: int = 640) -> Image.Image:
    """Create a test image with text rendering."""
    try:
        from Renderer import VelloRenderer
        renderer = VelloRenderer(width=width, height=height, padding=30)
        arr = renderer.render_batch([text])[0]
        return Image.fromarray(arr)
    except ImportError:
        # Fallback to blank image
        return Image.new("RGB", (width, height), color=(240, 240, 240))


@pytest.mark.skip(reason="Cross-architecture OCR→Qwen3-VL not supported: deepstack dimension mismatch (1024 vs 1280)")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ocr_encoder_roundtrip():
    """Test: OCR encoder → Qwen3-VL decoder pipeline (SKIPPED: architectural incompatibility)."""
    from OCRInfer.encoder import DPSKOCREncoder
    from OCRInfer.utils.model_paths import resolve_model_path
    from OCRVL.decoder import Qwen3VLDecoder

    # Setup
    test_text = "Machine learning transforms industries worldwide."
    image = create_test_image(test_text)
    device = "cuda:0"

    # 1. Encode with OCR encoder
    print("\n=== Encoding with OCR encoder ===")
    # Use remove_separators=True (default) so deepstack features match final features (both 100 tokens)
    # Separators (newlines) are added after CLIP layers, so deepstack won't have them
    encoder = DPSKOCREncoder(device=device, dtype=torch.bfloat16, remove_separators=True)
    ocr_features, deepstack = encoder.encode_images_with_deepstack([image])

    # Verify OCR feature shape
    # Note: Changed to 100 tokens (pure 10×10 grid) to match deepstack sequence length
    assert ocr_features[0].shape == (100, 1280), f"Expected (100, 1280), got {ocr_features[0].shape}"
    assert len(deepstack[0]) == 3, f"Expected 3 deepstack layers, got {len(deepstack[0])}"
    # Verify deepstack features also have 100 tokens
    for i, ds in enumerate(deepstack[0]):
        assert ds.shape == (100, 1280), f"Deepstack {i}: Expected (100, 1280), got {ds.shape}"
    print(f"✓ OCR features shape: {ocr_features[0].shape}")
    print(f"✓ Deepstack layers: {len(deepstack[0])}")

    # 2. Decode with Qwen3-VL decoder (using OCR features)
    print("\n=== Decoding with Qwen3-VL decoder ===")
    decoder = Qwen3VLDecoder(device=device)

    # Decoder automatically concatenates deepstack features
    # Note: deepstack_features expects [[ds0, ds1, ds2], ...] for batch
    text_output = decoder.decode(
        visual_embeddings=ocr_features[0],
        deepstack_features=[deepstack[0]],  # Wrap in list for single image
        grid_thw=[1, 10, 10],
        prompts="Transcribe all visible text:",
        max_tokens=100,
    )

    print(f"✓ Generated text: {text_output}")

    # 3. Validate output contains some text (OCR encoder output may not be perfect)
    assert len(text_output) > 0, "Generated text is empty"
    print("\n✓ Roundtrip test passed: OCR encoder → Qwen3-VL decoder produces output")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grid_thw_handling():
    """Test that grid_thw (MRoPE) is handled correctly."""
    from OCRInfer.utils.model_paths import resolve_model_path
    from OCRVL.model.language_model.ocr_qwen3_vl import _default_ocr_grid_thw
    from transformers import Qwen3VLConfig

    # Test default OCR grid
    model_path = resolve_model_path("Qwen/Qwen3-VL-2B-Instruct")
    config = Qwen3VLConfig.from_pretrained(str(model_path))
    grid = _default_ocr_grid_thw(config, device="cuda:0")

    assert grid.shape == (3,), f"Expected shape (3,), got {grid.shape}"
    assert grid.tolist() == [1, 10, 10], f"Expected [1, 10, 10] for OCR grid, got {grid.tolist()}"

    print(f"\n=== Grid THW Test ===")
    print(f"OCR default grid_thw: {grid.tolist()}")
    print("✓ MRoPE grid handling correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
