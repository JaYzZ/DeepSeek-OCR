#!/usr/bin/env python3
"""
Cross-architecture test: OCR encoder -> Qwen3-VL decoder.

NOTE: This test is skipped by default due to architectural incompatibility.
"""

import pytest
import torch
from PIL import Image
from Renderer import VelloRenderer
from transformers import Qwen3VLConfig

from OCRInfer.encoder import DPSKOCREncoder
from OCRInfer.utils.model_paths import resolve_model_path
from OCRVL.model.language_model.ocr_qwen3_vl import _default_ocr_grid_thw
from Qwen.decoder import Qwen3VLDecoder


def create_test_image(text: str, width: int = 640, height: int = 640) -> Image.Image:
    renderer = VelloRenderer(width=width, height=height, padding=30)
    arr = renderer.render_batch([text])[0]
    return Image.fromarray(arr)


@pytest.mark.skip(reason="Cross-architecture OCR->Qwen3-VL not supported: deepstack dimension mismatch (1024 vs 1280)")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ocr_encoder_roundtrip():
    test_text = "Machine learning transforms industries worldwide."
    image = create_test_image(test_text)
    device = "cuda:0"

    encoder = DPSKOCREncoder(device=device, dtype=torch.bfloat16, remove_separators=True)
    ocr_features, deepstack = encoder.encode_images_with_deepstack([image])

    assert ocr_features[0].shape == (100, 1280)
    assert len(deepstack[0]) == 3
    for ds in deepstack[0]:
        assert ds.shape == (100, 1280)

    decoder = Qwen3VLDecoder(device=device)
    text_output = decoder.decode(
        visual_embeddings=ocr_features[0],
        deepstack_features=[deepstack[0]],
        grid_thw=[1, 10, 10],
        prompts="Transcribe all visible text:",
        max_tokens=100,
    )

    assert len(text_output) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grid_thw_handling():
    model_path = resolve_model_path("Qwen/Qwen3-VL-2B-Instruct")
    config = Qwen3VLConfig.from_pretrained(str(model_path))
    grid = _default_ocr_grid_thw(config, device="cuda:0")

    assert grid.shape == (3,)
    assert grid.tolist() == [1, 10, 10]
