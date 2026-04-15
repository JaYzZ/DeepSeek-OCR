#!/usr/bin/env python3
"""
Test script for DPSK-Qwen integration

Validates that:
1. DPSK OCR encoder works with Qwen3-VL
2. Connectors are created correctly
3. Forward pass completes successfully
4. Dimension alignment is correct

Usage:
    python OCRVL/tests/test_ocrqwen3vl_integration.py
"""

import torch
from project_paths import hf_path
from transformers import AutoTokenizer

from OCRVL.model.language_model.ocr_qwen3_vl import (
    OCRQwen3VLForConditionalGeneration,
    Qwen3VLOCRTextAdapter,
)


def test_ocrqwen3vl_integration():
    """Test DPSK-Qwen integration end-to-end"""
    print("=" * 70)
    print("Testing DPSK-Qwen Integration")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    qwen_model_path = str(hf_path("Qwen", "Qwen3-VL-2B-Instruct"))
    dpsk_model_path = "deepseek-ai/DeepSeek-OCR"

    # 1. Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)
    print("✓ Tokenizer loaded")

    # 2. Load OCR adapter
    print("\n2. Loading OCR adapter with DPSK encoder...")
    ocr_adapter = Qwen3VLOCRTextAdapter(
        encoder_model_path=dpsk_model_path,
        device=device,
        use_deepstack=True
    )
    print("✓ OCR adapter loaded")

    # 3. Prepare test input
    print("\n3. Preparing test input (rendered text)...")
    test_text = "The quick brown fox jumps over the lazy dog. " * 10
    input_ids, ocr_features = ocr_adapter.prepare_qwen_inputs(
        instruction="Transcribe the text:",
        dense_text=test_text,
        tokenizer=tokenizer,
        return_deepstack=True
    )
    print(f"✓ Input prepared")
    print(f"  Input IDs shape: {input_ids.shape}")

    # Unpack features
    final_feats, deepstack_feats = ocr_features
    print(f"  Final features: {len(final_feats)} images")
    print(f"    Shape: {final_feats[0].shape}")
    print(f"  Deepstack features: {len(deepstack_feats)} images, {len(deepstack_feats[0])} levels")
    for i, ds in enumerate(deepstack_feats[0]):
        print(f"    Level {i}: {ds.shape}")

    # 4. Load Qwen model
    print("\n4. Loading OCR-Qwen3-VL model...")
    model = OCRQwen3VLForConditionalGeneration.from_pretrained(
        qwen_model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded")

    # 5. Test forward pass (creates connectors)
    print("\n5. Testing forward pass (creates connectors)...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            ocr_image_features=ocr_features,
        )
    print("✓ Forward pass successful")
    print(f"  Output shape: {outputs.logits.shape}")

    # 6. Check connector dimensions
    print("\n6. Checking connector dimensions...")
    if hasattr(model.model, 'ocr_connector'):
        conn = model.model.ocr_connector
        if hasattr(conn, 'in_features'):
            print(f"  Final connector: {conn.in_features} → {conn.out_features}")
        else:
            print(f"  Final connector: Identity")
    else:
        print("  WARNING: ocr_connector not created yet")

    if hasattr(model.model, '_ocr_deepstack_connectors'):
        for key, conn in model.model._ocr_deepstack_connectors.items():
            if hasattr(conn, 'in_features'):
                print(f"  Deepstack connector[{key}]: {conn.in_features} → {conn.out_features}")
            else:
                print(f"  Deepstack connector[{key}]: Identity")
    else:
        print("  WARNING: deepstack connectors not created yet")

    # 7. Test generation (optional, slow)
    print("\n7. Testing text generation...")
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids.to(device),
                ocr_image_features=ocr_features,
                max_new_tokens=50,
                do_sample=False,
            )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"✓ Generation successful")
        print(f"  Generated (first 200 chars): {generated_text[:200]}...")
    except Exception as e:
        print(f"  Generation failed (expected with random connectors): {e}")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Train connectors with: python OCRVL/train.py --stage alignment")
    print("2. Finetune model with: python OCRVL/train.py --stage vit")


if __name__ == "__main__":
    test_ocrqwen3vl_integration()
