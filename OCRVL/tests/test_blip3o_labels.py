#!/usr/bin/env python3
"""Test BLIP3o dataset label creation fix"""

import torch
from OCRVL.data.blip3o_dataset import BLIP3oAlignmentDataset
from OCRVL.model.language_model.ocr_qwen3_vl import Qwen3VLOCRTextAdapter
from transformers import AutoTokenizer


def test_label_creation():
    """Test that labels are created correctly (vision tokens masked)"""

    print("Initializing tokenizer and OCR adapter...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", trust_remote_code=True)
    ocr_adapter = Qwen3VLOCRTextAdapter(
        encoder_model_path="deepseek-ai/DeepSeek-OCR",
        device="cuda:0",
        use_deepstack=True
    )

    # Get vision token IDs
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_token_ids = {vision_start_id, vision_end_id, image_pad_id}

    print(f"Vision token IDs: start={vision_start_id}, end={vision_end_id}, pad={image_pad_id}")

    print("\nLoading BLIP3o dataset (0.0001% sample)...")
    dataset = BLIP3oAlignmentDataset(
        dataset_type="long",
        tokenizer=tokenizer,
        ocr_adapter=ocr_adapter,
        sample_percentage=0.0001,  # Tiny sample for testing
        use_images=False,  # Use text rendering to test the fix
        image_to_caption_ratio=0.0,
        seed=42
    )

    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return False

    print(f"Dataset loaded: {len(dataset)} samples\n")

    # Test a sample
    print("Testing sample 0...")
    sample = dataset[0]

    input_ids = sample['input_ids']
    labels = sample['labels']

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    # Check that vision tokens in input_ids are masked in labels
    vision_token_count = 0
    masked_vision_count = 0
    text_token_count = 0
    unmasked_text_count = 0

    for i, (inp_id, label_id) in enumerate(zip(input_ids, labels)):
        inp_val = inp_id.item()
        label_val = label_id.item()

        if inp_val in vision_token_ids:
            vision_token_count += 1
            if label_val == -100:
                masked_vision_count += 1
        else:
            text_token_count += 1
            if label_val != -100:
                unmasked_text_count += 1

    print(f"\nAnalysis:")
    print(f"  Vision tokens in input: {vision_token_count}")
    print(f"  Vision tokens masked in labels: {masked_vision_count}")
    print(f"  Text tokens in input: {text_token_count}")
    print(f"  Text tokens with labels: {unmasked_text_count}")

    # Verification
    success = True

    if vision_token_count > 0:
        if masked_vision_count == vision_token_count:
            print("  ✓ All vision tokens are masked in labels")
        else:
            print(f"  ✗ ERROR: {vision_token_count - masked_vision_count} vision tokens are NOT masked!")
            success = False

    if text_token_count > 0:
        if unmasked_text_count > 0:
            print(f"  ✓ {unmasked_text_count}/{text_token_count} text tokens have labels")
        else:
            print("  ✗ ERROR: No text tokens have labels!")
            success = False

    # Show sample of input_ids and labels
    print("\nFirst 20 tokens:")
    print("  input_ids:", input_ids[:20].tolist())
    print("  labels:   ", labels[:20].tolist())

    return success

if __name__ == "__main__":
    success = test_label_creation()
    if success:
        print("\n✓ Test PASSED! Labels are correctly masking vision tokens.")
        sys.exit(0)
    else:
        print("\n✗ Test FAILED!")
        sys.exit(1)
