#!/usr/bin/env python3
"""
Test BLIP3o dataset integration before starting training

Validates:
1. BLIP3o tar files are accessible
2. Images and captions load correctly
3. OCR adapter works with BLIP3o data
4. Dataset can create batches for training

Usage:
    python OCRVL/tests/test_blip3o_dataset.py
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

print("=" * 70)
print("Testing BLIP3o Dataset Integration")
print("=" * 70)

# Test 1: Import dataset
print("\n1. Importing BLIP3o dataset loader...")
try:
    from OCRVL.data.blip3o_dataset import BLIP3oAlignmentDataset
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load small sample without OCR adapter
print("\n2. Loading small sample (100 samples, no OCR)...")
try:
    dataset = BLIP3oAlignmentDataset(
        short_caption_path="/share/project/xiyan/huggingface/BLIP3o/BLIP3o-Pretrain-Short-Caption/00000.tar",
        long_caption_path="/share/project/xiyan/huggingface/BLIP3o/BLIP3o-Pretrain-Long-Caption/sa_000000.tar",
        tokenizer=None,
        ocr_adapter=None,
        mix_ratio=0.5,
        max_samples=100,
        use_images=False,  # Don't use images for quick test
    )
    print(f"✓ Loaded dataset with {len(dataset)} samples")

    # Check a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"  Sample keys: {sample.keys()}")
    else:
        print("✗ No samples loaded!")
        sys.exit(1)

except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load with tokenizer
print("\n3. Testing with tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded")
except Exception as e:
    print(f"✗ Tokenizer loading failed: {e}")
    sys.exit(1)

# Test 4: Load with OCR adapter (small sample)
print("\n4. Testing with OCR adapter (may take 1-2 min to load encoder)...")
try:
    from OCRVL.model.language_model.ocr_qwen3_vl import Qwen3VLOCRTextAdapter

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ocr_adapter = Qwen3VLOCRTextAdapter(
        encoder_model_path="deepseek-ai/DeepSeek-OCR",
        device=device,
        use_deepstack=True
    )
    print("✓ OCR adapter loaded")

    # Create dataset with OCR
    dataset_with_ocr = BLIP3oAlignmentDataset(
        short_caption_path="/share/project/xiyan/huggingface/BLIP3o/BLIP3o-Pretrain-Short-Caption/00000.tar",
        long_caption_path="/share/project/xiyan/huggingface/BLIP3o/BLIP3o-Pretrain-Long-Caption/sa_000000.tar",
        tokenizer=tokenizer,
        ocr_adapter=ocr_adapter,
        mix_ratio=0.5,
        max_samples=10,  # Just 10 samples for quick test
        use_images=True,
        image_to_caption_ratio=0.5,
    )
    print(f"✓ Dataset with OCR created ({len(dataset_with_ocr)} samples)")

except Exception as e:
    print(f"✗ OCR adapter test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Load a sample with features
print("\n5. Testing sample loading with features...")
try:
    sample = dataset_with_ocr[0]
    print(f"✓ Sample loaded")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")

    if sample['ocr_image_features'] is not None:
        if isinstance(sample['ocr_image_features'], tuple):
            final_feats, deepstack_feats = sample['ocr_image_features']
            print(f"  OCR features (final): {len(final_feats)} images")
            if len(final_feats) > 0:
                print(f"    Shape: {final_feats[0].shape}")
            print(f"  OCR features (deepstack): {len(deepstack_feats)} images")
            if len(deepstack_feats) > 0:
                print(f"    Levels: {len(deepstack_feats[0])}")
        else:
            print(f"  OCR features: {len(sample['ocr_image_features'])} images")
    else:
        print("  OCR features: None (text-only mode)")

except Exception as e:
    print(f"✗ Sample loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Create dataloader
print("\n6. Testing dataloader creation...")
try:
    from torch.utils.data import DataLoader
    from OCRVL.train import collate_fn

    dataloader = DataLoader(
        dataset_with_ocr,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))
    print(f"✓ Batch created")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")

except Exception as e:
    print(f"✗ Dataloader test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All tests passed!")
print("=" * 70)
print("\nYou can now start training with:")
print("  ./OCRVL/scripts/train_blip3o_alignment.sh")
print("\nOr run directly:")
print("  python OCRVL/train.py --stage alignment --use_blip3o")
print("=" * 70)
