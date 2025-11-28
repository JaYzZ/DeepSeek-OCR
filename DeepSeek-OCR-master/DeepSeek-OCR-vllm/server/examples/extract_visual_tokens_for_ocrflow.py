"""
Example: Extract pure visual tokens for OCRFlow from DeepSeek-OCR API

Shows how to get the 100 content-dependent visual tokens from the 111-token output.
"""

import requests
import numpy as np
from utils.visual_token_utils import extract_pure_visual_tokens, reshape_to_spatial_grid


def example_extract_from_api():
    """Example: Get 100 pure visual tokens from API response"""

    # Step 1: Call /text-to-vistok endpoint
    with open('test_images/test_render_large_font.png', 'rb') as f:
        response = requests.post(
            'http://localhost:8003/text-to-vistok',
            files={'file': f},
            data={'output_format': 'binary'}
        )

    print(f"Response size: {len(response.content)} bytes")
    # Expected: 568,320 bytes = 111 tokens × 1280 dims × 4 bytes/float32

    # Step 2: Parse binary response
    vistok_111 = np.frombuffer(response.content, dtype=np.float32)
    vistok_111 = vistok_111.reshape(-1, 111, 1280)
    print(f"Full output shape: {vistok_111.shape}")  # (1, 111, 1280)

    # Step 3: Extract 100 pure visual tokens
    vistok_100 = extract_pure_visual_tokens(vistok_111)
    print(f"Pure visual tokens shape: {vistok_100.shape}")  # (1, 100, 1280)

    # Step 4 (Optional): Reshape to spatial grid
    spatial_grid = reshape_to_spatial_grid(vistok_100)
    print(f"Spatial grid shape: {spatial_grid.shape}")  # (1, 10, 10, 1280)

    # Now you have:
    # - vistok_100[0]: 100 content-dependent visual features [100, 1280]
    # - spatial_grid[0]: Same features as 10×10 grid [10, 10, 1280]

    print("\n✓ Successfully extracted pure visual tokens for OCRFlow!")
    return vistok_100, spatial_grid


def example_batch_processing():
    """Example: Process multiple images in batch"""

    image_paths = [
        'test_images/test_render_large_font.png',
        'test_images/test_doc_font32.png',
    ]

    # Collect all visual tokens
    all_vistok_111 = []

    for img_path in image_paths:
        with open(img_path, 'rb') as f:
            response = requests.post(
                'http://localhost:8003/text-to-vistok',
                files={'file': f},
                data={'output_format': 'binary'}
            )

        vistok = np.frombuffer(response.content, dtype=np.float32)
        vistok = vistok.reshape(1, 111, 1280)
        all_vistok_111.append(vistok)

    # Stack into batch
    batch_vistok_111 = np.concatenate(all_vistok_111, axis=0)
    print(f"Batch shape (111 tokens): {batch_vistok_111.shape}")  # (2, 111, 1280)

    # Extract pure visual tokens for all images at once
    batch_vistok_100 = extract_pure_visual_tokens(batch_vistok_111)
    print(f"Batch shape (100 tokens): {batch_vistok_100.shape}")  # (2, 100, 1280)

    print("\n✓ Successfully processed batch!")
    return batch_vistok_100


def verify_extraction():
    """Verify that extraction removes exactly the right indices"""

    # Create test data with known pattern
    test_111 = np.arange(111 * 3).reshape(111, 3)  # [111, 3] for easy inspection

    # Extract pure visual tokens
    test_100 = extract_pure_visual_tokens(test_111)

    print("Original shape:", test_111.shape)  # (111, 3)
    print("Extracted shape:", test_100.shape)  # (100, 3)

    # Check which indices were kept
    kept_indices = []
    for row in range(10):
        start = row * 11
        kept_indices.extend(range(start, start + 10))

    # Verify extraction is correct
    expected = test_111[kept_indices]
    assert np.array_equal(test_100, expected), "Extraction mismatch!"

    print(f"\nKept indices: {kept_indices[:20]}... (first 20)")
    print(f"Removed indices: 10, 21, 32, 43, 54, 65, 76, 87, 98, 109, 110")
    print("\n✓ Extraction verified correct!")


if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek-OCR: Extract Pure Visual Tokens for OCRFlow")
    print("=" * 60)

    print("\n1. Verification Test")
    print("-" * 60)
    verify_extraction()

    print("\n2. Single Image Example")
    print("-" * 60)
    try:
        vistok_100, spatial = example_extract_from_api()
        print(f"\nSaved visual tokens shape: {vistok_100.shape}")
        print(f"  - For OCRFlow training: Use vistok_100 [batch, 100, 1280]")
        print(f"  - For spatial operations: Use spatial [batch, 10, 10, 1280]")
    except Exception as e:
        print(f"Note: Server not running or test image not found: {e}")

    print("\n3. Batch Processing Example")
    print("-" * 60)
    try:
        batch_vistok = example_batch_processing()
        print(f"\nBatch visual tokens ready for OCRFlow training!")
    except Exception as e:
        print(f"Note: Server not running or test images not found: {e}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("  • API returns: [batch, 111, 1280] (with structural tokens)")
    print("  • Utils extract: [batch, 100, 1280] (pure visual features)")
    print("  • OCRFlow uses: The 100 content-dependent features")
    print("=" * 60)
