#!/usr/bin/env python3
"""
End-to-end OCR test using server test images
"""

import requests
import json
import base64
import sys
from pathlib import Path
from difflib import SequenceMatcher

SERVER_URL = "http://localhost:8010"

# Map test images to their expected ground truth
TEST_CASES = [
    {
        "name": "Test 1 - Simple Text (result_with_boxes)",
        "image_path": "/tmp/ocr_test_proper_1/result_with_boxes.jpg",
        "gt_path": "/tmp/ocr_test_proper_1/result.mmd",
    },
    {
        "name": "Test 2 - Medium Text (result_with_boxes)",
        "image_path": "/tmp/ocr_test_proper_2/result_with_boxes.jpg",
        "gt_path": "/tmp/ocr_test_proper_2/result.mmd",
    },
    {
        "name": "Test 3 - Long Text (result_with_boxes)",
        "image_path": "/tmp/ocr_test_proper_3/result_with_boxes.jpg",
        "gt_path": "/tmp/ocr_test_proper_3/result.mmd",
    },
]


def load_ground_truth(gt_path: str):
    """Load ground truth text"""
    with open(gt_path, 'r') as f:
        return f.read().strip()


def ocr_image_file(image_path: str, server_url: str = SERVER_URL):
    """Perform OCR on image file"""
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    url = f"{server_url}/ocr"

    payload = {
        "image_base64": image_base64,
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "temperature": 0.0,
        "max_tokens": 8192,
        "ngram_size": 30,
        "window_size": 90,
        "output_format": "markdown",
    }

    print(f"  → Performing OCR...")
    response = requests.post(url, json=payload, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(f"OCR failed: {response.status_code} - {response.text}")

    result = response.json()

    if not result.get("success"):
        raise RuntimeError(f"OCR failed: {result.get('error')}")

    ocr_text = result.get("text", "")
    print(f"  ✓ OCR completed: {len(ocr_text)} chars")

    return ocr_text


def compare_texts(original: str, ocr_text: str):
    """Compare texts and return metrics"""
    orig_normalized = " ".join(original.split())
    ocr_normalized = " ".join(ocr_text.split())

    similarity = SequenceMatcher(None, orig_normalized, ocr_normalized).ratio()
    char_accuracy = similarity * 100

    orig_words = orig_normalized.split()
    ocr_words = ocr_normalized.split()
    word_similarity = SequenceMatcher(None, orig_words, ocr_words).ratio()
    word_accuracy = word_similarity * 100

    return {
        "char_accuracy": char_accuracy,
        "word_accuracy": word_accuracy,
        "original_chars": len(orig_normalized),
        "ocr_chars": len(ocr_normalized),
        "exact_match": orig_normalized == ocr_normalized
    }


def run_ocr_test(test_case: dict):
    """Run OCR test on single image"""
    print(f"\n{'='*80}")
    print(f"Test: {test_case['name']}")
    print(f"{'='*80}")
    print(f"Image: {test_case['image_path']}")
    print(f"GT: {test_case['gt_path']}")

    try:
        # Load ground truth
        gt_text = load_ground_truth(test_case['gt_path'])
        print(f"Ground Truth: {len(gt_text)} chars")
        print(f"Preview: {gt_text[:100]}...")

        # Perform OCR
        print(f"\nPerforming OCR on image...")
        ocr_text = ocr_image_file(test_case['image_path'])

        # Compare
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        print(f"\nOCR preview: {ocr_text[:100]}...")

        metrics = compare_texts(gt_text, ocr_text)

        print(f"\n📊 Accuracy:")
        print(f"  Character-level: {metrics['char_accuracy']:.2f}%")
        print(f"  Word-level:      {metrics['word_accuracy']:.2f}%")
        print(f"  Exact match:     {'✓ YES' if metrics['exact_match'] else '✗ NO'}")

        # Show differences if not exact match
        if not metrics['exact_match']:
            print(f"\n  GT length:  {metrics['original_chars']} chars")
            print(f"  OCR length: {metrics['ocr_chars']} chars")

        return {
            "name": test_case['name'],
            "success": True,
            "metrics": metrics,
            "ground_truth": gt_text,
            "ocr_text": ocr_text
        }

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "name": test_case['name'],
            "success": False,
            "error": str(e)
        }


def main():
    print("\n" + "="*80)
    print("DeepSeek-OCR End-to-End Test")
    print("="*80)
    print(f"Server: {SERVER_URL}")
    print(f"Tests: {len(TEST_CASES)}")

    # Check server
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        health = response.json()
        print(f"✓ Server: {health['status']}, Model: {health['model_loaded']}")
    except Exception as e:
        print(f"❌ Server check failed: {e}")
        sys.exit(1)

    # Run tests
    results = []
    for test_case in TEST_CASES:
        result = run_ocr_test(test_case)
        results.append(result)

    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\nTotal: {len(results)}")
    print(f"Passed: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_char = sum(r['metrics']['char_accuracy'] for r in successful) / len(successful)
        avg_word = sum(r['metrics']['word_accuracy'] for r in successful) / len(successful)
        exact = sum(1 for r in successful if r['metrics']['exact_match'])

        print(f"\n📊 Overall Accuracy:")
        print(f"  Character-level: {avg_char:.2f}%")
        print(f"  Word-level:      {avg_word:.2f}%")
        print(f"  Exact matches:   {exact}/{len(successful)}")

    if failed:
        print(f"\n❌ Failed:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Unknown')}")

    # Save detailed results
    output_file = "/tmp/ocr_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
