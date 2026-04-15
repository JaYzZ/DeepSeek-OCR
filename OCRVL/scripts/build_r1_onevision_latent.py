#!/usr/bin/env python3
"""
R1-OneVision Latent Dataset Builder with Pre-extracted Cross-Attention Features

This script creates a training dataset with:
1. Main question images (saved to disk, using existing format)
2. Rendered CoT thinking images (saved to disk, using existing format)
3. Pre-extracted combined features using cross-attention (I + T → 100 tokens)
   - Saved as .latent.pt file adjacent to corresponding rendered image

Existing image format: {dataset_name}__{hash}.png
Latent features format: {dataset_name}__{hash}.latent.pt (same hash as the image)

Output JSONL format:
    {
        "question": "What is 2+2?",
        "answer": "4",
        "image": "OCRVL/data/r1_onevision_images/{dataset}__{hash}.png",
        "cot_images": ["OCRVL/data/r1_onevision_images/{dataset}__{hash_cot_0}.png", ...],
        "num_latent_steps": 3
    }

The latent supervision features are stored alongside the rendered CoT images
as .latent.pt files, to be loaded during training.

Usage:
    # Build all datasets
    python OCRVL/scripts/build_r1_onevision_latent.py --all

    # Build specific datasets
    python OCRVL/scripts/build_r1_onevision_latent.py --dataset ai2d doc_vqa

    # Quick test with limited samples
    python OCRVL/scripts/build_r1_onevision_latent.py --all --max-samples 100
"""

import argparse
import hashlib
import json
import base64
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import torch
import logging
from Renderer import VelloRenderer
from project_paths import get_deepseek_ocr_dir, hf_path

# Add parent directory to path for imports
script_dir = Path(__file__).parent.parent
repo_root = script_dir.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

VELLO_AVAILABLE = True


def get_hash_filename(dataset_name: str, sample_id: str, suffix: str) -> str:
    """
    Generate deterministic hash-based filename.

    Format: {dataset_name}__{hash}.{suffix}
    Matches existing format in r1_onevision_images/
    """
    # Use same hash format as existing: dataset_name__{hash}
    hash_input = f"{sample_id}_{suffix}".encode('utf-8')
    hash_hex = hashlib.sha256(hash_input).hexdigest()[:12]
    return f"{dataset_name}__{hash_hex}"


def decode_and_save_image(
    base64_string: str,
    output_dir: Path,
    dataset_name: str,
    sample_id: str,
    suffix: str,
) -> str:
    """
    Decode base64 image and save to disk with hash-based filename.

    Uses existing format: {dataset_name}__{hash}.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]

    image_data = base64.b64decode(base64_string)

    filename = get_hash_filename(dataset_name, sample_id, suffix)
    image_path = output_dir / f"{filename}.png"

    with open(image_path, 'wb') as f:
        f.write(image_data)

    return str(image_path)


def extract_thinking_and_answer(content: str) -> Tuple[str, str]:
    """Extract thinking section and final answer from R1-Onevision format."""
    content = content.lstrip('\n')
    parts = content.split('\n\n', 2)

    if len(parts) >= 3:
        thinking = parts[0].strip()
        answer = parts[2].strip()
    elif len(parts) == 2:
        thinking = parts[0].strip()
        answer = parts[1].strip()
    else:
        thinking = ""
        answer = content.strip()

    return thinking, answer


def chunk_thinking_text(
    thinking: str,
    max_chars_per_chunk: int = 800,
    min_chunks: int = 1,
) -> List[str]:
    """
    Split thinking text into chunks for rendering.

    Args:
        thinking: Raw thinking text
        max_chars_per_chunk: Maximum characters per chunk
        min_chunks: Minimum number of chunks (even for short text)

    Returns:
        List of text chunks
    """
    if not thinking or len(thinking.strip()) < 10:
        return []

    thinking = thinking.strip()

    # For short text, still create at least min_chunks
    if len(thinking) <= max_chars_per_chunk:
        # Split into sentences if possible
        sentences = [s.strip() for s in thinking.split('.') if s.strip()]
        if not sentences:
            return [thinking] if thinking else []

        # If few sentences, return as is
        if len(sentences) <= min_chunks:
            return [thinking] if thinking else []

        # Distribute sentences across min_chunks
        chunks = []
        sentences_per_chunk = max(1, len(sentences) // min_chunks)
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = '. '.join(sentences[i:i + sentences_per_chunk])
            if chunk:
                chunks.append(chunk + '.')
        return chunks

    # For long text, chunk by max_chars_per_chunk
    chunks = []
    current_chunk = ""
    words = thinking.split()

    for word in words:
        test_chunk = current_chunk + " " + word if current_chunk else word
        if len(test_chunk) <= max_chars_per_chunk:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def render_text_to_image(
    text: str,
    renderer: 'VelloRenderer',
    output_dir: Path,
    filename_no_ext: str,
) -> Optional[str]:
    """
    Render text as an image using VelloRenderer.

    Saves as: {filename_no_ext}.png
    """
    if not text or len(text.strip()) < 5:
        return None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        images = renderer.render_batch([text])

        if images and len(images) > 0:
            from PIL import Image
            import numpy as np

            img_array = images[0]
            if isinstance(img_array, np.ndarray):
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = img_array

            image_path = output_dir / f"{filename_no_ext}.png"
            pil_image.save(image_path)

            return str(image_path)

    except Exception as e:
        logger.warning(f"  Failed to render text: {e}")

    return None


def extract_and_save_latent_features(
    query_image_path: str,
    cot_image_paths: List[str],
    encoder,  # DPSKOCRCrossAttentionEncoder
    output_path: str,
) -> bool:
    """
    Extract combined latent features using cross-attention encoder and save to disk.

    Saves as: {output_path}.latent.pt

    Args:
        query_image_path: Path to main question image (I)
        cot_image_paths: List of paths to CoT rendered images (T)
        encoder: Cross-attention encoder
        output_path: Base path for saving features (without extension)

    Returns:
        True if successful, False otherwise
    """
    from PIL import Image

    try:
        # Load images
        query_image = Image.open(query_image_path).convert("RGB")
        cot_images = [Image.open(p).convert("RGB") for p in cot_image_paths]

        # Combine using cross-attention
        results = encoder.combine_images(
            query_images=[query_image],
            kv_images=cot_images,
            return_attention=False,
            return_components=False,
        )

        # Get combined features [100, 1280]
        combined_features = results[0]

        # Save to disk adjacent to the CoT image
        latent_path = Path(output_path + ".latent.pt")
        torch.save(combined_features.cpu(), latent_path)

        return True

    except Exception as e:
        logger.warning(f"  Failed to extract features: {e}")
        return False


def create_latent_training_sample(
    question_text: str,
    thinking: str,
    answer: str,
    main_image_path: str,
    cot_image_paths: List[str],
    num_latent_steps: int,
) -> dict:
    """
    Create a latent training sample.

    JSONL format:
    {
        "question": "What is 2+2?",
        "answer": "4",
        "image": "OCRVL/data/r1_onevision_images/{dataset}__{hash}.png",
        "cot_images": ["OCRVL/data/r1_onevision_images/{dataset}__{hash_cot_0}.png", ...],
        "num_latent_steps": 3
    }

    The latent supervision features are stored alongside each CoT image as .latent.pt files.
    """
    return {
        "question": question_text,
        "answer": answer,
        "image": main_image_path,
        "cot_images": cot_image_paths,
        "num_latent_steps": num_latent_steps,
    }


def process_parquet_file(
    parquet_path: Path,
    output_jsonl: Path,
    images_dir: Path,
    dataset_name: str,
    renderer: Optional['VelloRenderer'],
    encoder,  # DPSKOCRCrossAttentionEncoder
    max_chars_per_chunk: int = 800,
    max_samples: Optional[int] = None,
) -> dict:
    """Process a single parquet file with full preprocessing pipeline."""
    df = pd.read_parquet(parquet_path)

    if max_samples:
        df = df.head(max_samples)

    stats = {
        'total_samples': len(df),
        'with_thinking': 0,
        'rendered_cot': 0,
        'extracted_features': 0,
        'errors': 0
    }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, 'w') as f_out:
        for idx, row in df.iterrows():
            try:
                sample_id = row['id']

                # Save main image (V) - use existing format
                main_image_path = decode_and_save_image(
                    row['image'],
                    images_dir,
                    dataset_name,
                    sample_id,
                    "main",  # suffix for hash generation
                )

                # Parse conversations
                conversations = row['conversations']
                if len(conversations) < 2:
                    stats['errors'] += 1
                    continue

                question_text = conversations[0]['value']
                assistant_msg = conversations[1]['value']

                # Extract thinking and answer
                thinking, answer = extract_thinking_and_answer(assistant_msg)

                # Render question text to image (for latent encoding)
                question_image_path = None
                if renderer:
                    question_suffix = "question"
                    question_filename = get_hash_filename(dataset_name, sample_id, question_suffix)
                    question_image_path = render_text_to_image(
                        question_text,
                        renderer,
                        images_dir,
                        question_filename,
                    )

                # Render CoT text to images if available
                cot_image_paths = []
                if thinking and renderer:
                    stats['with_thinking'] += 1

                    # Chunk thinking for rendering
                    thinking_chunks = chunk_thinking_text(
                        thinking,
                        max_chars_per_chunk=max_chars_per_chunk,
                        min_chunks=1,
                    )

                    num_latent_steps = max(1, len(thinking_chunks))

                    for chunk_idx, chunk in enumerate(thinking_chunks):
                        # Use same hash format: dataset_name__{hash}
                        chunk_suffix = f"cot_{chunk_idx}"
                        filename_no_ext = get_hash_filename(dataset_name, sample_id, chunk_suffix)

                        cot_path = render_text_to_image(
                            chunk,
                            renderer,
                            images_dir,
                            filename_no_ext,
                        )
                        if cot_path:
                            cot_image_paths.append(cot_path)
                            stats['rendered_cot'] += 1

                # Extract latent features: I (original image) attends to [question_rendered, cot_rendered]
                # K=V = [question rendered image, all CoT rendered images]
                if cot_image_paths and encoder:
                    # Build KV images list: question + all CoT
                    kv_image_paths = []
                    if question_image_path:
                        kv_image_paths.append(question_image_path)
                    kv_image_paths.extend(cot_image_paths)

                    # Use the first CoT image's path (without extension) as base for features
                    # This way features are saved adjacent to the first CoT image
                    first_cot_path = Path(cot_image_paths[0]).stem  # filename without .png
                    latent_base_path = str(images_dir / first_cot_path)

                    if extract_and_save_latent_features(
                        query_image_path=main_image_path,
                        cot_image_paths=kv_image_paths,  # Now includes question + CoT
                        encoder=encoder,
                        output_path=latent_base_path,
                    ):
                        stats['extracted_features'] += 1

                # Create sample without thinking text (images are pre-rendered)
                num_latent_steps = len(cot_image_paths) if cot_image_paths else 0
                sample = create_latent_training_sample(
                    question_text=question_text,
                    thinking="",  # Not storing raw thinking, only rendered images
                    answer=answer,
                    main_image_path=main_image_path,
                    cot_image_paths=cot_image_paths,
                    num_latent_steps=num_latent_steps,
                )

                # Write to JSONL
                f_out.write(json.dumps(sample) + '\n')

                # Progress
                if (idx + 1) % 50 == 0:
                    logger.info(f"  Progress: {idx + 1}/{len(df)} samples, "
                              f"{stats['rendered_cot']} rendered, "
                              f"{stats['extracted_features']} features extracted")

            except Exception as e:
                stats['errors'] += 1
                logger.warning(f"  Error processing sample {idx}: {e}")

    logger.info(f"  Progress: {len(df)}/{len(df)} samples")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='R1-OneVision Latent Dataset Builder with Pre-extracted Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build all datasets
    python OCRVL/scripts/build_r1_onevision_latent.py --all

    # Build specific datasets
    python OCRVL/scripts/build_r1_onevision_latent.py --dataset ai2d doc_vqa

    # Quick test with limited samples
    python OCRVL/scripts/build_r1_onevision_latent.py --all --max-samples 100

Output Format:
    Each JSONL line contains:
    {
        "question": "What is 2+2?",
        "answer": "4",
        "image": "OCRVL/data/r1_onevision_images/{dataset}__{hash}.png",
        "cot_images": ["OCRVL/data/r1_onevision_images/{dataset}__{hash_cot_0}.png", ...],
        "num_latent_steps": 3
    }

    The latent supervision features are stored as:
        OCRVL/data/r1_onevision_images/{dataset}__{hash_cot_0}.latent.pt

    During training, load the .latent.pt file and use it as the reference for loss computation.
        """
    )

    parser.add_argument('--dataset', nargs='+', help='Dataset name(s) to convert')
    parser.add_argument('--all', action='store_true', help='Convert all datasets')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples per dataset (for testing)')
    parser.add_argument('--base-dir',
                        default=str(get_deepseek_ocr_dir()),
                        help='Base directory of OCRVL repository')
    parser.add_argument('--data-dir',
                        default=str(hf_path('Fancy-MLLM', 'R1-Onevision')),
                        help='Path to R1-Onevision data')
    parser.add_argument('--output-dir',
                        default='OCRVL/llamafactory/data',
                        help='Output directory for JSONL files')
    parser.add_argument('--images-dir',
                        default='OCRVL/data/r1_onevision_images',
                        help='Directory for images (uses existing format)')
    parser.add_argument('--model-path',
                        default='OCRVL/checkpoints/OCR-Qwen3-VL-2B',
                        help='Path to OCR-Qwen3-VL model (for cross-attention encoder)')
    parser.add_argument('--max-chars-per-chunk', type=int, default=800,
                        help='Maximum characters per CoT chunk')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess even if output exists')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for encoder (cuda:0, cuda:1, cpu)')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir

    logger.info("=" * 70)
    logger.info("R1-OneVision Latent Dataset Builder")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Data directory:   {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Max samples:      {args.max_samples or 'All'}")
    logger.info(f"Max chars/chunk:  {args.max_chars_per_chunk}")
    logger.info("")

    # Initialize VelloRenderer for CoT rendering
    renderer = None
    if VELLO_AVAILABLE:
        logger.info("Initializing VelloRenderer for CoT rendering...")
        try:
            renderer = VelloRenderer(
                width=640,
                height=640,
                padding=20,
                min_font_size=8.0,
                max_font_size=16.0,
                preserve_newlines=True
            )
            logger.info("✓ Renderer initialized")
        except Exception as e:
            logger.warning(f"✗ Failed to initialize VelloRenderer: {e}")
    else:
        logger.warning("VelloRenderer not available. CoT will not be rendered.")
        logger.warning("Install with: cd Renderer && maturin develop --release")

    # Initialize cross-attention encoder for feature extraction
    encoder = None
    model_path = base_dir / args.model_path
    if model_path.exists():
        logger.info("Initializing Cross-Attention Encoder...")
        try:
            from OCRInfer.encoder import DPSKOCRCrossAttentionEncoder

            device = args.device
            dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32

            encoder = DPSKOCRCrossAttentionEncoder(
                model_path=str(model_path),
                device=device,
                dtype=dtype,
            )
            logger.info(f"✓ Cross-Attention encoder initialized on {device}")
        except Exception as e:
            logger.warning(f"✗ Failed to initialize encoder: {e}")
            logger.warning("  CoT features will not be extracted")
    else:
        logger.warning(f"Model path not found: {model_path}")
        logger.warning("  Cross-attention encoder not initialized")

    if not renderer:
        logger.error("Renderer is not available. Exiting.")
        return 1

    logger.info("")

    # Determine datasets
    if args.all:
        datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    elif args.dataset:
        datasets = args.dataset
    else:
        parser.error("Either --dataset or --all must be specified")

    total_stats = {
        'total_samples': 0,
        'with_thinking': 0,
        'rendered_cot': 0,
        'extracted_features': 0,
        'errors': 0
    }

    for dataset_name in sorted(datasets):
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            logger.info(f"Skipping {dataset_name} (directory not found)")
            continue

        parquet_files = list(dataset_dir.glob('*.parquet'))
        if not parquet_files:
            logger.info(f"Skipping {dataset_name} (no parquet files)")
            continue

        parquet_path = parquet_files[0]
        output_jsonl = output_dir / f'r1_onevision_{dataset_name}_latent.jsonl'

        # Check if output exists
        if output_jsonl.exists() and output_jsonl.stat().st_size > 0 and not args.force:
            line_count = sum(1 for _ in open(output_jsonl))
            source_count = len(pd.read_parquet(parquet_path))
            if line_count >= source_count:
                logger.info(f"Skipping {dataset_name} (complete: {line_count} samples)")
                total_stats['total_samples'] += source_count
                total_stats['rendered_cot'] += line_count  # Approximate
                total_stats['extracted_features'] += line_count
                continue

        # Sanitize dataset name for filename prefix
        dataset_prefix = dataset_name.replace('(', '_').replace(')', '_').replace('-', '_')

        logger.info(f"Processing {dataset_name}...")
        stats = process_parquet_file(
            parquet_path=parquet_path,
            output_jsonl=output_jsonl,
            images_dir=images_dir,
            dataset_name=dataset_prefix,
            renderer=renderer,
            encoder=encoder,
            max_chars_per_chunk=args.max_chars_per_chunk,
            max_samples=args.max_samples,
        )

        logger.info(f"  Samples: {stats['total_samples']}, "
                   f"With thinking: {stats['with_thinking']}, "
                   f"Rendered CoT: {stats['rendered_cot']}, "
                   f"Features: {stats['extracted_features']}, "
                   f"Errors: {stats['errors']}")
        logger.info("")

        for key in total_stats:
            total_stats[key] += stats[key]

    # Cleanup
    if renderer:
        try:
            renderer.shutdown()
        except:
            pass

    logger.info("=" * 70)
    logger.info("Conversion Complete!")
    logger.info("=" * 70)
    logger.info(f"Total samples:       {total_stats['total_samples']}")
    logger.info(f"With thinking:       {total_stats['with_thinking']}")
    logger.info(f"Rendered CoT:        {total_stats['rendered_cot']}")
    logger.info(f"Extracted features:  {total_stats['extracted_features']}")
    logger.info(f"Errors:              {total_stats['errors']}")
    logger.info("")
    logger.info(f"JSONL files:          {output_dir}")
    logger.info(f"Images directory:    {images_dir}")
    logger.info("")
    logger.info("✓ Ready to train!")
    logger.info("")
    logger.info("Usage in training:")
    logger.info("  Load CoT images from cot_images field")
    logger.info("  Load latent supervision from adjacent .latent.pt files")
    logger.info("  Each .latent.pt file contains [100, 1280] tensor")
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
