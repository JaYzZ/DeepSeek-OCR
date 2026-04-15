"""
Pre-compute Visual Tokens for Fast Training

This script pre-computes and caches visual tokens from text data,
eliminating the CPU bottleneck during training.

Architecture:
1. Read text from FineWeb/OpenWebMath parquet files
2. Chunk text into variable-length pieces
3. Render text to images and encode to visual tokens
4. Save visual tokens as .pt files with metadata

Output structure:
    cache_dir/
        chunks/
            00000000.pt  # Contains: {'tokens': [111, 1280], 'text_hash': str, 'word_count': int}
            00000001.pt
            ...
        metadata.json   # Contains: total_chunks, config, etc.
        pairs.json      # Contains: list of (chunk_i, chunk_i+1) pair indices

Usage:
    python OCRFlow/scripts/precompute_vistok.py \
        --dataset_type fineweb \
        --fineweb_subset 10BT \
        --output_dir ./cache/vistok_fineweb_10bt \
        --max_chunks 1000000 \
        --num_render_workers 32
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import hashlib
import random
from typing import List, Dict, Iterator, Optional

import torch
import pandas as pd
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from project_paths import hf_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_words(text: str) -> int:
    return len(text.split())


def chunk_text_by_words(
    text: str,
    target_words: int = 500,
    min_words: int = 10,
    max_words: int = 900,
    variance: float = 0.5,
    mixed_length: bool = True,
) -> List[str]:
    """Chunk text into variable-length pieces"""
    words = text.split()
    total_words = len(words)

    if total_words < min_words * 2:
        return []

    chunks = []
    current_pos = 0

    # Mixed-length distributions
    length_distributions = [
        (30, 0.5, 0.2),   # Short
        (100, 0.4, 0.3),  # Medium
        (400, 0.5, 0.5),  # Long
    ] if mixed_length else [(target_words, variance, 1.0)]

    while current_pos < total_words:
        if mixed_length:
            r = random.random()
            cumulative = 0
            for t_words, t_var, prob in length_distributions:
                cumulative += prob
                if r < cumulative:
                    chunk_size = int(t_words * random.uniform(1 - t_var, 1 + t_var))
                    break
        else:
            chunk_size = int(target_words * random.uniform(1 - variance, 1 + variance))

        chunk_size = max(min_words, min(max_words, chunk_size))

        remaining = total_words - current_pos
        if remaining < min_words:
            break

        if remaining - chunk_size < min_words and remaining <= max_words:
            chunk_size = remaining

        end_pos = min(current_pos + chunk_size, total_words)
        chunk_words = words[current_pos:end_pos]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        current_pos = end_pos

    if len(chunks) < 2:
        return []

    return chunks


def text_hash(text: str) -> str:
    return hashlib.md5(text[:200].encode()).hexdigest()[:16]


def iter_documents(parquet_files: List[Path], min_doc_words: int = 100) -> Iterator[str]:
    """Iterate over documents from parquet files"""
    for parquet_path in parquet_files:
        try:
            df = pd.read_parquet(parquet_path)
            if "text" not in df.columns:
                continue

            for _, row in df.iterrows():
                text = row["text"]
                if count_words(text) >= min_doc_words:
                    yield text
        except Exception as e:
            logger.warning(f"Error reading {parquet_path}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Pre-compute visual tokens")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="fineweb",
                       choices=["fineweb", "openwebmath", "multi"])
    parser.add_argument("--fineweb_path", type=str,
                       default=str(hf_path("HuggingFaceFW", "fineweb-edu")))
    parser.add_argument("--fineweb_subset", type=str, default="10BT")
    parser.add_argument("--openwebmath_path", type=str,
                       default=str(hf_path("open-web-math", "open-web-math")))

    # Chunking
    parser.add_argument("--target_words", type=int, default=500)
    parser.add_argument("--min_words", type=int, default=10)
    parser.add_argument("--max_words", type=int, default=900)
    parser.add_argument("--variance", type=float, default=0.5)
    parser.add_argument("--mixed_length", action="store_true", default=True)
    parser.add_argument("--min_doc_words", type=int, default=100)

    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_chunks", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=64)

    # Encoder
    parser.add_argument("--encoder_model", type=str, default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_render_workers", type=int, default=32)

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Find parquet files
    if args.dataset_type == "fineweb":
        if args.fineweb_subset:
            data_path = Path(args.fineweb_path) / "sample" / args.fineweb_subset
        else:
            data_path = Path(args.fineweb_path) / "data"
        parquet_files = sorted(data_path.glob("**/*.parquet"))
    elif args.dataset_type == "openwebmath":
        data_path = Path(args.openwebmath_path)
        if (data_path / "data").exists():
            data_path = data_path / "data"
        parquet_files = sorted(data_path.glob("**/*.parquet"))
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    logger.info(f"Found {len(parquet_files)} parquet files")

    # Initialize encoder + renderer directly.
    logger.info("Initializing DPSK OCR encoder...")
    from PIL import Image
    from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
    from Renderer.pil_renderer import PILRenderer, render_to_pil
    try:
        from Renderer import VelloRenderer  # type: ignore
    except Exception:
        VelloRenderer = None

    encoder = DPSKOCREncoder(
        model_path=args.encoder_model,
        device=args.device,
        dtype=torch.bfloat16,
    )

    vello = None
    if VelloRenderer is not None:
        try:
            vello = VelloRenderer(width=640, height=640, padding=20)
            logger.info("Using VelloRenderer for precompute")
        except Exception:
            vello = None
    pil_renderer = None if vello is not None else PILRenderer(
        width=640, height=640, num_workers=args.num_render_workers
    )

    def render_texts(texts: List[str]) -> List[Image.Image]:
        if vello is not None:
            arrays = vello.render_batch(list(texts))
            return [Image.fromarray(arr) for arr in arrays]
        if pil_renderer is not None:
            return pil_renderer.render_batch_pil(list(texts))
        return [render_to_pil(t, width=640, height=640) for t in texts]

    # Process documents
    logger.info("Processing documents...")

    chunk_metadata = []
    all_pairs = []  # (input_idx, target_idx) pairs for Markovian training
    chunk_idx = 0
    pending_chunks = []
    pending_start_idx = 0
    doc_chunk_indices = []  # Track chunk indices per document

    pbar = tqdm(total=args.max_chunks, desc="Processing chunks")

    for doc_text in iter_documents(parquet_files, args.min_doc_words):
        if chunk_idx >= args.max_chunks:
            break

        # Chunk document
        doc_chunks = chunk_text_by_words(
            doc_text,
            target_words=args.target_words,
            min_words=args.min_words,
            max_words=args.max_words,
            variance=args.variance,
            mixed_length=args.mixed_length,
        )

        if len(doc_chunks) < 2:
            continue

        # Track document's chunk indices for pair generation
        doc_start_idx = chunk_idx

        # Add chunks to pending batch
        for chunk in doc_chunks:
            if chunk_idx >= args.max_chunks:
                break
            pending_chunks.append(chunk)
            chunk_idx += 1

            # Process batch when full
            if len(pending_chunks) >= args.batch_size:
                # Encode batch
                try:
                    images = render_texts([t[:6000] for t in pending_chunks])
                    tokens_list = encoder.encode_images(images, return_global=False, return_local=True)

                    # Save each chunk
                    for i, (chunk_text, tokens) in enumerate(zip(pending_chunks, tokens_list)):
                        if tokens is None:
                            continue

                        idx = pending_start_idx + i
                        output_path = chunks_dir / f"{idx:08d}.pt"

                        data = {
                            'tokens': tokens.cpu().half(),
                            'text_hash': text_hash(chunk_text),
                            'word_count': count_words(chunk_text),
                        }

                        torch.save(data, output_path)
                        chunk_metadata.append({
                            'idx': idx,
                            'hash': data['text_hash'],
                            'words': data['word_count'],
                        })

                except Exception as e:
                    logger.warning(f"Encoding error: {e}")

                pbar.update(len(pending_chunks))
                pending_start_idx = chunk_idx
                pending_chunks = []

        # Generate pairs for this document (consecutive chunks)
        doc_end_idx = chunk_idx
        for i in range(doc_start_idx, doc_end_idx - 1):
            all_pairs.append((i, i + 1))

    # Process remaining chunks
    if pending_chunks:
        try:
            images = render_texts([t[:6000] for t in pending_chunks])
            tokens_list = encoder.encode_images(images, return_global=False, return_local=True)

            for i, (chunk_text, tokens) in enumerate(zip(pending_chunks, tokens_list)):
                if tokens is None:
                    continue

                idx = pending_start_idx + i
                output_path = chunks_dir / f"{idx:08d}.pt"

                data = {
                    'tokens': tokens.cpu().half(),
                    'text_hash': text_hash(chunk_text),
                    'word_count': count_words(chunk_text),
                }

                torch.save(data, output_path)
                chunk_metadata.append({
                    'idx': idx,
                    'hash': data['text_hash'],
                    'words': data['word_count'],
                })

        except Exception as e:
            logger.warning(f"Encoding error: {e}")

        pbar.update(len(pending_chunks))

    pbar.close()

    # Save metadata
    metadata = {
        'total_chunks': len(chunk_metadata),
        'total_pairs': len(all_pairs),
        'config': vars(args),
        'created_at': datetime.now().isoformat(),
        'chunks': chunk_metadata,
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "pairs.json", 'w') as f:
        json.dump(all_pairs, f)

    logger.info(f"Done! Saved {len(chunk_metadata)} chunks and {len(all_pairs)} pairs to {output_dir}")
    logger.info(f"  Chunks: {chunks_dir}")
    logger.info(f"  Metadata: {output_dir / 'metadata.json'}")
    logger.info(f"  Pairs: {output_dir / 'pairs.json'}")


if __name__ == "__main__":
    main()
