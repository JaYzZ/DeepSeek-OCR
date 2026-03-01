#!/usr/bin/env python3
"""
Build R1-OneVision Thinking Dataset for Qwen3VL Native Training

Uses Qwen3VL Native Vision Encoder for end-to-end feature extraction:
1. Question image → Vision Encoder (ViT + Projector) → LLM space (2048-dim)
2. Thinking text → Adaptive renderer → Images → Vision Encoder → LLM space
3. Latent format:
   - latent_ground_truth: Thinking image features (for injection at <latent>)
   - latent_supervision: Original image features (for OT loss reference)

All features are cached in .feature_cache directory and reused across runs.
Features are already in LLM hidden dimension space - ready for direct injection and OT loss.
"""

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from PIL import Image

# Setup path for local imports
script_dir = Path(__file__).parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(repo_root))

# Local imports
from Qwen.scripts.adaptive_vello_renderer import AdaptiveVelloRenderer
from OCRVL.encoder.qwen3vl_encoder import Qwen3VLEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_image_fast(path: Path) -> Optional[Image.Image]:
    """Load a single image with error handling using turbojpeg for speed."""
    try:
        # Try turbojpeg first (much faster for JPEG)
        import turbojpeg
        jpeg = turbojpeg.TurboJPEG()

        with open(path, 'rb') as f:
            img_array = jpeg.decode(f.read())

        # Convert numpy array to PIL Image
        return Image.fromarray(img_array)
    except Exception:
        # Fall back to PIL for non-JPEG or errors
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logging.warning(f"Error loading {path}: {e}")
            return None


def load_images_threaded(paths: List[Path], max_workers: int = 8) -> List[Optional[Image.Image]]:
    """Load images in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(load_image_fast, paths))


def compress_newlines(text: str) -> str:
    """Compress multiple consecutive newlines to a single newline."""
    # Replace 2+ consecutive newlines with a single newline
    return re.sub(r'\n{2,}', '\n', text)


def format_cot_subsequences(thinking_chunks: Optional[List[str]]) -> str:
    """
    Format COT text to match CE-loss "subsequence" composition.

    Expected shape:
      <think>[subseq1]<think_sep>[subseq2]<think_sep>...</think>
    """
    chunks = [c.strip() for c in (thinking_chunks or []) if c and c.strip()]
    if not chunks:
        return ""
    return f"<think>{'<think_sep>'.join(chunks)}</think>"


def chunk_thinking_text(
    thinking: str,
    max_chars: int,
) -> List[str]:
    """
    Chunk thinking text intelligently into balanced chunks.
    
    Rules:
    - Chunk at sentence boundaries (don't break mid-sentence)
    - Multiple newlines are good chunking positions
    - Chunks should be evenly sized (within 20% variance)
    """
    # First compress newlines
    thinking = compress_newlines(thinking)
    
    # If fits in one chunk, return as-is
    if len(thinking) <= max_chars:
        return [thinking]
    
    # Calculate how many chunks we need
    num_chunks = (len(thinking) + max_chars - 1) // max_chars
    
    # Split into sentence fragments (keeping delimiters)
    # Split on: '. ' (sentence end), '\n' (newlines which were already compressed)
    fragments = re.split(r'(\. \n|\. |\n)', thinking)
    
    # Reconstruct fragments with their delimiters
    sentences = []
    for i in range(0, len(fragments) - 1, 2):
        if i + 1 < len(fragments):
            sentences.append(fragments[i] + fragments[i + 1])
        else:
            sentences.append(fragments[i])
    if fragments and fragments[-1]:
        sentences.append(fragments[-1])
    
    # Filter empty sentences
    sentences = [s for s in sentences if s.strip()]
    
    # Now distribute sentences into chunks
    target_size = len(thinking) / num_chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + sentence
        
        # If adding this sentence stays within 20% of target, add it
        if len(test_chunk) <= target_size * 1.2 or not current_chunk:
            current_chunk = test_chunk
        else:
            # Start new chunk
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Post-processing: ensure no chunk exceeds max_chars
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            # Split this chunk further at sentence boundary
            split_point = chunk.rfind('. ', 0, max_chars)
            if split_point == -1:
                split_point = max_chars
            
            final_chunks.append(chunk[:split_point + 1])
            chunk = chunk[split_point + 1:].strip()
        
        if chunk:
            final_chunks.append(chunk)
    
    return final_chunks


def extract_thinking_and_answer(
    content: str,
    max_chars: Optional[int] = 4800,
    return_chunks: bool = True,
) -> Tuple[str, str] | Tuple[List[str], str]:
    """
    Extract thinking section and final answer from R1-Onevision format.

    Format: <think> thinking content </think> final answer

    Args:
        content: Assistant message containing thinking tags
        max_chars: Maximum characters per chunk (default: 4800)
        return_chunks: If True (default), return list of chunks.
                      If False, return only first chunk for backward compatibility.

    Returns:
        If return_chunks=True: (list of thinking chunks, answer)
        If return_chunks=False: (first thinking chunk or full thinking, answer)
    """
    # Find opening and closing tags
    open_tag = '<think>'
    close_tag = '</think>'
    
    open_pos = content.find(open_tag)
    if open_pos < 0:
        return "", content.strip()
    
    close_pos = content.find(close_tag, open_pos + len(open_tag))
    if close_pos < 0:
        return "", content.strip()
    
    # Extract content between tags (thinking to be rendered)
    thinking_start = open_pos + len(open_tag)
    thinking_raw = content[thinking_start:close_pos].strip()

    # Answer is everything after closing tag
    answer_start = close_pos + len(close_tag)
    answer_raw = content[answer_start:].strip()

    # If no answer, thinking is the answer
    if not answer_raw:
        answer = thinking_raw
    else:
        answer = answer_raw

    # Potential risk that <image> and </image> tags in the answer, remove them
    answer = re.sub(r'<\/?image>', '', answer).strip()

    # Process thinking based on max_chars
    if max_chars is not None and len(thinking_raw) > max_chars:
        # Use intelligent chunking
        thinking_chunks = chunk_thinking_text(thinking_raw, max_chars)

        if return_chunks:
            # Return all chunks as a list
            return thinking_chunks, answer
        else:
            # Return only first chunk (backward compatibility)
            thinking = thinking_chunks[0]
    else:
        # Compress newlines even if no chunking needed
        thinking = compress_newlines(thinking_raw)
        if return_chunks:
            return [thinking], answer

    return thinking, answer


def get_hash_filename(dataset_name: str, sample_id: str, suffix: str = "") -> str:
    """Generate deterministic filename for images and cache.

    Format: {sanitized_dataset}_{sample_id}[_{suffix}]

    Uses sample_id directly (not hashed) for traceability.
    Single underscore between all components.

    Examples:
        get_hash_filename("Geometry3K(MathV360K)", "abc123", "")
            -> "Geometry3K_MathV360K_abc123"
        get_hash_filename("Geometry3K(MathV360K)", "abc123", "question_text")
            -> "Geometry3K_MathV360K_abc123_question_text"
        get_hash_filename("Geometry3K(MathV360K)", "abc123", "thinking_0")
            -> "Geometry3K_MathV360K_abc123_thinking_0"

    Args:
        dataset_name: Dataset name (e.g., "Geometry3K(MathV360K)")
        sample_id: Sample ID (used directly, not hashed)
        suffix: Optional suffix for original, question_text, or thinking chunks

    Returns:
        Filename without extension
    """
    # Replace parentheses to create consistent naming
    sanitized = dataset_name.replace('(', '_').replace(')', '')

    # Single underscore between all components
    if suffix:
        return f"{sanitized}_{sample_id}_{suffix}"
    else:
        return f"{sanitized}_{sample_id}"


def extract_and_save_latent_features(
    query_image_path: str,
    thinking_image_paths: List[str],
    encoder: Qwen3VLEncoder,
    output_path: str,
    skip_existing: bool = True,
) -> bool:
    """
    Extract combined latent features using Qwen3VL cross-attention encoder.

    Saves as: {output_path}.latent.pt
    Output shape: [seq, hidden_dim] where seq depends on query image size

    Args:
        query_image_path: Path to main question image (Q)
        thinking_image_paths: List of paths to thinking rendered images (K=V)
        encoder: Qwen3VL cross-attention encoder
        output_path: Base path for saving features (without extension)
        skip_existing: Skip if latent file already exists (default: True)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if latent already exists
        latent_path = Path(output_path + ".latent.pt")
        if skip_existing and latent_path.exists():
            logger.info(f"  ✓ Skipping existing latent: {latent_path.name}")
            return True

        # Load images
        query_image = Image.open(query_image_path).convert("RGB")
        thinking_images = [Image.open(p).convert("RGB") for p in thinking_image_paths]

        # Combine using cross-attention
        results = encoder.combine_images(
            query_images=[query_image],
            thinking_images=thinking_images,
        )

        # Get combined features [seq, hidden_dim] where seq depends on query image size
        combined_features = results[0]

        # Save to disk
        latent_path = Path(output_path + ".latent.pt")
        torch.save(combined_features.cpu(), latent_path)

        return True

    except Exception as e:
        logger.warning(f"  Failed to extract features: {e}")
        return False


def process_parquet_file(
    parquet_path: Path,
    output_jsonl: Path,
    images_dir: Path,
    dataset_name: str,
    renderer: AdaptiveVelloRenderer,
    encoder: Qwen3VLEncoder,
    max_chars_per_chunk: int = 4800,
    max_samples: Optional[int] = None,
    file_mode: str = 'w',
) -> dict:
    """Process a single parquet file with full pipeline."""
    df = pd.read_parquet(parquet_path)

    if max_samples:
        df = df.head(max_samples)

    stats = {
        'total_samples': len(df),
        'with_thinking': 0,
        'rendered_thinking': 0,
        'extracted_features': 0,
        'reused_images': 0,
        'errors': 0
    }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, file_mode) as f_out:
        for idx, row in df.iterrows():
            try:
                sample_id = row['id']

                # Progress logging every 20 samples
                if stats['total_samples'] % 20 == 0 and stats['total_samples'] > 0:
                    logger.info(f"  Progress: {stats['total_samples']}/{len(df)} samples, "
                              f"{stats['extracted_features']} features extracted")

                # Check if main image exists in OCRVL cache (reuse if available)
                filename = get_hash_filename(dataset_name, sample_id, "main")
                ocrvl_image_path = repo_root / "OCRVL/data/r1_onevision_images" / f"{filename}.png"

                if ocrvl_image_path.exists():
                    # Reuse existing question image from OCRVL
                    image_path = ocrvl_image_path
                    stats['reused_images'] += 1
                else:
                    # Decode and save new image
                    if isinstance(row['image'], str) and ',' in row['image']:
                        image_data = row['image'].split(',', 1)[1]
                    else:
                        image_data = row['image']

                    image_bytes = base64.b64decode(image_data)
                    image_path = images_dir / f"{filename}.png"

                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)

                # Parse conversations
                conversations = row['conversations']
                if len(conversations) < 2:
                    stats['errors'] += 1
                    continue

                question_text = conversations[0]['value']
                assistant_msg = conversations[1]['value']

                # Extract thinking chunks and answer in one step (handles chunking internally)
                thinking_chunks, answer = extract_thinking_and_answer(
                    assistant_msg,
                    max_chars=max_chars_per_chunk,
                    return_chunks=True,
                )

                # Render thinking text to adaptive-sized images
                thinking_image_paths = []
                if thinking_chunks:
                    stats['with_thinking'] += 1

                    for chunk_idx, chunk in enumerate(thinking_chunks):
                        chunk_suffix = f"thinking_{chunk_idx}"
                        filename_no_ext = get_hash_filename(dataset_name, sample_id, chunk_suffix)
                        chunk_path = images_dir / f"{filename_no_ext}.png"

                        # Render with adaptive sizing
                        if renderer.render(chunk, str(chunk_path)):
                            thinking_image_paths.append(str(chunk_path))
                            stats['rendered_thinking'] += 1

                # Extract latent features using cross-attention
                latent_supervision_paths = []
                if thinking_image_paths:
                    # Use first thinking image's path as base for latent features
                    first_thinking_path = Path(thinking_image_paths[0]).stem
                    latent_base_path = str(images_dir / first_thinking_path)

                    if extract_and_save_latent_features(
                        query_image_path=str(image_path),
                        thinking_image_paths=thinking_image_paths,
                        encoder=encoder,
                        output_path=latent_base_path,
                    ):
                        stats['extracted_features'] += 1
                        # Store the latent file path (without .png, will be used with .latent.pt during training)
                        latent_supervision_paths.append(latent_base_path + ".latent.pt")

                # Build messages (no thinking image in messages, only latent supervision)
                # Adaptively add <image> token only if question_text doesn't already have one
                if '<image>' in question_text:
                    # Question already has <image> token, use as-is
                    user_content = question_text
                else:
                    # Question doesn't have <image> token, add one
                    user_content = f"<image>\n{question_text}"
                messages = [
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]

                # Create sample
                sample = {
                    "messages": messages,
                    "images": [str(image_path)],
                    "latent_supervision": latent_supervision_paths,
                    "num_latent_steps": len(latent_supervision_paths),
                    "task": "r1_onevision_thinking"
                }

                # Write to JSONL
                f_out.write(json.dumps(sample) + '\n')

                # Progress
                if (idx + 1) % 50 == 0:
                    logger.info(f"  Progress: {idx + 1}/{len(df)} samples, "
                              f"{stats['rendered_thinking']} rendered, "
                              f"{stats['extracted_features']} features extracted")

            except Exception as e:
                stats['errors'] += 1
                import traceback
                logger.warning(f"  Error processing sample {sample_id}: {e}\n{traceback.format_exc()}")

    logger.info(f"  Progress: {len(df)}/{len(df)} samples")
    return stats


def main_render_only(args, renderer):
    """Phase 1: Pre-render all question and thinking images to disk."""
    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Determine datasets
    if args.all:
        datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    elif args.dataset:
        datasets = args.dataset
    else:
        raise ValueError("Either --dataset or --all must be specified")

    logger.info(f"Processing {len(datasets)} datasets")
    logger.info("")

    total_stats = {
        'total_samples': 0,
        'question_images': 0,
        'question_text_images': 0,
        'thinking_chunks': 0,
        'thinking_images': 0,
        'errors': 0,
    }

    for dataset_name in datasets:
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"Skipping {dataset_name} (directory not found)")
            continue

        parquet_files = sorted(dataset_dir.glob('*.parquet'))
        if not parquet_files:
            logger.warning(f"Skipping {dataset_name} (no parquet files)")
            continue
        logger.info(f"Processing {dataset_name} ({len(parquet_files)} shards)...")

        stats = {'total_samples': 0, 'question_images': 0, 'question_text_images': 0, 'thinking_chunks': 0, 'thinking_images': 0, 'errors': 0}

        for shard_idx, parquet_path in enumerate(parquet_files):
            logger.info(f"  Shard {shard_idx + 1}/{len(parquet_files)}: {parquet_path.name}")

            df = pd.read_parquet(parquet_path)
            if args.max_samples:
                df = df.head(args.max_samples)

            for idx, row in df.iterrows():
                try:
                    sample_id = row['id']
                    stats['total_samples'] += 1

                    # Parse conversations
                    conversations = row.get('conversations', [])
                    if hasattr(conversations, 'tolist'):
                        conversations = conversations.tolist()
                    elif not isinstance(conversations, list):
                        conversations = list(conversations) if conversations is not None else []

                    if len(conversations) < 2:
                        continue

                    # Extract question text and thinking
                    question_text = conversations[0]['value']
                    assistant_msg = conversations[1]['value']
                    thinking_chunks, answer = extract_thinking_and_answer(
                        assistant_msg,
                        max_chars=args.max_chars_per_chunk,
                        return_chunks=True,
                    )

                    # 1. Save visual question image
                    filename = get_hash_filename(dataset_name, sample_id, "main")
                    image_path = images_dir / f"{filename}.png"

                    if not image_path.exists():
                        image_col = row.get('image')
                        if isinstance(image_col, str):
                            # Handle both data URL format and raw base64
                            if ',' in image_col:
                                # data:image/png;base64,XXXX format
                                _, imgstr = image_col.split(',', 1)
                                image_data = base64.b64decode(imgstr)
                            else:
                                # Raw base64 format
                                image_data = base64.b64decode(image_col)
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(image_path, 'wb') as f:
                                f.write(image_data)
                            stats['question_images'] += 1

                    # 2. Render question text
                    question_text_filename = get_hash_filename(dataset_name, sample_id, "question_text")
                    question_text_path = images_dir / f"{question_text_filename}.png"

                    if not question_text_path.exists():
                        logger.info(f"    Rendering question text: {len(question_text)} chars -> {question_text_path.name}")
                        success = renderer.render(question_text, str(question_text_path))
                        logger.info(f"      Render result: {success}, exists: {question_text_path.exists()}")
                        if success:
                            stats['question_text_images'] += 1
                        else:
                            logger.warning(f"      Failed to render {question_text_path.name}")

                    # 3. Render thinking chunks
                    if thinking_chunks:
                        stats['thinking_chunks'] += len(thinking_chunks)

                        for chunk_idx, chunk in enumerate(thinking_chunks):
                            chunk_filename = get_hash_filename(dataset_name, sample_id, f"thinking_{chunk_idx}")
                            chunk_path = images_dir / f"{chunk_filename}.png"

                            if not chunk_path.exists():
                                logger.info(f"    Rendering thinking chunk {chunk_idx}: {len(chunk)} chars -> {chunk_path.name}")
                                success = renderer.render(chunk, str(chunk_path))
                                logger.info(f"      Render result: {success}, exists: {chunk_path.exists()}")
                                if success:
                                    stats['thinking_images'] += 1
                                else:
                                    logger.warning(f"      Failed to render {chunk_path.name}")

                    # Progress logging
                    if stats['total_samples'] % 100 == 0:
                        logger.info(f"  Progress: {stats['total_samples']}/{len(df)} samples, "
                                  f"{stats['question_text_images']} question text, "
                                  f"{stats['thinking_images']} thinking images")

                except Exception as e:
                    stats['errors'] += 1
                    logger.warning(f"  Error: {e}")

        # Log dataset completion
        logger.info(f"  ✓ {dataset_name}: {stats['total_samples']} samples, "
                  f"{stats['question_text_images']} question text, "
                  f"{stats['thinking_images']} thinking images rendered")

        for key in total_stats:
            total_stats[key] += stats[key]

    logger.info("")
    logger.info("=" * 70)
    logger.info("Rendering Summary:")
    logger.info(f"  Total samples:        {total_stats['total_samples']}")
    logger.info(f"  Question images:      {total_stats['question_images']}")
    logger.info(f"  Question text imgs:   {total_stats['question_text_images']}")
    logger.info(f"  Thinking chunks:      {total_stats['thinking_chunks']}")
    logger.info(f"  Thinking images:      {total_stats['thinking_images']}")
    logger.info(f"  Errors:               {total_stats['errors']}")
    logger.info("=" * 70)
    logger.info(f"\nImages saved to: {images_dir}")
    logger.info("Next step: Run with --encode-only to extract latents")


def main_encode_only_ddp(args):
    """Multi-GPU encode phase - splits samples across GPUs within each dataset."""
    local_rank = args.local_rank
    world_size = args.world_size
    device = f'cuda:{local_rank}'

    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize encoder (device is handled internally by Qwen3VLEncoder)
    encoder = Qwen3VLEncoder(
        model_name_or_path="/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking",
        device=device,
        dtype=torch.bfloat16,
        use_vllm_kernels=True,  # Enable optimized vLLM kernels for speed
    )

    if local_rank == 0:
        logger.info(f"Sample-level splitting across {world_size} GPUs")
        logger.info("")

    # Call main_ddp which properly handles DDP for encoding
    return main_ddp(args)


def main_encode_only(args, encoder):
    """Phase 2: Batch encode pre-rendered images with cross-attention (OPTIMIZED).

    Multi-GPU: Splits samples across ranks, each encodes only its share.
    """
    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get DDP parameters
    local_rank = getattr(args, 'local_rank', 0)
    world_size = getattr(args, 'world_size', 1)

    # Determine datasets
    if args.all:
        datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    elif args.dataset:
        datasets = args.dataset
    else:
        raise ValueError("Either --dataset or --all must be specified")

    if local_rank == 0:
        logger.info(f"Processing {len(datasets)} datasets")
        logger.info(f"Images dir: {images_dir}")
        if world_size > 1:
            logger.info(f"Multi-GPU: {world_size} ranks, sample-level splitting")
        logger.info("")

    total_stats = {
        'total_samples': 0,
        'features_cached': 0,
        'extracted_features': 0,
        'errors': 0,
    }

    for dataset_name in datasets:
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"Skipping {dataset_name} (directory not found)")
            continue

        parquet_files = sorted(dataset_dir.glob('*.parquet'))
        if not parquet_files:
            logger.warning(f"Skipping {dataset_name} (no parquet files)")
            continue

        if local_rank == 0:
            logger.info(f"Processing {dataset_name} ({len(parquet_files)} shards)...")

        output_jsonl = output_dir / f"r1_onevision_{dataset_name}_thinking.jsonl"

        # Use rank-specific output file for multi-GPU
        if world_size > 1:
            output_jsonl = output_dir / f"r1_onevision_{dataset_name}_thinking_rank{local_rank}.jsonl"

        stats = {'total_samples': 0, 'features_cached': 0, 'extracted_features': 0, 'errors': 0}

        for shard_idx, parquet_path in enumerate(parquet_files):
            if local_rank == 0:
                logger.info(f"  Shard {shard_idx + 1}/{len(parquet_files)}: {parquet_path.name}")

            df = pd.read_parquet(parquet_path)
            if args.max_samples:
                df = df.head(args.max_samples)

            # Split samples across ranks for multi-GPU
            if world_size > 1:
                samples_per_rank = (len(df) + world_size - 1) // world_size
                start_idx = local_rank * samples_per_rank
                end_idx = min(start_idx + samples_per_rank, len(df))
                df = df.iloc[start_idx:end_idx]
                if local_rank == 0:
                    logger.info(f"  Split {len(df)} samples across {world_size} GPUs (this rank: {len(df)})")

            file_mode = 'w' if shard_idx == 0 else 'a'

            # Phase 2a: Collect all samples with thinking images
            samples_to_encode = []
            query_image_paths = []

            for idx, row in df.iterrows():
                try:
                    sample_id = row['id']
                    stats['total_samples'] += 1

                    conversations = row.get('conversations', [])
                    if hasattr(conversations, 'tolist'):
                        conversations = conversations.tolist()
                    elif not isinstance(conversations, list):
                        conversations = list(conversations) if conversations is not None else []

                    if len(conversations) < 2:
                        logger.info(f"    Sample {idx}: SKIP (conversations < 2)")
                        continue

                    question_text = conversations[0]['value']
                    assistant_msg = conversations[1]['value']

                    # Extract thinking with same params as render phase for consistency
                    thinking, answer = extract_thinking_and_answer(
                        assistant_msg,
                        max_chars=args.max_chars_per_chunk,
                        return_chunks=True,
                    )

                    filename = get_hash_filename(dataset_name, sample_id, "main")
                    query_image_path = images_dir / f"{filename}.png"

                    if not query_image_path.exists():
                        logger.info(f"    Sample {idx}: SKIP (query image {query_image_path.name} not found)")
                        continue

                    # Collect thinking image paths (question text + thinking chunks)
                    thinking_image_paths = []

                    # 1. Add question text image first
                    question_text_filename = get_hash_filename(dataset_name, sample_id, "question_text")
                    question_text_path = images_dir / f"{question_text_filename}.png"
                    if local_rank == 0:
                        logger.info(f"    Checking {question_text_path.name}: exists={question_text_path.exists()}")
                    if question_text_path.exists():
                        thinking_image_paths.append(str(question_text_path))

                    # 2. Add thinking chunk images
                    # thinking is already a list of chunks from extract_thinking_and_answer(return_chunks=True)
                    if thinking:
                        if local_rank == 0:
                            logger.info(f"    Thinking has {len(thinking)} chunks")
                        for chunk_idx, chunk in enumerate(thinking):
                            chunk_filename = get_hash_filename(dataset_name, sample_id, f"thinking_{chunk_idx}")
                            chunk_path = images_dir / f"{chunk_filename}.png"
                            if local_rank == 0:
                                logger.info(f"    Checking {chunk_path.name}: exists={chunk_path.exists()}")
                            if chunk_path.exists():
                                thinking_image_paths.append(str(chunk_path))

                    if local_rank == 0:
                        logger.info(f"    Total thinking_image_paths: {len(thinking_image_paths)}")
                    if thinking_image_paths:
                        samples_to_encode.append({
                            'sample_id': sample_id,
                            'question_text': question_text,
                            'answer': answer,
                            'query_image_path': str(query_image_path),
                            'query_cache_key': filename,
                            'question_text_path': str(question_text_path),  # Add question_text.pt path
                            'thinking_chunks': thinking,  # Actual thinking text for 'cot' field
                            'thinking_image_paths': thinking_image_paths,  # For encoding (includes question_text + thinking)
                        })
                        query_image_paths.append(str(query_image_path))

                except Exception as e:
                    stats['errors'] += 1
                    logger.warning(f"  Error processing sample: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

            if local_rank == 0:
                logger.info(f"  Found {len(samples_to_encode)} samples with thinking images")

            # Phase 2b: Collect ALL unique images (query + question + thinking) together
            if local_rank == 0:
                logger.info(f"  Collecting all unique images...")
            all_image_paths = set()

            for sample in samples_to_encode:
                # Add query image
                all_image_paths.add(sample['query_image_path'])
                # Add all thinking images (question text + thinking chunks)
                for path in sample['thinking_image_paths']:
                    all_image_paths.add(path)

            all_image_paths = sorted(all_image_paths)
            if local_rank == 0:
                logger.info(f"  Found {len(all_image_paths)} total unique images")

            # Phase 2c: Batch encode ALL images together (no grouping by resolution)
            unified_cache_dir = images_dir / ".feature_cache"
            unified_cache_dir.mkdir(exist_ok=True)

            batch_size = 64  # Larger batch for all images together

            # Filter: only encode images that aren't already cached
            images_to_encode = []
            for img_path in all_image_paths:
                filename = Path(img_path).stem
                cache_path = unified_cache_dir / f"{filename}.pt"
                if not cache_path.exists():
                    images_to_encode.append(img_path)

            if local_rank == 0:
                logger.info(f"  Encoding {len(images_to_encode)}/{len(all_image_paths)} images (rest cached)...")

            if not images_to_encode:
                if local_rank == 0:
                    logger.info(f"  ✓ All features already cached!")
            else:
                for i in range(0, len(images_to_encode), batch_size):
                    batch_paths = images_to_encode[i:i+batch_size]
                    batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

                    # Encode batch through ViT + patch merger
                    # Returns unpadded per-image features for true any-resolution efficiency
                    output = encoder.encode_images(batch_images)
                    l_features = output.features  # List[Tensor], each [actual_tokens, 2048]
                    grid_thw = output.grid_thw    # [batch, 3]

                    # Save each with both features + grid_thw
                    for img_path, l_feat, grid in zip(batch_paths, l_features, grid_thw):
                        filename = Path(img_path).stem
                        cache_path = unified_cache_dir / f"{filename}.pt"
                        torch.save({
                            'latent': l_feat.cpu(),  # [actual_tokens, 2048] - no padding!
                            'grid_thw': grid.cpu(),   # [3]
                        }, cache_path)
                        stats['features_cached'] += 1

                    if local_rank == 0 and (i + batch_size) % (batch_size * 10) == 0:
                        logger.info(f"    Cached {min(i + batch_size, len(images_to_encode))}/{len(images_to_encode)} features")

            if local_rank == 0:
                logger.info(f"  ✓ All features cached to {unified_cache_dir}")

            # Phase 2d: Process each sample using cached features (NO encoding!)
            if local_rank == 0:
                logger.info(f"  Combining cached features...")
            with open(output_jsonl, file_mode) as f_out:
                for sample in samples_to_encode:
                    try:
                        # Cache paths
                        query_cache_path = str(unified_cache_dir / f"{Path(sample['query_image_path']).stem}.pt")
                        # thinking_image_paths includes question_text + thinking chunks.
                        # For training, latent_ground_truth should contain ONLY thinking chunks (exclude question_text),
                        # and the number of <latent> placeholders must match len(latent_ground_truth).
                        thinking_cache_paths = []
                        for tp in sample["thinking_image_paths"]:
                            stem = Path(tp).stem
                            if "question_text" in stem:
                                continue
                            thinking_cache_paths.append(str(unified_cache_dir / f"{stem}.pt"))

                        num_thinking_chunks = len(thinking_cache_paths)
                        if num_thinking_chunks == 0:
                            # No thinking chunks -> skip (otherwise we would build <think></think> with empty latent GT).
                            continue

                        # latent_ground_truth: One entry per thinking chunk (for injection at <latent>)
                        latent_ground_truth = thinking_cache_paths

                        # latent_supervision: Main question image feature (used by OT/MSE/REPA/NCE losses)
                        latent_supervision = [query_cache_path]

                        # Reconstruct assistant message with thinking tags and latent placeholders
                        # Format: <think><latent><think_sep><latent>...</think>{answer}
                        # Note: <latent> and <think_sep> are TEXT tokens with CE loss
                        latent_placeholders = '<think_sep>'.join(['<latent>'] * num_thinking_chunks)
                        thinking_content = f"<think>{latent_placeholders}</think>{sample['answer']}"

                        # Write JSONL entry in llamafactory format
                        # Adaptively add <image> token only if question_text doesn't already have one
                        question_text = sample['question_text']
                        if '<image>' in question_text:
                            user_content = question_text
                        else:
                            user_content = f'<image>\n{question_text}'
                        json_entry = {
                            'messages': [
                                {'role': 'user', 'content': user_content},
                                {'role': 'assistant', 'content': thinking_content}
                            ],
                            'images': [sample['query_image_path']],
                            'latent_ground_truth': latent_ground_truth,
                            'latent_supervision': latent_supervision,
                            'num_latent_steps': num_thinking_chunks,
                            # Mirror CE-loss subsequence boundaries with <think_sep>.
                            # This stays a string for backward compatibility with existing JSONL.
                            'cot': format_cot_subsequences(sample.get('thinking_chunks')),
                            'task': 'r1_onevision_thinking',
                        }

                        f_out.write(json.dumps(json_entry) + '\n')
                        stats['features_cached'] += 1

                        # Progress
                        if local_rank == 0 and stats['features_cached'] % 100 == 0:
                            logger.info(f"    Progress: {stats['features_cached']}/{len(samples_to_encode)} samples combined")

                    except Exception as e:
                        stats['errors'] += 1
                        logger.warning(f"  Error combining {sample['sample_id']}: {e}")

        if local_rank == 0 or world_size == 1:
            logger.info(f"  ✓ {dataset_name}: {stats['extracted_features']} features extracted")

        for key in total_stats:
            total_stats[key] += stats[key]

    # Merge rank files on rank 0 if using multi-GPU
    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()  # Wait for all ranks to finish

        if local_rank == 0:
            logger.info("")
            logger.info("Merging rank files...")
            for dataset_name in datasets:
                base_jsonl = output_dir / f"r1_onevision_{dataset_name}_thinking.jsonl"
                rank_files = list(output_dir.glob(f"r1_onevision_{dataset_name}_thinking_rank*.jsonl"))

                if rank_files:
                    # Concatenate all rank files
                    with open(base_jsonl, 'w') as f_out:
                        for rank_file in sorted(rank_files, key=lambda x: int(x.stem.split('rank')[1])):
                            with open(rank_file, 'r') as f_in:
                                f_out.write(f_in.read())
                            # Delete rank file after merging
                            rank_file.unlink()

                    logger.info(f"  ✓ Merged {len(rank_files)} rank files -> {base_jsonl.name}")

    if local_rank == 0:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Encoding Summary:")
        logger.info(f"  Total samples:          {total_stats['total_samples']}")
        logger.info(f"  Features cached:        {total_stats['features_cached']}")
        logger.info(f"  Extracted features:     {total_stats['extracted_features']}")
        logger.info(f"  Errors:                 {total_stats['errors']}")
        logger.info("=" * 70)

    # Merge all dataset JSONL files into one (only on rank 0)
    if local_rank == 0 and len(datasets) > 1:
        logger.info("")
        logger.info("Merging all datasets into single JSONL...")

        merged_jsonl = output_dir / "r1_onevision_thinking.jsonl"
        total_samples = 0

        with open(merged_jsonl, 'w') as f_out:
            for dataset_name in datasets:
                dataset_jsonl = output_dir / f"r1_onevision_{dataset_name}_thinking.jsonl"
                if dataset_jsonl.exists():
                    sample_count = 0
                    with open(dataset_jsonl, 'r') as f_in:
                        for line in f_in:
                            f_out.write(line)
                            sample_count += 1
                    total_samples += sample_count
                    logger.info(f"  ✓ {dataset_name}: {sample_count} samples")

        logger.info(f"  ✓ Merged {total_samples} total samples -> {merged_jsonl.name}")

        # Delete per-dataset JSONLs after merge
        logger.info("")
        logger.info("Cleaning up per-dataset JSONLs...")
        for dataset_name in datasets:
            dataset_jsonl = output_dir / f"r1_onevision_{dataset_name}_thinking.jsonl"
            if dataset_jsonl.exists():
                dataset_jsonl.unlink()
                logger.info(f"  ✓ Deleted {dataset_jsonl.name}")

        logger.info("")

def main():
    parser = argparse.ArgumentParser(
        description='Build R1-OneVision Thinking Dataset with Qwen3VL Native Cross-Attention'
    )
    parser.add_argument('--dataset', nargs='+', help='Dataset name(s) to convert')
    parser.add_argument('--all', action='store_true', help='Convert all datasets')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples per dataset (for testing)')
    parser.add_argument('--base-dir',
                        default='/share/project/xiyan/sources/DeepSeek-OCR',
                        help='Base directory of repository')
    parser.add_argument('--data-dir',
                        default='/share/project/xiyan/huggingface/Fancy-MLLM/R1-Onevision',
                        help='Path to R1-Onevision data')
    parser.add_argument('--output-dir',
                        default='Qwen/data',
                        help='Output directory for JSONL files')
    parser.add_argument('--images-dir',
                        default='Qwen/data/r1_onevision_images',
                        help='Directory for rendered images and latent files')
    parser.add_argument('--max-chars-per-chunk', type=int, default=4800,
                        help='Maximum characters per thinking chunk (default: 4800)')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for cross-attention encoder')
    parser.add_argument('--cache-chunks', action='store_true',
                        help='Pre-render all unique chunks in batch before encoding (faster)')
    parser.add_argument('--cache-batch-size', type=int, default=64,
                        help='Batch size for pre-rendering chunks (default: 64)')
    parser.add_argument('--encode-batch-size', type=int, default=8,
                        help='Batch size for feature encoding (default: 8)')
    parser.add_argument('--render-only', action='store_true',
                        help='Phase 1: Only render images (CPU-bound, fast)')
    parser.add_argument('--encode-only', action='store_true',
                        help='Phase 2: Only encode pre-rendered images (GPU-bound, batch processing)')
    parser.add_argument('--query-batch-size', type=int, default=32,
                        help='Batch size for query image encoding (default: 32, only used with --encode-only)')
    parser.add_argument('--thinking-batch-size', type=int, default=16,
                        help='Batch size for thinking image encoding (default: 16, only used with --encode-only)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use (will launch torchrun automatically if > 1)')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for DDP (set automatically by torchrun)')
    parser.add_argument('--world-size', type=int, default=1,
                        help='World size for DDP (set automatically by torchrun)')

    args = parser.parse_args()

    # Auto-launch with torchrun if --num-gpus > 1 and not already in torchrun
    if args.num_gpus > 1 and 'LOCAL_RANK' not in os.environ:
        # Rebuild command line args without --num-gpus
        torchrun_args = [
            'torchrun',
            f'--nproc_per_node={args.num_gpus}',
            sys.argv[0],  # Current script
        ]

        # Add all args except --num-gpus and its value
        skip_next = False
        for i, arg in enumerate(sys.argv[1:]):
            if skip_next:
                skip_next = False
                continue
            if arg == '--num-gpus':
                skip_next = True  # Skip the next arg (the value)
                continue
            if arg.startswith('--num-gpus='):
                continue  # Skip --num-gpus=N format
            torchrun_args.append(arg)

        print(f"Launching {args.num_gpus} GPUs with torchrun...")
        print(f"Command: {' '.join(torchrun_args)}")
        sys.exit(subprocess.call(torchrun_args))

    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir

    # Get DDP parameters from environment (set by torchrun)
    # IMPORTANT: Update args with environment values so main_ddp gets correct values
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.world_size = int(os.environ.get('WORLD_SIZE', args.world_size))

    local_rank = args.local_rank
    world_size = args.world_size

    logger.info("=" * 70)
    logger.info("R1-OneVision Thinking Dataset Builder (Qwen3VL Native)")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Data directory:   {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Max samples:      {args.max_samples or 'All'}")
    logger.info(f"Max chars/chunk:  {args.max_chars_per_chunk}")
    logger.info(f"Device:           {args.device}")
    logger.info(f"World size:       {world_size}")
    logger.info(f"Render only:      {args.render_only}")
    logger.info(f"Encode only:      {args.encode_only}")
    logger.info("")

    # Handle two-phase mode
    if args.render_only:
        logger.info("=" * 70)
        logger.info("PHASE 1: Pre-rendering all images (question + thinking)")
        logger.info("=" * 70)
        logger.info("This will render all images to disk without encoding.")
        logger.info("Run with --encode-only after this completes.")
        logger.info("")

        # Rendering is CPU-bound, no GPU benefit from multi-GPU
        # Use single render path regardless of world_size
        renderer = AdaptiveVelloRenderer()
        return main_render_only(args, renderer)

    if args.encode_only:
        logger.info("=" * 70)
        logger.info("PHASE 2: Batch encoding pre-rendered images")
        logger.info("=" * 70)
        logger.info("This will batch encode all pre-rendered images.")
        logger.info("")

        if world_size > 1:
            # Multi-GPU encode
            if not dist.is_initialized():
                backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                dist.init_process_group(backend=backend)

            if local_rank == 0:
                logger.info("Multi-GPU encode mode initialized")
                logger.info(f"  World size: {world_size}")
                logger.info("")

            return main_encode_only_ddp(args)
        else:
            # Single-GPU encode
            encoder = Qwen3VLEncoder(
                model_name_or_path="/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking",
                device=args.device,
                dtype=torch.bfloat16,
                use_vllm_kernels=True,  # Enable optimized vLLM kernels for speed
            )
            return main_encode_only(args, encoder)

    # Route to DDP or single-GPU mode
    if world_size > 1:
        # DDP mode: initialize distributed training

        # Initialize process group if not already initialized
        if not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)

        if local_rank == 0:
            logger.info("DDP mode initialized")
            logger.info(f"  Local rank: {local_rank}")
            logger.info(f"  World size: {world_size}")
            logger.info("")

        # Call DDP main function (args.local_rank is now correctly set)
        return main_ddp(args)

    # Single-GPU mode: continue with normal processing
    # Initialize adaptive renderer
    logger.info("Initializing Adaptive Text Renderer...")
    renderer = AdaptiveVelloRenderer()
    logger.info("✓ Renderer initialized")
    logger.info("")

    # Initialize cross-attention encoder
    logger.info("Initializing Qwen3VL Cross-Attention Encoder...")
    encoder = Qwen3VLEncoder(
        model_name_or_path="/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking",
        device=args.device,
        dtype=torch.bfloat16,
        use_vllm_kernels=True,  # Enable optimized vLLM kernels for speed
    )
    logger.info("✓ Cross-Attention encoder initialized")
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
        'rendered_thinking': 0,
        'extracted_features': 0,
        'reused_images': 0,
        'errors': 0
    }

    for dataset_name in sorted(datasets):
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            logger.info(f"Skipping {dataset_name} (directory not found)")
            continue

        parquet_files = sorted(dataset_dir.glob('*.parquet'))
        if not parquet_files:
            logger.info(f"Skipping {dataset_name} (no parquet files)")
            continue

        output_jsonl = output_dir / f'r1_onevision_{dataset_name}_thinking.jsonl'

        logger.info(f"Processing {dataset_name} ({len(parquet_files)} shards)...")
        stats = {
            'total_samples': 0,
            'with_thinking': 0,
            'rendered_thinking': 0,
            'extracted_features': 0,
            'reused_images': 0,
            'errors': 0
        }

        for shard_idx, parquet_path in enumerate(parquet_files):
            logger.info(f"  Shard {shard_idx + 1}/{len(parquet_files)}: {parquet_path.name}")
            file_mode = 'w' if shard_idx == 0 else 'a'
            shard_stats = process_parquet_file(
                parquet_path=parquet_path,
                output_jsonl=output_jsonl,
                images_dir=images_dir,
                dataset_name=dataset_name,
                renderer=renderer,
                encoder=encoder,
                max_chars_per_chunk=args.max_chars_per_chunk,
                max_samples=args.max_samples,
                file_mode=file_mode,
            )
            for key in stats:
                stats[key] += shard_stats[key]

        logger.info(f"  Samples: {stats['total_samples']}, "
                   f"With thinking: {stats['with_thinking']}, "
                   f"Reused images: {stats['reused_images']}, "
                   f"Rendered: {stats['rendered_thinking']}, "
                   f"Features: {stats['extracted_features']}, "
                   f"Errors: {stats['errors']}")
        logger.info("")

        for key in total_stats:
            total_stats[key] += stats[key]

    # Merge all datasets into single JSONL
    if len(datasets) > 1:
        logger.info("")
        logger.info("Merging all datasets into single JSONL...")

        merged_jsonl = output_dir / "r1_onevision_thinking.jsonl"
        total_samples = 0

        with open(merged_jsonl, 'w') as f_out:
            for dataset_name in datasets:
                dataset_jsonl = output_dir / f"r1_onevision_{dataset_name}_thinking.jsonl"
                if dataset_jsonl.exists():
                    sample_count = 0
                    with open(dataset_jsonl, 'r') as f_in:
                        for line in f_in:
                            f_out.write(line)
                            sample_count += 1
                    total_samples += sample_count
                    logger.info(f"  ✓ {dataset_name}: {sample_count} samples")

        logger.info(f"  ✓ Merged {total_samples} total samples -> {merged_jsonl.name}")

        # Delete per-dataset JSONLs after merge
        logger.info("")
        logger.info("Cleaning up per-dataset JSONLs...")
        for dataset_name in datasets:
            dataset_jsonl = output_dir / f"r1_onevision_{dataset_name}_thinking.jsonl"
            if dataset_jsonl.exists():
                dataset_jsonl.unlink()
                logger.info(f"  ✓ Deleted {dataset_jsonl.name}")

        logger.info("")

    logger.info("=" * 70)
    logger.info("Conversion Complete!")
    logger.info("=" * 70)
    logger.info(f"Total samples:       {total_stats['total_samples']}")
    logger.info(f"With thinking:       {total_stats['with_thinking']}")
    logger.info(f"Reused images:       {total_stats['reused_images']}")
    logger.info(f"Rendered thinking:   {total_stats['rendered_thinking']}")
    logger.info(f"Extracted features:  {total_stats['extracted_features']}")
    logger.info(f"Errors:              {total_stats['errors']}")
    logger.info("")
    logger.info(f"JSONL files:          {output_dir}")
    logger.info(f"Images & latents:    {images_dir}")
    logger.info("")
    logger.info("✓ Ready to train!")
    logger.info("")
    logger.info("Each sample has:")
    logger.info("  - messages: user/assistant format")
    logger.info("  - images: [question_image.png]")
    logger.info("  - latent_supervision: [question__thinking_0.latent.pt, ...]")
    logger.info("  - num_latent_steps: number of thinking chunks")
    logger.info("")
    logger.info("During training:")
    logger.info("  1. Load question image via Qwen3VL native vision tower")
    logger.info("  2. Load .latent.pt files containing [seq, hidden_dim] features")
    logger.info("  3. Inject latents at <latent> token positions")
    logger.info("  4. Loss: Cross-entropy on answer + MSE on latent reconstruction")

    return 0


def main_ddp(args):
    """
    DDP-enabled main function for distributed processing.

    Args:
        args: Namespace with additional fields:
            - local_rank: int
            - world_size: int
            - device: str (e.g., 'cuda:0')
            - batch_size: int
            - ddp_enabled: bool
    """
    local_rank = args.local_rank
    world_size = args.world_size
    # In DDP mode, each rank uses its own GPU
    device = f'cuda:{local_rank}'
    batch_size = getattr(args, 'encode_batch_size', 8)

    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    output_dir = base_dir / args.output_dir
    images_dir = base_dir / args.images_dir

    if local_rank == 0:
        logger.info("=" * 70)
        logger.info("R1-OneVision Build (DDP Mode)")
        logger.info("=" * 70)
        logger.info(f"Rank: {local_rank}/{world_size}")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {batch_size}")
        logger.info("")

    # Initialize renderer (each rank has its own)
    renderer = AdaptiveVelloRenderer()

    # Initialize encoder on this GPU (device is handled internally by Qwen3VLEncoder)
    encoder = Qwen3VLEncoder(
        model_name_or_path="/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking",
        device=device,  # e.g., "cuda:0", "cuda:1", etc.
        dtype=torch.bfloat16,
        use_vllm_kernels=True,
    )

    # Initialize tokenizer for cot_token_ids (pre-tokenize CoT for efficient training)
    tokenizer = AutoTokenizer.from_pretrained(
        "/share/project/xiyan/sources/DeepSeek-OCR/Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking",
        trust_remote_code=True,
        use_fast=True,
    )

    if local_rank == 0:
        logger.info("✓ Encoder initialized")
        logger.info("")

    # Determine datasets
    if args.all:
        all_datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    elif args.dataset:
        all_datasets = args.dataset
    else:
        raise ValueError("Either --dataset or --all must be specified")

    # NOTE: All ranks process ALL datasets
    # Sample-level splitting happens within process_parquet_file_ddp()
    # This ensures better load balancing than dataset-level splitting

    if local_rank == 0:
        logger.info(f"Total datasets: {len(all_datasets)}")
        logger.info(f"Sample-level splitting across {world_size} GPUs")
        logger.info("")

    total_stats = {
        'total_samples': 0,
        'with_thinking': 0,
        'rendered_thinking': 0,
        'extracted_features': 0,
        'reused_images': 0,
        'errors': 0
    }

    for dataset_name in all_datasets:
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            if local_rank == 0:
                logger.info(f"Skipping {dataset_name} (directory not found)")
            continue

        parquet_files = sorted(dataset_dir.glob('*.parquet'))
        if not parquet_files:
            if local_rank == 0:
                logger.info(f"Skipping {dataset_name} (no parquet files)")
            continue
        output_jsonl = output_dir / f'r1_onevision_{dataset_name}_thinking.jsonl'

        if local_rank == 0:
            logger.info(f"[Rank {local_rank}] Processing {dataset_name} ({len(parquet_files)} shards)...")

        stats = {
            'total_samples': 0,
            'with_thinking': 0,
            'rendered_thinking': 0,
            'extracted_features': 0,
            'reused_images': 0,
            'errors': 0
        }

        for shard_idx, parquet_path in enumerate(parquet_files):
            if local_rank == 0:
                logger.info(f"  Shard {shard_idx + 1}/{len(parquet_files)}: {parquet_path.name}")
            file_mode = 'w' if shard_idx == 0 else 'a'
            shard_stats = process_parquet_file_ddp(
                parquet_path=parquet_path,
                output_jsonl=output_jsonl,
                images_dir=images_dir,
                dataset_name=dataset_name,
                renderer=renderer,
                encoder=encoder,
                tokenizer=tokenizer,
                max_chars_per_chunk=args.max_chars_per_chunk,
                max_samples=args.max_samples,
                batch_size=batch_size,
                local_rank=local_rank,
                world_size=world_size,
                file_mode=file_mode,
            )
            for key in stats:
                stats[key] += shard_stats[key]

        for key in total_stats:
            total_stats[key] += stats[key]

    # Synchronize all ranks
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Merge rank files on rank 0
    if local_rank == 0:
        logger.info("")
        logger.info("Merging rank files...")
        for dataset_name in all_datasets:
            output_jsonl = output_dir / f'r1_onevision_{dataset_name}_thinking.jsonl'
            rank_files = list(output_dir.glob(f"{output_jsonl.stem}_rank*{output_jsonl.suffix}"))

            if rank_files:
                # Concatenate all rank files
                with open(output_jsonl, 'w') as f_out:
                    for rank_file in sorted(rank_files, key=lambda x: int(x.stem.split('rank')[1])):
                        with open(rank_file, 'r') as f_in:
                            f_out.write(f_in.read())
                        # Delete rank file after merging
                        rank_file.unlink()

                logger.info(f"  ✓ Merged {len(rank_files)} rank files -> {output_jsonl.name}")

        # Merge all datasets into single JSONL
        if len(all_datasets) > 1:
            logger.info("")
            logger.info("Merging all datasets into single JSONL...")

            merged_jsonl = output_dir / "r1_onevision_thinking.jsonl"
            total_samples = 0

            with open(merged_jsonl, 'w') as f_out:
                for dataset_name in all_datasets:
                    dataset_jsonl = output_dir / f"r1_onevision_{dataset_name}_thinking.jsonl"
                    if dataset_jsonl.exists():
                        sample_count = 0
                        with open(dataset_jsonl, 'r') as f_in:
                            for line in f_in:
                                f_out.write(line)
                                sample_count += 1
                        total_samples += sample_count
                        logger.info(f"  ✓ {dataset_name}: {sample_count} samples")

            logger.info(f"  ✓ Merged {total_samples} total samples -> {merged_jsonl.name}")

            # Delete per-dataset JSONLs after merge
            logger.info("")
            logger.info("Cleaning up per-dataset JSONLs...")
            for dataset_name in all_datasets:
                dataset_jsonl = output_dir / f"r1_onevision_{dataset_name}_thinking.jsonl"
                if dataset_jsonl.exists():
                    dataset_jsonl.unlink()
                    logger.info(f"  ✓ Deleted {dataset_jsonl.name}")

        logger.info("")

    # Only rank 0 prints summary
    if local_rank == 0:
        logger.info("")
        logger.info("=" * 70)
        logger.info("DDP Build Summary (All Ranks)")
        logger.info("=" * 70)
        for key, val in total_stats.items():
            logger.info(f"{key.replace('_', ' ').title()}: {val}")
        logger.info("=" * 70)

    return 0


def process_parquet_file_ddp(
    parquet_path: Path,
    output_jsonl: Path,
    images_dir: Path,
    dataset_name: str,
    renderer: AdaptiveVelloRenderer,
    encoder: torch.nn.parallel.DistributedDataParallel,
    tokenizer,
    max_chars_per_chunk: int = 4800,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    local_rank: int = 0,
    world_size: int = 1,
    file_mode: str = 'w',
) -> dict:
    """Process parquet file with batched feature extraction for DDP."""
    df = pd.read_parquet(parquet_path)

    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)

    # Split data across ranks
    samples_per_rank = (len(df) + world_size - 1) // world_size
    start_idx = local_rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, len(df))
    df = df.iloc[start_idx:end_idx]

    stats = {
        'total_samples': len(df),
        'with_thinking': 0,
        'rendered_thinking': 0,
        'extracted_features': 0,
        'reused_images': 0,
        'errors': 0
    }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Only rank 0 writes to output file (or write to separate files per rank)
    output_jsonl_rank = output_jsonl.parent / f"{output_jsonl.stem}_rank{local_rank}{output_jsonl.suffix}"

    with open(output_jsonl_rank, file_mode) as f_out:
        # Process in batches
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            for idx, row in batch_df.iterrows():
                try:
                    sample_id = row['id']

                    # Progress logging every 20 samples
                    if stats['total_samples'] % 20 == 0 and stats['total_samples'] > 0:
                        logger.info(f"  Progress: {stats['total_samples']}/{len(df)} samples, "
                                  f"{stats['extracted_features']} features extracted")

                    # Parse conversations properly
                    # Handle both list and numpy array types for conversations
                    conversations = row.get('conversations', [])
                    # Convert to list if it's a numpy array or similar
                    if hasattr(conversations, 'tolist'):
                        conversations = conversations.tolist()
                    elif not isinstance(conversations, list):
                        conversations = list(conversations) if conversations is not None else []

                    if len(conversations) >= 2:
                        question_text = conversations[0]['value']
                        assistant_msg = conversations[1]['value']
                    else:
                        stats['errors'] += 1
                        continue

                    # Keep chunking consistent with render/encode phases (controls thinking_{i}.png count).
                    thinking, answer = extract_thinking_and_answer(
                        assistant_msg,
                        max_chars=max_chars_per_chunk,
                        return_chunks=True,
                    )

                    if not thinking:
                        # Save question image even without thinking
                        filename = get_hash_filename(dataset_name, sample_id, "main")
                        ocrvl_image_path = repo_root / "OCRVL/data/r1_onevision_images" / f"{filename}.png"

                        if ocrvl_image_path.exists():
                            image_path = ocrvl_image_path
                            stats['reused_images'] += 1
                        else:
                            image_path = images_dir / f"{filename}.png"
                            if isinstance(row.get('image'), str) and ',' in row['image']:
                                format, imgstr = row['image'].split(',', 1)
                                image_data = base64.b64decode(imgstr)
                                image_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(image_path, 'wb') as f_img:
                                    f_img.write(image_data)

                        # Cache original image features
                        unified_cache_dir = images_dir / ".feature_cache"
                        unified_cache_dir.mkdir(exist_ok=True)

                        original_image_path_obj = Path(image_path)
                        cache_path = unified_cache_dir / f"{original_image_path_obj.stem}.pt"

                        if not cache_path.exists():
                            try:
                                img = Image.open(image_path).convert("RGB")
                                output = encoder.encode_images([img])
                                torch.save({
                                    'latent': output.features[0].cpu(),
                                    'grid_thw': output.grid_thw[0].cpu(),
                                }, cache_path)
                            except Exception as e:
                                logger.warning(f"  Failed to encode image: {e}")

                        # Write sample without thinking (no latent supervision/ground truth)
                        # Adaptively add <image> token only if question_text doesn't already have one
                        if '<image>' in question_text:
                            user_content = question_text
                        else:
                            user_content = f'<image>\n{question_text}'
                        sample = {
                            'id': sample_id,
                            'messages': [
                                {'role': 'user', 'content': user_content},
                                {'role': 'assistant', 'content': answer or ''}
                            ],
                            'images': [str(image_path)],
                            'latent_ground_truth': [],      # No thinking chunks
                            'latent_supervision': [],        # No supervision
                            'num_latent_steps': 0,
                            'task': 'r1_onevision_thinking'
                        }
                        f_out.write(json.dumps(sample) + '\n')
                        continue

                    stats['with_thinking'] += 1

                    # Save main question image
                    filename = get_hash_filename(dataset_name, sample_id, "main")
                    ocrvl_image_path = repo_root / "OCRVL/data/r1_onevision_images" / f"{filename}.png"

                    if ocrvl_image_path.exists():
                        image_path = ocrvl_image_path
                        stats['reused_images'] += 1
                    else:
                        image_path = images_dir / f"{filename}.png"
                        if isinstance(row.get('image'), str) and ',' in row['image']:
                            format, imgstr = row['image'].split(',', 1)
                            image_data = base64.b64decode(imgstr)
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(image_path, 'wb') as f_img:
                                f_img.write(image_data)

                    # thinking is already a list of chunks from return_chunks=True
                    # Just use it directly (already chunked)
                    thinking_chunks = thinking if isinstance(thinking, list) else [thinking]

                    # Render thinking chunks to images (stored in Qwen cache)
                    thinking_image_paths = []
                    for chunk_idx, chunk in enumerate(thinking_chunks):
                        chunk_suffix = f"thinking_{chunk_idx}"
                        chunk_filename = get_hash_filename(dataset_name, sample_id, chunk_suffix)
                        chunk_path = images_dir / f"{chunk_filename}.png"

                        # Check if already rendered (cache hit)
                        if not chunk_path.exists():
                            # Render missing chunk - don't skip!
                            success = renderer.render(chunk, str(chunk_path))
                            if not success:
                                logger.warning(f"  Failed to render chunk {chunk_idx}, skipping...")
                                continue

                        # Verify file exists before using it
                        if chunk_path.exists():
                            thinking_image_paths.append(str(chunk_path))
                            stats['rendered_thinking'] += 1
                        else:
                            logger.warning(f"  Chunk {chunk_idx} file not found after rendering, skipping...")

                    # Collect all image paths for batch encoding
                    all_image_paths = [str(image_path)] + thinking_image_paths

                    # Batch encode all images and cache features
                    # This caches both original image and thinking images
                    unified_cache_dir = images_dir / ".feature_cache"
                    unified_cache_dir.mkdir(exist_ok=True)

                    # Filter out already cached images
                    images_to_encode = []
                    for img_path_str in all_image_paths:
                        img_path = Path(img_path_str)
                        cache_path = unified_cache_dir / f"{img_path.stem}.pt"
                        if not cache_path.exists():
                            images_to_encode.append(img_path)

                    # Encode missing images in one batch
                    if images_to_encode:
                        try:
                            batch_images = [Image.open(p).convert("RGB") for p in images_to_encode]
                            # Encode batch to L-1 features
                            output = encoder.encode_images(batch_images)
                            l_features_batch = output.features  # [batch, seq, 2048]
                            grid_thw_batch = output.grid_thw    # [batch, 3]

                            # Save each feature to cache
                            for img_path, l_feat, grid in zip(images_to_encode, l_features_batch, grid_thw_batch):
                                cache_path = unified_cache_dir / f"{img_path.stem}.pt"
                                torch.save({
                                    'latent': l_feat.cpu(),
                                    'grid_thw': grid.cpu(),
                                }, cache_path)

                            stats['rendered_thinking'] += len(images_to_encode)
                        except Exception as e:
                            logger.warning(f"  Failed to encode images: {e}")
                            stats['errors'] += 1
                            continue

                    # Load cached features for thinking images (for latent_ground_truth)
                    # Only include thinking chunks, NOT question_text
                    latent_ground_truth_paths = []
                    for thinking_path_str in thinking_image_paths:
                        thinking_path = Path(thinking_path_str)
                        # Skip question_text, only include thinking_0, thinking_1, etc.
                        if 'question_text' in thinking_path.stem:
                            continue
                        cache_path = unified_cache_dir / f"{thinking_path.stem}.pt"
                        if cache_path.exists():
                            # Use cached features directly
                            latent_ground_truth_paths.append(str(cache_path))

                    # Load cached features for original image (for latent_supervision)
                    # Only ONE entry - the main question image
                    original_image_path = Path(image_path)
                    original_cache_path = unified_cache_dir / f"{original_image_path.stem}.pt"
                    latent_supervision_paths = []

                    if original_cache_path.exists():
                        # Use cached features directly
                        latent_supervision_paths.append(str(original_cache_path))

                    # Note: question_text.pt path is available in sample['question_text_path'] if needed

                    # Count thinking chunks (exclude question_text)
                    num_thinking_chunks = len(latent_ground_truth_paths)

                    if not latent_ground_truth_paths or not latent_supervision_paths:
                        stats['errors'] += 1
                        continue

                    # latent_ground_truth: one entry per thinking chunk
                    # latent_supervision: one entry (main image) - used as supervision target for all thinking chunks
                    # No length check needed since they serve different purposes now

                    stats['extracted_features'] += 1

                    # Build thinking placeholders for messages (only thinking chunks, not question_text)
                    # Each thinking chunk gets <latent>, separated by <think_sep>
                    latent_placeholders = '<think_sep>'.join(['<latent>'] * num_thinking_chunks)
                    thinking_content = "<think>" + latent_placeholders + "</think>" + answer

                    # Get actual thinking text for 'cot' field
                    # 'thinking' variable contains the list of thinking chunks
                    cot_text = format_cot_subsequences(thinking if isinstance(thinking, list) else [thinking] if thinking else [])

                    # Write JSONL entry with proper separation
                    # Adaptively add <image> token only if question_text doesn't already have one
                    if '<image>' in question_text:
                        user_content = question_text
                    else:
                        user_content = f'<image>\n{question_text}'
                    # Build sample
                    question_pt = str(unified_cache_dir / f"{get_hash_filename(dataset_name, sample_id, 'question_text')}.pt")

                    sample = {
                        'id': sample_id,
                        'messages': [
                            {'role': 'user', 'content': user_content},
                            {'role': 'assistant', 'content': thinking_content}
                        ],
                        'images': [str(image_path)],
                        'question': question_pt,
                        'latent_ground_truth': latent_ground_truth_paths,  # Only thinking images
                        'latent_supervision': latent_supervision_paths,      # Only main image (one entry)
                        'num_latent_steps': num_thinking_chunks,  # Only thinking chunks count
                        # Mirror CE-loss subsequence boundaries with <think_sep>.
                        # This stays a string for backward compatibility with existing JSONL.
                        'cot': cot_text,
                        # Pre-tokenized CoT for efficient training (avoids re-tokenizing every batch)
                        'cot_token_ids': tokenizer.encode(cot_text, add_special_tokens=False),
                        'task': 'r1_onevision_thinking'
                    }
                    f_out.write(json.dumps(sample) + '\n')

                except Exception as e:
                    stats['errors'] += 1
                    if local_rank == 0:
                        import traceback
                        logger.warning(f"  Error processing sample {sample_id}: {e}\n{traceback.format_exc()}")

    return stats


if __name__ == '__main__':
    sys.exit(main())
