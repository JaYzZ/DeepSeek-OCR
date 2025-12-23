"""
Markovian Chunk Dataset for Visual Token Training

Variable-length word-based chunking for Markovian single-step prediction.
Each document is chunked into variable-length pieces, and consecutive pairs
are used for training: C_i → C_{i+1}

Chunking Strategy:
- Target: ~500 words per chunk (variance for robustness)
- Range: 10-900 words per chunk
- Each sample becomes 2+ chunks
- Training pairs: every consecutive (C_i, C_{i+1}) pair

Example:
    Document with 2000 words might become:
    [C1: 450 words] [C2: 600 words] [C3: 350 words] [C4: 600 words]

    Training pairs generated:
    (C1, C2), (C2, C3), (C3, C4)

    Each pair is one training sample for single-step Markovian prediction.

Memory Efficiency:
- Uses DPSKOCREncoder (~400M params) instead of full model (~7B)
- Saves ~13GB VRAM compared to loading full DeepSeek-OCR

Usage:
    from OCRFlow.training.markovian_dataset import create_markovian_dataloader

    dataloader = create_markovian_dataloader(
        dataset_type="fineweb",
        fineweb_subset="10BT",
        model_path="deepseek-ai/DeepSeek-OCR",
        batch_size=8,
    )

    for input_chunk, target_chunk in dataloader:
        # input_chunk: [batch, 111, 1280]
        # target_chunk: [batch, 111, 1280]
        loss = model.compute_loss_pair(input_chunk, target_chunk)
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Iterator, Tuple
import logging
import random
import hashlib

logger = logging.getLogger(__name__)


def count_words(text: str) -> int:
    """Count words in text (simple whitespace split)"""
    return len(text.split())


def chunk_text_by_words(
    text: str,
    target_words: int = 500,
    min_words: int = 10,
    max_words: int = 900,
    variance: float = 0.5,
    mixed_length: bool = False,
) -> List[str]:
    """
    Chunk text into variable-length pieces by word count.

    Args:
        text: Input text to chunk
        target_words: Target words per chunk (default: 500)
        min_words: Minimum words per chunk (default: 10)
        max_words: Maximum words per chunk (default: 900)
        variance: Variance factor for chunk size (default: 0.5)
                  Actual chunk size = target_words * uniform(1-variance, 1+variance)
        mixed_length: If True, use mixed-length strategy for short answer generation
                      Samples from [short, medium, long] distributions

    Returns:
        List of text chunks (at least 2 chunks if text is long enough)
    """
    words = text.split()
    total_words = len(words)

    # If text is too short for 2 chunks, return empty
    if total_words < min_words * 2:
        return []

    chunks = []
    current_pos = 0

    # Mixed-length distributions for short answer capability
    # Format: (target_words, variance, probability)
    length_distributions = [
        (30, 0.5, 0.2),   # Short: 15-45 words (20% probability)
        (100, 0.4, 0.3),  # Medium: 60-140 words (30% probability)
        (400, 0.5, 0.5),  # Long: 200-600 words (50% probability)
    ] if mixed_length else [(target_words, variance, 1.0)]

    while current_pos < total_words:
        # Sample chunk size
        if mixed_length:
            # Sample from mixed distributions
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

        # Don't leave tiny remainder
        remaining = total_words - current_pos
        if remaining < min_words:
            break

        # If remainder would be too small after this chunk, take it all
        if remaining - chunk_size < min_words and remaining <= max_words:
            chunk_size = remaining

        # Extract chunk
        end_pos = min(current_pos + chunk_size, total_words)
        chunk_words = words[current_pos:end_pos]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        current_pos = end_pos

    # Ensure at least 2 chunks for training pairs
    if len(chunks) < 2:
        return []

    return chunks


class MarkovianChunkDataset(IterableDataset):
    """
    Dataset that chunks documents and yields consecutive pairs for Markovian training.

    Each document is split into variable-length chunks, and all consecutive
    pairs (C_i, C_{i+1}) are yielded as training samples.

    Uses memory-efficient DPSKOCREncoder (~400M params) instead of full model (~7B).
    """

    def __init__(
        self,
        data_sources: Dict[str, Dict],
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        target_words: int = 500,
        min_words: int = 10,
        max_words: int = 900,
        variance: float = 0.5,
        min_doc_words: int = 100,  # Minimum words for a document to be used
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        encode_batch_size: int = 24,
        num_render_workers: int = 16,  # High parallelism for text rendering
        mixed_length: bool = False,  # Enable mixed-length chunks for short answer capability
    ):
        """
        Args:
            data_sources: Dict of data source configs
            model_path: Path to DeepSeek-OCR model
            device: Device for encoder
            target_words: Target words per chunk
            min_words: Minimum words per chunk
            max_words: Maximum words per chunk
            variance: Variance factor for chunk sizes
            min_doc_words: Minimum words for document to be used
            shuffle: Shuffle data sources
            max_samples: Maximum training pairs (None = unlimited)
            cache_dir: Optional cache directory
            encode_batch_size: Batch size for encoder
            num_render_workers: Number of parallel text rendering workers
            mixed_length: If True, use mixed-length chunking (short/medium/long)
                         for short answer generation capability
        """
        self.data_sources = data_sources
        self.model_path = model_path
        self.device = device
        self.target_words = target_words
        self.min_words = min_words
        self.max_words = max_words
        self.variance = variance
        self.min_doc_words = min_doc_words
        self.mixed_length = mixed_length
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.encode_batch_size = encode_batch_size
        self.num_render_workers = num_render_workers

        # Encoder will be lazily initialized
        self.encoder = None

        # Cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Normalize weights
        total_weight = sum(src.get("weight", 1.0) for src in data_sources.values())
        self.weights = {
            name: src.get("weight", 1.0) / total_weight
            for name, src in data_sources.items()
        }

        # Find parquet files
        self.parquet_files = {}
        for name, config in data_sources.items():
            path = Path(config["path"])
            data_dir = path / "data" if (path / "data").exists() else path
            files = list(data_dir.glob("**/*.parquet"))
            self.parquet_files[name] = sorted(files)
            logger.info(f"Found {len(files)} parquet files for {name}")

    def _text_hash(self, text: str) -> str:
        """Hash text for caching"""
        return hashlib.md5(text[:200].encode()).hexdigest()[:16]

    def _get_cache_path(self, text_hash: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{text_hash}.pt"

    def _load_from_cache(self, text_hash: str) -> Optional[torch.Tensor]:
        cache_path = self._get_cache_path(text_hash)
        if cache_path and cache_path.exists():
            try:
                return torch.load(cache_path, map_location="cpu")
            except:
                return None
        return None

    def _save_to_cache(self, text_hash: str, tensor: torch.Tensor):
        cache_path = self._get_cache_path(text_hash)
        if cache_path:
            try:
                torch.save(tensor.cpu(), cache_path)
            except:
                pass

    def _iter_documents(self, source_name: str) -> Iterator[str]:
        """Iterate over documents from a data source"""
        files = self.parquet_files.get(source_name, [])
        if self.shuffle:
            files = files.copy()
            random.shuffle(files)

        for parquet_path in files:
            try:
                df = pd.read_parquet(parquet_path)

                if "text" not in df.columns:
                    continue

                if self.shuffle:
                    df = df.sample(frac=1.0).reset_index(drop=True)

                for _, row in df.iterrows():
                    text = row["text"]
                    # Filter by minimum document length
                    if count_words(text) >= self.min_doc_words:
                        yield text

            except Exception as e:
                logger.warning(f"Error reading {parquet_path}: {e}")
                continue

    def _init_encoder(self):
        """Lazily initialize the encoder"""
        if self.encoder is None:
            from PIL import Image
            from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
            from Renderer.pil_renderer import PILRenderer, render_to_pil
            try:
                from Renderer import VelloRenderer  # type: ignore
            except Exception:
                VelloRenderer = None

            logger.info(
                f"Initializing DPSK OCR encoder on {self.device} with {self.num_render_workers} render workers..."
            )
            self.encoder = DPSKOCREncoder(
                model_path=self.model_path,
                device=self.device,
                dtype=torch.bfloat16,
            )

            vello = None
            if VelloRenderer is not None:
                try:
                    vello = VelloRenderer(width=640, height=640, padding=20)
                    logger.info("MarkovianChunkDataset using VelloRenderer")
                except Exception:
                    vello = None
            pil_renderer = None if vello is not None else PILRenderer(
                width=640, height=640, num_workers=self.num_render_workers
            )

            def render_texts(texts: List[str]) -> List[Image.Image]:
                if vello is not None:
                    arrays = vello.render_batch(list(texts))
                    return [Image.fromarray(arr) for arr in arrays]
                if pil_renderer is not None:
                    return pil_renderer.render_batch_pil(list(texts))
                return [render_to_pil(t, width=640, height=640) for t in texts]

            self._render_texts = render_texts

    def _encode_chunk(self, chunk_text: str) -> Optional[torch.Tensor]:
        """Encode a single chunk text to visual tokens"""
        text_hash = self._text_hash(chunk_text)

        # Check cache
        cached = self._load_from_cache(text_hash)
        if cached is not None:
            return cached

        # Encode
        try:
            # Truncate to ~6000 chars (roughly 900 words * 6-7 chars/word)
            truncated = chunk_text[:6000]
            images = self._render_texts([truncated])
            tensors = self.encoder.encode_images(images, return_global=False, return_local=True)
            if tensors:
                tensor = tensors[0].cpu()
                self._save_to_cache(text_hash, tensor)
                return tensor
        except Exception as e:
            logger.warning(f"Encoding error: {e}")

        return None

    def _encode_chunks_batch(self, chunk_texts: List[str]) -> List[Optional[torch.Tensor]]:
        """Encode multiple chunks in batch"""
        results = [None] * len(chunk_texts)
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(chunk_texts):
            text_hash = self._text_hash(text)
            cached = self._load_from_cache(text_hash)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(text[:6000])
                uncached_indices.append(i)

        # Encode uncached
        if uncached_texts:
            try:
                images = self._render_texts(uncached_texts)
                tensors = self.encoder.encode_images(images, return_global=False, return_local=True)
                for idx, tensor in zip(uncached_indices, tensors):
                    tensor_cpu = tensor.cpu()
                    results[idx] = tensor_cpu
                    text_hash = self._text_hash(chunk_texts[idx])
                    self._save_to_cache(text_hash, tensor_cpu)
            except Exception as e:
                logger.warning(f"Batch encoding error: {e}")

        return results

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over (input_chunk, target_chunk) pairs.

        Yields:
            Tuple of (input_chunk, target_chunk), each [111, 1280]
        """
        # Initialize encoder (lazy loading)
        self._init_encoder()

        # Create iterators for each source
        source_iters = {name: self._iter_documents(name) for name in self.data_sources}
        source_names = list(self.weights.keys())
        source_weights = [self.weights[name] for name in source_names]

        sample_count = 0

        # Buffer for batch encoding
        pending_pairs: List[Tuple[str, str]] = []  # (input_text, target_text)

        while True:
            if self.max_samples and sample_count >= self.max_samples:
                return

            # Get next document
            source = random.choices(source_names, weights=source_weights, k=1)[0]

            try:
                doc_text = next(source_iters[source])
            except StopIteration:
                # Reinitialize
                source_iters[source] = self._iter_documents(source)
                try:
                    doc_text = next(source_iters[source])
                except StopIteration:
                    continue

            # Chunk document
            chunks = chunk_text_by_words(
                doc_text,
                target_words=self.target_words,
                min_words=self.min_words,
                max_words=self.max_words,
                variance=self.variance,
                mixed_length=self.mixed_length,
            )

            if len(chunks) < 2:
                continue

            # Create consecutive pairs
            for i in range(len(chunks) - 1):
                pending_pairs.append((chunks[i], chunks[i + 1]))

            # Process in batches
            while len(pending_pairs) >= self.encode_batch_size:
                batch_pairs = pending_pairs[:self.encode_batch_size]
                pending_pairs = pending_pairs[self.encode_batch_size:]

                # Encode all texts in batch
                all_texts = []
                for inp, tgt in batch_pairs:
                    all_texts.extend([inp, tgt])

                encoded = self._encode_chunks_batch(all_texts)

                # Yield pairs
                for i, (inp_text, tgt_text) in enumerate(batch_pairs):
                    inp_tensor = encoded[i * 2]
                    tgt_tensor = encoded[i * 2 + 1]

                    if inp_tensor is not None and tgt_tensor is not None:
                        yield inp_tensor, tgt_tensor
                        sample_count += 1

                        if self.max_samples and sample_count >= self.max_samples:
                            return

        # Process remaining pairs
        if pending_pairs:
            all_texts = []
            for inp, tgt in pending_pairs:
                all_texts.extend([inp, tgt])

            encoded = self._encode_chunks_batch(all_texts)

            for i, (inp_text, tgt_text) in enumerate(pending_pairs):
                inp_tensor = encoded[i * 2]
                tgt_tensor = encoded[i * 2 + 1]

                if inp_tensor is not None and tgt_tensor is not None:
                    yield inp_tensor, tgt_tensor
                    sample_count += 1

                    if self.max_samples and sample_count >= self.max_samples:
                        return


def create_markovian_dataloader(
    dataset_type: str = "fineweb",
    model_path: str = "deepseek-ai/DeepSeek-OCR",
    fineweb_path: str = "/share/project/xiyan/huggingface/HuggingFaceFW/fineweb-edu",
    fineweb_subset: Optional[str] = "10BT",
    openwebmath_path: str = "/share/project/xiyan/huggingface/open-web-math/open-web-math",
    fineweb_weight: float = 0.7,
    openwebmath_weight: float = 0.3,
    batch_size: int = 8,
    encode_batch_size: int = 16,
    target_words: int = 500,
    min_words: int = 10,
    max_words: int = 900,
    variance: float = 0.5,
    min_doc_words: int = 100,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    num_workers: int = 0,
    num_render_workers: int = 16,
    mixed_length: bool = False,
) -> DataLoader:
    """
    Create dataloader for Markovian chunk training.

    Each batch contains (input_chunks, target_chunks) pairs for single-step prediction.
    Uses memory-efficient DPSKOCREncoder (~400M params) instead of full model (~7B).

    Args:
        dataset_type: "fineweb", "openwebmath", or "multi"
        model_path: Path to DeepSeek-OCR model
        fineweb_path: Path to FineWeb-Edu dataset
        fineweb_subset: FineWeb subset ("10BT", "100BT", "350BT")
        openwebmath_path: Path to OpenWebMath
        fineweb_weight: Weight for FineWeb in multi mode
        openwebmath_weight: Weight for OpenWebMath in multi mode
        batch_size: Batch size (number of pairs per batch)
        encode_batch_size: Encoder batch size (default: 16)
        target_words: Target words per chunk (default: 500)
        min_words: Min words per chunk (default: 10)
        max_words: Max words per chunk (default: 900)
        variance: Chunk size variance (default: 0.5)
        min_doc_words: Min words for document to be used
        max_samples: Max training pairs
        cache_dir: Cache directory
        device: Encoder device
        num_workers: DataLoader workers
        num_render_workers: Number of parallel text rendering workers (default: 16)
        mixed_length: Enable mixed-length chunking for short answer capability

    Returns:
        DataLoader yielding (input_batch, target_batch) tuples
        Each tensor has shape [batch, 111, 1280]
    """
    # Resolve FineWeb path
    if fineweb_subset:
        fineweb_data_path = Path(fineweb_path) / "sample" / fineweb_subset
        logger.info(f"Using FineWeb-Edu subset: {fineweb_subset}")
    else:
        fineweb_data_path = Path(fineweb_path) / "data"

    # Configure data sources
    if dataset_type == "fineweb":
        data_sources = {
            "fineweb": {"path": str(fineweb_data_path), "weight": 1.0}
        }
    elif dataset_type == "openwebmath":
        data_sources = {
            "openwebmath": {"path": openwebmath_path, "weight": 1.0}
        }
    elif dataset_type == "multi":
        data_sources = {
            "fineweb": {"path": str(fineweb_data_path), "weight": fineweb_weight},
            "openwebmath": {"path": openwebmath_path, "weight": openwebmath_weight},
        }
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Create dataset (encoder initialized lazily inside for memory efficiency)
    dataset = MarkovianChunkDataset(
        data_sources=data_sources,
        model_path=model_path,
        device=device,
        target_words=target_words,
        min_words=min_words,
        max_words=max_words,
        variance=variance,
        min_doc_words=min_doc_words,
        max_samples=max_samples,
        cache_dir=cache_dir,
        encode_batch_size=encode_batch_size,
        num_render_workers=num_render_workers,
        mixed_length=mixed_length,
    )

    # Collate function for pairs
    def collate_pairs(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        inputs = torch.stack([pair[0].float() for pair in batch], dim=0)
        targets = torch.stack([pair[1].float() for pair in batch], dim=0)
        return inputs, targets

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing MarkovianChunkDataset...")
    print()

    # Test chunking function
    print("=== Testing chunk_text_by_words ===")
    test_text = " ".join(["word"] * 2000)  # 2000 words
    chunks = chunk_text_by_words(test_text, target_words=500, variance=0.3)
    print(f"Input: {count_words(test_text)} words")
    print(f"Output: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {count_words(chunk)} words")
    print()

    # Test with real dataset (small sample)
    print("=== Testing MarkovianChunkDataset ===")
    dataloader = create_markovian_dataloader(
        dataset_type="fineweb",
        fineweb_subset="10BT",
        max_samples=16,
        batch_size=4,
        encode_batch_size=4,
        target_words=500,
        min_words=50,
        max_words=800,
    )

    print("\nIterating over dataloader...")
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        if i >= 2:
            break

    print("\n✓ Test passed!")
