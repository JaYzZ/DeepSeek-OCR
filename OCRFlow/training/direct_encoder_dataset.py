"""
Direct Encoder Dataset for Visual Token Training

Uses the DeepSeek-OCR vision encoder directly (no HTTP server) for faster throughput.
Supports FineWeb-Edu, OpenWebMath, and mixed datasets.

Key features:
- Direct encoder integration (no HTTP overhead)
- Fast parallel text rendering (9x faster with multiprocessing)
- Native GPU batching
- Simpler deployment (no separate server process)

Usage:
    from OCRFlow.training.direct_encoder_dataset import create_direct_dataloader

    dataloader = create_direct_dataloader(
        dataset_type="multi",
        model_path="deepseek-ai/DeepSeek-OCR",
        batch_size=8,
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Iterator, Union
import logging
import random
import hashlib
import pickle
from dataclasses import dataclass
from project_paths import hf_path

# Text rendering is provided by ./Renderer.
try:
    from Renderer import VelloRenderer  # type: ignore
except Exception:  # pragma: no cover
    VelloRenderer = None
from Renderer.pil_renderer import PILRenderer, render_to_pil

logger = logging.getLogger(__name__)


def estimate_token_count(text: str) -> int:
    """Estimate token count (~4 chars per token)"""
    return len(text) // 4


# Alias for compatibility
def render_text_to_image(
    text: str,
    width: int = 640,
    height: int = 640,
    font_size: int = 18,
    **kwargs
) -> Image.Image:
    """Render text to PIL Image (uses optimized renderer)"""
    return render_to_pil(text, width=width, height=height, font_size=font_size)


class _BatchTextRenderer:
    """Local batch renderer (Vello when available, else PILRenderer)."""

    def __init__(self, num_workers: int = 4, font_size: int = 18, width: int = 640, height: int = 640):
        self.font_size = font_size
        self.width = width
        self.height = height
        self._vello = None
        if VelloRenderer is not None:
            try:
                self._vello = VelloRenderer(width=width, height=height)
            except Exception:
                self._vello = None
        self._pil = None if self._vello is not None else PILRenderer(
            num_workers=num_workers, width=width, height=height
        )

    def render_batch(self, texts: List[str]) -> List[Image.Image]:
        if self._vello is not None:
            arrays = self._vello.render_batch(list(texts))
            return [Image.fromarray(arr) for arr in arrays]
        return self._pil.render_batch_pil(texts)  # type: ignore[union-attr]


class DirectVisionEncoder:
    """
    Direct vision encoder wrapper for DeepSeek-OCR.

    Loads the model once and provides efficient batched encoding.
    """

    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self._initialized = False
        self._renderer = None  # Lazy init parallel renderer

    def initialize(self):
        """Lazy initialization of the model"""
        if self._initialized:
            return

        logger.info(f"Loading DeepSeek-OCR encoder from {self.model_path}...")

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()

        # Import processor
        from OCRInfer.process.image_process import DeepseekOCRProcessor
        self.processor = DeepseekOCRProcessor()

        self._initialized = True
        logger.info("✓ Encoder initialized")

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Encode images to visual tokens [111, 1280] each.

        Args:
            images: List of PIL Images (rendered text at 640x640)

        Returns:
            List of visual token tensors [111, 1280]
        """
        if not self._initialized:
            self.initialize()

        from PIL import ImageOps

        # Process images
        pixel_values_list = []
        for image in images:
            global_view = ImageOps.pad(
                image,
                (self.processor.base_size, self.processor.base_size),
                color=tuple(int(x * 255) for x in self.processor.image_transform.mean)
            )
            pixel_values = self.processor.image_transform(global_view)
            pixel_values_list.append(pixel_values)

        pixel_values = torch.stack(pixel_values_list, dim=0).to(
            device=self.device, dtype=self.dtype
        )

        batch_size = len(images)

        # Encode through vision model
        sam_features = self.model.model.sam_model(pixel_values)
        clip_features = self.model.model.vision_model(pixel_values, sam_features)

        # Concatenate features
        features = torch.cat(
            (
                clip_features[:, 1:],
                sam_features.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )

        # Project
        features = self.model.model.projector(features)

        # Add newline tokens and view separator
        _, hw, dim = features.shape
        side = int(hw ** 0.5)

        embeddings_list = []
        for jdx in range(batch_size):
            img_features = features[jdx].view(side, side, dim)
            newline = self.model.model.image_newline[None, None, :].expand(side, 1, dim)
            img_features = torch.cat([img_features, newline], dim=1)
            img_features = img_features.view(-1, dim)

            combined = torch.cat(
                [img_features, self.model.model.view_seperator[None, :]],
                dim=0
            )
            embeddings_list.append(combined)

        return embeddings_list

    def encode_texts(self, texts: List[str], chunk_size: int = 1000, use_parallel: bool = True) -> List[torch.Tensor]:
        """
        Encode texts to visual tokens.

        Args:
            texts: List of text strings
            chunk_size: Characters per chunk (for chunking long texts)
            use_parallel: Use parallel rendering for batches (9x faster)

        Returns:
            List of visual token tensors [111, 1280]
        """
        # Truncate texts
        truncated = [text[:chunk_size] for text in texts]

        # Render texts to images (use parallel for batches > 2)
        if use_parallel and len(truncated) > 2:
            if self._renderer is None:
                self._renderer = _BatchTextRenderer(num_workers=4, font_size=18)
            images = self._renderer.render_batch(truncated)
        else:
            images = [render_text_to_image(text) for text in truncated]

        # Encode images
        return self.encode_images(images)


# Global encoder instance (lazy loaded)
_global_encoder: Optional[DirectVisionEncoder] = None


def get_global_encoder(
    model_path: str = "deepseek-ai/DeepSeek-OCR",
    device: str = "cuda",
) -> DirectVisionEncoder:
    """Get or create global encoder instance"""
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = DirectVisionEncoder(model_path=model_path, device=device)
    return _global_encoder


class DirectEncoderDataset(IterableDataset):
    """
    Dataset that uses direct encoder for on-the-fly visual token generation.

    Much faster than HTTP server approach (22x speedup).
    """

    def __init__(
        self,
        data_sources: Dict[str, Dict],
        encoder: DirectVisionEncoder,
        min_tokens: int = 100,
        max_tokens: int = 1200,
        chunk_size: int = 1000,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        encode_batch_size: int = 24,
    ):
        """
        Args:
            data_sources: Dict of data source configs
                {
                    "fineweb": {"path": "/path/to/fineweb", "weight": 0.7},
                    "openwebmath": {"path": "/path/to/openwebmath", "weight": 0.3},
                }
            encoder: DirectVisionEncoder instance
            min_tokens: Minimum estimated token count
            max_tokens: Maximum estimated token count
            chunk_size: Characters per visual chunk
            shuffle: Shuffle data
            max_samples: Maximum samples (None = unlimited)
            cache_dir: Optional cache directory for visual tokens
            encode_batch_size: Batch size for encoder (higher = faster but more memory)
        """
        self.data_sources = data_sources
        self.encoder = encoder
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.encode_batch_size = encode_batch_size

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

        # Find parquet files for each source
        self.parquet_files = {}
        for name, config in data_sources.items():
            path = Path(config["path"])
            data_dir = path / "data" if (path / "data").exists() else path
            files = list(data_dir.glob("**/*.parquet"))
            self.parquet_files[name] = sorted(files)
            logger.info(f"Found {len(files)} parquet files for {name}")

    def _get_cache_path(self, text_hash: str, source: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / source / f"{text_hash}.pt"

    def _load_from_cache(self, text_hash: str, source: str) -> Optional[torch.Tensor]:
        cache_path = self._get_cache_path(text_hash, source)
        if cache_path and cache_path.exists():
            try:
                return torch.load(cache_path, map_location="cpu")
            except:
                return None
        return None

    def _save_to_cache(self, text_hash: str, source: str, tensor: torch.Tensor):
        cache_path = self._get_cache_path(text_hash, source)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.save(tensor.cpu(), cache_path)
            except:
                pass

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text[:500].encode()).hexdigest()

    def _iter_source(self, source_name: str) -> Iterator[str]:
        """Iterate over texts from a data source"""
        files = self.parquet_files.get(source_name, [])
        if self.shuffle:
            files = files.copy()
            random.shuffle(files)

        for parquet_path in files:
            try:
                df = pd.read_parquet(parquet_path)

                if "text" not in df.columns:
                    continue

                # Estimate token counts if needed
                if "token_count" not in df.columns:
                    df["token_count"] = df["text"].apply(estimate_token_count)

                # Filter by length
                df = df[
                    (df["token_count"] >= self.min_tokens) &
                    (df["token_count"] <= self.max_tokens)
                ]

                if self.shuffle:
                    df = df.sample(frac=1.0).reset_index(drop=True)

                for _, row in df.iterrows():
                    yield row["text"]

            except Exception as e:
                logger.warning(f"Error reading {parquet_path}: {e}")
                continue

    def __iter__(self):
        """Iterate with batched encoding for efficiency"""
        # Initialize encoder
        self.encoder.initialize()

        # Create iterators for each source
        source_iters = {name: self._iter_source(name) for name in self.data_sources}
        source_names = list(self.weights.keys())
        source_weights = [self.weights[name] for name in source_names]

        sample_count = 0
        text_buffer = []
        source_buffer = []

        while True:
            # Fill buffer
            while len(text_buffer) < self.encode_batch_size:
                # Weighted source selection
                source = random.choices(source_names, weights=source_weights, k=1)[0]

                try:
                    text = next(source_iters[source])
                    text_buffer.append(text)
                    source_buffer.append(source)
                except StopIteration:
                    # Reinitialize exhausted source
                    source_iters[source] = self._iter_source(source)
                    try:
                        text = next(source_iters[source])
                        text_buffer.append(text)
                        source_buffer.append(source)
                    except StopIteration:
                        # Source truly exhausted
                        break

            if not text_buffer:
                break

            # Check cache first
            cached_tensors = []
            uncached_texts = []
            uncached_indices = []

            for i, (text, source) in enumerate(zip(text_buffer, source_buffer)):
                text_hash = self._text_hash(text)
                cached = self._load_from_cache(text_hash, source)
                if cached is not None:
                    cached_tensors.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Encode uncached texts
            if uncached_texts:
                try:
                    encoded = self.encoder.encode_texts(uncached_texts, self.chunk_size)

                    # Save to cache and collect results
                    for idx, tensor in zip(uncached_indices, encoded):
                        text_hash = self._text_hash(text_buffer[idx])
                        self._save_to_cache(text_hash, source_buffer[idx], tensor)
                        cached_tensors.append((idx, tensor.cpu()))

                except Exception as e:
                    logger.warning(f"Encoding error: {e}")
                    text_buffer.clear()
                    source_buffer.clear()
                    continue

            # Sort by original index and yield
            cached_tensors.sort(key=lambda x: x[0])

            for _, tensor in cached_tensors:
                yield tensor
                sample_count += 1

                if self.max_samples and sample_count >= self.max_samples:
                    return

            # Clear buffers
            text_buffer.clear()
            source_buffer.clear()


def create_direct_dataloader(
    dataset_type: str = "multi",
    model_path: str = "deepseek-ai/DeepSeek-OCR",
    fineweb_path: str = str(hf_path("HuggingFaceFW", "fineweb-edu")),
    fineweb_subset: Optional[str] = None,
    openwebmath_path: str = str(hf_path("open-web-math", "open-web-math")),
    fineweb_weight: float = 0.7,
    openwebmath_weight: float = 0.3,
    train_data: Optional[str] = None,
    batch_size: int = 8,
    encode_batch_size: int = 8,
    min_tokens: int = 100,
    max_tokens: int = 1200,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    num_workers: int = 0,
) -> DataLoader:
    """
    Create dataloader with direct encoder integration.

    Args:
        dataset_type: "fineweb", "openwebmath", or "multi"
        model_path: Path to DeepSeek-OCR model
        fineweb_path: Path to FineWeb-Edu dataset root
        fineweb_subset: FineWeb subset to use:
            - None: Use full dataset (all CommonCrawl dumps)
            - "10BT": 10 billion tokens (~27GB, fastest for testing)
            - "100BT": 100 billion tokens (~267GB, good balance)
            - "350BT": 350 billion tokens (~930GB, most diverse)
        openwebmath_path: Path to OpenWebMath dataset
        fineweb_weight: Weight for FineWeb in multi mode
        openwebmath_weight: Weight for OpenWebMath in multi mode
        train_data: Override path for single dataset mode
        batch_size: Training batch size
        encode_batch_size: Batch size for encoder (affects speed/memory)
        min_tokens: Minimum token count filter
        max_tokens: Maximum token count filter
        max_samples: Maximum samples (None = unlimited)
        cache_dir: Cache directory for encoded tokens
        device: Device for encoder
        num_workers: DataLoader workers (0 recommended for IterableDataset)

    Returns:
        DataLoader yielding visual token batches [batch, 111, 1280]
    """
    # Resolve FineWeb path with subset
    if fineweb_subset:
        valid_subsets = ["10BT", "100BT", "350BT"]
        if fineweb_subset not in valid_subsets:
            raise ValueError(f"fineweb_subset must be one of {valid_subsets}, got {fineweb_subset}")
        fineweb_data_path = Path(fineweb_path) / "sample" / fineweb_subset
        logger.info(f"Using FineWeb-Edu subset: {fineweb_subset} ({fineweb_data_path})")
    else:
        fineweb_data_path = Path(fineweb_path) / "data"
        logger.info(f"Using full FineWeb-Edu dataset ({fineweb_data_path})")

    # Configure data sources
    if dataset_type == "fineweb":
        data_sources = {
            "fineweb": {"path": str(train_data or fineweb_data_path), "weight": 1.0}
        }
    elif dataset_type == "openwebmath":
        data_sources = {
            "openwebmath": {"path": train_data or openwebmath_path, "weight": 1.0}
        }
    elif dataset_type == "multi":
        data_sources = {
            "fineweb": {"path": str(fineweb_data_path), "weight": fineweb_weight},
            "openwebmath": {"path": openwebmath_path, "weight": openwebmath_weight},
        }
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Create encoder
    encoder = get_global_encoder(model_path=model_path, device=device)

    # Create dataset
    dataset = DirectEncoderDataset(
        data_sources=data_sources,
        encoder=encoder,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        max_samples=max_samples,
        cache_dir=cache_dir,
        encode_batch_size=encode_batch_size,
    )

    # Create dataloader
    def collate_fn(batch):
        return torch.stack([t.float() for t in batch], dim=0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing DirectEncoderDataset...")

    # Test with small sample
    dataloader = create_direct_dataloader(
        dataset_type="fineweb",
        max_samples=16,
        batch_size=4,
        encode_batch_size=4,
    )

    print("\nIterating over dataloader...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}: shape = {batch.shape}, dtype = {batch.dtype}")
        if i >= 2:
            break

    print("\n✓ Test passed!")
