"""
Rolling Cache Dataset for Maximum GPU Utilization

Architecture:
- Encoder runs continuously in background on GPU:0, filling a cache
- Training samples randomly from cache on GPU:1 (never waits for encoder)
- Cache is constantly refreshed with new encodings (rolling)
- Variable chunking (50-900 words) provides natural randomness

Memory cache: Fast, GPU memory resident (~5.7GB for 10K pairs)

Benefits:
- Training runs at full GPU speed (~500+ pairs/s)
- Encoder runs at its natural speed (~65 samples/s)
- No blocking between encoder and training
- Randomness from: variable chunking + random cache sampling + constant refresh

Usage:
    from OCRFlow.training.rolling_cache_dataset import create_rolling_cache_dataloader

    cache, encoder, dataloader = create_rolling_cache_dataloader(
        cache_size=10000,
        encoder_device="cuda:0",
        training_device="cuda:1",
    )

    encoder.start()  # Start background encoding

    for inputs, targets in dataloader:
        # Training loop
        loss = model(inputs, targets)
        ...

    encoder.stop()
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import threading
import time
import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterator
import pandas as pd
from project_paths import hf_path

logger = logging.getLogger(__name__)


def count_words(text: str) -> int:
    return len(text.split())


def chunk_text_variable(
    text: str,
    min_words: int = 50,
    max_words: int = 900,
    target_mean: int = 500,
) -> List[str]:
    """
    Chunk text with variable length for randomness.
    Each chunk is randomly sized between min_words and max_words.
    """
    words = text.split()
    total_words = len(words)

    if total_words < min_words * 2:
        return []

    chunks = []
    current_pos = 0

    while current_pos < total_words:
        # Random chunk size with bias towards target_mean
        chunk_size = int(random.triangular(min_words, max_words, target_mean))
        chunk_size = max(min_words, min(max_words, chunk_size))

        remaining = total_words - current_pos
        if remaining < min_words:
            break

        if remaining - chunk_size < min_words and remaining <= max_words:
            chunk_size = remaining

        end_pos = min(current_pos + chunk_size, total_words)
        chunks.append(" ".join(words[current_pos:end_pos]))
        current_pos = end_pos

    return chunks if len(chunks) >= 2 else []


class RollingTokenCache:
    """
    Thread-safe rolling cache for visual tokens in GPU memory.

    - Encoder writes new tokens continuously (overwrites old)
    - Training reads random samples instantly
    - Circular buffer - old tokens replaced as new ones come in
    """

    def __init__(
        self,
        cache_size: int = 10000,
        token_dim: int = 1280,
        token_len: int = 111,
        device: str = "cuda:1",
    ):
        self.cache_size = cache_size
        self.token_dim = token_dim
        self.token_len = token_len
        self.device = device

        # Pre-allocate cache tensors on training device
        self.input_cache = torch.zeros(
            cache_size, token_len, token_dim,
            dtype=torch.float16, device=device
        )
        self.target_cache = torch.zeros(
            cache_size, token_len, token_dim,
            dtype=torch.float16, device=device
        )

        # Cache state
        self.write_idx = 0
        self.valid_count = 0

        # Thread safety for writes
        self.lock = threading.Lock()

        # Stats
        self.total_written = 0
        self.total_read = 0

        logger.info(
            f"RollingTokenCache initialized: {cache_size} pairs, "
            f"~{cache_size * 2 * token_len * token_dim * 2 / 1e9:.1f}GB on {device}"
        )

    def add_pairs(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor):
        """
        Add batch of token pairs to cache (rolling - overwrites old).

        Args:
            input_tokens: [batch, 111, 1280] on any device
            target_tokens: [batch, 111, 1280] on any device
        """
        batch_size = len(input_tokens)

        with self.lock:
            for i in range(batch_size):
                idx = self.write_idx % self.cache_size

                # Copy to cache (move to training device)
                self.input_cache[idx] = input_tokens[i].half().to(self.device)
                self.target_cache[idx] = target_tokens[i].half().to(self.device)

                self.write_idx += 1

            self.valid_count = min(self.write_idx, self.cache_size)
            self.total_written += batch_size

    def sample_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample random batch from cache.
        Non-blocking - returns immediately with random samples.

        Returns:
            (inputs, targets) each [batch, 111, 1280] or None if cache empty
        """
        if self.valid_count < batch_size:
            return None

        # Random indices (lock-free read)
        indices = torch.randint(0, self.valid_count, (batch_size,), device=self.device)

        # Direct tensor access
        inputs = self.input_cache[indices].float()
        targets = self.target_cache[indices].float()

        self.total_read += batch_size

        return inputs, targets

    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "valid_count": self.valid_count,
            "cache_size": self.cache_size,
            "fill_ratio": self.valid_count / self.cache_size,
            "total_written": self.total_written,
            "total_read": self.total_read,
            "refresh_rate": self.total_written / max(1, self.valid_count),
        }


class EncoderWorker:
    """
    Background encoder that continuously fills the cache.
    Runs on separate GPU, writes to cache on training GPU.
    """

    def __init__(
        self,
        cache: RollingTokenCache,
        data_sources: Dict[str, Dict],
        encoder_device: str = "cuda:0",
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        encode_batch_size: int = 24,
        num_render_workers: int = 48,
        min_words: int = 50,
        max_words: int = 900,
        target_mean: int = 500,
        min_doc_words: int = 100,
        augment_preset: str = "medium",
    ):
        self.cache = cache
        self.data_sources = data_sources
        self.encoder_device = encoder_device
        self.model_path = model_path
        self.encode_batch_size = encode_batch_size
        self.num_render_workers = num_render_workers
        self.min_words = min_words
        self.max_words = max_words
        self.target_mean = target_mean
        self.min_doc_words = min_doc_words
        self.augment_preset = augment_preset

        self.encoder = None
        self.stop_event = threading.Event()
        self.thread = None

        # Find parquet files
        self.parquet_files = {}
        for name, config in data_sources.items():
            path = Path(config["path"])
            data_dir = path / "data" if (path / "data").exists() else path
            files = list(data_dir.glob("**/*.parquet"))
            self.parquet_files[name] = sorted(files)
            logger.info(f"Found {len(files)} parquet files for {name}")

    def _iter_documents(self, source_name: str) -> Iterator[str]:
        """Iterate over documents from parquet files"""
        files = self.parquet_files.get(source_name, [])
        random.shuffle(files)

        for parquet_path in files:
            try:
                df = pd.read_parquet(parquet_path)
                if "text" not in df.columns:
                    continue

                df = df.sample(frac=1.0).reset_index(drop=True)
                for _, row in df.iterrows():
                    text = row["text"]
                    if count_words(text) >= self.min_doc_words:
                        yield text
            except Exception as e:
                continue

    def _collect_pairs(self, source_iters, source_names) -> List[Tuple[str, str]]:
        """Collect batch of text pairs with variable chunking"""
        pairs = []

        while len(pairs) < self.encode_batch_size:
            source = random.choice(source_names)

            try:
                doc_text = next(source_iters[source])
            except StopIteration:
                source_iters[source] = self._iter_documents(source)
                try:
                    doc_text = next(source_iters[source])
                except StopIteration:
                    continue

            # Variable length chunking for randomness
            chunks = chunk_text_variable(
                doc_text,
                min_words=self.min_words,
                max_words=self.max_words,
                target_mean=self.target_mean,
            )

            for i in range(len(chunks) - 1):
                pairs.append((chunks[i], chunks[i + 1]))
                if len(pairs) >= self.encode_batch_size:
                    break

        return pairs[:self.encode_batch_size]

    def _encoder_loop(self):
        """Main encoder loop - runs continuously"""
        from PIL import Image
        from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
        from Renderer.pil_renderer import PILRenderer, render_to_pil
        try:
            from Renderer import VelloRenderer  # type: ignore
        except Exception:
            VelloRenderer = None

        logger.info(f"Initializing DPSK OCR encoder on {self.encoder_device}...")
        self.encoder = DPSKOCREncoder(
            model_path=self.model_path,
            device=self.encoder_device,
            dtype=torch.bfloat16,
        )

        # Local batch renderer (Vello if available, else PIL).
        vello = None
        if VelloRenderer is not None:
            try:
                vello = VelloRenderer(width=640, height=640, padding=20)
                logger.info("EncoderWorker using VelloRenderer")
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

        source_iters = {name: self._iter_documents(name) for name in self.data_sources}
        source_names = list(self.data_sources.keys())

        logger.info("Encoder loop started - continuously filling cache")

        encode_times = []

        while not self.stop_event.is_set():
            try:
                start = time.time()

                # Collect text pairs with variable chunking
                pairs = self._collect_pairs(source_iters, source_names)

                # Flatten to texts
                texts = []
                for inp, tgt in pairs:
                    texts.extend([inp, tgt])

                # Render then encode.
                images = render_texts([t[:6000] for t in texts])

                if self.augment_preset != "none":
                    from OCRFlow.utils.image_augmentation import augment_batch, get_augment_config
                    aug_config = get_augment_config(self.augment_preset)
                    arrays = augment_batch(images, **aug_config)
                    images = [Image.fromarray(arr) for arr in arrays]

                tokens = self.encoder.encode_images(images, return_global=False, return_local=True)

                # Split into input/target pairs
                input_tokens = torch.stack([tokens[i] for i in range(0, len(tokens), 2)], dim=0)
                target_tokens = torch.stack([tokens[i] for i in range(1, len(tokens), 2)], dim=0)

                # Add to cache (rolling - overwrites old)
                self.cache.add_pairs(input_tokens, target_tokens)

                encode_time = time.time() - start
                encode_times.append(encode_time)

                # Log progress periodically
                if len(encode_times) % 10 == 0:
                    avg_time = sum(encode_times[-10:]) / 10
                    rate = self.encode_batch_size / avg_time
                    stats = self.cache.get_stats()
                    logger.info(
                        f"Encoder: {rate:.1f} pairs/s, "
                        f"cache fill: {stats['fill_ratio']*100:.1f}%, "
                        f"total written: {stats['total_written']}, "
                        f"refreshed: {stats['refresh_rate']:.1f}x"
                    )

            except Exception as e:
                logger.error(f"Encoder error: {e}")
                if self.stop_event.is_set():
                    break
                time.sleep(1)

        logger.info("Encoder loop stopped")

    def start(self):
        """Start encoder thread"""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self.thread.start()
        logger.info("Encoder worker started")

    def stop(self):
        """Stop encoder thread"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Encoder worker stopped")


class RollingCacheDataset(IterableDataset):
    """
    Dataset that samples from rolling cache.
    Never blocks - if cache is empty, waits briefly.
    """

    def __init__(
        self,
        cache: RollingTokenCache,
        batch_size: int = 32,
        min_cache_fill: float = 0.1,
    ):
        self.cache = cache
        self.batch_size = batch_size
        self.min_cache_fill = min_cache_fill

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Wait for cache to fill initially
        logger.info(f"Waiting for cache to fill to {self.min_cache_fill*100:.0f}%...")
        while self.cache.valid_count < self.cache.cache_size * self.min_cache_fill:
            time.sleep(0.5)
        logger.info(f"Cache ready: {self.cache.valid_count} pairs")

        while True:
            batch = self.cache.sample_batch(self.batch_size)
            if batch is not None:
                yield batch
            else:
                time.sleep(0.01)


def create_rolling_cache_dataloader(
    dataset_type: str = "fineweb",
    fineweb_path: str = str(hf_path("HuggingFaceFW", "fineweb-edu")),
    fineweb_subset: str = "10BT",
    cache_size: int = 10000,
    encoder_device: str = "cuda:0",
    training_device: str = "cuda:1",
    encode_batch_size: int = 64,
    training_batch_size: int = 64,
    num_render_workers: int = 48,
    min_words: int = 50,
    max_words: int = 900,
    min_cache_fill: float = 0.1,
    augment_preset: str = "medium",
) -> Tuple[RollingTokenCache, EncoderWorker, DataLoader]:
    """
    Create rolling cache dataloader.

    Args:
        dataset_type: Type of dataset (fineweb, etc.)
        fineweb_path: Path to FineWeb dataset
        fineweb_subset: Subset to use (e.g., "10BT")
        cache_size: Number of token pairs to cache (rolling)
        encoder_device: Device for vision encoder
        training_device: Device for training
        encode_batch_size: Batch size for encoding
        training_batch_size: Batch size for training
        num_render_workers: Number of parallel rendering workers
        min_words: Minimum words per chunk
        max_words: Maximum words per chunk
        min_cache_fill: Minimum cache fill ratio before training starts
        augment_preset: Augmentation intensity ("none", "light", "medium", "heavy")

    Returns:
        cache: RollingTokenCache instance
        encoder: EncoderWorker instance (call .start() to begin)
        dataloader: DataLoader for training
    """
    # Resolve path
    if fineweb_subset:
        data_path = Path(fineweb_path) / "sample" / fineweb_subset
    else:
        data_path = Path(fineweb_path) / "data"

    data_sources = {"fineweb": {"path": str(data_path), "weight": 1.0}}

    # Create cache on training device
    cache = RollingTokenCache(
        cache_size=cache_size,
        device=training_device,
    )

    # Create encoder worker
    encoder = EncoderWorker(
        cache=cache,
        data_sources=data_sources,
        encoder_device=encoder_device,
        encode_batch_size=encode_batch_size,
        num_render_workers=num_render_workers,
        min_words=min_words,
        max_words=max_words,
        augment_preset=augment_preset,
    )

    # Create dataset
    dataset = RollingCacheDataset(
        cache=cache,
        batch_size=training_batch_size,
        min_cache_fill=min_cache_fill,
    )

    # DataLoader (batch_size=1 since dataset already batches)
    def collate_fn(batch):
        inputs, targets = batch[0]
        return inputs, targets

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return cache, encoder, dataloader


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing RollingCacheDataset...")
    print()

    cache, encoder, dataloader = create_rolling_cache_dataloader(
        cache_size=1000,
        encoder_device="cuda:0",
        training_device="cuda:1",
        encode_batch_size=64,
        training_batch_size=64,
        min_cache_fill=0.1,
    )

    # Start encoder
    encoder.start()

    print("Running training simulation...")
    start = time.time()
    total_pairs = 0

    for i, (inputs, targets) in enumerate(dataloader):
        total_pairs += len(inputs)

        if i % 10 == 0:
            elapsed = time.time() - start
            stats = cache.get_stats()
            print(f"Step {i}: {total_pairs} pairs, {total_pairs/elapsed:.1f} pairs/s, cache: {stats['fill_ratio']*100:.1f}%")

        if i >= 100:
            break

    elapsed = time.time() - start
    print(f"\nTotal: {total_pairs} pairs in {elapsed:.1f}s = {total_pairs/elapsed:.1f} pairs/s")

    encoder.stop()
    print("Done!")
