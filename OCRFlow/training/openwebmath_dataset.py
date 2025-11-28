"""
OpenWebMath Dataset Loader for Visual Token Training

Loads mathematical text from OpenWebMath dataset and converts to visual tokens via DeepSeek OCR server.
OpenWebMath contains high-quality mathematical web content with LaTeX equations.

OpenWebMath structure:
- Parquet files with columns: text, url, date
- ~14.7B tokens of mathematical content
- Extracted from Common Crawl with math-focused filtering

Usage:
    from OCRFlow.training.openwebmath_dataset import create_openwebmath_dataloaders

    dataloader = create_openwebmath_dataloaders(
        data_root="/share/project/xiyan/huggingface/open-web-math/open-web-math",
        server_url="http://localhost:8010",
        cache_dir="./vistok_cache_math",
    )
"""

import torch
from torch.utils.data import IterableDataset
from pathlib import Path
import json
import numpy as np
import requests
import hashlib
import pickle
from typing import Optional, List
import logging
import pandas as pd
import random

logger = logging.getLogger(__name__)


def estimate_token_count(text: str) -> int:
    """
    Estimate token count from text length.
    Rough approximation: ~4 characters per token for English text.
    Math content may have different ratios due to LaTeX.
    """
    return len(text) // 4


class OpenWebMathVistokDataset(IterableDataset):
    """
    Streaming dataset for OpenWebMath → Visual Tokens

    Designed for mathematical content pretraining:
    - Streams from parquet files (memory efficient)
    - Caches visual tokens to disk
    - Filters by estimated token count
    - Handles LaTeX and mathematical notation

    Args:
        data_root: Root directory containing OpenWebMath data/
        server_url: DeepSeek OCR server URL
        cache_dir: Directory to cache visual tokens
        min_tokens: Minimum estimated token count (default: 100)
        max_tokens: Maximum estimated token count (default: 1200)
        chunk_size: Tokens per visual chunk (default: 1000)
        shuffle: Shuffle parquet files
        max_samples: Maximum samples to load (None = all)
    """

    def __init__(
        self,
        data_root: str,
        server_url: str = "http://localhost:8010",
        cache_dir: Optional[str] = None,
        min_tokens: int = 100,
        max_tokens: int = 1200,
        chunk_size: int = 1000,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.server_url = server_url.rstrip('/')
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.max_samples = max_samples

        # Setup cache with dataset-specific prefix
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching enabled: {self.cache_dir}")

        # Find all parquet files
        self.parquet_files = self._find_parquet_files()
        logger.info(f"Found {len(self.parquet_files)} OpenWebMath parquet files")

        # Test server connection
        self._test_server_connection()

    def _find_parquet_files(self) -> List[Path]:
        """Find all parquet files in data directory"""
        # OpenWebMath structure: data_root/data/*.parquet or data_root/*.parquet
        data_dir = self.data_root / "data"
        if data_dir.exists():
            parquet_files = list(data_dir.glob("**/*.parquet"))
        else:
            # Try root directory
            parquet_files = list(self.data_root.glob("**/*.parquet"))

        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.data_root}")

        # Sort for reproducibility
        parquet_files = sorted(parquet_files)

        return parquet_files

    def _test_server_connection(self):
        """Test connection to DeepSeek OCR server"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.ok:
                logger.info(f"✓ DeepSeek OCR server connected: {self.server_url}")
            else:
                logger.warning(f"Server returned status {response.status_code}")
        except Exception as e:
            logger.error(f"✗ Cannot connect to server: {e}")
            raise RuntimeError(f"Server not available at {self.server_url}")

    def _get_text_id(self, text: str, url: str = "") -> str:
        """Generate unique ID for text (OpenWebMath may not have 'id' column)"""
        # Use hash of text + url for unique ID
        content = f"{url}:{text[:500]}"  # Use first 500 chars to avoid long hashes
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, text_id: str) -> Optional[Path]:
        """Get cache file path for a text ID"""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"owm_{text_id}.pkl"

    def _load_from_cache(self, text_id: str) -> Optional[torch.Tensor]:
        """Load cached visual tokens"""
        if not self.cache_dir:
            return None

        cache_path = self._get_cache_path(text_id)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    return data['visual_tokens']
            except Exception as e:
                logger.debug(f"Failed to load cache {cache_path}: {e}")
                return None
        return None

    def _save_to_cache(self, text_id: str, visual_tokens: torch.Tensor):
        """Save visual tokens to cache"""
        if not self.cache_dir:
            return

        cache_path = self._get_cache_path(text_id)
        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'visual_tokens': visual_tokens,
                        'text_id': text_id,
                    }, f)
            except Exception as e:
                logger.debug(f"Failed to save cache {cache_path}: {e}")

    def _fetch_visual_tokens(self, text: str) -> torch.Tensor:
        """
        Fetch visual tokens from DeepSeek OCR server

        Returns:
            Visual tokens [111, 1280] for single chunk
            or [num_chunks, 111, 1280] for multiple chunks
        """
        try:
            response = requests.post(
                f"{self.server_url}/text-to-vistok",
                json={
                    "texts": [text],
                    "chunk_size": self.chunk_size,
                    "render_width": 640,
                    "render_height": 640,
                    "font_size": 18,
                    "include_rendered_images": False,
                    "skip_embeddings": False,
                    "output_format": "binary",
                },
                timeout=30,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Server returned status {response.status_code}")

            # Parse binary response
            shape_str = response.headers.get('X-Tensor-Shape')
            if not shape_str:
                raise RuntimeError("Server did not return tensor shape")

            shape = json.loads(shape_str)

            # Convert binary to numpy
            tensor_np = np.frombuffer(response.content, dtype=np.float32)
            tensor_np = tensor_np.reshape(shape)

            # Convert to torch
            visual_tokens = torch.from_numpy(tensor_np).to(torch.float32)

            # If single chunk, squeeze to [111, 1280]
            if visual_tokens.shape[0] == 1:
                visual_tokens = visual_tokens.squeeze(0)

            return visual_tokens

        except Exception as e:
            logger.error(f"Failed to fetch visual tokens: {e}")
            raise

    def _process_parquet_file(self, parquet_path: Path):
        """Process a single parquet file and yield samples"""
        try:
            df = pd.read_parquet(parquet_path)

            # OpenWebMath columns: text, url, date (may vary)
            if 'text' not in df.columns:
                logger.warning(f"No 'text' column in {parquet_path}, skipping")
                return

            # Estimate token counts if not present
            if 'token_count' not in df.columns:
                df['token_count'] = df['text'].apply(estimate_token_count)

            # Filter by token count
            df = df[
                (df['token_count'] >= self.min_tokens) &
                (df['token_count'] <= self.max_tokens)
            ]

            # Optionally shuffle rows
            if self.shuffle:
                df = df.sample(frac=1.0).reset_index(drop=True)

            for idx, row in df.iterrows():
                text = row['text']
                url = row.get('url', '')

                # Generate unique ID
                text_id = self._get_text_id(text, url)

                # Try cache first
                visual_tokens = self._load_from_cache(text_id)

                # Fetch from server if not cached
                if visual_tokens is None:
                    try:
                        visual_tokens = self._fetch_visual_tokens(text)
                        # Cache for future use
                        self._save_to_cache(text_id, visual_tokens)
                    except Exception as e:
                        logger.warning(f"Skipping sample {text_id[:8]}...: {e}")
                        continue

                # Yield visual tokens
                if visual_tokens.dim() == 3:
                    for chunk_idx in range(visual_tokens.shape[0]):
                        yield visual_tokens[chunk_idx]
                else:
                    yield visual_tokens

        except Exception as e:
            logger.error(f"Error processing {parquet_path}: {e}")

    def __iter__(self):
        """Iterate over dataset"""
        parquet_files = self.parquet_files.copy()
        if self.shuffle:
            random.shuffle(parquet_files)

        sample_count = 0

        for parquet_path in parquet_files:
            logger.info(f"Processing OpenWebMath: {parquet_path.name}...")

            for visual_tokens in self._process_parquet_file(parquet_path):
                yield visual_tokens

                sample_count += 1
                if self.max_samples and sample_count >= self.max_samples:
                    logger.info(f"Reached max_samples limit: {self.max_samples}")
                    return


def create_openwebmath_dataloaders(
    data_root: str,
    server_url: str = "http://localhost:8010",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    min_tokens: int = 100,
    max_tokens: int = 1200,
    chunk_size: int = 1000,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
):
    """
    Create dataloader for OpenWebMath dataset

    Args:
        data_root: Root directory of OpenWebMath
        server_url: DeepSeek OCR server URL
        cache_dir: Cache directory for visual tokens
        batch_size: Batch size
        num_workers: Number of workers (0 recommended for IterableDataset)
        min_tokens: Minimum estimated token count
        max_tokens: Maximum estimated token count
        chunk_size: Tokens per visual chunk
        shuffle: Shuffle files
        max_samples: Maximum samples to load

    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader

    dataset = OpenWebMathVistokDataset(
        data_root=data_root,
        server_url=server_url,
        cache_dir=cache_dir,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        chunk_size=chunk_size,
        shuffle=shuffle,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing OpenWebMathVistokDataset...")

    data_root = "/share/project/xiyan/huggingface/open-web-math/open-web-math"
    cache_dir = "./test_cache_owm"

    try:
        dataset = OpenWebMathVistokDataset(
            data_root=data_root,
            server_url="http://localhost:8010",
            cache_dir=cache_dir,
            min_tokens=100,
            max_tokens=500,
            max_samples=5,
        )

        print("\nIterating over dataset...")
        for idx, vistok in enumerate(dataset):
            print(f"Sample {idx + 1}: shape = {vistok.shape}")
            if idx >= 4:
                break

        print("\n✓ OpenWebMath dataset test passed!")

    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        print("Note: Requires DeepSeek-OCR server running at http://localhost:8010")
