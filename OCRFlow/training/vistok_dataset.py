"""
Visual Token Dataset for OCRFlow Training

Loads text data and converts to visual tokens (vistok) via DeepSeek-OCR server API.
Supports optional disk caching to avoid repeated API calls.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random
import hashlib
import requests
import base64
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class VistokDataset(Dataset):
    """
    Text-only dataset that converts text to visual tokens via server API.

    Expected format:
        data_root/
        └── texts.jsonl  (one JSON per line)
            {"text": "Content here...", "metadata": {...}}

    Or:
        data_root/
        ├── text1.txt
        ├── text2.txt
        └── ...

    Args:
        data_root: Root directory containing text data
        server_url: DeepSeek-OCR server URL (default: http://localhost:8010)
        max_tokens_per_chunk: Maximum tokens per image chunk (default: 1200)
        image_size: Image size for rendering (default: 640)
        max_samples: Maximum number of samples to load (None = all)
        prompts: List of prompt templates to randomly choose from
        cache_dir: Optional directory to cache vistok tensors (None = no caching)
    """

    def __init__(
        self,
        data_root: str,
        server_url: str = "http://localhost:8010",
        max_tokens_per_chunk: int = 1200,
        image_size: int = 640,
        max_samples: Optional[int] = None,
        prompts: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        self.server_url = server_url
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.image_size = image_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching enabled: {self.cache_dir}")

        # Default prompts
        if prompts is None:
            self.prompts = [
                "<image>\n<|grounding|>Convert the document to markdown.",
                "<image>\nFree OCR.",
                "<image>\n<|grounding|>OCR this image.",
                "<image>\n<|grounding|>Extract text from this document.",
            ]
        else:
            self.prompts = prompts

        # Load texts
        self.texts = self._load_texts(max_samples)
        logger.info(f"Loaded {len(self.texts)} text samples from {data_root}")

        # Test server connection
        self._test_server_connection()

    def _load_texts(self, max_samples: Optional[int]) -> List[str]:
        """Load texts from JSONL or text files"""
        texts = []

        # Try JSONL format first
        jsonl_file = self.data_root / "texts.jsonl"
        if jsonl_file.exists():
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if max_samples and idx >= max_samples:
                        break
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if text.strip():
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
            return texts

        # Otherwise, load individual .txt files
        txt_files = sorted(self.data_root.glob("*.txt"))
        for idx, txt_file in enumerate(txt_files):
            if max_samples and idx >= max_samples:
                break
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text.strip():
                        texts.append(text)
            except Exception as e:
                logger.warning(f"Failed to load {txt_file}: {e}")
                continue

        if not texts:
            raise ValueError(
                f"No texts found in {self.data_root}. "
                f"Expected either texts.jsonl or *.txt files"
            )

        return texts

    def _test_server_connection(self):
        """Test connection to server"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.ok:
                logger.info(f"✓ Server connected: {self.server_url}")
            else:
                logger.warning(f"Server responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"✗ Cannot connect to server at {self.server_url}: {e}")
            raise RuntimeError(f"Server not available at {self.server_url}")

    def _get_cache_path(self, text: str) -> Path:
        """Get cache file path for a text (using hash)"""
        if not self.cache_dir:
            return None
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return self.cache_dir / f"vistok_{text_hash}.pt"

    def _load_from_cache(self, text: str) -> Optional[torch.Tensor]:
        """Load cached vistok if available"""
        if not self.cache_dir:
            return None

        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                return None
        return None

    def _save_to_cache(self, text: str, vistok: torch.Tensor):
        """Save vistok to cache"""
        if not self.cache_dir:
            return

        cache_path = self._get_cache_path(text)
        try:
            torch.save(vistok, cache_path)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def _text_to_vistok(self, text: str) -> torch.Tensor:
        """
        Convert text to visual tokens via server API.

        Returns:
            Tensor of shape [num_chunks, 111, 1280]
            Each chunk is 111 tokens (100 visual + 10 newline + 1 separator)
        """
        # Call server API
        response = requests.post(
            f"{self.server_url}/text-to-vistok",
            json={
                "texts": [text],
                "output_format": "json",
                "chunk_size": self.max_tokens_per_chunk,
                "render_width": self.image_size,
                "render_height": self.image_size,
            },
            timeout=60
        )

        if not response.ok:
            raise RuntimeError(f"Server request failed: {response.status_code}")

        result = response.json()

        if not result.get('success'):
            error = result.get('error', 'Unknown error')
            raise RuntimeError(f"Server error: {error}")

        # Decode visual tokens from response
        all_chunks = []
        for text_result in result['results']:
            for chunk in text_result['chunks']:
                # Decode base64 tensor
                vistok_b64 = chunk['visual_tokens_base64']
                embedding_shape = chunk['embedding_shape']

                vistok_bytes = base64.b64decode(vistok_b64)
                vistok_np = np.frombuffer(vistok_bytes, dtype=np.float32)
                vistok_np = vistok_np.reshape(embedding_shape)

                # Convert to tensor [111, 1280]
                vistok = torch.from_numpy(vistok_np)
                all_chunks.append(vistok)

        # Stack chunks [num_chunks, 111, 1280]
        if len(all_chunks) > 0:
            return torch.stack(all_chunks)
        else:
            raise RuntimeError("No visual tokens returned from server")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            (vistok, prompt)
            vistok: Tensor of shape [num_chunks, 111, 1280]
            prompt: Random prompt string
        """
        text = self.texts[idx]

        # Try loading from cache
        vistok = self._load_from_cache(text)

        if vistok is None:
            # Call server API to get vistok
            vistok = self._text_to_vistok(text)

            # Save to cache
            self._save_to_cache(text, vistok)

        # Random prompt selection
        prompt = random.choice(self.prompts)

        return vistok, prompt


def collate_vistok_chunks(batch):
    """
    Custom collate function to handle variable number of chunks per sample.

    Args:
        batch: List of (vistok, prompt) tuples
            vistok: [num_chunks_i, 111, 1280]

    Returns:
        vistok_batch: [total_chunks, 111, 1280] - all chunks concatenated
        prompts: List of prompts repeated for each chunk
        chunk_counts: List of chunk counts per sample
    """
    all_chunks = []
    all_prompts = []
    chunk_counts = []

    for vistok, prompt in batch:
        # vistok: [num_chunks, 111, 1280]
        num_chunks = vistok.shape[0]
        chunk_counts.append(num_chunks)

        # Flatten chunks
        for i in range(num_chunks):
            all_chunks.append(vistok[i])  # [111, 1280]
            all_prompts.append(prompt)

    # Stack all chunks
    vistok_batch = torch.stack(all_chunks)  # [total_chunks, 111, 1280]

    return vistok_batch, all_prompts, chunk_counts


def create_vistok_dataloaders(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    server_url: str = "http://localhost:8010",
    batch_size: int = 16,
    num_workers: int = 4,
    max_tokens_per_chunk: int = 1200,
    image_size: int = 640,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders for vistok training.

    Args:
        train_data_path: Path to training text data
        val_data_path: Path to validation text data (optional)
        server_url: DeepSeek-OCR server URL
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_tokens_per_chunk: Max tokens per image chunk
        image_size: Image size for rendering
        max_train_samples: Max training samples
        max_val_samples: Max validation samples
        cache_dir: Optional cache directory (None = no caching)

    Returns:
        (train_loader, val_loader) tuple
    """
    # Create datasets
    train_dataset = VistokDataset(
        train_data_path,
        server_url=server_url,
        max_tokens_per_chunk=max_tokens_per_chunk,
        image_size=image_size,
        max_samples=max_train_samples,
        cache_dir=cache_dir,
    )

    val_dataset = None
    if val_data_path is not None:
        val_dataset = VistokDataset(
            val_data_path,
            server_url=server_url,
            max_tokens_per_chunk=max_tokens_per_chunk,
            image_size=image_size,
            max_samples=max_val_samples,
            cache_dir=cache_dir,
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_vistok_chunks,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_vistok_chunks,
            pin_memory=True,
        )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    import tempfile
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("Testing VistokDataset...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy text data
        texts = [
            "# Document 1\n\nThis is the first test document.",
            "# Document 2\n\nAnother document for testing.\n\n- Item 1\n- Item 2",
        ]

        # Save as JSONL
        jsonl_path = Path(tmpdir) / "texts.jsonl"
        with open(jsonl_path, 'w') as f:
            for text in texts:
                json.dump({"text": text}, f)
                f.write('\n')

        print(f"Created test data in {tmpdir}")

        # Create dataset (requires server running)
        try:
            dataset = VistokDataset(
                tmpdir,
                server_url="http://localhost:8010",
                cache_dir=None  # No caching for test
            )

            print(f"Dataset size: {len(dataset)}")

            # Test __getitem__
            vistok, prompt = dataset[0]
            print(f"Vistok shape: {vistok.shape}")  # [num_chunks, 111, 1280]
            print(f"Prompt: {prompt}")

            # Test dataloader
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                collate_fn=collate_vistok_chunks
            )
            vistok_batch, prompts, chunk_counts = next(iter(loader))
            print(f"Batch vistok shape: {vistok_batch.shape}")  # [total_chunks, 111, 1280]
            print(f"Chunk counts: {chunk_counts}")

            print("✓ VistokDataset test passed!")

        except Exception as e:
            print(f"✗ Test failed: {e}")
            print("Note: Requires DeepSeek-OCR server running at http://localhost:8010")
