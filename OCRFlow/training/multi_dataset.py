"""
Multi-Dataset Loader for Visual Token Training

Combines multiple data sources (FineWeb-Edu, OpenWebMath, etc.) with configurable
mixing ratios for pretraining. Supports curriculum learning and domain balancing.

Usage:
    from OCRFlow.training.multi_dataset import create_multi_dataloaders

    dataloader = create_multi_dataloaders(
        datasets={
            "fineweb": {
                "path": "/path/to/fineweb-edu",
                "weight": 0.7,  # 70% of samples
            },
            "openwebmath": {
                "path": "/path/to/open-web-math",
                "weight": 0.3,  # 30% of samples
            },
        },
        server_url="http://localhost:8010",
        cache_dir="./vistok_cache",
    )
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import random
import logging
from typing import Optional, Dict, List, Iterator, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    path: str
    weight: float = 1.0
    min_tokens: int = 100
    max_tokens: int = 1200
    max_samples: Optional[int] = None


# Default dataset configurations
DEFAULT_CONFIGS = {
    "fineweb": DatasetConfig(
        path="/share/project/xiyan/huggingface/HuggingFaceFW/fineweb-edu",
        weight=0.7,
        min_tokens=100,
        max_tokens=1200,
    ),
    "openwebmath": DatasetConfig(
        path="/share/project/xiyan/huggingface/open-web-math/open-web-math",
        weight=0.3,
        min_tokens=100,
        max_tokens=1200,
    ),
}


class MultiDatasetVistok(IterableDataset):
    """
    Multi-source dataset that combines FineWeb-Edu, OpenWebMath, etc.

    Implements weighted sampling across datasets with on-the-fly mixing.

    Args:
        datasets: Dict of dataset configs {name: DatasetConfig}
        server_url: DeepSeek OCR server URL
        cache_dir: Base cache directory (subdirs created per dataset)
        chunk_size: Tokens per visual chunk
        shuffle: Shuffle within datasets
        max_samples: Total maximum samples (None = unlimited)
    """

    def __init__(
        self,
        datasets: Dict[str, DatasetConfig],
        server_url: str = "http://localhost:8010",
        cache_dir: Optional[str] = None,
        chunk_size: int = 1000,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.datasets_config = datasets
        self.server_url = server_url
        self.base_cache_dir = Path(cache_dir) if cache_dir else None
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.max_samples = max_samples

        # Normalize weights
        total_weight = sum(cfg.weight for cfg in datasets.values())
        self.weights = {name: cfg.weight / total_weight for name, cfg in datasets.items()}

        # Create iterators lazily
        self._iterators: Dict[str, Iterator] = {}

        logger.info(f"MultiDataset initialized with {len(datasets)} sources:")
        for name, cfg in datasets.items():
            logger.info(f"  - {name}: weight={self.weights[name]:.2f}, path={cfg.path}")

    def _get_cache_dir(self, dataset_name: str) -> Optional[Path]:
        """Get cache directory for a specific dataset"""
        if not self.base_cache_dir:
            return None
        cache_dir = self.base_cache_dir / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _create_dataset_iterator(self, name: str) -> Iterator:
        """Create iterator for a specific dataset"""
        cfg = self.datasets_config[name]
        cache_dir = self._get_cache_dir(name)

        if name == "fineweb":
            from .fineweb_dataset import FineWebEduVistokDataset
            dataset = FineWebEduVistokDataset(
                data_root=cfg.path,
                server_url=self.server_url,
                cache_dir=str(cache_dir) if cache_dir else None,
                min_tokens=cfg.min_tokens,
                max_tokens=cfg.max_tokens,
                chunk_size=self.chunk_size,
                shuffle=self.shuffle,
                max_samples=cfg.max_samples,
            )
        elif name == "openwebmath":
            from .openwebmath_dataset import OpenWebMathVistokDataset
            dataset = OpenWebMathVistokDataset(
                data_root=cfg.path,
                server_url=self.server_url,
                cache_dir=str(cache_dir) if cache_dir else None,
                min_tokens=cfg.min_tokens,
                max_tokens=cfg.max_tokens,
                chunk_size=self.chunk_size,
                shuffle=self.shuffle,
                max_samples=cfg.max_samples,
            )
        else:
            raise ValueError(f"Unknown dataset type: {name}")

        return iter(dataset)

    def _get_next_from_dataset(self, name: str) -> Optional[torch.Tensor]:
        """Get next sample from a dataset, reinitializing if exhausted"""
        if name not in self._iterators:
            self._iterators[name] = self._create_dataset_iterator(name)

        try:
            return next(self._iterators[name])
        except StopIteration:
            # Reinitialize iterator for continuous training
            logger.info(f"Dataset '{name}' exhausted, reinitializing...")
            self._iterators[name] = self._create_dataset_iterator(name)
            try:
                return next(self._iterators[name])
            except StopIteration:
                return None

    def __iter__(self):
        """Iterate with weighted sampling across datasets"""
        dataset_names = list(self.weights.keys())
        weights = [self.weights[name] for name in dataset_names]

        sample_count = 0

        while True:
            # Weighted random selection of dataset
            selected = random.choices(dataset_names, weights=weights, k=1)[0]

            # Get sample from selected dataset
            sample = self._get_next_from_dataset(selected)

            if sample is not None:
                yield sample
                sample_count += 1

                if self.max_samples and sample_count >= self.max_samples:
                    logger.info(f"Reached max_samples: {self.max_samples}")
                    return
            else:
                # All datasets exhausted
                logger.warning(f"Dataset '{selected}' returned None")
                break


class InterleavedMultiDataset(IterableDataset):
    """
    Alternative: Round-robin interleaving across datasets.

    Instead of weighted sampling, takes samples in strict rotation.
    Better for ensuring all datasets are represented evenly in each batch.
    """

    def __init__(
        self,
        datasets: Dict[str, DatasetConfig],
        server_url: str = "http://localhost:8010",
        cache_dir: Optional[str] = None,
        chunk_size: int = 1000,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        samples_per_dataset: int = 1,  # Samples from each dataset before switching
    ):
        self.datasets_config = datasets
        self.server_url = server_url
        self.base_cache_dir = Path(cache_dir) if cache_dir else None
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.samples_per_dataset = samples_per_dataset

        self._iterators: Dict[str, Iterator] = {}

    def _get_cache_dir(self, dataset_name: str) -> Optional[Path]:
        if not self.base_cache_dir:
            return None
        cache_dir = self.base_cache_dir / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _create_dataset_iterator(self, name: str) -> Iterator:
        cfg = self.datasets_config[name]
        cache_dir = self._get_cache_dir(name)

        if name == "fineweb":
            from .fineweb_dataset import FineWebEduVistokDataset
            dataset = FineWebEduVistokDataset(
                data_root=cfg.path,
                server_url=self.server_url,
                cache_dir=str(cache_dir) if cache_dir else None,
                min_tokens=cfg.min_tokens,
                max_tokens=cfg.max_tokens,
                chunk_size=self.chunk_size,
                shuffle=self.shuffle,
                max_samples=cfg.max_samples,
            )
        elif name == "openwebmath":
            from .openwebmath_dataset import OpenWebMathVistokDataset
            dataset = OpenWebMathVistokDataset(
                data_root=cfg.path,
                server_url=self.server_url,
                cache_dir=str(cache_dir) if cache_dir else None,
                min_tokens=cfg.min_tokens,
                max_tokens=cfg.max_tokens,
                chunk_size=self.chunk_size,
                shuffle=self.shuffle,
                max_samples=cfg.max_samples,
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

        return iter(dataset)

    def __iter__(self):
        dataset_names = list(self.datasets_config.keys())
        sample_count = 0

        while True:
            for name in dataset_names:
                if name not in self._iterators:
                    self._iterators[name] = self._create_dataset_iterator(name)

                for _ in range(self.samples_per_dataset):
                    try:
                        sample = next(self._iterators[name])
                        yield sample
                        sample_count += 1

                        if self.max_samples and sample_count >= self.max_samples:
                            return
                    except StopIteration:
                        # Reinitialize
                        self._iterators[name] = self._create_dataset_iterator(name)
                        try:
                            sample = next(self._iterators[name])
                            yield sample
                            sample_count += 1
                        except StopIteration:
                            break


def create_multi_dataloaders(
    datasets: Optional[Dict[str, Any]] = None,
    server_url: str = "http://localhost:8010",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    chunk_size: int = 1000,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    interleaved: bool = False,
    samples_per_dataset: int = 1,
):
    """
    Create dataloader combining multiple datasets.

    Args:
        datasets: Dict of dataset configs. If None, uses defaults (FineWeb 70% + OpenWebMath 30%)
            Format: {
                "name": {
                    "path": "/path/to/data",
                    "weight": 0.7,  # Sampling weight
                    "min_tokens": 100,
                    "max_tokens": 1200,
                    "max_samples": None,
                }
            }
        server_url: DeepSeek OCR server URL
        cache_dir: Base cache directory
        batch_size: Batch size
        chunk_size: Tokens per visual chunk
        shuffle: Shuffle within datasets
        max_samples: Total maximum samples
        interleaved: Use round-robin instead of weighted sampling
        samples_per_dataset: For interleaved, samples per dataset before switching

    Returns:
        DataLoader

    Example:
        # Default: 70% FineWeb-Edu + 30% OpenWebMath
        dataloader = create_multi_dataloaders()

        # Custom mix
        dataloader = create_multi_dataloaders(
            datasets={
                "fineweb": {"path": "/path/to/fineweb", "weight": 0.6},
                "openwebmath": {"path": "/path/to/openwebmath", "weight": 0.4},
            }
        )
    """
    # Use defaults if not specified
    if datasets is None:
        datasets = {
            "fineweb": {
                "path": "/share/project/xiyan/huggingface/HuggingFaceFW/fineweb-edu",
                "weight": 0.7,
            },
            "openwebmath": {
                "path": "/share/project/xiyan/huggingface/open-web-math/open-web-math",
                "weight": 0.3,
            },
        }

    # Convert to DatasetConfig objects
    dataset_configs = {}
    for name, cfg in datasets.items():
        if isinstance(cfg, DatasetConfig):
            dataset_configs[name] = cfg
        else:
            dataset_configs[name] = DatasetConfig(
                path=cfg.get("path", DEFAULT_CONFIGS.get(name, DatasetConfig(path="")).path),
                weight=cfg.get("weight", 1.0),
                min_tokens=cfg.get("min_tokens", 100),
                max_tokens=cfg.get("max_tokens", 1200),
                max_samples=cfg.get("max_samples"),
            )

    # Create dataset
    if interleaved:
        dataset = InterleavedMultiDataset(
            datasets=dataset_configs,
            server_url=server_url,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            shuffle=shuffle,
            max_samples=max_samples,
            samples_per_dataset=samples_per_dataset,
        )
    else:
        dataset = MultiDatasetVistok(
            datasets=dataset_configs,
            server_url=server_url,
            cache_dir=cache_dir,
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

    print("Testing MultiDatasetVistok...")
    print("=" * 60)

    # Test with default config (requires server)
    try:
        dataloader = create_multi_dataloaders(
            cache_dir="./test_cache_multi",
            max_samples=10,
            batch_size=2,
        )

        print("\nIterating over multi-dataset...")
        for idx, batch in enumerate(dataloader):
            print(f"Batch {idx + 1}: shape = {batch.shape}")
            if idx >= 4:
                break

        print("\n✓ Multi-dataset test passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Note: Requires DeepSeek-OCR server and data paths to exist")
