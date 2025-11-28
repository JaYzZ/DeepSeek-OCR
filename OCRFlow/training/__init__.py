"""
OCRFlow Training Module

Contains training utilities:
- dataset: OCR dataset loaders
- fineweb_dataset: FineWeb-Edu dataset for visual token training
- openwebmath_dataset: OpenWebMath dataset for math content
- multi_dataset: Combined multi-source dataset with mixing
- direct_encoder_dataset: Direct encoder integration (22x faster than server)
- loss: Flow matching loss functions
"""

from .loss import RectifiedFlowLoss
from .dataset import OCRDataset
from .fineweb_dataset import FineWebEduVistokDataset, create_fineweb_dataloaders
from .openwebmath_dataset import OpenWebMathVistokDataset, create_openwebmath_dataloaders
from .multi_dataset import (
    MultiDatasetVistok,
    InterleavedMultiDataset,
    create_multi_dataloaders,
    DatasetConfig,
)
from .direct_encoder_dataset import (
    DirectVisionEncoder,
    DirectEncoderDataset,
    create_direct_dataloader,
    get_global_encoder,
)

__all__ = [
    "RectifiedFlowLoss",
    "OCRDataset",
    "FineWebEduVistokDataset",
    "create_fineweb_dataloaders",
    "OpenWebMathVistokDataset",
    "create_openwebmath_dataloaders",
    "MultiDatasetVistok",
    "InterleavedMultiDataset",
    "create_multi_dataloaders",
    "DatasetConfig",
    # Direct encoder (22x faster)
    "DirectVisionEncoder",
    "DirectEncoderDataset",
    "create_direct_dataloader",
    "get_global_encoder",
]
