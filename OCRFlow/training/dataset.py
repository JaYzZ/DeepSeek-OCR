"""
Dataset Loader for OCRFlow Training

Simple dataset for loading images and captions for OCRFlow training.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torchvision.transforms as transforms


class OCRDataset(Dataset):
    """
    Dataset for OCRFlow training.

    Expects data in format:
        - images/: folder with images
        - captions.json: {
            "image1.jpg": "caption text",
            ...
          }

    Args:
        data_root: Root directory containing images/ and captions.json
        image_size: Size to resize images to
        transform: Optional additional transforms
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 640,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / "images"
        self.image_size = image_size

        # Load captions
        caption_file = self.data_root / "captions.json"
        if caption_file.exists():
            with open(caption_file, 'r') as f:
                self.captions = json.load(f)
        else:
            # If no captions file, use image filenames as captions
            self.captions = {
                img.name: f"<image>\n<|grounding|>OCR this image."
                for img in self.image_dir.glob("*")
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']
            }

        self.image_files = list(self.captions.keys())

        # Default transform: resize and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            (image_tensor, caption)
        """
        img_file = self.image_files[idx]
        caption = self.captions[img_file]

        # Load image
        img_path = self.image_dir / img_file
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        image = self.transform(image)

        return image, caption


def create_dataloaders(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 640,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create train and validation dataloaders.

    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size

    Returns:
        (train_loader, val_loader) tuple
    """
    # Create datasets
    train_dataset = OCRDataset(train_data_path, image_size=image_size)

    val_dataset = None
    if val_data_path is not None:
        val_dataset = OCRDataset(val_data_path, image_size=image_size)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing OCRDataset...")

    # Create a dummy dataset for testing
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()

        # Create dummy images
        for i in range(5):
            img = Image.new('RGB', (640, 640), color=(i*50, 0, 0))
            img.save(img_dir / f"image_{i}.jpg")

        # Create captions
        captions = {
            f"image_{i}.jpg": f"<image>\nCaption {i}"
            for i in range(5)
        }
        with open(Path(tmpdir) / "captions.json", 'w') as f:
            json.dump(captions, f)

        # Create dataset
        dataset = OCRDataset(tmpdir, image_size=640)

        print(f"Dataset size: {len(dataset)}")
        assert len(dataset) == 5

        # Test __getitem__
        img, caption = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Caption: {caption}")
        assert img.shape == (3, 640, 640)

        print("✓ OCRDataset test passed!")
