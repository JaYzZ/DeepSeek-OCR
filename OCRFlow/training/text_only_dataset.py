"""
Text-Only Dataset for OCRFlow Training

Loads text data and renders it on-the-fly during training.
No need for document image datasets!
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random
from typing import Optional, Tuple, List
import torchvision.transforms as T

from OCRFlow.utils.text_rendering import render_text_to_image, render_markdown_to_image


class TextOnlyDataset(Dataset):
    """
    Dataset that loads text-only data and renders to images on-the-fly

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
        image_size: Size to render images (width and height)
        font_size: Base font size for rendering
        use_markdown: If True, interpret text as markdown
        max_samples: Maximum number of samples to load (None = all)
        prompts: List of prompt templates to randomly choose from
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 640,
        font_size: int = 16,
        use_markdown: bool = True,
        max_samples: Optional[int] = None,
        prompts: Optional[List[str]] = None,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.font_size = font_size
        self.use_markdown = use_markdown

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

        print(f"Loaded {len(self.texts)} text samples from {data_root}")

        # Image transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

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
                print(f"Warning: Failed to load {txt_file}: {e}")
                continue

        if not texts:
            raise ValueError(
                f"No texts found in {self.data_root}. "
                f"Expected either texts.jsonl or *.txt files"
            )

        return texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            (rendered_image_tensor, prompt_text)
        """
        text = self.texts[idx]

        # Truncate very long texts
        max_chars = 2000  # Fits in ~640x640 image
        if len(text) > max_chars:
            text = text[:max_chars]

        # Render text to image
        try:
            if self.use_markdown:
                img = render_markdown_to_image(
                    text,
                    width=self.image_size,
                    height=self.image_size,
                    base_font_size=self.font_size,
                )
            else:
                img = render_text_to_image(
                    text,
                    width=self.image_size,
                    height=self.image_size,
                    font_size=self.font_size,
                )
        except Exception as e:
            # Fallback to simple rendering if markdown fails
            print(f"Warning: Rendering failed for sample {idx}, using fallback: {e}")
            img = render_text_to_image(
                text,
                width=self.image_size,
                height=self.image_size,
                font_size=self.font_size,
            )

        # Convert to tensor
        img_tensor = self.transform(img)

        # Random prompt selection
        prompt = random.choice(self.prompts)

        return img_tensor, prompt


def create_text_only_dataloaders(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 640,
    font_size: int = 16,
    use_markdown: bool = True,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders for text-only training

    Args:
        train_data_path: Path to training text data
        val_data_path: Path to validation text data (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size for rendering
        font_size: Base font size
        use_markdown: Interpret text as markdown
        max_train_samples: Max training samples
        max_val_samples: Max validation samples

    Returns:
        (train_loader, val_loader) tuple
    """
    # Create datasets
    train_dataset = TextOnlyDataset(
        train_data_path,
        image_size=image_size,
        font_size=font_size,
        use_markdown=use_markdown,
        max_samples=max_train_samples,
    )

    val_dataset = None
    if val_data_path is not None:
        val_dataset = TextOnlyDataset(
            val_data_path,
            image_size=image_size,
            font_size=font_size,
            use_markdown=use_markdown,
            max_samples=max_val_samples,
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing TextOnlyDataset...")

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy text data
        texts = [
            "# Document 1\n\nThis is the first test document.",
            "# Document 2\n\nAnother document for testing.\n\n- Item 1\n- Item 2",
            "Simple plain text without markdown.",
            "# Long Document\n\n" + "Lorem ipsum. " * 100,
        ]

        # Save as JSONL
        jsonl_path = Path(tmpdir) / "texts.jsonl"
        with open(jsonl_path, 'w') as f:
            for text in texts:
                json.dump({"text": text}, f)
                f.write('\n')

        # Create dataset
        dataset = TextOnlyDataset(tmpdir, image_size=640, use_markdown=True)

        print(f"Dataset size: {len(dataset)}")
        assert len(dataset) == len(texts)

        # Test __getitem__
        img, prompt = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Prompt: {prompt}")
        assert img.shape == (3, 640, 640)
        assert isinstance(prompt, str)

        # Test dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch_img, batch_prompts = next(iter(loader))
        print(f"Batch image shape: {batch_img.shape}")
        print(f"Batch prompts: {batch_prompts}")
        assert batch_img.shape == (2, 3, 640, 640)

        print("✓ TextOnlyDataset test passed!")
