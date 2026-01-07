"""
LLaVA Mix665k Dataset for OCRVL Training

Handles the full 665K LLaVA instruction-following dataset with support for:
- File-based images (COCO, Visual Genome)
- Parquet-embedded images (GQA, OCR-VQA, TextVQA)
"""

import json
from typing import Dict, Any, List
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

from OCRVL.data.parquet_image_loader import Mix665kImageLoader


class LLaVAMix665kDataset(Dataset):
    """
    LLaVA Mix665k instruction-following dataset.
    
    Format:
        {
            "id": "000000033471",
            "image": "coco/train2017/000000033471.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat are the colors..."},
                {"from": "gpt", "value": "The image features..."}
            ]
        }
    """
    
    def __init__(
        self,
        json_path: str,
        images_dir: str,
        tokenizer,
        ocr_adapter,
        max_length: int = 2048,
        image_token: str = "<image>",
    ):
        """
        Args:
            json_path: Path to llava_v1_5_mix665k.json
            images_dir: Base directory containing images/ subdirectory
            tokenizer: Qwen tokenizer
            ocr_adapter: Qwen3VLOCRTextAdapter for encoding images
            max_length: Maximum sequence length
            image_token: Token to replace with image features
        """
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.ocr_adapter = ocr_adapter
        self.max_length = max_length
        self.image_token = image_token
        
        # Load data
        print(f"Loading LLaVA Mix665k from {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        print(f"✓ Loaded {len(self.data):,} samples")
        
        # Initialize image loader with symlinked parquet datasets
        print(f"Initializing image loader from {images_dir}...")
        self.image_loader = Mix665kImageLoader(images_dir)
        print(f"✓ Image loader ready")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item['image']
        image = self.image_loader.load_image(image_path)
        
        if image is None:
            # Fallback to error handling
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Parse conversations
        conversations = item['conversations']
        if len(conversations) < 2:
            raise ValueError(f"Invalid conversation format in sample {idx}")
        
        # Extract human question and GPT response
        human_msg = None
        gpt_msg = None
        for msg in conversations:
            if msg['from'] == 'human':
                human_msg = msg['value']
            elif msg['from'] == 'gpt':
                gpt_msg = msg['value']
        
        if human_msg is None or gpt_msg is None:
            raise ValueError(f"Missing human or gpt message in sample {idx}")
        
        # Remove <image> token from instruction (will be added by adapter)
        instruction = human_msg.replace(self.image_token, "").strip()
        
        # Encode image + instruction
        input_ids, ocr_features = self.ocr_adapter.prepare_qwen_inputs_from_images(
            instruction=instruction,
            images=[image],
            tokenizer=self.tokenizer,
            return_deepstack=True
        )
        
        # Encode response
        response_ids = self.tokenizer(
            gpt_msg,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Concatenate: [image features] [instruction] [response]
        full_input_ids = torch.cat([input_ids.squeeze(0), response_ids], dim=0)
        
        # Create labels: mask instruction part, predict response only
        labels = full_input_ids.clone()
        labels[:len(input_ids.squeeze(0))] = -100  # Ignore loss on instruction
        
        # Truncate if needed
        if len(full_input_ids) > self.max_length:
            full_input_ids = full_input_ids[:self.max_length]
            labels = labels[:self.max_length]
            # Also truncate ocr_features if needed
            # This is a simplification - proper implementation should handle this better
        
        return {
            "input_ids": full_input_ids,
            "labels": labels,
            "ocr_image_features": ocr_features,
            "image_path": image_path,  # For debugging
            "sample_id": item.get('id', str(idx)),
        }


def create_llava_mix665k_dataloader(
    json_path: str,
    images_dir: str,
    tokenizer,
    ocr_adapter,
    batch_size: int = 4,
    max_length: int = 2048,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """
    Create DataLoader for LLaVA Mix665k dataset.
    
    Args:
        json_path: Path to llava_v1_5_mix665k.json
        images_dir: Base directory containing images/ (e.g., .../LLaVA-Instruct-150K/images)
        tokenizer: Qwen tokenizer
        ocr_adapter: Qwen3VLOCRTextAdapter
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    dataset = LLaVAMix665kDataset(
        json_path=json_path,
        images_dir=images_dir,
        tokenizer=tokenizer,
        ocr_adapter=ocr_adapter,
        max_length=max_length,
    )
    
    # Custom collate function to handle variable-length sequences
    def collate_fn(batch):
        # Pad sequences to same length within batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids = []
        labels = []
        ocr_features = []
        
        for item in batch:
            # Pad input_ids
            pad_len = max_len - len(item['input_ids'])
            padded_input = torch.cat([
                item['input_ids'],
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
            input_ids.append(padded_input)
            
            # Pad labels
            padded_labels = torch.cat([
                item['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)  # -100 = ignore
            ])
            labels.append(padded_labels)
            
            ocr_features.append(item['ocr_image_features'])
        
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'ocr_image_features': torch.stack(ocr_features) if ocr_features[0] is not None else None,
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from transformers import AutoTokenizer
    from OCRVL.model.language_model.ocr_qwen3_vl import Qwen3VLOCRTextAdapter
    
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    ocr_adapter = Qwen3VLOCRTextAdapter()
    
    # Create dataset
    dataset = LLaVAMix665kDataset(
        json_path="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",
        images_dir="/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/images",
        tokenizer=tokenizer,
        ocr_adapter=ocr_adapter,
    )
    
    print(f"\nTesting dataset...")
    print(f"Total samples: {len(dataset):,}")
    
    # Test first sample
    print(f"\nLoading first sample...")
    sample = dataset[0]
    print(f"✓ Sample loaded successfully")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  image_path: {sample['image_path']}")
    print(f"  sample_id: {sample['sample_id']}")
