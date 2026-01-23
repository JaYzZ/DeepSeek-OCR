"""
OCR Thinking Training Dataset

This module provides a dataset for training with OCR-encoded thinking tokens.

Format: Question + <|thinking_start|><|latent_step|><|thinking_sep|><|latent_step|>...<|thinking_end|> + Answer

The number of latent steps (k) is ADAPTIVE - determined by how many chunks
the OCR adapter produces from the thinking text. Short thinking might yield 1
step, long thinking might yield 5+ steps. The model learns from this
variable-length thinking representation.

Example:
    Short thinking (1 chunk):
        "Simple answer." -> 1 image -> <|thinking_start|><|latent_step|><|thinking_end|>

    Long thinking (3 chunks):
        "Step 1... Step 2... Step 3..." -> 3 images -> <|thinking_start|><|latent_step|><|thinking_sep|><|latent_step|><|thinking_sep|><|latent_step|><|thinking_end|>

Each chunk is rendered to an image and encoded as [100, 1280] features (10x10 grid).

Key features:
- Uses thinking tokens (<|thinking_start|>, <|latent_step|>, <|thinking_sep|>, <|thinking_end|>)
- OCR-encoded thinking text (no SUMMARY/CAPTION/REASONING tags)
- Pre-encoded OCR features are passed directly to training
- Adaptive number of thinking steps per sample (determined by OCR adapter)
- Compatible with LlamaFactory training

Expected JSONL format:
    {
        "id": "...",
        "image": "path/to/image.png",
        "question": "What is 2+2?",
        "thinking": "To solve 2+2, I need to add the numbers...",
        "answer": "4"
    }
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from OCRVL.utils.latent_tokens import (
    get_thinking_token_ids,
    build_sequence_with_thinking,
    # Backward compatibility
    get_latent_token_ids,
    build_sequence_with_latents,
)

logger = logging.getLogger(__name__)


class OCRThinkingDataset(Dataset):
    """Dataset for OCR-encoded thinking token training.

    This dataset:
    1. Loads question/thinking/answer from JSONL
    2. Tokenizes question and answer
    3. Encodes thinking text as OCR features (via adapter) - ADAPTIVE chunking
    4. Builds sequence with thinking token placeholders
    5. Returns thinking supervision and positions for injection

    The number of thinking steps is determined purely by the OCR adapter when
    encoding the thinking text - no pre-computation or estimation needed.

    Args:
        jsonl_path: Path to JSONL data file
        tokenizer: Tokenizer for text encoding
        ocr_adapter: OCR adapter for encoding thinking text
        max_length: Maximum sequence length
        max_samples: Optional limit on number of samples
        include_images: Whether to load images (for VQA tasks)
        image_base_dir: Base directory for images (if include_images=True)
        use_prerendered: If True, use pre-rendered cached images instead of on-the-fly rendering
        prerendered_base_dir: Base directory for pre-rendered images (defaults to image_base_dir)

    JSONL Format (text mode, use_prerendered=False):
        {
            "question": "What is 2+2?",
            "thinking": "To solve 2+2, I need to add...",
            "answer": "4",
            "image": "path/to/image.png"
        }

    JSONL Format (LlamaFactory ShareGPT - auto-detected by presence of "messages" field):
        {
            "messages": [
                {"role": "user", "content": "<image>\nWhat is 2+2?"},
                {"role": "assistant", "content": "<control_chars><image><control_chars>\nTo solve 2+2...\n4"}
            ],
            "images": ["path/to/v_image.png", "path/to/thinking_image.png"]
        }

    JSONL Format (custom prerendered mode, use_prerendered=True):
        {
            "question_images": ["question_rendered_0.png"],
            "thinking_images": ["thinking_rendered_0.png", "thinking_rendered_1.png"],
            "answer": "4",
            "image": "path/to/image.png"
        }
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizerBase,
        ocr_adapter: Any,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        include_images: bool = False,
        image_base_dir: Optional[str] = None,
        use_prerendered: bool = False,
        prerendered_base_dir: Optional[str] = None,
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.ocr_adapter = ocr_adapter
        self.max_length = max_length
        self.include_images = include_images
        self.image_base_dir = image_base_dir or ""
        self.use_prerendered = use_prerendered
        self.prerendered_base_dir = prerendered_base_dir or image_base_dir or ""

        # Verify thinking tokens exist
        try:
            self.token_ids = get_thinking_token_ids(tokenizer)
        except ValueError as e:
            raise ValueError(
                f"Thinking tokens not found in tokenizer. Please add them first: {e}"
            )

        # Load data
        self.data = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.data.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(self.data)} samples from {jsonl_path}")

        # Count samples with thinking (text or prerendered)
        with_thinking_text = sum(1 for item in self.data if item.get('thinking', '').strip())
        with_thinking_images = sum(1 for item in self.data if item.get('thinking_images'))
        with_thinking = with_thinking_text + with_thinking_images
        logger.info(f"  Samples with thinking: {with_thinking}/{len(self.data)}")
        if self.use_prerendered:
            logger.info(f"  Mode: PRERENDERED (using cached images)")

    def __len__(self):
        return len(self.data)

    def _encode_thinking(self, thinking_text: str) -> List[torch.Tensor]:
        """Encode thinking text as OCR features (renders text then encodes).

        Args:
            thinking_text: Thinking text to encode

        Returns:
            List of [100, 1280] tensors (one per chunk, 10×10 grid)

        Note:
            The final [100, 1280] features can be decoded back to text using
            pre-learned line separator and view separator embeddings.
        """
        if not thinking_text or not thinking_text.strip():
            return []

        try:
            # Use OCR adapter to encode thinking text
            ocr_result = self.ocr_adapter.text_to_ocr_features(
                thinking_text,
                tokenizer=self.tokenizer
            )

            # Extract final features (ignore deepstack - not needed for thinking training)
            if isinstance(ocr_result, tuple):
                final_feats, _ = ocr_result
            else:
                final_feats = ocr_result

            # Ensure list format
            if not isinstance(final_feats, list):
                final_feats = [final_feats]

            return final_feats

        except Exception as e:
            logger.warning(f"Failed to encode thinking: {e}")
            return []

    def _encode_prerendered_thinking(
        self, thinking_image_paths: List[str]
    ) -> List[torch.Tensor]:
        """Encode pre-rendered thinking images as OCR features.

        This skips the rendering step and directly encodes cached images.
        Much faster than on-the-fly rendering.

        Args:
            thinking_image_paths: List of paths to pre-rendered thinking images

        Returns:
            List of [100, 1280] tensors (one per image, 10×10 grid)
        """
        if not thinking_image_paths:
            return []

        try:
            from PIL import Image

            # Load all images
            images = []
            for img_path in thinking_image_paths:
                # Handle relative paths
                if not os.path.isabs(img_path):
                    # Try image_base_dir first (for LlamaFactory format)
                    if self.image_base_dir:
                        full_path = os.path.join(self.image_base_dir, img_path)
                    else:
                        full_path = img_path
                else:
                    full_path = img_path

                try:
                    img = Image.open(full_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {full_path}: {e}")
                    continue

            if not images:
                return []

            # Encode images directly using the encoder (skip rendering)
            encoder = self.ocr_adapter.encoder
            ocr_result = encoder.encode_images(images, return_global=False, return_local=True)

            # Ensure list format
            if not isinstance(ocr_result, list):
                ocr_result = [ocr_result]

            return ocr_result

        except Exception as e:
            logger.warning(f"Failed to encode prerendered thinking: {e}")
            return []

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Auto-detect format
        is_llamafactory_format = 'messages' in item

        if is_llamafactory_format:
            # LlamaFactory ShareGPT format (existing R1-OneVision data)
            # Format: {"messages": [...], "images": [...]}
            messages = item.get('messages', [])
            images = item.get('images', [])

            if not messages or len(messages) < 2:
                raise ValueError(f"Invalid sample at idx {idx}: not enough messages")

            user_msg = messages[0]
            assistant_msg = messages[1]

            user_content = user_msg.get('content', '')
            assistant_content = assistant_msg.get('content', '')

            # Extract answer from assistant (after thinking section)
            # For now, use full assistant content as answer
            answer = assistant_content

            # Extract thinking images from assistant
            # Count <image> tags in assistant
            assistant_image_count = assistant_content.count('<image>')

            thinking_image_paths = []
            question_ids = []

            if len(images) > 0:
                # First image is V image (from user)
                v_image_path = images[0]
            else:
                v_image_path = None

            # Remaining images are thinking images (from assistant)
            if assistant_image_count > 0 and len(images) > 1:
                thinking_image_paths = images[1:1 + assistant_image_count]

            # Encode prerendered thinking images
            thinking_feats = self._encode_prerendered_thinking(thinking_image_paths)

        elif self.use_prerendered:
            # Custom prerendered mode: question/thinking are images, answer is text
            thinking_image_paths = item.get('thinking_images', [])
            question_image_paths = item.get('question_images', [])
            answer = item.get('answer', '')
            v_image_path = item.get('image')

            # For prerendered mode, we use a dummy question tokenization
            question_ids = []

            # Encode prerendered thinking images
            thinking_feats = self._encode_prerendered_thinking(thinking_image_paths)

        else:
            # Text mode: question/thinking are text
            question = item.get('question', '')
            thinking = item.get('thinking', '')
            answer = item.get('answer', '')
            v_image_path = item.get('image')

            # Tokenize question
            question_ids = self.tokenizer(question, add_special_tokens=False).input_ids

            # Encode thinking text as OCR features
            thinking_feats = self._encode_thinking(thinking)

        # Tokenize answer
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids

        num_thinking_steps = len(thinking_feats)

        # Build sequence with thinking placeholders (using separator format)
        input_ids, labels, thinking_start, thinking_end = build_sequence_with_thinking(
            question_ids=question_ids,
            answer_ids=answer_ids,
            num_steps=num_thinking_steps,
            tokenizer=self.tokenizer,
            include_newline=True,
        )

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        # Create latent step position mask (marks <|latent_step|> tokens only)
        latent_positions = torch.zeros(len(input_ids), dtype=torch.bool)
        if num_thinking_steps > 0 and thinking_start < len(input_ids):
            # Find positions of <|latent_step|> tokens (not including separators or boundaries)
            step_id = self.token_ids.get("<|latent_step|>", None)
            if step_id is not None:
                # Check each position in the thinking section
                for i in range(thinking_start, min(thinking_end, len(input_ids))):
                    if i < len(input_ids) and input_ids[i] == step_id:
                        latent_positions[i] = True

        # Load image if needed (for VQA tasks)
        image = None
        if self.include_images:
            # Determine image path based on format
            if is_llamafactory_format:
                image_path = v_image_path
            else:
                image_path = item.get('image')

            if image_path:
                if not os.path.isabs(image_path) and self.image_base_dir:
                    image_path = os.path.join(self.image_base_dir, image_path)

                try:
                    from PIL import Image
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")

        return {
            'input_ids': input_ids,
            'labels': labels,
            'latent_supervision': thinking_feats,  # List of [100, 1280] tensors
            'latent_positions': latent_positions,  # Boolean mask for <|latent_step|> tokens
            'image': image,
            'has_thinking': num_thinking_steps > 0,
        }


def ocr_thinking_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """Collate function for OCR thinking training dataset.

    Args:
        batch: List of dataset items
        pad_token_id: Token ID for padding

    Returns:
        Batched dictionary with:
        - input_ids: [batch_size, max_seq_len]
        - attention_mask: [batch_size, max_seq_len]
        - labels: [batch_size, max_seq_len]
        - latent_supervision: List of lists of [100, 1280] tensors (10×10 grid, thinking features)
        - latent_positions: [batch_size, max_seq_len] (marks <|latent_step|> tokens)
        - images: List of PIL images (or None)
        - has_thinking: List of bool indicating if each sample has thinking
    """
    # Find max length
    max_len = max(item['input_ids'].shape[0] for item in batch)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_latent_supervision = []
    batch_latent_positions = []
    batch_images = []
    batch_has_thinking = []

    for item in batch:
        ids = item['input_ids']
        labels_item = item['labels']
        latent_pos = item['latent_positions']
        pad_len = max_len - ids.shape[0]

        if pad_len > 0:
            # Pad
            ids_padded = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
            labels_padded = torch.cat([labels_item, torch.full((pad_len,), -100, dtype=labels_item.dtype)])
            latent_pos_padded = torch.cat([latent_pos, torch.zeros((pad_len,), dtype=torch.bool)])
            attn_mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)])
        else:
            ids_padded = ids
            labels_padded = labels_item
            latent_pos_padded = latent_pos
            attn_mask = torch.ones(len(ids))

        batch_input_ids.append(ids_padded)
        batch_labels.append(labels_padded)
        batch_attention_mask.append(attn_mask)
        batch_latent_positions.append(latent_pos_padded)
        batch_latent_supervision.append(item['latent_supervision'])
        batch_images.append(item.get('image'))
        batch_has_thinking.append(item['has_thinking'])

    return {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'labels': torch.stack(batch_labels),
        'latent_supervision': batch_latent_supervision,
        'latent_positions': torch.stack(batch_latent_positions),
        'images': batch_images,
        'has_thinking': batch_has_thinking,
    }


# Convenience function to create sample OCR thinking training data
def create_sample_jsonl(
    output_path: str,
    num_samples: int = 10,
    include_images: bool = False,
):
    """Create a sample JSONL file for testing OCR thinking training.

    Args:
        output_path: Path to save JSONL file
        num_samples: Number of samples to generate
        include_images: Whether to include image field
    """
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "id": f"sample_{i}",
            "question": f"What is {i} + {i}?",
            "thinking": f"To solve {i} + {i}, I need to add the two numbers together. "
                       f"The first number is {i} and the second number is also {i}. "
                       f"When I add them, I get {i + i}.",
            "answer": f"{i + i}",
        }
        for i in range(num_samples)
    ]

    if include_images:
        for sample in samples:
            sample['image'] = f"images/{sample['id']}.png"

    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    logger.info(f"Created sample JSONL with {num_samples} samples at {output_path}")
