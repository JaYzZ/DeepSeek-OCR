"""Dataset for thinking-with-latent-tokens training using LLaVA-CoT format."""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any
import logging
import json
import os
import re
from PIL import Image

logger = logging.getLogger(__name__)


def extract_reasoning(text: str) -> tuple[str, str]:
    """Extract reasoning section from LLaVA-CoT formatted response.

    Format:
        <SUMMARY>...</SUMMARY>
        <CAPTION>...</CAPTION>
        <REASONING>...(chain of thought)...</REASONING>
        <CONCLUSION>...(final answer)...</CONCLUSION>

    Returns:
        (reasoning_text, conclusion_text)
    """
    # Extract reasoning between <REASONING> and </REASONING>
    reasoning_pattern = r'<REASONING>(.*?)</REASONING>'
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL | re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract conclusion
    conclusion_pattern = r'<CONCLUSION>(.*?)</CONCLUSION>'
    conclusion_match = re.search(conclusion_pattern, text, re.DOTALL | re.IGNORECASE)
    conclusion = conclusion_match.group(1).strip() if conclusion_match else text

    # If no structured format, use full text as reasoning and extract last sentence as conclusion
    if not reasoning and not conclusion_match:
        sentences = text.split('. ')
        if len(sentences) > 1:
            reasoning = '. '.join(sentences[:-1]) + '.'
            conclusion = sentences[-1]
        else:
            reasoning = ""
            conclusion = text

    return reasoning, conclusion


class LLaVACoTDataset(Dataset):
    """Dataset for LLaVA-CoT-100k format with thinking-latent training.

    Expected JSONL format:
    {
        "id": "...",
        "image": "sqa/train/20839/image.png",
        "conversations": [
            {"from": "human", "value": "Question text..."},
            {"from": "gpt", "value": "<SUMMARY>...</SUMMARY><REASONING>...</REASONING><CONCLUSION>...</CONCLUSION>"}
        ]
    }
    """

    def __init__(
        self,
        jsonl_path: str,
        image_base_dir: str,
        tokenizer,
        ocr_adapter,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.image_base_dir = image_base_dir
        self.tokenizer = tokenizer
        self.ocr_adapter = ocr_adapter
        self.max_length = max_length

        # Load data from JSONL
        self.data = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.data.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(self.data)} samples from {jsonl_path}")

        # Get special token IDs
        self.think_start_id = tokenizer.convert_tokens_to_ids("<think>")
        self.think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        self.vision_start = "<|vision_start|>"
        self.vision_end = "<|vision_end|>"
        self.image_pad = "<|image_pad|>"
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_path = os.path.join(self.image_base_dir, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}. Using dummy image.")
            image = Image.new('RGB', (640, 640), color='white')

        # Parse conversations
        conversations = item['conversations']
        human_msg = conversations[0]['value']  # Question
        gpt_msg = conversations[1]['value']    # Response with CoT

        # Extract reasoning and conclusion from GPT response
        reasoning, conclusion = extract_reasoning(gpt_msg)

        # If no reasoning found, skip thinking tokens and use regular training
        has_thinking = len(reasoning) > 10  # At least some meaningful reasoning

        if has_thinking:
            # Render and encode reasoning text with DPSK OCR
            try:
                ocr_result = self.ocr_adapter.text_to_ocr_features(
                    reasoning,
                    tokenizer=self.tokenizer
                )

                if isinstance(ocr_result, tuple):
                    final_feats, _ = ocr_result  # Ignore deepstack for now
                else:
                    final_feats = ocr_result

                # Ensure it's a list
                if not isinstance(final_feats, list):
                    final_feats = [final_feats]

            except Exception as e:
                logger.warning(f"Failed to encode reasoning for sample {idx}: {e}. Skipping thinking.")
                has_thinking = False
                final_feats = []
        else:
            final_feats = []

        # Build sequence
        # Format: <|im_start|>user\n<image>\n{question}<|im_end|>\n
        #         <|im_start|>assistant\n<think><latent_1>...</think>{answer}<|im_end|>

        # Tokenize components
        user_start = f"{self.im_start}user\n"
        user_start_ids = self.tokenizer(user_start, add_special_tokens=False).input_ids

        # Image placeholder (100 tokens for Qwen3-VL)
        image_placeholder = f"{self.vision_start}{self.image_pad * 100}{self.vision_end}\n"
        image_placeholder_ids = self.tokenizer(image_placeholder, add_special_tokens=False).input_ids

        # Question
        question_ids = self.tokenizer(human_msg, add_special_tokens=False).input_ids

        user_end_ids = self.tokenizer(f"\n{self.im_end}\n", add_special_tokens=False).input_ids

        # Assistant start
        assistant_start_ids = self.tokenizer(f"{self.im_start}assistant\n", add_special_tokens=False).input_ids

        # Build sequence parts
        sequence_parts = [
            user_start_ids,
            image_placeholder_ids,
            question_ids,
            user_end_ids,
            assistant_start_ids,
        ]

        # Add thinking tokens if available
        if has_thinking and len(final_feats) > 0:
            think_start = [self.think_start_id]
            think_end = [self.think_end_id]

            # Create vision placeholders for each latent chunk
            latent_placeholders = []
            for _ in range(len(final_feats)):
                placeholder = f"{self.vision_start}{self.image_pad}{self.vision_end}"
                placeholder_ids = self.tokenizer(placeholder, add_special_tokens=False).input_ids
                latent_placeholders.extend(placeholder_ids)

            sequence_parts.extend([
                think_start,
                latent_placeholders,
                think_end,
                self.tokenizer("\n", add_special_tokens=False).input_ids,
            ])

        # Answer/conclusion
        answer_ids = self.tokenizer(conclusion, add_special_tokens=False).input_ids
        assistant_end_ids = self.tokenizer(f"{self.im_end}", add_special_tokens=False).input_ids

        sequence_parts.extend([answer_ids, assistant_end_ids])

        # Concatenate all parts
        input_ids = []
        for part in sequence_parts:
            input_ids.extend(part if isinstance(part, list) else part.tolist())
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Create labels: -100 for everything except answer
        # Loss is only computed on the conclusion/answer tokens
        labels = torch.full_like(input_ids, -100)

        # Find answer start position
        if has_thinking and len(final_feats) > 0:
            # Answer starts after: user_block + assistant_start + <think> + latents + </think> + \n
            answer_start_offset = sum(len(p) if isinstance(p, list) else len(p.tolist())
                                     for p in sequence_parts[:-2])  # All except answer and end
        else:
            # Answer starts after: user_block + assistant_start
            answer_start_offset = sum(len(p) if isinstance(p, list) else len(p.tolist())
                                     for p in sequence_parts[:-2])

        answer_len = len(answer_ids) + len(assistant_end_ids)
        labels[answer_start_offset:answer_start_offset + answer_len] = input_ids[answer_start_offset:answer_start_offset + answer_len]

        # Create latent position mask (True where latent tokens should be)
        latent_positions = torch.zeros(len(input_ids), dtype=torch.bool)

        if has_thinking and len(final_feats) > 0:
            # Find latent token positions: after <think>, before </think>
            think_start_offset = sum(len(p) if isinstance(p, list) else len(p.tolist())
                                    for p in sequence_parts[:6])  # Up to and including <think>
            latent_placeholders_len = len(latent_placeholders)
            latent_positions[think_start_offset:think_start_offset + latent_placeholders_len] = True

        return {
            'input_ids': input_ids,
            'labels': labels,
            'latent_supervision': final_feats if has_thinking else [],
            'latent_positions': latent_positions,
            'image': image,  # PIL Image for OCR encoding
            'has_thinking': has_thinking,
        }


def llava_cot_collate_fn(batch, ocr_adapter=None):
    """Collate function for LLaVA-CoT dataset with padding.

    Args:
        batch: List of dataset items
        ocr_adapter: OCR adapter for encoding real images (optional, will use from batch if None)
    """
    # Find max length
    max_len = max(item["input_ids"].shape[0] for item in batch)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_latent_supervision = []
    batch_latent_positions = []
    batch_images = []

    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - ids.shape[0]

        if pad_len > 0:
            # Pad with zeros
            ids_padded = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            labels_padded = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            latent_pos_padded = torch.cat([item["latent_positions"], torch.zeros(pad_len, dtype=torch.bool)])
            attn_mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)])
        else:
            ids_padded = ids
            labels_padded = item["labels"]
            latent_pos_padded = item["latent_positions"]
            attn_mask = torch.ones(len(ids))

        batch_input_ids.append(ids_padded)
        batch_labels.append(labels_padded)
        batch_attention_mask.append(attn_mask)
        batch_latent_positions.append(latent_pos_padded)
        batch_latent_supervision.append(item["latent_supervision"])
        batch_images.append(item["image"])

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),
        "latent_supervision": batch_latent_supervision,  # List of lists
        "latent_positions": torch.stack(batch_latent_positions),
        "images": batch_images,  # PIL Images
    }
