#!/usr/bin/env python3
"""
LLaVA-Instruct-150K Dataset for OCRVL Stage 2 VIT Training

Converts LLaVA multi-turn conversations to cumulative bundled format:
- Each k-turn conversation creates k samples (one per answer)
- Each sample bundles full conversation history up to that point

Bundled format with cumulative context:
- Turn 1: <real_img> [rendered_Q1] → A1
- Turn 2: <real_img> [rendered_Q1+A1+Q2] → A2
- Turn 3: <real_img> [rendered_Q1+A1+Q2+A2+Q3] → A3

Benefits:
1. Preserves dependencies (Q2 sees A1, Q3 sees A1+A2)
2. Efficient encoding (2 images per sample: real + bundled text)
3. Shorter sequences (no autoregressive accumulation)

Rendering Performance:
- Vello renderer: ~1565 img/s (GPU-accelerated, cached instance)
- Vision encoder: ~206 img/s per GPU
- Rendering is NOT the bottleneck

Loss: Only computed on current answer tokens (Ai)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


# Global cached Vello renderer (reused across samples for performance)
_VELLO_RENDERER = None
_VELLO_AVAILABLE = False

def _get_vello_renderer(image_size=(640, 640)):
    """Get cached Vello renderer instance (create once, reuse many times)"""
    global _VELLO_RENDERER, _VELLO_AVAILABLE

    if _VELLO_RENDERER is None:
        try:
            from Renderer import VelloRenderer, VELLO_AVAILABLE
            _VELLO_AVAILABLE = VELLO_AVAILABLE
            if _VELLO_AVAILABLE:
                _VELLO_RENDERER = VelloRenderer(width=image_size[0], height=image_size[1], padding=20)
        except Exception:
            _VELLO_AVAILABLE = False
            _VELLO_RENDERER = None

    return _VELLO_RENDERER


def render_text_to_image(text: str, image_size=(640, 640)) -> Image.Image:
    """Render text to image using cached VelloRenderer (1565 img/s) or PIL fallback (130 img/s)

    Performance:
    - Vello: ~1565 img/s (GPU-accelerated Vulkan, cached instance)
    - PIL: ~130 img/s (CPU-based, fallback only)

    Vello is 12x faster and matches vision encoder throughput (~206 img/s).
    """
    text = str(text).strip()

    vr = _get_vello_renderer(image_size)
    if vr is not None:
        try:
            pil_img = vr.render_batch_pil([text])[0]
            return pil_img
        except Exception as e:
            print(f"Warning: Vello rendering failed ({e}), falling back to PIL")

    # PIL fallback (slow, but always available)
    from PIL import ImageDraw, ImageFont
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    margin = 16
    max_w = image_size[0] - 2 * margin
    y = margin
    for line in text.split('\n'):
        words = line.split()
        cur = ""
        for w in words:
            t = (cur + " " + w) if cur else w
            bbox = draw.textbbox((0, 0), t, font=font)
            if bbox[2] - bbox[0] <= max_w:
                cur = t
            else:
                if cur:
                    draw.text((margin, y), cur, fill='black', font=font)
                    y += 26
                cur = w
        if cur:
            draw.text((margin, y), cur, fill='black', font=font)
            y += 26
        y += 6
    return img


class LLaVAInstructDataset(Dataset):
    """
    LLaVA-Instruct-150K dataset for OCRVL Stage 2 VIT training.

    Cumulative bundled format:
    - Each k-turn conversation creates k samples (one per answer)
    - Turn 1: <real_img> [rendered_Q1] → A1
    - Turn 2: <real_img> [rendered_Q1+A1+Q2] → A2
    - Turn 3: <real_img> [rendered_Q1+A1+Q2+A2+Q3] → A3

    Vision encoding happens in collate_fn via BATCHED encoding for efficiency.

    Args:
        json_path: Path to llava_instruct_150k.json (or llava_v1_5_mix665k.json)
        image_dir: Directory containing images/ subdirectory with all datasets
                   For Mix665k: /path/to/LLaVA-Instruct-150K/images
                   This directory should contain: coco/, vg/, gqa/, ocr_vqa/, textvqa/
        render_questions: Whether to render questions as images (default True)
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        render_questions: bool = True,
        seed: int = 42,
    ):
        """Initialize LLaVA-Instruct dataset.

        Loads all conversations from the JSON file and expands them into
        training samples using cumulative bundled format (k samples per k-turn conversation).
        """
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.render_questions = render_questions
        self.rng = random.Random(seed)

        # Initialize Mix665k image loader for parquet + file-based images
        from OCRVL.data.parquet_image_loader import Mix665kImageLoader
        print(f"Initializing Mix665k image loader from {image_dir}...")
        self.image_loader = Mix665kImageLoader(str(image_dir))
        print(f"✓ Image loader ready")

        # Load conversations (support JSON array and JSONL line-delimited)
        if self.json_path.suffix.lower() == '.jsonl':
            self.conversations = []
            with open(self.json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.conversations.append(json.loads(line))
                    except Exception:
                        # Skip malformed lines to avoid aborting on large corpora
                        continue
        else:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.conversations = json.load(f)

        # Expand conversations into training samples
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """Prepare samples with cumulative context bundling

        For each conversation with k turns, creates k samples:
        - Sample 1: Q1 → A1
        - Sample 2: Q1+A1+Q2 → A2
        - Sample 3: Q1+A1+Q2+A2+Q3 → A3

        Each sample gets the full conversation history up to that point.
        """
        samples = []

        for conv in self.conversations:
            # Skip text-only conversations (no image)
            if 'image' not in conv:
                continue

            image_id = conv['image']
            turns = conv['conversations']

            # Extract Q&A pairs
            qa_pairs = []
            for i in range(0, len(turns), 2):
                if i + 1 >= len(turns):
                    break  # Skip incomplete pairs

                question = turns[i]['value'].replace('<image>\n', '').replace('<image>', '').strip()
                answer = turns[i + 1]['value'].strip()
                qa_pairs.append({'q': question, 'a': answer})

            if not qa_pairs:
                continue

            # Create one sample per turn with cumulative context (use all turns)
            for turn_idx in range(len(qa_pairs)):
                # Build cumulative context up to this turn
                # Context: Q1, A1, Q2, A2, ..., Q(i-1), A(i-1), Q(i)
                context_pairs = qa_pairs[:turn_idx + 1]  # Include current question

                sample = {
                    'image_id': image_id,
                    'context_pairs': context_pairs,  # Full history up to current turn
                    'turn_idx': turn_idx,  # Which turn this is (0-indexed)
                    'total_turns': len(qa_pairs),  # Total turns in conversation
                }
                samples.append(sample)

        # Shuffle samples (use all data, no filtering)
        self.rng.shuffle(samples)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample with PIL images for BATCHED encoding in collate_fn.

        Returns PIL images instead of encoded features so collate_fn can
        encode the entire batch at once (much faster than serial encoding).

        Returns:
            {
                'real_image': PIL Image,
                'bundled_context_text': str (cumulative Q&A context),
                'current_answer': str,
                'render_questions': bool,
            }
        """
        sample = self.samples[idx]

        # Load real image using Mix665k image loader (handles both file and parquet)
        image_id = sample['image_id']
        real_image = self.image_loader.load_image(image_id)

        if real_image is None:
            # Fallback: create blank image if loading fails
            print(f"Warning: Failed to load image {image_id}, using blank placeholder")
            real_image = Image.new('RGB', (640, 640), color='white')

        # Build cumulative context: Q1\nA1\nQ2\nA2\n...\nQi
        context_pairs = sample['context_pairs']
        turn_idx = sample['turn_idx']

        context_text_parts = []
        for i, qa in enumerate(context_pairs):
            context_text_parts.append(f"Q: {qa['q']}")
            if i < turn_idx:  # Include answer only for previous turns
                context_text_parts.append(f"A: {qa['a']}")

        bundled_context = "\n\n".join(context_text_parts)
        current_answer = context_pairs[turn_idx]['a']

        return {
            'real_image': real_image,  # PIL image (will be batch-encoded in collate_fn)
            'bundled_context_text': bundled_context,  # Text context (will be rendered+encoded in collate_fn if RENDER=1)
            'current_answer': current_answer,
            'render_questions': self.render_questions,
        }


def create_llava_collate_fn(tokenizer):
    """
    Collate function for LLaVA-Instruct dataset.

    CPU-ONLY: Builds sequences with vision token placeholders (no GPU encoding).
    Vision encoding happens in training loop after parallel data loading.

    Format:
    - Turn 1: <real_img> [rendered_Q1] → A1
    - Turn 2: <real_img> [rendered_Q1+A1+Q2] → A2

    Loss: Only compute on current answer tokens (Ai)
    """
    def collate_fn(batch):
        batch_input_ids = []
        batch_labels = []
        batch_images_to_encode = []  # Collect images for encoding
        batch_image_indices = []  # Track which positions need features

        # Random number generator for ordering (fixed seed for reproducibility)
        import random
        rng = random.Random(42)

        # Qwen3-VL chat tokens
        user_start_ids = torch.tensor([151644, 872, 198], dtype=torch.long)
        user_end_ids = torch.tensor([151645, 198], dtype=torch.long)
        assistant_start_ids = torch.tensor([151644, 77091, 198], dtype=torch.long)
        assistant_end_ids = torch.tensor([151645, 198], dtype=torch.long)

        # Vision token placeholder (model will fill with actual features)
        # Use 100 tokens as placeholder - model expects this count
        VISION_TOKEN_COUNT = 100
        vision_placeholder = "<|vision_start|>" + "<|image_pad|>" * VISION_TOKEN_COUNT + "<|vision_end|>"

        for sample in batch:
            bundled_ctx_text = sample['bundled_context_text']
            current_answer = sample['current_answer']
            render_questions = sample['render_questions']
            real_image = sample['real_image']

            sample_images = []

            # Qwen3-VL official format: All images in ONE user block
            # <|im_start|>user<img1><img2><|im_end|><|im_start|>assistant...
            # Random ordering: 50% real first, 50% rendered first (for robustness)

            # Start ONE user block
            sequence_parts = [user_start_ids]
            label_parts = [torch.full_like(user_start_ids, -100)]

            # Collect images with random ordering
            if render_questions:
                # RENDER=1: Both real image and rendered context
                vr = _get_vello_renderer((640, 640))
                if vr is not None:
                    try:
                        context_image = vr.render_batch_pil([bundled_ctx_text])[0]
                    except Exception as e:
                        print(f"Warning: Vello rendering failed ({e}), falling back to PIL")
                        context_image = render_text_to_image(bundled_ctx_text)
                else:
                    context_image = render_text_to_image(bundled_ctx_text)

                # Random ordering: 50% real first, 50% rendered first
                if rng.random() < 0.5:
                    # Real image first (default order)
                    images_in_order = [real_image, context_image]
                else:
                    # Rendered context first
                    images_in_order = [context_image, real_image]

                # Add images to sequence
                for img in images_in_order:
                    img_ids = tokenizer(vision_placeholder, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                    sequence_parts.append(img_ids)
                    label_parts.append(torch.full_like(img_ids, -100))
                    sample_images.append(img)
            else:
                # RENDER=0: Only real image + text prompt
                real_img_ids = tokenizer(vision_placeholder, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                sequence_parts.append(real_img_ids)
                label_parts.append(torch.full_like(real_img_ids, -100))
                sample_images.append(real_image)

                # Add text prompt
                ctx_text_ids = tokenizer(bundled_ctx_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
                sequence_parts.append(ctx_text_ids)
                label_parts.append(torch.full_like(ctx_text_ids, -100))

            # Close user block
            sequence_parts.append(user_end_ids)
            label_parts.append(torch.full_like(user_end_ids, -100))

            # Answer
            answer_ids = tokenizer(current_answer, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
            sequence_parts.extend([assistant_start_ids, answer_ids, assistant_end_ids])
            label_parts.extend([
                torch.full_like(assistant_start_ids, -100),
                answer_ids.clone(),
                assistant_end_ids.clone()  # Compute loss on EOS! Model must learn when to stop
            ])

            # Concatenate
            full_input_ids = torch.cat(sequence_parts, dim=0)
            labels = torch.cat(label_parts, dim=0)

            batch_input_ids.append(full_input_ids)
            batch_labels.append(labels)
            batch_images_to_encode.append(sample_images)

        # Pad sequences
        max_len = max(ids.size(0) for ids in batch_input_ids)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(batch_input_ids, batch_labels):
            seq_len = input_ids.size(0)
            padding_len = max_len - seq_len

            padded_input_ids.append(
                torch.cat([input_ids, torch.full((padding_len,), pad_token_id, dtype=input_ids.dtype)])
            )
            padded_labels.append(
                torch.cat([labels, torch.full((padding_len,), -100, dtype=labels.dtype)])
            )
            attention_masks.append(
                torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(padded_labels),
            "images_to_encode": batch_images_to_encode,  # List of lists of PIL images
        }

    return collate_fn

