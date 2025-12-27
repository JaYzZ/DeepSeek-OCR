#!/usr/bin/env python3
"""
LLaVA-Instruct-150K Dataset for OCRVL Stage 2 VIT Training

Converts LLaVA multi-turn conversations to OCRVL pure vision format with:
1. Single-turn QA samples (loss weight: 0.2)
2. Multi-turn QA samples (loss weight: 1.0)

Format:
- Input: [Real image] + [Rendered questions]
- Output: Formatted answers
- Prompt: <|im_start|>assistant\n (pure vision mode)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


def render_text_to_image(text: str, image_size=(640, 640)) -> Image.Image:
    """Render text to image using VelloRenderer or PIL fallback"""
    try:
        from Renderer import VelloRenderer, VELLO_AVAILABLE
    except Exception:
        VELLO_AVAILABLE = False
        VelloRenderer = None

    text = str(text).strip()

    if VELLO_AVAILABLE and VelloRenderer is not None:
        try:
            vr = VelloRenderer(width=image_size[0], height=image_size[1], padding=20)
            pil_img = vr.render_batch_pil([text])[0]
            try:
                vr.shutdown()
            except Exception:
                pass
            return pil_img
        except Exception:
            pass

    # PIL fallback
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

    Generates both single-turn and multi-turn QA samples:
    - Single-turn: 1 question → 1 answer (loss weight 0.2)
    - Multi-turn: All questions → All answers (loss weight 1.0)

    Args:
        json_path: Path to llava_instruct_150k.json
        image_dir: Directory containing COCO images
        single_turn_ratio: Ratio of single-turn samples (default 0.5)
        max_turns: Maximum number of turns to include (default None = all)
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        single_turn_ratio: float = 0.5,
        max_turns: Optional[int] = None,
        render_questions: bool = True,
        seed: int = 42
    ):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.single_turn_ratio = single_turn_ratio
        self.max_turns = max_turns
        self.render_questions = render_questions
        self.rng = random.Random(seed)

        # Load conversations
        with open(self.json_path) as f:
            self.conversations = json.load(f)

        # Expand conversations into training samples
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """Prepare both single-turn and multi-turn samples"""
        samples = []

        for conv in self.conversations:
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

            # Limit turns if specified
            if self.max_turns:
                qa_pairs = qa_pairs[:self.max_turns]

            # Create multi-turn sample (all Q&A)
            multi_turn_sample = {
                'image_id': image_id,
                'qa_pairs': qa_pairs,
                'sample_type': 'multi_turn',
                'loss_weight': 1.0
            }
            samples.append(multi_turn_sample)

            # Create single-turn samples (one per Q&A pair)
            for qa in qa_pairs:
                single_turn_sample = {
                    'image_id': image_id,
                    'qa_pairs': [qa],
                    'sample_type': 'single_turn',
                    'loss_weight': 0.2
                }
                samples.append(single_turn_sample)

        # Shuffle samples
        self.rng.shuffle(samples)

        # Filter by single_turn_ratio
        # Keep all multi-turn, subsample single-turn
        multi_turn = [s for s in samples if s['sample_type'] == 'multi_turn']
        single_turn = [s for s in samples if s['sample_type'] == 'single_turn']

        # Calculate how many single-turn to keep
        target_single = int(len(multi_turn) * self.single_turn_ratio / (1 - self.single_turn_ratio))
        single_turn = single_turn[:target_single]

        final_samples = multi_turn + single_turn
        self.rng.shuffle(final_samples)

        return final_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample ready for OCRVL training.

        Returns:
            {
                'image': PIL Image (real image),
                'rendered_text': PIL Image or None (rendered questions if render_questions=True),
                'question_text': str (raw question text),
                'target_text': str (formatted answers),
                'sample_type': str ('single_turn' or 'multi_turn'),
                'loss_weight': float (0.2 or 1.0)
            }
        """
        sample = self.samples[idx]

        # Load real image
        image_path = self.image_dir / sample['image_id']

        # Try multiple path variations for COCO images
        if not image_path.exists():
            # Try with COCO_train2014_ prefix
            image_id_no_ext = sample['image_id'].rsplit('.', 1)[0]
            coco_path = self.image_dir / 'train2014' / f"COCO_train2014_{image_id_no_ext}.jpg"
            if coco_path.exists():
                image_path = coco_path
            else:
                # Try with different extensions
                for ext in ['.jpg', '.png', '.jpeg']:
                    test_path = self.image_dir / (image_id_no_ext + ext)
                    if test_path.exists():
                        image_path = test_path
                        break

        try:
            real_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Fallback: create blank image
            real_image = Image.new('RGB', (640, 640), color='white')

        # Format questions and answers
        qa_pairs = sample['qa_pairs']

        if len(qa_pairs) == 1:
            # Single-turn format
            question_text = qa_pairs[0]['q']
            answer_text = qa_pairs[0]['a']
        else:
            # Multi-turn format: Q1:\n...\nQ2:\n...\n
            questions = []
            answers = []
            for i, qa in enumerate(qa_pairs, 1):
                questions.append(f"Q{i}: {qa['q']}")
                answers.append(f"A{i}: {qa['a']}")

            question_text = '\n'.join(questions)
            answer_text = '\n'.join(answers)

        # Conditionally render questions as image
        rendered_questions = render_text_to_image(question_text) if self.render_questions else None

        return {
            'image': real_image,
            'rendered_text': rendered_questions,  # None if render_questions=False
            'question_text': question_text,  # Always provide raw text
            'target_text': answer_text,
            'sample_type': sample['sample_type'],
            'loss_weight': sample['loss_weight']
        }


def create_llava_collate_fn(tokenizer, ocr_adapter):
    """
    Collate function for LLaVA-Instruct dataset with loss weighting and random ordering.

    Supports two modes based on sample['rendered_text']:
    - RENDER=1 (rendered_text is Image): Pure vision mode with rendered questions as images
      - 50%: <|im_start|>user\n[real_image][rendered_questions]<|im_end|>\n<|im_start|>assistant\n[answers]
      - 50%: <|im_start|>user\n[rendered_questions][real_image]<|im_end|>\n<|im_start|>assistant\n[answers]
    - RENDER=0 (rendered_text is None): Hybrid mode with text prompts
      - <|im_start|>user\n[real_image]<|im_end|>\n<|im_start|>user\n[question_text]<|im_end|>\n<|im_start|>assistant\n[answers]

    Loss weight: 0.2 (single-turn) or 1.0 (multi-turn)
    """
    import random
    rng = random.Random(42)  # Use fixed seed for reproducibility per epoch

    def collate_fn(batch):
        batch_input_ids = []
        batch_labels = []
        batch_ocr_features = []
        batch_loss_weights = []

        # Qwen3-VL chat tokens
        user_start_ids = torch.tensor([151644, 872, 198], dtype=torch.long)  # <|im_start|>user\n
        user_end_ids = torch.tensor([151645, 198], dtype=torch.long)         # <|im_end|>\n
        assistant_start_ids = torch.tensor([151644, 77091, 198], dtype=torch.long)  # <|im_start|>assistant\n

        for sample in batch:
            real_image = sample['image']
            rendered_text = sample['rendered_text']  # PIL Image or None
            question_text = sample['question_text']  # str
            target_text = sample['target_text']
            loss_weight = sample['loss_weight']

            # Encode real image (no text instruction)
            img_input_ids, img_ocr_features = ocr_adapter.prepare_qwen_inputs_from_images(
                instruction="",
                images=[real_image],
                tokenizer=tokenizer,
                return_deepstack=True,
                render_instruction=False
            )

            # Encode target answers
            answer_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

            if rendered_text is not None:
                # RENDER=1: Pure vision mode - encode rendered questions as image
                text_input_ids, text_ocr_features = ocr_adapter.prepare_qwen_inputs_from_images(
                    instruction="",
                    images=[rendered_text],
                    tokenizer=tokenizer,
                    return_deepstack=True,
                    render_instruction=False
                )

                # Random ordering: 50% image first, 50% question first
                if rng.random() < 0.5:
                    # Order 1: <Real Image> + <Rendered Question>
                    full_input_ids = torch.cat([
                        user_start_ids,
                        img_input_ids.squeeze(0),
                        text_input_ids.squeeze(0),
                        user_end_ids,
                        assistant_start_ids,
                        answer_ids
                    ], dim=0)

                    # Combine OCR features: [real_image, rendered_questions]
                    if isinstance(img_ocr_features, tuple) and isinstance(text_ocr_features, tuple):
                        combined_final = img_ocr_features[0] + text_ocr_features[0]
                        combined_deepstack = img_ocr_features[1] + text_ocr_features[1]
                        ocr_features = (combined_final, combined_deepstack)
                    else:
                        ocr_features = img_ocr_features
                else:
                    # Order 2: <Rendered Question> + <Real Image>
                    full_input_ids = torch.cat([
                        user_start_ids,
                        text_input_ids.squeeze(0),
                        img_input_ids.squeeze(0),
                        user_end_ids,
                        assistant_start_ids,
                        answer_ids
                    ], dim=0)

                    # Combine OCR features: [rendered_questions, real_image]
                    if isinstance(img_ocr_features, tuple) and isinstance(text_ocr_features, tuple):
                        combined_final = text_ocr_features[0] + img_ocr_features[0]
                        combined_deepstack = text_ocr_features[1] + img_ocr_features[1]
                        ocr_features = (combined_final, combined_deepstack)
                    else:
                        ocr_features = text_ocr_features

                # Labels: mask everything before answers
                labels = full_input_ids.clone()
                vision_and_chat_count = len(user_start_ids) + len(img_input_ids.squeeze(0)) + len(text_input_ids.squeeze(0)) + len(user_end_ids) + len(assistant_start_ids)
                labels[:vision_and_chat_count] = -100

            else:
                # RENDER=0: Hybrid mode - use text prompt
                # Format: <|im_start|>user\n[real_image]<|im_end|>\n<|im_start|>user\n[question_text]<|im_end|>\n<|im_start|>assistant\n[answers]
                question_ids = tokenizer(question_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

                full_input_ids = torch.cat([
                    user_start_ids,
                    img_input_ids.squeeze(0),
                    user_end_ids,
                    user_start_ids,
                    question_ids,
                    user_end_ids,
                    assistant_start_ids,
                    answer_ids
                ], dim=0)

                # Only real image features (no rendered text)
                ocr_features = img_ocr_features

                # Labels: mask everything before answers
                labels = full_input_ids.clone()
                vision_and_chat_count = len(user_start_ids) + len(img_input_ids.squeeze(0)) + len(user_end_ids) + len(user_start_ids) + len(question_ids) + len(user_end_ids) + len(assistant_start_ids)
                labels[:vision_and_chat_count] = -100

            batch_input_ids.append(full_input_ids)
            batch_labels.append(labels)
            batch_ocr_features.append(ocr_features)
            batch_loss_weights.append(loss_weight)

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

        # Combine OCR features
        final_feats_combined = []
        deepstack_feats_combined = []

        for ocr_feat in batch_ocr_features:
            if isinstance(ocr_feat, tuple):
                final, ds = ocr_feat
                final_feats_combined.extend(final)
                deepstack_feats_combined.extend(ds)
            else:
                final_feats_combined.extend(ocr_feat)

        combined_ocr_features = (final_feats_combined, deepstack_feats_combined) if deepstack_feats_combined else final_feats_combined

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(padded_labels),
            "ocr_image_features": combined_ocr_features,
            "loss_weights": torch.tensor(batch_loss_weights, dtype=torch.float32)  # Per-sample loss weights
        }

    return collate_fn
