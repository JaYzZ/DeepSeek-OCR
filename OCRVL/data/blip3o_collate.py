#!/usr/bin/env python3
"""
Collate function for BLIP3o training - handles task-specific formatting
Separates concerns: Dataset returns raw data, collate handles all task logic
"""
import random
import torch

# Task 1: Image captioning instructions (variance)
IMAGE_CAPTION_INSTRUCTIONS = [
    "Describe the image:",
    "What's in this image?",
    "Describe what you see:",
    "What does this image show?",
    "Provide a description of the image:",
    "What do you see in the image?",
    "Can you describe this image?",
    "Tell me about this image:",
]

# Task 2: Text rendering OCR instructions (variance)
TEXT_OCR_INSTRUCTIONS = [
    "What does the text say?",
    "Read the text:",
    "Transcribe the text:",
    "What text is shown?",
    "Extract the text:",
    "What is written here?",
    "Read what's written:",
    "Transcribe what you see:",
]

# Task 3: Image-text matching question templates (variance)
MATCHING_QUESTIONS = [
    "Is the text caption correct for the image?",
    "Does the caption accurately describe the image?",
    "Is this caption matching the image content?",
    "Does the text correctly describe what's in the image?",
    "Is the description appropriate for the image?",
    "Does this text match the image?",
    "Is the caption suitable for the image?",
    "Does the given text describe the image correctly?",
    "Is this an accurate caption for the image?",
    "Does the caption align with the image content?",
]

# Task 3: Positive answer templates (variance)
POSITIVE_ANSWERS = [
    "Yes, the caption accurately describes the image content.",
    "Yes, the text correctly matches the image.",
    "Yes, this is an appropriate description of the image.",
    "Yes, the caption aligns well with the image content.",
    "Yes, the text provides an accurate description of the image.",
    "Yes, the description matches the image.",
    "Yes, this caption is correct for the image.",
]

# Task 3: Negative answer templates (variance)
NEGATIVE_ANSWERS = [
    "No, the caption does not match the image content.",
    "No, the text incorrectly describes the image.",
    "No, this caption is not appropriate for the image.",
    "No, the text does not align with the image content.",
    "No, this is an inaccurate description of the image.",
    "No, the description doesn't match the image.",
    "No, this caption is incorrect for the image.",
]


def create_blip3o_collate_fn(tokenizer, ocr_adapter, task_ratios=(0.4, 0.4, 0.2), seed=42):
    """
    Create collate function for BLIP3o dataset.

    Task distribution (per batch, not per sample):
    - Task 1 (40% of batches): Image captioning
    - Task 2 (40% of batches): Text rendering OCR
    - Task 3 (20% of batches): Contrastive matching (1 pos + 3 neg per sample → 4x batch size)

    Args:
        tokenizer: Tokenizer for text
        ocr_adapter: OCR adapter for encoding images/text
        task_ratios: (Task1, Task2, Task3) probabilities for batch task selection
        seed: Random seed for reproducibility

    Returns:
        Collate function that returns:
            - input_ids: [batch_size, seq_len]
            - attention_mask: [batch_size, seq_len]
            - labels: [batch_size, seq_len]
            - ocr_image_features: List of (final_feats, deepstack_feats) tuples
            - task_id: int (1, 2, or 3) for loss tracking
    """
    rng = random.Random(seed)
    batch_counter = [0]  # Mutable counter for deterministic task selection

    def collate_fn(batch):
        """
        Collate function that handles all task-specific formatting.

        Input batch: List of {"image": PIL.Image, "caption": str, "source": str}
        Output batch: {"input_ids", "attention_mask", "labels", "ocr_image_features", "task_id"}
        """
        # Determine task for THIS ENTIRE BATCH (not per sample)
        # Use batch counter for deterministic task selection across epochs
        batch_counter[0] += 1
        task_seed = seed + batch_counter[0]
        task_rng = random.Random(task_seed)
        task_choice = task_rng.random()

        if task_choice < task_ratios[0]:
            task = 1  # Image captioning
        elif task_choice < task_ratios[0] + task_ratios[1]:
            task = 2  # Text rendering OCR
        else:
            task = 3  # Contrastive matching

        batch_input_ids = []
        batch_labels = []
        batch_ocr_features = []

        if task == 3:
            # Task 3: Contrastive matching - 1 positive + 3 negatives per sample
            # This creates 4x accumulation to avoid OOM
            for sample in batch:
                image = sample["image"]
                pos_caption = sample["caption"]

                # Create 1 positive + 3 negatives
                for idx, is_positive in enumerate([True, False, False, False]):
                    if is_positive:
                        caption_to_use = pos_caption
                        answer = task_rng.choice(POSITIVE_ANSWERS)
                    else:
                        # Sample random negative caption from batch (excluding current)
                        neg_candidates = [s for s in batch if s["caption"] != pos_caption]
                        if not neg_candidates:
                            neg_candidates = batch  # Fallback if all captions are same
                        neg_sample = task_rng.choice(neg_candidates)
                        caption_to_use = neg_sample["caption"]
                        answer = task_rng.choice(NEGATIVE_ANSWERS)

                    # Encode real image (no instruction for Task 3)
                    img_input_ids, img_ocr_features = ocr_adapter.prepare_qwen_inputs_from_images(
                        instruction="",
                        images=[image],
                        tokenizer=tokenizer,
                        return_deepstack=True,
                        render_instruction=False
                    )

                    # Encode caption as rendered text (no instruction)
                    caption_input_ids, caption_ocr_features = ocr_adapter.prepare_qwen_inputs(
                        instruction="",
                        dense_text=caption_to_use,
                        tokenizer=tokenizer,
                        return_deepstack=True,
                        render_instruction=False
                    )

                    # Question (random template)
                    question = task_rng.choice(MATCHING_QUESTIONS)
                    question_ids = tokenizer(question, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

                    # Answer
                    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

                    # Concatenate: image_tokens + caption_tokens + question_tokens + answer_tokens
                    full_input_ids = torch.cat([
                        img_input_ids.squeeze(0),
                        caption_input_ids.squeeze(0),
                        question_ids,
                        answer_ids
                    ], dim=0)

                    # Combine OCR features from both image and rendered caption
                    if isinstance(img_ocr_features, tuple) and isinstance(caption_ocr_features, tuple):
                        # Both have (final_feats, deepstack_feats) format
                        combined_final = img_ocr_features[0] + caption_ocr_features[0]
                        combined_deepstack = img_ocr_features[1] + caption_ocr_features[1]
                        ocr_features = (combined_final, combined_deepstack)
                    else:
                        # Fallback (shouldn't happen with return_deepstack=True)
                        ocr_features = img_ocr_features

                    # Labels: mask everything except answer
                    labels = full_input_ids.clone()
                    answer_start_idx = len(img_input_ids.squeeze(0)) + len(caption_input_ids.squeeze(0)) + len(question_ids)
                    labels[:answer_start_idx] = -100  # Mask image + caption + question

                    batch_input_ids.append(full_input_ids)
                    batch_labels.append(labels)
                    batch_ocr_features.append(ocr_features)

        elif task == 1:
            # Task 1: Image captioning
            for sample in batch:
                image = sample["image"]
                caption = sample["caption"]

                # Random instruction template
                instruction = task_rng.choice(IMAGE_CAPTION_INSTRUCTIONS)

                # Encode image with instruction rendered as vision tokens
                input_ids, ocr_features = ocr_adapter.prepare_qwen_inputs_from_images(
                    instruction=instruction,
                    images=[image],
                    tokenizer=tokenizer,
                    return_deepstack=True,
                    render_instruction=True  # Render instruction as image
                )

                # Target caption
                response_ids = tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

                # Concatenate: vision_tokens + response_tokens
                full_input_ids = torch.cat([input_ids.squeeze(0), response_ids], dim=0)

                # Labels: mask vision tokens, predict caption
                labels = full_input_ids.clone()
                labels[:len(input_ids.squeeze(0))] = -100

                batch_input_ids.append(full_input_ids)
                batch_labels.append(labels)
                batch_ocr_features.append(ocr_features)

        else:  # task == 2
            # Task 2: Text rendering OCR
            for sample in batch:
                caption = sample["caption"]

                # Random instruction template
                instruction = task_rng.choice(TEXT_OCR_INSTRUCTIONS)

                # Encode rendered text with instruction as vision tokens
                input_ids, ocr_features = ocr_adapter.prepare_qwen_inputs(
                    instruction=instruction,
                    dense_text=caption,
                    tokenizer=tokenizer,
                    return_deepstack=True,
                    render_instruction=True  # Render instruction as image
                )

                # Target: same caption
                response_ids = tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

                # Concatenate: vision_tokens + response_tokens
                full_input_ids = torch.cat([input_ids.squeeze(0), response_ids], dim=0)

                # Labels: mask vision tokens, predict caption
                labels = full_input_ids.clone()
                labels[:len(input_ids.squeeze(0))] = -100

                batch_input_ids.append(full_input_ids)
                batch_labels.append(labels)
                batch_ocr_features.append(ocr_features)

        # Pad sequences to max length in batch
        max_len = max(ids.size(0) for ids in batch_input_ids)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(batch_input_ids, batch_labels):
            seq_len = input_ids.size(0)
            padding_len = max_len - seq_len

            # Pad input_ids
            padded_input_ids.append(
                torch.cat([input_ids, torch.full((padding_len,), pad_token_id, dtype=input_ids.dtype)])
            )

            # Pad labels (-100 for padding)
            padded_labels.append(
                torch.cat([labels, torch.full((padding_len,), -100, dtype=labels.dtype)])
            )

            # Attention mask (1 for real tokens, 0 for padding)
            attention_masks.append(
                torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)])
            )

        # Combine OCR features from all samples in batch
        # batch_ocr_features is a list of (final_feats, deepstack_feats) tuples
        # Model expects: (list_of_final_feats, list_of_deepstack_feats)
        final_feats_combined = []
        deepstack_feats_combined = []

        for ocr_feat in batch_ocr_features:
            if isinstance(ocr_feat, tuple):
                final, ds = ocr_feat
                final_feats_combined.extend(final)
                deepstack_feats_combined.extend(ds)
            else:
                # Fallback for list-only format
                final_feats_combined.extend(ocr_feat)

        # Return as tuple if we have deepstack features, otherwise just list
        combined_ocr_features = (final_feats_combined, deepstack_feats_combined) if deepstack_feats_combined else final_feats_combined

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(padded_labels),
            "ocr_image_features": combined_ocr_features,  # Properly combined OCR features
            "task_id": task,  # For separate loss tracking
        }

    return collate_fn
