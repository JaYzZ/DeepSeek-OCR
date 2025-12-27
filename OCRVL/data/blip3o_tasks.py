#!/usr/bin/env python3
"""
BLIP3o task formatting - handles task-specific encoding in training loop.

This module provides functions to format raw BLIP3o samples into model inputs
for different tasks. Task selection and expansion happens in the training loop.
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
    ("Is the following caption correct for the image?", 1),
    ("Does this caption accurately describe the image?", 1),
    ("Is this caption matching the image content?", 1),
    ("Does the text correctly describe what's in the image?", 1),
    ("Is the description appropriate for the image?", 1),
    ("Does this text match the image?", 1),
    ("Caption to verify:", 2),
    ("Check if this caption is correct:", 1),
    ("Verify the caption:", 2),
    ("Is this an accurate description?", 1),
]

# Task 3: Positive/Negative answer templates
POSITIVE_ANSWERS = [
    "Yes, the caption accurately describes the image content.",
    "Yes, the text correctly matches the image.",
    "Yes, this is an appropriate description of the image.",
    "Yes, the caption aligns well with the image content.",
    "Yes, the text provides an accurate description of the image.",
    "Yes, the description matches the image.",
    "Yes, this caption is correct for the image.",
]

NEGATIVE_ANSWERS = [
    "No, the caption does not match the image content.",
    "No, the text incorrectly describes the image.",
    "No, this caption is not appropriate for the image.",
    "No, the text does not align with the image content.",
    "No, this is an inaccurate description of the image.",
    "No, the description doesn't match the image.",
    "No, this caption is incorrect for the image.",
]


def format_task1_batch(batch, tokenizer, ocr_adapter, rng):
    """
    Task 1: Image captioning
    Format: [Natural image] + [Rendered instruction] -> Caption (or reverse order)

    Args:
        batch: List of {"image": PIL.Image, "caption": str}
        tokenizer: Tokenizer
        ocr_adapter: OCR adapter for encoding
        rng: Random number generator for variance

    Returns:
        List with 1 formatted batch (for consistency with task 3 which returns 4)
    """
    batch_input_ids = []
    batch_labels = []
    batch_ocr_features = []

    for sample in batch:
        image = sample["image"]
        caption = sample["caption"]
        instruction = rng.choice(IMAGE_CAPTION_INSTRUCTIONS)

        # Encode natural image (NO text instruction)
        img_input_ids, img_ocr_features = ocr_adapter.prepare_qwen_inputs_from_images(
            instruction="",
            images=[image],
            tokenizer=tokenizer,
            return_deepstack=True,
            render_instruction=False
        )

        # Encode rendered instruction as vision tokens
        inst_input_ids, inst_ocr_features = ocr_adapter.prepare_qwen_inputs(
            instruction="",
            dense_text=instruction,
            tokenizer=tokenizer,
            return_deepstack=True,
            render_instruction=False
        )

        # Target caption (text tokens)
        response_ids = tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

        # Random ordering: instruction first (50%) or last (50%)
        if rng.random() < 0.5:
            full_input_ids = torch.cat([inst_input_ids.squeeze(0), img_input_ids.squeeze(0), response_ids], dim=0)
            if isinstance(img_ocr_features, tuple):
                ocr_features = (inst_ocr_features[0] + img_ocr_features[0], inst_ocr_features[1] + img_ocr_features[1])
            else:
                ocr_features = inst_ocr_features
        else:
            full_input_ids = torch.cat([img_input_ids.squeeze(0), inst_input_ids.squeeze(0), response_ids], dim=0)
            if isinstance(img_ocr_features, tuple):
                ocr_features = (img_ocr_features[0] + inst_ocr_features[0], img_ocr_features[1] + inst_ocr_features[1])
            else:
                ocr_features = img_ocr_features

        # Labels: mask vision tokens, predict caption
        labels = full_input_ids.clone()
        vision_token_count = len(img_input_ids.squeeze(0)) + len(inst_input_ids.squeeze(0))
        labels[:vision_token_count] = -100

        batch_input_ids.append(full_input_ids)
        batch_labels.append(labels)
        batch_ocr_features.append(ocr_features)

    return [_pad_and_combine(batch_input_ids, batch_labels, batch_ocr_features, tokenizer)]


def format_task2_batch(batch, tokenizer, ocr_adapter, rng):
    """
    Task 2: Text rendering OCR
    Format: [Rendered caption] + [Rendered instruction] -> Caption (or reverse order)

    Returns:
        List with 1 formatted batch (for consistency with task 3 which returns 4)
    """
    batch_input_ids = []
    batch_labels = []
    batch_ocr_features = []

    for sample in batch:
        caption = sample["caption"]
        instruction = rng.choice(TEXT_OCR_INSTRUCTIONS)

        # Encode rendered caption as vision tokens
        caption_input_ids, caption_ocr_features = ocr_adapter.prepare_qwen_inputs(
            instruction="",
            dense_text=caption,
            tokenizer=tokenizer,
            return_deepstack=True,
            render_instruction=False
        )

        # Encode rendered instruction as vision tokens
        inst_input_ids, inst_ocr_features = ocr_adapter.prepare_qwen_inputs(
            instruction="",
            dense_text=instruction,
            tokenizer=tokenizer,
            return_deepstack=True,
            render_instruction=False
        )

        # Target: same caption (text tokens)
        response_ids = tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

        # Random ordering
        if rng.random() < 0.5:
            full_input_ids = torch.cat([inst_input_ids.squeeze(0), caption_input_ids.squeeze(0), response_ids], dim=0)
            if isinstance(caption_ocr_features, tuple):
                ocr_features = (inst_ocr_features[0] + caption_ocr_features[0], inst_ocr_features[1] + caption_ocr_features[1])
            else:
                ocr_features = inst_ocr_features
        else:
            full_input_ids = torch.cat([caption_input_ids.squeeze(0), inst_input_ids.squeeze(0), response_ids], dim=0)
            if isinstance(caption_ocr_features, tuple):
                ocr_features = (caption_ocr_features[0] + inst_ocr_features[0], caption_ocr_features[1] + inst_ocr_features[1])
            else:
                ocr_features = caption_ocr_features

        # Labels: mask vision tokens, predict caption
        labels = full_input_ids.clone()
        vision_token_count = len(caption_input_ids.squeeze(0)) + len(inst_input_ids.squeeze(0))
        labels[:vision_token_count] = -100

        batch_input_ids.append(full_input_ids)
        batch_labels.append(labels)
        batch_ocr_features.append(ocr_features)

    return [_pad_and_combine(batch_input_ids, batch_labels, batch_ocr_features, tokenizer)]


def format_task3_batch(batch, tokenizer, ocr_adapter, rng):
    """
    Task 3: Contrastive matching with efficient negative sampling via caption shifting

    Given batch of N samples with (image_i, caption_i) pairs:
    - Forward 1 (shift=0): All positives (img_i, cap_i) → "Yes"
    - Forward 2 (shift=1): All negatives (img_i, cap_{i+1}) → "No"
    - Forward 3 (shift=2): All negatives (img_i, cap_{i+2}) → "No"
    - Forward 4 (shift=3): All negatives (img_i, cap_{i+3}) → "No"

    Returns 4 sub-batches (one per shift) to avoid OOM.
    Each sub-batch has N samples.

    Requirement: N >= 4 for proper negative sampling
    """
    if len(batch) < 4:
        # Need at least 4 samples for shifting
        # Fall back to task 1
        return [format_task1_batch(batch, tokenizer, ocr_adapter, rng)]

    N = len(batch)
    images = [sample["image"] for sample in batch]
    captions = [sample["caption"] for sample in batch]

    # Encode all images once (reuse across 4 forwards)
    encoded_images = []
    for image in images:
        img_input_ids, img_ocr_features = ocr_adapter.prepare_qwen_inputs_from_images(
            instruction="",
            images=[image],
            tokenizer=tokenizer,
            return_deepstack=True,
            render_instruction=False
        )
        encoded_images.append((img_input_ids, img_ocr_features))

    # Create 4 sub-batches (one for each shift: 0, 1, 2, 3)
    sub_batches = []

    for shift in range(4):
        batch_input_ids = []
        batch_labels = []
        batch_ocr_features = []

        # Determine if positive (shift=0) or negative (shift>0)
        is_positive = (shift == 0)

        for i in range(N):
            # Get pre-encoded image
            img_input_ids, img_ocr_features = encoded_images[i]

            # Get caption (shifted for negatives)
            caption_idx = (i + shift) % N
            caption = captions[caption_idx]

            # Choose answer
            if is_positive:
                answer = rng.choice(POSITIVE_ANSWERS)
            else:
                answer = rng.choice(NEGATIVE_ANSWERS)

            # Render question + caption template
            question, sep_style = rng.choice(MATCHING_QUESTIONS)
            if sep_style == 0:
                template_text = f"{question}\n{caption}"
            elif sep_style == 1:
                template_text = f"{question} {caption}"
            else:
                template_text = f"{question} {caption}"

            template_input_ids, template_ocr_features = ocr_adapter.prepare_qwen_inputs(
                instruction="",
                dense_text=template_text,
                tokenizer=tokenizer,
                return_deepstack=True,
                render_instruction=False
            )

            # Answer (text tokens)
            answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

            # Concatenate: image + template + answer
            full_input_ids = torch.cat([
                img_input_ids.squeeze(0),
                template_input_ids.squeeze(0),
                answer_ids
            ], dim=0)

            # Combine OCR features
            if isinstance(img_ocr_features, tuple):
                ocr_features = (
                    img_ocr_features[0] + template_ocr_features[0],
                    img_ocr_features[1] + template_ocr_features[1]
                )
            else:
                ocr_features = img_ocr_features

            # Labels: mask vision tokens, predict answer only
            labels = full_input_ids.clone()
            vision_token_count = len(img_input_ids.squeeze(0)) + len(template_input_ids.squeeze(0))
            labels[:vision_token_count] = -100

            batch_input_ids.append(full_input_ids)
            batch_labels.append(labels)
            batch_ocr_features.append(ocr_features)

        # Pad and combine this sub-batch
        formatted = _pad_and_combine(batch_input_ids, batch_labels, batch_ocr_features, tokenizer)
        sub_batches.append(formatted)

    return sub_batches  # Returns list of 4 batches (shift 0, 1, 2, 3)


def _pad_and_combine(batch_input_ids, batch_labels, batch_ocr_features, tokenizer):
    """Helper: Pad sequences and combine features"""
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
    }
