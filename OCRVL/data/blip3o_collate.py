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
# Format: (question_template, separator_style)
# separator_style: 0=newline, 1=space, 2=colon+space
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


def create_blip3o_collate_fn(tokenizer, ocr_adapter):
    """
    Create SIMPLE collate function for BLIP3o dataset - just batch raw samples.

    Task logic is handled in the training loop, NOT here.

    Args:
        tokenizer: Tokenizer (not used, kept for compatibility)
        ocr_adapter: OCR adapter (not used, kept for compatibility)

    Returns:
        Collate function that returns raw batched samples
    """
    def collate_fn(batch):
        """
        Simple collate: just return the batch as-is.

        Input batch: List of {"image": PIL.Image, "caption": str, "source": str}
        Output batch: Same list (no processing)
        """
        return batch

    return collate_fn


def create_blip3o_collate_fn_OLD_MOVED_TO_TRAINING(tokenizer, ocr_adapter, task_ratios=(0.4, 0.4, 0.2), seed=42):
    """
    OLD VERSION - TASK LOGIC MOVED TO TRAINING LOOP

    This function is kept for reference but should not be used.
    Task selection and formatting now happens in train_one_epoch().
    """
    rng = random.Random(seed)
    batch_counter = [0]

    def collate_fn(batch):
        # DEPRECATED - use simple collate above
        batch_counter[0] += 1
        task_seed = seed + batch_counter[0]
        task_rng = random.Random(task_seed)
        task_choice = task_rng.random()

        if task_ratios[2] > 0:
            if task_choice < task_ratios[0]:
                task = 1
            elif task_choice < task_ratios[0] + task_ratios[1]:
                task = 2
            else:
                task = 3
        else:
            task_1_prob = task_ratios[0] / (task_ratios[0] + task_ratios[1])
            if task_choice < task_1_prob:
                task = 1
            else:
                task = 2

        batch_input_ids = []
        batch_labels = []
        batch_ocr_features = []

        if task == 3:
            # Task 3: Contrastive matching - 1 positive + 3 negatives per sample
            # PURE VISION FORMAT: [Natural image] + [Rendered "{instruction}: {caption}"] -> Yes/No
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

                    # Encode natural image (NO text instruction)
                    img_input_ids, img_ocr_features = ocr_adapter.prepare_qwen_inputs_from_images(
                        instruction="",  # EMPTY - pure vision!
                        images=[image],
                        tokenizer=tokenizer,
                        return_deepstack=True,
                        render_instruction=False
                    )

                    # Encode rendered template with natural formatting
                    question, sep_style = task_rng.choice(MATCHING_QUESTIONS)
                    if sep_style == 0:  # newline
                        template_text = f"{question}\n{caption_to_use}"
                    elif sep_style == 1:  # space
                        template_text = f"{question} {caption_to_use}"
                    else:  # 2: colon+space
                        template_text = f"{question} {caption_to_use}"

                    template_input_ids, template_ocr_features = ocr_adapter.prepare_qwen_inputs(
                        instruction="",  # EMPTY - pure vision!
                        dense_text=template_text,
                        tokenizer=tokenizer,
                        return_deepstack=True,
                        render_instruction=False
                    )

                    # Answer (text tokens)
                    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

                    # Concatenate: image_vision_tokens + template_vision_tokens + answer_text_tokens
                    full_input_ids = torch.cat([
                        img_input_ids.squeeze(0),
                        template_input_ids.squeeze(0),
                        answer_ids
                    ], dim=0)

                    # Combine OCR features from image and template
                    if isinstance(img_ocr_features, tuple) and isinstance(template_ocr_features, tuple):
                        combined_final = img_ocr_features[0] + template_ocr_features[0]
                        combined_deepstack = img_ocr_features[1] + template_ocr_features[1]
                        ocr_features = (combined_final, combined_deepstack)
                    else:
                        ocr_features = img_ocr_features

                    # Labels: mask vision tokens, predict answer only
                    labels = full_input_ids.clone()
                    vision_token_count = len(img_input_ids.squeeze(0)) + len(template_input_ids.squeeze(0))
                    labels[:vision_token_count] = -100  # Mask all vision tokens

                    batch_input_ids.append(full_input_ids)
                    batch_labels.append(labels)
                    batch_ocr_features.append(ocr_features)

        elif task == 1:
            # Task 1: Image captioning
            # PURE VISION FORMAT: Randomly choose instruction position
            # Type 1: [Natural image] + [Rendered instruction] -> Caption
            # Type 2: [Rendered instruction] + [Natural image] -> Caption
            for sample in batch:
                image = sample["image"]
                caption = sample["caption"]

                # Random instruction template
                instruction = task_rng.choice(IMAGE_CAPTION_INSTRUCTIONS)

                # Encode natural image (NO text instruction)
                img_input_ids, img_ocr_features = ocr_adapter.prepare_qwen_inputs_from_images(
                    instruction="",  # EMPTY - pure vision!
                    images=[image],
                    tokenizer=tokenizer,
                    return_deepstack=True,
                    render_instruction=False
                )

                # Encode rendered instruction as vision tokens
                inst_input_ids, inst_ocr_features = ocr_adapter.prepare_qwen_inputs(
                    instruction="",  # EMPTY - pure vision!
                    dense_text=instruction,
                    tokenizer=tokenizer,
                    return_deepstack=True,
                    render_instruction=False
                )

                # Target caption (text tokens)
                response_ids = tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

                # Qwen3-VL chat tokens: <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n
                user_start_ids = torch.tensor([151644, 872, 198], dtype=torch.long)  # <|im_start|>user\n
                user_end_ids = torch.tensor([151645, 198], dtype=torch.long)         # <|im_end|>\n
                assistant_start_ids = torch.tensor([151644, 77091, 198], dtype=torch.long)  # <|im_start|>assistant\n

                # Random ordering: instruction first (50%) or last (50%)
                instruction_first = task_rng.random() < 0.5

                if instruction_first:
                    # Type 1: <|im_start|>user\n[Instruction][Image]<|im_end|>\n<|im_start|>assistant\n[Caption]
                    full_input_ids = torch.cat([
                        user_start_ids,
                        inst_input_ids.squeeze(0),
                        img_input_ids.squeeze(0),
                        user_end_ids,
                        assistant_start_ids,
                        response_ids
                    ], dim=0)
                    # Combine OCR features: instruction first
                    if isinstance(img_ocr_features, tuple) and isinstance(inst_ocr_features, tuple):
                        combined_final = inst_ocr_features[0] + img_ocr_features[0]
                        combined_deepstack = inst_ocr_features[1] + img_ocr_features[1]
                        ocr_features = (combined_final, combined_deepstack)
                    else:
                        ocr_features = inst_ocr_features
                else:
                    # Type 2: <|im_start|>user\n[Image][Instruction]<|im_end|>\n<|im_start|>assistant\n[Caption]
                    full_input_ids = torch.cat([
                        user_start_ids,
                        img_input_ids.squeeze(0),
                        inst_input_ids.squeeze(0),
                        user_end_ids,
                        assistant_start_ids,
                        response_ids
                    ], dim=0)
                    # Combine OCR features: image first
                    if isinstance(img_ocr_features, tuple) and isinstance(inst_ocr_features, tuple):
                        combined_final = img_ocr_features[0] + inst_ocr_features[0]
                        combined_deepstack = img_ocr_features[1] + inst_ocr_features[1]
                        ocr_features = (combined_final, combined_deepstack)
                    else:
                        ocr_features = img_ocr_features

                # Labels: mask vision tokens + chat tokens, predict caption only
                labels = full_input_ids.clone()
                # Mask: user_start (3) + vision_tokens + user_end (2) + assistant_start (3)
                vision_token_count = len(img_input_ids.squeeze(0)) + len(inst_input_ids.squeeze(0))
                chat_and_vision_count = 3 + vision_token_count + 2 + 3  # Total tokens before caption
                labels[:chat_and_vision_count] = -100  # Mask everything before caption

                batch_input_ids.append(full_input_ids)
                batch_labels.append(labels)
                batch_ocr_features.append(ocr_features)

        else:  # task == 2
            # Task 2: Text rendering OCR
            # PURE VISION FORMAT: Randomly choose instruction position
            # Type 1: [Rendered caption] + [Rendered instruction] -> Caption
            # Type 2: [Rendered instruction] + [Rendered caption] -> Caption
            for sample in batch:
                caption = sample["caption"]

                # Random instruction template
                instruction = task_rng.choice(TEXT_OCR_INSTRUCTIONS)

                # Encode rendered caption as vision tokens (NO text instruction)
                caption_input_ids, caption_ocr_features = ocr_adapter.prepare_qwen_inputs(
                    instruction="",  # EMPTY - pure vision!
                    dense_text=caption,
                    tokenizer=tokenizer,
                    return_deepstack=True,
                    render_instruction=False
                )

                # Encode rendered instruction as vision tokens
                inst_input_ids, inst_ocr_features = ocr_adapter.prepare_qwen_inputs(
                    instruction="",  # EMPTY - pure vision!
                    dense_text=instruction,
                    tokenizer=tokenizer,
                    return_deepstack=True,
                    render_instruction=False
                )

                # Target: same caption (text tokens)
                response_ids = tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

                # Qwen3-VL chat tokens: <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n
                user_start_ids = torch.tensor([151644, 872, 198], dtype=torch.long)  # <|im_start|>user\n
                user_end_ids = torch.tensor([151645, 198], dtype=torch.long)         # <|im_end|>\n
                assistant_start_ids = torch.tensor([151644, 77091, 198], dtype=torch.long)  # <|im_start|>assistant\n

                # Random ordering: instruction first (50%) or last (50%)
                instruction_first = task_rng.random() < 0.5

                if instruction_first:
                    # Type 1: <|im_start|>user\n[Instruction][Caption]<|im_end|>\n<|im_start|>assistant\n[Response]
                    full_input_ids = torch.cat([
                        user_start_ids,
                        inst_input_ids.squeeze(0),
                        caption_input_ids.squeeze(0),
                        user_end_ids,
                        assistant_start_ids,
                        response_ids
                    ], dim=0)
                    # Combine OCR features: instruction first
                    if isinstance(caption_ocr_features, tuple) and isinstance(inst_ocr_features, tuple):
                        combined_final = inst_ocr_features[0] + caption_ocr_features[0]
                        combined_deepstack = inst_ocr_features[1] + caption_ocr_features[1]
                        ocr_features = (combined_final, combined_deepstack)
                    else:
                        ocr_features = inst_ocr_features
                else:
                    # Type 2: <|im_start|>user\n[Caption][Instruction]<|im_end|>\n<|im_start|>assistant\n[Response]
                    full_input_ids = torch.cat([
                        user_start_ids,
                        caption_input_ids.squeeze(0),
                        inst_input_ids.squeeze(0),
                        user_end_ids,
                        assistant_start_ids,
                        response_ids
                    ], dim=0)
                    # Combine OCR features: caption first
                    if isinstance(caption_ocr_features, tuple) and isinstance(inst_ocr_features, tuple):
                        combined_final = caption_ocr_features[0] + inst_ocr_features[0]
                        combined_deepstack = caption_ocr_features[1] + inst_ocr_features[1]
                        ocr_features = (combined_final, combined_deepstack)
                    else:
                        ocr_features = caption_ocr_features

                # Labels: mask vision tokens + chat tokens, predict caption only
                labels = full_input_ids.clone()
                # Mask: user_start (3) + vision_tokens + user_end (2) + assistant_start (3)
                vision_token_count = len(caption_input_ids.squeeze(0)) + len(inst_input_ids.squeeze(0))
                chat_and_vision_count = 3 + vision_token_count + 2 + 3  # Total tokens before caption
                labels[:chat_and_vision_count] = -100  # Mask everything before caption

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
