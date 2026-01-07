#!/usr/bin/env python3
"""
BLIP3o task formatting - CPU-only preparation, GPU encoding moved to training loop

Returns PIL images + metadata instead of encoded features.
This enables parallel data loading with num_workers > 0 and batch encoding.
"""
import random
import torch
from PIL import Image

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
    "Please transcribe all text in the image.",
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

# Import Vello renderer for text rendering (CPU-only)
_VELLO_RENDERER = None
_VELLO_AVAILABLE = False

def _get_vello_renderer(image_size=(640, 640)):
    """Get cached Vello renderer instance"""
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

def _render_text(text, image_size=(640, 640)):
    """Render text to PIL image using Vello (CPU-only)"""
    vr = _get_vello_renderer(image_size)
    if vr is not None:
        try:
            return vr.render_batch_pil([text])[0]
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


def format_caption_batch(batch, tokenizer, rng):
    """
    Image captioning (CPU-only preparation)
    Format: 50% chance: [Natural image] alone -> Caption
            50% chance: [Natural image] + [Rendered instruction] -> Caption (random order)

    Returns:
        List with 1 dict containing PIL images + metadata (no encoding)
    """
    all_images = []  # Collect PIL images for batch encoding
    image_counts = []  # Track how many images per sample
    response_texts = []  # Response texts
    orderings = []  # Track image ordering per sample

    for sample in batch:
        natural_image = sample["image"]  # PIL Image
        caption = sample["caption"]

        # 50% chance: just use original image with no text
        if rng.random() < 0.5:
            all_images.append(natural_image)
            image_counts.append(1)
            orderings.append("img_only")
        else:
            # Use image + rendered instruction
            instruction = rng.choice(IMAGE_CAPTION_INSTRUCTIONS)
            instruction_image = _render_text(instruction)

            # Random ordering
            if rng.random() < 0.5:
                # Instruction first
                all_images.extend([instruction_image, natural_image])
                orderings.append("inst_first")
            else:
                # Image first
                all_images.extend([natural_image, instruction_image])
                orderings.append("img_first")

            image_counts.append(2)

        response_texts.append(caption)

    return [{
        "images": all_images,  # PIL images for GPU encoding
        "image_counts": image_counts,
        "response_texts": response_texts,
        "orderings": orderings,
        "tokenizer": tokenizer,
    }]


def format_ocr_batch(batch, tokenizer, rng):
    """
    OCR Transcription Task (CPU-only preparation)

    For BLIP3o (text captions):
        Format: Rendered caption text with embedded instruction -> Caption text
        Renders: "{caption_text}\n\nPlease transcribe all text in the image"
        Model outputs: caption_text

    For DocLayNet (document images):
        Format: Document image + optional rendered OCR instruction -> Layout description
        50% chance: [Document image] alone -> Layout description
        50% chance: [Document image] + [Rendered OCR instruction] -> Layout description (random order)
        Model outputs: layout description

    Returns:
        List with 1 dict containing PIL images + metadata (no encoding)
    """
    all_images = []
    image_counts = []
    response_texts = []
    orderings = []

    for sample in batch:
        # Check if this is a real image (DocLayNet) or text caption (BLIP3o)
        has_image = "image" in sample
        caption = sample["caption"]

        if has_image:
            # DocLayNet: Real document image
            # Format similar to caption task, but with OCR instructions
            document_image = sample["image"]

            # 50% chance: use document image with rendered OCR instruction
            if rng.random() < 0.5:
                # Use document image + rendered OCR instruction
                instruction = rng.choice(TEXT_OCR_INSTRUCTIONS)
                instruction_image = _render_text(instruction)

                # Random ordering
                if rng.random() < 0.5:
                    # Instruction first
                    all_images.extend([instruction_image, document_image])
                    orderings.append("inst_first")
                else:
                    # Document first
                    all_images.extend([document_image, instruction_image])
                    orderings.append("img_first")

                image_counts.append(2)
            else:
                # Just document image, no instruction
                all_images.append(document_image)
                image_counts.append(1)
                orderings.append("img_only")

        else:
            # BLIP3o: Text caption - render with embedded OCR instruction
            instruction = rng.choice(TEXT_OCR_INSTRUCTIONS)
            combined_text = f"{caption}\n\n{instruction}"
            rendered_image = _render_text(combined_text)

            all_images.append(rendered_image)
            image_counts.append(1)
            orderings.append("ocr_single")

        response_texts.append(caption)

    return [{
        "images": all_images,
        "image_counts": image_counts,
        "response_texts": response_texts,
        "orderings": orderings,
        "tokenizer": tokenizer,
    }]


def format_task3_batch(batch, tokenizer, rng):
    """
    Task 3: Contrastive matching (CPU-only preparation)

    Returns 4 sub-batches (one per shift) with PIL images + metadata
    """
    if len(batch) < 4:
        # Fallback to caption task
        return format_caption_batch(batch, tokenizer, rng)

    N = len(batch)
    images = [sample["image"] for sample in batch]
    captions = [sample["caption"] for sample in batch]

    # Create 4 sub-batches (one for each shift: 0, 1, 2, 3)
    sub_batches = []

    for shift in range(4):
        all_images = []
        image_counts = []
        response_texts = []
        orderings = []  # Not used for task 3 (fixed order: img + template)

        is_positive = (shift == 0)

        for i in range(N):
            natural_image = images[i]

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

            template_image = _render_text(template_text)

            # Fixed order: natural image + template
            all_images.extend([natural_image, template_image])
            image_counts.append(2)
            response_texts.append(answer)
            orderings.append("img_template")

        sub_batches.append({
            "images": all_images,
            "image_counts": image_counts,
            "response_texts": response_texts,
            "orderings": orderings,
            "tokenizer": tokenizer,
        })

    return sub_batches  # 4 sub-batches
