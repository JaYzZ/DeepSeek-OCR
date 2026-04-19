#!/usr/bin/env python3
"""
Generate composite images from transparent evaluation results.

Creates a composite image for each sample showing input image(s) with
ground truth and generated text overlaid.
"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from project_paths import resolve_project_path


def extract_display_output(full_output: str) -> str:
    """Return the final visible answer span after the last thinking marker."""
    return str(full_output or "").rsplit("</think>", 1)[-1].strip()


def text_width(text: str, font) -> int:
    bbox = font.getbbox(text or " ")
    return bbox[2] - bbox[0]


def line_height(font, extra_spacing: int = 4) -> int:
    bbox = font.getbbox("Ag")
    return (bbox[3] - bbox[1]) + extra_spacing


def wrap_paragraph(paragraph: str, font, max_width: int) -> list[str]:
    if not paragraph:
        return [""]

    words = paragraph.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        if current_line and text_width(test_line, font) <= max_width:
            current_line.append(word)
            continue
        if not current_line and text_width(word, font) <= max_width:
            current_line = [word]
            continue
        if current_line:
            lines.append(' '.join(current_line))
            current_line = []

        if text_width(word, font) <= max_width:
            current_line = [word]
            continue

        chunk = ""
        for ch in word:
            candidate = f"{chunk}{ch}"
            if chunk and text_width(candidate, font) > max_width:
                lines.append(chunk)
                chunk = ch
            else:
                chunk = candidate
        current_line = [chunk]

    if current_line:
        lines.append(' '.join(current_line))

    return lines or [""]


def wrap_text(text: str, font, max_width: int) -> list[str]:
    """Wrap text to fit within max_width while preserving explicit newlines."""
    if not text:
        return []

    lines: list[str] = []
    for paragraph in str(text).splitlines():
        lines.extend(wrap_paragraph(paragraph, font, max_width))
    return lines or [""]


def create_composite_image(
    result: dict,
    repo_root: Path,
    output_path: Path,
    img_width: int = 800,
    font_size: int = 16,
    padding: int = 20,
):
    """Create composite image for a single sample."""

    # Load images
    images = []
    for img_path in result.get('images', []):
        full_path = resolve_project_path(img_path, repo_root=repo_root)
        if full_path.exists():
            img = Image.open(full_path).convert('RGB')
            images.append(img)

    # For bbox_ocr tasks, draw bbox on the first image
    if result.get('task') == 'bbox_ocr' and images:
        instruction = result.get('instruction', '')
        # Parse bbox from instruction: "Transcribe the text in [x1, y1, x2, y2]:<image>"
        import re
        bbox_match = re.search(r'\[([^\]]+)\]', instruction)
        if bbox_match:
            try:
                bbox_str = bbox_match.group(1)
                bbox_coords = [float(x.strip()) for x in bbox_str.split(',')]
                if len(bbox_coords) == 4:
                    # Draw bbox on image (Qwen3VL format: [0, 1000] coordinates)
                    img = images[0]
                    draw = ImageDraw.Draw(img)
                    x1, y1, x2, y2 = bbox_coords
                    # Convert from [0, 1000] to absolute coordinates
                    abs_x1 = int(x1 * img.width / 1000)
                    abs_y1 = int(y1 * img.height / 1000)
                    abs_x2 = int(x2 * img.width / 1000)
                    abs_y2 = int(y2 * img.height / 1000)
                    # Draw red rectangle with thick outline
                    draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2],
                                 outline='red', width=5)
                    # Add semi-transparent overlay (optional visualization)
                    # Create a semi-transparent overlay
                    overlay = Image.new('RGBA', img.size, (255, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2],
                                          fill=(255, 0, 0, 30))  # 30/255 alpha
                    # Composite overlay onto image
                    images[0] = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse bbox for {result.get('id')}: {e}")

    if not images:
        print(f"Warning: No images found for {result.get('id')}")
        return

    # For VQA, skip the rendered question image (second image)
    # But keep the question text in instruction field
    instruction = result.get('instruction', '')
    if result.get('task') == 'visual_question_answering' and len(images) == 2:
        images = [images[0]]  # Keep only the content image
        # Note: instruction already contains "Answer the question:" from metadata

    # Resize images to fit width
    max_img_width = img_width - 2 * padding
    resized_images = []
    for img in images:
        ratio = max_img_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_img_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(img)

    # Calculate total height
    total_img_height = sum(img.height for img in resized_images)
    spacing = 15

    # Fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size + 4)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 2)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    # Get text content
    ground_truth = result.get('ground_truth', '')
    full_output = str(result.get('generated_answer', '') or '')
    generated = str(
        result.get('generated_answer_display')
        or extract_display_output(full_output)
        or full_output
        or '[EMPTY]'
    )
    question_text = result.get('question_text', '')

    # Measure text with explicit newline preservation
    text_width = max_img_width
    text_line_height = line_height(text_font, extra_spacing=2)
    _, gt_lines = measure_text_height(ground_truth, text_font, text_width, text_line_height)
    _, gen_lines = measure_text_height(generated, text_font, text_width, text_line_height)
    _, q_lines = measure_text_height(question_text, text_font, text_width, text_line_height) if question_text else (0, 0)

    text_area_height = 0
    if instruction:
        text_area_height += 30  # Instruction label
        _, inst_lines = measure_text_height(instruction, text_font, text_width, text_line_height)
        text_area_height += inst_lines * text_line_height
        if question_text:
            text_area_height += 5  # Spacing before question
            text_area_height += 30  # Question label
            text_area_height += q_lines * text_line_height
        text_area_height += 10  # Spacing after instruction/question
    text_area_height += 30  # Ground truth label + content
    text_area_height += gt_lines * text_line_height
    text_area_height += 10  # Spacing after ground truth
    text_area_height += 30  # Generated label + content
    text_area_height += gen_lines * text_line_height
    text_area_height += 60  # Title and spacing

    total_height = total_img_height + text_area_height + (len(images) - 1) * spacing + 3 * padding

    # Create composite image
    composite = Image.new('RGB', (img_width, total_height), color='white')
    draw = ImageDraw.Draw(composite)

    # Draw title
    sample_id = result.get('id', 'unknown')
    task = result.get('task', 'unknown').replace('_', ' ').title()
    title = f"Sample: {sample_id} | Task: {task}"
    draw.text((padding, padding), title, fill='white', font=title_font)

    # Title background
    title_bbox = draw.textbbox((padding, padding), title, font=title_font)
    draw.rectangle([padding - 5, padding - 5, img_width - padding + 5, title_bbox[3] + 10],
                   fill='#2196F3', outline=None)
    draw.text((padding, padding), title, fill='white', font=title_font)

    y_offset = title_bbox[3] + 15

    # Draw images
    for img in resized_images:
        composite.paste(img, (padding, y_offset))
        y_offset += img.height + spacing

    y_offset += 10

    # Draw instruction/question (if present)
    # For VQA, show both "Answer the question:" instruction and the actual question text
    if instruction:
        draw.text((padding, y_offset), "Instruction:", fill='#9C27B0', font=label_font)
        y_offset += 20

        lines = wrap_text(instruction, text_font, text_width)
        for line in lines:
            draw.text((padding, y_offset), line, fill='#333', font=text_font)
            y_offset += text_line_height

        # For VQA, also show the actual question text
        question_text = result.get('question_text', '')
        if question_text:
            y_offset += 5
            draw.text((padding, y_offset), "Question:", fill='#9C27B0', font=label_font)
            y_offset += 20
            lines = wrap_text(question_text, text_font, text_width)
            for line in lines:
                draw.text((padding, y_offset), line, fill='#333', font=text_font)
                y_offset += text_line_height

        y_offset += 10

    # Draw ground truth
    draw.text((padding, y_offset), "Ground Truth:", fill='#4CAF50', font=label_font)
    y_offset += 20
    lines = wrap_text(ground_truth, text_font, text_width)
    for line in lines:
        draw.text((padding, y_offset), line, fill='#333', font=text_font)
        y_offset += text_line_height
    y_offset += 10

    draw.text((padding, y_offset), "Answer:", fill='#FF9800', font=label_font)
    y_offset += 20
    lines = wrap_text(generated, text_font, text_width)
    for line in lines:
        draw.text((padding, y_offset), line, fill='#333', font=text_font)
        y_offset += text_line_height

    # Save composite
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output_path, optimize=True, quality=95)
    print(f"✓ {output_path.name}")


def measure_text_height(text: str, font, max_width: int, measured_line_height: int | None = None) -> tuple[int, int]:
    """Measure text height and line count."""
    lines = wrap_text(text, font, max_width)
    line_count = len(lines)
    height = line_count * (measured_line_height or line_height(font))
    return height, line_count


def generate_composite_images(
    results_json: Path,
    output_dir: Path,
    repo_root: Path,
):
    """Generate composite images for all samples."""

    with open(results_json, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    step = data.get('step', 'unknown')

    output_dir = output_dir / f"step_{step}_composite"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(results)} composite images...")

    for idx, result in enumerate(results, 1):
        sample_id = result.get('id', f'sample_{idx}')
        output_path = output_dir / f"{idx:02d}_{sample_id}.png"

        create_composite_image(
            result=result,
            repo_root=repo_root,
            output_path=output_path,
        )

    print(f"✓ Generated {len(results)} composite images in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate composite images from transparent eval results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to step_N_results.json file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: same dir as results/step_N_composite/)",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repository root (default: auto-detect)",
    )

    args = parser.parse_args()

    results_json = Path(args.results).resolve()
    if not results_json.exists():
        print(f"Error: Results file not found: {results_json}")
        return 1

    # Auto-detect output path
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = results_json.parent

    # Auto-detect repo root
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        repo_root = results_json
        for _ in range(6):
            repo_root = repo_root.parent
        repo_root = repo_root.parent.parent.parent

    generate_composite_images(
        results_json=results_json,
        output_dir=output_dir,
        repo_root=repo_root,
    )

    return 0


if __name__ == "__main__":
    exit(main())
