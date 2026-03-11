#!/usr/bin/env python3
"""
Visualize DocLayNet markdown examples.

Shows 1x3 layout:
- Left: Original image with bbox overlays
- Middle: Raw markdown text
- Right: Rendered markdown (HTML preview)
"""

import argparse
import base64
import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from markdown import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    markdown = lambda x: x  # Fallback: return as-is


def load_jsonl_samples(jsonl_path: Path, num_samples: int = 5) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if len(samples) >= num_samples:
                    break
    return samples


def draw_text_pil(text: str, width: int = 600, max_height: int = 800,
                  bg_color: str = 'white', text_color: str = 'black',
                  font_size: int = 14) -> np.ndarray:
    """Draw text on PIL image."""
    # Create image
    img = Image.new('RGB', (width, max_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Wrap text and draw line by line
    lines = text.split('\n')
    y_offset = 10
    line_height = font_size + 4

    for line in lines:
        # Simple word wrap
        words = line.split(' ')
        current_line = ''

        for word in words:
            test_line = current_line + ' ' + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width_test = bbox[2] - bbox[0]

            if width_test <= width - 20:
                current_line = test_line
            else:
                if current_line:
                    draw.text((10, y_offset), current_line, fill=text_color, font=font)
                    y_offset += line_height
                current_line = word

        if current_line:
            draw.text((10, y_offset), current_line, fill=text_color, font=font)
            y_offset += line_height

        if y_offset > max_height - line_height:
            break

    # Crop to actual content
    actual_height = min(y_offset + 10, max_height)
    img = img.crop((0, 0, width, actual_height))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_markdown_html(md_text: str, width: int = 600) -> np.ndarray:
    """Render markdown to HTML and capture as image using wkhtmltoimage approach."""
    # Convert markdown to HTML
    html_content = markdown(md_text)

    # Create styled HTML
    full_html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'DejaVu Sans', sans-serif;
                font-size: 14px;
                line-height: 1.6;
                padding: 20px;
                margin: 0;
                background: white;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Try to use wkhtmltoimage if available
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_path = f.name
        f.write(full_html)

    try:
        output_path = html_path.replace('.html', '.png')

        # Try wkhtmltoimage
        result = subprocess.run(
            ['wkhtmltoimage', '--width', str(width), '--height', '800',
             '--enable-local-file-access', html_path, output_path],
            capture_output=True, timeout=5
        )

        if result.returncode == 0 and Path(output_path).exists():
            img = cv2.imread(output_path)
            Path(output_path).unlink()
            Path(html_path).unlink()
            if img is not None:
                return img
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        print(f"Warning: wkhtmltoimage failed ({e}); falling back to text rendering.")

    # Fallback: render as text with basic markdown formatting
    Path(html_path).unlink()
    return render_markdown_as_text(md_text, width)


def render_markdown_as_text(md_text: str, width: int = 600) -> np.ndarray:
    """Fallback: render markdown as formatted text."""
    # Simple markdown to formatted text conversion
    lines = md_text.split('\n')
    formatted_lines = []

    for line in lines:
        # Headers
        if line.startswith('###'):
            formatted_lines.append(f"  {line[3:]}")
        elif line.startswith('##'):
            formatted_lines.append(f" {line[2:]}")
        elif line.startswith('#'):
            formatted_lines.append(line[1:])
        # Lists
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            formatted_lines.append(f"  • {line.strip()[2:]}")
        elif line.strip().startswith(tuple('0123456789')):
            formatted_lines.append(f"  {line.strip()}")
        # Tables
        elif '|' in line:
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)

    formatted_text = '\n'.join(formatted_lines)
    return draw_text_pil(formatted_text, width=width, font_size=13)


def visualize_markdown_sample(
    sample: Dict,
    image_dir: Path,
    output_path: Path
):
    """Create 1x3 visualization of markdown sample."""
    # Header bar height for titles
    HEADER_HEIGHT = 50
    HEADER_COLOR = (240, 240, 240)  # Light gray
    TEXT_COLOR = (0, 0, 0)  # Black

    # Get image path
    img_rel_path = sample.get('images', [None])[0]
    if img_rel_path is None:
        img_rel_path = sample.get('image', '')

    img_path = Path(img_rel_path)
    if not img_path.is_absolute():
        # Try to find the image
        img_path = image_dir / img_path.name

    if not img_path.exists():
        print(f"  Image not found: {img_path}")
        return

    # Load original image
    orig_img = cv2.imread(str(img_path))
    if orig_img is None:
        print(f"  Failed to load: {img_path}")
        return

    # Resize original to fit (max height 800)
    scale = min(800 / orig_img.shape[0], 1.0)
    new_width = int(orig_img.shape[1] * scale)
    new_height = int(orig_img.shape[0] * scale)
    orig_img_resized = cv2.resize(orig_img, (new_width, new_height))

    # Get markdown text
    messages = sample.get('messages', [])
    markdown_text = ""
    for msg in messages:
        if msg.get('role') == 'assistant':
            markdown_text = msg.get('content', '')
            break

    if not markdown_text:
        markdown_text = sample.get('answer', '')

    # Panel 1: Original image (keep as-is)
    panel1 = orig_img_resized

    # Panel 2: Raw markdown text
    panel2 = draw_text_pil(
        f"# RAW MARKDOWN TEXT\n\n{markdown_text}",
        width=600,
        max_height=panel1.shape[0],
        font_size=12
    )

    # Panel 3: Rendered markdown
    panel3 = render_markdown_html(markdown_text, width=600)
    panel3 = cv2.resize(panel3, (600, panel1.shape[0]))

    # Ensure all panels have same height
    max_height = max(panel1.shape[0], panel2.shape[0], panel3.shape[0])

    # Pad panels to same height
    def pad_panel(panel, target_height):
        pad = target_height - panel.shape[0]
        if pad > 0:
            return cv2.copyMakeBorder(panel, 0, pad, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return panel

    panel1 = pad_panel(panel1, max_height)
    panel2 = pad_panel(panel2, max_height)
    panel3 = pad_panel(panel3, max_height)

    # Add header bar to each panel
    def add_header(panel, title):
        # Create header bar
        header = np.full((HEADER_HEIGHT, panel.shape[1], 3), HEADER_COLOR, dtype=np.uint8)
        # Add title text centered
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = (panel.shape[1] - text_size[0]) // 2
        text_y = HEADER_HEIGHT - 15
        cv2.putText(header, title, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
        # Stack header on top of panel
        return np.vstack([header, panel])

    panel1 = add_header(panel1, "Original Image")
    panel2 = add_header(panel2, "Raw Markdown")
    panel3 = add_header(panel3, "Rendered Preview")

    # Concatenate horizontally
    canvas = np.hstack([panel1, panel2, panel3])

    # Save
    cv2.imwrite(str(output_path), canvas)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize markdown examples")
    parser.add_argument('--jsonl', required=True,
                        help='Path to markdown JSONL file')
    parser.add_argument('--images-dir', required=True,
                        help='Directory containing images')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples from {jsonl_path}...")
    samples = load_jsonl_samples(jsonl_path, args.num_samples)

    print(f"Visualizing {len(samples)} samples...")
    print()

    for i, sample in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)}:")
        output_path = output_dir / f"markdown_example_{i+1}.png"
        visualize_markdown_sample(sample, images_dir, output_path)

    print("Done!")


if __name__ == '__main__':
    main()
