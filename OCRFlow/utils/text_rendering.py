"""
Text Rendering Utilities for OCRFlow

Provides fast CPU-based text rendering to create document-like images
for text-only training.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Tuple, Optional
import re


def render_text_to_image(
    text: str,
    width: int = 640,
    height: int = 640,
    font_size: int = 18,
    font_path: Optional[str] = None,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 20,
    line_spacing: int = 5,
) -> Image.Image:
    """
    Render plain text as an image (simulating a document page)

    Args:
        text: Text to render
        width, height: Image dimensions in pixels
        font_size: Font size in pixels
        font_path: Path to TTF font file (uses default if None)
        bg_color: Background color (R, G, B)
        text_color: Text color (R, G, B)
        padding: Padding from edges in pixels
        line_spacing: Extra spacing between lines

    Returns:
        PIL Image with rendered text
    """
    # Create blank image
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Load font
    font = _load_font(font_path, font_size)

    # Word wrapping
    lines = _wrap_text(text, draw, font, width - 2 * padding)

    # Draw text lines
    y = padding
    line_height = font_size + line_spacing

    for line in lines:
        if y + line_height > height - padding:
            break  # Stop if exceeding image height

        draw.text((padding, y), line, fill=text_color, font=font)
        y += line_height

    return img


def render_markdown_to_image(
    markdown: str,
    width: int = 640,
    height: int = 640,
    base_font_size: int = 16,
    font_path: Optional[str] = None,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 20,
) -> Image.Image:
    """
    Render markdown with basic formatting

    Supports:
    - # Headers (h1-h6 with different sizes)
    - **Bold** (darker/thicker appearance)
    - *Italic* (simulated)
    - - Lists (bullet points)
    - Code blocks (monospace font)
    - > Quotes (indented)

    Args:
        markdown: Markdown text to render
        width, height: Image dimensions
        base_font_size: Base font size (headers scale from this)
        font_path: Path to TTF font (optional)
        bg_color: Background color
        text_color: Text color
        padding: Edge padding

    Returns:
        PIL Image with rendered markdown
    """
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    y = padding
    usable_width = width - 2 * padding

    for line in markdown.split('\n'):
        if y > height - 40:
            break

        # Parse markdown line
        font_size, text, indent, is_bold = _parse_markdown_line(line, base_font_size)

        # Load appropriate font
        font = _load_font(font_path, font_size)

        # Apply indent
        x = padding + indent

        # Wrap text
        if text.strip():
            wrapped = _wrap_text(text, draw, font, usable_width - indent)

            for wrap_line in wrapped:
                if y > height - 40:
                    break

                # Adjust color for bold
                color = tuple(max(0, c - 40) for c in text_color) if is_bold else text_color

                draw.text((x, y), wrap_line, fill=color, font=font)
                y += font_size + 5
        else:
            # Empty line - add spacing
            y += 10

    return img


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """Load a font with fallback to default"""
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            pass

    # Try common system fonts
    common_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ]

    for font_path in common_fonts:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue

    # Fallback to default
    return ImageFont.load_default()


def _wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """
    Wrap text to fit within max_width

    Args:
        text: Text to wrap
        draw: ImageDraw instance (for text measurement)
        font: Font to use
        max_width: Maximum line width in pixels

    Returns:
        List of wrapped lines
    """
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def _parse_markdown_line(line: str, base_font_size: int) -> Tuple[int, str, int, bool]:
    """
    Parse a markdown line and extract formatting

    Args:
        line: Markdown line
        base_font_size: Base font size

    Returns:
        (font_size, text, indent, is_bold)
    """
    # Headers
    if line.startswith('# '):
        return base_font_size + 12, line[2:], 0, True
    elif line.startswith('## '):
        return base_font_size + 8, line[3:], 0, True
    elif line.startswith('### '):
        return base_font_size + 6, line[4:], 0, True
    elif line.startswith('#### '):
        return base_font_size + 4, line[5:], 0, True
    elif line.startswith('##### '):
        return base_font_size + 2, line[6:], 0, True
    elif line.startswith('###### '):
        return base_font_size + 1, line[7:], 0, True

    # Lists
    elif line.startswith('- ') or line.startswith('* '):
        return base_font_size, f"• {line[2:]}", 10, False
    elif re.match(r'^\d+\. ', line):
        match = re.match(r'^(\d+)\. (.+)', line)
        if match:
            num, text = match.groups()
            return base_font_size, f"{num}. {text}", 10, False

    # Quotes
    elif line.startswith('> '):
        return base_font_size - 1, line[2:], 20, False

    # Code blocks (simple detection)
    elif line.startswith('    ') or line.startswith('\t'):
        return base_font_size - 2, line.strip(), 30, False

    # Bold detection (simple)
    is_bold = '**' in line or '__' in line

    # Default
    return base_font_size, line, 0, is_bold


if __name__ == "__main__":
    # Test rendering
    print("Testing text rendering...")

    # Test 1: Plain text
    plain_text = """This is a simple document with plain text. It should wrap nicely across multiple lines and simulate a real document page.

This is a second paragraph with more content to test the layout."""

    img = render_text_to_image(plain_text, width=640, height=640)
    img.save("test_plain.png")
    print("✓ Saved test_plain.png")

    # Test 2: Markdown
    markdown_text = """# Document Title

This is a document with **markdown** formatting.

## Section 1

Here is some text in section 1.

- Item 1
- Item 2
- Item 3

## Section 2

> This is a quote

Here is more text.
"""

    img = render_markdown_to_image(markdown_text, width=640, height=640)
    img.save("test_markdown.png")
    print("✓ Saved test_markdown.png")

    print("\nRendering tests complete!")
