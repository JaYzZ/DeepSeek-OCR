"""
Text Rendering Utilities for DeepSeek-OCR Server

Provides high-performance text-to-image rendering with automatic backend selection:
- Vello GPU renderer (~1565 img/s) - Primary if available
- Skia renderer (~778 img/s) - Secondary fallback
- PIL fallback (~130 img/s) - Last resort

The Renderer module handles backend selection and fallback logic.
"""

import logging
from typing import Tuple, Optional
from PIL import Image

# Try to use the Renderer module (with GPU acceleration and Skia fallback)
try:
    from Renderer import VelloRenderer, VELLO_AVAILABLE
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    VELLO_AVAILABLE = False

try:
    from Renderer import render_text_skia_pil, SKIA_AVAILABLE
except ImportError:
    SKIA_AVAILABLE = False
    render_text_skia_pil = None

logger = logging.getLogger(__name__)

# Global renderer instance (lazy-loaded)
_renderer_instance = None


def _get_renderer(width: int = 640, height: int = 640):
    """Get or create the global renderer instance"""
    global _renderer_instance

    if not RENDERER_AVAILABLE:
        logger.debug("Renderer module not available, using PIL fallback")
        return None

    if _renderer_instance is None:
        try:
            _renderer_instance = VelloRenderer(width=width, height=height)
            logger.info(f"✓ Initialized Vello GPU renderer: {_renderer_instance}")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU renderer: {e}")
            _renderer_instance = None

    return _renderer_instance


def render_text_to_image(
    text: str,
    width: int = 640,
    height: int = 640,
    font_size: int = 20,
    padding: int = 30,
    line_spacing: int = 6,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Render text as a document-style image

    Uses best available renderer: Vello GPU → Skia → PIL.

    Args:
        text: Text content to render (600-1300 words recommended)
        width: Image width in pixels
        height: Image height in pixels
        font_size: Font size (only used for PIL fallback)
        padding: Padding around text (used by Skia and PIL)
        line_spacing: Additional spacing between lines (only used for PIL fallback)
        bg_color: Background color RGB tuple (PIL fallback only)
        text_color: Text color RGB tuple (PIL fallback only)

    Returns:
        PIL Image with rendered text
    """
    # Try GPU-accelerated Vello renderer first
    renderer = _get_renderer(width, height)
    if renderer is not None:
        try:
            return renderer.render_batch_pil([text])[0]
        except Exception as e:
            logger.warning(f"Vello GPU renderer failed: {e}, trying Skia...")

    # Try Skia renderer second
    if SKIA_AVAILABLE and render_text_skia_pil is not None:
        try:
            logger.info("Using Skia renderer")
            return render_text_skia_pil(text, width=width, height=height, padding=padding)
        except Exception as e:
            logger.warning(f"Skia renderer failed: {e}, falling back to PIL")

    # Fallback to PIL (always available)
    logger.info("Using PIL renderer (last resort)")
    return _render_text_pil(
        text=text,
        width=width,
        height=height,
        font_size=font_size,
        padding=padding,
        line_spacing=line_spacing,
        bg_color=bg_color,
        text_color=text_color,
    )


def _render_text_pil(
    text: str,
    width: int = 640,
    height: int = 640,
    font_size: int = 20,
    padding: int = 30,
    line_spacing: int = 6,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Render text using PIL (fallback implementation)

    Args:
        text: Text content to render
        width: Image width in pixels
        height: Image height in pixels
        font_size: Font size
        padding: Padding around text
        line_spacing: Additional spacing between lines
        bg_color: Background color RGB tuple
        text_color: Text color RGB tuple

    Returns:
        PIL Image with rendered text
    """
    import textwrap
    from PIL import ImageDraw, ImageFont

    # Create image with white background
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)

    # Try to load a decent font, fall back to default if not available
    try:
        # Try common system fonts
        font_paths = [
            "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except (OSError, IOError):
                continue

        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Calculate usable text area
    text_width = width - 2 * padding

    # Word wrap text
    # Estimate characters per line based on font size
    chars_per_line = max(40, text_width // (font_size // 2))
    wrapped_lines = []
    for paragraph in text.split('\n'):
        if paragraph.strip():
            wrapped_lines.extend(textwrap.wrap(paragraph, width=chars_per_line))
        else:
            wrapped_lines.append('')  # Preserve empty lines

    # Calculate line height
    # Get actual font metrics if possible
    try:
        bbox = draw.textbbox((0, 0), "Ay", font=font)
        line_height = bbox[3] - bbox[1] + line_spacing
    except:
        line_height = font_size + line_spacing

    # Draw text
    y = padding
    for line in wrapped_lines:
        if y + line_height > height - padding:
            # Truncate if text exceeds image height
            break

        draw.text((padding, y), line, fill=text_color, font=font)
        y += line_height

    return image


def chunk_text_by_tokens(text: str, chunk_size: int = 1000, tokenizer=None) -> list:
    """
    Chunk text into pieces of approximately chunk_size tokens each

    Args:
        text: Input text
        chunk_size: Target token count per chunk (default: 1000)
        tokenizer: Optional tokenizer (if None, uses word-based approximation)

    Returns:
        List of text chunks
    """
    if tokenizer is not None:
        # Use actual tokenizer
        tokens = tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks
    else:
        # Rough approximation: ~1.3 tokens per word
        words = text.split()
        words_per_chunk = int(chunk_size / 1.3)

        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunk_words = words[i:i + words_per_chunk]
            chunks.append(' '.join(chunk_words))

        return chunks


def estimate_token_count(text: str, tokenizer=None) -> int:
    """
    Estimate token count in text

    Args:
        text: Input text
        tokenizer: Optional tokenizer (if None, uses rough approximation)

    Returns:
        Approximate token count
    """
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    else:
        # Rough approximation: ~1.3 tokens per word for English
        return int(len(text.split()) * 1.3)


def validate_text_chunk(text: str, min_tokens: int = 100, max_tokens: int = 2000, tokenizer=None) -> Tuple[bool, str]:
    """
    Validate text chunk is within acceptable token count
    (Note: This is now mainly for sanity checking, not enforcing limits)

    Args:
        text: Input text
        min_tokens: Minimum token count (default: 100)
        max_tokens: Maximum token count (default: 2000)
        tokenizer: Optional tokenizer for accurate counting

    Returns:
        Tuple of (is_valid, error_message)
    """
    token_count = estimate_token_count(text, tokenizer)

    if token_count < min_tokens:
        return False, f"Text too short: {token_count} tokens (minimum {min_tokens})"
    elif token_count > max_tokens:
        return False, f"Text too long: {token_count} tokens (maximum {max_tokens}). Will be automatically chunked."
    else:
        return True, ""
