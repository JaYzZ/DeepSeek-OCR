"""
Renderer Module - GPU-Accelerated Text Rendering

Provides multiple rendering backends with automatic fallback:
- Vello GPU renderer (1565+ img/s) - Best performance
- Skia renderer (778+ img/s) - CPU-based, good quality
"""

from .vello_renderer_wrapper import VelloRenderer, is_available, VELLO_AVAILABLE

try:
    from .skia_renderer import render_text_skia_pil, SkiaRenderer
    SKIA_AVAILABLE = True
except ImportError:
    SKIA_AVAILABLE = False
    render_text_skia_pil = None
    SkiaRenderer = None

__all__ = [
    'VelloRenderer',
    'is_available',
    'VELLO_AVAILABLE',
    'SkiaRenderer',
    'render_text_skia_pil',
    'SKIA_AVAILABLE'
]
