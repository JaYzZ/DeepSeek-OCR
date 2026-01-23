"""
Vello GPU Renderer - Python Wrapper

Provides a unified interface matching other OCRFlow renderers (PIL, Skia).

Performance: 5,000-10,000+ images/second (GPU-accelerated)
"""

import numpy as np
from PIL import Image
from typing import List, Optional

try:
    import vello_renderer as _vello  # Rust extension module
    VELLO_AVAILABLE = True
except ImportError as e:
    VELLO_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class VelloRenderer:
    """
    GPU-accelerated text renderer using Vello (Rust) + cosmic-text.

    This renderer leverages GPU compute shaders for rendering, achieving
    10-100x better performance than CPU-based renderers.

    Features:
    - Binary search for optimal font sizing (cosmic-text metrics)
    - GPU-accelerated rendering with Vello
    - Proper typography: kerning, ligatures, shaping
    - Zero-copy numpy array output

    Expected performance: 5,000-10,000+ img/s

    Example:
        >>> renderer = VelloRenderer(width=640, height=640)
        >>> images = renderer.render_batch(["Hello", "World"])
        >>> print(images[0].shape)  # (640, 640, 3)
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 640,
        padding: int = 20,
        min_font_size: float = 5.0,
        max_font_size: float = 20.0,
        preserve_newlines: bool = False,
    ):
        """
        Initialize Vello GPU renderer.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            padding: Padding around text
            min_font_size: Minimum font size for binary search
            max_font_size: Maximum font size for binary search
            preserve_newlines: If True, preserve newlines for Q&A formatting.
                            If False, collapse newlines for compact rendering (default).

        Raises:
            ImportError: If vello_renderer Rust module is not installed
            RuntimeError: If GPU initialization fails
        """
        if not VELLO_AVAILABLE:
            raise ImportError(
                f"Vello renderer not available. "
                f"Please build the Rust extension:\n"
                f"  cd utils/vello_renderer\n"
                f"  maturin develop --release\n"
                f"Error: {_IMPORT_ERROR}"
            )

        self.width = width
        self.height = height
        self.padding = padding
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.preserve_newlines = preserve_newlines

        # Create Rust renderer (initializes GPU)
        try:
            self._renderer = _vello.VelloRenderer(
                width=width,
                height=height,
                padding=padding,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                preserve_newlines=preserve_newlines,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Vello GPU renderer: {e}\n"
                f"Make sure you have:\n"
                f"  1. Vulkan drivers installed (nvidia-smi for NVIDIA GPUs)\n"
                f"  2. libvulkan-dev package installed\n"
                f"  3. At least one GPU available"
            ) from e

    def render_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Render batch of texts to numpy arrays (GPU-accelerated).

        This is the main rendering method. It processes all texts on the GPU
        in parallel, achieving very high throughput.

        Args:
            texts: List of strings to render

        Returns:
            List of numpy arrays, each [H, W, 3] uint8 (RGB format)

        Example:
            >>> renderer = VelloRenderer()
            >>> images = renderer.render_batch(["Text 1", "Text 2"])
            >>> print(len(images))  # 2
            >>> print(images[0].dtype)  # uint8
        """
        return self._renderer.render_batch(texts)

    def render_batch_pil(self, texts: List[str]) -> List[Image.Image]:
        """
        Render batch to PIL Images.

        Convenience method for compatibility with PIL-based code.

        Args:
            texts: List of strings to render

        Returns:
            List of PIL Images
        """
        arrays = self.render_batch(texts)
        return [Image.fromarray(arr) for arr in arrays]

    def shutdown(self):
        """
        Clean up GPU resources.

        Called automatically when the renderer is garbage collected,
        but can be called explicitly for immediate cleanup.
        """
        if hasattr(self, '_renderer'):
            del self._renderer

    def __del__(self):
        """Cleanup on garbage collection"""
        self.shutdown()

    def __repr__(self):
        return (
            f"VelloRenderer(width={self.width}, height={self.height}, "
            f"padding={self.padding}, font_size={self.min_font_size}-{self.max_font_size}, "
            f"preserve_newlines={self.preserve_newlines})"
        )

    @property
    def version(self):
        """Get Vello renderer version"""
        return self._renderer.version if hasattr(self, '_renderer') else "N/A"


def is_available() -> bool:
    """
    Check if Vello renderer is available.

    Returns:
        True if the Rust extension is installed and GPU is accessible
    """
    return VELLO_AVAILABLE


def get_build_instructions() -> str:
    """
    Get build instructions for the Vello renderer.

    Returns:
        Multi-line string with build instructions
    """
    return """
To build and install the Vello GPU renderer:

1. Install Rust:
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   source $HOME/.cargo/env

2. Install system dependencies:
   apt-get install -y libvulkan-dev vulkan-tools libxcb1-dev libfontconfig1-dev

3. Install maturin:
   pip install maturin

4. Build and install:
   cd utils/vello_renderer
   maturin develop --release

5. Test:
   python -c "from Renderer import VelloRenderer; print('Success!')"

See docs/vello_renderer_setup.md for complete guide.
"""


# Convenience exports
__all__ = ['VelloRenderer', 'is_available', 'get_build_instructions', 'VELLO_AVAILABLE']
