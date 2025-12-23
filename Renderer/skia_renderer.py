"""
Skia-based Text Renderer with Binary Search Font Sizing

Advantages over PIL:
1. Proper kerning, ligatures, and anti-aliasing (C++ implementation)
2. Fast font metrics (no Python overhead)
3. Binary search for optimal font size
4. Can output both images and PDFs
5. Highly parallelizable with multiprocessing

Expected performance: 500-1000+ images/s with multiprocessing
"""

import numpy as np
import skia
from PIL import Image
from typing import List, Optional, Tuple
from pathlib import Path
import multiprocessing as mp
import time

# Font paths
FONT_PATHS = [
    "/usr/lib/python3/dist-packages/mkdocs/themes/readthedocs/fonts/Lato-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def find_font() -> str:
    """Find available font"""
    for path in FONT_PATHS:
        if Path(path).exists():
            return path
    return FONT_PATHS[0]


_FONT_PATH = None


def ensure_font():
    global _FONT_PATH
    if _FONT_PATH is None:
        _FONT_PATH = find_font()
    return _FONT_PATH


def binary_search_font_size(
    text: str,
    typeface: skia.Typeface,
    width: int,
    height: int,
    padding: int = 20,
    min_size: float = 9.0,
    max_size: float = 20.0,
    tolerance: float = 0.5,
) -> float:
    """
    Binary search for optimal font size that fits text in box.

    Uses Skia's fast C++ font metrics for measurement.

    Args:
        text: Text to render
        typeface: Skia typeface
        width, height: Box dimensions
        padding: Padding around text
        min_size, max_size: Font size search range
        tolerance: Stop when range < tolerance

    Returns:
        Optimal font size
    """
    usable_width = width - 2 * padding
    usable_height = height - 2 * padding

    low = min_size
    high = max_size
    optimal_size = min_size

    while high - low > tolerance:
        mid = (low + high) / 2.0
        font = skia.Font(typeface, mid)

        # Measure text with word wrapping
        words = text.split()
        if not words:
            return min_size

        # Simulate word wrapping
        space_width = font.measureText(' ')
        line_height = font.getSpacing()

        current_line_width = 0
        num_lines = 1

        for i, word in enumerate(words):
            word_width = font.measureText(word)

            if i == 0:
                current_line_width = word_width
            else:
                # Check if adding word + space exceeds width
                if current_line_width + space_width + word_width > usable_width:
                    # New line
                    num_lines += 1
                    current_line_width = word_width
                else:
                    current_line_width += space_width + word_width

        total_height = num_lines * line_height

        # Check if it fits
        if total_height <= usable_height:
            optimal_size = mid
            low = mid  # Try larger
        else:
            high = mid  # Too big, try smaller

    return optimal_size


def render_text_skia(
    text: str,
    width: int = 640,
    height: int = 640,
    padding: int = 20,
    font_size: Optional[float] = None,
    min_font_size: float = 9.0,
    max_font_size: float = 20.0,
) -> np.ndarray:
    """
    Render text using Skia with optimal font sizing.

    Returns RGB numpy array [H, W, 3] uint8.
    """
    # Load font
    font_path = ensure_font()
    typeface = skia.Typeface.MakeFromFile(font_path)
    if typeface is None:
        # Fallback to default
        typeface = skia.Typeface('Arial')

    # Find optimal font size
    if font_size is None:
        font_size = binary_search_font_size(
            text, typeface, width, height, padding,
            min_font_size, max_font_size
        )

    # Create surface
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()

    # White background
    canvas.clear(skia.ColorWHITE)

    # Setup font and paint
    font = skia.Font(typeface, font_size)
    paint = skia.Paint(Color=skia.ColorBLACK, AntiAlias=True)

    # Render text with word wrapping
    words = text.split()
    if not words:
        # Return white image
        image = surface.makeImageSnapshot()
        return np.array(image, copy=False)

    space_width = font.measureText(' ')
    line_height = font.getSpacing()

    x = padding
    y = padding + font.getMetrics().fDescent  # Start at baseline

    for i, word in enumerate(words):
        word_width = font.measureText(word)

        # Check if need to wrap
        if i > 0 and x + word_width > width - padding:
            x = padding
            y += line_height

        # Stop if out of vertical space
        if y > height - padding:
            break

        # Draw word
        canvas.drawString(word, x, y, font, paint)
        x += word_width + space_width

    # Convert to numpy array
    image = surface.makeImageSnapshot()
    # Skia returns RGBA, convert to RGB
    array = np.array(image, copy=False)
    if array.shape[2] == 4:
        array = array[:, :, :3]

    return array


def render_text_skia_pil(
    text: str,
    width: int = 640,
    height: int = 640,
    padding: int = 20,
    font_size: Optional[float] = None,
    min_font_size: float = 9.0,
    max_font_size: float = 20.0,
) -> Image.Image:
    """Render text to PIL Image"""
    arr = render_text_skia(text, width, height, padding, font_size, min_font_size, max_font_size)
    return Image.fromarray(arr)


# Worker function for multiprocessing
def _render_worker_skia(args):
    """Worker function that takes (text, width, height, padding, min_fs, max_fs)"""
    text, width, height, padding, min_fs, max_fs = args
    return render_text_skia(text, width, height, padding, None, min_fs, max_fs)


def _init_worker():
    """Initialize worker process"""
    ensure_font()


class SkiaRenderer:
    """
    High-performance Skia-based text renderer with multiprocessing.

    Features:
    - Binary search for optimal font size (uses fast C++ metrics)
    - Proper kerning and ligatures
    - Process pool for parallel rendering
    - Returns numpy arrays (faster than PIL)

    Expected performance: 500-1000+ images/s with 16+ workers
    """

    def __init__(
        self,
        num_workers: int = None,
        width: int = 640,
        height: int = 640,
        padding: int = 20,
        min_font_size: float = 9.0,
        max_font_size: float = 20.0,
    ):
        self.num_workers = num_workers or min(16, max(4, mp.cpu_count() // 2))
        self.width = width
        self.height = height
        self.padding = padding
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size

        # Initialize font
        ensure_font()

        # Process pool (lazy init)
        self.pool = None

    def _ensure_pool(self):
        if self.pool is None:
            ctx = mp.get_context('fork')
            self.pool = ctx.Pool(processes=self.num_workers, initializer=_init_worker)

    def _make_args(self, texts: List[str]):
        """Create arguments for worker function"""
        return [
            (t, self.width, self.height, self.padding, self.min_font_size, self.max_font_size)
            for t in texts
        ]

    def render_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Render batch of texts to numpy arrays.

        Returns list of [H, W, 3] uint8 arrays.
        """
        # Use serial rendering for small batches
        if len(texts) <= 2:
            return [render_text_skia(
                t, self.width, self.height, self.padding,
                None, self.min_font_size, self.max_font_size
            ) for t in texts]

        # Use pool for parallel rendering
        self._ensure_pool()
        args = self._make_args(texts)
        return self.pool.map(_render_worker_skia, args)

    def render_batch_pil(self, texts: List[str]) -> List[Image.Image]:
        """Render batch to PIL Images"""
        arrays = self.render_batch(texts)
        return [Image.fromarray(arr) for arr in arrays]

    def shutdown(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        self.shutdown()


def benchmark_skia():
    """Benchmark the Skia renderer"""
    print("=" * 70)
    print("SkiaRenderer Benchmark")
    print("=" * 70)

    # Generate test texts
    base = "Machine learning is a branch of AI. " * 20
    texts = [base] * 64

    # Warmup
    renderer = SkiaRenderer(num_workers=16)
    renderer.render_batch(texts[:4])

    # Benchmark
    num_iterations = 5
    times = []

    for _ in range(num_iterations):
        start = time.time()
        images = renderer.render_batch(texts)
        elapsed = time.time() - start
        times.append(elapsed)

    min_time = min(times)
    rate = len(texts) / min_time

    print(f"Batch size: {len(texts)}")
    print(f"Best time: {min_time*1000:.1f} ms")
    print(f"Rate: {rate:.1f} img/s")

    renderer.shutdown()

    # Compare with different worker counts
    print("\nWorker scaling:")
    for num_workers in [4, 8, 12, 16]:
        renderer = SkiaRenderer(num_workers=num_workers)
        renderer.render_batch(texts[:4])  # warmup

        start = time.time()
        for _ in range(3):
            renderer.render_batch(texts)
        elapsed = (time.time() - start) / 3

        rate = len(texts) / elapsed
        print(f"  {num_workers} workers: {rate:.1f} img/s")
        renderer.shutdown()


if __name__ == "__main__":
    benchmark_skia()
