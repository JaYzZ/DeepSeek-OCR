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
import textwrap
import math
import re

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


def _empty_layout_result() -> tuple[str, dict]:
    return "", {
        "layout_type": "plain",
        "preserve_newlines": True,
        "num_lines": 1,
        "max_line_chars": 0,
    }


def ensure_font():
    global _FONT_PATH
    if _FONT_PATH is None:
        _FONT_PATH = find_font()
    return _FONT_PATH


def prepare_text_for_rendering(
    text: str,
    short_line_threshold: int = 20,
    min_lines_for_reflow: int = 8,
    max_canvas_size: int = 4096,
    measurement_padding: int = 20,
    measurement_font_size: float = 10.0,
    min_canvas_size: int = 32,
    measurement_divisor: int = 32,
) -> tuple[str, dict]:
    """Binary search over width to find minimum H*W that fits in max_canvas_size."""
    if not text or not text.strip():
        return _empty_layout_result()

    # Normalize tabs to spaces and drop blank lines.
    text = text.expandtabs(4)
    lines = []
    for line in text.split("\n"):
        if line.strip():
            lines.append(line)

    if not lines:
        return _empty_layout_result()

    list_prefix = re.compile(r"^(\s*(?:[-*+]\s+|\d+\.\s+|[A-Za-z][\.\)]\s+))(.+?)\s*$")
    quote_prefix = re.compile(r"^(\s*>\s*)(.+?)\s*$")
    header_prefix = re.compile(r"^(\s*#+\s*)(.+?)\s*$")
    max_char_width = 1000
    line_pack_separator = "    "

    def classify_line(line: str, inside_fence: bool) -> str:
        stripped = line.strip()
        if not stripped:
            return "blank"
        if stripped.startswith(("```", "~~~")):
            return "immutable"
        if inside_fence:
            return "immutable"
        if line.startswith("    "):
            return "immutable"
        if stripped.startswith("|") or stripped.count("|") >= 2:
            return "immutable"
        if "\t" in line:
            return "immutable"
        if any(c in stripped for c in ("┌", "┐", "└", "┘", "│", "─", "├", "┤", "┬", "┴", "┼")):
            return "immutable"
        if set(stripped.replace("|", "").replace("+", "")) <= {"-", "=", " ", ":"} and len(stripped) > 5:
            return "immutable"
        return "mutable"

    def split_immutable_line(line: str, width: int) -> list[str]:
        if len(line) <= width:
            return [line]
        indent_len = len(line) - len(line.lstrip(" "))
        prefix = line[:indent_len]
        content = line[indent_len:]
        available = max(1, width - indent_len)
        wrapped = textwrap.wrap(
            content,
            width=available,
            break_long_words=True,
            break_on_hyphens=False,
            drop_whitespace=False,
            replace_whitespace=False,
        )
        if not wrapped:
            return [prefix]
        return [prefix + part for part in wrapped]

    def wrap_plain_text(content: str, width: int) -> list[str]:
        wrapped = textwrap.wrap(
            content,
            width=max(1, width),
            break_long_words=False,
            break_on_hyphens=False,
        )
        return wrapped or [content.strip()]

    def wrap_list_item(prefix: str, content: str, width: int) -> list[str]:
        body_width = max(8, width - len(prefix))
        wrapped = textwrap.wrap(
            content,
            width=body_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not wrapped:
            return [prefix.rstrip()]
        out = [prefix + wrapped[0]]
        continuation = prefix[: len(prefix) - len(prefix.lstrip(" "))]
        out.extend(continuation + line for line in wrapped[1:])
        return out

    def wrap_prefixed_item(prefix: str, content: str, width: int) -> list[str]:
        body_width = max(8, width - len(prefix))
        wrapped = textwrap.wrap(
            content,
            width=body_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not wrapped:
            return [prefix.rstrip()]
        out = [prefix + wrapped[0]]
        continuation = " " * len(prefix)
        out.extend(continuation + line for line in wrapped[1:])
        return out

    items = []
    mutable_item_count = 0
    mutable_char_total = 0
    shortest_line_len = None
    longest_line_len = 0
    inside_fence = False
    for line in lines:
        line_len = len(line)
        if shortest_line_len is None or line_len < shortest_line_len:
            shortest_line_len = line_len
        if line_len > longest_line_len:
            longest_line_len = line_len

        line_class = classify_line(line, inside_fence)
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            inside_fence = not inside_fence
        if line_class == "blank":
            items.append({"kind": "blank", "text": ""})
            continue
        if line_class == "immutable":
            items.append({"kind": "immutable", "text": line})
            continue
        list_match = list_prefix.match(line)
        if list_match:
            prefix, content = list_match.groups()
            items.append({"kind": "mutable", "prefix": prefix, "prefix_kind": "list", "content": content})
            mutable_item_count += 1
            mutable_char_total += len(prefix) + len(content)
            continue
        quote_match = quote_prefix.match(line)
        if quote_match:
            prefix, content = quote_match.groups()
            items.append({"kind": "mutable", "prefix": prefix, "content": content})
            mutable_item_count += 1
            mutable_char_total += len(prefix) + len(content)
            continue
        header_match = header_prefix.match(line)
        if header_match:
            prefix, content = header_match.groups()
            items.append({"kind": "mutable", "prefix": prefix, "content": content})
            mutable_item_count += 1
            mutable_char_total += len(prefix) + len(content)
            continue
        leading_ws_len = len(line) - len(line.lstrip(" "))
        if leading_ws_len > 0:
            prefix = line[:leading_ws_len]
            content = line[leading_ws_len:].strip()
            items.append({"kind": "mutable", "prefix": prefix, "content": content})
            mutable_item_count += 1
            mutable_char_total += len(prefix) + len(content)
            continue
        content = line.strip()
        items.append({"kind": "mutable", "content": content})
        mutable_item_count += 1
        mutable_char_total += len(content)
    # Get font for measurement
    font_path = ensure_font()
    typeface = skia.Typeface.MakeFromFile(font_path)
    if typeface is None:
        typeface = skia.Typeface("Arial")
    font = skia.Font(typeface, measurement_font_size)

    def render_item_with_width(item: dict, width: int) -> list[str]:
        if item.get("prepacked"):
            return [item["content"]]
        if item["kind"] == "immutable":
            return split_immutable_line(item["text"], width)
        if "prefix" in item:
            if item.get("prefix_kind") == "list":
                return wrap_list_item(item["prefix"], item["content"], width)
            return wrap_prefixed_item(item["prefix"], item["content"], width)
        return wrap_plain_text(item["content"], width)

    def build_packed_items(all_items: list[dict], width: int) -> list[dict]:
        packed_items = []
        pending_plain_lines = []
        pending_plain_len = 0
        separator_width = len(line_pack_separator)

        def flush_pending_plain_lines() -> None:
            nonlocal pending_plain_len
            if not pending_plain_lines:
                return
            packed_items.append(
                {
                    "kind": "mutable",
                    "content": line_pack_separator.join(pending_plain_lines),
                    "prepacked": True,
                }
            )
            pending_plain_lines.clear()
            pending_plain_len = 0

        for item in all_items:
            if item["kind"] != "mutable":
                flush_pending_plain_lines()
                packed_items.append(item)
                continue
            if "prefix" in item:
                flush_pending_plain_lines()
                packed_items.append(item)
                continue

            line_len = len(item["content"])
            if line_len > width:
                flush_pending_plain_lines()
                packed_items.append(item)
                continue

            if not pending_plain_lines:
                pending_plain_lines.append(item["content"])
                pending_plain_len = line_len
                continue

            candidate_len = pending_plain_len + separator_width + line_len
            if candidate_len <= width:
                pending_plain_lines.append(item["content"])
                pending_plain_len = candidate_len
            else:
                flush_pending_plain_lines()
                pending_plain_lines.append(item["content"])
                pending_plain_len = line_len

        flush_pending_plain_lines()
        return packed_items

    def render_all_with_width(all_items: list[dict], width: int) -> list[str]:
        rendered = []
        for item in all_items:
            if item["kind"] == "blank":
                if rendered and rendered[-1] != "":
                    rendered.append("")
                elif not rendered:
                    rendered.append("")
                continue
            rendered.extend(render_item_with_width(item, width))
        while rendered and rendered[-1] == "":
            rendered.pop()
        return rendered

    wrap_measure_cache: dict[int, tuple[list[str], int, int]] = {}
    score_cache: dict[int, tuple[float, float, int]] = {}

    def wrap_and_measure(char_width: int) -> tuple[list[str], int, int]:
        cached = wrap_measure_cache.get(char_width)
        if cached is not None:
            return cached
        packed_items = build_packed_items(items, char_width)
        wrapped = render_all_with_width(packed_items, char_width)

        if not wrapped:
            result = (wrapped, min_canvas_size, min_canvas_size)
            wrap_measure_cache[char_width] = result
            return result

        max_line_w = max((font.measureText(l) for l in wrapped), default=0.0)
        metrics = font.getMetrics()
        line_h = font.getSpacing()
        h = (-metrics.fAscent) + max(0, len(wrapped) - 1) * line_h + metrics.fDescent

        cw = max(min_canvas_size, int(math.ceil(max_line_w + 2 * measurement_padding)))
        ch = max(min_canvas_size, int(math.ceil(h + 2 * measurement_padding)))
        # Snap to grid
        cw = ((cw + measurement_divisor - 1) // measurement_divisor) * measurement_divisor
        ch = ((ch + measurement_divisor - 1) // measurement_divisor) * measurement_divisor
        result = (wrapped, cw, ch)
        wrap_measure_cache[char_width] = result
        return result

    if mutable_item_count < 2 and mutable_char_total < short_line_threshold * 2:
        wrapped, _, _ = wrap_and_measure(max(8, min(max_char_width, longest_line_len)))
        result = "\n".join(wrapped)
        return result, {
            "layout_type": "plain_packed",
            "preserve_newlines": True,
            "num_lines": len(wrapped),
            "max_line_chars": max((len(l) for l in wrapped), default=0),
        }

    if len(lines) < min_lines_for_reflow and mutable_char_total < short_line_threshold * min_lines_for_reflow:
        wrapped, _, _ = wrap_and_measure(max(8, min(max_char_width, longest_line_len)))
        result = "\n".join(wrapped)
        return result, {
            "layout_type": "plain_packed",
            "preserve_newlines": True,
            "num_lines": len(wrapped),
            "max_line_chars": max((len(l) for l in wrapped), default=0),
        }

    min_char = max(8, shortest_line_len or 0)
    max_char = min(max_char_width, longest_line_len)
    max_char = max(min_char, max_char)

    def get_area(char_width: int) -> int:
        """Get canvas area for given char width, or infinity if doesn't fit."""
        wrapped, cw, ch = wrap_and_measure(char_width)
        if cw <= max_canvas_size and ch <= max_canvas_size:
            return cw * ch
        return float('inf')

    def get_score(char_width: int) -> tuple[float, float, int]:
        """Score width by area first, then by square-ness."""
        cached = score_cache.get(char_width)
        if cached is not None:
            return cached
        _, cw, ch = wrap_and_measure(char_width)
        if cw > max_canvas_size or ch > max_canvas_size:
            result = (float("inf"), float("inf"), char_width)
            score_cache[char_width] = result
            return result
        longer = max(cw, ch)
        shorter = max(1, min(cw, ch))
        aspect_ratio = longer / shorter
        result = (cw * ch, aspect_ratio, char_width)
        score_cache[char_width] = result
        return result

    lo, hi = min_char, max_char
    min_valid = None
    while lo <= hi:
        mid = (lo + hi) // 2
        _, cw, ch = wrap_and_measure(mid)
        if cw <= max_canvas_size and ch <= max_canvas_size:
            min_valid = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if min_valid is None:
        best_wrapped, best_cw, best_ch = wrap_and_measure(max_char)
    else:
        best_wrapped = None
        best_cw = best_ch = 0
        best_area = float('inf')
        left, right = min_valid, max_char
        while right - left > 3:
            mid = (left + right) // 2
            low_score = get_score(left)
            mid_score = get_score(mid)
            high_score = get_score(right)
            ranked = sorted(
                [
                    (left, low_score),
                    (mid, mid_score),
                    (right, high_score),
                ],
                key=lambda item: item[1],
            )
            chosen_widths = {ranked[0][0], ranked[1][0]}
            if chosen_widths == {left, mid}:
                right = mid
            elif chosen_widths == {mid, right}:
                left = mid
            else:
                left_half_score = (low_score[1], low_score[0], left)
                right_half_score = (high_score[1], high_score[0], right)
                if left_half_score <= right_half_score:
                    right = mid
                else:
                    left = mid

        for char_width in range(left, right + 1):
            wrapped, cw, ch = wrap_and_measure(char_width)
            if cw <= max_canvas_size and ch <= max_canvas_size:
                area = cw * ch
                if area < best_area:
                    best_area = area
                    best_wrapped = wrapped
                    best_cw, best_ch = cw, ch

        if best_wrapped is None:
            best_wrapped, best_cw, best_ch = wrap_and_measure(min_valid)

    result = "\n".join(best_wrapped)
    return result, {
        "layout_type": "plain_packed",
        "preserve_newlines": True,
        "num_lines": len(best_wrapped),
        "max_line_chars": max((len(l) for l in best_wrapped), default=0),
    }


def snap_canvas_to_grid(
    width: int,
    height: int,
    divisor: int = 32,
    min_size: int = 32,
    max_size: int = 4096,
) -> tuple[int, int]:
    """Snap the measured canvas upward to the nearest allowed grid slot."""
    # If already exceeds max_size, snap upward and allow overflow
    if width > max_size:
        # Snap upward to nearest divisor, no clamping
        width = ((width + divisor - 1) // divisor) * divisor
    else:
        width = max(min_size, width)
        width = ((width + divisor - 1) // divisor) * divisor
        width = min(width, max_size)

    if height > max_size:
        height = ((height + divisor - 1) // divisor) * divisor
    else:
        height = max(min_size, height)
        height = ((height + divisor - 1) // divisor) * divisor
        height = min(height, max_size)

    return width, height


def measure_finalized_text_canvas(
    text: str,
    padding: int = 20,
    font_size: float = 10.0,
    min_size: int = 32,
) -> tuple[int, int]:
    """Measure the canvas size for finalized text with explicit line breaks."""
    font_path = ensure_font()
    typeface = skia.Typeface.MakeFromFile(font_path)
    if typeface is None:
        typeface = skia.Typeface("Arial")
    font = skia.Font(typeface, font_size)

    # Measure text
    lines = text.split('\n')
    if not lines:
        return min_size, min_size

    line_height = font.getSpacing()
    metrics = font.getMetrics()
    max_width = max((font.measureText(line) for line in lines), default=0.0)
    total_height = (-metrics.fAscent) + max(0, len(lines) - 1) * line_height + metrics.fDescent

    width = max(min_size, int(math.ceil(max_width + 2 * padding)))
    height = max(min_size, int(math.ceil(total_height + 2 * padding)))
    return width, height


def binary_search_font_size(
    text: str,
    typeface: skia.Typeface,
    width: int,
    height: int,
    padding: int = 20,
    min_size: float = 9.0,
    max_size: float = 20.0,
    tolerance: float = 0.5,
    preserve_newlines: bool = False,
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

        if preserve_newlines:
            lines = text.split("\n")
            if not any(line for line in lines):
                return min_size
            line_height = font.getSpacing()
            metrics = font.getMetrics()
            max_line_width = max((font.measureText(line) for line in lines), default=0.0)
            total_height = (-metrics.fAscent) + max(0, len(lines) - 1) * line_height + metrics.fDescent
            fits = max_line_width <= usable_width and total_height <= usable_height
        else:
            words = text.split()
            if not words:
                return min_size

            space_width = font.measureText(' ')
            line_height = font.getSpacing()
            current_line_width = 0
            num_lines = 1

            for i, word in enumerate(words):
                word_width = font.measureText(word)

                if i == 0:
                    current_line_width = word_width
                else:
                    if current_line_width + space_width + word_width > usable_width:
                        num_lines += 1
                        current_line_width = word_width
                    else:
                        current_line_width += space_width + word_width

            total_height = num_lines * line_height
            fits = total_height <= usable_height

        if fits:
            optimal_size = mid
            low = mid
        else:
            high = mid

    return optimal_size


def render_text_skia(
    text: str,
    width: int = 640,
    height: int = 640,
    padding: int = 20,
    font_size: Optional[float] = None,
    min_font_size: float = 9.0,
    max_font_size: float = 20.0,
    preserve_newlines: bool = False,
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
            min_font_size, max_font_size, preserve_newlines=preserve_newlines
        )

    # Create surface
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()

    # White background
    canvas.clear(skia.ColorWHITE)

    # Setup font and paint
    font = skia.Font(typeface, font_size)
    paint = skia.Paint(Color=skia.ColorBLACK, AntiAlias=True)

    if preserve_newlines:
        lines = text.split("\n")
        if not any(line for line in lines):
            image = surface.makeImageSnapshot()
            return np.array(image, copy=False)

        line_height = font.getSpacing()
        baseline = padding - font.getMetrics().fAscent
        for line in lines:
            if baseline > height - padding:
                break
            canvas.drawString(line, padding, baseline, font, paint)
            baseline += line_height
    else:
        words = text.split()
        if not words:
            image = surface.makeImageSnapshot()
            return np.array(image, copy=False)

        space_width = font.measureText(' ')
        line_height = font.getSpacing()

        x = padding
        y = padding + font.getMetrics().fDescent

        for i, word in enumerate(words):
            word_width = font.measureText(word)

            if i > 0 and x + word_width > width - padding:
                x = padding
                y += line_height

            if y > height - padding:
                break

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
    preserve_newlines: bool = False,
) -> Image.Image:
    """Render text to PIL Image"""
    arr = render_text_skia(
        text,
        width,
        height,
        padding,
        font_size,
        min_font_size,
        max_font_size,
        preserve_newlines=preserve_newlines,
    )
    return Image.fromarray(arr)


# Worker function for multiprocessing
def _render_worker_skia(args):
    """Worker function that takes renderer args."""
    text, width, height, padding, min_fs, max_fs, preserve_newlines = args
    return render_text_skia(
        text,
        width,
        height,
        padding,
        None,
        min_fs,
        max_fs,
        preserve_newlines=preserve_newlines,
    )


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
        preserve_newlines: bool = True,
    ):
        self.num_workers = num_workers or min(16, max(4, mp.cpu_count() // 2))
        self.width = width
        self.height = height
        self.padding = padding
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.preserve_newlines = preserve_newlines

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
            (
                t,
                self.width,
                self.height,
                self.padding,
                self.min_font_size,
                self.max_font_size,
                self.preserve_newlines,
            )
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
                None, self.min_font_size, self.max_font_size,
                preserve_newlines=self.preserve_newlines,
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
