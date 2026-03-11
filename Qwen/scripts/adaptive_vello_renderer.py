#!/usr/bin/env python3
"""
Content-Based Adaptive Vello Renderer for Qwen3VL

Calculates optimal H×W dimensions based on actual text measurement:
- Uses minimum readable font size
- Calculates actual space needed with proper line wrapping
- Supports asymmetric H×W for efficiency
- Optimized for Qwen3VL's native variable resolution (64-1536)
"""

import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from Renderer.vello_renderer_wrapper import VelloRenderer, is_available


def analyze_text_layout(text: str) -> dict:
    """
    Analyze text to detect layout patterns including dialog, code, markdown, LaTeX, diagrams, and tables.

    Returns:
        dict with:
        - has_structure: bool - whether text has explicit structure
        - preserve_newlines: bool - whether to preserve newlines
        - use_monospace: bool - whether to use monospace font
        - max_line_chars: int - max chars per line (for code width calculation)
        - num_lines: int - number of lines
        - layout_type: str - 'diagram', 'table', 'tree', 'ascii_art', 'dialog', 'code', 'markdown', 'latex', 'list', 'structured', 'paragraph', 'plain'
    """
    lines = text.split('\n')
    num_lines = len(lines)
    non_empty_lines = [l for l in lines if l.strip()]

    if num_lines == 1:
        # Check for inline LaTeX or short code or chemical formulas
        has_latex = '$' in text or '\\' in text
        has_chemical = any(c in text for c in ['→', '←', '↔']) or ('+' in text and any(c in text for c in ['H', 'O', 'C', 'N']))
        has_code = '  ' in text or '\t' in text  # Has indentation

        return {
            'has_structure': False,
            'preserve_newlines': False,
            'use_monospace': has_code or has_chemical,
            'max_line_chars': len(text),
            'num_lines': 1,
            'layout_type': 'latex' if has_latex else ('diagram' if has_chemical else ('code' if has_code else 'plain')),
        }

    # Calculate max line length
    max_line_chars = max(len(l) for l in lines) if lines else 0

    # Detect ASCII art / symbolic diagrams (HIGHEST PRIORITY - check first)
    box_chars = ['┌', '┐', '└', '┘', '│', '─', '├', '┤', '┬', '┴', '┼', '╔', '╗', '╚', '╝', '║', '═']
    has_box_drawing = any(any(c in l for c in box_chars) for l in lines)

    # ASCII art patterns: repeated special chars in alignment
    special_char_lines = sum(1 for l in non_empty_lines if sum(1 for c in l if c in '+-|*=#_') > len(l) * 0.3)
    has_ascii_art = special_char_lines >= 3

    # Detect arrows (flowcharts, diagrams)
    arrow_chars = ['→', '←', '↔', '↑', '↓', '⇒', '⇐', '->', '<-', '=>']
    has_arrows = any(any(arrow in l for arrow in arrow_chars) for l in lines)

    is_diagram = (
        has_box_drawing or
        (has_ascii_art and num_lines >= 3) or
        (has_arrows and num_lines >= 2)
    )

    # Detect tree structures
    tree_chars = ['├──', '└──', '│', '├─', '└─', '├', '└', '├─', '└─']
    has_tree_structure = any(any(tc in l for tc in tree_chars) for l in lines)

    is_tree = has_tree_structure and num_lines >= 3

    # Detect tables
    has_table_borders = any(l.count('|') >= 2 for l in lines)
    has_separator_line = any(set(l.strip().replace('|', '').replace('+', '')) <= {'-', ' ', '='} and len(l.strip()) > 5 for l in lines)

    is_table = (
        has_table_borders and
        (has_separator_line or sum(1 for l in lines if '|' in l) >= 3)
    )

    # Detect dialog/QA patterns (check BEFORE other detections)
    dialog_patterns = ['Q:', 'A:', 'User:', 'Assistant:', 'Human:', 'AI:', 'Question:', 'Answer:']
    lines_with_dialog = sum(1 for l in non_empty_lines if any(l.strip().startswith(p) for p in dialog_patterns))
    is_dialog = (
        lines_with_dialog >= 2 and  # At least 2 dialog turns
        num_lines >= 2
    )

    # Detect code patterns
    has_indentation = any(l.startswith('    ') or l.startswith('\t') for l in non_empty_lines)
    has_braces = sum(1 for l in non_empty_lines if '{' in l or '}' in l) >= 2
    has_semicolons = sum(1 for l in non_empty_lines if l.rstrip().endswith(';')) >= 2
    code_keywords = ['def ', 'class ', 'import ', 'function ', 'const ', 'let ', 'var ', 'public ', 'private ']
    has_code_keywords = any(any(kw in l for kw in code_keywords) for l in non_empty_lines)

    is_code = (
        not is_dialog and  # Don't treat dialog as code even if it has keywords
        (has_indentation or
         (has_braces and has_semicolons) or
         has_code_keywords or
         (num_lines >= 3 and max_line_chars > 60 and has_indentation))
    )

    # Detect markdown patterns
    has_headers = any(l.startswith('#') for l in lines)
    has_md_list = any(l.strip().startswith(('- ', '* ', '+ ')) or
                      (l.strip()[:3].rstrip('.').isdigit() and '. ' in l[:5])
                      for l in non_empty_lines)
    has_code_block = '```' in text or '~~~' in text
    has_bold_italic = '**' in text or '__' in text or '*' in text or '_' in text

    is_markdown = (
        not is_dialog and
        (has_headers or
         has_code_block or
         (has_md_list and num_lines >= 3) or
         (has_bold_italic and has_headers))
    )

    # Detect LaTeX patterns
    has_math_env = any(env in text for env in ['\\begin{', '\\end{', '\\frac', '\\sum', '\\int'])
    has_display_math = '$$' in text or '\\[' in text or '\\]' in text
    has_inline_math = '$' in text and text.count('$') >= 2

    is_latex = has_math_env or has_display_math or (has_inline_math and num_lines >= 2)

    # Calculate average line length (excluding empty lines)
    if non_empty_lines:
        avg_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
    else:
        avg_line_length = 0

    # Detect list pattern (short lines, many newlines)
    is_list = (
        not is_dialog and
        not is_code and
        not is_markdown and
        not is_latex and
        num_lines >= 3 and
        avg_line_length < 50 and
        len(non_empty_lines) >= 3
    )

    # Detect structured content (multiple paragraphs or sections)
    empty_line_count = len([l for l in lines if not l.strip()])
    is_structured = (
        not is_dialog and
        not is_code and
        not is_markdown and
        not is_latex and
        not is_list and
        (empty_line_count >= 2 or num_lines >= 5)
    )

    # Determine layout type and settings (PRIORITY ORDER matters!)
    if is_diagram:
        layout_type = 'diagram'
        preserve_newlines = True
        use_monospace = True  # Spatial alignment critical
    elif is_tree:
        layout_type = 'tree'
        preserve_newlines = True
        use_monospace = True  # Indentation critical
    elif is_table:
        layout_type = 'table'
        preserve_newlines = True
        use_monospace = True  # Column alignment critical
    elif is_dialog:
        layout_type = 'dialog'
        preserve_newlines = True
        use_monospace = False
    elif is_code:
        layout_type = 'code'
        preserve_newlines = True
        use_monospace = True
    elif is_markdown:
        layout_type = 'markdown'
        preserve_newlines = True
        use_monospace = has_code_block  # Use monospace if has code blocks
    elif is_latex:
        layout_type = 'latex'
        preserve_newlines = True
        use_monospace = False
    elif is_list:
        layout_type = 'list'
        preserve_newlines = True
        use_monospace = False
    elif is_structured:
        layout_type = 'structured'
        preserve_newlines = True
        use_monospace = False
    elif num_lines >= 3:
        layout_type = 'paragraph'
        preserve_newlines = False
        use_monospace = False
    else:
        layout_type = 'plain'
        preserve_newlines = False
        use_monospace = False

    return {
        'has_structure': any([is_diagram, is_tree, is_table, is_dialog, is_code, is_markdown, is_latex, is_list, is_structured]),
        'preserve_newlines': preserve_newlines,
        'use_monospace': use_monospace,
        'max_line_chars': max_line_chars,
        'num_lines': num_lines,
        'layout_type': layout_type,
    }


def calculate_text_layout(
    text: str,
    font_size: float = 12.0,           # Font size in points
    char_width_px: float = 7.0,        # Approximate character width at this font size
    line_height_px: float = 18.0,      # Line height in pixels
    max_line_width_chars: int = 80,    # Maximum characters per line before wrapping
    padding: int = 20,                 # Padding around text
    preserve_newlines: bool = False,   # Whether to preserve explicit newlines
    use_monospace: bool = False,       # Whether text is code (affects width calculation)
) -> Tuple[int, int]:
    """
    Calculate actual text layout dimensions based on content measurement.

    This estimates the real space needed by analyzing:
    - Character count and line wrapping
    - Explicit newlines in text
    - Actual pixel dimensions with given font metrics
    - Special handling for code/monospace (no wrapping)

    Args:
        text: Text to render
        font_size: Font size in points
        char_width_px: Average character width in pixels at this font size
        line_height_px: Line height in pixels
        max_line_width_chars: Max characters per line before wrapping (ignored if preserve_newlines)
        padding: Padding in pixels
        preserve_newlines: If True, respect explicit newlines (for lists/structured text/code)
        use_monospace: If True, adjust for monospace fonts (wider chars)

    Returns:
        (width, height) tuple in pixels
    """
    # Split by explicit newlines
    lines = text.split('\n')

    # Adjust char width for monospace (typically slightly wider)
    if use_monospace:
        char_width_px = char_width_px * 1.1  # Monospace fonts are ~10% wider

    # Estimate wrapped lines
    total_lines = 0
    max_line_chars = 0

    for line in lines:
        if len(line) == 0:
            total_lines += 1  # Empty line (for structure)
            continue

        if preserve_newlines:
            # Keep lines as-is, don't wrap (critical for code/lists/markdown)
            total_lines += 1
            max_line_chars = max(max_line_chars, len(line))
        else:
            # Estimate word wrapping for this line (for paragraphs)
            line_chars = len(line)
            wrapped_lines = max(1, math.ceil(line_chars / max_line_width_chars))
            total_lines += wrapped_lines

            # Track longest line (after wrapping)
            chars_per_wrapped_line = min(line_chars, max_line_width_chars)
            max_line_chars = max(max_line_chars, chars_per_wrapped_line)

    # Calculate dimensions with padding
    width = int(max_line_chars * char_width_px + 2 * padding)
    height = int(total_lines * line_height_px + 2 * padding)

    return width, height


def calculate_adaptive_dimensions(
    text: str,
    min_font_size: float = 8.0,          # Vello's minimum font size
    min_size: int = 64,                  # Minimum dimension (Qwen3VL lower bound)
    max_size: int = 1536,                # Maximum dimension
    vit_divisor: int = 32,               # Round to multiple (for Qwen3VL efficiency)
    padding: int = 15,                   # Padding in pixels (small for efficiency)
    safety_multiplier: float = 1.5,      # Safety margin (50% extra to handle Vello auto-sizing)
    aspect_ratio_constraint: Optional[float] = 4.0,  # max W/H or H/W ratio
    allow_asymmetric: bool = True,       # if False, force square
) -> Tuple[int, int, dict]:
    """
    Calculate adaptive H×W dimensions based on actual text content measurement.

    Approach:
    1. Measure space needed for text at Vello's minimum font size
    2. Apply safety margin (Vello will auto-size font between min and max)
    3. Clamp to valid range [min_size, max_size]
    4. Round to vit_divisor for Qwen3VL efficiency

    Args:
        text: Text to render
        min_font_size: Vello's minimum font size (used for size calculation)
        min_size: Minimum dimension for either H or W (Qwen3VL requires 64+)
        max_size: Maximum dimension for either H or W
        vit_divisor: Round dimensions to multiple of this (32 for Qwen3VL)
        padding: Padding around text in pixels
        safety_multiplier: Extra space multiplier (e.g., 1.5 = 50% extra)
        aspect_ratio_constraint: Maximum aspect ratio (prevents extreme rectangles)
        allow_asymmetric: If False, returns square dimensions

    Returns:
        (width, height, layout_info) tuple
    """
    num_chars = len(text)

    if num_chars == 0:
        return (min_size, min_size)

    # Analyze text layout to detect structure
    layout_info = analyze_text_layout(text)

    # Calculate text layout at Vello's minimum font size
    # Font metrics (approximate, for any font size):
    # - char width: ~0.58 * font_size
    # - line height: ~1.5 * font_size
    char_width = min_font_size * 0.58
    line_height = min_font_size * 1.5

    width, height = calculate_text_layout(
        text=text,
        font_size=min_font_size,
        char_width_px=char_width,
        line_height_px=line_height,
        max_line_width_chars=80,  # Reasonable line length for readability
        padding=padding,
        preserve_newlines=layout_info['preserve_newlines'],
        use_monospace=layout_info['use_monospace'],
    )

    # Apply safety multiplier to prevent truncation
    # Vello's font sizing is automatic, so we need extra space
    width = int(width * safety_multiplier)
    height = int(height * safety_multiplier)

    # Apply aspect ratio constraint if specified.
    # IMPORTANT: keep content area by expanding the smaller side, never
    # shrinking the larger side (shrinking can re-introduce truncation).
    if aspect_ratio_constraint is not None and allow_asymmetric:
        actual_ratio = max(width, height) / max(min(width, height), 1)
        if actual_ratio > aspect_ratio_constraint:
            # Constrain to max ratio by growing the smaller dimension.
            if width > height:
                height = int(width / aspect_ratio_constraint)
            else:
                width = int(height / aspect_ratio_constraint)

    # Force square if requested
    if not allow_asymmetric:
        size = max(width, height)
        width = height = size

    # Clamp to valid range
    width = max(min_size, min(max_size, width))
    height = max(min_size, min(max_size, height))

    # Round to divisor for Qwen3VL efficiency
    width = ((width + vit_divisor - 1) // vit_divisor) * vit_divisor
    height = ((height + vit_divisor - 1) // vit_divisor) * vit_divisor

    return width, height, layout_info


def reflow_short_newline_text(
    text: str,
    layout_info: dict,
    short_line_threshold: int = 20,
    min_lines_for_reflow: int = 8,
    target_line_width_chars: int = 80,
) -> tuple[str, dict]:
    """
    Reflow text when explicit newlines create many very short lines.

    This keeps newlines in the final text, but uses fewer lines by wrapping merged
    content to a target width.
    """
    if not text or not text.strip():
        return text, layout_info

    if not layout_info.get("preserve_newlines", False):
        return text, layout_info

    # Never rewrite layout-sensitive formats where explicit '\n' carries structure.
    layout_type = layout_info.get("layout_type", "")
    reflow_allowed_layouts = {"list", "structured", "markdown"}
    if layout_type not in reflow_allowed_layouts:
        return text, layout_info

    lines = text.split("\n")
    non_empty = [l.strip() for l in lines if l.strip()]
    if len(non_empty) < min_lines_for_reflow:
        return text, layout_info

    def is_hard_layout_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.startswith(("```", "#", ">", "|")):
            return True
        if len(line) - len(line.lstrip()) >= 4:  # indentation/code
            return True
        if any(c in s for c in ("┌", "┐", "└", "┘", "│", "─", "{", "}", ";")):
            return True
        # Keep markdown bullets as explicit structure.
        if s.startswith(("- ", "* ", "+ ")):
            return True
        return False

    number_prefix = re.compile(r"^\s*(\d+)\.\s+")

    def parse_line_number(line: str) -> Optional[int]:
        m = number_prefix.match(line)
        return int(m.group(1)) if m else None

    def wrap_with_best_width(text: str, min_width: int = 56) -> List[str]:
        """Search width candidates and pick the wrap with smallest estimated render cost."""
        best_wrapped: Optional[List[str]] = None
        best_score: Optional[float] = None
        best_max_len: Optional[int] = None

        for width in range(min_width, target_line_width_chars + 1, 4):
            wrapped = textwrap.wrap(
                text,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            if not wrapped:
                continue

            max_len = max(len(x) for x in wrapped)
            area = max_len * len(wrapped)

            short_thr = max(10, short_line_threshold // 2)
            very_short_thr = max(6, short_thr // 2)
            short_lines = sum(1 for x in wrapped if len(x) < short_thr)
            very_short_lines = sum(1 for x in wrapped if len(x) < very_short_thr)
            orphan = 1 if len(wrapped[-1]) < short_thr else 0

            score = (
                area
                + short_lines * 80
                + very_short_lines * 160
                + orphan * 140
            )

            if (
                best_score is None
                or score < best_score
                or (score == best_score and max_len < (best_max_len or 10**9))
            ):
                best_score = score
                best_wrapped = wrapped
                best_max_len = max_len

        return best_wrapped if best_wrapped else [text]

    def optimize_numbered_adjacent_lines(lines: List[str]) -> List[str]:
        """
        For numbered short lines, search adjacent line-number groups for a compact wrap.
        Uses local look-ahead and picks the smallest estimated render cost.
        """
        out: List[str] = []
        i = 0
        n = len(lines)
        while i < n:
            best_score: Optional[float] = None
            best_k = 1
            best_wrapped: List[str] = [lines[i]]

            max_k = min(6, n - i)
            for k in range(1, max_k + 1):
                chunk = lines[i:i + k]
                nums = [parse_line_number(x) for x in chunk]
                if any(x is None for x in nums):
                    if k > 1:
                        break
                if k > 1 and all(x is not None for x in nums):
                    if any(nums[j] != nums[j - 1] + 1 for j in range(1, len(nums))):
                        break

                merged = " ".join(x.strip() for x in chunk if x.strip())
                wrapped = wrap_with_best_width(merged, min_width=52)
                max_len = max(len(x) for x in wrapped) if wrapped else 0
                score = max_len * len(wrapped)
                # Mild bias to merge adjacent numbered lines to reduce line count.
                score -= (k - 1) * 20

                if (
                    best_score is None
                    or score < best_score
                    or (score == best_score and max_len < max(len(x) for x in best_wrapped))
                ):
                    best_score = score
                    best_k = k
                    best_wrapped = wrapped

            out.extend(best_wrapped)
            i += best_k

        return out

    blocks: List[List[str]] = []
    cur: List[str] = []
    for line in lines:
        if line.strip():
            cur.append(line)
        else:
            if cur:
                blocks.append(cur)
                cur = []
            blocks.append([])  # explicit paragraph break
    if cur:
        blocks.append(cur)

    rebuilt: List[str] = []
    for block in blocks:
        if not block:
            rebuilt.append("")
            continue
        # Mixed-content block: keep hard lines intact, reflow soft line segments.
        soft_segment: List[str] = []

        def flush_soft_segment() -> None:
            nonlocal soft_segment
            if not soft_segment:
                return
            stripped = [s.strip() for s in soft_segment if s.strip()]
            if not stripped:
                rebuilt.extend(soft_segment)
                soft_segment = []
                return

            # Keep semantic paragraph boundaries:
            # - long lines are wrapped individually (newline preserved),
            # - consecutive short lines are merged and re-wrapped.
            runs: List[tuple[bool, List[str]]] = []
            current_kind = None
            current_lines: List[str] = []

            for line in stripped:
                is_short = len(line) < short_line_threshold
                if current_kind is None or is_short == current_kind:
                    current_kind = is_short
                    current_lines.append(line)
                else:
                    runs.append((current_kind, current_lines))
                    current_kind = is_short
                    current_lines = [line]
            if current_lines:
                runs.append((bool(current_kind), current_lines))

            for is_short_run, run_lines in runs:
                avg_len = sum(len(s) for s in run_lines) / len(run_lines)
                dynamic_width = int(min(100, max(56, avg_len * 2.2)))
                width = max(56, min(target_line_width_chars, dynamic_width))

                if is_short_run and len(run_lines) >= 2:
                    numbered_ratio = sum(1 for x in run_lines if parse_line_number(x) is not None) / len(run_lines)
                    if numbered_ratio >= 0.6:
                        rebuilt.extend(optimize_numbered_adjacent_lines(run_lines))
                    else:
                        merged = " ".join(run_lines)
                        rebuilt.extend(wrap_with_best_width(merged, min_width=56))
                else:
                    for line in run_lines:
                        if len(line) > width:
                            rebuilt.extend(wrap_with_best_width(line, min_width=56))
                        else:
                            rebuilt.append(line)
            soft_segment = []

        for line in block:
            if is_hard_layout_line(line):
                flush_soft_segment()
                rebuilt.append(line)
            else:
                soft_segment.append(line)
        flush_soft_segment()

    # Clean trailing empty lines introduced by block reconstruction.
    while rebuilt and rebuilt[-1] == "":
        rebuilt.pop()

    if not rebuilt:
        return text, layout_info

    updated = dict(layout_info)
    updated["preserve_newlines"] = True
    updated["layout_type"] = f"{layout_info.get('layout_type', 'unknown')}_shortline_reflow"
    updated["num_lines"] = len(rebuilt)
    updated["max_line_chars"] = max(len(w) for w in rebuilt) if rebuilt else 0

    return "\n".join(rebuilt), updated


class AdaptiveVelloRenderer:
    """
    Content-based adaptive renderer for Qwen3VL.

    Features:
    - Measures actual text space requirements
    - Asymmetric H×W based on content (e.g., 192×256 for tall lists)
    - Minimum 64×64 (Qwen3VL lower bound)
    - Maximum 1536×1536
    - All dimensions divisible by 32
    - GPU-accelerated Vello rendering
    """

    def __init__(
        self,
        min_vello_font: float = 8.0,     # Vello minimum font size
        max_vello_font: float = 10.0,    # Vello maximum font size
        min_size: int = 64,
        max_size: int = 1536,
        vit_divisor: int = 32,
        padding: int = 15,               # Small padding for efficiency (reduced from 40)
        safety_multiplier: float = 1.5,  # 50% extra (increased to handle Vello's auto-sizing)
        aspect_ratio_constraint: Optional[float] = 4.0,
        allow_asymmetric: bool = True,
        max_retries: int = 3,            # Max attempts to fit text (auto-retry if truncated)
        edge_margin: int = 5,            # Minimum margin from edges (for truncation detection)
        thinking_padding: int = 9,       # Smaller padding for thinking chunks
        thinking_safety_multiplier: float = 1.25,  # Reduced safety for thinking chunks
        short_line_wrap_threshold: int = 20,       # Reflow if many lines shorter than this
        short_line_min_lines: int = 8,             # Minimum lines to trigger reflow
        short_line_target_width_chars: int = 80,   # Wrapped line width for reflowed content
    ):
        """
        Initialize content-based adaptive renderer.

        Uses Vello's min font size for calculation, then lets Vello auto-size
        between min and max font to fill the canvas optimally.

        Args:
            min_vello_font: Minimum font size for Vello renderer (used for size calculation)
            max_vello_font: Maximum font size for Vello renderer
            min_size: Minimum canvas dimension (64 for Qwen3VL)
            max_size: Maximum canvas dimension
            vit_divisor: Round dimensions to multiple of this
            padding: Padding around text in pixels
            safety_multiplier: Extra space multiplier (e.g., 1.5 = 50% extra)
            aspect_ratio_constraint: Maximum aspect ratio (None for no limit)
            allow_asymmetric: Enable asymmetric H×W rendering
            max_retries: Maximum attempts to fit text (auto-retry with larger size if truncated)
            edge_margin: Minimum margin from edges in pixels (for truncation detection)
        """
        if not is_available():
            raise ImportError("Vello renderer not available!")

        self.min_vello_font = min_vello_font
        self.max_vello_font = max_vello_font
        self.min_size = min_size
        self.max_size = max_size
        self.vit_divisor = vit_divisor
        self.padding = padding
        self.safety_multiplier = safety_multiplier
        self.aspect_ratio_constraint = aspect_ratio_constraint
        self.allow_asymmetric = allow_asymmetric
        self.max_retries = max_retries
        self.edge_margin = edge_margin
        self.thinking_padding = thinking_padding
        self.thinking_safety_multiplier = thinking_safety_multiplier
        self.short_line_wrap_threshold = short_line_wrap_threshold
        self.short_line_min_lines = short_line_min_lines
        self.short_line_target_width_chars = short_line_target_width_chars

    def _is_truncated(self, image: np.ndarray) -> bool:
        """
        Detect if text is truncated by checking for non-white pixels near edges.

        Args:
            image: Rendered image (H, W, 3) in RGB format

        Returns:
            True if text appears to touch edges (likely truncated)
        """
        # White threshold (Vello uses white background)
        white_threshold = 250  # Pixels with value > 250 are considered white

        # Check all edges within edge_margin pixels
        margin = self.edge_margin

        # Top edge
        if np.any(image[:margin, :, :] < white_threshold):
            return True

        # Bottom edge
        if np.any(image[-margin:, :, :] < white_threshold):
            return True

        # Left edge
        if np.any(image[:, :margin, :] < white_threshold):
            return True

        # Right edge
        if np.any(image[:, -margin:, :] < white_threshold):
            return True

        return False

    def render(
        self,
        text: str,
        output_path: str,
        thinking_mode: bool = False,
    ) -> bool:
        """
        Render single text to image file.

        Convenience method for rendering a single text to a file.

        Args:
            text: Text to render
            output_path: Path to save rendered image

        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False

        try:
            # Render with adaptive sizing
            results = self.render_batch(
                [text],
                return_dimensions=True,
                thinking_mode=thinking_mode,
            )

            if not results or len(results) == 0:
                return False

            image, (width, height) = results[0]

            # Save atomically to avoid partially-written PNGs if interrupted.
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
            Image.fromarray(image).save(tmp_path, format="PNG")
            tmp_path.replace(output_path)

            return True

        except Exception as e:
            import traceback
            print(f"Error rendering text (len={len(text) if text else 0}): {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            return False

    def render_batch(
        self,
        texts: List[str],
        return_dimensions: bool = False,
        thinking_mode: bool = False,
    ) -> List[np.ndarray] | List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Render batch of texts with content-based adaptive dimensions.

        Each text gets its own optimal H×W dimensions based on actual content.
        Automatically validates and retries if text is truncated.

        Args:
            texts: List of texts to render
            return_dimensions: If True, return (image, (W, H)) tuples

        Returns:
            List of rendered images or (image, dimensions) tuples
        """
        results = []

        for text in texts:
            # Thinking preset: smaller padding and tighter safety to reduce token inflation.
            base_padding = self.thinking_padding if thinking_mode else self.padding
            base_safety = self.thinking_safety_multiplier if thinking_mode else self.safety_multiplier

            # Reflow short newline-heavy content while preserving '\n' in output text.
            layout_info = analyze_text_layout(text)
            prepared_text, layout_info = reflow_short_newline_text(
                text=text,
                layout_info=layout_info,
                short_line_threshold=self.short_line_wrap_threshold,
                min_lines_for_reflow=self.short_line_min_lines,
                target_line_width_chars=self.short_line_target_width_chars,
            )

            # Try rendering with increasing sizes until it fits
            for attempt in range(self.max_retries):
                # Calculate optimal dimensions for this text
                # Use Vello's min font size for calculation
                # Apply additional multiplier on retries
                retry_multiplier = 1.0 + (0.2 * attempt)  # 1.0, 1.2, 1.4, ...
                effective_safety = base_safety * retry_multiplier

                width, height, layout_info = calculate_adaptive_dimensions(
                    prepared_text,
                    min_font_size=self.min_vello_font,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    vit_divisor=self.vit_divisor,
                    padding=base_padding,
                    safety_multiplier=effective_safety,
                    aspect_ratio_constraint=self.aspect_ratio_constraint,
                    allow_asymmetric=self.allow_asymmetric,
                )

                # Create Vello renderer with these dimensions
                # Use preserve_newlines based on layout detection
                renderer = VelloRenderer(
                    width=width,
                    height=height,
                    padding=base_padding,
                    min_font_size=self.min_vello_font,
                    max_font_size=self.max_vello_font,
                    preserve_newlines=layout_info['preserve_newlines'],
                )

                # Render
                images = renderer.render_batch([prepared_text])
                image = images[0]

                # Cleanup
                renderer.shutdown()

                # Check if text is truncated
                is_truncated = self._is_truncated(image)

                if not is_truncated:
                    # Success! Text fits properly
                    if return_dimensions:
                        results.append((image, (width, height)))
                    else:
                        results.append(image)
                    break
                elif attempt == self.max_retries - 1:
                    # Last attempt, use what we have
                    print(f"Warning: Text may be truncated after {self.max_retries} attempts (final size: {width}×{height})")
                    if return_dimensions:
                        results.append((image, (width, height)))
                    else:
                        results.append(image)
                else:
                    # Retry with larger size
                    pass  # Continue to next attempt

        return results


def main():
    """Test content-based adaptive rendering with layout detection."""
    import time

    output_dir = Path("./Qwen")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test cases covering different layout patterns
    test_cases = [
        ("short", "Hello world!"),
        ("medium", "The quick brown fox jumps over the lazy dog. " * 5),
        ("tall_list", "\n".join([f"Item {i}" for i in range(1, 11)])),
        ("paragraph", "DeepSeek-OCR uses content-based adaptive rendering. " * 10),
        ("structured", "Section 1: Introduction\nThis is the intro.\n\nSection 2: Methods\nDescribes the methods.\n\nSection 3: Results\nShows the results."),
        ("bullet_list", "• First item\n• Second item\n• Third item\n• Fourth item\n• Fifth item"),
        ("code_python", """def hello_world():
    print("Hello, world!")
    for i in range(10):
        print(f"Number: {i}")
    return True"""),
        ("code_json", """{
  "name": "adaptive_renderer",
  "version": "1.0",
  "features": ["layout_detection", "code_support"],
  "status": "active"
}"""),
        ("markdown", """# Main Title

## Introduction
This is a markdown document with **bold** and *italic* text.

### Features
- Feature 1
- Feature 2
- Feature 3

```python
def example():
    return "code block"
```

## Conclusion
End of document."""),
        ("latex", r"""Consider the equation:
$$\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$

We can also write inline math like $E = mc^2$ and:
$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$"""),
    ]

    print("=" * 80)
    print("CONTENT-BASED ADAPTIVE RENDERING TEST (WITH LAYOUT DETECTION)")
    print("=" * 80)

    renderer = AdaptiveVelloRenderer()

    for name, text in test_cases:
        # Analyze layout first
        layout_info = analyze_text_layout(text)

        start = time.time()
        results = renderer.render_batch([text], return_dimensions=True)
        render_time = (time.time() - start) * 1000

        image, (width, height) = results[0]

        # Save
        output_path = output_dir / f"content_{name}.png"
        Image.fromarray(image).save(output_path)

        # Analyze
        gray = np.mean(image, axis=2)
        non_white = np.where(gray <= 240)
        if len(non_white[0]) > 0:
            content_h = non_white[0].max() - non_white[0].min() + 1
            content_w = non_white[1].max() - non_white[1].min() + 1
        else:
            content_h = content_w = 0

        utilization = (content_h * content_w) / (width * height) if width * height > 0 else 0
        qwen_tokens = (height // 32) * (width // 32)

        print(f"\n{name}:")
        print(f"  Chars: {len(text)}")
        print(f"  Layout: {layout_info['layout_type']} (newlines={layout_info['preserve_newlines']}, mono={layout_info['use_monospace']})")
        print(f"  Canvas: {width}×{height}")
        print(f"  Content: {content_w}×{content_h}")
        print(f"  Utilization: {utilization:.1%}")
        print(f"  Qwen3VL tokens: {qwen_tokens}")
        print(f"  Time: {render_time:.2f}ms")

    print(f"\n✓ Images saved to {output_dir.absolute()}")
    print("\nLayout Detection Summary:")
    print("  - 'plain': Single line → wrap naturally")
    print("  - 'paragraph': Multi-line text → wrap naturally")
    print("  - 'list': Short lines with newlines → preserve structure")
    print("  - 'structured': Sections with empty lines → preserve structure")
    print("  - 'code': Indentation/braces/keywords → preserve lines, monospace")
    print("  - 'markdown': Headers/lists/code blocks → preserve structure")
    print("  - 'latex': Math formulas → preserve structure, no wrapping")


if __name__ == "__main__":
    main()
