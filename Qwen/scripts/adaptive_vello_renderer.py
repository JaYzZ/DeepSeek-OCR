#!/usr/bin/env python3
"""
Content-Based Adaptive Vello Renderer for Qwen3VL

Calculates optimal H×W dimensions based on actual text measurement:
- Uses minimum readable font size
- Calculates actual space needed with proper line wrapping
- Supports asymmetric H×W for efficiency
- Optimized for Qwen3VL's native variable resolution policy
"""

import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from PIL import Image

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from Renderer.skia_renderer import prepare_text_for_rendering
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
    min_size: int = 32,                  # Minimum dimension before processor-side upscaling
    max_size: int = 4096,                # Maximum dimension
    vit_divisor: int = 32,               # Round to multiple (for Qwen3VL efficiency)
    padding: int = 15,                   # Padding in pixels (small for efficiency)
    safety_multiplier: float = 1.5,      # Safety margin (50% extra to handle Vello auto-sizing)
    aspect_ratio_constraint: Optional[float] = 200.0,  # Very loose cap for extreme rectangles
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
        min_size: Minimum dimension for either H or W before processor-side resizing
        max_size: Maximum dimension for either H or W
        vit_divisor: Round dimensions to multiple of this (32 for Qwen3VL)
        padding: Padding around text in pixels
        safety_multiplier: Extra space multiplier (e.g., 1.5 = 50% extra)
        aspect_ratio_constraint: Maximum aspect ratio (set high enough to rarely bind)
        allow_asymmetric: If False, returns square dimensions

    Returns:
        (width, height, layout_info) tuple
    """
    num_chars = len(text)

    if num_chars == 0:
        empty_layout = analyze_text_layout(text)
        return min_size, min_size, empty_layout

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
) -> tuple[str, dict]:
    """Unified newline-aware preprocessing for short, ragged text blocks."""
    return _pack_text_preserving_layout(
        text=text,
        layout_info=layout_info,
        short_line_threshold=short_line_threshold,
        min_lines_for_reflow=min_lines_for_reflow,
    )


def _pack_text_preserving_layout(
    text: str,
    layout_info: dict,
    short_line_threshold: int = 20,
    min_lines_for_reflow: int = 8,
) -> tuple[str, dict]:
    """Pack mutable text into even lines while preserving immutable layout.

    Strategy:
    - Split text into strict immutable and mutable chunks.
    - Strict immutable chunks preserve exact line layout.
    - Strict immutable chunks define the minimum packed line width.
    - Mutable chunks are merged/re-wrapped against that minimum width.
    - Width is inferred once from aggregate stats, keeping preprocessing O(n).
    """
    if not text or not text.strip():
        return text, layout_info

    def normalize_blank_lines(text_value: str) -> str:
        text_value = text_value.expandtabs(4)
        normalized_lines: list[str] = []
        previous_blank = False
        for raw_line in text_value.split("\n"):
            is_blank = not raw_line.strip()
            if is_blank:
                if previous_blank:
                    continue
                normalized_lines.append("")
                previous_blank = True
            else:
                normalized_lines.append(raw_line)
                previous_blank = False
        while normalized_lines and normalized_lines[0] == "":
            normalized_lines.pop(0)
        while normalized_lines and normalized_lines[-1] == "":
            normalized_lines.pop()
        return "\n".join(normalized_lines)

    text = normalize_blank_lines(text)
    if not text:
        return text, layout_info

    lines = text.split("\n")
    if len(lines) < 2 and len(text) < short_line_threshold * 2:
        return text, layout_info

    list_prefix = re.compile(r"^(\s*(?:[-*+]\s+|\d+\.\s+|[A-Za-z][\.\)]\s+))(.+?)\s*$")
    quote_prefix = re.compile(r"^(\s*>\s*)(.+?)\s*$")
    header_prefix = re.compile(r"^(\s*#+\s*)(.+?)\s*$")
    line_pack_separator = "    "  # Tab-width visual gap between originally separate packed lines.

    def classify_line(line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return "blank"
        if stripped.startswith(("```", "~~~")):
            return "immutable"
        if stripped.startswith("|"):
            return "immutable"
        if any(c in stripped for c in ("┌", "┐", "└", "┘", "│", "─", "├", "┤", "┬", "┴", "┼")):
            return "immutable"
        if stripped.count("|") >= 2:
            return "immutable"
        if "\t" in line:
            return "immutable"
        return "mutable"

    def longest_token_len(text_value: str) -> int:
        tokens = re.split(r"\s+", text_value.strip())
        return max((len(tok) for tok in tokens if tok), default=1)

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

    def render_item_with_width(item: dict, width: int) -> list[str]:
        if item.get("prepacked"):
            return [item["content"]]
        if "prefix" in item:
            if item.get("prefix_kind") == "list":
                return wrap_list_item(item["prefix"], item["content"], width)
            return wrap_prefixed_item(item["prefix"], item["content"], width)
        return wrap_plain_text(item["content"], width)

    def infer_global_width(block_items: list[dict], min_width: int) -> int:
        total_chars = sum(len(item["content"]) + len(item.get("prefix", "")) for item in block_items)
        longest_token = max(longest_token_len(item["content"]) for item in block_items)
        width_floor = max(min_width, longest_token + 2)

        item_count = max(len(block_items), 1)
        avg_len = max(1, total_chars // item_count)
        return max(width_floor, avg_len)

    def render_all_with_width(all_items: list[dict], width: int) -> list[str]:
        rendered: list[str] = []
        for item in all_items:
            kind = item["kind"]
            if kind == "blank":
                if rendered and rendered[-1] != "":
                    rendered.append("")
                elif not rendered:
                    rendered.append("")
                continue
            if kind == "immutable":
                rendered.append(item["text"])
                continue
            rendered.extend(render_item_with_width(item, width))
        while rendered and rendered[-1] == "":
            rendered.pop()
        return rendered

    def build_packed_items(all_items: list[dict], width: int) -> list[dict]:
        packed_items: list[dict] = []
        pending_plain_lines: list[str] = []
        pending_plain_len = 0
        separator_width = 4

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

    def score_width(all_items: list[dict], width: int) -> tuple[int, int, int, int]:
        packed_items = build_packed_items(all_items, width)
        rendered = render_all_with_width(packed_items, width)
        visible_lines = [line for line in rendered if line]
        if not visible_lines:
            return (0, 0, 0, width)

        max_len = max(len(line) for line in visible_lines)
        line_count = len(visible_lines)
        est_width_px = max_len * 3
        est_height_px = line_count * 12
        est_area = est_width_px * est_height_px
        return (est_area, est_height_px, est_width_px, max_len)

    def choose_balanced_width(all_items: list[dict], min_width: int) -> int:
        mutable_widths = [
            len(item.get("prefix", "")) + len(item["content"])
            for item in all_items
            if item["kind"] == "mutable"
        ]
        if not mutable_widths:
            return min_width

        lower = infer_global_width([item for item in all_items if item["kind"] == "mutable"], min_width)
        upper = max(lower, max(mutable_widths) // 2)
        if lower >= upper:
            return lower

        left = lower
        right = upper
        while right - left > 3:
            mid = (left + right) // 2
            mid_score = score_width(all_items, mid)
            next_score = score_width(all_items, mid + 1)
            if next_score <= mid_score:
                left = mid + 1
            else:
                right = mid

        best_width = left
        best_score: Optional[tuple[int, int, int, int]] = None
        for width in range(left, right + 1):
            score = score_width(all_items, width)
            if best_score is None or score < best_score:
                best_score = score
                best_width = width
        return best_width

    items: list[dict] = []
    immutable_width_floor = 0
    for line in lines:
        line_class = classify_line(line)
        if line_class == "blank":
            items.append({"kind": "blank", "text": ""})
            continue
        if line_class == "immutable":
            items.append({"kind": "immutable", "text": line})
            immutable_width_floor = max(immutable_width_floor, len(line))
            continue
        list_match = list_prefix.match(line)
        if list_match:
            prefix, content = list_match.groups()
            items.append({"kind": "mutable", "prefix": prefix, "prefix_kind": "list", "content": content})
            immutable_width_floor = max(immutable_width_floor, len(prefix))
            continue
        quote_match = quote_prefix.match(line)
        header_match = header_prefix.match(line)
        if quote_match:
            prefix, content = quote_match.groups()
            items.append({"kind": "mutable", "prefix": prefix, "content": content})
            immutable_width_floor = max(immutable_width_floor, len(prefix))
            continue
        if header_match:
            prefix, content = header_match.groups()
            items.append({"kind": "mutable", "prefix": prefix, "content": content})
            immutable_width_floor = max(immutable_width_floor, len(prefix))
            continue
        leading_ws_len = len(line) - len(line.lstrip(" "))
        if leading_ws_len > 0:
            prefix = line[:leading_ws_len]
            items.append({"kind": "mutable", "prefix": prefix, "content": line[leading_ws_len:].strip()})
            immutable_width_floor = max(immutable_width_floor, len(prefix))
            continue
        items.append({"kind": "mutable", "content": line.strip()})

    mutable_items = [item for item in items if item["kind"] == "mutable"]
    mutable_char_total = sum(
        len(item["content"]) + len(item.get("prefix", ""))
        for item in mutable_items
    )

    if len(mutable_items) < 2 and immutable_width_floor == 0 and mutable_char_total < short_line_threshold * 2:
        return text, layout_info

    if len(lines) < min_lines_for_reflow and immutable_width_floor == 0 and mutable_char_total < short_line_threshold * min_lines_for_reflow:
        return text, layout_info

    if not mutable_items:
        packed_lines = []
        for item in items:
            kind = item["kind"]
            if kind == "blank":
                if packed_lines and packed_lines[-1] != "":
                    packed_lines.append("")
                elif not packed_lines:
                    packed_lines.append("")
                continue
            if kind == "immutable":
                packed_lines.append(item["text"])
                continue
        while packed_lines and packed_lines[-1] == "":
            packed_lines.pop()
    else:
        min_width = max(8, immutable_width_floor)
        global_width = choose_balanced_width(items, min_width)
        packed_items = build_packed_items(items, global_width)
        packed_lines = render_all_with_width(packed_items, global_width)

    if not packed_lines:
        return text, layout_info

    updated = dict(layout_info)
    updated["preserve_newlines"] = True
    updated["layout_type"] = f"{layout_info.get('layout_type', 'unknown')}_packed"
    updated["num_lines"] = len(packed_lines)
    updated["max_line_chars"] = max((len(line) for line in packed_lines), default=0)
    return "\n".join(packed_lines), updated

def build_piecewise_bucket_sizes(min_size: int, max_size: int) -> List[int]:
    """Build canonical dimension buckets for adaptive render snapping."""
    bucket_values = set()
    bucket_values.update(range(32, 257, 32))
    bucket_values.update(range(320, 1025, 64))
    bucket_values.update(range(1280, 4097, 256))

    sizes = [size for size in sorted(bucket_values) if min_size <= size <= max_size]
    if min_size not in sizes:
        sizes.insert(0, min_size)
    if max_size not in sizes:
        sizes.append(max_size)
    return sorted(set(sizes))


class AdaptiveVelloRenderer:
    """
    Content-based adaptive renderer for Qwen3VL.

    Features:
    - Measures actual text space requirements
    - Asymmetric H×W based on content (e.g., 192×256 for tall lists)
    - Minimum 32×32 render canvas
    - Maximum 4096×4096
    - All dimensions divisible by 32
    - GPU-accelerated Vello rendering
    """

    def __init__(
        self,
        min_vello_font: float = 8.0,     # Vello minimum font size
        max_vello_font: float = 10.0,    # Vello maximum font size
        min_size: int = 32,
        max_size: int = 4096,
        vit_divisor: int = 32,
        padding: int = 12,               # Small padding for efficiency
        safety_multiplier: float = 1.5,  # 50% extra (increased to handle Vello's auto-sizing)
        aspect_ratio_constraint: Optional[float] = 200.0,
        allow_asymmetric: bool = True,
        max_retries: int = 3,            # Max attempts to fit text (auto-retry if truncated)
        edge_margin: int = 5,            # Minimum margin from edges (for truncation detection)
        thinking_padding: int = 8,       # Smaller padding for thinking chunks
        thinking_safety_multiplier: float = 1.25,  # Reduced safety for thinking chunks
        short_line_wrap_threshold: int = 20,       # Reflow if many lines shorter than this
        short_line_min_lines: int = 8,             # Minimum lines to trigger reflow
        short_line_target_width_chars: int = 80,   # Wrapped line width for reflowed content
        minimize_after_fit: bool = True,           # Shrink each successful render to the smallest fitting bucket
    ):
        """
        Initialize content-based adaptive renderer.

        Uses Vello's min font size for calculation, then lets Vello auto-size
        between min and max font to fill the canvas optimally.

        Args:
            min_vello_font: Minimum font size for Vello renderer (used for size calculation)
            max_vello_font: Maximum font size for Vello renderer
            min_size: Minimum canvas dimension
            max_size: Maximum canvas dimension
            vit_divisor: Round dimensions to multiple of this
            padding: Padding around text in pixels
            safety_multiplier: Extra space multiplier (e.g., 1.5 = 50% extra)
            aspect_ratio_constraint: Maximum aspect ratio (None for no limit)
            allow_asymmetric: Enable asymmetric H×W rendering
            max_retries: Maximum attempts to fit text (auto-retry with larger size if truncated)
            edge_margin: Minimum margin from edges in pixels (for truncation detection)
            minimize_after_fit: If True, shrink each successful render to the smallest fitting bucket
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
        self.minimize_after_fit = minimize_after_fit
        self.dimension_buckets = build_piecewise_bucket_sizes(self.min_size, self.max_size)
        self._renderer_cache: Dict[tuple[int, int, int, bool], VelloRenderer] = {}

    def _get_renderer(
        self,
        width: int,
        height: int,
        padding: int,
        preserve_newlines: bool,
    ) -> VelloRenderer:
        key = (width, height, padding, preserve_newlines)
        renderer = self._renderer_cache.get(key)
        if renderer is None:
            renderer = VelloRenderer(
                width=width,
                height=height,
                padding=padding,
                min_font_size=self.min_vello_font,
                max_font_size=self.max_vello_font,
                preserve_newlines=preserve_newlines,
            )
            self._renderer_cache[key] = renderer
        return renderer

    def _render_batch_once(
        self,
        texts: List[str],
        width: int,
        height: int,
        padding: int,
        preserve_newlines: bool,
    ) -> List[np.ndarray]:
        renderer = self._get_renderer(width, height, padding, preserve_newlines)
        return renderer.render_batch(texts)

    def _round_up_dim(self, value: int) -> int:
        value = max(self.min_size, min(self.max_size, value))
        return ((value + self.vit_divisor - 1) // self.vit_divisor) * self.vit_divisor

    def _snap_dim_up(self, value: int) -> int:
        rounded = self._round_up_dim(value)
        for bucket in self.dimension_buckets:
            if bucket >= rounded:
                return bucket
        return self.dimension_buckets[-1]

    def _render_once(
        self,
        text: str,
        width: int,
        height: int,
        padding: int,
        preserve_newlines: bool,
    ) -> np.ndarray:
        return self._render_batch_once([text], width, height, padding, preserve_newlines)[0]

    def shutdown(self) -> None:
        for renderer in self._renderer_cache.values():
            renderer.shutdown()
        self._renderer_cache.clear()
        return None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def _shrink_axis(
        self,
        text: str,
        width: int,
        height: int,
        padding: int,
        preserve_newlines: bool,
        axis: str,
        best_image: Optional[np.ndarray] = None,
    ) -> tuple[int, np.ndarray]:
        current_dim = width if axis == "width" else height
        candidate_dims = [dim for dim in self.dimension_buckets if dim <= current_dim]
        if not candidate_dims:
            candidate_dims = [self.dimension_buckets[0]]

        best_dim = current_dim
        if best_image is None:
            best_image = self._render_once(text, width, height, padding, preserve_newlines)

        low = 0
        high = len(candidate_dims) - 1
        while low <= high:
            mid_idx = (low + high) // 2
            mid = candidate_dims[mid_idx]
            if mid >= best_dim:
                high = mid_idx - 1
                continue
            test_width = mid if axis == "width" else width
            test_height = mid if axis == "height" else height
            image = self._render_once(text, test_width, test_height, padding, preserve_newlines)

            if self._is_truncated(image):
                low = mid_idx + 1
            else:
                best_dim = mid
                best_image = image
                high = mid_idx - 1

        return best_dim, best_image

    def _minimize_canvas(
        self,
        text: str,
        width: int,
        height: int,
        padding: int,
        preserve_newlines: bool,
        best_image: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, int, int]:
        width, image = self._shrink_axis(
            text, width, height, padding, preserve_newlines, axis="width", best_image=best_image
        )
        height, image = self._shrink_axis(
            text,
            width,
            height,
            padding,
            preserve_newlines,
            axis="height",
            best_image=image,
        )
        return image, width, height

    def _prepare_text(self, text: str) -> tuple[str, dict]:
        return prepare_text_for_rendering(
            text=text,
            short_line_threshold=self.short_line_wrap_threshold,
            min_lines_for_reflow=self.short_line_min_lines,
        )

    def _estimate_min_canvas(
        self,
        text: str,
        padding: int,
        preserve_newlines: bool,
    ) -> tuple[int, int]:
        lines = text.split("\n") if preserve_newlines else [text.replace("\n", " ").strip()]
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return self.min_size, self.min_size

        max_line_chars = max(len(line) for line in non_empty_lines)
        line_count = len(non_empty_lines) if preserve_newlines else 1

        # Fixed visual budgeting for pre-render search.
        char_width_px = 3.0
        line_height_px = 12.0
        min_width = self._snap_dim_up(int(max_line_chars * char_width_px + 2 * padding))
        min_height = self._snap_dim_up(int(line_count * line_height_px + 2 * padding))
        return min_width, min_height

    def _iter_canvas_candidates(
        self,
        min_width: int,
        min_height: int,
    ) -> List[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        if not self.allow_asymmetric:
            for size in self.dimension_buckets:
                if size < max(min_width, min_height):
                    continue
                candidates.append((size, size))
            return candidates

        for width in self.dimension_buckets:
            for height in self.dimension_buckets:
                if width < min_width or height < min_height:
                    continue
                if self.aspect_ratio_constraint is not None:
                    ratio = max(width, height) / max(min(width, height), 1)
                    if ratio > self.aspect_ratio_constraint:
                        continue
                candidates.append((width, height))

        candidates.sort(key=lambda dims: (dims[0] * dims[1], max(dims), dims[0], dims[1]))
        return candidates

    def _render_adaptive_candidate(
        self,
        text: str,
        base_padding: int,
        base_safety: float,
        preserve_newlines_override: Optional[bool] = None,
    ) -> tuple[np.ndarray, int, int, bool]:
        layout_info = analyze_text_layout(text)
        preserve_newlines = (
            preserve_newlines_override
            if preserve_newlines_override is not None
            else layout_info["preserve_newlines"]
        )
        min_width, min_height = self._estimate_min_canvas(text, base_padding, preserve_newlines)
        final_image: Optional[np.ndarray] = None
        final_width = self.min_size
        final_height = self.min_size
        final_truncated = True

        for width, height in self._iter_canvas_candidates(min_width, min_height):
            image = self._render_once(text, width, height, base_padding, preserve_newlines)
            is_truncated = self._is_truncated(image)

            final_image = image
            final_width = width
            final_height = height
            final_truncated = is_truncated

            if not is_truncated:
                return image, width, height, False

        assert final_image is not None
        return final_image, final_width, final_height, final_truncated

    def _render_grouped_candidates(
        self,
        candidate_specs: List[Optional[dict]],
        base_padding: int,
        base_safety: float,
    ) -> List[Optional[tuple[np.ndarray, int, int, bool]]]:
        results: List[Optional[tuple[np.ndarray, int, int, bool]]] = [None] * len(candidate_specs)
        last_attempts: List[Optional[tuple[np.ndarray, int, int, bool]]] = [None] * len(candidate_specs)
        pending = [idx for idx, spec in enumerate(candidate_specs) if spec is not None]

        min_requirements: Dict[int, tuple[int, int]] = {}
        for idx in pending:
            spec = candidate_specs[idx]
            assert spec is not None
            min_requirements[idx] = self._estimate_min_canvas(
                spec["text"],
                base_padding,
                spec["preserve_newlines"],
            )

        candidate_pool = sorted(
            {
                dims
                for idx in pending
                for dims in self._iter_canvas_candidates(*min_requirements[idx])
            },
            key=lambda dims: (dims[0] * dims[1], max(dims), dims[0], dims[1]),
        )

        for width, height in candidate_pool:
            if not pending:
                break
            grouped: Dict[tuple[int, int, int, bool], List[int]] = {}

            for idx in pending:
                spec = candidate_specs[idx]
                assert spec is not None
                min_width, min_height = min_requirements[idx]
                if width < min_width or height < min_height:
                    continue
                preserve_newlines = spec["preserve_newlines"]
                grouped.setdefault((width, height, base_padding, preserve_newlines), []).append(idx)

            next_pending = []
            for (width, height, padding, preserve_newlines), idxs in grouped.items():
                texts = [candidate_specs[idx]["text"] for idx in idxs if candidate_specs[idx] is not None]
                images = self._render_batch_once(texts, width, height, padding, preserve_newlines)
                for idx, image in zip(idxs, images):
                    is_truncated = self._is_truncated(image)
                    last_attempts[idx] = (image, width, height, is_truncated)
                    if is_truncated:
                        next_pending.append(idx)
                        continue

                    final_width = width
                    final_height = height
                    results[idx] = (image, final_width, final_height, False)

            pending = next_pending

        for idx in pending:
            results[idx] = last_attempts[idx]

        return results

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
        final_results: List[Optional[tuple[np.ndarray, int, int, bool]]] = [None] * len(texts)

        primary_specs: List[Optional[dict]] = []
        for text in texts:
            prepared_text, layout_info = self._prepare_text(text)
            primary_specs.append(
                {
                    "text": prepared_text,
                    "preserve_newlines": layout_info["preserve_newlines"],
                }
            )

        # Thinking preset: smaller padding and tighter safety to reduce token inflation.
        base_padding = self.thinking_padding if thinking_mode else self.padding
        base_safety = self.thinking_safety_multiplier if thinking_mode else self.safety_multiplier

        grouped_results = self._render_grouped_candidates(primary_specs, base_padding, base_safety)
        for idx, result in enumerate(grouped_results):
            final_results[idx] = result

        for idx, result in enumerate(final_results):
            if result is None:
                image, width, height, is_truncated = self._render_adaptive_candidate(
                    primary_specs[idx]["text"],
                    base_padding=base_padding,
                    base_safety=base_safety,
                    preserve_newlines_override=primary_specs[idx]["preserve_newlines"],
                )
            else:
                image, width, height, is_truncated = result
            if is_truncated:
                print(
                    f"Warning: Text may be truncated after {self.max_retries} attempts "
                    f"(final size: {width}×{height})"
                )
            if return_dimensions:
                results.append((image, (width, height)))
            else:
                results.append(image)

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
