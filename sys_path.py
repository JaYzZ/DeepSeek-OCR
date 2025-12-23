"""Shared helper for guarded sys.path updates across the project."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union


def _add_sys_path(path: Union[str, "Path"]) -> None:
    """Append an existing directory to sys.path if it's not already present."""
    abs_path = Path(path).expanduser().resolve()
    if abs_path.is_dir():
        str_path = str(abs_path)
        if str_path not in sys.path:
            sys.path.append(str_path)
