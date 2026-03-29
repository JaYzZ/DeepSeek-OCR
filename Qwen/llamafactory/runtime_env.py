#!/usr/bin/env python3
"""Helpers for Qwen runtime environment variables."""

from __future__ import annotations

import os
from typing import Optional


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a non-empty runtime env value."""
    value = os.environ.get(name)
    if value is not None and value != "":
        return value
    return default


def get_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean runtime env flag."""
    raw = get_env(name, None)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
