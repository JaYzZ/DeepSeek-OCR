"""Helpers for resolving local HuggingFace model paths."""

from __future__ import annotations

import os
from pathlib import Path
from project_paths import get_huggingface_dir

DEFAULT_MODEL_BASE = get_huggingface_dir()


def resolve_model_path(model_id: str, *, model_base: str | os.PathLike | None = None) -> str:
    """Return the local mirror path if it exists, otherwise the raw model id."""
    base_dir = Path(model_base or DEFAULT_MODEL_BASE)
    local_path = base_dir / model_id
    if local_path.exists():
        return str(local_path)
    return model_id
