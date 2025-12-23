"""Helpers for resolving local HuggingFace model paths.

All models are stored under `/share/project/xiyan/huggingface/{model_id}` where
`model_id` matches the public HuggingFace identifier (e.g., `Qwen/Qwen3-VL-7B`
→ `/share/project/xiyan/huggingface/Qwen/Qwen3-VL-7B`). If the local copy is
missing, we fall back to the raw model id so callers can still download/cache
as usual.
"""

from __future__ import annotations

import os
from pathlib import Path

# Default location for mirrored HuggingFace models.
DEFAULT_MODEL_BASE = Path("/share/project/xiyan/huggingface")
# Optional override for environments that mirror models elsewhere.
MODEL_BASE_ENV = "HF_LOCAL_MODEL_DIR"


def resolve_model_path(model_id: str, *, model_base: str | os.PathLike | None = None) -> str:
    """Return the local mirror path if it exists, otherwise the raw model id."""
    base_dir = Path(
        model_base
        or os.environ.get(MODEL_BASE_ENV, DEFAULT_MODEL_BASE)
    )
    local_path = base_dir / model_id
    if local_path.exists():
        return str(local_path)
    return model_id
