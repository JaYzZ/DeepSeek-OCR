"""Path helpers derived from the single project-root env variable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR_ENV = "ROOT_DIR"
DEFAULT_PROJECT_ROOT = Path("/share/project/xiyan")


def get_project_root() -> Path:
    """Return the top-level project root controlled by ROOT_DIR."""
    return Path(os.environ.get(ROOT_DIR_ENV, DEFAULT_PROJECT_ROOT)).expanduser()


def get_sources_dir() -> Path:
    return get_project_root() / "sources"


def get_deepseek_ocr_dir() -> Path:
    return get_sources_dir() / "DeepSeek-OCR"


def get_huggingface_dir() -> Path:
    return get_project_root() / "huggingface"


def get_envs_dir() -> Path:
    return get_project_root() / "envs"


def get_env_python(env_name: str = "ocrflow") -> Path:
    return get_envs_dir() / env_name / "bin" / "python"


def deepseek_ocr_path(*parts: str) -> Path:
    return get_deepseek_ocr_dir().joinpath(*parts)


def sources_path(*parts: str) -> Path:
    return get_sources_dir().joinpath(*parts)


def hf_path(*parts: str) -> Path:
    return get_huggingface_dir().joinpath(*parts)


def env_path(*parts: str) -> Path:
    return get_envs_dir().joinpath(*parts)
