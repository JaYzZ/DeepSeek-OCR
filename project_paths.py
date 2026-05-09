"""Path helpers derived from the single project-root env variable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR_ENV = "ROOT_DIR"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def resolve_project_path(path: str | Path, repo_root: str | Path | None = None) -> Path:
    """Resolve a path against the repo root or the global project root.

    Absolute paths are returned unchanged.

    Relative paths are first resolved against ``repo_root`` (or the
    DeepSeek-OCR repo root by default). If that candidate does not exist, the
    global ``ROOT_DIR`` project root is tried next. This supports mixed asset
    references such as repo-local ``Qwen/...`` paths and project-level
    ``huggingface/...`` or ``sources/...`` paths.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    bases = [Path(repo_root).expanduser() if repo_root is not None else get_deepseek_ocr_dir()]
    project_root = get_project_root()
    if project_root not in bases:
        bases.append(project_root)
    for base in bases:
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return bases[0] / candidate


def sources_path(*parts: str) -> Path:
    return get_sources_dir().joinpath(*parts)


def hf_path(*parts: str) -> Path:
    return get_huggingface_dir().joinpath(*parts)


def env_path(*parts: str) -> Path:
    return get_envs_dir().joinpath(*parts)
