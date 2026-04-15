"""
Global evaluation configuration module.

Provides consistent paths and environment loading for all benchmark scripts
when run from the evaluation root directory.
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from project_paths import hf_path

# Get the evaluation root directory
# This works whether script is run directly or as a module
_SCRIPT_DIR = Path(__file__).parent.absolute()
_EVAL_ROOT = _SCRIPT_DIR

# Paths
ENV_FILE = _EVAL_ROOT / ".env"
DATA_ROOT = _EVAL_ROOT / "data"
MODEL_ROOT = hf_path("Qwen")

# Load environment variables from .env
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"✓ Loaded environment from {ENV_FILE}")
else:
    print(f"⚠️  Warning: .env file not found at {ENV_FILE}")


def get_data_path(*parts):
    """Get a path relative to the data directory."""
    return str(DATA_ROOT / "/".join(parts))


def get_results_path(benchmark, *parts):
    """Get a path relative to a benchmark's results directory."""
    return str(_EVAL_ROOT / benchmark / "results" / "/".join(parts))


def resolve_path(path):
    """Resolve a path relative to evaluation root if not absolute."""
    if os.path.isabs(path):
        return path

    # Handle paths that were relative to benchmark subdirectories (e.g., ../data/...)
    # These should be resolved relative to evaluation root, not with their ../ prefix
    path_str = str(path)

    # If path starts with ../, it was relative to a benchmark subdirectory
    # Replace ../ with nothing to make it relative to evaluation root
    if path_str.startswith('../'):
        path_str = path_str[3:]  # Remove ../ prefix

    # Join with evaluation root
    resolved = (_EVAL_ROOT / path_str)

    # Resolve any remaining .. but don't follow symlinks outside eval root
    try:
        resolved = resolved.resolve()
    except (OSError, RuntimeError):
        # If resolve fails, use absolute path without resolve
        resolved = _EVAL_ROOT / path_str

    return str(resolved)


# Default model paths
QWEN3_VL_2B_THINKING = MODEL_ROOT / "Qwen3-VL-2B-Thinking"
QWEN3_VL_7B = MODEL_ROOT / "Qwen3-VL-7B"
