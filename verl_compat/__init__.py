"""Repo-local compatibility helpers for the VERL runtime."""

from .bootstrap import apply_runtime_compat_patches, patch_runtime_env, patch_worker_env_vars

__all__ = ["apply_runtime_compat_patches", "patch_runtime_env", "patch_worker_env_vars"]
