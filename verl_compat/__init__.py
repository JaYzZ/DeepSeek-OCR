"""Repo-local compatibility helpers for the VERL runtime."""

__all__ = ["apply_runtime_compat_patches", "patch_runtime_env", "patch_worker_env_vars"]


def __getattr__(name: str):
    if name in __all__:
        from . import bootstrap

        return getattr(bootstrap, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
