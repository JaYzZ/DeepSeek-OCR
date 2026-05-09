"""Qwen package entrypoint.

Keep package import side effects minimal.

Training-time patch application is handled by explicit integration imports and
sitecustomize, so the package root must not eagerly re-import the latent
integration module during package initialization.
"""

from __future__ import annotations

__all__ = []
