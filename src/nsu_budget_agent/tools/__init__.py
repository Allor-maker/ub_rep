"""Read-only local production tools; no subprocess, shell or network backend."""

from .local import LocalToolExecutor

__all__ = ["LocalToolExecutor"]
