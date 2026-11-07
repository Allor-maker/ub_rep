"""Explicit solver registration and replaceable, side-effect-free policy."""

from .baseline import BaselinePlanner, next_not_before_wakeup
from .registry import StaticSolverRegistry

__all__ = ["BaselinePlanner", "StaticSolverRegistry", "next_not_before_wakeup"]
