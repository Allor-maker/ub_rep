"""Single gateway entry point; solvers should depend on LLMSession, not this module."""

from .gateway import NeuralDeepGateway

__all__ = ["NeuralDeepGateway"]
