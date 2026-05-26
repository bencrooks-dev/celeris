"""celeris — a JIT compiler/runtime for a statically-typed numeric subset of Python."""

__version__ = "0.1.0"

from .runtime import fast_runtime

__all__ = ["fast_runtime", "__version__"]
