"""Shared pytest fixtures and skip markers for the celeris test suite.

The reusable marker objects below let tests opt in to capabilities that may be
absent in a given environment:

- ``needs_clang``   — requires a ``clang++`` on PATH (runtime source-gen / golden kernels)
- ``needs_llvmlite``— requires the optional ``llvmlite`` package (LLVM backend)
- ``needs_native``  — requires the CMake-built ``celeris_native`` pybind11 module
"""

import importlib.util
import shutil

import pytest

needs_clang = pytest.mark.skipif(
    shutil.which("clang++") is None,
    reason="clang++ not available",
)

needs_llvmlite = pytest.mark.skipif(
    importlib.util.find_spec("llvmlite") is None,
    reason="llvmlite not installed",
)


def _native_available():
    return importlib.util.find_spec("celeris_native") is not None


needs_native = pytest.mark.skipif(
    not _native_available(),
    reason="celeris_native not built",
)
