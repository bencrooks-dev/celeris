"""Shared pytest fixtures and capability markers for the celeris test suite.

The named markers below let tests opt in to capabilities that may be absent in a
given environment. They are real, selectable markers (``-m "needs_llvmlite"``
selects them); a ``pytest_runtest_setup`` hook skips a marked test when its
capability is missing:

- ``needs_clang``   — requires a ``clang++`` on PATH (runtime source-gen / golden kernels)
- ``needs_llvmlite``— requires the optional ``llvmlite`` package (LLVM backend)
- ``needs_native``  — requires the CMake-built ``celeris_native`` pybind11 module
"""

import importlib.util
import shutil

import pytest

# Named capability markers: `-m "needs_clang"` selects them; the hook skips
# them when the capability is absent.
needs_clang = pytest.mark.needs_clang
needs_llvmlite = pytest.mark.needs_llvmlite
needs_native = pytest.mark.needs_native

_CAP = {
    "needs_clang": (lambda: shutil.which("clang++") is None, "clang++ not available"),
    "needs_llvmlite": (lambda: importlib.util.find_spec("llvmlite") is None, "llvmlite not installed"),
    "needs_native": (lambda: importlib.util.find_spec("celeris_native") is None, "celeris_native not built"),
}


def pytest_runtest_setup(item):
    for marker in item.iter_markers():
        cap = _CAP.get(marker.name)
        if cap and cap[0]():
            pytest.skip(cap[1])
