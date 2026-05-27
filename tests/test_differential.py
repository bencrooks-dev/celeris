"""Differential correctness harness: every available backend must agree with
plain-Python execution for each kernel. Plain Python (the undecorated function)
is ground truth. Backends that cannot compile a given shape are skipped."""
import importlib.util
import shutil

import numpy as np
import pytest

from celeris.parser import parse_function
from celeris.verifier import verify_ir
from celeris.passes import optimize
from celeris.backends import available_backends
from celeris.types import F64Array, F64Array2D


def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        y[i] = a * x[i] + y[i]

def scale(a: float, x: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        y[i] = a * x[i]

def vsum(x: F64Array, n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + x[i]
    return acc

def dot(x: F64Array, y: F64Array, n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + x[i] * y[i]
    return acc

def fused(a: float, b: float, x: F64Array, y: F64Array, z: F64Array, n: int) -> None:
    for i in range(n):
        z[i] = a * x[i] + b * y[i] + z[i]

def sum_evens(n: int) -> int:
    s = 0
    i = 0
    while i < n:
        if i % 2 == 0:
            s = s + i
        i = i + 1
    return s

def floordiv_loop(n: int) -> int:
    acc = 0
    for i in range(n):
        acc = acc + ((i - 5) // 2)   # negative operands: Python floor semantics
    return acc

def mod_loop(n: int) -> int:
    acc = 0
    for i in range(n):
        acc = acc + ((i - 5) % 3)    # negative operands: Python floored modulo
    return acc

def chain(a: float, x: F64Array, t: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        t[i] = a * x[i]
    for i in range(n):
        y[i] = t[i] + 1.0

def shifted_chain(a: float, x: F64Array, t: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        t[i + 1] = a * x[i]
    for i in range(n):
        y[i] = t[i]

def rowsum(a: F64Array2D, y: F64Array, m: int, k: int) -> None:
    for i in range(m):
        acc = 0.0
        for j in range(k):
            acc = acc + a[i, j]
        y[i] = acc


def _run_saxpy(f):
    x = np.arange(16, dtype=np.float64); y = np.ones(16, dtype=np.float64)
    f(2.5, x, y, 16); return y

def _run_scale(f):
    x = np.arange(16, dtype=np.float64); y = np.zeros(16, dtype=np.float64)
    f(3.0, x, y, 16); return y

def _run_sum(f):
    x = np.arange(64, dtype=np.float64); return np.float64(f(x, 64))

def _run_dot(f):
    x = np.arange(32, dtype=np.float64); y = np.arange(32, dtype=np.float64)
    return np.float64(f(x, y, 32))

def _run_fused(f):
    x = np.arange(16, dtype=np.float64); y = np.arange(16, dtype=np.float64) * 2.0
    z = np.ones(16, dtype=np.float64); f(1.5, 2.0, x, y, z, 16); return z

def _run_sum_evens(f):
    return np.int64(f(20))

def _run_floordiv(f):
    return np.int64(f(12))

def _run_mod(f):
    return np.int64(f(12))

def _run_chain(f):
    x = np.arange(16, dtype=np.float64)
    t = np.zeros(16, dtype=np.float64)
    y = np.zeros(16, dtype=np.float64)
    f(2.0, x, t, y, 16)
    return y

def _run_shifted_chain(f):
    x = np.arange(16, dtype=np.float64)
    t = np.zeros(17, dtype=np.float64)   # size n+1 so t[i+1] and t[i] are in bounds
    y = np.zeros(16, dtype=np.float64)
    f(2.0, x, t, y, 16)
    return y

def _run_rowsum(f):
    # CONTIGUOUS 2-D array: every backend (interpreter+sourcegen+llvm) compiles
    # and must agree under the hardened CASES asserts below.
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    y = np.zeros(3, dtype=np.float64); f(a, y, 3, 4); return y


# celeris's own backends. The global backend registry is process-wide and other
# tests register throwaway backends into it (e.g. test_backends_registry's
# "dummy"); restrict the oracle comparison to celeris's real backends so a
# leaked foreign backend cannot poison this harness under any collection order.
_CELERIS_BACKENDS = {"interpreter", "sourcegen", "kernels", "llvm"}


CASES = [
    ("saxpy", saxpy, _run_saxpy),
    ("scale", scale, _run_scale),
    ("sum", vsum, _run_sum),
    ("dot", dot, _run_dot),
    ("fused", fused, _run_fused),
    ("sum_evens", sum_evens, _run_sum_evens),
    ("floordiv_loop", floordiv_loop, _run_floordiv),
    ("mod_loop", mod_loop, _run_mod),
    ("chain", chain, _run_chain),
    ("shifted_chain", shifted_chain, _run_shifted_chain),
    ("rowsum", rowsum, _run_rowsum),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_all_backends_agree_with_python(case):
    name, pyfunc, run = case
    expected = run(pyfunc)                       # plain Python = ground truth
    k = parse_function(pyfunc); verify_ir(k); k = optimize(k)
    tested = []
    for be in available_backends():
        if be.name not in _CELERIS_BACKENDS:
            continue                              # ignore foreign/test backends
        try:
            compiled = be.compile(k)
        except Exception:
            continue                              # backend can't handle this shape
        got = run(compiled)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9,
                                   err_msg=f"{name} via {be.name} backend")
        tested.append(be.name)
    assert "interpreter" in tested, f"{name}: interpreter (oracle) did not run"
    if shutil.which("clang++") is not None:
        assert "sourcegen" in tested, f"{name}: sourcegen failed to compile a supported kernel"
    if importlib.util.find_spec("llvmlite") is not None:
        assert "llvm" in tested, f"{name}: llvm failed to compile a supported kernel"


def test_chain_is_actually_fused():
    from celeris.parser import parse_function
    from celeris.passes import optimize
    k = optimize(parse_function(chain))
    fors = [s for s in k["body"] if s["op"] == "for"]
    assert len(fors) == 1 and len(fors[0]["body"]) == 2, "chain must fuse to one loop"


def test_shifted_chain_is_fused():
    from celeris.parser import parse_function
    from celeris.passes import optimize
    k = optimize(parse_function(shifted_chain))
    fors = [s for s in k["body"] if s["op"] == "for"]
    assert len(fors) == 1 and len(fors[0]["body"]) == 2


def test_prange_parallel_matches_serial_oracle():
    from celeris.types import prange
    from celeris.parser import parse_function
    from celeris.verifier import verify_ir
    from celeris.passes import optimize
    from celeris.backends.interpreter import InterpreterBackend
    from celeris.backends.sourcegen import SourceGenBackend
    import shutil
    def psaxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in prange(n):
            y[i] = a * x[i] + y[i]
    n = 8192
    k = parse_function(psaxpy); verify_ir(k); k = optimize(k)
    def run(fn):
        x = np.arange(n, dtype=np.float64); y = np.ones(n, dtype=np.float64)
        fn(2.0, x, y, n); return y
    expected = run(psaxpy)                               # plain-Python serial oracle
    np.testing.assert_allclose(run(InterpreterBackend().compile(k)), expected, atol=1e-9)
    if shutil.which("clang++"):
        np.testing.assert_allclose(run(SourceGenBackend().compile(k)), expected, atol=1e-9)


def test_prange_reduction_serial_fallback_correct():
    from celeris.types import prange
    from celeris.parser import parse_function
    from celeris.backends.sourcegen import SourceGenBackend
    import shutil
    import pytest as _pt
    if not shutil.which("clang++"):
        _pt.skip("clang++ not available")
    def psum(x: F64Array, n: int) -> float:
        acc = 0.0
        for i in prange(n):
            acc = acc + x[i]
        return acc
    fn = SourceGenBackend().compile(parse_function(psum))
    x = np.arange(500, dtype=np.float64)
    assert abs(fn(x, 500) - x.sum()) < 1e-9


def test_rowsum_transposed_view_all_backends_agree():
    """A 2-D rowsum over a transposed (non-contiguous) NumPy view. The interpreter
    indexes the real NumPy array (the oracle); the native backends use general
    per-dim strides, so they must agree even though the view is not C-contiguous."""
    from celeris.parser import parse_function
    from celeris.verifier import verify_ir
    from celeris.passes import optimize
    from celeris.backends.interpreter import InterpreterBackend
    from celeris.backends.sourcegen import SourceGenBackend
    import shutil
    k = parse_function(rowsum); verify_ir(k); k = optimize(k)
    def run(fn):
        base = np.arange(12, dtype=np.float64).reshape(4, 3)
        a = base.T                                       # shape (3, 4), non-contiguous
        assert not a.flags["C_CONTIGUOUS"]
        y = np.zeros(3, dtype=np.float64); fn(a, y, 3, 4); return y
    expected = run(rowsum)                               # plain-Python NumPy oracle
    np.testing.assert_allclose(run(InterpreterBackend().compile(k)), expected, atol=1e-9)
    if shutil.which("clang++"):
        np.testing.assert_allclose(run(SourceGenBackend().compile(k)), expected, atol=1e-9)


def scale2d(a: F64Array2D, s: float, m: int, k: int) -> None:
    for i in range(m):
        for j in range(k):
            a[i, j] = a[i, j] * s + 1.0

def aug2d(a: F64Array2D, s: float, m: int, k: int) -> None:
    for i in range(m):
        for j in range(k):
            a[i, j] += s

def test_2d_inplace_write_matches_oracle():
    """In-place 2-D write through the strided lval path, validated against the
    plain-Python oracle on BOTH a contiguous array and a transposed view.

    ``scale2d`` uses a plain assign to a 2-D target (``a[i, j] = ...``);
    ``aug2d`` exercises 2-D augmented assignment (``a[i, j] += ...``), which the
    parser synthesises through ``lval_index_nd`` — the regression this guards."""
    from celeris.parser import parse_function
    from celeris.verifier import verify_ir
    from celeris.passes import optimize
    from celeris.backends.interpreter import InterpreterBackend
    from celeris.backends.sourcegen import SourceGenBackend
    import shutil
    for pyfunc in (scale2d, aug2d):
        k = parse_function(pyfunc); verify_ir(k); k = optimize(k)
        def run(fn, make):
            a = make(); fn(a, 3.0, a.shape[0], a.shape[1]); return a
        for make in (lambda: np.arange(12, dtype=np.float64).reshape(3, 4),
                     lambda: np.arange(12, dtype=np.float64).reshape(4, 3).T):
            expected = run(pyfunc, make)                 # plain-Python oracle
            np.testing.assert_allclose(
                run(InterpreterBackend().compile(k), make), expected, atol=1e-9)
            if shutil.which("clang++"):
                np.testing.assert_allclose(
                    run(SourceGenBackend().compile(k), make), expected, atol=1e-9)
