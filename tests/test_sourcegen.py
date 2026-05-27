import pytest
import numpy as np
from celeris.backends.sourcegen import SourceGenBackend
from celeris.parser import parse_function
from celeris.types import F64Array
from conftest import needs_clang
pytestmark = needs_clang

def test_sourcegen_saxpy():
    def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in range(n):
            y[i] = a * x[i] + y[i]
    fn = SourceGenBackend().compile(parse_function(saxpy))
    x = np.arange(1000, dtype=np.float64); y = np.zeros(1000, dtype=np.float64)
    fn(2.0, x, y, 1000)
    np.testing.assert_allclose(y, 2.0*np.arange(1000))

def test_sourcegen_scalar_return():
    def f(a: float, b: float) -> float:
        return a*b + 1.0
    fn = SourceGenBackend().compile(parse_function(f))
    assert abs(fn(3.0, 4.0) - 13.0) < 1e-9

def test_sourcegen_reduction_sum():
    def s(x: F64Array, n: int) -> float:
        acc = 0.0
        for i in range(n):
            acc = acc + x[i]
        return acc
    fn = SourceGenBackend().compile(parse_function(s))
    x = np.arange(50, dtype=np.float64)
    assert abs(fn(x, 50) - x.sum()) < 1e-9

def test_sourcegen_floordiv_negative_matches_python():
    def f(a: int, b: int) -> int:
        return a // b
    fn = SourceGenBackend().compile(parse_function(f))
    assert fn(-7, 2) == (-7 // 2) == -4

def test_sourcegen_floormod_negative_matches_python():
    def f(a: int, b: int) -> int:
        return a % b
    fn = SourceGenBackend().compile(parse_function(f))
    assert fn(-7, 2) == (-7 % 2) == 1

def test_sourcegen_available():
    assert SourceGenBackend().available() is True  # clang present in this env

def test_sourcegen_malicious_name_is_sanitized():
    import os
    import pathlib
    import celeris.ir as ir
    from celeris.backends.sourcegen import _CACHE, _safe_name

    # the helper itself never yields a path-traversal stem
    assert "/" not in _safe_name("../../evil")
    assert ".." not in _safe_name("../../evil")
    assert _safe_name("../../evil") == "______evil"

    # a hand-built (verifier-bypassing) IR with a path-traversal name. The real
    # C symbol name "../../evil" is not a valid C identifier, so clang rejects it
    # and compile raises -- but the key guarantee is that no .cpp/.so file is
    # written outside the cache dir.
    bad = ir.kernel("../../evil", [ir.param("a", "f64")], "f64",
                    [ir.ret(ir.var("a", "f64"))])

    cache = pathlib.Path(os.path.expanduser("~/.celeris_cache"))
    before = set(cache.glob("*")) if cache.exists() else set()
    escaped = [pathlib.Path(os.path.expanduser("~/evil.so")),
               pathlib.Path(os.path.expanduser("~/evil.cpp")),
               pathlib.Path("/tmp/pwn.so"), pathlib.Path("/tmp/pwn.cpp")]

    with pytest.raises(Exception):
        SourceGenBackend().compile(bad)

    # no file escaped the cache dir...
    for p in escaped:
        assert not p.exists()
    # ...and any new cache artifacts stay inside the cache dir with a safe stem
    after = set(cache.glob("*")) if cache.exists() else set()
    for p in (after - before):
        assert p.parent == cache
        assert ".." not in p.name


def test_sourcegen_floordiv_with_literal_compiles():
    def f(n: int) -> int:
        acc = 0
        for i in range(n):
            acc = acc + ((i - 5) // 2)
        return acc
    fn = SourceGenBackend().compile(parse_function(f))   # must NOT raise CompileError
    assert fn(12) == sum((i - 5) // 2 for i in range(12))

def test_sourcegen_mod_with_literal_compiles():
    def f(n: int) -> int:
        acc = 0
        for i in range(n):
            acc = acc + ((i - 5) % 3)
        return acc
    fn = SourceGenBackend().compile(parse_function(f))
    assert fn(12) == sum((i - 5) % 3 for i in range(12))


def test_sourcegen_int_const_returned_from_float_fn():
    def f(a: float) -> float:
        return 1
    fn = SourceGenBackend().compile(parse_function(f))
    assert abs(fn(2.0) - 1.0) < 1e-9


def test_sourcegen_prange_threaded_correct():
    from celeris.types import prange
    def psaxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in prange(n):
            y[i] = a * x[i] + y[i]
    n = 8192
    fn = SourceGenBackend().compile(parse_function(psaxpy))
    x = np.arange(n, dtype=np.float64); y = np.ones(n, dtype=np.float64)
    fn(2.0, x, y, n)
    np.testing.assert_allclose(y, 2.0 * np.arange(n) + 1.0)


def test_sourcegen_prange_emits_threads_for_large_independent():
    from celeris.backends.sourcegen import emit_cpp
    from celeris.types import prange
    def psaxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in prange(n):
            y[i] = a * x[i] + y[i]
    src = emit_cpp(parse_function(psaxpy))
    assert "std::thread" in src and "join" in src


def test_sourcegen_prange_reduction_stays_serial():
    from celeris.backends.sourcegen import emit_cpp
    from celeris.types import prange
    def psum(x: F64Array, n: int) -> float:
        acc = 0.0
        for i in prange(n):
            acc = acc + x[i]
        return acc
    src = emit_cpp(parse_function(psum))
    assert "std::thread" not in src          # reduction -> not parallelized
    fn = SourceGenBackend().compile(parse_function(psum))
    x = np.arange(100, dtype=np.float64)
    assert abs(fn(x, 100) - x.sum()) < 1e-9  # serial fallback correct


def test_sourcegen_2d_contiguous():
    from celeris.types import F64Array2D, F64Array
    def rowsum(a: F64Array2D, y: F64Array, m: int, k: int) -> None:
        for i in range(m):
            acc = 0.0
            for j in range(k):
                acc = acc + a[i, j]
            y[i] = acc
    fn = SourceGenBackend().compile(parse_function(rowsum))
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    y = np.zeros(3, dtype=np.float64); fn(a, y, 3, 4)
    np.testing.assert_allclose(y, a.sum(axis=1))

def test_sourcegen_2d_transposed_view_strides():
    from celeris.types import F64Array2D, F64Array
    def rowsum(a: F64Array2D, y: F64Array, m: int, k: int) -> None:
        for i in range(m):
            acc = 0.0
            for j in range(k):
                acc = acc + a[i, j]
            y[i] = acc
    fn = SourceGenBackend().compile(parse_function(rowsum))
    base = np.arange(12, dtype=np.float64).reshape(4, 3)
    a = base.T                     # non-contiguous view (shape 3x4)
    assert not a.flags["C_CONTIGUOUS"]
    y = np.zeros(3, dtype=np.float64); fn(a, y, 3, 4)
    np.testing.assert_allclose(y, a.sum(axis=1))   # general strides => correct on a view
