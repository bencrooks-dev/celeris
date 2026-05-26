import pytest, numpy as np

pytestmark = pytest.mark.needs_llvmlite

pytest.importorskip("llvmlite")
from celeris.backends.llvm import LLVMBackend
from celeris.parser import parse_function
from celeris.types import F64Array

def test_available():
    assert LLVMBackend().available() is True

def test_scalar():
    def f(a: float, b: float) -> float:
        return a*b + 2.0
    assert abs(LLVMBackend().compile(parse_function(f))(3.0, 4.0) - 14.0) < 1e-9

def test_saxpy():
    def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in range(n):
            y[i] = a*x[i] + y[i]
    fn = LLVMBackend().compile(parse_function(saxpy))
    x = np.arange(64, dtype=np.float64); y = np.zeros(64, dtype=np.float64)
    fn(2.0, x, y, 64)
    np.testing.assert_allclose(y, 2.0*np.arange(64))

def test_reduction_sum():
    def s(x: F64Array, n: int) -> float:
        acc = 0.0
        for i in range(n):
            acc = acc + x[i]
        return acc
    fn = LLVMBackend().compile(parse_function(s))
    x = np.arange(100, dtype=np.float64)
    assert abs(fn(x, 100) - x.sum()) < 1e-9

def test_while_and_if_matches_python():
    def f(n: int) -> int:
        s = 0; i = 0
        while i < n:
            if i % 2 == 0:
                s = s + i
            i = i + 1
        return s
    fn = LLVMBackend().compile(parse_function(f))
    assert fn(10) == sum(i for i in range(10) if i % 2 == 0)

def test_floordiv_negative_matches_python():
    def f(a: int, b: int) -> int:
        return a // b
    fn = LLVMBackend().compile(parse_function(f))
    assert fn(-7, 2) == -7 // 2 == -4


def test_llvm_int_const_returned_from_float_fn():
    def f(a: float) -> float:
        return 1
    fn = LLVMBackend().compile(parse_function(f))
    assert abs(fn(2.0) - 1.0) < 1e-9


def test_llvm_cmp_assigned_then_returned():
    def f(a: int, b: int) -> int:
        c = a < b
        return c
    fn = LLVMBackend().compile(parse_function(f))
    assert fn(1, 2) == 1 and fn(2, 1) == 0
