import pytest
import numpy as np
from celeris.backends.kernels import KernelBackend, REGISTRY
from celeris.parser import parse_function
from celeris.types import F64Array
from conftest import needs_clang
pytestmark = needs_clang

def _saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        y[i] = a * x[i] + y[i]

def _dot(x: F64Array, y: F64Array, n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + x[i] * y[i]
    return acc

def _sum(x: F64Array, n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + x[i]
    return acc

def test_registry_has_core_shapes():
    assert {"saxpy", "dot", "sum"} <= set(REGISTRY.keys())

def test_saxpy_matches_and_runs():
    be = KernelBackend()
    ir = parse_function(_saxpy)
    assert be.matches(ir) is True
    fn = be.compile(ir)
    x = np.arange(8, dtype=np.float64); y = np.ones(8, dtype=np.float64)
    fn(2.0, x, y, 8)
    np.testing.assert_allclose(y, 2.0*np.arange(8)+1.0)

def test_dot_matches_and_runs():
    be = KernelBackend()
    ir = parse_function(_dot)
    assert be.matches(ir) is True
    fn = be.compile(ir)
    x = np.arange(10, dtype=np.float64); y = np.arange(10, dtype=np.float64)
    assert abs(fn(x, y, 10) - float(x @ y)) < 1e-9

def test_sum_matches_and_runs():
    be = KernelBackend()
    ir = parse_function(_sum)
    assert be.matches(ir) is True
    fn = be.compile(ir)
    x = np.arange(20, dtype=np.float64)
    assert abs(fn(x, 20) - x.sum()) < 1e-9

def test_unknown_shape_returns_bool_and_compile_raises():
    from celeris.errors import CompileError
    def weird(x: F64Array, n: int) -> float:
        acc = 0.0
        for i in range(n):
            acc = acc + x[i]*x[i]*x[i] + 1.0
        return acc
    be = KernelBackend()
    ir = parse_function(weird)
    assert isinstance(be.matches(ir), bool)
    if not be.matches(ir):
        with pytest.raises(CompileError):
            be.compile(ir)
