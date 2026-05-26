import numpy as np
from celeris import fast_runtime
from celeris.types import F64Array

def test_compiles_and_runs():
    @fast_runtime
    def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in range(n):
            y[i] = a * x[i] + y[i]
    x = np.arange(100, dtype=np.float64); y = np.zeros(100, dtype=np.float64)
    saxpy(2.0, x, y, 100)
    np.testing.assert_allclose(y, 2.0 * np.arange(100))

def test_falls_back_on_unsupported():
    @fast_runtime
    def uses_dict(n: int) -> int:
        d = {}                 # unsupported -> must fall back to Python and still work
        for i in range(n):
            d[i] = i
        return len(d)
    assert uses_dict(3) == 3

def test_cache_reuses_compiled():
    @fast_runtime
    def f(a: float) -> float:
        return a + 1.0
    f(1.0); f(2.0)
    assert len(f.__celeris_cache__) >= 1

def test_debug_emits_ir(capsys):
    @fast_runtime(debug=True)
    def f(a: float) -> float:
        return a * 2.0
    f(1.0)
    out = capsys.readouterr().out
    assert "IR" in out

def test_explicit_backend_interpreter():
    @fast_runtime(backend="interpreter")
    def f(a: float, b: float) -> float:
        return a * b + 1.0
    assert abs(f(3.0, 4.0) - 13.0) < 1e-9

def test_wrapped_accessible():
    @fast_runtime
    def f(a: float) -> float:
        return a
    assert f.__celeris_wrapped__.__name__ == "f"

def test_result_matches_python_for_reduction():
    @fast_runtime
    def s(x: F64Array, n: int) -> float:
        acc = 0.0
        for i in range(n):
            acc = acc + x[i]
        return acc
    x = np.arange(50, dtype=np.float64)
    assert abs(s(x, 50) - x.sum()) < 1e-9
