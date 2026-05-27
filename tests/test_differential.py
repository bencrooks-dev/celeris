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
from celeris.types import F64Array


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
