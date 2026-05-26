"""Honest benchmark: pure-Python loop vs NumPy vs celeris @fast_runtime.

No fabricated speedups, and no rigged timing: input/output buffers are allocated
OUTSIDE the timed region (see ``_median``), so allocation cost is never charged to
the measured op.

celeris fuses a multiply-add expression into ONE pass with NO temporary, whereas
NumPy's ``a*x + y`` must allocate an intermediate array for ``a*x``. So celeris can
legitimately win even on a single fused expression. Where a NumPy op maps to one
already-optimized primitive (a plain copy, a single BLAS call), celeris will not
beat it. The genuine wins are therefore (a) huge vs pure-Python loops, and (b) loop
fusion on multi-op expressions where NumPy materializes intermediate temporaries.
"""
import statistics
import time

import numpy as np

from celeris import fast_runtime
from celeris.types import F64Array


@fast_runtime
def cel_saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        y[i] = a * x[i] + y[i]


@fast_runtime
def cel_fused(a: float, b: float, c: float, x: F64Array, y: F64Array,
              z: F64Array, d: F64Array, n: int) -> None:
    for i in range(n):
        d[i] = a * x[i] + b * y[i] + c * z[i]


def _py_saxpy(a, x, y, n):
    for i in range(n):
        y[i] = a * x[i] + y[i]


def _median(prepare, reps):
    """prepare() -> a zero-arg callable whose inputs are already allocated;
    only the returned callable is timed (allocation happens in prepare())."""
    prepare()()  # warm-up (also triggers JIT compile)
    ts = []
    for _ in range(reps):
        call = prepare()                       # allocate inputs OUTSIDE the timer
        t0 = time.perf_counter(); call(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def run(n: int = 1_000_000, reps: int = 7) -> None:
    a, b, c = 2.5, 1.5, 0.5
    x = np.random.rand(n); y = np.random.rand(n)
    print(f"celeris benchmark — N={n:,}, reps={reps}\n")

    # saxpy: single fused multiply-add
    if n <= 20000:
        xl = list(x)
        t_py = _median(lambda: (lambda yy=list(y): _py_saxpy(a, xl, yy, n)), max(reps, 1))
    else:
        t_py = float("nan")  # pure-Python loop at 1e6 is too slow to bother timing fully
    t_np = _median(lambda: (lambda yy=y.copy(): np.add(np.multiply(a, x), yy, out=yy)), reps)
    t_cel = _median(lambda: (lambda yy=y.copy(): cel_saxpy(a, x, yy, n)), reps)
    print("saxpy   y = a*x + y    (fused multiply-add)")
    if t_py == t_py:  # not nan
        print(f"  pure Python loop : {t_py*1e3:9.3f} ms  ({t_py/t_cel:6.1f}x slower than celeris)")
    print(f"  NumPy            : {t_np*1e3:9.3f} ms")
    print(f"  celeris          : {t_cel*1e3:9.3f} ms   (vs NumPy: {t_np/t_cel:.2f}x)\n")

    # fused: NumPy materializes temporaries; celeris fuses one pass
    x1, x2, x3 = np.random.rand(n), np.random.rand(n), np.random.rand(n)
    d = np.random.rand(n)
    t_np2 = _median(lambda: (lambda dd=d.copy(): np.add(
        np.add(np.multiply(a, x1), np.multiply(b, x2)),
        np.multiply(c, x3), out=dd)), reps)
    t_cel2 = _median(lambda: (lambda dd=d.copy(): cel_fused(a, b, c, x1, x2, x3, dd, n)), reps)
    print("fused   d = a*x + b*y + c*z   (NumPy makes temporaries; celeris fuses)")
    print(f"  NumPy            : {t_np2*1e3:9.3f} ms")
    print(f"  celeris          : {t_cel2*1e3:9.3f} ms   (vs NumPy: {t_np2/t_cel2:.2f}x)\n")

    print("Interpretation: celeris fuses a*x + y into ONE pass with NO temporary, while\n"
          "NumPy must allocate an intermediate for a*x — so celeris can legitimately win\n"
          "even on this single expression. Where a NumPy op maps to one already-optimized\n"
          "primitive (a plain copy, a single BLAS call), celeris would not beat it. The\n"
          "clearest wins are vs pure-Python loops and the multi-op fused case above,\n"
          "where NumPy materializes several intermediate temporaries and celeris does not.")


if __name__ == "__main__":
    run()
