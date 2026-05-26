"""saxpy example — y[i] = a*x[i] + y[i].

celeris compiles this kernel on the first call (golden-kernel / llvmlite /
source-gen, whichever is available) and transparently falls back to the
original Python function if the kernel is unsupported on this machine. Either
way the numeric result is the same, which is what we verify below.

Run it directly:  python examples/saxpy.py
"""
import numpy as np

from celeris import fast_runtime
from celeris.types import F64Array


@fast_runtime
def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        y[i] = a * x[i] + y[i]


def main():
    n = 1000
    a = 2.5
    x = np.arange(n, dtype=np.float64)
    y = np.linspace(1.0, 2.0, n)

    # Reference (computed before saxpy mutates y in place).
    expected = a * x + y

    saxpy(a, x, y, n)

    np.testing.assert_allclose(y, expected)
    print("saxpy OK")


if __name__ == "__main__":
    main()
