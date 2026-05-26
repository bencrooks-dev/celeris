"""reduction example — accumulator loops for sum and dot product.

celeris compiles these reduction kernels on the first call and transparently
falls back to the original Python function if compilation is unsupported on
this machine; the numeric result is identical either way, which is what we
verify against NumPy below.

Run it directly:  python examples/reduction.py
"""
import numpy as np

from celeris import fast_runtime
from celeris.types import F64Array


@fast_runtime
def array_sum(x: F64Array, n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + x[i]
    return acc


@fast_runtime
def dot(x: F64Array, y: F64Array, n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + x[i] * y[i]
    return acc


def main():
    n = 500
    x = np.linspace(-1.0, 1.0, n)
    y = np.cos(np.linspace(0.0, 3.0, n))

    assert abs(array_sum(x, n) - float(x.sum())) < 1e-9
    assert abs(dot(x, y, n) - float(x @ y)) < 1e-9

    print("reduction OK")


if __name__ == "__main__":
    main()
