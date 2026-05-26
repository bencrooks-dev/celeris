# celeris

[![CI](https://img.shields.io/badge/CI-pending-lightgrey.svg)](https://github.com/bencrooks-dev/celeris/actions) <!-- CI badge placeholder; wired up in Wave 8 -->

**A JIT compiler for a statically-typed numeric subset of Python, exposed via a `@fast_runtime` decorator.**

celeris is a readable, deliberately restricted re-implementation of the Numba architecture:
a JIT compiler for a statically-typed numeric subset of Python, exposed via a `@fast_runtime`
decorator. It is **not** a full Python compiler. It compiles tight numeric kernels — the kind
that show up in inner loops — and transparently falls back to the original Python function for
anything outside its supported subset.

The goal is pedagogical clarity and honest engineering: every layer (frontend → IR → backends
→ bindings) is small enough to read in one sitting, and every backend is cross-checked against a
pure-Python reference interpreter by a differential test harness.

## Quickstart

```python
import numpy as np
from celeris import fast_runtime
from celeris.types import F64Array


@fast_runtime
def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        y[i] = a * x[i] + y[i]


x = np.arange(1_000_000, dtype=np.float64)
y = np.zeros(1_000_000, dtype=np.float64)
saxpy(2.0, x, y, x.size)          # compiled on first call, cached thereafter
np.testing.assert_allclose(y, 2.0 * np.arange(1_000_000))
```

If a function uses anything outside the supported subset, `@fast_runtime` does not raise — it
quietly returns the original Python function so your code keeps working.

## Supported subset (v0.1)

Inside a `@fast_runtime`-decorated region, celeris supports:

- **Numeric scalars** — `i32`, `i64`, `f32`, `f64` (Python `int` → `i64`, `float` → `f64`).
- **Typed 1-D arrays** — `F64Array`, `F32Array`, `I64Array`, `I32Array` (NumPy or any
  buffer supporting `__getitem__`/`__setitem__`).
- **Control flow** — `for i in range(...)`, `while`, `if`/`else`.
- **Arithmetic** — `+ - * / // % **`, comparisons, boolean ops, and a small intrinsic
  whitelist (`sqrt`, `exp`, `log`, `sin`, `cos`, `fabs`, `floor`, `fmax`, `fmin`, `len`).
- **Simple reductions** — accumulator loops (sum, dot, etc.).

**Not supported in compiled regions** (these trigger fallback to pure Python): classes,
dicts/sets, exceptions, generators, closures over non-locals, dynamic/duck-typed calls,
recursion, and multi-dimensional indexing. Mandatory type annotations are required on all
parameters and the return type.

## Install

celeris is pure-Python at its core; the native backends are optional.

| Command | What you get |
| --- | --- |
| `pip install -e .` | Core: parser, IR, verifier, passes, pure-Python interpreter backend. |
| `pip install -e .[llvm]` | Adds the optional `llvmlite` in-process JIT backend. |
| `pip install -e .[native]` | Adds `pybind11` for the CMake-built production `celeris_native` module. |
| `pip install -e .[dev]` | Test + lint toolchain (`pytest`, `numpy`, `ruff`). |

The C++ source-gen and golden-kernel backends require a `clang++` on your `PATH`; they are
detected at runtime and skipped when absent.

## Backend tiers

`@fast_runtime` dispatches through a priority chain, taking the first backend that can compile
the kernel and falling back gracefully:

1. **Golden kernels** — a registry of hand-tuned C++ templates for recognized shapes (saxpy,
   dot, sum, scale, fused elementwise), matched by an IR fingerprint. The fast path.
2. **llvmlite** *(optional)* — in-process LLVM JIT for general kernels.
3. **C++ source-gen** — emit C++ from the IR, compile with `clang++ -O3`, load via `ctypes`.
4. **Python fallback** — the pure-Python interpreter (reference) and, ultimately, the original
   undecorated function. Always available.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layered architecture and tiered dispatch.
- [docs/IR_SPEC.md](docs/IR_SPEC.md) — the JSON IR schema.
- [docs/ROADMAP.md](docs/ROADMAP.md) — v0.1 → v1.0 plan.
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, TDD workflow, native build.

## License

Apache-2.0. See [LICENSE](LICENSE).
