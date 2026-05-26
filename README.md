# celeris

[![CI](https://github.com/bencrooks-dev/celeris/actions/workflows/ci.yml/badge.svg)](https://github.com/bencrooks-dev/celeris/actions/workflows/ci.yml)

**A JIT compiler for a statically-typed numeric subset of Python, exposed via a `@fast_runtime` decorator. It is not a full Python compiler.**

celeris is a readable, deliberately restricted re-implementation of the Numba architecture:
a JIT compiler for a statically-typed numeric subset of Python, exposed via a `@fast_runtime`
decorator. To be clear up front, it is not a full Python compiler. It compiles tight numeric
kernels — the kind that show up in inner loops — and transparently falls back to the original
Python function for anything outside its supported subset.

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
- **Arithmetic** — `+ - * / // % **`, comparisons, and boolean ops. Division follows
  Python semantics: `/` is true division and always yields `f64`; `//` is floor division;
  `%` is floored modulo (the sign follows the divisor, matching CPython, not C).
- **Intrinsics** — a small whitelist: `sqrt`, `exp`, `log`, `sin`, `cos`, `fabs`, `floor`,
  `fmax`, `fmin`, and `len`.
- **Simple reductions** — accumulator loops (sum, dot, etc.).

**Not supported** (any of these triggers transparent fallback to pure Python): classes,
dicts/sets, exceptions, generators, closures over non-locals, dynamic or duck-typed calls
(only direct calls to the intrinsic whitelist and `range` are allowed), recursion,
multi-dimensional indexing (`a[i, j]`), and slicing (`a[i:j]`). Type annotations are
mandatory on every parameter and on the return type.

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

The native `celeris_native` module is built separately, via CMake + pybind11:

```bash
pip install -e .[native]      # pulls in pybind11
cmake -S . -B build
cmake --build build           # produces celeris_native + the celeris_core static lib
```

Once built, the `needs_native` tests exercise it instead of skipping.

## Backend tiers

`@fast_runtime` dispatches through a priority chain, taking the first backend that can compile
the kernel and falling back gracefully:

1. **Golden kernels** — a registry of hand-tuned C++ templates for recognized shapes (saxpy,
   scale, sum, dot), matched by an IR fingerprint. These emit `__restrict__` pointer params,
   unlocking vectorization the generic path can't assume. The fast path.
2. **llvmlite** *(optional)* — in-process LLVM JIT for general kernels.
3. **C++ source-gen** — emit C++ from the IR, compile with `clang++ -O3`, load via `ctypes`.
4. **Interpreter (reference)** — a pure-Python tree-walker over the IR. It is the correctness
   oracle the differential harness checks every other backend against, and the always-available
   compiled path.
5. **Python fallback** — if no backend can compile the kernel, `@fast_runtime` returns the
   original undecorated function unchanged.

A standalone C++ core (`celeris_native`, built from `src/celeris/_native` via pybind11/CMake)
is also available for embedding the IR pipeline in C++ hosts.

**A note on performance, honestly:** a single-op kernel like saxpy is *memory-bound* — it
moves more bytes than it does arithmetic — so a compiled celeris kernel lands at roughly
NumPy-equivalent throughput, not dramatically faster. The architecture's intended win is
**fusion**: collapsing a chain of array operations into one pass over memory, avoiding the
temporaries NumPy allocates between each operation. (Loop fusion is a v1.0 roadmap item, not
yet shipped — see [docs/ROADMAP.md](docs/ROADMAP.md).) We ship no fabricated benchmark
numbers; run `benchmarks/benchmark.py` on your own hardware.

## Comparison to Numba

celeris is best understood as a *readable, deliberately restricted re-implementation of the
Numba architecture*, built for clarity rather than coverage. Numba is a mature, production
JIT supporting a far larger Python subset, NumPy semantics, and CUDA. celeris is not a
competitor: it trades breadth for a codebase small enough to read end-to-end, with each
backend cross-checked against a reference interpreter. Reach for Numba in production; reach
for celeris to understand how a tiered numeric JIT actually works.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layered architecture and tiered dispatch.
- [docs/IR_SPEC.md](docs/IR_SPEC.md) — the JSON IR schema.
- [docs/ROADMAP.md](docs/ROADMAP.md) — v0.1 → v1.0 plan.
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, TDD workflow, native build, and the
  **differential test harness** (`tests/test_differential.py`), which is how we keep every
  backend honest against the reference interpreter. Read it before sending a patch.

## License

celeris is licensed under Apache-2.0; see [LICENSE](LICENSE). It vendors one third-party
header (nlohmann/json, MIT); attributions are recorded in [NOTICE](NOTICE).
