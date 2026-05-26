# Contributing to celeris

Thanks for your interest. celeris is built to be read, so contributions that keep each layer
small and obvious are especially welcome.

## Dev setup

```bash
python -m venv .venv
.venv/bin/pip install -e .[dev]
```

This installs the core package plus the test and lint toolchain (`pytest`, `numpy`, `ruff`).
To work on the optional backends, add the relevant extra:

```bash
.venv/bin/pip install -e .[dev,llvm]      # llvmlite JIT backend
.venv/bin/pip install -e .[dev,native]    # pybind11 production module
```

The C++ source-gen and golden-kernel backends additionally require a `clang++` on your `PATH`.

## TDD expectation

Every task is test-first, no exceptions. The workflow for any change is:

1. Write the failing test that pins the behavior you want.
2. Run it and confirm it fails for the right reason.
3. Implement the smallest change that makes it pass.
4. Run the suite and confirm green.
5. Self-review the diff, then commit.

New behavior without a test that would fail in its absence will not be merged.

## Running the tests

```bash
.venv/bin/python -m pytest -q
```

Tests that need optional capabilities skip automatically when those capabilities are absent,
driven by the markers in `tests/conftest.py`:

- `needs_clang` — requires a `clang++` on `PATH`.
- `needs_llvmlite` — requires the optional `llvmlite` package.
- `needs_native` — requires the CMake-built `celeris_native` module.

Select or exclude them with `-m`, e.g. `pytest -m "not needs_llvmlite and not needs_native"`.

## Differential test harness

The correctness backbone of celeris is `tests/test_differential.py`. It parametrizes a set of
representative kernels (saxpy, dot, sum, scale, a fused 3-term elementwise, a `while` loop, an
`if`-branch kernel), then runs each one through pure Python, the reference interpreter, and
every *available* native backend, asserting they all agree (within `1e-9` for floats, exact for
ints). Run just the harness with:

```bash
.venv/bin/python -m pytest tests/test_differential.py -v
```

When you add a backend or a golden kernel, add the corresponding case here — this is the oracle
that de-risks the more aggressive backends.

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
```

CI runs `ruff check`; keep it clean.

## Building the native module

The standalone C++ core and its pybind11 binding build via CMake:

```bash
pip install cmake pybind11
cmake -S . -B build
cmake --build build
```

This produces the `celeris_native` module (and the `celeris_core` static library). The optional
C++-side LLVM lowering seam is gated behind the `CELERIS_LLVM` CMake cache variable. Once built,
the `needs_native` tests run instead of skipping.

## Reporting issues

Open an issue with a minimal reproduction. For a compiled-kernel bug, the most useful thing you
can include is the IR — call your decorated function with `@fast_runtime(debug=True)` and paste
the dumped IR.
