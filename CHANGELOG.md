# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet.

### Changed
- Nothing yet.

### Fixed
- Nothing yet.

## [0.5.0] - 2026-05-27

### Added
- 2-D arrays — the first step of the tensor memory model. New parameter markers
  `F64Array2D` / `F32Array2D` / `I64Array2D` / `I32Array2D` annotate a 2-D array; each lowers to
  the IR type `{"ptr": <elem>, "ndim": 2}` (1-D arrays stay `{"ptr": <elem>}`, an implicit
  `ndim` of `1`, so existing kernels are unchanged). Element access uses `a[i, j]`, which the
  parser lowers to a new `index` IR node carrying an `"indices"` list (one expression per
  dimension) instead of the 1-D single `"index"`; the verifier requires the index arity to equal
  the array's `ndim` and each index to be integer-typed.
- General strides — 2-D access is correct for **any** memory layout, not just contiguous arrays.
  Native backends (C++ source-gen, llvm) receive each 2-D array as a data pointer **plus one
  `int64` stride per dimension, measured in elements**, and lower `a[i, j]` to a flat offset
  `data[Σ idx_d · stride_d]`. Because the strides are passed at call time from the NumPy array's
  own `strides`, non-contiguous NumPy views — slices of a larger buffer, and `.T` transposes —
  compute correctly without a copy. The pure-Python interpreter indexes the real NumPy array
  natively (`arr[(i, j)]`), so it remains the strided oracle the differential harness checks the
  native backends against, including a transposed-view case.

### Changed
- The golden-kernel tier declines any IR that contains an N-D `index` node (no 2-D golden
  templates in this release), so 2-D kernels route to the source-gen or llvm backends.

### Notes
- Out of scope for v0.5.0 (tracked in the roadmap): slicing and row-views (`a[i, :]`, `a[1:5]`),
  broadcasting, and arrays of rank ≥ 3 are still rejected and fall back to pure Python. A 2-D
  access must use exactly two integer indices.

## [0.4.0] - 2026-05-27

### Added
- `prange` parallel loops. Writing `for i in prange(n):` parses identically to `range` but
  marks the `for` IR node `parallel: true` — a hint, not a guarantee. The C++ source-gen
  backend executes a `prange` loop with `std::thread` chunking **only** when it can prove the
  loop is independent: unit positive step (`step == 1`), no `return` in the body, no scalar
  writes (so no reductions or loop-carried temporaries), and every array *write* indexed at
  exactly `i`. When all hold and the trip count is at least 4096 iterations, the loop body is
  split into contiguous chunks across up to 8 worker threads (`std::thread::hardware_concurrency`,
  clamped); below 4096 iterations it runs the same body serially to avoid thread overhead. Any
  loop that fails the independence predicate — reductions, offset writes (`y[i+1] = …`),
  non-unit step, or a `return` body — falls back to a normal serial loop, so it is correct by
  construction. The pure-Python interpreter runs every `prange` loop serially and is the
  oracle the differential harness checks the threaded source-gen output against. The
  golden-kernel and llvm backends **decline** parallel loops (kernels' `matches` returns
  `False`; llvm raises `CompileError`), so a `prange` kernel routes to the threaded source-gen
  path. Loop fusion only merges two loops when their `parallel` flags match.

### Changed
- The C++ source-gen prelude now includes `<thread>`, `<vector>`, and `<algorithm>`, and the
  runtime `clang++` invocation passes `-pthread`, to support the threaded `prange` codegen.

## [0.3.0] - 2026-05-27

### Changed
- Loop fusion now handles constant affine offsets (`a[i ± c]`, `c` an integer literal) on
  written arrays, generalizing the v0.2.0 "subscript must be exactly the loop variable" rule
  into a provably-safe superset. For two adjacent unit-step loops (`step == 1`), each written
  array's cross-loop access pairs — L1 at offset `cx`, L2 at offset `cy`, with at least one
  write among them — must satisfy `cy ≤ cx`; this is exactly the condition under which the
  fused interleaving preserves the unfused flow/anti/output dependence order. So a producer at
  `t[i+1]` followed by a consumer at `t[i]` fuses, while a forward-read dependence (`t[i]` then
  `t[i+1]`) is declined. The pass stays conservative: variable offsets (`a[i+k]`), non-unit
  step (and the strict exactly-`i` fallback it triggers), and multi-array broadcasting remain
  out of scope and fall back to leaving the loops unfused. Read-only arrays are still
  unrestricted, and a fused loop is still a normal `for` IR node — no backend changes — with
  the differential harness cross-checking an in-bounds affine stencil chain against the
  pure-Python oracle.

## [0.2.0] - 2026-05-26

### Added
- Loop-fusion optimization pass (`fuse_loops`): adjacent `for` loops over the same iteration
  space (identical loop var, start, stop, step) fuse into a single loop body — the "one pass,
  no temporary" win — applied left-to-right to a fixpoint. Fusion runs only when a
  conservative, provably-safe legality predicate holds (no `return` in either body; every
  subscript of any array written in either body is exactly the loop variable; no shared scalar
  dependence between the two bodies). Read-only arrays may use any index. Anything else is left
  untouched (correct, just unfused). Wired into `optimize()` as `fold → fuse → DCE`; backends
  need no changes since a fused loop is a normal `for` IR node, and the differential harness
  cross-checks the fused output against the pure-Python oracle.

## [0.1.0] - 2026-05-26

### Added
- Initial public release: the `@fast_runtime` JIT decorator for a statically-typed numeric
  subset of Python.
- Pure-Python frontend: AST parser + subset validator, typed JSON IR with constructors and
  round-trip, an independent IR verifier (the trust boundary for native backends), and
  constant-folding + dead-code-elimination passes.
- Type system with scalar (`i32`/`i64`/`f32`/`f64`) and 1-D array markers
  (`F64Array`/`F32Array`/`I64Array`/`I32Array`) plus numeric promotion rules.
- Tiered backend dispatch: golden-kernel registry (hand-tuned C++ templates matched by IR
  fingerprint), optional `llvmlite` JIT, C++ source-gen via runtime `clang++`, and an
  always-available pure-Python interpreter reference, with graceful fallback to the original
  Python function.
- Standalone C++ core (C ABI + golden saxpy + LLVM lowering seam stub) and a `pybind11`
  production binding, built via CMake.
- Differential correctness harness cross-checking every available backend against pure Python,
  a benchmark suite, runnable examples, and GitHub Actions CI.

[Unreleased]: https://github.com/bencrooks-dev/celeris/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/bencrooks-dev/celeris/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/bencrooks-dev/celeris/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bencrooks-dev/celeris/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bencrooks-dev/celeris/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bencrooks-dev/celeris/releases/tag/v0.1.0
