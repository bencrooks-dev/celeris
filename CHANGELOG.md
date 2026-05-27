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

[Unreleased]: https://github.com/bencrooks-dev/celeris/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bencrooks-dev/celeris/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bencrooks-dev/celeris/releases/tag/v0.1.0
