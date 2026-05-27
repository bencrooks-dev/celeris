# Roadmap

celeris ships in small, independently useful increments. Each milestone is fully tested and
keeps the existing public API (`@fast_runtime`, `celeris.types`) stable.

| Version | Theme | Scope |
| --- | --- | --- |
| **v0.1** | Frontend + reference | AST parser & subset validator, typed JSON IR with verifier and constant-fold/DCE passes, pure-Python interpreter backend, `@fast_runtime` with cache + graceful Python fallback. The whole pipeline runs end to end with no native dependencies. |
| **v0.2** | Typed IR + C++ source-gen + loop fusion | IR → C++ emitter, runtime `clang++ -O3` compile, `ctypes` load and marshaling, on-disk per-kernel cache (`~/.celeris_cache/`). First real speedups. **Shipped (v0.2.0):** the provably-safe loop-fusion pass — adjacent elementwise loops over the same iteration space fuse into one body (`fold → fuse → DCE`), the "one pass, no temporary" win. |
| **v0.3** | pybind11 production binding | Standalone C++ core (C ABI + golden saxpy + LLVM seam stub) and the `pybind11` module built via CMake; demonstrates the production binding path. |
| **v0.4** | LLVM ORC backend | Optional `llvmlite` in-process JIT: structured-control-flow → SSA lowering with PHI nodes for loop induction variables, opt-level 2/3 pipeline, ORC/MCJIT execution. Exercised by a dedicated CI job. |
| **v0.5** | Tensor memory model | Multi-dimensional arrays, strides, and basic slicing; promotes the array markers beyond 1-D and lays groundwork for tiling passes. |
| **v1.0** | Stable kernel compiler | Stabilized IR schema and public API, the golden-kernel registry as a documented extension point, fusion extensions (affine-offset dependence analysis) and loop tiling/blocking, and a published, semver-guaranteed release. |

## Explicitly out of scope (for now)

These are tracked deliberately as *not* part of the v0.1→v1.0 line above, to keep the project
honest about what it is:

- Type inference that drops mandatory annotations.
- **Fusion across affine offsets.** The shipped v0.2.0 fusion pass requires every subscript of
  any written array to be exactly the loop variable; loops where a written array is touched at
  an offset (e.g. `a[i+1]`) are intentionally *declined*. Affine-offset dependence analysis is
  a v1.0 fusion extension.
- Loop tiling / blocking (v1.0 optimization pass).
- `prange` / multi-threading.
- Recursion and a general call graph between compiled kernels.
- A persistent on-disk kernel cache shared across processes.
- GPU backends.
- PyPI publishing (installed from source / GitHub during this line of work).

If and when these land, they will get their own roadmap entry; until then, anything outside the
[supported subset](../README.md#supported-subset-v01) falls back to pure Python.
