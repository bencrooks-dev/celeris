# Roadmap

celeris ships in small, independently useful increments. Each milestone is fully tested and
keeps the existing public API (`@fast_runtime`, `celeris.types`) stable.

| Version | Theme | Scope |
| --- | --- | --- |
| **v0.1** | Frontend + reference | AST parser & subset validator, typed JSON IR with verifier and constant-fold/DCE passes, pure-Python interpreter backend, `@fast_runtime` with cache + graceful Python fallback. The whole pipeline runs end to end with no native dependencies. |
| **v0.2** | Typed IR + C++ source-gen + loop fusion | IR → C++ emitter, runtime `clang++ -O3` compile, `ctypes` load and marshaling, on-disk per-kernel cache (`~/.celeris_cache/`). First real speedups. **Shipped (v0.2.0):** the provably-safe loop-fusion pass — adjacent elementwise loops over the same iteration space fuse into one body (`fold → fuse → DCE`), the "one pass, no temporary" win. |
| **v0.3** | Affine-offset fusion | **Shipped (v0.3.0):** the fusion legality check generalizes from "every written subscript is exactly the loop variable" to **constant affine offsets** (`a[i ± c]`, `c` an integer literal) via a provably-safe `cy ≤ cx` dependence test on unit-step loops — a strict superset of the v0.2.0 rule. A producer at `t[i+1]` feeding a consumer at `t[i]` now fuses; forward-read dependences, variable offsets, and non-unit step still decline. (The standalone C++ core and `pybind11` production binding, built via CMake, shipped earlier in the v0.1 line.) |
| **v0.4** | `prange` threaded parallelism | **Shipped (v0.4.0):** `for i in prange(n)` marks a loop `parallel` (a hint), and the C++ source-gen backend runs it with `std::thread` chunking when it can *prove* independence — unit positive step, no `return`, no scalar writes (so no reductions), every array write at exactly `i` — and the trip count is at least 4096 iterations; everything else falls back to serial, correct by construction. The interpreter and llvm backends run serial; the golden-kernel and llvm backends decline parallel loops so a `prange` kernel routes to the threaded source-gen path. Correctness is enforced by the differential harness (threaded output == serial interpreter oracle). (The original v0.4 "LLVM ORC backend" idea is moot — the optional `llvmlite` in-process JIT shipped back in the v0.1 line.) |
| **v0.5** | Tensor memory model | **Shipped (v0.5.0):** 2-D arrays with `a[i, j]` element indexing and **general strides**. New markers `F64Array2D` / `F32Array2D` / `I64Array2D` / `I32Array2D` lower to `{"ptr": <elem>, "ndim": 2}`; `a[i, j]` becomes an `index` IR node with an `"indices"` list. The native backends pass each 2-D array as a data pointer plus one element-stride per dimension and lower an index to `data[Σ idx_d · stride_d]`, so non-contiguous NumPy views (slices, `.T` transposes) compute correctly without a copy; the interpreter indexes the NumPy array natively as the strided oracle. **Still future** (see below): slicing / row-views (`a[i, :]`, `a[1:5]`), broadcasting, arrays of rank ≥ 3, 2-D golden kernels, and the tiling passes this lays groundwork for. |
| **v1.0** | Stable kernel compiler | Stabilized IR schema and public API, the golden-kernel registry as a documented extension point, the remaining fusion extensions (variable-offset and non-unit-step dependence analysis) and loop tiling/blocking, and a published, semver-guaranteed release. |

## Explicitly out of scope (for now)

These are tracked deliberately as *not* part of the v0.1→v1.0 line above, to keep the project
honest about what it is:

- Type inference that drops mandatory annotations.
- **Variable-offset and non-unit-step fusion.** The shipped v0.3.0 fusion pass handles
  *constant* affine offsets on written arrays (`a[i ± c]`, `c` an integer literal) for unit-step
  loops via the `cy ≤ cx` dependence test. Variable offsets (`a[i+k]`, where `k` is not a
  literal), non-unit / negative step, and multi-array broadcasting are still intentionally
  *declined* (they fall back to leaving the loops unfused) and remain future fusion extensions.
- Loop tiling / blocking (future optimization pass).
- **Parallel reductions and richer parallel loops.** The shipped v0.4.0 `prange` threads only
  provably-independent, unit-step, elementwise loops via the source-gen backend. Parallel
  reductions (atomics / per-thread partials), offset-write and non-unit-step parallel loops, and
  parallelism inside the llvm backend are intentionally *declined* (they fall back to serial) and
  remain future extensions.
- **Slicing, broadcasting, and rank ≥ 3 arrays.** The shipped v0.5.0 tensor model supports 2-D
  arrays with `a[i, j]` element indexing (exactly `ndim` integer indices) and general strides.
  Slicing and row-views (`a[i, :]`, `a[1:5]`), broadcasting, arrays of rank ≥ 3, 2-D golden
  kernels, and 2-D parallelism in `prange` / the llvm backend are intentionally *declined* (they
  fall back to pure Python) and remain future extensions.
- OpenMP / GPU backends.
- Recursion and a general call graph between compiled kernels.
- A persistent on-disk kernel cache shared across processes.
- PyPI publishing (installed from source / GitHub during this line of work).

If and when these land, they will get their own roadmap entry; until then, anything outside the
[supported subset](../README.md#supported-subset-v01) falls back to pure Python.
