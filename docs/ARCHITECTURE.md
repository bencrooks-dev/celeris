# Architecture

celeris is a small, layered JIT pipeline. Each layer has one job and a narrow, JSON-shaped
contract with its neighbours, so any stage can be read, tested, or replaced in isolation.

## Layers

```
        Python source (a @fast_runtime-decorated function)
                              │
                              ▼
  ┌───────────────────────────────────────────────────────────┐
  │ FRONTEND                                                    │
  │   parser.py    AST → subset validation → typed IR           │
  │   types.py     annotation → IR type, promotion rules        │
  └───────────────────────────────────────────────────────────┘
                              │  typed IR (dict / JSON)
                              ▼
  ┌───────────────────────────────────────────────────────────┐
  │ IR + ANALYSIS                                               │
  │   ir.py        node constructors, schema, dumps/loads       │
  │   verifier.py  independent structural + type check          │
  │   passes.py    constant folding, loop fusion, dead-code elim │
  └───────────────────────────────────────────────────────────┘
                              │  verified, optimized IR
                              ▼
  ┌───────────────────────────────────────────────────────────┐
  │ BACKENDS (tiered dispatch — see below)                      │
  │   kernels.py     golden hand-tuned C++ templates            │
  │   llvm.py        optional llvmlite in-process JIT            │
  │   sourcegen.py   IR → C++ → clang -O3 → .so → ctypes         │
  │   interpreter.py pure-Python reference (always available)    │
  └───────────────────────────────────────────────────────────┘
                              │  callable (or Python fallback)
                              ▼
  ┌───────────────────────────────────────────────────────────┐
  │ BINDINGS (standalone C++ core, built via CMake)             │
  │   _native/celeris_core.{hpp,cpp}  C ABI + LLVM seam stub     │
  │   _native/bindings.cpp            pybind11 production module  │
  └───────────────────────────────────────────────────────────┘
```

The frontend produces a typed IR (a plain JSON-serializable dict tree, see
[IR_SPEC.md](IR_SPEC.md)). The verifier re-checks that IR from scratch — it never trusts the
parser — because the IR is the trust boundary handed to native code. Optimization passes then
rewrite the IR before it reaches a backend.

## Optimizer pass order

`optimize()` runs three pure passes in a fixed order — `fold → fuse → DCE`:

1. **Constant folding** (`fold_constants`) — recursively collapse `binop`/`cmp` nodes whose
   operands are all `const` into a single `const`, simplifying the tree fusion then analyzes.
2. **Loop fusion** (`fuse_loops`) — merge adjacent `for` loops over the same iteration space
   into one loop body, applied left-to-right to a fixpoint (so a run of three fusable loops
   collapses to one). It is a no-op unless a conservative legality predicate proves the merge
   safe: **two adjacent `for` loops fuse iff they share an identical iteration space (var,
   start, stop, step), neither body contains a `return`, the written-array dependence test
   below holds, and there is no shared scalar dependence between the bodies**
   (`writes(L1) ∩ refs(L2) = ∅` and `writes(L2) ∩ refs(L1) = ∅`, excluding the loop var).
   The written-array test (`_can_fuse` condition 4) generalizes the v0.2.0 "exactly the loop
   variable" rule to **constant affine offsets** (`a[i ± c]`, `c` an integer literal): when the
   step is the unit positive literal (`step == 1`), every subscript of an array *written* in
   either body must parse to a constant offset of the loop var, and for each written array
   every cross-loop access pair — L1 at offset `cx`, L2 at offset `cy`, with at least one write
   among them — must satisfy `cy ≤ cx`. With unit stride `i ↦ i + c` is injective and
   contiguous, so each element is written once and `cy ≤ cx` is exactly the condition that the
   fused interleaving (L1's statement before L2's within each iteration `i`) preserves the
   unfused flow/anti/output dependence order. A producer at `t[i+1]` feeding a consumer at
   `t[i]` therefore fuses, whereas a forward-read dependence (`t[i]` then `t[i+1]`) is declined.
   A non-affine subscript of a written array (variable offset `a[i+k]`, `2*i`, `i%2`, …) is
   undecidable here and declines; a non-unit step abandons the contiguity premise and falls
   back to the strict exactly-`i` rule. Read-only arrays carry no cross-loop dependence and may
   use any index. Anything that fails the predicate is left untouched — correct, just unfused.
3. **Dead-code elimination** (`eliminate_dead_code`) — drop `assign` statements to a local
   variable whose name is never read anywhere in the (now-fused) kernel.

Each pass is pure (never mutates its input) and the whole pipeline is idempotent. A fused loop
is a normal `for` IR node, so backends need no changes; the differential harness cross-checks
the fused output against the pure-Python interpreter oracle.

## 2-D memory model (`ndim` + general strides)

As of v0.5.0 the array markers extend beyond 1-D: `F64Array2D` (and its `F32`/`I64`/`I32`
variants) annotate a 2-D array, lowering to the IR type `{"ptr": <elem>, "ndim": 2}`. The
dimensionality lives on the *type* — a bare `{"ptr": ...}` is an implicit `ndim` of `1`, so 1-D
kernels are byte-for-byte unchanged. An `a[i, j]` access parses to an `index` node carrying an
`"indices"` list (one expression per dimension) instead of the 1-D single `"index"`, and the
verifier requires the index arity to equal the array's `ndim` with each index integer-typed.

The model uses **general strides**, so a 2-D array is never assumed contiguous. The native
backends (source-gen and llvm) receive each `ndim ≥ 2` array as a **data pointer plus one
`int64` element-stride per dimension** across the C ABI; the source-gen signature for a 2-D `a`
is `(<elem>* a, int64_t a_s0, int64_t a_s1)`. A multi-dimensional index lowers to a single flat
offset — `data[Σ_d idx_d · stride_d]` — and the marshalling layer fills the stride arguments at
call time from the NumPy array's own `strides` (in elements, `arr.strides[d] // arr.itemsize`).
Because the strides are supplied per-call, non-contiguous NumPy views (a slice of a larger
buffer, or a `.T` transpose) compute correctly without forcing a copy. The pure-Python
interpreter sidesteps the offset math and indexes the real NumPy array natively
(`arr[tuple(indices)]`), so it stays the strided correctness oracle — the differential harness
checks the native backends against it, including a transposed-view case. The golden-kernel tier
has no 2-D templates in this release and declines any IR containing an N-D `index` node, so 2-D
kernels route to source-gen or llvm. Slicing/row-views, broadcasting, and rank ≥ 3 are out of
scope (see the roadmap).

## Tiered-dispatch model

The hand-tuned kernel tier is **not** the primary codegen path — that doesn't scale. It is a
*specialization fast-path*: a registry keyed by a normalized IR fingerprint, mapping recognized
shapes (saxpy, scale, sum, dot) onto hand-optimized C++ templates.
General source-gen is the fallback when no golden kernel matches; the pure-Python interpreter is
the portable reference; the original Python function is the ultimate safety net.

`@fast_runtime` walks a priority chain and takes the first tier that can compile the kernel:

1. **Golden kernels** (`kernels.py`) — fingerprint hit → hand-tuned template. Fastest, narrowest.
2. **llvmlite** (`llvm.py`, optional) — in-process LLVM JIT for general kernels.
3. **C++ source-gen** (`sourcegen.py`) — emit C++, compile with `clang++ -O3 -march=native`,
   load via `ctypes`. General, requires a compiler at runtime.
4. **Interpreter** (`interpreter.py`) — pure-Python tree-walker. Always available; the reference
   oracle for the differential harness.
5. **Python fallback** — on any `UnsupportedFeature` / `VerifyError` / `CompileError`, return the
   original undecorated function so user code never breaks.

This mirrors how MKL / oneDNN / cuDNN ship hand-tuned kernels for known shapes and fall back to a
general path otherwise, and it gives a clean A/B harness (golden vs generic vs reference).

## Parallel dispatch (`prange`)

`for i in prange(...)` parses identically to `range` but sets `parallel: true` on the `for` IR
node — a hint, not a guarantee. The tiered dispatch routes a parallel loop to the one backend
that can act on the hint safely:

- **Golden kernels** (`kernels.py`) **decline** parallel loops — `matches()` returns `False`
  via `ir.has_parallel_loop(...)` — so a `prange` kernel never matches a (serial) golden template.
- **llvm** (`llvm.py`) **declines** parallel loops — `compile()` raises `CompileError` when
  `has_parallel_loop(...)` is true, since that backend lowers serially.
- **C++ source-gen** (`sourcegen.py`) is therefore the backend that handles parallel loops, and
  it emits a `std::thread`-chunked loop **only** when the loop is provably independent. The
  interpreter reference (`interpreter.py`) ignores the flag and runs every loop serially, which
  makes it the serial oracle the differential harness checks the threaded output against.

The independence predicate lives in `sourcegen._is_parallelizable(node)` and requires **all** of:

1. the node is marked `parallel` (it came from `prange`);
2. unit positive step — `step` is the integer constant `1` (`celeris_floordiv`/negative/variable
   steps abandon the simple chunking premise);
3. no `return` anywhere in the body (`_has_return`);
4. no scalar writes in the body — via `passes._collect`, the set of written scalar names must be
   empty, which rules out reductions (`acc = acc + x[i]`) and loop-carried temporaries;
5. every array *write* is indexed at exactly the loop variable `i` (offset writes like
   `y[i+1] = …` are rejected); array *reads* may use any index.

When the predicate holds, `_emit_parallel_for` emits a guarded threaded loop: if the trip count
`hi - lo < 4096` it runs the body in a single serial loop (avoiding thread-spawn overhead);
otherwise it splits `[lo, hi)` into contiguous chunks across `std::thread::hardware_concurrency()`
workers, clamped to at most 8, each running a serial sub-loop, and joins them. Each worker thread
captures the array pointers and scalar parameters by value (`[=]`); because the predicate
guarantees disjoint writes (each element written once, at its own `i`) and no shared mutable
scalar, the chunks are race-free and the threaded result equals the serial one. Any loop that
fails the predicate falls through to the ordinary serial `for` emitter — correct, just not
threaded. The prelude gains `<thread>`, `<vector>`, `<algorithm>` and the `clang++` invocation
passes `-pthread`. Loop fusion (`_can_fuse`) only merges two loops when their `parallel` flags
match, so fusing never silently drops or invents the hint.

This is intentionally narrow: it threads only embarrassingly-parallel elementwise loops.
Parallel reductions (atomics / per-thread partials), offset and non-unit-step parallel loops,
parallelism in the llvm backend, and OpenMP/GPU backends are out of scope (see the roadmap).

## Two distinct native layers (do not conflate)

- **(A) Runtime JIT backends** — Python-driven, shell out to `clang` at runtime, load via
  `ctypes` (or run in-process via `llvmlite`). No CMake needed. This is the working speedup path.
- **(B) Standalone C++ core** — `_native/celeris_core.{hpp,cpp}` exposes the C ABI
  (`celeris_compile`/`free`/`strategy` + golden kernels + an LLVM lowering seam stub) and
  `bindings.cpp` is the pybind11 production module. Built via CMake. It demonstrates the
  production binding path and marks where C++-side LLVM lowering would go.

## Honest comparison

| Project | What it is | How celeris differs |
| --- | --- | --- |
| **Numba** | Production LLVM JIT for a large NumPy/Python subset, type inference, `prange`, CUDA. | celeris is a readable re-implementation of the same *architecture* at a fraction of the scope: mandatory annotations, 1-D arrays only, no GPU. It ships a deliberately minimal `prange` — the source-gen backend threads only provably-independent elementwise loops, with no parallel reductions. |
| **Cython** | Ahead-of-time Python→C transpiler with its own superset syntax and a build step. | celeris is a runtime JIT on plain annotated Python; no separate syntax, no AOT build (the optional CMake core aside). |
| **PyPy** | Whole-program tracing JIT for general Python via meta-tracing. | celeris compiles only explicitly-marked numeric kernels and falls back to CPython for everything else; far narrower, far simpler. |
| **Mojo** | A new language (Python superset) with its own MLIR-based compiler and runtime. | celeris is not a language; it is a library that compiles a subset of *existing* Python. |
| **MLIR** | A reusable compiler-IR infrastructure with dialects and lowering frameworks. | celeris's IR is a single, fixed, JSON dict schema — deliberately tiny and non-extensible, optimized for readability over generality. |
| **LLVM ORC** | LLVM's modern on-request JIT linking APIs. | celeris reaches LLVM only through the optional `llvmlite` backend; ORC-level concerns are abstracted away, and the C++ LLVM seam is a documented stub. |

celeris does not aim to beat any of these. It aims to be the smallest honest thing that
demonstrates the full Numba-style pipeline end to end.
