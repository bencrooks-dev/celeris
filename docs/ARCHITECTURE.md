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
  │   passes.py    constant folding, dead-code elimination      │
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
| **Numba** | Production LLVM JIT for a large NumPy/Python subset, type inference, `prange`, CUDA. | celeris is a readable re-implementation of the same *architecture* at a fraction of the scope: mandatory annotations, 1-D arrays only, no GPU, no threads. |
| **Cython** | Ahead-of-time Python→C transpiler with its own superset syntax and a build step. | celeris is a runtime JIT on plain annotated Python; no separate syntax, no AOT build (the optional CMake core aside). |
| **PyPy** | Whole-program tracing JIT for general Python via meta-tracing. | celeris compiles only explicitly-marked numeric kernels and falls back to CPython for everything else; far narrower, far simpler. |
| **Mojo** | A new language (Python superset) with its own MLIR-based compiler and runtime. | celeris is not a language; it is a library that compiles a subset of *existing* Python. |
| **MLIR** | A reusable compiler-IR infrastructure with dialects and lowering frameworks. | celeris's IR is a single, fixed, JSON dict schema — deliberately tiny and non-extensible, optimized for readability over generality. |
| **LLVM ORC** | LLVM's modern on-request JIT linking APIs. | celeris reaches LLVM only through the optional `llvmlite` backend; ORC-level concerns are abstracted away, and the C++ LLVM seam is a documented stub. |

celeris does not aim to beat any of these. It aims to be the smallest honest thing that
demonstrates the full Numba-style pipeline end to end.
