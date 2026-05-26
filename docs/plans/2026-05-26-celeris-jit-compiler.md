# Celeris — JIT Compiler & Runtime Plan

> **Plan ID:** `plan:celeris-jit`
> **For agentic workers:** Use `/athena-build` to execute this plan wave-by-wave. Steps use checkbox (`- [ ]`) syntax for tracking. Each task is TDD: write failing test → verify fail → implement → verify pass → commit.

**Goal:** Ship a complete, professional-grade open-source JIT compiler/runtime for a statically-typed numeric subset of Python, exposed via a `@fast_runtime` decorator, with tiered native backends and a clean LLVM seam.

**Approach:** A pure-Python frontend (AST → validated typed IR) feeds a pluggable backend layer. Backends form a tiered dispatch: hand-tuned kernel registry → optional llvmlite JIT → generic C++ source-gen (clang at runtime) → graceful fallback to the original Python function. A separately-buildable C++ core (C ABI + pybind11) demonstrates the production binding path and marks where C++-side LLVM lowering goes. Everything is verified by a differential test harness that cross-checks every backend against pure Python.

**Tech Stack:** Python 3.10+ (stdlib `ast`, `ctypes`, `inspect`), C++17 (Apple clang / gcc), `pybind11` (optional, production binding), `llvmlite` (optional, LLVM backend), `numpy` (array tests + benchmarks), `pytest`, CMake (optional native build), GitHub Actions CI, Apache-2.0.

**Key design decision — tiered dispatch (resolves the "skip the hand-written kernel?" question):**
The hand-tuned kernel tier is *not* the primary codegen path (that doesn't scale). It is a **specialization fast-path** — a registry keyed by a normalized IR fingerprint, mapping recognized shapes (saxpy, dot, gemv, sum/reduction, fused elementwise) to hand-optimized C++ templates. General source-gen is the fallback when no golden kernel matches; the pure-Python interpreter is the portable reference; the original Python function is the ultimate safety fallback. This mirrors MKL/oneDNN/cuDNN (hand-tuned kernels for known shapes, general path otherwise) and gives a clean A/B harness (golden vs generic vs reference).

**Two distinct native layers (do not conflate):**
- **(A) Runtime JIT backends** — Python-driven, shell out to `clang` at runtime, load via `ctypes`. No CMake needed. This is the working speedup path: `kernels.py`, `sourcegen.py`, plus optional `llvm.py` (in-process via llvmlite) and `interpreter.py` (reference).
- **(B) Standalone C++ core** — `src/celeris/_native/celeris_core.{hpp,cpp}` exposes the C ABI (`celeris_compile/free/strategy` + golden kernels + LLVM lowering seam stub) and `bindings.cpp` is the pybind11 production module. Built via CMake. Demonstrates the production binding + the C++ LLVM seam. Tests skip when not built.

---

## File Structure

**Create:**
- `pyproject.toml` — packaging; extras `[llvm]`, `[native]`, `[dev]`; pytest config
- `src/celeris/__init__.py` — public API: `fast_runtime`, `__version__`
- `src/celeris/errors.py` — `UnsupportedFeature`, `TypeErrorIR`, `CompileError`, `VerifyError`
- `src/celeris/types.py` — IR type system + array markers (`F64Array`, `F32Array`, `I64Array`, `I32Array`) + promotion rules + annotation mapping
- `src/celeris/ir.py` — typed IR node constructors, schema version, `dumps`/`loads` (JSON round-trip)
- `src/celeris/parser.py` — `parse_function(fn) -> dict`: source → AST → validate subset → typed IR
- `src/celeris/verifier.py` — `verify_ir(ir) -> None`: independent structural + type verification (never trusts parser)
- `src/celeris/passes.py` — `optimize(ir) -> ir`: constant folding + dead-code elimination
- `src/celeris/backends/__init__.py` — registry, `get_backend`, `available_backends`, default dispatch chain
- `src/celeris/backends/base.py` — `Backend` Protocol (`name`, `available()`, `compile(ir)`)
- `src/celeris/backends/interpreter.py` — pure-Python IR interpreter (reference; always available)
- `src/celeris/backends/sourcegen.py` — IR → C++ source → clang `-O3` → `.so` → ctypes
- `src/celeris/backends/kernels.py` — golden-kernel registry + IR fingerprint matcher (hand-tuned C++ templates)
- `src/celeris/backends/llvm.py` — optional llvmlite JIT (skips when llvmlite absent)
- `src/celeris/runtime.py` — `@fast_runtime` decorator: capture, parse, verify, optimize, cache, dispatch, fallback, debug
- `src/celeris/_native/celeris_core.hpp` — C ABI declarations
- `src/celeris/_native/celeris_core.cpp` — IR verify + pattern match + hand-written kernels + LLVM seam stub
- `src/celeris/_native/bindings.cpp` — pybind11 production module
- `CMakeLists.txt` — builds the C++ core static lib + pybind11 module
- `tests/conftest.py` — fixtures + skip markers (`needs_clang`, `needs_llvmlite`, `needs_native`)
- `tests/test_types.py`, `tests/test_ir_roundtrip.py`, `tests/test_parser.py`, `tests/test_verifier.py`, `tests/test_passes.py`, `tests/test_interpreter.py`, `tests/test_sourcegen.py`, `tests/test_kernels.py`, `tests/test_llvm.py`, `tests/test_runtime.py`, `tests/test_differential.py`
- `benchmarks/benchmark.py` — Python loop vs NumPy vs celeris
- `examples/saxpy.py`, `examples/reduction.py`, `examples/agent_loop.py`
- `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/IR_SPEC.md`
- `README.md`, `CONTRIBUTING.md`, `LICENSE`, `CHANGELOG.md`, `.gitignore`
- `.github/workflows/ci.yml` — matrix: core (clang only) + dedicated llvmlite job + lint

> Repo is already `git init`'d on `main` with `user.name`/`user.email` configured. The bootstrap commit contains this plan.

---

## Wave 1 — Repo scaffold & packaging

### Task 1.1: Package scaffold + packaging metadata
**Files:** Create `pyproject.toml`, `src/celeris/__init__.py`, `.gitignore`, `LICENSE`, `tests/conftest.py`

- [ ] **Step 1: Write the failing test** — `tests/test_smoke.py`
```python
def test_import_and_version():
    import celeris
    assert isinstance(celeris.__version__, str)
    assert celeris.__version__.count(".") >= 2
    assert callable(celeris.fast_runtime)
```
- [ ] **Step 2: Verify it fails** — `pytest tests/test_smoke.py -v` → FAIL (`ModuleNotFoundError: celeris`)
- [ ] **Step 3: Implement**
  - `pyproject.toml`: `[build-system]` setuptools; `[project]` name=`celeris`, version=`0.1.0`, requires-python `>=3.10`, src layout (`[tool.setuptools.packages.find] where=["src"]`); `[project.optional-dependencies]` `llvm=["llvmlite>=0.42"]`, `native=["pybind11>=2.11"]`, `dev=["pytest>=8","numpy>=1.24","ruff"]`; `[tool.pytest.ini_options]` `testpaths=["tests"]`, register markers `needs_clang`, `needs_llvmlite`, `needs_native`.
  - `src/celeris/__init__.py`: `__version__ = "0.1.0"`; `from .runtime import fast_runtime`. *(temporary stub `fast_runtime` until Wave 7; for this task define `from .runtime import fast_runtime` guarded — instead, for Wave 1, define a placeholder in `__init__` re-exported, and the real one lands in 7. Simplest: create `runtime.py` now with a no-op passthrough decorator so the import works, replaced in Wave 7.)*
  - `src/celeris/runtime.py` (placeholder): `def fast_runtime(fn=None, **kw):\n    def deco(f): return f\n    return deco(fn) if callable(fn) else deco`
  - `.gitignore`: `__pycache__/`, `*.so`, `*.dylib`, `build/`, `dist/`, `*.egg-info/`, `.pytest_cache/`, `.celeris_cache/`, `_skbuild/`, `*.o`
  - `LICENSE`: full Apache-2.0 text, copyright `2026 Ben Crooks`.
  - `tests/conftest.py`: define skip helpers — `shutil.which("clang++")` → `needs_clang`; `importlib.util.find_spec("llvmlite")` → `needs_llvmlite`; native module import → `needs_native`. Provide pytest markers via `pytest.mark.skipif`.
- [ ] **Step 4: Verify pass** — `pip install -e .` then `pytest tests/test_smoke.py -v` → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "chore: package scaffold, packaging metadata, license [plan:celeris-jit] [wave:1/task:1]"`

---

## Wave 2 — Foundations (parallel)
No interdependencies; run together. Depend only on Wave 1 scaffold.

### Task 2.1: Type system + array markers
**Files:** Create `src/celeris/types.py`, `tests/test_types.py`

- [ ] **Step 1: Failing test**
```python
from celeris.types import (annotation_to_type, unify_numeric, is_float, is_int,
                           F64Array, I64Array, SCALAR, PTR)

def test_scalar_annotations():
    assert annotation_to_type("int") == "i64"
    assert annotation_to_type("float") == "f64"

def test_array_annotations():
    assert annotation_to_type("F64Array") == {"ptr": "f64"}
    assert annotation_to_type("I64Array") == {"ptr": "i64"}

def test_promotion():
    assert unify_numeric("i64", "i64") == "i64"
    assert unify_numeric("i64", "f64") == "f64"   # int+float -> float
    assert unify_numeric("f32", "f64") == "f64"

def test_predicates():
    assert is_float("f64") and is_float("f32")
    assert is_int("i32") and is_int("i64")
    assert not is_float("i64")
```
- [ ] **Step 2: Verify fail** — `pytest tests/test_types.py -v` → FAIL (import error)
- [ ] **Step 3: Implement** — define scalar marker classes `f32`, `i32`; array marker classes `F64Array`, `F32Array`, `I64Array`, `I32Array`; `_SCALAR = {"int":"i64","float":"f64","f32":"f32","i32":"i32"}`; `_ARRAY = {"F64Array":{"ptr":"f64"}, ...}`; `annotation_to_type(name)` raises `UnsupportedFeature` (from `celeris.errors`) on unknown; `unify_numeric`, `is_float`, `is_int` per the design doc rules (any float → f64; int mix → i64).
- [ ] **Step 4: Verify pass** — `pytest tests/test_types.py -v` → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: IR type system and array markers [plan:celeris-jit] [wave:2/task:1]"`

### Task 2.2: Error hierarchy
**Files:** Create `src/celeris/errors.py`, `tests/test_errors.py`

- [ ] **Step 1: Failing test**
```python
import celeris.errors as e
def test_hierarchy():
    assert issubclass(e.UnsupportedFeature, e.CelerisError)
    assert issubclass(e.TypeErrorIR, e.CelerisError)
    assert issubclass(e.VerifyError, e.CelerisError)
    assert issubclass(e.CompileError, e.CelerisError)
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — `class CelerisError(Exception)`; subclasses `UnsupportedFeature`, `TypeErrorIR`, `VerifyError`, `CompileError`, each with a docstring describing when raised.
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: error hierarchy [plan:celeris-jit] [wave:2/task:2]"`

### Task 2.3: Project docs stubs
**Files:** Create `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/IR_SPEC.md`

- [ ] **Step 1: Failing test** — `tests/test_docs_present.py`
```python
import pathlib
def test_docs_exist():
    root = pathlib.Path(__file__).resolve().parents[1]
    for f in ["README.md","CONTRIBUTING.md","CHANGELOG.md",
              "docs/ARCHITECTURE.md","docs/ROADMAP.md","docs/IR_SPEC.md"]:
        p = root / f
        assert p.exists() and p.stat().st_size > 200, f
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — substantive stubs (>200 bytes each). README: project pitch (honest positioning vs Numba — "a readable, restricted re-implementation of the Numba architecture"), quickstart, supported subset summary, install matrix. ARCHITECTURE: layered diagram + tiered dispatch from this plan. ROADMAP: v0.1→v1.0 table. IR_SPEC: the IR schema. CONTRIBUTING: dev setup, TDD expectation, how to run differential tests. CHANGELOG: `## [Unreleased]` + `## [0.1.0]`.
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "docs: project documentation stubs [plan:celeris-jit] [wave:2/task:3]"`

---

## Wave 3 — IR core

### Task 3.1: Typed IR constructors + JSON round-trip
**Depends on:** 2.1, 2.2
**Files:** Create `src/celeris/ir.py`, `tests/test_ir_roundtrip.py`

- [ ] **Step 1: Failing test**
```python
import celeris.ir as ir
def test_constructors_and_roundtrip():
    e = ir.binop("+", "f64", ir.var("a","f64"), ir.const("f64", 1.0))
    assert e["k"] == "binop" and e["type"] == "f64"
    kern = ir.kernel("k", [ir.param("a","f64")], "f64",
                     [ir.ret(ir.var("a","f64"))])
    blob = ir.dumps(kern)
    assert ir.loads(blob) == kern
    assert ir.SCHEMA_VERSION >= 1

def test_for_node():
    f = ir.for_("i", ir.const("i64",0), ir.var("n","i64"), ir.const("i64",1), [])
    assert f["op"] == "for" and f["var"] == "i"
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — IR is a JSON-serializable dict tree (per design §4). Provide constructor helpers: `const(type,value)`, `var(type,name)` → note arg order in tests is `var(name,type)`; **fix: define `var(name, type)`**, `index(array,index,type)`, `binop(op,type,lhs,rhs)`, `cmp(op,lhs,rhs)`, `boolop(op,args)`, `call(fn,type,args)`, `cast(type,value)`; statements `assign(target,value)`, `augassign(binop,target,value)`, `for_(var,start,stop,step,body)`, `while_(cond,body)`, `if_(cond,then,els)`, `ret(value)`; structural `param(name,type)`, `kernel(name,params,ret,body)`, `module(kernels)`. `SCHEMA_VERSION = 1`. `dumps(obj)` = `json.dumps` with sorted keys; `loads(s)` = `json.loads`. *(Self-review note: keep `var(name, type)` ordering consistent everywhere — parser & backends must match.)*
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: typed IR constructors and JSON round-trip [plan:celeris-jit] [wave:3/task:1]"`

---

## Wave 4 — Frontend & analysis (parallel)
All depend on Wave 3 IR + Wave 2 types/errors. Independent of each other.

### Task 4.1: AST parser + subset validator
**Files:** Create `src/celeris/parser.py`, `tests/test_parser.py`
**Reference:** Design doc `parser.py` (prior conversation turn) — port it, adapting imports to `celeris.ir`, `celeris.types`, `celeris.errors`, and emit IR via `ir.py` constructors (not raw dicts) so node shapes stay canonical.

- [ ] **Step 1: Failing test**
```python
import pytest
from celeris.parser import parse_function
from celeris.types import F64Array
from celeris.errors import UnsupportedFeature

def test_saxpy_ir():
    def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in range(n):
            y[i] = a * x[i] + y[i]
    ir = parse_function(saxpy)
    assert ir["name"] == "saxpy"
    assert [p["type"] for p in ir["params"]] == ["f64", {"ptr":"f64"}, {"ptr":"f64"}, "i64"]
    assert ir["ret"] == "void"
    assert ir["body"][0]["op"] == "for"

def test_rejects_classes():
    def bad(x: int) -> int:
        class C: ...
        return x
    with pytest.raises(UnsupportedFeature):
        parse_function(bad)

def test_rejects_unknown_call():
    def bad(x: float) -> float:
        return print(x)  # not whitelisted
    with pytest.raises(UnsupportedFeature):
        parse_function(bad)

def test_rejects_multidim_index():
    def bad(x: F64Array) -> float:
        return x[1, 2]
    with pytest.raises(UnsupportedFeature):
        parse_function(bad)

def test_requires_annotations():
    def bad(x) -> int:
        return x
    with pytest.raises(UnsupportedFeature):
        parse_function(bad)
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — port design-doc parser: `_Validator(ast.NodeVisitor)` with `_ALLOWED_NODES` allowlist + targeted `visit_Call`/`visit_Subscript` rejections; `IRBuilder` with symbol table, annotation→type via `types.annotation_to_type`, tiny first-assignment local inference, `range()` lowering, intrinsic whitelist (`sqrt,exp,log,sin,cos,fabs,floor,fmax,fmin,len`). Strip decorators before parsing. Handle py3.9+ slice shape. Emit nodes through `ir.py` constructors.
- [ ] **Step 4: Verify pass** → `pytest tests/test_parser.py -v` → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: AST parser and subset validator [plan:celeris-jit] [wave:4/task:1]"`

### Task 4.2: IR verifier
**Files:** Create `src/celeris/verifier.py`, `tests/test_verifier.py`

- [ ] **Step 1: Failing test**
```python
import pytest, celeris.ir as ir
from celeris.verifier import verify_ir
from celeris.errors import VerifyError

def test_valid_ir_passes():
    k = ir.kernel("k",[ir.param("a","f64")],"f64",[ir.ret(ir.var("a","f64"))])
    verify_ir(k)  # no raise

def test_missing_type_rejected():
    bad = {"name":"k","params":[{"name":"a"}],"ret":"f64","body":[]}  # param has no type
    with pytest.raises(VerifyError):
        verify_ir(bad)

def test_index_into_non_pointer_rejected():
    k = ir.kernel("k",[ir.param("a","f64")],"f64",
        [ir.ret(ir.index("a", ir.const("i64",0), "f64"))])  # 'a' is scalar
    with pytest.raises(VerifyError):
        verify_ir(k)
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — recursive walk; every `Expr`/`LValue` must carry a valid `type`; binop operands must unify to the node type; index target must be `{"ptr":...}` and element type must match; `range`/loop bounds must be int; intrinsics must be whitelisted with correct arity. Raise `VerifyError` with node context. This is the trust boundary for the native backends — never assume the parser was correct.
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: independent IR verifier (trust boundary) [plan:celeris-jit] [wave:4/task:2]"`

### Task 4.3: Optimization passes (const-fold + DCE)
**Files:** Create `src/celeris/passes.py`, `tests/test_passes.py`

- [ ] **Step 1: Failing test**
```python
import celeris.ir as ir
from celeris.passes import optimize

def test_constant_folding():
    e = ir.binop("+","i64", ir.const("i64",2), ir.const("i64",3))
    k = ir.kernel("k",[],"i64",[ir.ret(e)])
    out = optimize(k)
    assert out["body"][0]["value"] == ir.const("i64",5)

def test_dead_assignment_removed():
    body = [ir.assign(ir.var("dead","i64"), ir.const("i64",1)),
            ir.ret(ir.const("i64",0))]
    k = ir.kernel("k",[],"i64",body)
    out = optimize(k)
    assert all(s["op"] != "assign" for s in out["body"])
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — `fold_constants(node)`: recursively evaluate binops/cmps over two `const` operands (integer + float arithmetic; respect `//`, `%`, `**`); `eliminate_dead_code(kernel)`: remove assignments to locals never read afterward (single backward liveness pass; conservative — keep anything read, keep all array stores and side-effecting calls). `optimize(ir)` runs fold then DCE; idempotent. Keep simple and provably safe (YAGNI — no loop-invariant motion yet; that's roadmap).
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: constant folding and dead-code elimination passes [plan:celeris-jit] [wave:4/task:3]"`

### Task 4.4: Backend Protocol + registry
**Files:** Create `src/celeris/backends/__init__.py`, `src/celeris/backends/base.py`, `tests/test_backends_registry.py`

- [ ] **Step 1: Failing test**
```python
from celeris.backends import register, get_backend, available_backends, default_chain
from celeris.backends.base import Backend

class _Dummy:
    name = "dummy"
    def available(self): return True
    def compile(self, ir): return lambda *a: 42

def test_register_and_get():
    register(_Dummy())
    assert get_backend("dummy").compile({})(1,2) == 42
    assert "dummy" in [b.name for b in available_backends()]

def test_default_chain_is_priority_ordered():
    # kernels (fast-path) before sourcegen before interpreter
    names = [b.name for b in default_chain()]
    assert names.index("kernels") < names.index("sourcegen")
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — `base.py`: `Backend` `Protocol` with `name: str`, `available() -> bool`, `compile(ir: dict) -> Callable`. `__init__.py`: module-level `_REGISTRY: dict[str,Backend]`; `register(b)`, `get_backend(name)` (raise `KeyError` if missing), `available_backends()` (those whose `available()` is True), `default_chain()` returning available backends ordered by priority `["kernels","llvm","sourcegen","interpreter"]`. Backends self-register on import (import them lazily in `__init__` inside a try/except so an optional backend's missing dep can't break the registry). *(Self-review: `default_chain` must tolerate `kernels`/`sourcegen` being unavailable — test asserts relative order only among present ones; guard accordingly.)*
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: backend Protocol and registry with priority chain [plan:celeris-jit] [wave:4/task:4]"`

---

## Wave 5 — Reference backend & C++ core (parallel)

### Task 5.1: Pure-Python IR interpreter backend
**Depends on:** 3.1, 4.4
**Files:** Create `src/celeris/backends/interpreter.py`, `tests/test_interpreter.py`

- [ ] **Step 1: Failing test**
```python
import numpy as np
from celeris.backends.interpreter import InterpreterBackend

def test_interpreter_saxpy():
    ir = {"name":"saxpy","ret":"void",
      "params":[{"name":"a","type":"f64"},{"name":"x","type":{"ptr":"f64"}},
                {"name":"y","type":{"ptr":"f64"}},{"name":"n","type":"i64"}],
      "body":[{"op":"for","var":"i",
        "start":{"k":"const","type":"i64","value":0},
        "stop":{"k":"var","type":"i64","name":"n"},
        "step":{"k":"const","type":"i64","value":1},
        "body":[{"op":"assign",
          "target":{"k":"index","array":"y","type":"f64",
                    "index":{"k":"var","type":"i64","name":"i"}},
          "value":{"k":"binop","op":"+","type":"f64",
            "lhs":{"k":"binop","op":"*","type":"f64",
              "lhs":{"k":"var","type":"f64","name":"a"},
              "rhs":{"k":"index","array":"x","type":"f64",
                     "index":{"k":"var","type":"i64","name":"i"}}},
            "rhs":{"k":"index","array":"y","type":"f64",
                   "index":{"k":"var","type":"i64","name":"i"}}}}]}]}
    fn = InterpreterBackend().compile(ir)
    x = np.arange(5, dtype=np.float64); y = np.ones(5, dtype=np.float64)
    fn(2.0, x, y, 5)
    np.testing.assert_allclose(y, 2.0*np.arange(5)+1.0)

def test_interpreter_always_available():
    assert InterpreterBackend().available() is True
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — tree-walking evaluator over the IR: an env dict maps param/local names to Python values (scalars) or array objects (anything supporting `__getitem__`/`__setitem__`, e.g. numpy or list). `compile(ir)` returns a closure `run(*args)` binding positional args to params, executing statements (`for`/`while`/`if`/`assign`/`augassign`/`return`), evaluating exprs (`const`/`var`/`index`/`binop`/`cmp`/`bool`/`call`/`cast`). Intrinsics dispatch to `math`. `//`,`%`,`**` per Python semantics on the typed values. Returns the `return` value or `None`. `available()` → True. Register the backend.
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: pure-Python IR interpreter backend (reference) [plan:celeris-jit] [wave:5/task:1]"`

### Task 5.2: C++ core (C ABI + verify + golden kernel + LLVM seam stub)
**Depends on:** none (self-contained C++)
**Files:** Create `src/celeris/_native/celeris_core.hpp`, `src/celeris/_native/celeris_core.cpp`, `tests/test_native_core.py`
**Reference:** Design doc `compiler.hpp`/`compiler.cpp` — port, renaming symbols to `celeris_*`.

- [ ] **Step 1: Failing test** (compiles the core with clang, calls via ctypes; skipped if no clang)
```python
import pytest, subprocess, shutil, ctypes, pathlib, json
import numpy as np
pytestmark = pytest.mark.needs_clang

CORE = pathlib.Path("src/celeris/_native/celeris_core.cpp")

@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    out = tmp_path_factory.mktemp("native") / "libceleris_core.so"
    # nlohmann/json is header-only & vendored under _native/third_party; -I it.
    inc = CORE.parent / "third_party"
    subprocess.run(["clang++","-O3","-std=c++17","-fPIC","-shared",
                    f"-I{inc}", str(CORE), "-o", str(out)], check=True)
    return ctypes.CDLL(str(out))

def test_saxpy_symbol(lib):
    lib.celeris_saxpy.restype = None
    lib.celeris_saxpy.argtypes = [ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64]
    x = np.arange(4, dtype=np.float64); y = np.ones(4, dtype=np.float64)
    lib.celeris_saxpy(3.0, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 4)
    np.testing.assert_allclose(y, 3.0*np.arange(4)+1.0)

def test_compile_strategy(lib):
    lib.celeris_compile.restype = ctypes.c_void_p
    lib.celeris_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.celeris_strategy.restype = ctypes.c_int
    lib.celeris_strategy.argtypes = [ctypes.c_void_p]
    saxpy_ir = json.dumps({"name":"saxpy","ret":"void","params":[],
        "body":[{"op":"for","var":"i","body":[
            {"op":"assign","target":{"k":"index"},
             "value":{"k":"binop","op":"+"}}]}]}).encode()
    err = ctypes.create_string_buffer(256)
    h = lib.celeris_compile(saxpy_ir, err, 256)
    assert h, err.value.decode()
    assert lib.celeris_strategy(h) == 1   # STRAT_HANDWRITTEN
    lib.celeris_free.argtypes = [ctypes.c_void_p]
    lib.celeris_free(h)
```
- [ ] **Step 2: Verify fail** → FAIL (file missing / compile error)
- [ ] **Step 3: Implement** — `celeris_core.hpp`/`.cpp` per design doc: `extern "C"` `celeris_compile/free/strategy` + `celeris_saxpy(double,const double*,double*,int64_t)`; `verify_ir` (defensive); `matches_saxpy` fingerprint; `lower_to_llvm` stub guarded by `#ifdef CELERIS_LLVM` returning false with a clear message (the documented C++ LLVM seam). Vendor `nlohmann/json.hpp` single header into `_native/third_party/nlohmann/json.hpp` (download once, commit it; MIT-licensed — note in NOTICE).
- [ ] **Step 4: Verify pass** → `pytest tests/test_native_core.py -v` → PASS (or skip if no clang)
- [ ] **Step 5: Commit** — `git commit -am "feat: C++ core — C ABI, verify, golden saxpy, LLVM seam stub [plan:celeris-jit] [wave:5/task:2]"`

---

## Wave 6 — JIT backends (parallel)

### Task 6.1: C++ source-gen backend (clang at runtime)
**Depends on:** 3.1, 4.2, 4.4
**Files:** Create `src/celeris/backends/sourcegen.py`, `tests/test_sourcegen.py` (`needs_clang`)

- [ ] **Step 1: Failing test**
```python
import pytest, numpy as np
from celeris.backends.sourcegen import SourceGenBackend
from celeris.parser import parse_function
from celeris.types import F64Array
pytestmark = pytest.mark.needs_clang

def test_sourcegen_saxpy():
    def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in range(n):
            y[i] = a * x[i] + y[i]
    fn = SourceGenBackend().compile(parse_function(saxpy))
    x = np.arange(1000, dtype=np.float64); y = np.zeros(1000, dtype=np.float64)
    fn(2.0, x, y, 1000)
    np.testing.assert_allclose(y, 2.0*np.arange(1000))

def test_sourcegen_scalar_return():
    def f(a: float, b: float) -> float:
        return a*b + 1.0
    fn = SourceGenBackend().compile(parse_function(f))
    assert abs(fn(3.0, 4.0) - 13.0) < 1e-9
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — `emit_cpp(ir) -> str`: 1:1 structured-IR → C++ (types map `i32/i64/f32/f64`→`int32_t/int64_t/float/double`, `{ptr:T}`→`T*`; `for`→`for`, etc.; `extern "C"` wrapper with the kernel name). `compile(ir)`: hash IR → cache dir `~/.celeris_cache/`; if `.so` missing, write `.cpp`, run `clang++ -O3 -std=c++17 -fPIC -shared -march=native`, load via `ctypes`. Set `argtypes`/`restype` from `ir["params"]`/`ir["ret"]`; the returned closure marshals numpy arrays (`arr.ctypes.data_as(POINTER(...))`) and scalars. On any failure raise `CompileError`. `available()` = `shutil.which("clang++") is not None`. Register.
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: C++ source-gen JIT backend [plan:celeris-jit] [wave:6/task:1]"`

### Task 6.2: Golden-kernel registry + fingerprint matcher
**Depends on:** 6.1 (reuses the clang compile+load helper), 4.4
**Files:** Create `src/celeris/backends/kernels.py`, `tests/test_kernels.py` (`needs_clang`)

- [ ] **Step 1: Failing test**
```python
import pytest, numpy as np
from celeris.backends.kernels import KernelBackend, fingerprint, REGISTRY
from celeris.parser import parse_function
from celeris.types import F64Array
pytestmark = pytest.mark.needs_clang

def test_saxpy_matches_golden():
    def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in range(n):
            y[i] = a * x[i] + y[i]
    be = KernelBackend()
    ir = parse_function(saxpy)
    assert be.matches(ir)               # fingerprint hit
    fn = be.compile(ir)
    x = np.arange(8, dtype=np.float64); y = np.ones(8, dtype=np.float64)
    fn(2.0, x, y, 8)
    np.testing.assert_allclose(y, 2.0*np.arange(8)+1.0)

def test_unknown_shape_does_not_match():
    def weird(x: F64Array, n: int) -> float:
        s = 0.0
        for i in range(n):
            s = s + x[i]*x[i]*x[i]      # not a registered shape
        return s
    # sum-of-cubes may or may not be registered; assert API, not a hit:
    assert isinstance(KernelBackend().matches(parse_function(weird)), bool)

def test_registry_has_core_shapes():
    assert {"saxpy","dot","sum"} <= set(REGISTRY.keys())
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — `fingerprint(ir) -> str`: normalize the IR (strip names, keep shape: op structure + types + index patterns) to a canonical hash. `REGISTRY: dict[str, GoldenKernel]` where each `GoldenKernel` has `(matcher(ir)->bool, cpp_template, signature)`. Register hand-tuned templates for `saxpy` (`y[i]=a*x[i]+y[i]`), `dot` (sum of `x[i]*y[i]`), `sum` (reduction), `scale` (`y[i]=a*x[i]`), `axpy_fused` (3-term fused). Templates use `#pragma omp simd` / `__restrict__` hints. `KernelBackend.matches(ir)` → any registry matcher hits. `compile(ir)` → pick matching template, compile+load via the shared sourcegen helper (`-O3 -march=native -fopenmp-simd`). `available()` = clang present. Register with higher priority than sourcegen. *(This is the explicit answer to "don't skip the hand-written kernels": they are a real, extensible fast-path tier, A/B-comparable against generic codegen in the differential + benchmark suites.)*
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: golden-kernel registry and IR fingerprint matcher [plan:celeris-jit] [wave:6/task:2]"`

### Task 6.3: Optional llvmlite JIT backend
**Depends on:** 3.1, 4.2, 4.4
**Files:** Create `src/celeris/backends/llvm.py`, `tests/test_llvm.py` (`needs_llvmlite`)

- [ ] **Step 1: Failing test**
```python
import pytest, numpy as np
pytest.importorskip("llvmlite")
from celeris.backends.llvm import LLVMBackend
from celeris.parser import parse_function
from celeris.types import F64Array

def test_llvm_scalar():
    def f(a: float, b: float) -> float:
        return a*b + 2.0
    fn = LLVMBackend().compile(parse_function(f))
    assert abs(fn(3.0, 4.0) - 14.0) < 1e-9

def test_llvm_saxpy():
    def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in range(n):
            y[i] = a*x[i] + y[i]
    fn = LLVMBackend().compile(parse_function(saxpy))
    x = np.arange(64, dtype=np.float64); y = np.zeros(64, dtype=np.float64)
    fn(2.0, x, y, 64)
    np.testing.assert_allclose(y, 2.0*np.arange(64))

def test_available_reflects_import():
    from celeris.backends.llvm import LLVMBackend
    assert LLVMBackend().available() is True   # since importorskip passed
```
- [ ] **Step 2: Verify fail** → FAIL (or skip if llvmlite absent — install in dev env to implement: `pip install llvmlite`)
- [ ] **Step 3: Implement** — `LLVMBackend.compile(ir)`: build an `llvmlite.ir.Module`; map IR types to LLVM types; for scalars, straight expression lowering; for control flow, create basic blocks and **PHI nodes for loop induction variables** (the hard part — structured→SSA). Run the llvmlite pass pipeline at opt-level 2/3. Materialize with `llvmlite.binding` MCJIT/ORC execution engine, get the function address, wrap in a `ctypes.CFUNCTYPE` matching the signature, return a marshaling closure (same array handling as sourcegen). `available()` = `importlib.util.find_spec("llvmlite") is not None`. Register at priority below kernels, above sourcegen. Guard import so absence never breaks the registry.
- [ ] **Step 4: Verify pass** → PASS (when llvmlite installed)
- [ ] **Step 5: Commit** — `git commit -am "feat: optional llvmlite JIT backend with structured->SSA lowering [plan:celeris-jit] [wave:6/task:3]"`

### Task 6.4: pybind11 production binding + CMake
**Depends on:** 5.2
**Files:** Create `src/celeris/_native/bindings.cpp`, `CMakeLists.txt`, `tests/test_pybind.py` (`needs_native`)

- [ ] **Step 1: Failing test**
```python
import pytest
celeris_native = pytest.importorskip("celeris_native")  # built via CMake
import numpy as np
def test_pybind_saxpy():
    x = np.arange(5, dtype=np.float64); y = np.ones(5, dtype=np.float64)
    celeris_native.saxpy(2.0, x, y)     # uses py::array_t buffer protocol
    np.testing.assert_allclose(y, 2.0*np.arange(5)+1.0)
```
- [ ] **Step 2: Verify fail** → FAIL/skip
- [ ] **Step 3: Implement** — `bindings.cpp` per design §8B: `PYBIND11_MODULE(celeris_native, m)` exposing `saxpy(double, py::array_t<double>, py::array_t<double>)` via `.request()` buffers, and `compile_kernel(std::string ir_json)` wrapping the C ABI with exceptions → Python. `CMakeLists.txt`: `find_package(pybind11)`, build `celeris_core` static lib + `pybind11_add_module(celeris_native bindings.cpp celeris_core.cpp)`, `-O3 -std=c++17`, optional `CELERIS_LLVM` cache var. Document `pip install cmake pybind11 && cmake -S . -B build && cmake --build build` in CONTRIBUTING.
- [ ] **Step 4: Verify pass** → PASS (when built) / skip
- [ ] **Step 5: Commit** — `git commit -am "feat: pybind11 production binding and CMake build [plan:celeris-jit] [wave:6/task:4]"`

---

## Wave 7 — Runtime decorator (end-to-end wiring)

### Task 7.1: `@fast_runtime` decorator
**Depends on:** 4.1, 4.2, 4.3, 4.4, 5.1 (and 6.x backends register when available)
**Files:** Modify `src/celeris/runtime.py` (replace placeholder), Create `tests/test_runtime.py`
**Reference:** Design doc `runtime.py` — port, wiring parse→verify→optimize→dispatch.

- [ ] **Step 1: Failing test**
```python
import numpy as np
from celeris import fast_runtime
from celeris.types import F64Array

def test_compiles_and_runs():
    @fast_runtime
    def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in range(n):
            y[i] = a*x[i] + y[i]
    x = np.arange(100, dtype=np.float64); y = np.zeros(100, dtype=np.float64)
    saxpy(2.0, x, y, 100)
    np.testing.assert_allclose(y, 2.0*np.arange(100))

def test_falls_back_on_unsupported():
    @fast_runtime
    def uses_dict(n: int) -> int:
        d = {}            # unsupported -> must fall back to Python, still work
        for i in range(n):
            d[i] = i
        return len(d)
    assert uses_dict(3) == 3

def test_cache_reuses_compiled():
    calls = {"n": 0}
    @fast_runtime(debug=False)
    def f(a: float) -> float:
        return a + 1.0
    f(1.0); f(2.0)
    assert len(f.__celeris_cache__) >= 1

def test_debug_emits_ir(capsys):
    @fast_runtime(debug=True)
    def f(a: float) -> float:
        return a*2.0
    f(1.0)
    out = capsys.readouterr().out
    assert "IR" in out
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — `fast_runtime(fn=None, *, backend=None, debug=False)`: on first call per signature, `parse_function` → `verify_ir` → `optimize`; pick backend (explicit `backend=` via `get_backend`, else first of `default_chain()` whose `compile` succeeds); cache compiled callable by `(qualname, arg-type-tags)`. On `UnsupportedFeature`/`TypeErrorIR`/`VerifyError`/`CompileError` (or any backend exception) → memoize + return original Python function (fallback). `debug=True` prints the IR (`ir.dumps`) and chosen backend. Expose `__celeris_cache__`, `__celeris_ir__()`, `__celeris_wrapped__`. Replace the Wave-1 placeholder.
- [ ] **Step 4: Verify pass** → `pytest tests/test_runtime.py -v` → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: @fast_runtime decorator with cache, dispatch, fallback, debug [plan:celeris-jit] [wave:7/task:1]"`

---

## Wave 8 — Cross-cutting: differential tests, benchmarks, examples, CI (parallel)

### Task 8.1: Differential correctness harness
**Depends on:** Wave 7
**Files:** Create `tests/test_differential.py`

- [ ] **Step 1: Failing test** — parametrize a set of kernels (saxpy, dot, sum, scale, fused 3-term, a `while` loop, an `if`-branch kernel). For each, run pure Python, interpreter backend, and every *available* native backend (sourcegen, kernels, llvm); assert all agree within `1e-9` (float) / exact (int).
```python
import numpy as np, pytest
from celeris.parser import parse_function
from celeris.backends.interpreter import InterpreterBackend
from celeris.backends import available_backends

KERNELS = [...]  # list of (pyfunc, args_factory, reference_fn)

@pytest.mark.parametrize("case", KERNELS, ids=lambda c: c.__name__)
def test_all_backends_agree(case):
    pyfunc, make_args, ref = case.fn, case.args, case.ref
    ir = parse_function(pyfunc)
    base = ref(*make_args())
    for be in [InterpreterBackend(), *available_backends()]:
        try: compiled = be.compile(ir)
        except Exception: continue
        got = compiled(*make_args())
        np.testing.assert_allclose(_result(got, make_args), base, rtol=0, atol=1e-9)
```
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — define the kernel cases + arg factories + reference functions; the harness is the project's correctness backbone (this is the oracle that de-risks the LLVM backend, per the plan rationale).
- [ ] **Step 4: Verify pass** → PASS (backends not available are skipped per-case)
- [ ] **Step 5: Commit** — `git commit -am "test: differential correctness harness across all backends [plan:celeris-jit] [wave:8/task:1]"`

### Task 8.2: Benchmark suite
**Files:** Create `benchmarks/benchmark.py`
**Reference:** Design doc `benchmark.py`.

- [ ] **Step 1: Test** — `tests/test_benchmark_runs.py`: import and run the benchmark at tiny N to assert it executes and reports all rows without error (no perf assertions).
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — port design `benchmark.py`: compares pure-Python loop, NumPy (`a*x+y` and in-place), and `@fast_runtime`; medians over reps; honest commentary printed (memory-bound, NumPy parity expected for single ops, fusion wins for multi-op). Add a fused multi-op case (`d = a*x + b*y + c*z`) where celeris can actually beat NumPy via avoided temporaries; report compile-latency separately from steady-state.
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "perf: benchmark suite (python vs numpy vs celeris) [plan:celeris-jit] [wave:8/task:2]"`

### Task 8.3: Examples
**Files:** Create `examples/saxpy.py`, `examples/reduction.py`, `examples/agent_loop.py`

- [ ] **Step 1: Test** — `tests/test_examples_run.py`: exec each example module; assert it runs and prints expected sentinel output.
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — `saxpy.py` (decorated saxpy + verification print); `reduction.py` (sum/dot reductions); `agent_loop.py` (a tight numeric scoring loop standing in for a lightweight agent-loop inner kernel — the stated niche). Each runnable as `python examples/X.py`.
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "docs: runnable examples (saxpy, reduction, agent loop) [plan:celeris-jit] [wave:8/task:3]"`

### Task 8.4: GitHub Actions CI
**Files:** Create `.github/workflows/ci.yml`

- [ ] **Step 1: Test** — `tests/test_ci_config.py`: parse the YAML; assert jobs `core` and `llvm` exist; assert the `llvm` job installs `llvmlite`; assert core matrix covers py3.10–3.12 on ubuntu+macos.
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — `ci.yml`: `core` job (matrix ubuntu/macos × py3.10/3.11/3.12; install clang via system; `pip install -e .[dev]`; `pytest -m "not needs_llvmlite and not needs_native"`); `llvm` job (ubuntu, py3.11; `pip install -e .[dev,llvm]`; `pytest -m "needs_llvmlite"` plus the differential suite — **this is the mitigation: llvmlite path tested every push without being a hard dependency**); `lint` job (`ruff check`). Cache pip.
- [ ] **Step 4: Verify pass** → `pytest tests/test_ci_config.py -v` → PASS (`pip install pyyaml` for the test or parse minimally)
- [ ] **Step 5: Commit** — `git commit -am "ci: GitHub Actions — core matrix + dedicated llvmlite job + lint [plan:celeris-jit] [wave:8/task:4]"`

### Task 8.5: README finalize + NOTICE
**Files:** Modify `README.md`; Create `NOTICE`
- [ ] **Step 1: Test** — extend `tests/test_docs_present.py`: README contains "Supported Subset", "Install", "Architecture", honest Numba comparison, CI badge placeholder; `NOTICE` credits vendored nlohmann/json (MIT).
- [ ] **Step 2: Verify fail** → FAIL
- [ ] **Step 3: Implement** — finalize README (badges, quickstart with `@fast_runtime`, subset table, backend tiers, honest performance section, contributing pointer) + `NOTICE`.
- [ ] **Step 4: Verify pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "docs: finalize README and NOTICE [plan:celeris-jit] [wave:8/task:5]"`

---

## Wave 9 — Publish

### Task 9.1: Create GitHub repo + push + tag
**Depends on:** all waves green
- [ ] **Step 1:** Run full suite locally: `pytest -q` (clang tests run; llvmlite/native skip if absent) → all pass/skip, zero failures.
- [ ] **Step 2:** `gh repo create bencrooks-dev/celeris --public --source=. --remote=origin --description "A readable, restricted re-implementation of the Numba architecture: a JIT compiler for a numeric subset of Python." --push`
- [ ] **Step 3:** Verify CI goes green on GitHub Actions (`gh run watch` / `gh run list`).
- [ ] **Step 4:** Tag release: `git tag -a v0.1.0 -m "celeris v0.1.0" && git push origin v0.1.0`; `gh release create v0.1.0 --title "celeris v0.1.0" --notes-file -` (from CHANGELOG).
- [ ] **Step 5:** Final commit if needed: `git commit -am "chore: v0.1.0 release [plan:celeris-jit] [wave:9/task:1]"`

---

## Verification (after all waves)
- [ ] `pytest -q` → all pass; skipped tests only where clang/llvmlite/native genuinely absent
- [ ] Differential harness: all available backends agree with pure Python
- [ ] `python benchmarks/benchmark.py` runs and reports honest numbers (no fabricated speedups)
- [ ] `python examples/*.py` all run
- [ ] CI green on GitHub (core matrix + llvmlite job + lint)
- [ ] Repo public at `github.com/bencrooks-dev/celeris`, `v0.1.0` released
- [ ] Fresh-clone sanity: `pip install -e .[dev]`; `import celeris; celeris.__version__`

## Out of scope (roadmap, not this build)
Type inference (drop mandatory annotations), multi-D/tensor memory model + slicing, loop fusion/tiling optimization passes, `prange`/threads, recursion/call graph, persistent on-disk kernel cache across processes, PyPI publish. Tracked in `docs/ROADMAP.md`.
