# Loop Fusion Pass Plan

> **Plan ID:** `plan:loop-fusion`
> **For agentic workers:** Use `/athena-build` to execute this plan wave-by-wave. Steps use checkbox (`- [ ]`) syntax. Each task is TDD: failing test → verify fail → implement → verify pass → commit. Work happens on branch `feat/loop-fusion`.

**Goal:** Add a provably-safe loop-fusion optimization pass that merges adjacent `for` loops over the same iteration space into one loop body, capturing the "one pass, no temporary" win.

**Approach:** New `fuse_loops(kernel)` in `src/celeris/passes.py`, wired into `optimize()` as `fold → fuse → DCE`, on by default. Fusion is applied only when a conservative legality predicate proves it safe; anything else is left untouched (correct, just unfused). Backends need **no** changes — a fused loop is a normal `for` IR node. Validated by the existing differential harness (fused output must equal the plain-Python oracle across all backends) plus structural unit tests for correct *declines*.

**Legality predicate** — fuse two adjacent `for` loops `L1`, `L2` iff ALL hold:
1. Structurally identical iteration space (`var`, `start`, `stop`, `step` equal).
2. Adjacent — consecutive `for` statements in the same block (applied left-to-right to a fixpoint).
3. No `return` anywhere in either body (recursively).
4. For any array **written** in either body, every subscript of that array (reads and writes, both bodies) is exactly `{"k":"var","name":var}`. Read-only arrays may use any index.
5. No shared scalar dependence: `writes(L1) ∩ refs(L2) = ∅` and `writes(L2) ∩ refs(L1) = ∅`, where `refs = reads ∪ writes` of scalar var names, excluding the loop var. (Loop-local temporaries are fine.)

**Tech Stack:** Python 3.10+, existing celeris IR (dict tree), pytest, ruff.

**Files:**
- Modify: `src/celeris/passes.py` — add `fuse_loops` + private helpers; insert into `optimize()`
- Test: `tests/test_passes.py` — unit tests (fuse positives, decline negatives, non-mutation, idempotency)
- Test: `tests/test_differential.py` — add a fusable multi-loop kernel case (end-to-end correctness across backends)
- Modify: `README.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md` — move fusion roadmap→shipped
- Modify: `pyproject.toml` — version bump `0.1.0` → `0.2.0`

---

## Wave 1 — The fusion pass

### Task 1.1: `fuse_loops` pass + wiring + unit tests
**Files:** Modify `src/celeris/passes.py`; Modify `tests/test_passes.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_passes.py`:
```python
import copy
import celeris.ir as ir
from celeris.passes import fuse_loops, optimize


def _loop(body, var="i", stop_name="n"):
    return ir.for_(var, ir.const("i64", 0), ir.var(stop_name, "i64"),
                   ir.const("i64", 1), body)

def _idx_assign(dst, src_arr, idx_expr):
    return ir.assign(ir.lval_index(dst, ir.var("i", "i64"), "f64"),
                     ir.index(src_arr, idx_expr, "f64"))

def _kernel(body):
    params = [ir.param("n", "i64"), ir.param("x", {"ptr": "f64"}),
              ir.param("y", {"ptr": "f64"}), ir.param("z", {"ptr": "f64"}),
              ir.param("b", {"ptr": "f64"})]
    return ir.kernel("k", params, "void", body)


def test_fuse_adjacent_elementwise():
    # for i: y[i]=x[i] ; for i: z[i]=y[i]  -> one loop, body of 2 (y written@i, read@i: ok)
    l1 = _loop([_idx_assign("y", "x", ir.var("i", "i64"))])
    l2 = _loop([_idx_assign("z", "y", ir.var("i", "i64"))])
    out = fuse_loops(_kernel([l1, l2]))
    fors = [s for s in out["body"] if s["op"] == "for"]
    assert len(fors) == 1 and len(fors[0]["body"]) == 2

def test_fuse_fixpoint_three_loops():
    loops = [_loop([_idx_assign("y", "x", ir.var("i", "i64"))]),
             _loop([_idx_assign("z", "y", ir.var("i", "i64"))]),
             _loop([_idx_assign("y", "z", ir.var("i", "i64"))])]
    out = fuse_loops(_kernel(loops))
    fors = [s for s in out["body"] if s["op"] == "for"]
    assert len(fors) == 1 and len(fors[0]["body"]) == 3

def test_decline_offset_write_index():
    # y written at i+1 (offset) -> decline
    off = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("y", off, "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([_idx_assign("z", "y", ir.var("i", "i64"))])
    out = fuse_loops(_kernel([l1, l2]))
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_decline_shared_accumulator():
    # for i: acc=acc+x[i] ; for i: y[i]=acc  -> writes(L1)={acc} ∩ refs(L2)={acc} -> decline
    l1 = _loop([ir.assign(ir.lval_var("acc", "f64"),
                          ir.binop("+", "f64", ir.var("acc", "f64"),
                                   ir.index("x", ir.var("i", "i64"), "f64")))])
    l2 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.var("acc", "f64"))])
    out = fuse_loops(_kernel([l1, l2]))
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_decline_return_in_body():
    l1 = _loop([ir.ret(ir.const("i64", 0))])
    l2 = _loop([_idx_assign("z", "y", ir.var("i", "i64"))])
    out = fuse_loops(_kernel([l1, l2]))
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_decline_different_iteration_space():
    l1 = _loop([_idx_assign("y", "x", ir.var("i", "i64"))], stop_name="n")
    l2 = _loop([_idx_assign("z", "x", ir.var("i", "i64"))], stop_name="m")
    k = _kernel([l1, l2])
    k["params"].append(ir.param("m", "i64"))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_readonly_array_offset_allowed():
    # both loops only READ b (read-only) with an offset; write y,z by i -> fusable
    off = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.index("b", off, "f64"))])
    l2 = _loop([_idx_assign("z", "b", ir.var("i", "i64"))])
    out = fuse_loops(_kernel([l1, l2]))
    assert len([s for s in out["body"] if s["op"] == "for"]) == 1

def test_fuse_nonmutating_and_idempotent():
    l1 = _loop([_idx_assign("y", "x", ir.var("i", "i64"))])
    l2 = _loop([_idx_assign("z", "y", ir.var("i", "i64"))])
    k = _kernel([l1, l2])
    before = copy.deepcopy(k)
    out1 = optimize(k)
    assert k == before                       # optimize did not mutate input
    assert out1 == optimize(out1)            # idempotent
    assert len([s for s in out1["body"] if s["op"] == "for"]) == 1  # fused via optimize
```

- [ ] **Step 2: Verify it fails** — `/Users/bencrooks/celeris/.venv/bin/python -m pytest tests/test_passes.py -k "fuse or decline or readonly" -v`
Expected: FAIL — `ImportError: cannot import name 'fuse_loops'`.

- [ ] **Step 3: Implement** — add to `src/celeris/passes.py` (imports `copy` already present):
```python
def _has_return(stmts) -> bool:
    for s in stmts:
        op = s["op"]
        if op == "return":
            return True
        if op in ("for", "while") and _has_return(s["body"]):
            return True
        if op == "if" and (_has_return(s["then"]) or _has_return(s["else"])):
            return True
    return False


def _collect(stmts, loopvar):
    """Gather (written arrays, all array accesses, scalar writes, scalar reads)
    for a loop body. Scalars exclude the loop variable. Array accesses are
    (array_name, index_expr) pairs for every index node (read or write)."""
    arr_writes, arr_access, s_writes, s_reads = set(), [], set(), set()

    def read_expr(n):
        k = n.get("k")
        if k == "var":
            if n["name"] != loopvar:
                s_reads.add(n["name"])
        elif k == "index":
            arr_access.append((n["array"], n["index"]))
            read_expr(n["index"])
        elif k in ("binop", "cmp"):
            read_expr(n["lhs"]); read_expr(n["rhs"])
        elif k == "bool":
            for a in n["args"]:
                read_expr(a)
        elif k == "call":
            for a in n["args"]:
                read_expr(a)
        elif k == "cast":
            read_expr(n["value"])
        # const: leaf

    def write_lval(t):
        if t["k"] == "var":
            if t["name"] != loopvar:
                s_writes.add(t["name"])
        else:  # index
            arr_writes.add(t["array"])
            arr_access.append((t["array"], t["index"]))
            read_expr(t["index"])

    def visit(ss):
        for s in ss:
            op = s["op"]
            if op == "assign":
                write_lval(s["target"]); read_expr(s["value"])
            elif op == "augassign":
                write_lval(s["target"])
                if s["target"]["k"] == "var":   # augassign also READS the target
                    if s["target"]["name"] != loopvar:
                        s_reads.add(s["target"]["name"])
                else:
                    arr_access.append((s["target"]["array"], s["target"]["index"]))
                read_expr(s["value"])
            elif op == "for":
                read_expr(s["start"]); read_expr(s["stop"]); read_expr(s["step"])
                visit(s["body"])
            elif op == "while":
                read_expr(s["cond"]); visit(s["body"])
            elif op == "if":
                read_expr(s["cond"]); visit(s["then"]); visit(s["else"])
            elif op == "return" and s["value"] is not None:
                read_expr(s["value"])

    visit(stmts)
    return arr_writes, arr_access, s_writes, s_reads


def _can_fuse(f1, f2) -> bool:
    if f1.get("op") != "for" or f2.get("op") != "for":
        return False
    if not (f1["var"] == f2["var"] and f1["start"] == f2["start"]
            and f1["stop"] == f2["stop"] and f1["step"] == f2["step"]):
        return False                                  # (1) iteration space
    var = f1["var"]
    if _has_return(f1["body"]) or _has_return(f2["body"]):
        return False                                  # (3) no return
    w1, acc1, sw1, sr1 = _collect(f1["body"], var)
    w2, acc2, sw2, sr2 = _collect(f2["body"], var)
    written = w1 | w2
    for arr, idx in acc1 + acc2:                       # (4) written arrays index==i
        if arr in written and not (idx.get("k") == "var" and idx.get("name") == var):
            return False
    if (sw1 & (sw2 | sr2)) or (sw2 & (sw1 | sr1)):     # (5) scalar dependence
        return False
    return True


def _fuse_block(stmts):
    processed = [_fuse_stmt(s) for s in stmts]         # recurse into nested bodies first
    out = []
    for s in processed:
        if (out and out[-1].get("op") == "for" and s.get("op") == "for"
                and _can_fuse(out[-1], s)):
            merged = dict(out[-1])
            merged["body"] = _fuse_block(list(out[-1]["body"]) + list(s["body"]))
            out[-1] = merged
        else:
            out.append(s)
    return out


def _fuse_stmt(s):
    op = s.get("op")
    if op in ("for", "while"):
        s = dict(s); s["body"] = _fuse_block(s["body"]); return s
    if op == "if":
        s = dict(s); s["then"] = _fuse_block(s["then"]); s["else"] = _fuse_block(s["else"]); return s
    return s


def fuse_loops(kernel):
    """Fuse adjacent same-iteration-space ``for`` loops into one, when provably
    safe (see the legality predicate in the loop-fusion plan). Pure: never
    mutates ``kernel``. A no-op whenever no adjacent pair is fusable."""
    work = copy.deepcopy(kernel)
    work["body"] = _fuse_block(work["body"])
    return work
```
Then update `optimize()` to insert fusion between fold and DCE:
```python
def optimize(kernel):
    work = copy.deepcopy(kernel)
    work["body"] = fold_constants(work.get("body", []))
    work = fuse_loops(work)
    return eliminate_dead_code(work)
```

- [ ] **Step 4: Verify pass** — `/Users/bencrooks/celeris/.venv/bin/python -m pytest tests/test_passes.py -v` → all PASS. Then full suite `pytest -q` → all green (no regressions), and `ruff check .` → clean.

- [ ] **Step 5: Commit**
```bash
git add src/celeris/passes.py tests/test_passes.py
git commit -m "feat: provably-safe loop-fusion pass (fold -> fuse -> DCE) [plan:loop-fusion] [wave:1/task:1]"
```

---

## Wave 2 — Validate end-to-end + docs (parallel)

### Task 2.1: Differential validation of fused kernels
**Depends on:** Task 1.1
**Files:** Modify `tests/test_differential.py`

- [ ] **Step 1: Add a fusable multi-loop kernel + its runner to the harness.** Append a kernel whose Python form has TWO loops; the optimizer fuses them, and every backend must still match the plain-Python oracle:
```python
def chain(a: float, x: F64Array, t: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        t[i] = a * x[i]
    for i in range(n):
        y[i] = t[i] + 1.0

def _run_chain(f):
    x = np.arange(16, dtype=np.float64)
    t = np.zeros(16, dtype=np.float64)
    y = np.zeros(16, dtype=np.float64)
    f(2.0, x, t, y, 16)
    return y
```
Add `("chain", chain, _run_chain)` to the `CASES` list.

- [ ] **Step 2: Verify it fails first** — temporarily (mentally) the case is new; run `pytest tests/test_differential.py -k chain -v`. Expected once added: PASS (it should pass immediately because fusion is correct) — so to honor red-green, first confirm the harness *exercises fusion*: add an assertion in this test module that the optimized `chain` IR has a single fused loop:
```python
def test_chain_is_actually_fused():
    from celeris.parser import parse_function
    from celeris.passes import optimize
    k = optimize(parse_function(chain))
    fors = [s for s in k["body"] if s["op"] == "for"]
    assert len(fors) == 1 and len(fors[0]["body"]) == 2, "chain must fuse to one loop"
```
Run it BEFORE Task 1.1 exists → FAIL (no fusion); after Task 1.1 → PASS.

- [ ] **Step 3: Run** — `/Users/bencrooks/celeris/.venv/bin/python -m pytest tests/test_differential.py -v`
Expected: `chain` agrees across interpreter + sourcegen + llvm (the hardened asserts require sourcegen+llvm to compile it), proving the FUSED code is correct against the oracle.

- [ ] **Step 4: Full suite** — `pytest -q` → all green.

- [ ] **Step 5: Commit**
```bash
git add tests/test_differential.py
git commit -m "test: differential validation of fused multi-loop kernels [plan:loop-fusion] [wave:2/task:1]"
```

### Task 2.2: Docs + version bump
**Files:** Modify `README.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`, `pyproject.toml`

- [ ] **Step 1: Write the failing test** — append to `tests/test_docs_present.py`:
```python
def test_changelog_mentions_loop_fusion():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert "fusion" in (root / "CHANGELOG.md").read_text().lower()

def test_version_bumped():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert '0.2.0' in (root / "pyproject.toml").read_text()
```

- [ ] **Step 2: Verify fail** — `pytest tests/test_docs_present.py -k "fusion or version" -v` → FAIL.

- [ ] **Step 3: Implement**
  - `pyproject.toml`: `version = "0.2.0"`.
  - `CHANGELOG.md`: add `## [0.2.0] - 2026-05-26` with a "Loop-fusion optimization pass (adjacent same-range elementwise loops fuse into one pass; provably-safe legality predicate)" entry.
  - `README.md`: in the architecture/backend-tiers section, change the fusion line from a roadmap caveat to shipped — state celeris now fuses adjacent elementwise loops over the same range (the "one pass, no temporary" win), with a one-line note on the conservative safety rule.
  - `docs/ROADMAP.md`: move loop fusion out of the future/v1.0 list into shipped (note it landed in v0.2.0); keep tiling/parallelism/affine-offset-fusion as future.
  - `docs/ARCHITECTURE.md`: in the optimizer description, document the pass order `fold → fuse → DCE` and the loop-fusion legality predicate (1-line summary).

- [ ] **Step 4: Verify pass** — `pytest tests/test_docs_present.py -v` → all PASS.

- [ ] **Step 5: Commit**
```bash
git add README.md docs/ROADMAP.md docs/ARCHITECTURE.md CHANGELOG.md pyproject.toml tests/test_docs_present.py
git commit -m "docs: move loop fusion roadmap->shipped; bump to 0.2.0 [plan:loop-fusion] [wave:2/task:2]"
```

---

## Wave 3 — Ship

### Task 3.1: PR, CI, merge, release
**Depends on:** Waves 1-2
- [ ] **Step 1:** Final local gate: `/Users/bencrooks/celeris/.venv/bin/python -m pytest -q` (all green) and `ruff check .` (clean).
- [ ] **Step 2:** Push the branch: `git push -u origin feat/loop-fusion`.
- [ ] **Step 3:** Open PR: `gh pr create --title "feat: loop-fusion optimization pass" --body "<summary of the pass + legality predicate + differential validation>"`.
- [ ] **Step 4:** Wait for CI green on the PR (`gh pr checks --watch` / `gh run watch`). If red, diagnose and fix (commit + push) before merging.
- [ ] **Step 5:** Merge: `gh pr merge --squash --delete-branch` (or `--merge`). Then on `main`: tag `git tag -a v0.2.0 -m "celeris v0.2.0 — loop fusion" && git push origin v0.2.0` and `gh release create v0.2.0 --title "celeris v0.2.0" --notes-file -` (from CHANGELOG [0.2.0]).

---

## Verification
After all waves:
- [ ] `pytest -q` all green; `ruff check .` clean
- [ ] Differential harness: `chain` fused-kernel agrees with the Python oracle across interpreter + sourcegen + llvm
- [ ] `optimize(parse_function(chain))` yields a single fused loop; negative cases (offset write, shared accumulator, return-in-body, different range) remain unfused
- [ ] CI green on the PR; merged to `main`; `v0.2.0` released
- [ ] No backend files changed (fused IR is a normal `for` loop)

## Out of scope (future)
Affine-offset dependence analysis (Approach B), loop tiling/blocking, parallelization (`prange`). Tracked in ROADMAP.
