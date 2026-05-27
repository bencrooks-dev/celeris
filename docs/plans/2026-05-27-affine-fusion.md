# Affine-Offset Loop Fusion Plan (v0.3.0)

> **Plan ID:** `plan:affine-fusion`
> **For agentic workers:** Use `/athena-build` to execute wave-by-wave. TDD per task. Branch: `feat/affine-fusion`.

**Goal:** Generalize the loop-fusion legality check from "written-array subscripts must be exactly the loop var" to **constant affine offsets** `i ± c` (c an integer literal), via a provably-safe dependence test — a strict superset of the v0.2 rule.

**Approach:** In `src/celeris/passes.py`, add `_affine_offset` (parse an index expr to a constant offset relative to the loop var), track a per-access write flag in `_collect`, and rewrite `_can_fuse` condition (4): when the loop step is a positive integer literal, for each written array require `cy ≤ cx` for every cross-loop access pair (L1 offset `cx`, L2 offset `cy`) with ≥1 write; otherwise fall back to the strict exactly-`i` rule. Read-only arrays stay unrestricted. No backend changes — a fused loop is still a normal `for`.

**Tech Stack:** Python 3.10+, existing IR, pytest, ruff.

**Files:**
- Modify: `src/celeris/passes.py` — `_affine_offset` (new), `_collect` (add write flag), `_can_fuse` (affine condition 4)
- Test: `tests/test_passes.py` — affine fuse-positives + decline cases; existing tests stay green
- Test: `tests/test_differential.py` — OOB-free affine stencil chain (`shifted_chain`)
- Modify: `README.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`, `pyproject.toml` (→ 0.3.0)

---

## Wave 1 — Affine legality

### Task 1.1: Affine-offset fusion in `passes.py` + unit tests
**Files:** Modify `src/celeris/passes.py`, `tests/test_passes.py`

- [ ] **Step 1: Write failing tests** — append to `tests/test_passes.py` (reuse the existing `_loop`/`_kernel` helpers added in the loop-fusion work; if absent, define locally as in the loop-fusion plan):
```python
def _wr(dst, idx_expr):
    return ir.assign(ir.lval_index(dst, idx_expr, "f64"), ir.index("x", ir.var("i", "i64"), "f64"))

def test_affine_backward_offset_fuses():
    # L1: t[i]=x[i] ; L2: y[i]=t[i]+t[i] but read t at offset 0 only -> already fuses;
    # exercise true affine: L1 writes t[i+1] (cx=1), L2 reads t[i] (cy=0) -> 0<=1 fuse
    off1 = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("t", off1, "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.index("t", ir.var("i", "i64"), "f64"))])
    k = _kernel([l1, l2]); k["params"].append(ir.param("t", {"ptr": "f64"}))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 1

def test_affine_forward_offset_read_declines():
    # L1 writes t[i] (cx=0) ; L2 reads t[i+1] (cy=1) -> 1<=0 false -> decline
    off1 = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("t", ir.var("i", "i64"), "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.index("t", off1, "f64"))])
    k = _kernel([l1, l2]); k["params"].append(ir.param("t", {"ptr": "f64"}))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_affine_variable_offset_declines():
    # t[i+k] where k is a variable -> non-affine -> decline
    offk = ir.binop("+", "i64", ir.var("i", "i64"), ir.var("k", "i64"))
    l1 = _loop([ir.assign(ir.lval_index("t", offk, "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.index("t", ir.var("i", "i64"), "f64"))])
    k = _kernel([l1, l2])
    for p in ("t", "k"):
        k["params"].append(ir.param(p, {"ptr": "f64"} if p == "t" else "i64"))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_affine_nonunit_step_with_offset_declines():
    # step=2 (non-unit) with a written-array offset -> strict fallback -> decline
    off1 = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    def l2step(body):
        return ir.for_("i", ir.const("i64", 0), ir.var("n", "i64"), ir.const("i64", 2), body)
    l1 = l2step([ir.assign(ir.lval_index("t", off1, "f64"),
                           ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = l2step([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                           ir.index("t", ir.var("i", "i64"), "f64"))])
    k = _kernel([l1, l2]); k["params"].append(ir.param("t", {"ptr": "f64"}))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_affine_output_dep_offset_fuses():
    # L1 writes t[i+1] (cx=1), L2 writes t[i] (cy=0) -> 0<=1 fuse (output dep ok)
    off1 = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("t", off1, "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([ir.assign(ir.lval_index("t", ir.var("i", "i64"), "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    k = _kernel([l1, l2]); k["params"].append(ir.param("t", {"ptr": "f64"}))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 1
```
(Existing v0.2 fusion tests — `test_fuse_adjacent_elementwise`, the decline cases, etc. — MUST remain unchanged and still pass, proving subsumption.)

- [ ] **Step 2: Verify fail** — `/Users/bencrooks/celeris/.venv/bin/python -m pytest tests/test_passes.py -k affine -v` → FAIL (current code declines all offsets, so the two *fuse* tests fail).

- [ ] **Step 3: Implement** in `src/celeris/passes.py`:
  (a) Add the affine-offset parser:
```python
def _affine_offset(idx, loopvar):
    """Constant offset c if idx is exactly `loopvar` (0) or `loopvar ± int-literal`
    (±c); None for any non-affine subscript (2*i, i+k, i%2, nested index, ...)."""
    if idx.get("k") == "var" and idx.get("name") == loopvar:
        return 0
    if idx.get("k") == "binop" and idx.get("op") in ("+", "-"):
        lhs, rhs = idx["lhs"], idx["rhs"]
        if (lhs.get("k") == "var" and lhs.get("name") == loopvar
                and rhs.get("k") == "const" and isinstance(rhs.get("value"), int)):
            c = int(rhs["value"])
            return c if idx["op"] == "+" else -c
        if (idx["op"] == "+" and rhs.get("k") == "var" and rhs.get("name") == loopvar
                and lhs.get("k") == "const" and isinstance(lhs.get("value"), int)):
            return int(lhs["value"])
    return None
```
  (b) Change `_collect` so array accesses carry a write flag — its accesses list becomes `(array_name, index_expr, is_write)`:
  - `write_lval` index target → append `(array, index, True)`.
  - `read_expr` index → append `(array, index, False)`.
  - augassign index target → append `(array, index, True)` (covers both read+write at the same offset; the write flag is the safe choice).
  Update the return so the accesses list is the 3-tuple form. (The `arr_writes` set, `s_writes`, `s_reads` are unchanged.)
  (c) Rewrite `_can_fuse` condition (4) (keep (1) iter-space, (2) handled by `_fuse_block`, (3) no-return, (5) scalar-dep unchanged):
```python
    w1, acc1, sw1, sr1 = _collect(f1["body"], var)
    w2, acc2, sw2, sr2 = _collect(f2["body"], var)
    written = w1 | w2
    step = f1["step"]
    step_pos_const = (step.get("k") == "const"
                      and isinstance(step.get("value"), int) and step["value"] > 0)
    if step_pos_const:
        for arr in written:
            l1 = [(_affine_offset(idx, var), iw) for (a, idx, iw) in acc1 if a == arr]
            l2 = [(_affine_offset(idx, var), iw) for (a, idx, iw) in acc2 if a == arr]
            if any(off is None for off, _ in l1 + l2):
                return False                      # non-affine access to a written array
            for cx, wx in l1:
                for cy, wy in l2:
                    if (wx or wy) and not (cy <= cx):
                        return False              # would reorder a real dependence
    else:
        for a, idx, iw in acc1 + acc2:            # non-unit/var step: strict exactly-i
            if a in written and not (idx.get("k") == "var" and idx.get("name") == var):
                return False
    if (sw1 & (sw2 | sr2)) or (sw2 & (sw1 | sr1)):
        return False                              # (5) scalar dependence
    return True
```
  Document the `cy ≤ cx` derivation in `_can_fuse`'s docstring (each written element is written once under the injective affine map; `cy ≤ cx` is exactly the condition that the fused interleaving preserves the unfused dependence order — covers flow, anti, and output deps).

- [ ] **Step 4: Verify pass** — `pytest tests/test_passes.py -v` → all PASS (new affine + all existing). Full suite `pytest -q` → green (subsumption: existing differential `chain` still fuses). `ruff check .` → clean.

- [ ] **Step 5: Commit**
```bash
git add src/celeris/passes.py tests/test_passes.py
git commit -m "feat: affine-offset loop fusion (cy<=cx dependence test) [plan:affine-fusion] [wave:1/task:1]"
```

---

## Wave 2 — Differential + docs (parallel)

### Task 2.1: OOB-free affine differential kernel
**Depends on:** Task 1.1
**Files:** Modify `tests/test_differential.py`

- [ ] **Step 1: Add an in-bounds affine stencil chain to the harness.** L1 writes `t[i+1]` (cx=1), L2 reads `t[i]` (cy=0) → fuses; `t` sized `n+1` keeps every index valid:
```python
def shifted_chain(a: float, x: F64Array, t: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        t[i + 1] = a * x[i]
    for i in range(n):
        y[i] = t[i]

def _run_shifted_chain(f):
    x = np.arange(16, dtype=np.float64)
    t = np.zeros(17, dtype=np.float64)   # size n+1 so t[i+1] and t[i] are in bounds
    y = np.zeros(16, dtype=np.float64)
    f(2.0, x, t, y, 16)
    return y
```
Append `("shifted_chain", shifted_chain, _run_shifted_chain)` to `CASES`.

- [ ] **Step 2: Add a fusion-happened assertion** (red before Task 1.1, green after):
```python
def test_shifted_chain_is_fused():
    from celeris.parser import parse_function
    from celeris.passes import optimize
    k = optimize(parse_function(shifted_chain))
    fors = [s for s in k["body"] if s["op"] == "for"]
    assert len(fors) == 1 and len(fors[0]["body"]) == 2
```

- [ ] **Step 3: Run** — `pytest tests/test_differential.py -v` → `shifted_chain` fuses AND matches the plain-Python oracle across interpreter + sourcegen + llvm (the hardened asserts require sourcegen+llvm to compile it). If any backend disagrees, that's a real bug → report BLOCKED, do not weaken.

- [ ] **Step 4: Full suite** — `pytest -q` → green.

- [ ] **Step 5: Commit**
```bash
git add tests/test_differential.py
git commit -m "test: in-bounds affine stencil fusion differential case [plan:affine-fusion] [wave:2/task:1]"
```

### Task 2.2: Docs + version bump
**Files:** Modify `README.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`, `pyproject.toml`, `tests/test_docs_present.py`

- [ ] **Step 1: Failing test** — append to `tests/test_docs_present.py`:
```python
def test_version_is_030():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert '0.3.0' in (root / "pyproject.toml").read_text()

def test_changelog_mentions_affine():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert "affine" in (root / "CHANGELOG.md").read_text().lower()
```

- [ ] **Step 2: Verify fail** — `pytest tests/test_docs_present.py -k "030 or affine" -v` → FAIL.

- [ ] **Step 3: Implement**
  - `pyproject.toml`: `version = "0.3.0"`.
  - `CHANGELOG.md`: `## [0.3.0] - 2026-05-27` — "Loop fusion now handles constant affine offsets (`a[i±c]`) via a `cy ≤ cx` dependence test; variable offsets, non-unit step, and broadcasting remain future."
  - `README.md` + `docs/ARCHITECTURE.md`: update the fusion description — fusion now fuses adjacent loops with constant affine offsets (still conservative/provably-safe; declines variable offsets, non-unit-step offset cases, forward-read deps). Cross-check against `_can_fuse`.
  - `docs/ROADMAP.md`: note affine-offset fusion landed in v0.3.0; keep variable-offset/non-unit-step fusion, tiling, parallelism as future.

- [ ] **Step 4: Verify pass** — `pytest tests/test_docs_present.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add README.md docs/ROADMAP.md docs/ARCHITECTURE.md CHANGELOG.md pyproject.toml tests/test_docs_present.py
git commit -m "docs: affine fusion shipped; bump to 0.3.0 [plan:affine-fusion] [wave:2/task:2]"
```

---

## Wave 3 — Ship

### Task 3.1: PR, CI, merge, release
- [ ] **Step 1:** `/Users/bencrooks/celeris/.venv/bin/python -m pytest -q` green; `ruff check .` clean.
- [ ] **Step 2:** `git push -u origin feat/affine-fusion`.
- [ ] **Step 3:** `gh pr create --title "feat: affine-offset loop fusion (v0.3.0)" --body "<summary: cy<=cx rule, subsumes exactly-i, differential-validated>"`.
- [ ] **Step 4:** Wait for CI green (`gh pr checks <n> --watch`). Fix + push if red.
- [ ] **Step 5:** `gh pr merge <n> --squash --delete-branch`; then on `main`: `git tag -a v0.3.0 -m "celeris v0.3.0 — affine-offset fusion" && git push origin v0.3.0`; `gh release create v0.3.0 --title "celeris v0.3.0" --notes-file -` (from CHANGELOG [0.3.0]).

---

## Verification
- [ ] `pytest -q` green; `ruff check .` clean
- [ ] All existing v0.2 fusion tests still pass (subsumption proven)
- [ ] New affine fuse-positives fuse; decline cases (forward read, variable offset, non-unit step) stay unfused
- [ ] `shifted_chain` fuses AND matches the Python oracle across interpreter + sourcegen + llvm
- [ ] CI green on PR; merged to `main`; `v0.3.0` released

## Out of scope (future)
Variable offsets (`i+k`), non-unit/negative-step offset fusion, multi-array broadcasting, tiling. Tracked in ROADMAP. (Next features: `prange` v0.4.0, tensor model v0.5.0.)
