"""celeris optimisation passes — constant folding, loop fusion and dead-code
elimination.

These passes operate on the canonical IR dicts produced by :mod:`celeris.ir`.
Every pass is *pure*: it never mutates its input and always returns a freshly
built tree, so callers can re-run a pass (idempotency) or compose passes freely.

* :func:`fold_constants` — recursively evaluate ``binop`` / ``cmp`` nodes whose
  operands are all ``const``, collapsing them to a single ``const``.
* :func:`fuse_loops` — merge adjacent ``for`` loops over the same iteration
  space into one body when a conservative legality predicate proves it safe.
* :func:`eliminate_dead_code` — drop ``assign`` statements to a local variable
  whose name is never read anywhere in the kernel.
* :func:`optimize` — run folding, then fusion, then DCE over a whole kernel.
"""

import copy

# Arithmetic operators usable inside a folded ``binop``. Division and modulo are
# handled specially (zero guard + int/float result coercion).
_ARITH = {"+", "-", "*", "/", "//", "%", "**"}

# Comparison operators usable inside a folded ``cmp`` (always yields an i1 0/1).
_CMP = {"<", "<=", ">", ">=", "==", "!="}


def _is_int_type(type):
    """True if ``type`` names an integer scalar (``i1``, ``i32``, ``i64`` ...)."""
    return isinstance(type, str) and type.startswith("i")


def _eval_arith(op, type, lhs, rhs):
    """Evaluate ``lhs op rhs`` for a ``binop``; return value or ``None`` to skip.

    Returns ``None`` (folding is skipped) for division/modulo by zero so the
    original node is preserved untouched.
    """
    if op in ("/", "//", "%") and rhs == 0:
        return None
    if op == "+":
        result = lhs + rhs
    elif op == "-":
        result = lhs - rhs
    elif op == "*":
        result = lhs * rhs
    elif op == "/":
        result = lhs / rhs
    elif op == "//":
        result = lhs // rhs
    elif op == "%":
        result = lhs % rhs
    elif op == "**":
        result = lhs ** rhs
    else:  # pragma: no cover - guarded by caller
        return None
    # The node's declared type drives the const type: int types coerce to int,
    # everything else (floats) coerce to float.
    if _is_int_type(type):
        return int(result)
    return float(result)


def _eval_cmp(op, lhs, rhs):
    """Evaluate a comparison ``lhs op rhs`` returning ``1``/``0`` (i1)."""
    if op == "<":
        result = lhs < rhs
    elif op == "<=":
        result = lhs <= rhs
    elif op == ">":
        result = lhs > rhs
    elif op == ">=":
        result = lhs >= rhs
    elif op == "==":
        result = lhs == rhs
    elif op == "!=":
        result = lhs != rhs
    else:  # pragma: no cover - guarded by caller
        return None
    return 1 if result else 0


def fold_constants(node):
    """Recursively fold constant expressions, returning a fresh node.

    Children are folded first; a ``binop``/``cmp`` whose ``lhs`` and ``rhs`` both
    fold to ``const`` is collapsed to a single ``const`` of the node's type
    (``i1`` for ``cmp``). Non-foldable nodes are returned structurally intact
    with their children folded. The input is never mutated.
    """
    if isinstance(node, list):
        return [fold_constants(child) for child in node]
    if not isinstance(node, dict):
        return node

    # Fold every value field first, leaving plain scalars (names, ops, raw
    # ``const`` values, the ``array`` slot of an index when it is a string) alone.
    folded = {}
    for key, value in node.items():
        if isinstance(value, (dict, list)):
            folded[key] = fold_constants(value)
        else:
            folded[key] = value

    kind = folded.get("k")

    if kind == "binop" and folded.get("op") in _ARITH:
        lhs, rhs = folded.get("lhs"), folded.get("rhs")
        if _is_const(lhs) and _is_const(rhs):
            value = _eval_arith(folded["op"], folded["type"], lhs["value"], rhs["value"])
            if value is not None:
                return {"k": "const", "type": folded["type"], "value": value}

    elif kind == "cmp" and folded.get("op") in _CMP:
        lhs, rhs = folded.get("lhs"), folded.get("rhs")
        if _is_const(lhs) and _is_const(rhs):
            value = _eval_cmp(folded["op"], lhs["value"], rhs["value"])
            if value is not None:
                return {"k": "const", "type": "i1", "value": value}

    return folded


def _is_const(node):
    """True if ``node`` is a ``const`` expression dict."""
    return isinstance(node, dict) and node.get("k") == "const"


def _collect_reads(node, reads):
    """Accumulate the names of every variable *read* anywhere in ``node``.

    A read is any ``{"k": "var", "name": ...}`` appearing in an expression
    position. Assignment targets are l-values, not reads, so the ``target`` of
    an ``assign`` is skipped (its name is only counted if read elsewhere); but
    an ``index`` target's ``index`` sub-expression *is* scanned, and so is the
    ``value`` of every statement.
    """
    if isinstance(node, list):
        for child in node:
            _collect_reads(child, reads)
        return
    if not isinstance(node, dict):
        return

    if node.get("op") == "assign":
        # The target is a write, not a read; but an index target's subscript
        # expression and the assigned value are both reads.
        target = node.get("target")
        if isinstance(target, dict) and target.get("k") == "index":
            _collect_reads(target.get("index"), reads)
            _collect_reads(target.get("array"), reads)
        _collect_reads(node.get("value"), reads)
        return

    if node.get("k") == "var":
        reads.add(node.get("name"))
        return

    for key, value in node.items():
        if isinstance(value, (dict, list)):
            _collect_reads(value, reads)


def _strip_dead(body, reads):
    """Return a copy of statement list ``body`` with dead var-assigns removed.

    Recurses into nested ``for``/``while``/``if`` bodies. Only drops an
    ``assign`` whose target is ``{"k": "var", "name": N}`` where ``N`` is read
    nowhere in the kernel; index stores, augassigns and everything else are
    preserved.
    """
    result = []
    for stmt in body:
        if isinstance(stmt, dict) and stmt.get("op") == "assign":
            target = stmt.get("target")
            if (
                isinstance(target, dict)
                and target.get("k") == "var"
                and target.get("name") not in reads
            ):
                continue  # dead: target var is never read anywhere
            result.append(stmt)
            continue

        if isinstance(stmt, dict) and stmt.get("op") in ("for", "while"):
            new_stmt = dict(stmt)
            new_stmt["body"] = _strip_dead(stmt.get("body", []), reads)
            result.append(new_stmt)
            continue

        if isinstance(stmt, dict) and stmt.get("op") == "if":
            new_stmt = dict(stmt)
            new_stmt["then"] = _strip_dead(stmt.get("then", []), reads)
            new_stmt["else"] = _strip_dead(stmt.get("else", []), reads)
            result.append(new_stmt)
            continue

        result.append(stmt)
    return result


def eliminate_dead_code(kernel):
    """Drop ``assign`` statements to a local var that is never read.

    Conservative whole-kernel liveness: a variable read is any ``var`` node in
    an expression position (loop bounds, conditions, returns, index subscripts,
    RHS of assigns). An ``assign`` to ``{"k": "var", "name": N}`` is dead iff
    ``N`` has zero reads anywhere in the kernel. Index stores, augassigns and
    calls are never removed. Returns a fresh kernel; the input is not mutated.
    """
    work = copy.deepcopy(kernel)
    reads = set()
    _collect_reads(work.get("body", []), reads)
    work["body"] = _strip_dead(work.get("body", []), reads)
    return work


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


def _collect(stmts, loopvar):
    """Gather (written arrays, all array accesses, scalar writes, scalar reads)
    for a loop body. Scalars exclude the loop variable. Array accesses are
    (array_name, index_expr, is_write) triples for every index node: the write
    flag is True for store targets (index l-values and augassign index targets),
    False for index reads."""
    arr_writes, arr_access, s_writes, s_reads = set(), [], set(), set()

    def read_expr(n):
        k = n.get("k")
        if k == "var":
            if n["name"] != loopvar:
                s_reads.add(n["name"])
        elif k == "index":
            arr_access.append((n["array"], n["index"], False))
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
            arr_access.append((t["array"], t["index"], True))
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
                    arr_access.append((s["target"]["array"], s["target"]["index"], True))
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
    """Conservative legality predicate for fusing two adjacent ``for`` loops.

    Condition (4) — affine-offset dependence test. When the loop step is the
    unit positive literal (``step == 1``), every subscript of a *written* array
    must be a constant affine offset of the loop var (``i ± c``). With unit
    stride the map ``i -> i + c`` is injective and contiguous, so each iteration
    writes a distinct element and the iteration sweep reaches every integer in
    range. For a written array, take any cross-loop access pair — L1 at offset
    ``cx``, L2 at offset ``cy`` — with at least one write among them. In the
    unfused program L1's whole sweep precedes L2's, so the dependence (flow /
    anti / output) between element ``cx`` of L1's iteration and element ``cy``
    of L2's runs in source order. Fusion interleaves them: within one fused
    iteration ``i`` the L1 statement (touching ``i + cx``) runs before the L2
    statement (touching ``i + cy``); across iterations ``i`` increases by one.
    The fused order preserves the unfused dependence order exactly when the L2
    access touches an element that L1 has already reached, i.e.
    ``i + cy <= i + cx`` ⇔ ``cy <= cx`` for every such pair. Read-only arrays
    carry no cross-loop dependence and are unrestricted. A non-affine subscript
    of a written array is undecidable here, so we decline. When the step is not
    the unit positive literal the contiguity premise of the derivation fails, so
    we fall back to the strict exactly-``i`` rule (declining any written-array
    offset).
    """
    if f1.get("op") != "for" or f2.get("op") != "for":
        return False
    if not (f1["var"] == f2["var"] and f1["start"] == f2["start"]
            and f1["stop"] == f2["stop"] and f1["step"] == f2["step"]):
        return False                                  # (1) iteration space
    if f1.get("parallel", False) != f2.get("parallel", False):
        return False                                  # (1) matching parallel flag
    var = f1["var"]
    if _has_return(f1["body"]) or _has_return(f2["body"]):
        return False                                  # (3) no return
    w1, acc1, sw1, sr1 = _collect(f1["body"], var)
    w2, acc2, sw2, sr2 = _collect(f2["body"], var)
    written = w1 | w2
    step = f1["step"]
    step_unit_const = (step.get("k") == "const"
                       and isinstance(step.get("value"), int) and step["value"] == 1)
    if step_unit_const:
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


def optimize(kernel):
    """Fold constants, fuse adjacent loops, then eliminate dead code over ``kernel``.

    Pure and idempotent: never mutates ``kernel`` and re-running ``optimize`` on
    its own output yields an equal kernel.
    """
    work = copy.deepcopy(kernel)
    work["body"] = fold_constants(work.get("body", []))
    work = fuse_loops(work)
    return eliminate_dead_code(work)
