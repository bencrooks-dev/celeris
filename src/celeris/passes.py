"""celeris optimisation passes — constant folding and dead-code elimination.

These passes operate on the canonical IR dicts produced by :mod:`celeris.ir`.
Every pass is *pure*: it never mutates its input and always returns a freshly
built tree, so callers can re-run a pass (idempotency) or compose passes freely.

* :func:`fold_constants` — recursively evaluate ``binop`` / ``cmp`` nodes whose
  operands are all ``const``, collapsing them to a single ``const``.
* :func:`eliminate_dead_code` — drop ``assign`` statements to a local variable
  whose name is never read anywhere in the kernel.
* :func:`optimize` — run folding then DCE over a whole kernel.
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


def optimize(kernel):
    """Fold constants then eliminate dead code over ``kernel``.

    Pure and idempotent: never mutates ``kernel`` and re-running ``optimize`` on
    its own output yields an equal kernel.
    """
    work = copy.deepcopy(kernel)
    work["body"] = fold_constants(work.get("body", []))
    return eliminate_dead_code(work)
