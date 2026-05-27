"""Pure-Python IR interpreter backend — the portable reference implementation.

Walks the typed IR and executes it in Python. It provides no speedup; its
purpose is correctness (it is the oracle the differential tests check against)
and portability (it runs anywhere, needing no compiler/LLVM).
"""
from __future__ import annotations

import math

from . import register

_INTRINSICS = {
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "sin": math.sin,
    "cos": math.cos, "fabs": math.fabs, "floor": math.floor,
    "fmax": max, "fmin": min, "len": len,
}


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class InterpreterBackend:
    """Executes typed IR directly in Python. Always available."""

    name = "interpreter"

    def available(self) -> bool:
        return True

    def compile(self, ir: dict):
        params = [p["name"] for p in ir["params"]]
        body = ir["body"]

        def run(*args):
            env = dict(zip(params, args))
            try:
                _exec_block(body, env)
            except _Return as r:
                return r.value
            return None

        return run


def _exec_block(stmts, env):
    for s in stmts:
        _exec_stmt(s, env)


def _exec_stmt(s, env):
    op = s["op"]
    if op == "assign":
        _store(s["target"], _eval(s["value"], env), env)
    elif op == "augassign":
        cur = _load_lval(s["target"], env)
        _store(s["target"], _apply(s["binop"], cur, _eval(s["value"], env)), env)
    elif op == "for":
        start = _eval(s["start"], env)
        stop = _eval(s["stop"], env)
        step = _eval(s["step"], env)
        if step == 0:
            raise ValueError("range() arg 3 must not be zero")
        i = start
        while (step > 0 and i < stop) or (step < 0 and i > stop):
            env[s["var"]] = i
            _exec_block(s["body"], env)
            i += step
    elif op == "while":
        while bool(_eval(s["cond"], env)):
            _exec_block(s["body"], env)
    elif op == "if":
        if bool(_eval(s["cond"], env)):
            _exec_block(s["then"], env)
        else:
            _exec_block(s["else"], env)
    elif op == "return":
        raise _Return(_eval(s["value"], env) if s["value"] is not None else None)
    else:
        raise ValueError(f"interpreter: unknown statement op '{op}'")


def _store(target, value, env):
    if target["k"] == "var":
        env[target["name"]] = value
    elif target["k"] == "index":
        if "indices" in target:
            env[target["array"]][tuple(_eval(ix, env) for ix in target["indices"])] = value
        else:
            env[target["array"]][_eval(target["index"], env)] = value
    else:
        raise ValueError(f"interpreter: bad store target '{target['k']}'")


def _load_lval(target, env):
    if target["k"] == "var":
        return env[target["name"]]
    if target["k"] == "index":
        if "indices" in target:
            return env[target["array"]][tuple(_eval(ix, env) for ix in target["indices"])]
        return env[target["array"]][_eval(target["index"], env)]
    raise ValueError(f"interpreter: bad lvalue '{target['k']}'")


def _eval(e, env):
    k = e["k"]
    if k == "const":
        return e["value"]
    if k == "var":
        return env[e["name"]]
    if k == "index":
        if "indices" in e:
            return env[e["array"]][tuple(_eval(ix, env) for ix in e["indices"])]
        return env[e["array"]][_eval(e["index"], env)]
    if k == "binop":
        return _apply(e["op"], _eval(e["lhs"], env), _eval(e["rhs"], env))
    if k == "cmp":
        return _compare(e["op"], _eval(e["lhs"], env), _eval(e["rhs"], env))
    if k == "bool":
        return _boolop(e["op"], [_eval(a, env) for a in e["args"]])
    if k == "call":
        return _INTRINSICS[e["fn"]](*[_eval(a, env) for a in e["args"]])
    if k == "cast":
        return _cast(e["type"], _eval(e["value"], env))
    raise ValueError(f"interpreter: unknown expr kind '{k}'")


def _apply(op, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    if op == "//":
        return a // b
    if op == "%":
        return a % b
    if op == "**":
        return a ** b
    raise ValueError(f"interpreter: unknown binop '{op}'")


def _compare(op, a, b):
    # Comparisons yield an int 0/1 to match the native backends' i-typed truth
    # value (the while/if bool(...) wrappers still work on ints).
    return int({"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b,
                "==": a == b, "!=": a != b}[op])


def _boolop(op, args):
    # Boolean ops yield an int 0/1 to match the native backends' representation.
    if op == "not":
        return int(not args[0])
    if op == "and":
        return int(all(args))
    if op == "or":
        return int(any(args))
    raise ValueError(f"interpreter: unknown boolop '{op}'")


def _cast(t, v):
    if t in ("i32", "i64"):
        return int(v)
    if t in ("f32", "f64"):
        return float(v)
    return v


register(InterpreterBackend())
