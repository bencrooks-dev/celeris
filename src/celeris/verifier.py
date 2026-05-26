"""celeris IR verifier — the trust boundary between front-end and back-ends.

The verifier independently re-checks the structure and types of an IR kernel
so that a corrupt or hand-crafted blob is rejected *before* any backend lowers
or executes it. It deliberately trusts nothing emitted by the parser: every
node shape, type annotation, and symbol reference is validated from scratch.

On the first violation :func:`verify_ir` raises
:class:`~celeris.errors.VerifyError` with a message naming the offending
construct; a clean kernel returns ``None``.
"""

from .errors import VerifyError
from .types import is_int, unify_numeric

#: Scalar type vocabulary the IR may carry. ``"void"`` marks the absence of a
#: value and is *not* a valid value/element type.
_SCALARS = {"i32", "i64", "f32", "f64", "i1", "void"}

#: Intrinsic functions the verifier accepts, mapped to their arity.
_INTRINSIC_ARITY = {
    "sqrt": 1, "exp": 1, "log": 1, "sin": 1, "cos": 1,
    "fabs": 1, "floor": 1, "fmax": 2, "fmin": 2, "len": 1,
}


# --- type predicates -----------------------------------------------------------

def _is_scalar_type(t) -> bool:
    """True if ``t`` is a known scalar type string (includes ``"void"``)."""
    return isinstance(t, str) and t in _SCALARS


def _is_int_type(t) -> bool:
    """True if ``t`` is an integer scalar (``"i32"`` or ``"i64"``)."""
    return is_int(t)


def _is_ptr(t) -> bool:
    """True if ``t`` is a pointer type ``{"ptr": <scalar>}``."""
    return isinstance(t, dict) and set(t.keys()) == {"ptr"}


def _is_value_type(t) -> bool:
    """True if ``t`` is a legal type for a value/element (scalar, non-void)."""
    return _is_scalar_type(t) and t != "void"


def _is_valid_type(t) -> bool:
    """True if ``t`` is any structurally valid type (scalar or pointer)."""
    if _is_scalar_type(t):
        return True
    if _is_ptr(t):
        return _is_value_type(t["ptr"])
    return False


# --- entry point ---------------------------------------------------------------

def verify_ir(k) -> None:
    """Verify IR kernel ``k``; raise :class:`VerifyError` on any violation."""
    if not isinstance(k, dict):
        raise VerifyError(f"kernel must be a dict, got {type(k).__name__}")

    for field in ("name", "params", "ret", "body"):
        if field not in k:
            raise VerifyError(f"kernel missing required field '{field}'")

    if not isinstance(k["name"], str):
        raise VerifyError("kernel 'name' must be a string")
    if not isinstance(k["params"], list):
        raise VerifyError("kernel 'params' must be a list")
    if not isinstance(k["body"], list):
        raise VerifyError("kernel 'body' must be a list")
    if not _is_valid_type(k["ret"]):
        raise VerifyError(f"kernel 'ret' has invalid type {k['ret']!r}")

    # Build the initial symbol table from parameters.
    syms = {}
    for p in k["params"]:
        if not isinstance(p, dict):
            raise VerifyError("each param must be a dict")
        if "name" not in p or not isinstance(p["name"], str):
            raise VerifyError("param missing a string 'name'")
        if "type" not in p:
            raise VerifyError(f"param '{p['name']}' missing 'type'")
        if not _is_valid_type(p["type"]):
            raise VerifyError(f"param '{p['name']}' has invalid type {p['type']!r}")
        syms[p["name"]] = p["type"]

    _verify_body(k["body"], syms)


# --- statement verification ----------------------------------------------------

def _verify_body(body, syms) -> None:
    """Verify a list of statements ``body`` against symbol table ``syms``."""
    if not isinstance(body, list):
        raise VerifyError("statement body must be a list")
    for stmt in body:
        _verify_stmt(stmt, syms)


def _verify_stmt(s, syms) -> None:
    """Verify a single statement ``s``, mutating ``syms`` for new locals."""
    if not isinstance(s, dict) or "op" not in s:
        raise VerifyError(f"statement must be a dict with 'op', got {s!r}")
    op = s["op"]

    if op == "assign":
        _verify_assign(s, syms)
    elif op == "augassign":
        _verify_augassign(s, syms)
    elif op == "for":
        _verify_for(s, syms)
    elif op == "while":
        _verify_while(s, syms)
    elif op == "if":
        _verify_if(s, syms)
    elif op == "return":
        _verify_return(s, syms)
    else:
        raise VerifyError(f"unknown statement op '{op}'")


def _verify_assign(s, syms) -> None:
    if "target" not in s or "value" not in s:
        raise VerifyError("assign missing 'target' or 'value'")
    _verify_expr(s["value"], syms)
    _verify_lvalue(s["target"], syms)


def _verify_augassign(s, syms) -> None:
    if "binop" not in s or "target" not in s or "value" not in s:
        raise VerifyError("augassign missing 'binop'/'target'/'value'")
    _verify_expr(s["value"], syms)
    _verify_lvalue(s["target"], syms)


def _verify_for(s, syms) -> None:
    if not isinstance(s.get("var"), str):
        raise VerifyError("for loop 'var' must be a string")
    for bound in ("start", "stop", "step"):
        if bound not in s:
            raise VerifyError(f"for loop missing '{bound}'")
        t = _verify_expr(s[bound], syms)
        if not _is_int_type(t):
            raise VerifyError(f"for loop '{bound}' must be int-typed, got {t!r}")
    # The loop variable is i64 within the body scope.
    inner = dict(syms)
    inner[s["var"]] = "i64"
    _verify_body(s.get("body", []), inner)


def _verify_while(s, syms) -> None:
    if "cond" not in s:
        raise VerifyError("while missing 'cond'")
    _verify_cond(s["cond"], syms)
    _verify_body(s.get("body", []), syms)


def _verify_if(s, syms) -> None:
    if "cond" not in s:
        raise VerifyError("if missing 'cond'")
    _verify_cond(s["cond"], syms)
    _verify_body(s.get("then", []), syms)
    _verify_body(s.get("else", []), syms)


def _verify_return(s, syms) -> None:
    if "value" not in s:
        raise VerifyError("return missing 'value' key")
    if s["value"] is not None:
        _verify_expr(s["value"], syms)


def _verify_cond(cond, syms) -> None:
    """A condition must be a boolean (``"i1"``) or an int/bool expression."""
    t = _verify_expr(cond, syms)
    if t != "i1" and not _is_int_type(t):
        raise VerifyError(f"condition must be i1 or int, got {t!r}")


# --- l-value verification ------------------------------------------------------

def _verify_lvalue(lv, syms) -> None:
    """Verify an assignment target ``lv`` (a ``var`` or ``index`` l-value)."""
    if not isinstance(lv, dict) or "k" not in lv:
        raise VerifyError(f"l-value must be a dict with 'k', got {lv!r}")
    if "type" not in lv:
        raise VerifyError("l-value missing 'type'")
    kind = lv["k"]

    if kind == "var":
        name = lv.get("name")
        if not isinstance(name, str):
            raise VerifyError("var l-value missing string 'name'")
        if not _is_value_type(lv["type"]):
            raise VerifyError(f"var l-value '{name}' has invalid type {lv['type']!r}")
        if name in syms:
            if syms[name] != lv["type"]:
                raise VerifyError(
                    f"var l-value '{name}' type {lv['type']!r} conflicts with {syms[name]!r}")
        else:
            # First assignment defines the local's type.
            # v0.1 leniency is deliberate: augassign to an undefined local is
            # accepted here (the parser rejects it; the verifier stays lenient).
            syms[name] = lv["type"]
    elif kind == "index":
        _verify_index_node(lv, syms, what="index l-value")
    else:
        raise VerifyError(f"unknown l-value kind '{kind}'")


def _verify_index_node(node, syms, what) -> None:
    """Shared check for ``index`` expressions and l-values."""
    array = node.get("array")
    if not isinstance(array, str):
        raise VerifyError(f"{what} missing string 'array'")
    if array not in syms:
        raise VerifyError(f"{what} references unknown array '{array}'")
    arr_t = syms[array]
    if not _is_ptr(arr_t):
        raise VerifyError(f"{what} into non-pointer '{array}' of type {arr_t!r}")
    elem = arr_t["ptr"]
    if node["type"] != elem:
        raise VerifyError(
            f"{what} element type {node['type']!r} does not match pointer element {elem!r}")
    if "index" not in node:
        raise VerifyError(f"{what} missing 'index' sub-expression")
    idx_t = _verify_expr(node["index"], syms)
    if not _is_int_type(idx_t):
        raise VerifyError(f"{what} subscript must be int-typed, got {idx_t!r}")


# --- expression verification ---------------------------------------------------

def _verify_expr(e, syms):
    """Verify expression ``e`` and return its (validated) type."""
    if not isinstance(e, dict) or "k" not in e:
        raise VerifyError(f"expression must be a dict with 'k', got {e!r}")
    if "type" not in e:
        raise VerifyError(f"expression of kind '{e['k']}' missing 'type'")
    kind = e["k"]

    if kind == "const":
        if not _is_value_type(e["type"]):
            raise VerifyError(f"const has invalid type {e['type']!r}")
        return e["type"]

    if kind == "var":
        name = e.get("name")
        if not isinstance(name, str):
            raise VerifyError("var expression missing string 'name'")
        if name not in syms:
            raise VerifyError(f"reference to unknown variable '{name}'")
        if syms[name] != e["type"]:
            raise VerifyError(
                f"var '{name}' type {e['type']!r} conflicts with declared {syms[name]!r}")
        return e["type"]

    if kind == "index":
        _verify_index_node(e, syms, what="index")
        return e["type"]

    if kind == "binop":
        if "lhs" not in e or "rhs" not in e:
            raise VerifyError("binop missing 'lhs' or 'rhs'")
        lt = _verify_expr(e["lhs"], syms)
        rt = _verify_expr(e["rhs"], syms)
        if not _is_value_type(lt) or not _is_value_type(rt):
            raise VerifyError(f"binop operands must be numeric, got {lt!r}, {rt!r}")
        expected = unify_numeric(lt, rt)
        if e["type"] != expected:
            raise VerifyError(
                f"binop result type {e['type']!r} should be {expected!r} for {lt!r} op {rt!r}")
        return e["type"]

    if kind == "cmp":
        if "lhs" not in e or "rhs" not in e:
            raise VerifyError("cmp missing 'lhs' or 'rhs'")
        lt = _verify_expr(e["lhs"], syms)
        rt = _verify_expr(e["rhs"], syms)
        # Operands must be scalar value types: comparing pointers/void is invalid.
        if not _is_value_type(lt) or not _is_value_type(rt):
            raise VerifyError(f"cmp operands must be scalar, got {lt!r}, {rt!r}")
        if e["type"] != "i1":
            raise VerifyError(f"cmp result type must be 'i1', got {e['type']!r}")
        return "i1"

    if kind == "bool":
        args = e.get("args")
        if not isinstance(args, list):
            raise VerifyError("bool expression 'args' must be a list")
        for a in args:
            at = _verify_expr(a, syms)
            # and/or/not operands must be scalar value types, never ptr/void.
            if not _is_value_type(at):
                raise VerifyError(f"bool operand must be scalar, got {at!r}")
        if e["type"] != "i1":
            raise VerifyError(f"bool result type must be 'i1', got {e['type']!r}")
        return "i1"

    if kind == "call":
        fn = e.get("fn")
        if fn not in _INTRINSIC_ARITY:
            raise VerifyError(f"call to unknown intrinsic '{fn}'")
        args = e.get("args")
        if not isinstance(args, list):
            raise VerifyError("call 'args' must be a list")
        arity = _INTRINSIC_ARITY[fn]
        if len(args) != arity:
            raise VerifyError(
                f"intrinsic '{fn}' expects {arity} arg(s), got {len(args)}")
        for a in args:
            _verify_expr(a, syms)
        if not _is_value_type(e["type"]):
            raise VerifyError(f"call to '{fn}' has invalid result type {e['type']!r}")
        return e["type"]

    if kind == "cast":
        if "value" not in e:
            raise VerifyError("cast missing 'value'")
        _verify_expr(e["value"], syms)
        if not _is_value_type(e["type"]):
            raise VerifyError(f"cast target must be a numeric type, got {e['type']!r}")
        return e["type"]

    raise VerifyError(f"unknown expression kind '{kind}'")
