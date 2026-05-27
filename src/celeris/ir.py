"""celeris intermediate representation — the canonical node contract.

The IR is a JSON-serialisable tree of plain ``dict`` nodes. This module is the
single source of truth for node shapes: the parser, verifier, optimisation
passes, and every backend bind to the exact dicts emitted by the constructors
below. Construct nodes only through these helpers so the shapes stay uniform.

Three families of nodes:

* **Expressions** carry a ``"k"`` discriminator and always a ``"type"``.
* **Statements** carry an ``"op"`` discriminator.
* **L-values** are assignment targets; they reuse the ``"k"`` shapes of the
  corresponding expressions (``var`` / ``index``).

:func:`dumps` / :func:`loads` provide a deterministic JSON round-trip.
"""

import json

#: Version stamped into every :func:`module` blob; bump on incompatible changes.
SCHEMA_VERSION = 1


# --- expressions ---------------------------------------------------------------

def const(type, value):
    """A literal ``value`` of scalar ``type``."""
    return {"k": "const", "type": type, "value": value}


def var(name, type):
    """A reference to variable ``name`` (note: name-first arg order)."""
    return {"k": "var", "type": type, "name": name}


def index(array, index, type):
    """Element ``array[index]``; ``type`` is the element type."""
    return {"k": "index", "type": type, "array": array, "index": index}


def index_nd(array, indices, type):
    """N-D array element read: ``array[indices[0], indices[1], ...]``.

    ``type`` is the element type. The presence of the ``"indices"`` key (a list)
    rather than a single ``"index"`` distinguishes the multi-dim node from the
    1-D :func:`index` node, keeping the 1-D path byte-for-byte unchanged.
    """
    return {"k": "index", "type": type, "array": array, "indices": indices}


def binop(op, type, lhs, rhs):
    """Binary arithmetic ``lhs op rhs`` producing ``type``."""
    return {"k": "binop", "op": op, "type": type, "lhs": lhs, "rhs": rhs}


def cmp(op, lhs, rhs):
    """Comparison ``lhs op rhs`` producing a boolean (``"i1"``)."""
    return {"k": "cmp", "op": op, "type": "i1", "lhs": lhs, "rhs": rhs}


def boolop(op, args):
    """Boolean ``and``/``or``/``not`` over ``args`` producing ``"i1"``."""
    return {"k": "bool", "op": op, "type": "i1", "args": args}


def call(fn, type, args):
    """A call to ``fn`` with ``args`` producing ``type``."""
    return {"k": "call", "fn": fn, "type": type, "args": args}


def cast(type, value):
    """Cast expression ``value`` to ``type``."""
    return {"k": "cast", "type": type, "value": value}


# --- l-values ------------------------------------------------------------------

def lval_var(name, type):
    """Assignment target: variable ``name`` of ``type``."""
    return {"k": "var", "name": name, "type": type}


def lval_index(array, index, type):
    """Assignment target: element ``array[index]`` of element ``type``."""
    return {"k": "index", "array": array, "type": type, "index": index}


def lval_index_nd(array, indices, type):
    """Assignment target: N-D element ``array[indices[0], ...]`` of ``type``."""
    return {"k": "index", "array": array, "type": type, "indices": indices}


# --- statements ----------------------------------------------------------------

def assign(target, value):
    """Assign ``value`` to l-value ``target``."""
    return {"op": "assign", "target": target, "value": value}


def augassign(binop, target, value):
    """Augmented assign ``target binop= value`` (``binop`` is the operator)."""
    return {"op": "augassign", "binop": binop, "target": target, "value": value}


def for_(var, start, stop, step, body, parallel=False):
    """Counted loop ``for var in range(start, stop, step): body``.

    ``parallel`` is a hint (set by the parser for ``prange`` loops) that the
    loop iterations are intended to run independently; backends may execute the
    loop in parallel when they can prove independence, and otherwise fall back
    to serial. The flag is always emitted as a ``bool``.
    """
    return {"op": "for", "var": var, "start": start, "stop": stop, "step": step,
            "body": body, "parallel": bool(parallel)}


def has_parallel_loop(node) -> bool:
    """True if any ``for`` node anywhere in ``node`` (kernel/stmt/list) is parallel."""
    if isinstance(node, dict):
        if node.get("op") == "for" and node.get("parallel"):
            return True
        return any(has_parallel_loop(v) for v in node.values())
    if isinstance(node, list):
        return any(has_parallel_loop(x) for x in node)
    return False


def while_(cond, body):
    """While loop ``while cond: body``."""
    return {"op": "while", "cond": cond, "body": body}


def if_(cond, then, els):
    """Conditional ``if cond: then else: els``."""
    return {"op": "if", "cond": cond, "then": then, "else": els}


def ret(value):
    """Return ``value`` (``None`` for a bare ``return``)."""
    return {"op": "return", "value": value}


# --- structural ----------------------------------------------------------------

def param(name, type):
    """A kernel parameter ``name`` of ``type``."""
    return {"name": name, "type": type}


def kernel(name, params, ret, body):
    """A kernel ``name`` with ``params``, return type ``ret`` and ``body``."""
    return {"name": name, "params": params, "ret": ret, "body": body}


def module(kernels):
    """A module of ``kernels``, stamped with :data:`SCHEMA_VERSION`."""
    return {"schema": SCHEMA_VERSION, "kernels": kernels}


# --- serialisation -------------------------------------------------------------

def dumps(obj):
    """Serialise an IR ``obj`` to a deterministic JSON string."""
    return json.dumps(obj, sort_keys=True)


def loads(s):
    """Parse a JSON string ``s`` back into an IR object."""
    return json.loads(s)
