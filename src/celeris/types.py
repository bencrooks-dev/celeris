"""celeris type system — the small numeric type vocabulary used by the IR.

Types are represented as plain Python values rather than classes so they are
cheap to compare and serialise:

* scalars are strings: ``"i32"``, ``"i64"``, ``"f32"``, ``"f64"``;
* ``"void"`` marks the absence of a value (a function returning ``None``);
* arrays are pointers, represented as ``{"ptr": <element-scalar>}``.

Callers annotate their Python source with the marker classes defined below
(e.g. ``f32``, ``F64Array``); :func:`annotation_to_type` maps the annotation's
name to its internal representation.
"""

from .errors import UnsupportedFeature


# --- annotation marker classes -------------------------------------------------
# These exist only to be referenced as type annotations in user source. They are
# never instantiated; celeris reads their *name* via ``annotation_to_type``.

class f32:
    """Marker annotation for a 32-bit float scalar."""


class i32:
    """Marker annotation for a 32-bit integer scalar."""


class F64Array:
    """Marker annotation for a contiguous array of 64-bit floats."""


class F32Array:
    """Marker annotation for a contiguous array of 32-bit floats."""


class I64Array:
    """Marker annotation for a contiguous array of 64-bit integers."""


class I32Array:
    """Marker annotation for a contiguous array of 32-bit integers."""


# --- annotation name -> internal type ------------------------------------------

_SCALAR = {"int": "i64", "float": "f64", "f32": "f32", "i32": "i32"}
_ARRAY = {
    "F64Array": {"ptr": "f64"},
    "F32Array": {"ptr": "f32"},
    "I64Array": {"ptr": "i64"},
    "I32Array": {"ptr": "i32"},
}

# C type names per scalar, kept for backends that emit C source.
SCALAR_C = {"i32": "int32_t", "i64": "int64_t", "f32": "float", "f64": "double"}


def annotation_to_type(name: str):
    """Map an annotation's name to its internal type representation.

    Returns a scalar string for ``int``/``float``/``f32``/``i32``, an array
    ``{"ptr": ...}`` mapping for the ``*Array`` markers, and ``"void"`` for
    ``None``. Raises :class:`~celeris.errors.UnsupportedFeature` otherwise.
    """
    if name in _SCALAR:
        return _SCALAR[name]
    if name in _ARRAY:
        return _ARRAY[name]
    if name == "None":
        return "void"
    raise UnsupportedFeature(f"unknown annotation '{name}'")


def is_float(t) -> bool:
    """True if ``t`` is a float scalar (``"f32"`` or ``"f64"``)."""
    return t in ("f32", "f64")


def is_int(t) -> bool:
    """True if ``t`` is an integer scalar (``"i32"`` or ``"i64"``)."""
    return t in ("i32", "i64")


def unify_numeric(a, b) -> str:
    """Promote two scalar types to a common type.

    Toy promotion rule: identical types are unchanged; if either operand is a
    float the result widens to ``"f64"``; otherwise the result is ``"i64"``.
    """
    if a == b:
        return a
    if is_float(a) or is_float(b):
        return "f64"
    return "i64"
