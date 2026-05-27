# IR Specification

The celeris IR is a plain, JSON-serializable dict tree. It is the contract between the frontend
(`parser.py`) and every backend, and the thing the verifier (`verifier.py`) re-checks from
scratch. `ir.dumps` / `ir.loads` round-trip it via `json` with sorted keys. The schema is
versioned by `ir.SCHEMA_VERSION` (currently `1`).

Two discriminator keys are used by convention:

- **statements** carry an `"op"` key (`assign`, `augassign`, `for`, `while`, `if`, `return`).
- **expressions and l-values** carry a `"k"` key (`const`, `var`, `index`, `binop`, `cmp`,
  `bool`, `call`, `cast`).

**Every expression and l-value carries a `type`.** A type is either a scalar string —
`"i32"`, `"i64"`, `"f32"`, `"f64"`, or `"void"` (kernel return only) — or an array (pointer)
type. A 1-D array is written `{"ptr": <scalar>}`, e.g. `{"ptr": "f64"}`. An N-D array carries an
explicit dimensionality: `{"ptr": <scalar>, "ndim": N}` (`N` an int ≥ 1), e.g.
`{"ptr": "f64", "ndim": 2}` for a 2-D `f64` array. A bare `{"ptr": ...}` has an implicit `ndim`
of `1`, so `{"ptr": "f64"}` and `{"ptr": "f64", "ndim": 1}` denote the same 1-D type;
`types.ndim_of(t)` returns `0` for scalars, `1` for a bare pointer, and the explicit `ndim`
otherwise. As of v0.5.0 the frontend emits `ndim` up to `2` (`F64Array2D`-style markers);
rank ≥ 3 is reserved for a future release.

## Structural nodes

| Node | Shape |
| --- | --- |
| **Module** | `{"k": "module", "kernels": [<Kernel>, ...]}` |
| **Kernel** | `{"name": str, "params": [<Param>, ...], "ret": <type>, "body": [<Stmt>, ...]}` |
| **Param** | `{"name": str, "type": <type>}` |

## Statements (`"op"`)

| Op | Shape |
| --- | --- |
| **assign** | `{"op": "assign", "target": <LValue>, "value": <Expr>}` |
| **augassign** | `{"op": "augassign", "binop": str, "target": <LValue>, "value": <Expr>}` |
| **for** | `{"op": "for", "var": str, "start": <Expr>, "stop": <Expr>, "step": <Expr>, "body": [<Stmt>, ...]}` |
| **while** | `{"op": "while", "cond": <Expr>, "body": [<Stmt>, ...]}` |
| **if** | `{"op": "if", "cond": <Expr>, "then": [<Stmt>, ...], "else": [<Stmt>, ...]}` |
| **return** | `{"op": "return", "value": <Expr> | null}` |

## L-values (`"k"`, assignable)

| Kind | Shape |
| --- | --- |
| **var** | `{"k": "var", "type": <type>, "name": str}` |
| **index** (1-D) | `{"k": "index", "type": <scalar>, "array": str, "index": <Expr>}` |
| **index** (N-D) | `{"k": "index", "type": <scalar>, "array": str, "indices": [<Expr>, ...]}` |

The `index` target's `array` must name a parameter/local of pointer type, and the node's `type`
must equal that element type `T`. A 1-D index carries a single `"index"` expression and the
array must have `ndim == 1`. An **N-D index** (introduced in v0.5.0, e.g. `a[i, j]`) instead
carries an `"indices"` list whose length must equal the array's `ndim`; each index expression
must be integer-typed. The two forms are mutually exclusive — a node has either `"index"` or
`"indices"`, never both — so 1-D nodes are unchanged and fully back-compatible.

## Expressions (`"k"`)

| Kind | Shape |
| --- | --- |
| **const** | `{"k": "const", "type": <scalar>, "value": <number>}` |
| **var** | `{"k": "var", "type": <type>, "name": str}` |
| **index** (1-D) | `{"k": "index", "type": <scalar>, "array": str, "index": <Expr>}` |
| **index** (N-D) | `{"k": "index", "type": <scalar>, "array": str, "indices": [<Expr>, ...]}` |
| **binop** | `{"k": "binop", "op": str, "type": <scalar>, "lhs": <Expr>, "rhs": <Expr>}` |
| **cmp** | `{"k": "cmp", "op": str, "type": "i1", "lhs": <Expr>, "rhs": <Expr>}` |
| **bool** | `{"k": "bool", "op": "and" | "or" | "not", "type": "i1", "args": [<Expr>, ...]}` |
| **call** | `{"k": "call", "fn": str, "type": <scalar>, "args": [<Expr>, ...]}` |
| **cast** | `{"k": "cast", "type": <scalar>, "value": <Expr>}` |

`binop` operands must unify to the node's `type` (see `types.unify_numeric`: any float operand
promotes to `f64`, an all-int mix to `i64`). `op` is one of `+ - * / // % **`. `/` is true
division and always yields `f64` (Python semantics); `//` is floor division and `%` is floored
modulo (sign follows divisor). `cmp` `op` is one
of `< <= > >= == !=` and yields an integer truth value. `call` `fn` must be in the intrinsic
whitelist (`sqrt`, `exp`, `log`, `sin`, `cos`, `fabs`, `floor`, `fmax`, `fmin`, `len`) with the
correct arity.

## Example: saxpy

Source:

```python
def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
    for i in range(n):
        y[i] = a * x[i] + y[i]
```

Compiled IR:

```json
{
  "name": "saxpy",
  "ret": "void",
  "params": [
    {"name": "a", "type": "f64"},
    {"name": "x", "type": {"ptr": "f64"}},
    {"name": "y", "type": {"ptr": "f64"}},
    {"name": "n", "type": "i64"}
  ],
  "body": [
    {
      "op": "for",
      "var": "i",
      "start": {"k": "const", "type": "i64", "value": 0},
      "stop": {"k": "var", "type": "i64", "name": "n"},
      "step": {"k": "const", "type": "i64", "value": 1},
      "body": [
        {
          "op": "assign",
          "target": {
            "k": "index", "array": "y", "type": "f64",
            "index": {"k": "var", "type": "i64", "name": "i"}
          },
          "value": {
            "k": "binop", "op": "+", "type": "f64",
            "lhs": {
              "k": "binop", "op": "*", "type": "f64",
              "lhs": {"k": "var", "type": "f64", "name": "a"},
              "rhs": {
                "k": "index", "array": "x", "type": "f64",
                "index": {"k": "var", "type": "i64", "name": "i"}
              }
            },
            "rhs": {
              "k": "index", "array": "y", "type": "f64",
              "index": {"k": "var", "type": "i64", "name": "i"}
            }
          }
        }
      ]
    }
  ]
}
```

## N-D arrays and the strided calling convention

A 2-D array parameter has type `{"ptr": <elem>, "ndim": 2}`, and an `a[i, j]` access compiles to
an `index` node with an `"indices"` list:

```json
{
  "k": "index", "type": "f64", "array": "a",
  "indices": [
    {"k": "var", "type": "i64", "name": "i"},
    {"k": "var", "type": "i64", "name": "j"}
  ]
}
```

The native backends use **general strides**, so an N-D array is *not* assumed contiguous. Each
N-D (`ndim ≥ 2`) array parameter is passed across the C ABI as a **data pointer followed by one
`int64` stride per dimension, measured in elements** (not bytes). For a 2-D array `a` the
generated function signature is `(<elem>* a, int64_t a_s0, int64_t a_s1, ...)`, and an
`index` node with `indices = [e0, e1]` lowers to a flat offset

```
a[(e0) * a_s0 + (e1) * a_s1]
```

i.e. `data[Σ_d idx_d · stride_d]`. The marshalling layer fills the stride arguments at call time
from the NumPy array's own `strides` (`arr.strides[d] // arr.itemsize`), so a non-contiguous
view — a slice of a larger buffer or a `.T` transpose — supplies the right strides and computes
correctly without a copy. 1-D arrays keep the original single-pointer convention (no stride
arguments). The pure-Python interpreter ignores this convention entirely and indexes the real
NumPy array with `arr[tuple(indices)]`, making it the strided oracle for the differential
harness.
