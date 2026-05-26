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
`"i32"`, `"i64"`, `"f32"`, `"f64"`, or `"void"` (kernel return only) — or a 1-D array type
written `{"ptr": <scalar>}`, e.g. `{"ptr": "f64"}`.

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
| **index** | `{"k": "index", "type": <scalar>, "array": str, "index": <Expr>}` |

The `index` target's `array` must name a parameter/local of type `{"ptr": T}`, and the node's
`type` must equal that element type `T`.

## Expressions (`"k"`)

| Kind | Shape |
| --- | --- |
| **const** | `{"k": "const", "type": <scalar>, "value": <number>}` |
| **var** | `{"k": "var", "type": <type>, "name": str}` |
| **index** | `{"k": "index", "type": <scalar>, "array": str, "index": <Expr>}` |
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
