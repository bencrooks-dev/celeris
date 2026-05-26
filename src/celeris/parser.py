"""celeris frontend: Python source -> validated, typed structured IR.

Parses a Python function with the stdlib ``ast`` module, validates that it uses
only the supported numeric subset, and lowers it to the canonical IR (see
``celeris.ir``). Anything outside the subset raises ``UnsupportedFeature`` so the
runtime can fall back to plain Python.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from . import ir
from .errors import TypeErrorIR, UnsupportedFeature
from .types import annotation_to_type, is_float, is_int, unify_numeric

_INTRINSICS = {"sqrt": "f64", "exp": "f64", "log": "f64", "sin": "f64",
               "cos": "f64", "fabs": "f64", "floor": "f64",
               "fmax": "f64", "fmin": "f64", "len": "i64"}

_BINOP = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
          ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
_CMP = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
        ast.Eq: "==", ast.NotEq: "!="}

_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg,
    ast.For, ast.While, ast.If, ast.Assign, ast.AugAssign, ast.Return,
    ast.Expr, ast.Pass, ast.Call, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.Name, ast.Constant, ast.Subscript,
    ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)


class _Validator(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsupportedFeature(
                f"{type(node).__name__} is not supported in the celeris subset "
                f"(line {getattr(node, 'lineno', '?')})")
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise UnsupportedFeature("only direct calls to intrinsics/range allowed")
        if node.func.id not in _INTRINSICS and node.func.id != "range":
            raise UnsupportedFeature(
                f"call to '{node.func.id}' not in intrinsic whitelist")
        if node.keywords:
            raise UnsupportedFeature("keyword arguments are not supported")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Tuple):
            raise UnsupportedFeature("multi-dimensional indexing not supported in v0.1")
        if isinstance(node.slice, ast.Slice):
            raise UnsupportedFeature("slicing not supported in v0.1")
        self.generic_visit(node)


class _IRBuilder:
    def __init__(self) -> None:
        self.symtab: dict[str, object] = {}
        self.ret_type: object = "void"

    # -- annotations
    def _annot(self, node) -> object:
        if node is None:
            return "void"
        if isinstance(node, ast.Constant) and node.value is None:
            return "void"
        if isinstance(node, ast.Name):
            return annotation_to_type(node.id)
        raise UnsupportedFeature("annotation must be a simple name or None")

    def _elem_type(self, array_name: str) -> str:
        t = self.symtab.get(array_name)
        if not (isinstance(t, dict) and "ptr" in t):
            raise TypeErrorIR(f"'{array_name}' is not an array")
        return t["ptr"]

    # -- top level
    def build(self, fn: ast.FunctionDef) -> dict:
        params = []
        for a in fn.args.args:
            if a.annotation is None:
                raise UnsupportedFeature(f"parameter '{a.arg}' must be annotated")
            t = self._annot(a.annotation)
            self.symtab[a.arg] = t
            params.append(ir.param(a.arg, t))
        ret = self._annot(fn.returns)
        self.ret_type = ret
        body = [s for s in (self._stmt(x) for x in fn.body) if s is not None]
        return ir.kernel(fn.name, params, ret, body)

    # -- statements
    def _stmt(self, s: ast.AST):
        if isinstance(s, ast.Assign):
            return self._assign(s)
        if isinstance(s, ast.AugAssign):
            return self._augassign(s)
        if isinstance(s, ast.For):
            return self._for(s)
        if isinstance(s, ast.While):
            return ir.while_(self._expr(s.test),
                             [x for x in (self._stmt(b) for b in s.body) if x is not None])
        if isinstance(s, ast.If):
            return ir.if_(self._expr(s.test),
                          [x for x in (self._stmt(b) for b in s.body) if x is not None],
                          [x for x in (self._stmt(b) for b in s.orelse) if x is not None])
        if isinstance(s, ast.Return):
            if s.value is None:
                return ir.ret(None)
            return ir.ret(self._cast(self._expr(s.value), self.ret_type))
        if isinstance(s, (ast.Pass, ast.Expr)):
            return None  # docstrings / pure bare expressions: no effect in subset
        raise UnsupportedFeature(f"statement {type(s).__name__} not supported")

    def _assign(self, s: ast.Assign):
        if len(s.targets) != 1:
            raise UnsupportedFeature("chained/tuple assignment not supported")
        tgt = s.targets[0]
        val = self._expr(s.value)
        if isinstance(tgt, ast.Name):
            if tgt.id not in self.symtab:
                self.symtab[tgt.id] = val["type"]
            lty = self.symtab[tgt.id]
            return ir.assign(ir.lval_var(tgt.id, lty), self._cast(val, lty))
        if isinstance(tgt, ast.Subscript):
            arr = tgt.value.id
            ety = self._elem_type(arr)
            return ir.assign(ir.lval_index(arr, self._expr(_sub_index(tgt)), ety),
                             self._cast(val, ety))
        raise UnsupportedFeature("unsupported assignment target")

    def _augassign(self, s: ast.AugAssign):
        op = _BINOP[type(s.op)]
        if isinstance(s.target, ast.Name):
            if s.target.id not in self.symtab:
                raise TypeErrorIR(
                    f"augmented assignment to undefined name '{s.target.id}'")
            ty = self.symtab[s.target.id]
            return ir.augassign(op, ir.lval_var(s.target.id, ty),
                                self._cast(self._expr(s.value), ty))
        if isinstance(s.target, ast.Subscript):
            arr = s.target.value.id
            ety = self._elem_type(arr)
            return ir.augassign(op, ir.lval_index(arr, self._expr(_sub_index(s.target)), ety),
                                self._cast(self._expr(s.value), ety))
        raise UnsupportedFeature("unsupported augmented-assignment target")

    def _for(self, s: ast.For):
        if not (isinstance(s.iter, ast.Call) and isinstance(s.iter.func, ast.Name)
                and s.iter.func.id == "range"):
            raise UnsupportedFeature("only 'for i in range(...)' is supported")
        if not isinstance(s.target, ast.Name):
            raise UnsupportedFeature("for-target must be a single name")
        args = s.iter.args
        def ci(v):
            return ir.const("i64", v)
        if len(args) == 1:
            start, stop, step = ci(0), self._expr(args[0]), ci(1)
        elif len(args) == 2:
            start, stop, step = self._expr(args[0]), self._expr(args[1]), ci(1)
        elif len(args) == 3:
            start, stop, step = self._expr(args[0]), self._expr(args[1]), self._expr(args[2])
        else:
            raise UnsupportedFeature("range takes 1-3 arguments")
        self.symtab[s.target.id] = "i64"
        body = [x for x in (self._stmt(b) for b in s.body) if x is not None]
        return ir.for_(s.target.id, start, stop, step, body)

    # -- expressions
    def _expr(self, e: ast.AST) -> dict:
        if isinstance(e, ast.Constant):
            if isinstance(e.value, bool):
                return ir.const("i1", int(e.value))
            if isinstance(e.value, int):
                return ir.const("i64", e.value)
            if isinstance(e.value, float):
                return ir.const("f64", e.value)
            raise UnsupportedFeature(f"constant {e.value!r} not supported")
        if isinstance(e, ast.Name):
            if e.id not in self.symtab:
                raise TypeErrorIR(f"use of undefined name '{e.id}'")
            return ir.var(e.id, self.symtab[e.id])
        if isinstance(e, ast.Subscript):
            arr = e.value.id
            return ir.index(arr, self._expr(_sub_index(e)), self._elem_type(arr))
        if isinstance(e, ast.BinOp):
            l, r = self._expr(e.left), self._expr(e.right)
            ty = unify_numeric(l["type"], r["type"])
            # '/' is true division (Python semantics): result is always f64,
            # and operands are cast to f64 by the _cast calls below.
            if _BINOP[type(e.op)] == "/":
                ty = "f64"
            return ir.binop(_BINOP[type(e.op)], ty, self._cast(l, ty), self._cast(r, ty))
        if isinstance(e, ast.UnaryOp):
            v = self._expr(e.operand)
            if isinstance(e.op, ast.Not):
                return ir.boolop("not", [v])
            if isinstance(e.op, ast.USub):
                return ir.binop("-", v["type"], ir.const(v["type"], 0), v)
            return v
        if isinstance(e, ast.Compare):
            if len(e.ops) != 1:
                raise UnsupportedFeature("chained comparisons not supported")
            return ir.cmp(_CMP[type(e.ops[0])], self._expr(e.left),
                          self._expr(e.comparators[0]))
        if isinstance(e, ast.BoolOp):
            op = "and" if isinstance(e.op, ast.And) else "or"
            return ir.boolop(op, [self._expr(x) for x in e.values])
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
            fn = e.func.id
            if fn not in _INTRINSICS:
                raise UnsupportedFeature(f"call to '{fn}' not allowed in an expression")
            return ir.call(fn, _INTRINSICS[fn], [self._expr(a) for a in e.args])
        raise UnsupportedFeature(f"expression {type(e).__name__} not supported")

    def _cast(self, e: dict, target) -> dict:
        if e["type"] == target or target in ("void", "i1"):
            return e
        src_ok = is_int(e["type"]) or is_float(e["type"]) or e["type"] == "i1"
        tgt_ok = is_int(target) or is_float(target)
        if src_ok and tgt_ok:
            return ir.cast(target, e)
        return e


def _sub_index(node: ast.Subscript) -> ast.AST:
    sl = node.slice
    return sl.value if isinstance(sl, ast.Index) else sl  # ast.Index only on py<3.9


def parse_function(fn) -> dict:
    """Parse a Python function object into validated typed IR (a dict)."""
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    fndef = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    fndef.decorator_list = []
    _Validator().visit(fndef)
    return _IRBuilder().build(fndef)
