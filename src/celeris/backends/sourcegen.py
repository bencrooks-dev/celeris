"""C++ source-gen JIT backend.

Lowers typed IR to C++ source (1:1 with the structured IR), compiles it with
``clang++ -O3`` into a shared library, and loads it via ctypes. This is the
general native path: it compiles ANY kernel in the supported subset, and
clang's -O3 pipeline (the LLVM optimizer) gives autovectorized, LLVM-quality
machine code without celeris needing the LLVM API.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import re
import shutil
import subprocess

from . import register
from .. import ir as _ir
from ..errors import CompileError
from ..passes import _collect, _has_return


def _safe_name(name: str) -> str:
    """Sanitize an IR name for use as a cache *filename* stem only.

    Prevents path traversal (e.g. ``name="../../tmp/pwn"``) from escaping the
    cache dir. The real C symbol name (``ir['name']``) is unaffected.
    """
    s = re.sub(r"\W", "_", str(name))
    return s or "kernel"

_CTYPE = {"i32": "int32_t", "i64": "int64_t", "f32": "float",
          "f64": "double", "void": "void", "i1": "int"}
_CTYPES = {"i32": ctypes.c_int32, "i64": ctypes.c_int64,
           "f32": ctypes.c_float, "f64": ctypes.c_double, "i1": ctypes.c_int}
_INTRIN = {"sqrt": "std::sqrt", "exp": "std::exp", "log": "std::log",
           "sin": "std::sin", "cos": "std::cos", "fabs": "std::fabs",
           "floor": "std::floor", "fmax": "std::fmax", "fmin": "std::fmin"}
_CACHE = pathlib.Path(os.path.expanduser("~/.celeris_cache"))

_PRELUDE = """\
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>
#include <algorithm>

template <class T> static inline T celeris_floordiv(T a, T b) {
    return static_cast<T>(std::floor(static_cast<double>(a) / static_cast<double>(b)));
}
template <class T> static inline T celeris_floormod(T a, T b) {
    return a - b * celeris_floordiv(a, b);
}
"""


def _cty(t) -> str:
    if isinstance(t, dict) and "ptr" in t:
        return _CTYPE[t["ptr"]] + "*"
    return _CTYPE[t]


def _ctypes_for(t):
    if isinstance(t, dict) and "ptr" in t:
        return ctypes.POINTER(_CTYPES[t["ptr"]])
    if t == "void":
        return None
    return _CTYPES[t]


def _collect_locals(stmts, params, out):
    """Hoist all assigned-var and loop-var names (not params) to function scope."""
    for s in stmts:
        op = s["op"]
        if op in ("assign", "augassign") and s["target"]["k"] == "var":
            n = s["target"]["name"]
            if n not in params and n not in out:
                out[n] = _cty(s["target"]["type"])
        if op == "for":
            if s["var"] not in params and s["var"] not in out:
                out[s["var"]] = "int64_t"
            _collect_locals(s["body"], params, out)
        elif op == "while":
            _collect_locals(s["body"], params, out)
        elif op == "if":
            _collect_locals(s["then"], params, out)
            _collect_locals(s["else"], params, out)


def _fmt_const(e) -> str:
    t, v = e["type"], e["value"]
    if t in ("f32", "f64"):
        return repr(float(v))
    if t == "i64":
        return f"(int64_t){int(v)}"
    if t == "i32":
        return f"(int32_t){int(v)}"
    return str(int(v))  # i1 -> plain 0/1


def _emit_expr(e) -> str:
    k = e["k"]
    if k == "const":
        return _fmt_const(e)
    if k == "var":
        return e["name"]
    if k == "index":
        return f"{e['array']}[{_emit_expr(e['index'])}]"
    if k == "binop":
        l, r, op = _emit_expr(e["lhs"]), _emit_expr(e["rhs"]), e["op"]
        if op in ("+", "-", "*"):
            return f"({l} {op} {r})"
        if op == "/":
            return f"((double)({l}) / (double)({r}))"
        if op == "//":
            return f"celeris_floordiv({l}, {r})"
        if op == "%":
            return f"celeris_floormod({l}, {r})"
        if op == "**":
            return f"std::pow({l}, {r})"
        raise CompileError(f"sourcegen: unknown binop '{op}'")
    if k == "cmp":
        return f"({_emit_expr(e['lhs'])} {e['op']} {_emit_expr(e['rhs'])})"
    if k == "bool":
        args = [_emit_expr(a) for a in e["args"]]
        if e["op"] == "not":
            return f"(!{args[0]})"
        joiner = " && " if e["op"] == "and" else " || "
        return "(" + joiner.join(args) + ")"
    if k == "call":
        fn = e["fn"]
        if fn not in _INTRIN:
            raise CompileError(f"sourcegen: intrinsic '{fn}' not supported")
        return f"{_INTRIN[fn]}({', '.join(_emit_expr(a) for a in e['args'])})"
    if k == "cast":
        return f"({_cty(e['type'])})({_emit_expr(e['value'])})"
    raise CompileError(f"sourcegen: unknown expr kind '{k}'")


def _emit_lval(t) -> str:
    if t["k"] == "var":
        return t["name"]
    return f"{t['array']}[{_emit_expr(t['index'])}]"


def _is_parallelizable(node) -> bool:
    if not node.get("parallel", False):
        return False
    step = node["step"]
    if not (step.get("k") == "const" and isinstance(step.get("value"), int) and step["value"] == 1):
        return False
    if _has_return(node["body"]):
        return False
    var = node["var"]
    _arr_w, accesses, s_writes, _s_reads = _collect(node["body"], var)
    if s_writes:                       # no scalar writes (no temps/reductions)
        return False
    for arr, idx, is_write in accesses:
        if is_write and not (idx.get("k") == "var" and idx.get("name") == var):
            return False               # writes must be at exactly i
    return True


def _emit_parallel_for(s, ind):
    v = s["var"]
    st, sp = _emit_expr(s["start"]), _emit_expr(s["stop"])
    body = _emit_stmts(s["body"], ind + "            ")
    return "\n".join([
        f"{ind}{{",
        f"{ind}  const int64_t _lo = {st}, _hi = {sp};",
        f"{ind}  if (_hi - _lo < 4096) {{",
        f"{ind}    for (int64_t {v} = _lo; {v} < _hi; ++{v}) {{",
        body,
        f"{ind}    }}",
        f"{ind}  }} else {{",
        f"{ind}    unsigned _nt = std::thread::hardware_concurrency();",
        f"{ind}    if (_nt == 0u) _nt = 1u; if (_nt > 8u) _nt = 8u;",
        f"{ind}    int64_t _chunk = (_hi - _lo + (int64_t)_nt - 1) / (int64_t)_nt;",
        f"{ind}    std::vector<std::thread> _ths;",
        f"{ind}    for (unsigned _t = 0u; _t < _nt; ++_t) {{",
        f"{ind}      int64_t _a = _lo + (int64_t)_t * _chunk;",
        f"{ind}      int64_t _b = std::min(_a + _chunk, _hi);",
        f"{ind}      if (_a >= _b) break;",
        f"{ind}      _ths.emplace_back([=]() {{",
        f"{ind}        for (int64_t {v} = _a; {v} < _b; ++{v}) {{",
        body,
        f"{ind}        }}",
        f"{ind}      }});",
        f"{ind}    }}",
        f"{ind}    for (auto& _th : _ths) _th.join();",
        f"{ind}  }}",
        f"{ind}}}",
    ])


def _emit_stmts(stmts, ind="    ") -> str:
    out = []
    for s in stmts:
        op = s["op"]
        if op == "assign":
            out.append(f"{ind}{_emit_lval(s['target'])} = {_emit_expr(s['value'])};")
        elif op == "augassign":
            lv = _emit_lval(s["target"])
            synth = {"k": "binop", "op": s["binop"], "type": s["target"]["type"],
                     "lhs": {"k": "var" if s["target"]["k"] == "var" else "index",
                             "type": s["target"]["type"],
                             **({"name": s["target"]["name"]} if s["target"]["k"] == "var"
                                else {"array": s["target"]["array"], "index": s["target"]["index"]})},
                     "rhs": s["value"]}
            out.append(f"{ind}{lv} = {_emit_expr(synth)};")
        elif op == "for":
            if _is_parallelizable(s):
                out.append(_emit_parallel_for(s, ind))
            else:
                st, sp, stp = _emit_expr(s["start"]), _emit_expr(s["stop"]), _emit_expr(s["step"])
                v = s["var"]
                out.append(f"{ind}for ({v} = {st}; (({stp}) > 0 ? {v} < ({sp}) : {v} > ({sp})); {v} += ({stp})) {{")
                out.append(_emit_stmts(s["body"], ind + "    "))
                out.append(f"{ind}}}")
        elif op == "while":
            out.append(f"{ind}while ({_emit_expr(s['cond'])}) {{")
            out.append(_emit_stmts(s["body"], ind + "    "))
            out.append(f"{ind}}}")
        elif op == "if":
            out.append(f"{ind}if ({_emit_expr(s['cond'])}) {{")
            out.append(_emit_stmts(s["then"], ind + "    "))
            if s["else"]:
                out.append(f"{ind}}} else {{")
                out.append(_emit_stmts(s["else"], ind + "    "))
            out.append(f"{ind}}}")
        elif op == "return":
            out.append(f"{ind}return {_emit_expr(s['value'])};" if s["value"] is not None else f"{ind}return;")
        else:
            raise CompileError(f"sourcegen: unknown statement op '{op}'")
    return "\n".join(out)


def emit_cpp(ir: dict) -> str:
    params = [p["name"] for p in ir["params"]]
    sig = ", ".join(f"{_cty(p['type'])} {p['name']}" for p in ir["params"])
    locs: dict[str, str] = {}
    _collect_locals(ir["body"], set(params), locs)
    decls = "\n".join(f"    {cty} {name} = 0;" for name, cty in locs.items())
    body = _emit_stmts(ir["body"])
    decls_block = (decls + "\n") if decls else ""
    return (f'{_PRELUDE}\nextern "C" {_cty(ir["ret"])} {ir["name"]}({sig}) {{\n'
            f'{decls_block}{body}\n}}\n')


class SourceGenBackend:
    """Compile IR to C++ via clang at runtime; load with ctypes."""

    name = "sourcegen"

    def available(self) -> bool:
        return shutil.which("clang++") is not None

    def compile(self, ir: dict):
        src = emit_cpp(ir)
        key = hashlib.sha1((_ir.dumps(ir) + "\n" + src).encode()).hexdigest()[:16]
        _CACHE.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(ir["name"])
        so = _CACHE / f"cel_{stem}_{key}.so"
        if not so.exists():
            cpp = _CACHE / f"cel_{stem}_{key}.cpp"
            cpp.write_text(src)
            cc = shutil.which("clang++") or "clang++"
            r = subprocess.run([cc, "-O3", "-std=c++17", "-fPIC", "-shared",
                                "-march=native", "-pthread", str(cpp), "-o", str(so)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise CompileError(f"clang++ failed:\n{r.stderr}\n--- source ---\n{src}")
        lib = ctypes.CDLL(str(so))
        fn = getattr(lib, ir["name"])
        fn.argtypes = [_ctypes_for(p["type"]) for p in ir["params"]]
        fn.restype = _ctypes_for(ir["ret"])
        ptypes = [p["type"] for p in ir["params"]]

        def call(*args):
            cargs = []
            for t, a in zip(ptypes, args):
                if isinstance(t, dict) and "ptr" in t:
                    cargs.append(a.ctypes.data_as(_ctypes_for(t)))
                else:
                    cargs.append(a)
            return fn(*cargs)

        return call


register(SourceGenBackend())
