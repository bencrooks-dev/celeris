"""Golden-kernel registry + IR fingerprint matcher — the hand-tuned fast-path.

Recognized IR shapes (saxpy, scale, sum, dot) are compiled with __restrict__
pointer params, which unlocks vectorization the generic source-gen backend
cannot apply (its pointers may alias). The emitted statement body reuses
source-gen's codegen, so semantics are identical to the verified generic path;
only the signature differs. This mirrors how MKL/oneDNN ship hand-tuned kernels
for known shapes with a general path behind them.

PRECONDITION: distinct array parameters are assumed not to alias (the BLAS
convention). Passing overlapping buffers to a golden kernel is undefined.
"""
from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from . import register
from .. import ir as _ir
from ..ir import has_parallel_loop
from ..errors import CompileError
from .sourcegen import (_CACHE, _collect_locals, _cty, _ctypes_for, _emit_stmts,
                        _PRELUDE, _safe_name)


# --------------------------------------------------------------------------
# Shape matchers. Each returns True if the IR matches that golden shape.
# Looseness is safe: compile() reuses source-gen codegen, so a matched kernel
# is always semantically correct; the matcher only decides whether the kernel
# qualifies for the __restrict__ fast-path.
# --------------------------------------------------------------------------
def _single_loop(ir) -> Optional[dict]:
    b = ir["body"]
    if len(b) == 1 and b[0]["op"] == "for" and len(b[0]["body"]) == 1:
        return b[0]
    return None


def _is_index(e) -> bool:
    return isinstance(e, dict) and e.get("k") == "index"


def _contains_nd_index(node) -> bool:
    """Recursively scan the IR for any N-D index node (one carrying "indices").

    There are no 2-D golden kernels in v0.5, so the golden tier must decline any
    IR containing an `index_nd` node and let it fall through to source-gen/llvm.
    """
    if isinstance(node, dict):
        if node.get("k") == "index" and "indices" in node:
            return True
        return any(_contains_nd_index(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains_nd_index(v) for v in node)
    return False


def _match_saxpy(ir) -> bool:
    loop = _single_loop(ir)
    if not loop:
        return False
    st = loop["body"][0]
    if st["op"] != "assign" or st["target"]["k"] != "index":
        return False
    v = st["value"]
    if v.get("k") != "binop" or v["op"] != "+":
        return False
    arr = st["target"]["array"]

    def is_self_idx(e):
        return _is_index(e) and e["array"] == arr

    def is_scaled_idx(e):
        return (isinstance(e, dict) and e.get("k") == "binop" and e["op"] == "*"
                and ((e["lhs"].get("k") == "var" and _is_index(e["rhs"]))
                     or (e["rhs"].get("k") == "var" and _is_index(e["lhs"]))))

    return ((is_scaled_idx(v["lhs"]) and is_self_idx(v["rhs"]))
            or (is_scaled_idx(v["rhs"]) and is_self_idx(v["lhs"])))


def _match_scale(ir) -> bool:
    loop = _single_loop(ir)
    if not loop:
        return False
    st = loop["body"][0]
    if st["op"] != "assign" or st["target"]["k"] != "index":
        return False
    v = st["value"]
    return (v.get("k") == "binop" and v["op"] == "*"
            and ((v["lhs"].get("k") == "var" and _is_index(v["rhs"]))
                 or (v["rhs"].get("k") == "var" and _is_index(v["lhs"]))))


def _reduction_acc(ir):
    """Return the accumulator name if ir is acc=init; for: acc = acc <op> ...; return acc."""
    b = ir["body"]
    if len(b) < 3:
        return None
    if b[0]["op"] != "assign" or b[0]["target"]["k"] != "var":
        return None
    acc = b[0]["target"]["name"]
    # find the single for-loop and a return acc
    loop = next((s for s in b if s["op"] == "for"), None)
    ret = b[-1]
    if loop is None or ret["op"] != "return":
        return None
    if not (ret["value"] and ret["value"].get("k") == "var" and ret["value"]["name"] == acc):
        return None
    if len(loop["body"]) != 1:
        return None
    s = loop["body"][0]
    # acc = acc <op> rhs   OR   acc += rhs
    if s["op"] == "augassign" and s["target"].get("name") == acc:
        return (acc, s["binop"], s["value"])
    if s["op"] == "assign" and s["target"].get("name") == acc:
        v = s["value"]
        if v.get("k") == "binop" and v["lhs"].get("k") == "var" and v["lhs"]["name"] == acc:
            return (acc, v["op"], v["rhs"])
    return None


def _match_sum(ir) -> bool:
    r = _reduction_acc(ir)
    return bool(r and r[1] == "+" and _is_index(r[2]))


def _match_dot(ir) -> bool:
    r = _reduction_acc(ir)
    if not (r and r[1] == "+"):
        return False
    rhs = r[2]
    return (rhs.get("k") == "binop" and rhs["op"] == "*"
            and _is_index(rhs["lhs"]) and _is_index(rhs["rhs"]))


@dataclass
class GoldenKernel:
    name: str
    matches: Callable[[dict], bool]


REGISTRY: dict[str, GoldenKernel] = {
    "saxpy": GoldenKernel("saxpy", _match_saxpy),
    "scale": GoldenKernel("scale", _match_scale),
    "sum": GoldenKernel("sum", _match_sum),
    "dot": GoldenKernel("dot", _match_dot),
}


def _emit_tuned(ir: dict) -> str:
    """Same as source-gen, but pointer params get __restrict__."""
    params = [p["name"] for p in ir["params"]]

    def one(p):
        cty = _cty(p["type"])
        if cty.endswith("*"):
            return f"{cty} __restrict__ {p['name']}"
        return f"{cty} {p['name']}"

    sig = ", ".join(one(p) for p in ir["params"])
    locs: dict[str, str] = {}
    _collect_locals(ir["body"], set(params), locs)
    decls = "".join(f"    {c} {n} = 0;\n" for n, c in locs.items())
    body = _emit_stmts(ir["body"])
    return (f'{_PRELUDE}\nextern "C" {_cty(ir["ret"])} {ir["name"]}({sig}) {{\n'
            f'{decls}{body}\n}}\n')


class KernelBackend:
    """Hand-tuned golden-kernel fast-path. Falls through (raises) on non-match."""

    name = "kernels"

    def available(self) -> bool:
        return shutil.which("clang++") is not None

    def matches(self, ir: dict) -> bool:
        # Parallel loops fall through to source-gen's threaded path.
        if has_parallel_loop(ir):
            return False
        # No 2-D golden kernels in v0.5: decline any N-D index_nd IR so it
        # falls through to source-gen/llvm.
        if _contains_nd_index(ir):
            return False
        return any(g.matches(ir) for g in REGISTRY.values())

    def compile(self, ir: dict):
        if not self.matches(ir):
            raise CompileError(f"no golden kernel matches '{ir.get('name')}'")
        src = _emit_tuned(ir)
        key = hashlib.sha1(("tuned\n" + _ir.dumps(ir) + "\n" + src).encode()).hexdigest()[:16]
        _CACHE.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(ir["name"])
        so = _CACHE / f"kern_{stem}_{key}.so"
        if not so.exists():
            cpp = _CACHE / f"kern_{stem}_{key}.cpp"
            cpp.write_text(src)
            cc = shutil.which("clang++") or "clang++"
            r = subprocess.run([cc, "-O3", "-std=c++17", "-fPIC", "-shared",
                                "-march=native", str(cpp), "-o", str(so)],
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


register(KernelBackend())
