"""Optional llvmlite JIT backend.

Lowers typed IR to LLVM IR and JIT-compiles in-process. Uses the
alloca-everything + mem2reg strategy: all params/locals get entry-block stack
slots accessed via load/store, so the optimizer inserts loop phi nodes (this is
how Clang emits -O0 code before promotion). Semantics match the interpreter
oracle: '/' is true division (fdiv), '//' Python floor division, '%' Python
floored modulo, cmp/bool yield i1.

Optional: imported only when llvmlite is present; the registry guards the import.
"""
from __future__ import annotations
import ctypes
from . import register
from ..errors import CompileError

try:
    import llvmlite.binding as llvm
    import llvmlite.ir as lir
    _HAVE = True
except Exception:
    _HAVE = False

_INIT = False
def _ensure():
    # llvmlite 0.47 / LLVM 20 initialises the core automatically (the old
    # ``llvm.initialize()`` now raises). Native target/asmprinter init are
    # retained as best-effort no-ops for portability across llvmlite versions.
    global _INIT
    if not _INIT:
        for _fn in ("initialize_native_target", "initialize_native_asmprinter"):
            f = getattr(llvm, _fn, None)
            if f is not None:
                try:
                    f()
                except Exception:
                    pass
        _INIT = True

_CT = {"i1": ctypes.c_int, "i32": ctypes.c_int32, "i64": ctypes.c_int64,
       "f32": ctypes.c_float, "f64": ctypes.c_double}

def _is_int(t): return t in ("i1", "i32", "i64")
def _is_float(t): return t in ("f32", "f64")

def _lty(t):
    if isinstance(t, dict) and "ptr" in t:
        return lir.PointerType(_lty(t["ptr"]))
    return {"i1": lir.IntType(1), "i32": lir.IntType(32), "i64": lir.IntType(64),
            "f32": lir.FloatType(), "f64": lir.DoubleType(), "void": lir.VoidType()}[t]

def _ct(t):
    if isinstance(t, dict) and "ptr" in t:
        return ctypes.POINTER(_CT[t["ptr"]])
    return None if t == "void" else _CT[t]

def _collect_locals(stmts, params, out):
    for s in stmts:
        op = s["op"]
        if op in ("assign", "augassign") and s["target"]["k"] == "var":
            n = s["target"]["name"]
            if n not in params and n not in out:
                out[n] = s["target"]["type"]
        if op == "for":
            if s["var"] not in params and s["var"] not in out:
                out[s["var"]] = "i64"
            _collect_locals(s["body"], params, out)
        elif op == "while":
            _collect_locals(s["body"], params, out)
        elif op == "if":
            _collect_locals(s["then"], params, out); _collect_locals(s["else"], params, out)


class _Lowerer:
    def __init__(self, b, func, slots, module):
        self.b, self.func, self.slots, self.module = b, func, slots, module

    def emit_block(self, stmts):
        for s in stmts:
            if self.b.block.is_terminated:
                break
            self.emit_stmt(s)

    def emit_stmt(self, s):
        op = s["op"]
        if op == "assign":
            self.store_lval(s["target"], self.eval(s["value"]))
        elif op == "augassign":
            cur = self.load_lval(s["target"])
            self.store_lval(s["target"], self.binop(s["binop"], s["target"]["type"], cur, self.eval(s["value"])))
        elif op == "return":
            self.b.ret_void() if s["value"] is None else self.b.ret(self.eval(s["value"]))
        elif op == "for":
            self.emit_for(s)
        elif op == "while":
            self.emit_while(s)
        elif op == "if":
            self.emit_if(s)
        else:
            raise CompileError(f"llvm: unknown stmt '{op}'")

    def eval(self, e):
        k, t = e["k"], e.get("type")
        if k == "const":
            return lir.Constant(_lty(t), e["value"])
        if k == "var":
            slot, sty = self.slots[e["name"]]; return self.b.load(slot, typ=_lty(sty))
        if k == "index":
            slot, sty = self.slots[e["array"]]
            ety = _lty(sty["ptr"])
            base = self.b.load(slot, typ=_lty(sty))
            ptr = self.b.gep(base, [self.eval(e["index"])], inbounds=True, source_etype=ety)
            return self.b.load(ptr, typ=ety)
        if k == "binop":
            return self.binop(e["op"], t, self.eval(e["lhs"]), self.eval(e["rhs"]))
        if k == "cmp":
            return self.cmp(e["op"], e["lhs"]["type"], self.eval(e["lhs"]), self.eval(e["rhs"]))
        if k == "bool":
            return self.boolop(e["op"], [self.eval(a) for a in e["args"]])
        if k == "call":
            return self.call(e["fn"], [self.eval(a) for a in e["args"]])
        if k == "cast":
            return self.cast(e["value"]["type"], t, self.eval(e["value"]))
        raise CompileError(f"llvm: unknown expr '{k}'")

    def binop(self, op, ty, l, r):
        I = _is_int(ty)
        if op == "+": return self.b.add(l, r) if I else self.b.fadd(l, r)
        if op == "-": return self.b.sub(l, r) if I else self.b.fsub(l, r)
        if op == "*": return self.b.mul(l, r) if I else self.b.fmul(l, r)
        if op == "/": return self.b.fdiv(l, r)            # true division -> f64
        if op == "//": return self._floordiv(ty, l, r)
        if op == "%": return self._floormod(ty, l, r)
        if op == "**":
            if I: raise CompileError("llvm: integer ** unsupported (falls back)")
            return self.call("__pow", [l, r])
        raise CompileError(f"llvm: unknown binop '{op}'")

    def _floordiv(self, ty, a, b):
        if _is_float(ty):
            return self.call("floor", [self.b.fdiv(a, b)])
        q = self.b.sdiv(a, b); r = self.b.srem(a, b)
        zero = lir.Constant(a.type, 0)
        rnz = self.b.icmp_signed("!=", r, zero)
        signs_diff = self.b.icmp_signed("<", self.b.xor(r, b), zero)
        need = self.b.and_(rnz, signs_diff)
        return self.b.select(need, self.b.sub(q, lir.Constant(a.type, 1)), q)

    def _floormod(self, ty, a, b):
        fd = self._floordiv(ty, a, b)
        return self.b.fsub(a, self.b.fmul(b, fd)) if _is_float(ty) else self.b.sub(a, self.b.mul(b, fd))

    def cmp(self, op, ty, l, r):
        return self.b.fcmp_ordered(op, l, r) if _is_float(ty) else self.b.icmp_signed(op, l, r)

    def _to_i1(self, v):
        if v.type == lir.IntType(1): return v
        if isinstance(v.type, lir.IntType): return self.b.icmp_signed("!=", v, lir.Constant(v.type, 0))
        return self.b.fcmp_ordered("!=", v, lir.Constant(v.type, 0.0))

    def boolop(self, op, args):
        bs = [self._to_i1(a) for a in args]
        if op == "not": return self.b.not_(bs[0])
        r = bs[0]
        for x in bs[1:]:
            r = self.b.and_(r, x) if op == "and" else self.b.or_(r, x)
        return r

    def cast(self, ft, tt, v):
        if ft == tt: return v
        lt = _lty(tt); order = {"i1": 1, "i32": 32, "i64": 64}
        if _is_int(ft) and _is_float(tt): return self.b.sitofp(v, lt)
        if _is_float(ft) and _is_int(tt): return self.b.fptosi(v, lt)
        if _is_int(ft) and _is_int(tt):
            if order[tt] > order[ft]:
                # i1 is boolean (0/1): zero-extend so true widens to 1, not -1.
                return self.b.zext(v, lt) if ft == "i1" else self.b.sext(v, lt)
            return self.b.trunc(v, lt) if order[tt] < order[ft] else v
        if _is_float(ft) and _is_float(tt):
            return self.b.fpext(v, lt) if (ft == "f32" and tt == "f64") else self.b.fptrunc(v, lt)
        return v

    def call(self, fn, args):
        intr = {"sqrt": "llvm.sqrt.f64", "sin": "llvm.sin.f64", "cos": "llvm.cos.f64",
                "exp": "llvm.exp.f64", "log": "llvm.log.f64", "floor": "llvm.floor.f64",
                "fabs": "llvm.fabs.f64", "fmax": "llvm.maxnum.f64", "fmin": "llvm.minnum.f64",
                "__pow": "llvm.pow.f64"}
        if fn not in intr:
            raise CompileError(f"llvm: intrinsic '{fn}' unsupported")
        dty = lir.DoubleType()
        name = intr[fn]; n = len(args)
        g = self.module.globals.get(name)
        if g is None:
            g = lir.Function(self.module, lir.FunctionType(dty, [dty] * n), name=name)
        cargs = [a if a.type == dty else (self.b.sitofp(a, dty) if isinstance(a.type, lir.IntType) else self.b.fpext(a, dty)) for a in args]
        return self.b.call(g, cargs)

    def store_lval(self, t, val):
        if t["k"] == "var":
            slot, _ = self.slots[t["name"]]; self.b.store(val, slot)
        else:
            slot, sty = self.slots[t["array"]]
            ety = _lty(sty["ptr"])
            base = self.b.load(slot, typ=_lty(sty))
            ptr = self.b.gep(base, [self.eval(t["index"])], inbounds=True, source_etype=ety)
            self.b.store(val, ptr)

    def load_lval(self, t):
        if t["k"] == "var":
            slot, sty = self.slots[t["name"]]; return self.b.load(slot, typ=_lty(sty))
        slot, sty = self.slots[t["array"]]
        ety = _lty(sty["ptr"])
        base = self.b.load(slot, typ=_lty(sty))
        ptr = self.b.gep(base, [self.eval(t["index"])], inbounds=True, source_etype=ety)
        return self.b.load(ptr, typ=ety)

    def emit_for(self, s):
        slot, _ = self.slots[s["var"]]
        self.b.store(self.eval(s["start"]), slot)
        cond = self.func.append_basic_block("for.cond"); body = self.func.append_basic_block("for.body"); end = self.func.append_basic_block("for.end")
        self.b.branch(cond); self.b.position_at_end(cond)
        i = self.b.load(slot, typ=lir.IntType(64)); stop = self.eval(s["stop"]); step = self.eval(s["step"])
        zero = lir.Constant(step.type, 0)
        c = self.b.select(self.b.icmp_signed(">", step, zero), self.b.icmp_signed("<", i, stop), self.b.icmp_signed(">", i, stop))
        self.b.cbranch(c, body, end); self.b.position_at_end(body)
        self.emit_block(s["body"])
        if not self.b.block.is_terminated:
            self.b.store(self.b.add(self.b.load(slot, typ=lir.IntType(64)), self.eval(s["step"])), slot); self.b.branch(cond)
        self.b.position_at_end(end)

    def emit_while(self, s):
        cond = self.func.append_basic_block("wh.cond"); body = self.func.append_basic_block("wh.body"); end = self.func.append_basic_block("wh.end")
        self.b.branch(cond); self.b.position_at_end(cond)
        self.b.cbranch(self._to_i1(self.eval(s["cond"])), body, end); self.b.position_at_end(body)
        self.emit_block(s["body"])
        if not self.b.block.is_terminated: self.b.branch(cond)
        self.b.position_at_end(end)

    def emit_if(self, s):
        c = self._to_i1(self.eval(s["cond"]))
        then = self.func.append_basic_block("if.then"); has_else = bool(s["else"])
        els = self.func.append_basic_block("if.else") if has_else else None
        end = self.func.append_basic_block("if.end")
        self.b.cbranch(c, then, els if has_else else end); self.b.position_at_end(then)
        self.emit_block(s["then"])
        if not self.b.block.is_terminated: self.b.branch(end)
        if has_else:
            self.b.position_at_end(els); self.emit_block(s["else"])
            if not self.b.block.is_terminated: self.b.branch(end)
        self.b.position_at_end(end)


class LLVMBackend:
    name = "llvm"

    def available(self) -> bool:
        return _HAVE

    def compile(self, ir: dict):
        if not _HAVE:
            raise CompileError("llvmlite not installed")
        from ..ir import has_parallel_loop
        if has_parallel_loop(ir):
            raise CompileError("llvm backend runs serial; deferring parallel loop to source-gen")
        try:
            _ensure()
            module = lir.Module(name="celeris")
            fnty = lir.FunctionType(_lty(ir["ret"]), [_lty(p["type"]) for p in ir["params"]])
            func = lir.Function(module, fnty, name=ir["name"])
            b = lir.IRBuilder(func.append_basic_block("entry"))
            slots = {}
            for p, arg in zip(ir["params"], func.args):
                arg.name = p["name"]
                slot = b.alloca(_lty(p["type"]), name=p["name"]); b.store(arg, slot)
                slots[p["name"]] = (slot, p["type"])
            locs = {}
            _collect_locals(ir["body"], {p["name"] for p in ir["params"]}, locs)
            for n, t in locs.items():
                slots[n] = (b.alloca(_lty(t), name=n), t)
            lw = _Lowerer(b, func, slots, module)
            lw.emit_block(ir["body"])
            if not b.block.is_terminated:
                b.ret_void() if isinstance(fnty.return_type, lir.VoidType) else b.ret(lir.Constant(fnty.return_type, 0))
            return self._jit(str(module), ir)
        except CompileError:
            raise
        except Exception as exc:
            raise CompileError(f"llvm lowering failed: {exc}") from exc

    def _jit(self, llvm_ir, ir):
        mod = llvm.parse_assembly(llvm_ir); mod.verify()
        tm = llvm.Target.from_default_triple().create_target_machine()
        pto = llvm.PipelineTuningOptions(speed_level=2, size_level=0)
        pb = llvm.create_pass_builder(tm, pto)
        pm = pb.getModulePassManager()
        pm.run(mod, pb)
        ee = llvm.create_mcjit_compiler(mod, tm)
        ee.finalize_object(); ee.run_static_constructors()
        addr = ee.get_function_address(ir["name"])
        cfty = ctypes.CFUNCTYPE(_ct(ir["ret"]), *[_ct(p["type"]) for p in ir["params"]])
        cfn = cfty(addr); ptypes = [p["type"] for p in ir["params"]]

        def call(*args):
            cargs = [(a.ctypes.data_as(_ct(t)) if isinstance(t, dict) and "ptr" in t else a) for t, a in zip(ptypes, args)]
            return cfn(*cargs)
        call._keepalive = (ee, mod)  # prevent GC of the JIT engine
        return call


if _HAVE:
    register(LLVMBackend())
