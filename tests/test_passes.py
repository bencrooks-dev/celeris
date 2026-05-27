import copy

import celeris.ir as ir
from celeris.passes import fold_constants, fuse_loops, optimize

def test_constant_folding_int():
    e = ir.binop("+","i64", ir.const("i64",2), ir.const("i64",3))
    assert fold_constants(e) == ir.const("i64",5)

def test_constant_folding_float_and_nested():
    e = ir.binop("*","f64", ir.binop("+","f64", ir.const("f64",1.0), ir.const("f64",2.0)),
                 ir.const("f64",4.0))
    assert fold_constants(e) == ir.const("f64",12.0)

def test_folding_preserves_nonconst():
    e = ir.binop("+","i64", ir.var("x","i64"), ir.const("i64",0))
    out = fold_constants(e)
    assert out["k"] == "binop" and out["lhs"] == ir.var("x","i64")

def test_dead_assignment_removed():
    body = [ir.assign(ir.lval_var("dead","i64"), ir.const("i64",1)),
            ir.ret(ir.const("i64",0))]
    k = ir.kernel("k", [], "i64", body)
    out = optimize(k)
    assert all(s["op"] != "assign" for s in out["body"])

def test_live_assignment_kept():
    body = [ir.assign(ir.lval_var("s","i64"), ir.const("i64",5)),
            ir.ret(ir.var("s","i64"))]
    out = optimize(ir.kernel("k", [], "i64", body))
    assert any(s["op"] == "assign" for s in out["body"])

def test_optimize_idempotent():
    body = [ir.ret(ir.binop("+","i64", ir.const("i64",2), ir.const("i64",3)))]
    k = ir.kernel("k", [], "i64", body)
    once = optimize(k); twice = optimize(once)
    assert once == twice
    assert once["body"][0]["value"] == ir.const("i64",5)


def _loop(body, var="i", stop_name="n"):
    return ir.for_(var, ir.const("i64", 0), ir.var(stop_name, "i64"),
                   ir.const("i64", 1), body)

def _idx_assign(dst, src_arr, idx_expr):
    return ir.assign(ir.lval_index(dst, ir.var("i", "i64"), "f64"),
                     ir.index(src_arr, idx_expr, "f64"))

def _kernel(body):
    params = [ir.param("n", "i64"), ir.param("x", {"ptr": "f64"}),
              ir.param("y", {"ptr": "f64"}), ir.param("z", {"ptr": "f64"}),
              ir.param("b", {"ptr": "f64"})]
    return ir.kernel("k", params, "void", body)


def test_fuse_adjacent_elementwise():
    # for i: y[i]=x[i] ; for i: z[i]=y[i]  -> one loop, body of 2 (y written@i, read@i: ok)
    l1 = _loop([_idx_assign("y", "x", ir.var("i", "i64"))])
    l2 = _loop([_idx_assign("z", "y", ir.var("i", "i64"))])
    out = fuse_loops(_kernel([l1, l2]))
    fors = [s for s in out["body"] if s["op"] == "for"]
    assert len(fors) == 1 and len(fors[0]["body"]) == 2

def test_fuse_fixpoint_three_loops():
    loops = [_loop([_idx_assign("y", "x", ir.var("i", "i64"))]),
             _loop([_idx_assign("z", "y", ir.var("i", "i64"))]),
             _loop([_idx_assign("y", "z", ir.var("i", "i64"))])]
    out = fuse_loops(_kernel(loops))
    fors = [s for s in out["body"] if s["op"] == "for"]
    assert len(fors) == 1 and len(fors[0]["body"]) == 3

def test_affine_backward_offset_write_then_read_fuses():
    # y written at i+1 (cx=1), y read at i (cy=0) -> 0<=1 -> fuse (flow dep preserved).
    # NOTE: under the v0.2 strict exactly-i rule this declined; the v0.3 affine
    # dependence test (cy<=cx) legally fuses it -- a strict superset relaxation.
    off = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("y", off, "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([_idx_assign("z", "y", ir.var("i", "i64"))])
    out = fuse_loops(_kernel([l1, l2]))
    assert len([s for s in out["body"] if s["op"] == "for"]) == 1

def test_decline_shared_accumulator():
    # for i: acc=acc+x[i] ; for i: y[i]=acc  -> writes(L1)={acc} ∩ refs(L2)={acc} -> decline
    l1 = _loop([ir.assign(ir.lval_var("acc", "f64"),
                          ir.binop("+", "f64", ir.var("acc", "f64"),
                                   ir.index("x", ir.var("i", "i64"), "f64")))])
    l2 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.var("acc", "f64"))])
    out = fuse_loops(_kernel([l1, l2]))
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_decline_return_in_body():
    l1 = _loop([ir.ret(ir.const("i64", 0))])
    l2 = _loop([_idx_assign("z", "y", ir.var("i", "i64"))])
    out = fuse_loops(_kernel([l1, l2]))
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_decline_different_iteration_space():
    l1 = _loop([_idx_assign("y", "x", ir.var("i", "i64"))], stop_name="n")
    l2 = _loop([_idx_assign("z", "x", ir.var("i", "i64"))], stop_name="m")
    k = _kernel([l1, l2])
    k["params"].append(ir.param("m", "i64"))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_readonly_array_offset_allowed():
    # both loops only READ b (read-only) with an offset; write y,z by i -> fusable
    off = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.index("b", off, "f64"))])
    l2 = _loop([_idx_assign("z", "b", ir.var("i", "i64"))])
    out = fuse_loops(_kernel([l1, l2]))
    assert len([s for s in out["body"] if s["op"] == "for"]) == 1

def test_fuse_nonmutating_and_idempotent():
    l1 = _loop([_idx_assign("y", "x", ir.var("i", "i64"))])
    l2 = _loop([_idx_assign("z", "y", ir.var("i", "i64"))])
    k = _kernel([l1, l2])
    before = copy.deepcopy(k)
    out1 = optimize(k)
    assert k == before                       # optimize did not mutate input
    assert out1 == optimize(out1)            # idempotent
    assert len([s for s in out1["body"] if s["op"] == "for"]) == 1  # fused via optimize


def _wr(dst, idx_expr):
    return ir.assign(ir.lval_index(dst, idx_expr, "f64"), ir.index("x", ir.var("i", "i64"), "f64"))

def test_affine_backward_offset_fuses():
    # L1: t[i]=x[i] ; L2: y[i]=t[i]+t[i] but read t at offset 0 only -> already fuses;
    # exercise true affine: L1 writes t[i+1] (cx=1), L2 reads t[i] (cy=0) -> 0<=1 fuse
    off1 = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("t", off1, "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.index("t", ir.var("i", "i64"), "f64"))])
    k = _kernel([l1, l2]); k["params"].append(ir.param("t", {"ptr": "f64"}))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 1

def test_affine_forward_offset_read_declines():
    # L1 writes t[i] (cx=0) ; L2 reads t[i+1] (cy=1) -> 1<=0 false -> decline
    off1 = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("t", ir.var("i", "i64"), "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.index("t", off1, "f64"))])
    k = _kernel([l1, l2]); k["params"].append(ir.param("t", {"ptr": "f64"}))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_affine_variable_offset_declines():
    # t[i+k] where k is a variable -> non-affine -> decline
    offk = ir.binop("+", "i64", ir.var("i", "i64"), ir.var("k", "i64"))
    l1 = _loop([ir.assign(ir.lval_index("t", offk, "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                          ir.index("t", ir.var("i", "i64"), "f64"))])
    k = _kernel([l1, l2])
    for p in ("t", "k"):
        k["params"].append(ir.param(p, {"ptr": "f64"} if p == "t" else "i64"))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_affine_nonunit_step_with_offset_declines():
    # step=2 (non-unit) with a written-array offset -> strict fallback -> decline
    off1 = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    def l2step(body):
        return ir.for_("i", ir.const("i64", 0), ir.var("n", "i64"), ir.const("i64", 2), body)
    l1 = l2step([ir.assign(ir.lval_index("t", off1, "f64"),
                           ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = l2step([ir.assign(ir.lval_index("y", ir.var("i", "i64"), "f64"),
                           ir.index("t", ir.var("i", "i64"), "f64"))])
    k = _kernel([l1, l2]); k["params"].append(ir.param("t", {"ptr": "f64"}))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2

def test_affine_output_dep_offset_fuses():
    # L1 writes t[i+1] (cx=1), L2 writes t[i] (cy=0) -> 0<=1 fuse (output dep ok)
    off1 = ir.binop("+", "i64", ir.var("i", "i64"), ir.const("i64", 1))
    l1 = _loop([ir.assign(ir.lval_index("t", off1, "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    l2 = _loop([ir.assign(ir.lval_index("t", ir.var("i", "i64"), "f64"),
                          ir.index("x", ir.var("i", "i64"), "f64"))])
    k = _kernel([l1, l2]); k["params"].append(ir.param("t", {"ptr": "f64"}))
    out = fuse_loops(k)
    assert len([s for s in out["body"] if s["op"] == "for"]) == 1


def test_fuse_two_parallel_loops_keeps_parallel():
    l1 = ir.for_("i", ir.const("i64",0), ir.var("n","i64"), ir.const("i64",1),
                 [_idx_assign("y", "x", ir.var("i","i64"))], parallel=True)
    l2 = ir.for_("i", ir.const("i64",0), ir.var("n","i64"), ir.const("i64",1),
                 [_idx_assign("z", "x", ir.var("i","i64"))], parallel=True)
    k = _kernel([l1, l2])
    out = fuse_loops(k)
    fors = [s for s in out["body"] if s["op"] == "for"]
    assert len(fors) == 1 and fors[0].get("parallel") is True

def test_no_fuse_parallel_with_serial():
    l1 = ir.for_("i", ir.const("i64",0), ir.var("n","i64"), ir.const("i64",1),
                 [_idx_assign("y", "x", ir.var("i","i64"))], parallel=True)
    l2 = ir.for_("i", ir.const("i64",0), ir.var("n","i64"), ir.const("i64",1),
                 [_idx_assign("z", "x", ir.var("i","i64"))], parallel=False)
    out = fuse_loops(_kernel([l1, l2]))
    assert len([s for s in out["body"] if s["op"] == "for"]) == 2
