import celeris.ir as ir
from celeris.passes import optimize, fold_constants

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
