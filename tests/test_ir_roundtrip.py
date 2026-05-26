import celeris.ir as ir


def test_constructors_and_roundtrip():
    e = ir.binop("+", "f64", ir.var("a", "f64"), ir.const("f64", 1.0))
    assert e["k"] == "binop" and e["type"] == "f64"
    assert e["lhs"] == {"k": "var", "type": "f64", "name": "a"}
    assert e["rhs"] == {"k": "const", "type": "f64", "value": 1.0}
    kern = ir.kernel("k", [ir.param("a", "f64")], "f64", [ir.ret(ir.var("a", "f64"))])
    blob = ir.dumps(kern)
    assert ir.loads(blob) == kern
    assert ir.SCHEMA_VERSION >= 1


def test_for_node():
    f = ir.for_("i", ir.const("i64", 0), ir.var("n", "i64"), ir.const("i64", 1), [])
    assert f["op"] == "for" and f["var"] == "i"


def test_index_and_lval():
    idx = ir.index("x", ir.var("i", "i64"), "f64")
    assert idx == {"k": "index", "type": "f64", "array": "x", "index": {"k": "var", "type": "i64", "name": "i"}}
    tgt = ir.lval_index("y", ir.var("i", "i64"), "f64")
    assert tgt["k"] == "index" and tgt["array"] == "y" and tgt["type"] == "f64"


def test_assign_and_if():
    a = ir.assign(ir.lval_var("s", "f64"), ir.const("f64", 0.0))
    assert a == {"op": "assign", "target": {"k": "var", "name": "s", "type": "f64"}, "value": {"k": "const", "type": "f64", "value": 0.0}}
    b = ir.if_(ir.cmp("<", ir.var("i", "i64"), ir.const("i64", 3)), [], [])
    assert b["op"] == "if" and "then" in b and "else" in b
