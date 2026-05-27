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


def test_for_parallel_flag():
    import celeris.ir as ir
    f = ir.for_("i", ir.const("i64", 0), ir.var("n", "i64"), ir.const("i64", 1), [], parallel=True)
    assert f["parallel"] is True
    f2 = ir.for_("i", ir.const("i64", 0), ir.var("n", "i64"), ir.const("i64", 1), [])
    assert f2["parallel"] is False

def test_has_parallel_loop():
    import celeris.ir as ir
    par = ir.for_("i", ir.const("i64",0), ir.var("n","i64"), ir.const("i64",1), [], parallel=True)
    ser = ir.for_("i", ir.const("i64",0), ir.var("n","i64"), ir.const("i64",1), [])
    assert ir.has_parallel_loop(ir.kernel("k", [], "void", [par])) is True
    assert ir.has_parallel_loop(ir.kernel("k", [], "void", [ser])) is False
    assert ir.has_parallel_loop(ir.kernel("k", [], "void", [ir.if_(ir.cmp("<", ir.const("i64",0), ir.const("i64",1)), [par], [])])) is True


def test_index_nd():
    import celeris.ir as ir
    e = ir.index_nd("a", [ir.var("i","i64"), ir.var("j","i64")], "f64")
    assert e == {"k":"index","type":"f64","array":"a",
                 "indices":[{"k":"var","type":"i64","name":"i"},{"k":"var","type":"i64","name":"j"}]}
    lv = ir.lval_index_nd("a", [ir.var("i","i64"), ir.var("j","i64")], "f64")
    assert lv["k"] == "index" and lv["array"] == "a" and len(lv["indices"]) == 2
