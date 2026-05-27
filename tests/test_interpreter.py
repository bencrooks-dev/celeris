import numpy as np
from celeris.backends.interpreter import InterpreterBackend

SAXPY_IR = {"name":"saxpy","ret":"void",
  "params":[{"name":"a","type":"f64"},{"name":"x","type":{"ptr":"f64"}},
            {"name":"y","type":{"ptr":"f64"}},{"name":"n","type":"i64"}],
  "body":[{"op":"for","var":"i",
    "start":{"k":"const","type":"i64","value":0},
    "stop":{"k":"var","type":"i64","name":"n"},
    "step":{"k":"const","type":"i64","value":1},
    "body":[{"op":"assign",
      "target":{"k":"index","array":"y","type":"f64",
                "index":{"k":"var","type":"i64","name":"i"}},
      "value":{"k":"binop","op":"+","type":"f64",
        "lhs":{"k":"binop","op":"*","type":"f64",
          "lhs":{"k":"var","type":"f64","name":"a"},
          "rhs":{"k":"index","array":"x","type":"f64",
                 "index":{"k":"var","type":"i64","name":"i"}}},
        "rhs":{"k":"index","array":"y","type":"f64",
               "index":{"k":"var","type":"i64","name":"i"}}}}]}]}

def test_interpreter_saxpy():
    fn = InterpreterBackend().compile(SAXPY_IR)
    x = np.arange(5, dtype=np.float64); y = np.ones(5, dtype=np.float64)
    fn(2.0, x, y, 5)
    np.testing.assert_allclose(y, 2.0*np.arange(5)+1.0)

def test_interpreter_always_available():
    assert InterpreterBackend().available() is True

def test_interpreter_scalar_return():
    ir = {"name":"f","ret":"f64",
      "params":[{"name":"a","type":"f64"},{"name":"b","type":"f64"}],
      "body":[{"op":"return","value":{"k":"binop","op":"+","type":"f64",
        "lhs":{"k":"binop","op":"*","type":"f64",
          "lhs":{"k":"var","type":"f64","name":"a"},
          "rhs":{"k":"var","type":"f64","name":"b"}},
        "rhs":{"k":"const","type":"f64","value":1.0}}}]}
    assert abs(InterpreterBackend().compile(ir)(3.0,4.0) - 13.0) < 1e-12

def test_interpreter_while_and_if():
    # s=0; i=0; while i<n: if i%2==0: s+=i ; i+=1 ; return s   (n via reduction)
    ir = {"name":"f","ret":"i64","params":[{"name":"n","type":"i64"}],
      "body":[
        {"op":"assign","target":{"k":"var","name":"s","type":"i64"},
         "value":{"k":"const","type":"i64","value":0}},
        {"op":"assign","target":{"k":"var","name":"i","type":"i64"},
         "value":{"k":"const","type":"i64","value":0}},
        {"op":"while","cond":{"k":"cmp","op":"<","type":"i1",
            "lhs":{"k":"var","type":"i64","name":"i"},
            "rhs":{"k":"var","type":"i64","name":"n"}},
         "body":[
           {"op":"if","cond":{"k":"cmp","op":"==","type":"i1",
              "lhs":{"k":"binop","op":"%","type":"i64",
                "lhs":{"k":"var","type":"i64","name":"i"},
                "rhs":{"k":"const","type":"i64","value":2}},
              "rhs":{"k":"const","type":"i64","value":0}},
            "then":[{"op":"augassign","binop":"+",
               "target":{"k":"var","name":"s","type":"i64"},
               "value":{"k":"var","type":"i64","name":"i"}}],
            "else":[]},
           {"op":"augassign","binop":"+",
            "target":{"k":"var","name":"i","type":"i64"},
            "value":{"k":"const","type":"i64","value":1}}]},
        {"op":"return","value":{"k":"var","type":"i64","name":"s"}}]}
    assert InterpreterBackend().compile(ir)(6) == 0+2+4

def test_interpreter_cmp_returns_int():
    from celeris.backends.interpreter import _compare, _boolop
    assert _compare("<", 1, 2) == 1 and isinstance(_compare("<",1,2), int)
    assert _boolop("not", [0]) == 1 and isinstance(_boolop("not",[0]), int)

def test_interpreter_negative_step_for():
    ir = {"name":"f","ret":"i64","params":[],
      "body":[
        {"op":"assign","target":{"k":"var","name":"s","type":"i64"},
         "value":{"k":"const","type":"i64","value":0}},
        {"op":"for","var":"i",
         "start":{"k":"const","type":"i64","value":5},
         "stop":{"k":"const","type":"i64","value":0},
         "step":{"k":"const","type":"i64","value":-1},
         "body":[{"op":"augassign","binop":"+",
           "target":{"k":"var","name":"s","type":"i64"},
           "value":{"k":"var","type":"i64","name":"i"}}]},
        {"op":"return","value":{"k":"var","type":"i64","name":"s"}}]}
    from celeris.backends.interpreter import InterpreterBackend
    assert InterpreterBackend().compile(ir)() == 5+4+3+2+1

def test_interpreter_step_zero_raises():
    import pytest
    ir = {"name":"f","ret":"void","params":[],
      "body":[{"op":"for","var":"i",
        "start":{"k":"const","type":"i64","value":0},
        "stop":{"k":"const","type":"i64","value":3},
        "step":{"k":"const","type":"i64","value":0},"body":[]}]}
    from celeris.backends.interpreter import InterpreterBackend
    with pytest.raises(ValueError):
        InterpreterBackend().compile(ir)()

def test_interpreter_2d_index():
    import celeris.ir as ir
    import numpy as np

    from celeris.backends.interpreter import InterpreterBackend
    # y[i] = a[i,0] + a[i,1]
    body = [ir.for_("i", ir.const("i64",0), ir.var("m","i64"), ir.const("i64",1),
        [ir.assign(ir.lval_index("y", ir.var("i","i64"), "f64"),
            ir.binop("+","f64",
                ir.index_nd("a", [ir.var("i","i64"), ir.const("i64",0)], "f64"),
                ir.index_nd("a", [ir.var("i","i64"), ir.const("i64",1)], "f64")))])]
    k = ir.kernel("k", [ir.param("a",{"ptr":"f64","ndim":2}), ir.param("y",{"ptr":"f64"}), ir.param("m","i64")], "void", body)
    fn = InterpreterBackend().compile(k)
    a = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.zeros(3, dtype=np.float64)
    fn(a, y, 3)
    np.testing.assert_allclose(y, a[:, 0] + a[:, 1])
