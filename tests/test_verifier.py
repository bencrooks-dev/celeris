import pytest, celeris.ir as ir
from celeris.verifier import verify_ir
from celeris.errors import VerifyError

def test_valid_scalar_kernel_passes():
    k = ir.kernel("k", [ir.param("a","f64")], "f64", [ir.ret(ir.var("a","f64"))])
    verify_ir(k)  # no raise

def test_valid_saxpy_passes():
    k = ir.kernel("saxpy",
        [ir.param("a","f64"), ir.param("x",{"ptr":"f64"}),
         ir.param("y",{"ptr":"f64"}), ir.param("n","i64")], "void",
        [ir.for_("i", ir.const("i64",0), ir.var("n","i64"), ir.const("i64",1),
            [ir.assign(ir.lval_index("y", ir.var("i","i64"), "f64"),
                ir.binop("+","f64",
                    ir.binop("*","f64", ir.var("a","f64"),
                             ir.index("x", ir.var("i","i64"), "f64")),
                    ir.index("y", ir.var("i","i64"), "f64")))])])
    verify_ir(k)

def test_param_missing_type_rejected():
    bad = {"name":"k","params":[{"name":"a"}],"ret":"f64","body":[]}
    with pytest.raises(VerifyError):
        verify_ir(bad)

def test_index_into_scalar_rejected():
    k = ir.kernel("k", [ir.param("a","f64")], "f64",
        [ir.ret(ir.index("a", ir.const("i64",0), "f64"))])  # 'a' is scalar, not ptr
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_unknown_intrinsic_rejected():
    k = ir.kernel("k", [ir.param("a","f64")], "f64",
        [ir.ret(ir.call("frobnicate","f64",[ir.var("a","f64")]))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_missing_kernel_fields_rejected():
    with pytest.raises(VerifyError):
        verify_ir({"name":"k","body":[]})  # no params / ret


# --- additional coverage ------------------------------------------------------

def test_non_dict_kernel_rejected():
    with pytest.raises(VerifyError):
        verify_ir(["not", "a", "dict"])

def test_name_not_str_rejected():
    with pytest.raises(VerifyError):
        verify_ir({"name":123,"params":[],"ret":"void","body":[]})

def test_params_not_list_rejected():
    with pytest.raises(VerifyError):
        verify_ir({"name":"k","params":"a","ret":"void","body":[]})

def test_body_not_list_rejected():
    with pytest.raises(VerifyError):
        verify_ir({"name":"k","params":[],"ret":"void","body":"x"})

def test_invalid_ret_type_rejected():
    with pytest.raises(VerifyError):
        verify_ir({"name":"k","params":[],"ret":"banana","body":[]})

def test_param_invalid_type_rejected():
    with pytest.raises(VerifyError):
        verify_ir({"name":"k","params":[{"name":"a","type":"banana"}],"ret":"void","body":[]})

def test_ptr_to_void_rejected():
    with pytest.raises(VerifyError):
        verify_ir({"name":"k","params":[{"name":"a","type":{"ptr":"void"}}],"ret":"void","body":[]})

def test_void_value_type_rejected():
    # a const can't carry void
    k = ir.kernel("k", [], "void", [ir.assign(ir.lval_var("s","void"), ir.const("void",0))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_expr_missing_type_rejected():
    bad_expr = {"k":"const","value":1.0}  # no type
    k = ir.kernel("k", [], "f64", [ir.ret(bad_expr)])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_unknown_expr_kind_rejected():
    k = ir.kernel("k", [], "f64", [ir.ret({"k":"weird","type":"f64"})])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_unknown_stmt_op_rejected():
    k = ir.kernel("k", [], "void", [{"op":"goto","target":"x"}])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_index_subexpr_not_int_rejected():
    k = ir.kernel("k", [ir.param("x",{"ptr":"f64"})], "f64",
        [ir.ret(ir.index("x", ir.const("f64",0.0), "f64"))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_index_wrong_element_type_rejected():
    # ptr is f64 but index node claims i32 element type
    k = ir.kernel("k", [ir.param("x",{"ptr":"f64"})], "i32",
        [ir.ret(ir.index("x", ir.const("i64",0), "i32"))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_index_unknown_array_rejected():
    k = ir.kernel("k", [], "f64",
        [ir.ret(ir.index("nope", ir.const("i64",0), "f64"))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_binop_wrong_result_type_rejected():
    # i64 + f64 should unify to f64, claiming i64 is wrong
    k = ir.kernel("k", [], "i64",
        [ir.ret(ir.binop("+","i64", ir.const("i64",1), ir.const("f64",1.0)))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_binop_valid_promotion_passes():
    k = ir.kernel("k", [], "f64",
        [ir.ret(ir.binop("+","f64", ir.const("i64",1), ir.const("f64",1.0)))])
    verify_ir(k)

def test_cmp_must_be_i1():
    bad = {"k":"cmp","op":"<","type":"f64","lhs":ir.const("i64",1),"rhs":ir.const("i64",2)}
    k = ir.kernel("k", [], "i1", [ir.ret(bad)])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_cmp_valid_passes():
    k = ir.kernel("k", [], "i1",
        [ir.ret(ir.cmp("<", ir.const("i64",1), ir.const("i64",2)))])
    verify_ir(k)

def test_bool_must_be_i1():
    bad = {"k":"bool","op":"and","type":"i64","args":[ir.cmp("<", ir.const("i64",1), ir.const("i64",2))]}
    k = ir.kernel("k", [], "i1", [ir.ret(bad)])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_bool_valid_passes():
    e = ir.boolop("and", [ir.cmp("<", ir.const("i64",1), ir.const("i64",2)),
                          ir.cmp(">", ir.const("i64",3), ir.const("i64",1))])
    k = ir.kernel("k", [], "i1", [ir.ret(e)])
    verify_ir(k)

def test_call_valid_intrinsic_passes():
    k = ir.kernel("k", [ir.param("a","f64")], "f64",
        [ir.ret(ir.call("sqrt","f64",[ir.var("a","f64")]))])
    verify_ir(k)

def test_call_wrong_arg_count_rejected():
    k = ir.kernel("k", [ir.param("a","f64")], "f64",
        [ir.ret(ir.call("sqrt","f64",[ir.var("a","f64"), ir.var("a","f64")]))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_call_fmax_two_args_passes():
    k = ir.kernel("k", [ir.param("a","f64"), ir.param("b","f64")], "f64",
        [ir.ret(ir.call("fmax","f64",[ir.var("a","f64"), ir.var("b","f64")]))])
    verify_ir(k)

def test_call_fmax_wrong_arg_count_rejected():
    k = ir.kernel("k", [ir.param("a","f64")], "f64",
        [ir.ret(ir.call("fmax","f64",[ir.var("a","f64")]))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_call_len_one_arg_passes():
    k = ir.kernel("k", [ir.param("x",{"ptr":"f64"})], "i64",
        [ir.ret(ir.call("len","i64",[ir.var("x",{"ptr":"f64"})]))])
    verify_ir(k)

def test_cast_valid_passes():
    k = ir.kernel("k", [ir.param("a","i64")], "f64",
        [ir.ret(ir.cast("f64", ir.var("a","i64")))])
    verify_ir(k)

def test_cast_to_void_rejected():
    k = ir.kernel("k", [ir.param("a","i64")], "void",
        [ir.ret(ir.cast("void", ir.var("a","i64")))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_for_non_int_bound_rejected():
    k = ir.kernel("k", [], "void",
        [ir.for_("i", ir.const("f64",0.0), ir.const("i64",10), ir.const("i64",1), [])])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_for_loop_var_is_i64_in_body():
    # loop var 'i' must be usable as i64 in the body
    k = ir.kernel("k", [ir.param("x",{"ptr":"f64"})], "void",
        [ir.for_("i", ir.const("i64",0), ir.const("i64",10), ir.const("i64",1),
            [ir.assign(ir.lval_index("x", ir.var("i","i64"), "f64"), ir.const("f64",1.0))])])
    verify_ir(k)

def test_while_valid_passes():
    k = ir.kernel("k", [ir.param("n","i64")], "void",
        [ir.while_(ir.cmp("<", ir.const("i64",0), ir.var("n","i64")), [])])
    verify_ir(k)

def test_if_valid_passes():
    k = ir.kernel("k", [], "void",
        [ir.if_(ir.cmp("<", ir.const("i64",1), ir.const("i64",2)),
                [ir.assign(ir.lval_var("s","f64"), ir.const("f64",1.0))],
                [ir.assign(ir.lval_var("t","f64"), ir.const("f64",2.0))])])
    verify_ir(k)

def test_local_var_defined_by_assign_then_used():
    k = ir.kernel("k", [], "f64",
        [ir.assign(ir.lval_var("s","f64"), ir.const("f64",0.0)),
         ir.ret(ir.var("s","f64"))])
    verify_ir(k)

def test_unknown_var_reference_rejected():
    k = ir.kernel("k", [], "f64", [ir.ret(ir.var("ghost","f64"))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_var_type_mismatch_rejected():
    k = ir.kernel("k", [ir.param("a","f64")], "f64",
        [ir.ret(ir.var("a","i64"))])  # 'a' is f64, claims i64
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_augassign_valid_passes():
    k = ir.kernel("k", [], "f64",
        [ir.assign(ir.lval_var("s","f64"), ir.const("f64",0.0)),
         ir.augassign("+", ir.lval_var("s","f64"), ir.const("f64",1.0)),
         ir.ret(ir.var("s","f64"))])
    verify_ir(k)

def test_assign_index_target_valid_passes():
    k = ir.kernel("k", [ir.param("x",{"ptr":"f64"})], "void",
        [ir.assign(ir.lval_index("x", ir.const("i64",0), "f64"), ir.const("f64",1.0))])
    verify_ir(k)

def test_assign_index_into_scalar_target_rejected():
    k = ir.kernel("k", [ir.param("a","f64")], "void",
        [ir.assign(ir.lval_index("a", ir.const("i64",0), "f64"), ir.const("f64",1.0))])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_return_none_passes():
    k = ir.kernel("k", [], "void", [ir.ret(None)])
    verify_ir(k)

def test_while_int_cond_passes():
    k = ir.kernel("k", [ir.param("n","i64")], "void",
        [ir.while_(ir.var("n","i64"), [])])
    verify_ir(k)

def test_cmp_over_pointers_rejected():
    k = ir.kernel("k",
        [ir.param("x",{"ptr":"f64"}), ir.param("y",{"ptr":"f64"})], "void",
        [ir.if_(ir.cmp("==", ir.var("x",{"ptr":"f64"}), ir.var("y",{"ptr":"f64"})), [], [])])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_bool_over_pointer_rejected():
    k = ir.kernel("k", [ir.param("x",{"ptr":"f64"})], "void",
        [ir.if_(ir.boolop("not",[ir.var("x",{"ptr":"f64"})]), [], [])])
    with pytest.raises(VerifyError):
        verify_ir(k)

def test_cmp_over_scalars_still_passes():
    k = ir.kernel("k", [ir.param("a","i64"), ir.param("b","i64")], "void",
        [ir.if_(ir.cmp("<", ir.var("a","i64"), ir.var("b","i64")), [], [])])
    verify_ir(k)  # no raise
