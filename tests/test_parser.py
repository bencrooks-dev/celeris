import pytest
from celeris.parser import parse_function
from celeris.types import F64Array
from celeris.errors import UnsupportedFeature

def test_saxpy_ir():
    def saxpy(a: float, x: F64Array, y: F64Array, n: int) -> None:
        for i in range(n):
            y[i] = a * x[i] + y[i]
    irk = parse_function(saxpy)
    assert irk["name"] == "saxpy"
    assert [p["type"] for p in irk["params"]] == ["f64", {"ptr":"f64"}, {"ptr":"f64"}, "i64"]
    assert irk["ret"] == "void"
    assert irk["body"][0]["op"] == "for"

def test_scalar_return():
    def f(a: float, b: float) -> float:
        return a*b + 1.0
    irk = parse_function(f)
    assert irk["ret"] == "f64" and irk["body"][-1]["op"] == "return"

def test_rejects_classes():
    def bad(x: int) -> int:
        class C: ...
        return x
    with pytest.raises(UnsupportedFeature):
        parse_function(bad)

def test_rejects_unknown_call():
    def bad(x: float) -> float:
        return print(x)
    with pytest.raises(UnsupportedFeature):
        parse_function(bad)

def test_rejects_multidim_index():
    def bad(x: F64Array) -> float:
        return x[1, 2]
    with pytest.raises(UnsupportedFeature):
        parse_function(bad)

def test_requires_annotations():
    def bad(x) -> int:
        return x
    with pytest.raises(UnsupportedFeature):
        parse_function(bad)

def test_while_and_if():
    def f(n: int) -> int:
        s = 0
        i = 0
        while i < n:
            if i % 2 == 0:
                s = s + i
            i = i + 1
        return s
    irk = parse_function(f)
    ops = [st["op"] for st in irk["body"]]
    assert "while" in ops and irk["ret"] == "i64"

def test_augassign_undefined_name_raises():
    from celeris.errors import CelerisError
    def bad(n: int) -> int:
        x += 1  # noqa: F821
        return x
    with pytest.raises(CelerisError):
        parse_function(bad)

def test_true_division_is_f64():
    def f(a: int, b: int) -> float:
        return a / b
    irk = parse_function(f)
    assert irk["body"][-1]["value"]["type"] == "f64"


def test_return_int_const_from_float_fn_is_cast():
    def f() -> float:
        return 0
    irk = parse_function(f)
    assert irk["body"][-1]["value"]["type"] == "f64"


def test_cmp_result_returned_as_int_is_cast():
    def f(a: int, b: int) -> int:
        c = a < b
        return c
    irk = parse_function(f)
    assert irk["body"][-1]["value"]["type"] == "i64"
