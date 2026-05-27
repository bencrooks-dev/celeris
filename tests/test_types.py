from celeris.types import (annotation_to_type, unify_numeric, is_float, is_int,
                           F64Array, I64Array, F32Array, I32Array)


def test_scalar_annotations():
    assert annotation_to_type("int") == "i64"
    assert annotation_to_type("float") == "f64"
    assert annotation_to_type("f32") == "f32"
    assert annotation_to_type("i32") == "i32"


def test_array_annotations():
    assert annotation_to_type("F64Array") == {"ptr": "f64"}
    assert annotation_to_type("F32Array") == {"ptr": "f32"}
    assert annotation_to_type("I64Array") == {"ptr": "i64"}
    assert annotation_to_type("I32Array") == {"ptr": "i32"}


def test_none_annotation():
    assert annotation_to_type("None") == "void"


def test_unknown_annotation_raises():
    import pytest
    from celeris.errors import UnsupportedFeature
    with pytest.raises(UnsupportedFeature):
        annotation_to_type("Banana")


def test_promotion():
    assert unify_numeric("i64", "i64") == "i64"
    assert unify_numeric("i64", "f64") == "f64"
    assert unify_numeric("f32", "f64") == "f64"
    assert unify_numeric("i32", "i64") == "i64"


def test_predicates():
    assert is_float("f64") and is_float("f32")
    assert is_int("i32") and is_int("i64")
    assert not is_float("i64")
    assert not is_int("f64")


def test_array2d_annotations():
    from celeris.types import annotation_to_type
    assert annotation_to_type("F64Array2D") == {"ptr": "f64", "ndim": 2}
    assert annotation_to_type("I64Array2D") == {"ptr": "i64", "ndim": 2}


def test_ndim_of():
    from celeris.types import ndim_of
    assert ndim_of({"ptr": "f64"}) == 1
    assert ndim_of({"ptr": "f64", "ndim": 2}) == 2
