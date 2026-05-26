import pytest
import subprocess
import ctypes
import pathlib
import json
import numpy as np
from conftest import needs_clang

pytestmark = needs_clang

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "src/celeris/_native/celeris_core.cpp"
INC = ROOT / "src/celeris/_native/third_party"


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    out = tmp_path_factory.mktemp("native") / "libceleris_core.so"
    subprocess.run(["clang++", "-O3", "-std=c++17", "-fPIC", "-shared",
                    f"-I{INC}", str(CORE), "-o", str(out)], check=True)
    return ctypes.CDLL(str(out))


def test_saxpy_symbol(lib):
    lib.celeris_saxpy.restype = None
    lib.celeris_saxpy.argtypes = [ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int64]
    x = np.arange(4, dtype=np.float64); y = np.ones(4, dtype=np.float64)
    lib.celeris_saxpy(3.0, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 4)
    np.testing.assert_allclose(y, 3.0 * np.arange(4) + 1.0)


def test_compile_strategy_saxpy(lib):
    lib.celeris_compile.restype = ctypes.c_void_p
    lib.celeris_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.celeris_strategy.restype = ctypes.c_int
    lib.celeris_strategy.argtypes = [ctypes.c_void_p]
    lib.celeris_free.argtypes = [ctypes.c_void_p]
    saxpy_ir = json.dumps({"name": "saxpy", "ret": "void", "params": [],
        "body": [{"op": "for", "var": "i", "body": [
            {"op": "assign", "target": {"k": "index"},
             "value": {"k": "binop", "op": "+"}}]}]}).encode()
    err = ctypes.create_string_buffer(256)
    h = lib.celeris_compile(saxpy_ir, err, 256)
    assert h, err.value.decode()
    assert lib.celeris_strategy(h) == 1   # STRAT_HANDWRITTEN
    lib.celeris_free(h)


def test_compile_unsupported_strategy(lib):
    lib.celeris_compile.restype = ctypes.c_void_p
    lib.celeris_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.celeris_strategy.restype = ctypes.c_int
    lib.celeris_strategy.argtypes = [ctypes.c_void_p]
    lib.celeris_free.argtypes = [ctypes.c_void_p]
    other = json.dumps({"name": "f", "ret": "f64", "params": [],
        "body": [{"op": "return", "value": {"k": "const"}}]}).encode()
    err = ctypes.create_string_buffer(256)
    h = lib.celeris_compile(other, err, 256)
    assert h, err.value.decode()
    assert lib.celeris_strategy(h) == 0   # STRAT_UNSUPPORTED (no LLVM built)
    lib.celeris_free(h)


def test_compile_bad_json_returns_null(lib):
    lib.celeris_compile.restype = ctypes.c_void_p
    lib.celeris_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    err = ctypes.create_string_buffer(256)
    h = lib.celeris_compile(b"{not json", err, 256)
    assert not h
    assert b"JSON" in err.value or len(err.value) > 0
