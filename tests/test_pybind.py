import pytest
import subprocess
import sys
import pathlib
import importlib.util
import numpy as np
from conftest import needs_clang

pybind11 = pytest.importorskip("pybind11")  # skip if pybind11 absent
pytestmark = needs_clang

ROOT = pathlib.Path(__file__).resolve().parents[1]
NATIVE = ROOT / "src/celeris/_native"

@pytest.fixture(scope="module")
def mod(tmp_path_factory):
    build = tmp_path_factory.mktemp("pybind")
    so = build / "celeris_native.so"
    includes = subprocess.run([sys.executable, "-m", "pybind11", "--includes"],
                              capture_output=True, text=True, check=True).stdout.split()
    cmd = ["clang++", "-O3", "-std=c++17", "-shared", "-fPIC", *includes,
           f"-I{NATIVE}", f"-I{NATIVE/'third_party'}",
           str(NATIVE / "bindings.cpp"), str(NATIVE / "celeris_core.cpp"),
           "-o", str(so)]
    if sys.platform == "darwin":
        cmd[5:5] = ["-undefined", "dynamic_lookup"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location("celeris_native", so)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_pybind_saxpy(mod):
    x = np.arange(5, dtype=np.float64); y = np.ones(5, dtype=np.float64)
    mod.saxpy(2.0, x, y)
    np.testing.assert_allclose(y, 2.0 * np.arange(5) + 1.0)

def test_pybind_compile_strategy(mod):
    import json
    ir = json.dumps({"name":"saxpy","ret":"void","params":[],
        "body":[{"op":"for","var":"i","body":[
          {"op":"assign","target":{"k":"index"},"value":{"k":"binop","op":"+"}}]}]})
    assert mod.compile_strategy(ir) == 1  # STRAT_HANDWRITTEN

def test_pybind_saxpy_rejects_wrong_dtype_y(mod):
    x = np.arange(4, dtype=np.float64); y = np.zeros(4, dtype=np.int64)
    with pytest.raises(Exception):
        mod.saxpy(1.0, x, y)

def test_pybind_saxpy_rejects_length_mismatch(mod):
    x = np.arange(4, dtype=np.float64); y = np.zeros(5, dtype=np.float64)
    with pytest.raises(Exception):
        mod.saxpy(1.0, x, y)

def test_pybind_saxpy_rejects_noncontiguous_y(mod):
    base = np.zeros(8, dtype=np.float64); y = base[::2]   # strided
    x = np.arange(4, dtype=np.float64)
    with pytest.raises(Exception):
        mod.saxpy(1.0, x, y)

def test_pybind_compile_strategy_bad_json_raises(mod):
    with pytest.raises(Exception):
        mod.compile_strategy("{not json")

def test_pybind_compile_strategy_unsupported_returns_zero(mod):
    import json
    ir = json.dumps({"name":"f","ret":"f64","params":[],
        "body":[{"op":"return","value":{"k":"const"}}]})
    assert mod.compile_strategy(ir) == 0
