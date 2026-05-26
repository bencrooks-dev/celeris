import importlib.util
import pathlib


def test_benchmark_runs():
    p = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("celeris_benchmark", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.run(n=512, reps=2)   # tiny: assert it runs end-to-end without error
