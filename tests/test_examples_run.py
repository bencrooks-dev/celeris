import importlib.util
import pathlib
import pytest

EXAMPLES = ["saxpy", "reduction", "agent_loop"]


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs(name, capsys):
    p = pathlib.Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.main()
    out = capsys.readouterr().out
    assert "OK" in out
