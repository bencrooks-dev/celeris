import pathlib

import yaml   # pyyaml is installed in the dev venv


def _load():
    p = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    return yaml.safe_load(p.read_text())


def test_jobs_present():
    cfg = _load()
    jobs = cfg["jobs"]
    assert {"core", "llvm", "lint"} <= set(jobs)


def test_core_matrix_covers_versions_and_os():
    cfg = _load()
    mat = cfg["jobs"]["core"]["strategy"]["matrix"]
    pys = [str(v) for v in mat["python-version"]]
    assert {"3.10", "3.11", "3.12"} <= set(pys)
    assert any("ubuntu" in str(o) for o in mat["os"]) and any("macos" in str(o) for o in mat["os"])


def test_llvm_job_installs_llvmlite():
    cfg = _load()
    steps = cfg["jobs"]["llvm"]["steps"]
    blob = " ".join(str(s.get("run", "")) for s in steps)
    assert "llvm" in blob.lower()  # installs the [llvm] extra / llvmlite
    assert "needs_llvmlite" in blob or "test_differential" in blob


def test_core_excludes_optional_markers():
    cfg = _load()
    steps = cfg["jobs"]["core"]["steps"]
    blob = " ".join(str(s.get("run", "")) for s in steps)
    assert "needs_llvmlite" in blob  # excluded via -m "not needs_llvmlite ..."
