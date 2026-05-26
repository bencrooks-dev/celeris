import pathlib


def test_docs_exist():
    root = pathlib.Path(__file__).resolve().parents[1]
    for f in ["README.md", "CONTRIBUTING.md", "CHANGELOG.md",
              "docs/ARCHITECTURE.md", "docs/ROADMAP.md", "docs/IR_SPEC.md"]:
        p = root / f
        assert p.exists() and p.stat().st_size > 200, f
