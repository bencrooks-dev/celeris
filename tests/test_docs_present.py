import pathlib


def _root():
    return pathlib.Path(__file__).resolve().parents[1]


def test_docs_exist():
    root = _root()
    for f in ["README.md", "CONTRIBUTING.md", "CHANGELOG.md",
              "docs/ARCHITECTURE.md", "docs/ROADMAP.md", "docs/IR_SPEC.md"]:
        p = root / f
        assert p.exists() and p.stat().st_size > 200, f


def test_readme_has_key_sections():
    txt = (_root() / "README.md").read_text()
    low = txt.lower()
    for needle in ["@fast_runtime", "supported subset", "install", "numba",
                   "not a full python compiler", "backend"]:
        assert needle.lower() in low, needle


def test_notice_credits_nlohmann():
    txt = (_root() / "NOTICE").read_text()
    assert "nlohmann" in txt.lower() and "mit" in txt.lower()
    assert "apache" in txt.lower()
