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


def test_changelog_mentions_loop_fusion():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert "fusion" in (root / "CHANGELOG.md").read_text().lower()

def test_version_bumped():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert '0.5.0' in (root / "pyproject.toml").read_text()

def test_changelog_mentions_affine():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert "affine" in (root / "CHANGELOG.md").read_text().lower()

def test_version_is_050():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert '0.5.0' in (root / "pyproject.toml").read_text()

def test_changelog_mentions_prange():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert "prange" in (root / "CHANGELOG.md").read_text().lower()

def test_changelog_mentions_tensor():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    txt = (root / "CHANGELOG.md").read_text().lower()
    assert "2-d" in txt or "tensor" in txt
