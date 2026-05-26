import celeris.errors as e


def test_hierarchy():
    assert issubclass(e.UnsupportedFeature, e.CelerisError)
    assert issubclass(e.TypeErrorIR, e.CelerisError)
    assert issubclass(e.VerifyError, e.CelerisError)
    assert issubclass(e.CompileError, e.CelerisError)
