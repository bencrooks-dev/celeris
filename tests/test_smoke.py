def test_import_and_version():
    import celeris
    assert isinstance(celeris.__version__, str)
    assert celeris.__version__.count(".") >= 2
    assert callable(celeris.fast_runtime)
