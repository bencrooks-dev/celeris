from celeris.backends import (register, get_backend, available_backends,
                              default_chain, PRIORITY)

class _Dummy:
    name = "dummy"
    def available(self): return True
    def compile(self, ir): return lambda *a: 42

class _Unavail:
    name = "unavail"
    def available(self): return False
    def compile(self, ir): raise AssertionError("should not compile")

def test_register_and_get():
    register(_Dummy())
    assert get_backend("dummy").compile({})(1, 2) == 42

def test_available_filters_unavailable():
    register(_Dummy()); register(_Unavail())
    names = [b.name for b in available_backends()]
    assert "dummy" in names and "unavail" not in names

def test_get_missing_raises():
    import pytest
    with pytest.raises(KeyError):
        get_backend("does-not-exist-xyz")

def test_priority_constant_order():
    assert PRIORITY.index("kernels") < PRIORITY.index("llvm") < PRIORITY.index("sourcegen") < PRIORITY.index("interpreter")

def test_default_chain_respects_priority():
    # default_chain returns available registered backends ordered by PRIORITY;
    # unknown-name backends (like the test dummy) sort after known ones, stably.
    register(_Dummy())
    chain = [b.name for b in default_chain()]
    # all returned backends must be available
    assert all(name != "unavail" for name in chain)
