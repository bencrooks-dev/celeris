"""celeris runtime: the @fast_runtime decorator.

Captures a function's source at first call, lowers it to typed IR, verifies and
optimizes the IR, then compiles it via the best available backend (golden
kernels -> llvmlite -> source-gen -> interpreter). Compiled kernels are cached
by signature. Any failure -- an unsupported feature, or no backend able to
compile -- falls back transparently to running the original Python function.
"""
from __future__ import annotations

import functools
import os

from . import ir as _ir
from .backends import default_chain, get_backend
from .errors import CelerisError
from .parser import parse_function
from .passes import optimize
from .verifier import verify_ir

_DEBUG_ENV = os.environ.get("CELERIS_DEBUG", "") not in ("", "0", "false", "False")


def _sig_key(fn, args):
    return (fn.__module__, fn.__qualname__, tuple(type(a).__name__ for a in args))


def fast_runtime(_fn=None, *, backend=None, debug=None):
    """JIT-compile a numeric-subset function at first call; fall back to plain
    Python on any failure. Usable as ``@fast_runtime`` or
    ``@fast_runtime(backend=..., debug=...)``."""
    dbg = _DEBUG_ENV if debug is None else debug

    def decorate(fn):
        cache = {}
        ir_holder = {}

        def _build_ir():
            if "ir" not in ir_holder:
                k = parse_function(fn)
                verify_ir(k)
                ir_holder["ir"] = optimize(k)
            return ir_holder["ir"]

        def _select_and_compile(k):
            if backend is not None:
                be = get_backend(backend)
                return be.name, be.compile(k)
            last = None
            for be in default_chain():
                try:
                    return be.name, be.compile(k)
                except Exception as exc:  # noqa: BLE001 - try next backend
                    last = exc
            raise CelerisError(
                f"no backend could compile '{fn.__qualname__}': {last}")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if kwargs:
                return fn(*args, **kwargs)  # the subset has no kwargs
            key = _sig_key(fn, args)
            compiled = cache.get(key)
            if compiled is None:
                try:
                    k = _build_ir()
                    if dbg:
                        print(f"[celeris] IR for {fn.__qualname__}:")
                        print(_ir.dumps(k))
                    name, compiled = _select_and_compile(k)
                    if dbg:
                        print(f"[celeris] compiled {fn.__qualname__} "
                              f"via {name} backend")
                except Exception as exc:  # noqa: BLE001 - any failure -> fallback
                    if dbg:
                        print(f"[celeris] '{fn.__qualname__}' not compiled "
                              f"({exc}); falling back to Python")
                    compiled = fn
                cache[key] = compiled
            return compiled(*args)

        wrapper.__celeris_cache__ = cache
        wrapper.__celeris_ir__ = _build_ir
        wrapper.__celeris_wrapped__ = fn
        return wrapper

    return decorate(_fn) if callable(_fn) else decorate
