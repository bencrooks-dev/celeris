"""Runtime entry point.

Wave-1 placeholder: a no-op passthrough decorator so that ``import celeris``
works end-to-end. The real ``@fast_runtime`` (parse -> verify -> optimize ->
dispatch -> fallback) lands in Wave 7 and replaces this module.
"""


def fast_runtime(fn=None, **kw):
    def deco(f):
        return f

    return deco(fn) if callable(fn) else deco
