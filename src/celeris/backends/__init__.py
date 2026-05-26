"""Backend registry and dispatch ordering for celeris.

Concrete backends (``interpreter``, ``sourcegen``, ``kernels``, ``llvm``) live
in submodules of this package and self-register on import via :func:`register`.
They are built in later waves; missing or broken optional backends are tolerated
so the registry always imports cleanly.

Dispatch order is governed by the static :data:`PRIORITY` constant: faster
backends sort first. :func:`default_chain` filters the registry to the
currently-available backends and orders them by this priority.
"""
from __future__ import annotations

from .base import Backend

# Static dispatch priority — fastest backends first. ``default_chain`` orders
# available backends by their index here; unknown-name backends sort last.
PRIORITY = ["kernels", "llvm", "sourcegen", "interpreter"]

_REGISTRY: dict[str, Backend] = {}

__all__ = [
    "Backend",
    "PRIORITY",
    "register",
    "get_backend",
    "available_backends",
    "default_chain",
]


def register(b: Backend) -> None:
    """Register a backend instance under its ``name``."""
    _REGISTRY[b.name] = b


def get_backend(name: str) -> Backend:
    """Return the registered backend named ``name`` (raises ``KeyError``)."""
    return _REGISTRY[name]


def available_backends() -> list[Backend]:
    """Return registered backends whose ``available()`` is True.

    Any backend whose ``available()`` raises is treated as unavailable and
    silently skipped.
    """
    out: list[Backend] = []
    for b in _REGISTRY.values():
        try:
            ok = b.available()
        except Exception:
            ok = False
        if ok:
            out.append(b)
    return out


def default_chain() -> list[Backend]:
    """Return available backends ordered by :data:`PRIORITY`.

    Backends whose name is not in :data:`PRIORITY` sort after all known ones,
    tie-broken by name for a stable order.
    """
    def _key(b: Backend):
        name = b.name
        return (PRIORITY.index(name) if name in PRIORITY else len(PRIORITY), name)

    return sorted(available_backends(), key=_key)


# Attempt to import the optional concrete backend modules so they self-register
# if present. Guarded so a missing or broken backend cannot break the registry.
for _mod in ("interpreter", "sourcegen", "kernels", "llvm"):
    try:
        __import__(f"{__name__}.{_mod}")
    except Exception:
        pass  # backend unavailable in this environment / not built yet
