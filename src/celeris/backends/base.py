"""Backend Protocol: the contract every celeris backend implements."""
from __future__ import annotations
from typing import Any, Callable, Protocol, runtime_checkable

@runtime_checkable
class Backend(Protocol):
    name: str
    def available(self) -> bool:
        """True if this backend can run in the current environment."""
        ...
    def compile(self, ir: dict) -> Callable[..., Any]:
        """Compile typed IR into a callable kernel. Raise on failure."""
        ...
