"""celeris error hierarchy.

All exceptions raised within celeris derive from :class:`CelerisError`, so
callers can catch the entire family with a single ``except CelerisError``.
The leaf classes mark distinct stages of the compile pipeline (source
parsing, IR type reconciliation, IR verification, and backend lowering),
which lets callers distinguish a user-facing limitation from an internal
invariant violation.
"""


class CelerisError(Exception):
    """Base class for every error raised by celeris."""


class UnsupportedFeature(CelerisError):
    """Raised when source uses syntax/features outside the supported numeric subset."""


class TypeErrorIR(CelerisError):
    """Raised when types fail to reconcile while building or checking IR."""


class VerifyError(CelerisError):
    """Raised when the independent IR verifier rejected an IR structure/type invariant."""


class CompileError(CelerisError):
    """Raised when a backend failed to lower/compile/load a kernel."""
