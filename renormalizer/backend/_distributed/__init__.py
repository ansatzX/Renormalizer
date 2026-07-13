"""Private distributed backend implementation components."""

from renormalizer.backend._distributed.terminal import (
    _FatalTransition,
    _LeaseCloseTransition,
    _ResourceAdmission,
    _TerminalLifecycleGate,
    _TerminalPhase,
)


__all__ = [
    "_FatalTransition",
    "_LeaseCloseTransition",
    "_ResourceAdmission",
    "_TerminalLifecycleGate",
    "_TerminalPhase",
]
