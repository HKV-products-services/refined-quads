"""Progress reporting and cancellation of a run."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from refined_quads.errors import Cancelled

_log = logging.getLogger("refined_quads._engine")

#: Engine callback: ``(percent or None, message)``; a truthy result cancels the run.
Callback = Callable[[int | None, str], object]


def fmt_elapsed(seconds: float) -> str:
    """Format a duration as ``"12s"`` or ``"3m 05s"``."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


class Reporter:
    """Report progress to an optional callback, with the elapsed time, and cancel on request.

    Parameters
    ----------
    callback : callable, optional
        Called as ``callback(percent or None, message)``; a truthy result cancels.
    """

    def __init__(self, callback: Callback | None) -> None:
        self.callback = callback
        self.started = time.perf_counter()

    def elapsed(self) -> str:
        """Time since the start, formatted."""
        return fmt_elapsed(time.perf_counter() - self.started)

    def __call__(self, pct: int | None = None, msg: str | None = None) -> None:
        """Report a step; raise :class:`Cancelled` when the callback asks for it."""
        suffix = ""
        if pct is not None or msg is not None:
            suffix = f"  [{self.elapsed()}]"
            _log.debug("[QuadRef] %s%s%s", f"{pct}% " if pct is not None else "", msg or "", suffix)
        if self.callback is not None:
            msg_out = (msg + suffix) if msg else (msg or "")
            if self.callback(pct, msg_out):
                raise Cancelled("mesh generation cancelled by the progress callback")
