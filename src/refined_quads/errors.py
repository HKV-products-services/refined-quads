"""Exceptions raised by refined-quads."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from refined_quads.mesh import Mesh


class InputError(ValueError):
    """Invalid input: the package raises this instead of silently adjusting the input."""


class Cancelled(Exception):
    """The run was cancelled through the progress callback."""


class OrthogonalityRepairError(RuntimeError):
    """The orthogonality repair could not make every internal edge FM-orthogonal.

    Parameters
    ----------
    message : str
        Description of the failure.
    bad_points : list of (float, float, float or None)
        Midpoint ``(x, y)`` of every remaining invalid internal edge and its FM
        orthogonality (``None`` for a degenerate edge).
    mesh : Mesh, optional
        The raw mesh as it was before the repair, to inspect or repair by hand.

    Attributes
    ----------
    bad_points : list of (float, float, float or None)
        As above.
    mesh : Mesh or None
        As above.
    """

    def __init__(
        self,
        message: str,
        bad_points: list[tuple[float, float, float | None]] | None = None,
        mesh: Mesh | None = None,
    ) -> None:
        super().__init__(message)
        self.bad_points = list(bad_points or [])
        self.mesh = mesh
