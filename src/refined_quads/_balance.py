"""2:1 balance of lattice squares."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from refined_quads._recognise import LatticeSquares
from refined_quads._topology import PointLookup


def balance_violations(squares: LatticeSquares, levels: int) -> NDArray[np.int64]:
    """Find sides where a lattice square meets a square more than one level coarser.

    Parameters
    ----------
    squares : LatticeSquares
        The lattice squares of a mesh.
    levels : int
        Highest refinement level of the lattice.

    Returns
    -------
    numpy.ndarray of int, shape (v, 2)
        Midpoint of each violating side of the finer square, in units of half the
        finest level (so that midpoints of the finest squares are integers).

    Notes
    -----
    Every side of every square is probed just outside its midpoint. A coarser
    neighbour along that side always covers the midpoint, because lattice squares
    are aligned to their level; a finer neighbour is found from its own side.
    """
    side = 2 ** (levels - squares.level)
    # Work in half units so that a probe half a unit outside a side is an integer point.
    x0 = 2 * squares.x * side
    y0 = 2 * squares.y * side
    s2 = 2 * side
    mid_x, mid_y = x0 + side, y0 + side
    probes = np.concatenate(
        [
            np.stack([x0 + s2 + 1, mid_y], axis=1),
            np.stack([x0 - 1, mid_y], axis=1),
            np.stack([mid_x, y0 + s2 + 1], axis=1),
            np.stack([mid_x, y0 - 1], axis=1),
        ]
    )
    sides = np.concatenate(
        [
            np.stack([x0 + s2, mid_y], axis=1),
            np.stack([x0, mid_y], axis=1),
            np.stack([mid_x, y0 + s2], axis=1),
            np.stack([mid_x, y0], axis=1),
        ]
    )
    probe_level = np.tile(squares.level, 4)
    violating = np.zeros(len(probes), dtype=bool)
    for level in range(levels - 1):
        here = squares.level == level
        if not here.any():
            continue
        lookup = PointLookup(np.stack([squares.x[here], squares.y[here]], axis=1))
        candidates = probe_level >= level + 2
        cell = np.floor_divide(probes[candidates], 2 * 2 ** (levels - level))
        violating[np.flatnonzero(candidates)[lookup.find(cell) >= 0]] = True
    return np.asarray(sides[violating], dtype=np.int64)
