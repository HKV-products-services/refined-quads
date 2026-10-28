"""The base grid of a run: integer lattice indices and their coordinates.

Cells are keys ``(ix0, iy0, ix1, iy1)`` in integer units of the finest cell
(``cell_size / 2**levels``) from the start of the base grid; coordinates are computed
as ``origin + index * unit``, only where geometry is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import shapely
from shapely.geometry.base import BaseGeometry

Key = tuple[int, int, int, int]
Leaf = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class Grid:
    """The base grid over a domain.

    Attributes
    ----------
    ox, oy : float
        Start of the base grid, a lattice point at or below the lower-left corner
        of the domain.
    unit : float
        Side of the finest cell, ``cell_size / 2**levels``.
    cell_size : float
        Base cell size.
    levels : int
        Highest refinement level.
    nx, ny : int
        Number of base cells in x and y.
    """

    ox: float
    oy: float
    unit: float
    cell_size: float
    levels: int
    nx: int
    ny: int

    @property
    def base(self) -> int:
        """Side of a base cell in units."""
        return int(2**self.levels)

    def box(self, ix0: int, iy0: int, ix1: int, iy1: int) -> BaseGeometry:
        """Return a cell as a shapely box in coordinates."""
        return shapely.box(
            self.ox + ix0 * self.unit,
            self.oy + iy0 * self.unit,
            self.ox + ix1 * self.unit,
            self.oy + iy1 * self.unit,
        )


def make_grid(
    bounds: tuple[float, float, float, float],
    cell_size: float,
    levels: int,
    origin: tuple[float, float] | None,
) -> Grid:
    """Lay the base grid over the bounds of the domain.

    Parameters
    ----------
    bounds : tuple of float
        ``(minx, miny, maxx, maxy)`` of the domain.
    cell_size : float
        Base cell size.
    levels : int
        Highest refinement level.
    origin : tuple of float or None
        A lattice point; by default the lattice runs through multiples of the cell
        size.

    Returns
    -------
    Grid
        The grid, starting at the largest lattice point at or below the lower-left
        corner of the bounds.
    """
    minx, miny, maxx, maxy = bounds
    if origin is None:
        ox = math.floor(minx / cell_size) * cell_size
        oy = math.floor(miny / cell_size) * cell_size
    else:
        ox = origin[0] + math.floor((minx - origin[0]) / cell_size) * cell_size
        oy = origin[1] + math.floor((miny - origin[1]) / cell_size) * cell_size
    return Grid(
        ox=ox,
        oy=oy,
        unit=cell_size / (2**levels),
        cell_size=cell_size,
        levels=levels,
        nx=math.ceil((maxx - ox) / cell_size),
        ny=math.ceil((maxy - oy) / cell_size),
    )


def neighbours(key: Key) -> list[Key]:
    """Return the same-size neighbour keys of a cell in the order N, S, E, W."""
    ix0, iy0, ix1, iy1 = key
    iw = ix1 - ix0
    ih = iy1 - iy0
    return [
        (ix0, iy1, ix1, iy1 + ih),
        (ix0, iy0 - ih, ix1, iy0),
        (ix1, iy0, ix1 + iw, iy1),
        (ix0 - iw, iy0, ix0, iy1),
    ]


def split(cells: dict[Key, int], key: Key, new_lvl: int) -> list[Key]:
    """Add the four children of ``key`` to ``cells`` at ``new_lvl``; return their keys.

    The caller removes ``key`` itself.
    """
    sx0, sy0, sx1, sy1 = key
    sxm = (sx0 + sx1) >> 1
    sym = (sy0 + sy1) >> 1
    subs = [(sx0, sy0, sxm, sym), (sxm, sy0, sx1, sym), (sx0, sym, sxm, sy1), (sxm, sym, sx1, sy1)]
    for sub in subs:
        cells[sub] = new_lvl
    return subs
