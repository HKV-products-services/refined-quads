"""Generate a refined quad mesh."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from refined_quads._engine import generate_cells
from refined_quads.errors import InputError
from refined_quads.mesh import Lattice, Mesh

_POLYGONAL = ("Polygon", "MultiPolygon")


def generate(
    domain: BaseGeometry,
    *,
    cell_size: float,
    refinement_areas: Iterable[tuple[BaseGeometry, int]] = (),
    transition_rows: int | Sequence[int] = 1,
) -> Mesh:
    """Generate a refined quad mesh over a domain.

    Parameters
    ----------
    domain : shapely.Polygon or shapely.MultiPolygon
        The area the mesh covers; cells are clipped to it.
    cell_size : float
        Side length of the base cell; any positive length.
    refinement_areas : iterable of (shapely geometry, int)
        Polygons paired with the refinement level their cells must reach (1 or
        higher); level ``L`` gives cells of ``cell_size / 2**L``. The highest level
        among the areas is the highest level of the mesh.
    transition_rows : int or sequence of int
        Rows of finer cells kept around a finer area before the mesh coarsens, per
        step from coarse to fine: the first value is for the step from level 0 to 1.
        One value per level step, or a single value for every step; each at least 1.

    Returns
    -------
    Mesh
        The mesh, its faces counterclockwise, on a lattice whose origin is the
        lower-left corner of the domain rounded down to a multiple of ``cell_size``.

    Raises
    ------
    InputError
        If the domain is empty or not polygonal, the cell size is not a positive
        length, a refinement area is empty or not polygonal, a level is not an
        integer of at least 1, or the transition rows are not one integer of at least
        1 per level step.
    OrthogonalityRepairError
        If the orthogonality repair cannot make every internal edge FM-orthogonal.

    Examples
    --------
    >>> from shapely import Point, box
    >>> mesh = generate(
    ...     box(0, 0, 400, 300),
    ...     cell_size=25.0,
    ...     refinement_areas=[(Point(200, 150).buffer(40), 2)],
    ...     transition_rows=[1, 1],
    ... )
    >>> mesh.lattice.unit
    6.25
    """
    _check_polygonal(domain, "domain")
    size = _positive_length(cell_size)
    areas = [_refinement_area(item, i) for i, item in enumerate(refinement_areas)]
    levels = max((level for _, level in areas), default=0)
    rows = _transition_rows(transition_rows, levels)

    vertices, cells, origin = generate_cells(
        domain, areas, cell_size=size, levels=levels, transition_rows=rows
    )
    return _to_mesh(vertices, cells, Lattice(origin, size, levels))


def _to_mesh(vertices: NDArray[np.float64], cells: list[list[int]], lattice: Lattice) -> Mesh:
    """Build a mesh of the used nodes, every face counterclockwise."""
    sizes = np.fromiter((len(cell) for cell in cells), dtype=np.int64, count=len(cells))
    flat = np.fromiter((v for cell in cells for v in cell), dtype=np.int64, count=int(sizes.sum()))
    used, flat = np.unique(flat, return_inverse=True)
    nodes = vertices[used]
    faces = np.full((len(cells), int(sizes.max())), -1, dtype=np.int64)
    rows = np.repeat(np.arange(len(cells)), sizes)
    cols = np.arange(len(flat)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    faces[rows, cols] = flat.reshape(-1)
    for n in np.unique(sizes):
        group = np.flatnonzero(sizes == n)
        xy = nodes[faces[group, :n]]
        area2 = (xy[..., 0] * np.roll(xy[..., 1], -1, axis=1)).sum(axis=1) - (
            np.roll(xy[..., 0], -1, axis=1) * xy[..., 1]
        ).sum(axis=1)
        clockwise = group[area2 < 0]
        faces[clockwise, :n] = faces[clockwise, n - 1 :: -1]
    return Mesh(nodes, faces, lattice)


def _check_polygonal(geom: object, what: str) -> None:
    if not isinstance(geom, BaseGeometry) or geom.geom_type not in _POLYGONAL:
        kind = geom.geom_type if isinstance(geom, BaseGeometry) else type(geom).__name__
        raise InputError(f"{what} must be a Polygon or MultiPolygon, got {kind}")
    if geom.is_empty:
        raise InputError(f"{what} is empty")


def _positive_length(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InputError(f"cell size must be a number, got {value!r}")
    size = float(value)
    if not (math.isfinite(size) and size > 0):
        raise InputError(f"cell size must be a positive length, got {value!r}")
    return size


def _integer(value: object) -> int | None:
    """Return ``value`` as int if it is an integral number (not a bool), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    return None


def _refinement_area(item: object, index: int) -> tuple[BaseGeometry, int]:
    if not isinstance(item, tuple | list) or len(item) != 2:
        raise InputError(f"refinement area {index} must be a (geometry, level) pair, got {item!r}")
    geom, raw_level = item
    _check_polygonal(geom, f"refinement area {index}")
    level = _integer(raw_level)
    if level is None or level < 1:
        raise InputError(
            f"refinement area {index}: the level must be an integer of at least 1, "
            f"got {raw_level!r}"
        )
    return geom, level


def _transition_rows(value: int | Sequence[int], levels: int) -> list[int]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        n = _integer(value)
        if n is None or n < 1:
            raise InputError(f"transition rows must be integers of at least 1, got {value!r}")
        return [n] * levels
    rows = list(value)
    if len(rows) != levels:
        raise InputError(
            f"transition rows need one value per level step from coarse to fine; the "
            f"refinement areas reach level {levels}, so {levels} values, got {len(rows)}"
        )
    result = []
    for step, raw in enumerate(rows):
        n = _integer(raw)
        if n is None or n < 1:
            raise InputError(
                f"transition rows must be integers of at least 1; got {raw!r} for the step "
                f"from level {step} to {step + 1}"
            )
        result.append(n)
    return result
