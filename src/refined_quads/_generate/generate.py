"""Generate a refined quad mesh."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

import numpy as np
import shapely
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from refined_quads._generate.engine import WATER_REMOVAL_RULES, RawRepairError, generate_cells
from refined_quads._io.gis import Areas, frame_areas, is_frame
from refined_quads.errors import InputError, OrthogonalityRepairError
from refined_quads.mesh import Lattice, Mesh, projected_crs

_POLYGONAL = ("Polygon", "MultiPolygon")

WaterRemovalRule = Literal["centre", "inside", "touches"]


def generate(
    domain: BaseGeometry | Areas | Any,  # noqa: ANN401 (a GeoDataFrame without geopandas)
    *,
    cell_size: float,
    refinement_areas: Iterable[tuple[BaseGeometry, int]] | Areas = (),
    transition_rows: int | Sequence[int] = 1,
    water_areas: Iterable[BaseGeometry] | Areas | Any = (),  # noqa: ANN401
    water_removal: WaterRemovalRule = "centre",
    crs: CRS | str | int | None = None,
    origin: tuple[float, float] | None = None,
    progress: Callable[[float, str], object] | None = None,
) -> Mesh:
    """Generate a refined quad mesh over a domain.

    Parameters
    ----------
    domain : shapely.Polygon, shapely.MultiPolygon, Areas or GeoDataFrame
        The area the mesh covers; its CRS is the CRS of the mesh and must be
        projected. A lattice square is kept whole when it overlaps the domain, so
        cells along the outer edge may reach slightly outside it. Holes in the domain
        are clipped like water areas, cell by cell, with the water removal rule.
    cell_size : float
        Side length of the base cell; any positive length.
    refinement_areas : iterable of (shapely geometry, int), or Areas
        Polygons (pairs, or :func:`read_refinement_areas`) paired with the
        refinement level their cells must reach (1 or higher); level ``L`` gives
        cells of ``cell_size / 2**L``. The highest level among the areas is the
        highest level of the mesh.
    transition_rows : int or sequence of int
        Rows of finer cells kept around a finer area before the mesh coarsens, per
        step from coarse to fine: the first value is for the step from level 0 to 1.
        One value per level step, or a single value for every step; each at least 1.
    water_areas : iterable of shapely geometry, Areas or GeoDataFrame
        Polygons where no cells are kept, because the 1D network carries the flow
        there. Giving water areas applies them.
    water_removal : {"centre", "inside", "touches"}
        Water removal rule, applied cell by cell to the finished mesh, as clipping
        with MeshKernel does: a cell is removed when its circumcentre (triangle) or
        centroid lies in water (``"centre"``, default, MeshKernel's
        ``FACES_WITH_INCLUDED_CIRCUMCENTERS``), only when it lies completely in water
        (``"inside"``), or as soon as it touches water (``"touches"``). The remaining
        small cells follow the water's edge.
    crs : pyproj.CRS or str or int, optional
        CRS of input that carries none (shapely geometries). Input without CRS
        raises an error unless it is given; input with its own CRS is transformed to
        the CRS of the domain.
    origin : tuple of float, optional
        A point of the lattice, so that the mesh lines up with an existing mesh or
        raster. By default the lower-left corner of the domain rounded down to a
        multiple of ``cell_size``.
    progress : callable, optional
        Called as ``progress(fraction, message)`` while the mesh is generated, with
        a fraction that rises from 0 to 1 and an English message. Returning ``True``
        cancels the run.

    Returns
    -------
    Mesh
        The mesh in the CRS of the domain, its faces counterclockwise, on the
        lattice through ``origin``.

    Raises
    ------
    InputError
        If input has no CRS and ``crs`` is not given, the CRS of the domain is
        geographic, the domain is empty or not polygonal, the cell size is not a positive
        length, a refinement area is empty or not polygonal, a level is not an
        integer of at least 1, the transition rows are not one integer of at least
        1 per level step, a water area is empty or not polygonal, the water removal
        rule is unknown, the origin is not two finite numbers, or no cells are left
        inside the domain.
    OrthogonalityRepairError
        If the orthogonality repair cannot make every internal edge FM-orthogonal;
        it carries the raw mesh (``mesh``) and the problem locations
        (``bad_points``).
    Cancelled
        If the progress callback returned ``True``.

    Examples
    --------
    >>> from shapely import Point, box
    >>> mesh = generate(
    ...     box(0, 0, 400, 300),
    ...     cell_size=25.0,
    ...     refinement_areas=[(Point(200, 150).buffer(40), 2)],
    ...     transition_rows=[1, 1],
    ...     crs="EPSG:28992",
    ... )
    >>> mesh.lattice.unit
    6.25
    """
    domain_geom, mesh_crs = _domain(domain, crs)
    size = _positive_length(cell_size)
    areas = _refinement_areas(refinement_areas, crs, mesh_crs)
    levels = max((level for _, level in areas), default=0)
    rows = _transition_rows(transition_rows, levels)
    water = _water_areas(water_areas, crs, mesh_crs) + domain_holes(domain_geom)
    domain_geom = domain_shells(domain_geom)
    if water_removal not in WATER_REMOVAL_RULES:
        allowed = ", ".join(repr(rule) for rule in WATER_REMOVAL_RULES)
        raise InputError(f"water removal rule must be one of {allowed}, got {water_removal!r}")

    lattice_origin = None if origin is None else _origin(origin)

    try:
        vertices, cells, grid_start = generate_cells(
            domain_geom,
            areas,
            cell_size=size,
            levels=levels,
            transition_rows=rows,
            water_areas=water,
            water_removal=water_removal,
            origin=lattice_origin,
            progress=None if progress is None else _ProgressAdapter(progress),
        )
    except RawRepairError as exc:
        raw_lattice = Lattice(lattice_origin or exc.grid_start, size, levels)
        raw = _to_mesh(exc.vertices, exc.cells, raw_lattice, mesh_crs)
        raise OrthogonalityRepairError(str(exc), exc.bad_points, mesh=raw) from None
    lattice = Lattice(lattice_origin or grid_start, size, levels)
    return _to_mesh(vertices, cells, lattice, mesh_crs)


class _ProgressAdapter:
    """Turn the engine's percentages (or None) into a rising fraction and a message."""

    def __init__(self, callback: Callable[[float, str], object]) -> None:
        self._callback = callback
        self._fraction = 0.0
        self._message = "Starting"

    def __call__(self, percent: int | None, message: str) -> bool:
        if percent is not None:
            self._fraction = max(self._fraction, min(percent, 100) / 100.0)
        if message:
            self._message = message
        return self._callback(self._fraction, self._message) is True


def _origin(value: object) -> tuple[float, float]:
    parts = list(value) if isinstance(value, Iterable) and not isinstance(value, str) else []
    if len(parts) != 2 or not all(
        isinstance(v, numbers.Real) and not isinstance(v, bool) for v in parts
    ):
        raise InputError(f"origin must be two numbers (x, y), got {value!r}")
    point = (float(parts[0]), float(parts[1]))
    if not all(math.isfinite(v) for v in point):
        raise InputError(f"origin must be finite, got {value!r}")
    return point


def domain_shells(domain: BaseGeometry) -> BaseGeometry:
    """Return the domain with its holes filled (the holes are clipped like water)."""
    parts = shapely.get_parts(domain)
    if not any(len(part.interiors) for part in parts):
        return domain
    return shapely.union_all(shapely.polygons(shapely.get_exterior_ring(parts)))


def domain_holes(domain: BaseGeometry) -> list[BaseGeometry]:
    """Return the holes of a domain as polygons, without the domain parts inside them.

    Parameters
    ----------
    domain : shapely.Polygon or shapely.MultiPolygon
        The domain.

    Returns
    -------
    list of shapely geometry
        One polygon per hole; an island of the domain inside a hole is cut out of it.
    """
    parts = shapely.get_parts(domain)
    if not any(len(part.interiors) for part in parts):
        return []
    shells = shapely.union_all(shapely.polygons(shapely.get_exterior_ring(parts)))
    holes = shapely.difference(shells, domain)
    return [hole for hole in shapely.get_parts(holes) if hole.area > 0]


def _domain(value: object, crs: object) -> tuple[BaseGeometry, CRS]:
    """Return the domain as one geometry and the CRS of the mesh."""
    if isinstance(value, Areas) or is_frame(value):
        areas = value if isinstance(value, Areas) else frame_areas(value, crs, "domain")
        geoms = np.asarray(areas.geometries, dtype=object)
        geom = geoms[0] if len(geoms) == 1 else shapely.union_all(geoms)
        own_crs: object = areas.crs
    else:
        geom = value
        own_crs = crs
    _check_polygonal(geom, "domain")
    if own_crs is None:
        raise InputError("the domain has no CRS; pass crs= to say which CRS it is in")
    return geom, projected_crs(own_crs)


def _refinement_areas(value: object, crs: object, mesh_crs: CRS) -> list[tuple[BaseGeometry, int]]:
    if is_frame(value):
        raise InputError(
            "refinement areas from a GeoDataFrame need their levels: use "
            "read_refinement_areas(frame, level=...)"
        )
    if isinstance(value, Areas):
        if value.levels is None:
            raise InputError(
                "refinement areas need a level per polygon: use "
                "read_refinement_areas(..., level=...)"
            )
        pairs = list(zip(value.to_crs(mesh_crs).geometries, value.levels, strict=True))
        return [_refinement_area(item, i) for i, item in enumerate(pairs)]
    if not isinstance(value, Iterable):
        raise InputError(f"refinement areas must be (geometry, level) pairs, got {value!r}")
    pairs = [_refinement_area(item, i) for i, item in enumerate(value)]
    if not pairs:
        return []
    geoms, levels = zip(*pairs, strict=True)
    return list(zip(_in_mesh_crs(geoms, crs, mesh_crs, "refinement areas"), levels, strict=True))


def _water_areas(value: object, crs: object, mesh_crs: CRS) -> list[BaseGeometry]:
    if isinstance(value, Areas) or is_frame(value):
        areas = value if isinstance(value, Areas) else frame_areas(value, crs, "water areas")
        geoms = list(areas.to_crs(mesh_crs).geometries)
    elif isinstance(value, Iterable):
        geoms = list(value)
        for i, geom in enumerate(geoms):
            _check_polygonal(geom, f"water area {i}")
        return _in_mesh_crs(geoms, crs, mesh_crs, "water areas") if geoms else []
    else:
        raise InputError(f"water areas must be polygons, got {value!r}")
    for i, geom in enumerate(geoms):
        _check_polygonal(geom, f"water area {i}")
    return geoms


def _in_mesh_crs(
    geoms: Sequence[BaseGeometry], crs: object, mesh_crs: CRS, what: str
) -> list[BaseGeometry]:
    """Geometries without own CRS, given in ``crs``, transformed to the mesh CRS."""
    if crs is None:
        raise InputError(f"the {what} have no CRS; pass crs= to say which CRS they are in")
    return list(Areas(tuple(geoms), CRS.from_user_input(crs)).to_crs(mesh_crs).geometries)


def _to_mesh(
    vertices: NDArray[np.float64], cells: list[list[int]], lattice: Lattice, crs: CRS | None
) -> Mesh:
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
    return Mesh(nodes, faces, lattice, crs=crs)


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
