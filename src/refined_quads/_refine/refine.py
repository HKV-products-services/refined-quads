"""Refine an existing refined quad mesh with new refinement areas (ADR-0001)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import shapely
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from refined_quads._common.grid import Key, Leaf
from refined_quads._common.recognise import lattice_squares
from refined_quads._generate.balance import balance_violations
from refined_quads._generate.engine import (
    WATER_REMOVAL_RULES,
    Existing,
    RawRepairError,
    refine_cells,
)
from refined_quads._generate.generate import (
    WaterRemovalRule,
    _check_polygonal,
    _in_mesh_crs,
    _ProgressAdapter,
    _refinement_area,
    _refinement_areas,
    _to_mesh,
    _transition_rows,
    _water_areas,
    domain_holes,
    domain_shells,
)
from refined_quads._io.gis import Areas, frame_areas, is_frame
from refined_quads.errors import InputError, MeshRecognitionError, OrthogonalityRepairError
from refined_quads.mesh import Lattice, Mesh, RefineResult, ReplacementMap


def refine(
    mesh: Mesh,
    refinement_areas: Iterable[tuple[BaseGeometry, int]] | Areas,
    *,
    transition_rows: int | Sequence[int] = 1,
    domain: BaseGeometry | Areas | None = None,
    water_areas: Iterable[BaseGeometry] | Areas = (),
    water_removal: WaterRemovalRule = "centre",
    crs: CRS | str | int | None = None,
    progress: Callable[[float, str], object] | None = None,
) -> RefineResult:
    """Refine an existing mesh locally with new refinement areas.

    The lattice and the lattice squares are recognised from the cells of the mesh
    (transition cells as parts of their square), the new areas are applied, the 2:1
    balance is restored where needed and the transitions around the changed squares
    are rebuilt. Squares that do not change keep their cells exactly, nodes included.

    Parameters
    ----------
    mesh : Mesh
        The existing mesh, for example from :func:`read_ugrid`.
    refinement_areas : iterable of (shapely geometry, int), or Areas
        The new refinement areas with their level (1 or higher, relative to the base
        cell of the mesh).
    transition_rows : int or sequence of int
        Transition rows per level step from coarse to fine, as for :func:`generate`:
        one value per step up to the highest level of the result (that of the mesh
        or of the new areas, whichever is higher), or one value for every step; each
        at least 0. Only the steps the new areas need are used.
    domain : shapely geometry or Areas, optional
        New cells outside it are removed; its holes act as water areas. It never adds
        cells: refining stays within the footprint of the mesh.
    water_areas : iterable of shapely geometry, or Areas
        New cells are removed by the water removal rule; existing cells stay.
    water_removal : {"centre", "inside", "touches"}
        Water removal rule.
    crs : pyproj.CRS or str or int, optional
        CRS of input without CRS (shapely geometries) when the mesh has a CRS; input
        with its own CRS is transformed to the CRS of the mesh.
    progress : callable, optional
        As for :func:`generate`.

    Returns
    -------
    RefineResult
        The new mesh (lattice levels raised if the new areas are finer) and the
        replacement map.

    Raises
    ------
    MeshRecognitionError
        If the mesh is not a refined quad mesh: nodes off its lattice, cells that
        fill no lattice square, or a broken 2:1 balance; it carries the locations.
        Duplicate nodes are accepted: cells that do not change keep theirs.
    InputError
        For invalid areas, transition rows, CRS or water removal rule (as
        :func:`generate`).
    OrthogonalityRepairError
        If the orthogonality repair cannot close; it carries the raw mesh.
    Cancelled
        If the progress callback returned ``True``.
    """
    mesh_crs = mesh.crs
    areas = _areas(refinement_areas, crs, mesh_crs)
    new_levels = max((level for _, level in areas), default=0)
    levels = max(mesh.lattice.levels, new_levels)
    rows = _transition_rows(transition_rows, levels)
    domain_geom = None if domain is None else _geometry(domain, crs, mesh_crs, "domain")
    water = list(_water(water_areas, crs, mesh_crs))
    if domain_geom is not None:
        water += domain_holes(domain_geom)
        domain_geom = domain_shells(domain_geom)
    if water_removal not in WATER_REMOVAL_RULES:
        allowed = ", ".join(repr(rule) for rule in WATER_REMOVAL_RULES)
        raise InputError(f"water removal rule must be one of {allowed}, got {water_removal!r}")

    lattice = mesh.lattice
    existing, old_cells = _recognise(mesh, 2 ** (levels - lattice.levels))
    new_lattice = Lattice(lattice.origin, lattice.cell_size, levels)
    try:
        vertices, cells = refine_cells(
            existing,
            areas,
            origin=lattice.origin,
            cell_size=lattice.cell_size,
            levels=levels,
            transition_rows=rows,
            domain=domain_geom,
            water_areas=water,
            water_removal=water_removal,
            progress=None if progress is None else _ProgressAdapter(progress),
        )
    except RawRepairError as exc:
        raw = _to_mesh(exc.vertices, exc.cells, new_lattice, mesh_crs)
        raise OrthogonalityRepairError(str(exc), exc.bad_points, mesh=raw) from None
    new_mesh = _to_mesh(vertices, cells, new_lattice, mesh_crs)
    return RefineResult(new_mesh, _replacement_map(new_mesh, old_cells))


def _recognise(mesh: Mesh, scale: int) -> tuple[Existing, list[list[tuple[int, int]]]]:
    """Recognise the lattice squares of a mesh; indices are multiplied by ``scale``.

    Returns the existing mesh for the engine (nodes compacted to the used ones) and
    every old cell as lattice points, in cell order.
    """
    lattice = mesh.lattice
    faces = mesh.faces
    used = np.unique(faces[faces >= 0])
    renumber = np.full(len(mesh.nodes), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))
    faces = np.where(faces >= 0, renumber[np.where(faces >= 0, faces, 0)], -1)
    nodes = mesh.nodes[used]

    indices, on_lattice = lattice.to_indices(nodes)
    if not on_lattice.all():
        bad = nodes[~on_lattice]
        raise MeshRecognitionError(
            f"{len(bad)} nodes are not on the lattice of the mesh (unit {lattice.unit:g}); "
            f"for example ({bad[0, 0]:.3f}, {bad[0, 1]:.3f})",
            [(float(x), float(y)) for x, y in bad],
        )
    squares = lattice_squares(indices, faces, lattice.levels)
    unrecognised = np.flatnonzero(squares.face_level < 0)
    if unrecognised.size:
        centres = [_centroid(nodes, faces[i]) for i in unrecognised]
        raise MeshRecognitionError(
            f"{unrecognised.size} cells are neither a lattice square nor transition cells "
            f"that fill one; for example at ({centres[0][0]:.3f}, {centres[0][1]:.3f})",
            centres,
        )
    violations = balance_violations(squares, lattice.levels)
    if len(violations):
        points = np.asarray(lattice.origin) + violations * (lattice.unit / 2.0)
        raise MeshRecognitionError(
            f"the mesh breaks the 2:1 balance at {len(points)} sides; for example at "
            f"({points[0, 0]:.3f}, {points[0, 1]:.3f})",
            [(float(x), float(y)) for x, y in points],
        )

    partial_squares = {
        (int(level), int(x), int(y))
        for level, x, y, part in zip(
            squares.level, squares.x, squares.y, squares.partial, strict=True
        )
        if part
    }
    partial_boxes = []
    partial_cells = []
    cells_of: dict[Key, list[list[int]]] = {}
    leaf_level: dict[Key, int] = {}
    old_cells: list[list[tuple[int, int]]] = []
    for i, face in enumerate(faces):
        cell = [int(v) for v in face if v >= 0]
        level = int(squares.face_level[i])
        side = 2 ** (lattice.levels - level)
        x0 = min(int(indices[v, 0]) for v in cell) // side * side
        y0 = min(int(indices[v, 1]) for v in cell) // side * side
        key = (x0 * scale, y0 * scale, (x0 + side) * scale, (y0 + side) * scale)
        if (level, x0 // side, y0 // side) in partial_squares:
            partial_cells.append(shapely.Polygon(nodes[cell]))
            if key not in leaf_level:
                (bx0, by0), (bx1, by1) = lattice.to_coordinates([[x0, y0], [x0 + side, y0 + side]])
                partial_boxes.append(shapely.box(bx0, by0, bx1, by1))
        leaf_level.setdefault(key, level)
        cells_of.setdefault(key, []).append(cell)
        old_cells.append([(int(indices[v, 0]) * scale, int(indices[v, 1]) * scale) for v in cell])
    leaves: list[Leaf] = [(*key, level) for key, level in leaf_level.items()]
    points = [(int(x) * scale, int(y) * scale) for x, y in indices]
    existing = Existing(
        leaves,
        cells_of,
        points,
        nodes.copy(),
        partial=shapely.union_all(partial_boxes) if partial_boxes else None,
        # A tolerance of 1e-6 lattice units for coordinates read from a file.
        footprint=(
            shapely.union_all(partial_cells).buffer(1e-6 * lattice.unit) if partial_cells else None
        ),
    )
    return existing, old_cells


def _replacement_map(mesh: Mesh, old_cells: list[list[tuple[int, int]]]) -> ReplacementMap:
    """Relate every new cell to the old cell it lies in; list the old cells that went."""
    indices, _ = mesh.lattice.to_indices(mesh.nodes)
    old_keys = {_canonical(cell): i for i, cell in enumerate(old_cells)}
    n_new = len(mesh.faces)
    source = np.full(n_new, -1, dtype=np.int64)
    unchanged = np.zeros(n_new, dtype=bool)
    changed: list[int] = []
    centroids: list[tuple[float, float]] = []
    for i, face in enumerate(mesh.faces):
        points = [(int(indices[v, 0]), int(indices[v, 1])) for v in face if v >= 0]
        old = old_keys.get(_canonical(points))
        if old is not None:
            source[i] = old
            unchanged[i] = True
        else:
            changed.append(i)
            centroids.append(tuple(np.mean(points, axis=0)))
    if changed:
        polygons = [shapely.Polygon(c) for c in old_cells]
        tree = shapely.STRtree(polygons)
        q, hit = tree.query(shapely.points(np.asarray(centroids)), predicate="intersects")
        first = np.full(len(changed), -1, dtype=np.int64)
        for qi, oi in sorted(zip(q.tolist(), hit.tolist(), strict=True), reverse=True):
            first[qi] = oi if first[qi] < 0 else min(first[qi], oi)
        source[changed] = first
    kept = set(source[unchanged].tolist())
    removed = np.array([i for i in range(len(old_cells)) if i not in kept], dtype=np.int64)
    return ReplacementMap(source=source, unchanged=unchanged, removed=removed)


def _canonical(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Return a cell as lattice points, counterclockwise, from its smallest point."""
    n = len(points)
    area2 = sum(
        points[k][0] * points[(k + 1) % n][1] - points[(k + 1) % n][0] * points[k][1]
        for k in range(n)
    )
    if area2 < 0:
        points = points[::-1]
    start = min(range(n), key=points.__getitem__)
    return tuple(points[start:] + points[:start])


def _centroid(nodes: NDArray[np.float64], face: NDArray[np.int64]) -> tuple[float, float]:
    xy = nodes[face[face >= 0]].mean(axis=0)
    return float(xy[0]), float(xy[1])


def _areas(value: object, crs: object, mesh_crs: CRS | None) -> list[tuple[BaseGeometry, int]]:
    if mesh_crs is not None:
        return _refinement_areas(value, crs, mesh_crs)
    if isinstance(value, Areas) or is_frame(value):
        raise InputError("the mesh has no CRS, so areas with a CRS cannot be transformed to it")
    if not isinstance(value, Iterable):
        raise InputError(f"refinement areas must be (geometry, level) pairs, got {value!r}")
    return [_refinement_area(item, i) for i, item in enumerate(value)]


def _water(value: object, crs: object, mesh_crs: CRS | None) -> list[BaseGeometry]:
    if mesh_crs is not None:
        return _water_areas(value, crs, mesh_crs)
    if isinstance(value, Areas) or is_frame(value):
        raise InputError("the mesh has no CRS, so water areas with a CRS cannot be transformed")
    if not isinstance(value, Iterable):
        raise InputError(f"water areas must be polygons, got {value!r}")
    geoms = list(value)
    for i, geom in enumerate(geoms):
        _check_polygonal(geom, f"water area {i}")
    return geoms


def _geometry(value: object, crs: object, mesh_crs: CRS | None, what: str) -> BaseGeometry:
    """One (merged) polygon in the CRS of the mesh."""
    if isinstance(value, Areas) or is_frame(value):
        if mesh_crs is None:
            raise InputError(f"the mesh has no CRS, so a {what} with a CRS cannot be transformed")
        areas = value if isinstance(value, Areas) else frame_areas(value, crs, what)
        geom = shapely.union_all(np.asarray(areas.to_crs(mesh_crs).geometries, dtype=object))
    else:
        _check_polygonal(value, what)
        geom = value  # type: ignore[assignment]
        if mesh_crs is not None:
            geom = _in_mesh_crs([geom], crs, mesh_crs, what)[0]
    _check_polygonal(geom, what)
    return geom
