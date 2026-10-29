"""The engine: run the steps that turn validated input into nodes and cells.

The engine is a port of the generator of qgis_dhydamo ``9a324bc``, one module per
step: areas (transition zones), quadtree, balance (2:1 and T-junctions), select
(domain and water), faces (transition patterns), conformity (head-to-toe and hanging
nodes) and orthogonality (FM measure and repair). The changes against the original
are deliberate and listed in the changelog:

- transition rows arrive per step from coarse to fine and are validated by the caller
  (the original took them from fine to coarse and counted 0 as 1);
- the highest refinement level follows from the refinement areas, without a minimum
  cell size (the original capped the levels so that the smallest cell was >= 1 m);
- no code for shapely < 2 and no explicit refine zones without transition rows;
- every step up to the final coordinates works in integer lattice indices, including
  the centre nodes the orthogonality repair adds (the original added those as float
  midpoints);
- errors are raised instead of being logged and skipped;
- messages are in English.

Cells are integers in units of the smallest cell (``cell_size / 2**levels``); integer
keys make lookups exact and fast, and coordinates are computed once, at the end, as
``origin + index * unit``. The order of all steps and of node creation follows the
original: the head-to-toe step depends on the node numbering.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from refined_quads._areas import Candidates, make_valid, prepare_refinement_areas
from refined_quads._balance import balance_2_to_1, eliminate_t_junctions
from refined_quads._clip import clip_cells
from refined_quads._conformity import count_hanging_nodes, head_to_toe, resolve_hanging_nodes
from refined_quads._faces import Nodes, emit_cells, split_sides
from refined_quads._grid import Grid, Key, Leaf, make_grid
from refined_quads._orthogonality import repair_orthogonality
from refined_quads._progress import Callback as Progress
from refined_quads._progress import Reporter
from refined_quads._quadtree import Queued, base_cells, build_quadtree, existing_cells
from refined_quads._select import WaterAreas, prepare_water, select_leaves
from refined_quads.errors import InputError, OrthogonalityRepairError

_log = logging.getLogger(__name__)

WATER_REMOVAL_RULES = ("centre", "inside", "touches")


class RawRepairError(OrthogonalityRepairError):
    """Repair failure with the raw nodes and cells, for the caller to build the raw mesh."""

    def __init__(
        self,
        message: str,
        bad_points: list[tuple[float, float, float | None]],
        vertices: NDArray[np.float64],
        cells: list[list[int]],
        grid_start: tuple[float, float],
    ) -> None:
        super().__init__(message, bad_points)
        self.vertices = vertices
        self.cells = cells
        self.grid_start = grid_start


def generate_cells(
    domain: BaseGeometry,
    refinement_areas: Sequence[tuple[BaseGeometry, int]],
    *,
    cell_size: float,
    levels: int,
    transition_rows: Sequence[int],
    water_areas: Sequence[BaseGeometry] = (),
    water_removal: str = "centre",
    origin: tuple[float, float] | None = None,
    progress: Progress | None = None,
) -> tuple[NDArray[np.float64], list[list[int]], tuple[float, float]]:
    """Generate a refined quad mesh; the input is validated by the caller.

    Parameters
    ----------
    domain : shapely geometry
        Polygonal area the mesh covers.
    refinement_areas : sequence of (shapely geometry, int)
        Polygonal areas with the refinement level (1 to ``levels``) their cells reach.
    cell_size : float
        Base cell size.
    levels : int
        Highest refinement level.
    transition_rows : sequence of int
        Per step from coarse to fine (index = level of the step), at least 1 each.
    water_areas : sequence of shapely geometry
        Areas where no cells are kept.
    water_removal : {"centre", "inside", "touches"}
        Remove a lattice square when its centre lies in water, when it lies
        completely in water, or as soon as it touches water.
    origin : tuple of float, optional
        A lattice point; the base grid starts at the largest lattice point at or
        below the lower-left corner of the domain. By default a multiple of the cell
        size.
    progress : callable, optional
        Called with ``(percent or None, message)``; a truthy result cancels.

    Returns
    -------
    vertices : numpy.ndarray of float, shape (n, 2)
        Node coordinates (some nodes may be unused).
    cells : list of list of int
        Node indices per cell.
    origin : tuple of float
        Start of the base grid, a lattice point: the lower-left corner of the domain
        rounded down to the lattice.

    Raises
    ------
    InputError
        If no cells are left inside the domain.
    RawRepairError
        If the orthogonality repair does not close.
    """
    report = Reporter(progress)
    domain_valid = make_valid(domain, "the domain")
    grid = make_grid(domain_valid.bounds, cell_size, levels, origin)
    water = prepare_water(water_areas, water_removal)
    candidates = prepare_refinement_areas(refinement_areas, cell_size, transition_rows, report)
    queue = base_cells(domain_valid, grid, candidates, report)
    vertices, cells = _run(queue, grid, candidates, domain_valid, water, None, progress, report)
    return vertices, cells, (grid.ox, grid.oy)


@dataclass(frozen=True)
class Existing:
    """An existing mesh as the start of refine, in lattice units of the finest new level.

    Attributes
    ----------
    leaves : list of Leaf
        Its lattice squares, in start order.
    cells_of : dict
        Its cells (node numbers) per lattice square.
    points : list of (int, int)
        Lattice point of every node, in node order.
    coordinates : numpy.ndarray of float, shape (n, 2)
        Stored coordinate of every node; kept for the nodes of the new mesh.
    partial : shapely geometry or None
        The lattice squares that their cells fill only partly (cells were removed
        after generation, such as clipping along water).
    footprint : shapely geometry or None
        The old cells of those squares: a new cell inside ``partial`` stays only when
        it lies completely within them, so the mesh never grows.
    """

    leaves: list[Leaf]
    cells_of: dict[Key, list[list[int]]]
    points: list[tuple[int, int]]
    coordinates: NDArray[np.float64]
    partial: BaseGeometry | None = None
    footprint: BaseGeometry | None = None


def refine_cells(
    existing: Existing,
    refinement_areas: Sequence[tuple[BaseGeometry, int]],
    *,
    origin: tuple[float, float],
    cell_size: float,
    levels: int,
    transition_rows: Sequence[int],
    domain: BaseGeometry | None = None,
    water_areas: Sequence[BaseGeometry] = (),
    water_removal: str = "centre",
    progress: Progress | None = None,
) -> tuple[NDArray[np.float64], list[list[int]]]:
    """Refine an existing mesh; the input is validated and recognised by the caller.

    Parameters
    ----------
    existing : Existing
        The recognised mesh.
    refinement_areas : sequence of (shapely geometry, int)
        The new refinement areas.
    origin : tuple of float
        Lattice point of index ``(0, 0)``.
    cell_size : float
        Base cell size.
    levels : int
        Highest refinement level of the result.
    transition_rows : sequence of int
        Per step from coarse to fine, for the new areas.
    domain : shapely geometry, optional
        Removes new cells outside it.
    water_areas : sequence of shapely geometry
        Remove new cells by the water removal rule.
    water_removal : {"centre", "inside", "touches"}
        Water removal rule.
    progress : callable, optional
        Called with ``(percent or None, message)``; a truthy result cancels.

    Returns
    -------
    vertices : numpy.ndarray of float, shape (n, 2)
        Node coordinates: the stored ones for existing nodes, then the new ones.
    cells : list of list of int
        Node indices per cell.
    """
    report = Reporter(progress)
    grid = Grid(ox=origin[0], oy=origin[1], unit=cell_size / (2**levels), cell_size=cell_size,
                levels=levels, nx=0, ny=0)  # fmt: skip
    domain_valid = None if domain is None else make_valid(domain, "the domain")
    water = prepare_water(water_areas, water_removal)
    candidates = prepare_refinement_areas(refinement_areas, cell_size, transition_rows, report)
    queue = existing_cells(existing.leaves, grid, candidates)
    return _run(queue, grid, candidates, domain_valid, water, existing, progress, report)


def _run(
    queue: list[Queued],
    grid: Grid,
    candidates: Candidates,
    domain: BaseGeometry | None,
    water: WaterAreas,
    existing: Existing | None,
    progress: Progress | None,
    report: Reporter,
) -> tuple[NDArray[np.float64], list[list[int]]]:
    """Run the steps from the quadtree to the repaired cells.

    For generate ``existing`` is None and domain and water select every lattice
    square. For refine they select only new squares, and an existing square whose
    split sides do not change keeps its own cells.
    """
    levels = grid.levels
    leaves = build_quadtree(queue, grid, candidates, report)

    report(40, "Quad Grid: 2:1 balancing...")
    cell_dict: dict[Key, int] = {(ix0, iy0, ix1, iy1): lvl for ix0, iy0, ix1, iy1, lvl in leaves}
    balance_2_to_1(cell_dict, levels, report)
    report(43, "Quad Grid: T-junction elimination (three-side halving)...")
    eliminate_t_junctions(cell_dict, levels, report)
    leaves = [(ix0, iy0, ix1, iy1, lvl) for (ix0, iy0, ix1, iy1), lvl in cell_dict.items()]

    report(50, f"Quad Grid: {len(leaves)} cells after balancing, clipping...")
    if existing is None:
        kept = select_leaves(leaves, domain, grid)
    else:
        old = {leaf[:4] for leaf in existing.leaves}
        new = [leaf for leaf in leaves if leaf[:4] not in old]
        selected = {leaf[:4] for leaf in select_leaves(new, domain, grid)}
        kept = [leaf for leaf in leaves if leaf[:4] in old or leaf[:4] in selected]
    if not kept:
        raise InputError("no cells are left inside the domain; check the domain and water areas")
    # Only the kept squares count as neighbours from here on.
    cell_dict = {(ix0, iy0, ix1, iy1): lvl for ix0, iy0, ix1, iy1, lvl in kept}

    report(70, f"Quad Grid: {len(kept)} valid cells, building nodes...")
    nodes = Nodes(None if existing is None else existing.points)
    sides = split_sides(cell_dict, kept)
    keep = None
    if existing is not None:
        old_dict = {leaf[:4]: leaf[4] for leaf in existing.leaves}
        old_sides = split_sides(old_dict, existing.leaves)
        keep = {
            key: cells
            for key, cells in existing.cells_of.items()
            if key in cell_dict and sides[key] == old_sides[key]
        }
    report(80, f"Quad Grid: converting {len(kept)} cells to mesh elements...")
    cells = emit_cells(kept, sides, nodes, report, keep)

    report(99, "Quad Grid: head-to-toe correction...")
    head_to_toe(cells, nodes, report)
    resolve_hanging_nodes(cells, nodes)
    hanging = count_hanging_nodes(cells, nodes)
    if hanging:
        _log.warning("[QuadRef] %d hanging nodes after the conformity pass", hanging)

    report(msg="Quad Grid: orthogonality check...")
    log = None if progress is None else (lambda m: progress(None, m))
    raw_cells = [list(c) for c in cells]  # the repair changes cells in place
    origin_xy = (grid.ox, grid.oy)
    try:
        fixed = (
            None
            if existing is None
            else {tuple(sorted(c)) for cs in existing.cells_of.values() for c in cs}
        )
        repair_orthogonality(cells, nodes, origin_xy, grid.unit, log=log, fixed=fixed)
    except OrthogonalityRepairError as exc:
        raise RawRepairError(
            str(exc), exc.bad_points, _coordinates(nodes, grid, existing), raw_cells, origin_xy
        ) from None

    vertices = _coordinates(nodes, grid, existing)
    cells = _remove_cells(cells, vertices, water, existing)
    if not cells:
        raise InputError("no cells are left inside the domain; check the domain and water areas")
    report(100, f"Quad Grid: done - {len(vertices)} nodes, {len(cells)} elements.")
    return vertices, cells


def _remove_cells(
    cells: list[list[int]],
    vertices: NDArray[np.float64],
    water: WaterAreas,
    existing: Existing | None,
) -> list[list[int]]:
    """Clip the finished mesh: water with MeshKernel, and for refine the old footprint.

    Generate clips the whole mesh, like generating without water and clipping with
    MeshKernel afterwards. Refine clips only new cells: MeshKernel gets the water
    within the area of the new cells, and a new cell in a partly filled old square
    goes when it does not lie completely within the old cells.
    """
    polygons = list(water.geometries)
    if existing is None:
        return clip_cells(vertices, cells, polygons, water.rule)
    old = {tuple(sorted(c)) for cs in existing.cells_of.values() for c in cs}
    new = [i for i, c in enumerate(cells) if tuple(sorted(c)) not in old]
    if new and existing.partial is not None and existing.footprint is not None:
        centroids = shapely.points(np.array([vertices[cells[i]].mean(axis=0) for i in new]))
        shapes = np.array([shapely.Polygon(vertices[cells[i]]) for i in new], dtype=object)
        outside = shapely.covers(existing.partial, centroids) & ~shapely.covers(
            existing.footprint, shapes
        )
        drop = {new[k] for k in np.flatnonzero(outside)}
        cells = [c for i, c in enumerate(cells) if i not in drop]
        new = [i for i, c in enumerate(cells) if tuple(sorted(c)) not in old]
    if not new or not polygons:
        return cells
    region = shapely.union_all([shapely.Polygon(vertices[cells[i]]) for i in new])
    within = [p for p in (shapely.intersection(w, region) for w in polygons) if not p.is_empty]
    return clip_cells(vertices, cells, within, water.rule)


def _coordinates(nodes: Nodes, grid: Grid, existing: Existing | None) -> NDArray[np.float64]:
    """Node coordinates ``origin + index * unit``, computed once; existing nodes keep theirs."""
    pts = np.array(nodes.points, dtype=np.float64).reshape(-1, 2)
    vertices = np.empty_like(pts)
    vertices[:, 0] = grid.ox + pts[:, 0] * grid.unit
    vertices[:, 1] = grid.oy + pts[:, 1] * grid.unit
    if existing is not None:
        vertices[: len(existing.coordinates)] = existing.coordinates
    return vertices
