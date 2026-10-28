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

import numpy as np
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from refined_quads._areas import prepare_refinement_areas
from refined_quads._balance import balance_2_to_1, eliminate_t_junctions
from refined_quads._conformity import count_hanging_nodes, head_to_toe, resolve_hanging_nodes
from refined_quads._faces import Nodes, emit_cells, split_sides
from refined_quads._grid import Key, make_grid
from refined_quads._orthogonality import repair_orthogonality
from refined_quads._progress import Callback as Progress
from refined_quads._progress import Reporter
from refined_quads._quadtree import build_quadtree
from refined_quads._select import prepare_water, select_leaves
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

    domain_valid = domain if domain.is_valid else domain.buffer(0)
    grid = make_grid(domain_valid.bounds, cell_size, levels, origin)
    water = prepare_water(water_areas, water_removal)
    candidates = prepare_refinement_areas(refinement_areas, cell_size, transition_rows, report)
    leaves = build_quadtree(prep(domain_valid), grid, candidates, report)

    report(40, "Quad Grid: 2:1 balancing...")
    cell_dict: dict[Key, int] = {(ix0, iy0, ix1, iy1): lvl for ix0, iy0, ix1, iy1, lvl in leaves}
    balance_2_to_1(cell_dict, levels, report)
    report(43, "Quad Grid: T-junction elimination (three-side halving)...")
    eliminate_t_junctions(cell_dict, levels, report)
    leaves = [(ix0, iy0, ix1, iy1, lvl) for (ix0, iy0, ix1, iy1), lvl in cell_dict.items()]

    report(50, f"Quad Grid: {len(leaves)} cells after balancing, clipping...")
    kept = select_leaves(leaves, domain_valid, grid, water)
    if not kept:
        raise InputError("no cells are left inside the domain; check the domain and water areas")
    # Only the kept squares count as neighbours from here on, or the edge of every
    # water hole gets an inconsistent midpoint; with many small water areas this added
    # up to thousands of pinch nodes (Parkstad).
    cell_dict = {(ix0, iy0, ix1, iy1): lvl for ix0, iy0, ix1, iy1, lvl in kept}

    report(70, f"Quad Grid: {len(kept)} valid cells, building nodes...")
    nodes = Nodes()
    sides = split_sides(cell_dict, kept)
    report(80, f"Quad Grid: converting {len(kept)} cells to mesh elements...")
    cells = emit_cells(kept, sides, nodes, report)

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
        repair_orthogonality(cells, nodes, origin_xy, grid.unit, log=log)
    except OrthogonalityRepairError as exc:
        raise RawRepairError(
            str(exc), exc.bad_points, _coordinates(nodes, grid.ox, grid.oy, grid.unit),
            raw_cells, origin_xy,
        ) from None  # fmt: skip

    vertices = _coordinates(nodes, grid.ox, grid.oy, grid.unit)
    report(100, f"Quad Grid: done - {len(vertices)} nodes, {len(cells)} elements.")
    return vertices, cells, origin_xy


def _coordinates(nodes: Nodes, ox: float, oy: float, unit: float) -> NDArray[np.float64]:
    """Node coordinates ``origin + index * unit``, computed once."""
    pts = np.array(nodes.points, dtype=np.float64).reshape(-1, 2)
    vertices = np.empty_like(pts)
    vertices[:, 0] = ox + pts[:, 0] * unit
    vertices[:, 1] = oy + pts[:, 1] * unit
    return vertices
