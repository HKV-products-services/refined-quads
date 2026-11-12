"""Refinement areas and their transition zones, prepared for the quadtree."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from refined_quads._common.progress import Reporter

_log = logging.getLogger("refined_quads._generate.engine")

#: A refinement area: its level, the prepared polygon and the prepared transition zones
#: with the level of the step they belong to, coarsest (largest) zone first.
Candidate = tuple[int, BaseGeometry, list[tuple[BaseGeometry, int]]]


@dataclass(frozen=True)
class Candidates:
    """The refinement areas of a run, with an STRtree on their outermost zones."""

    items: list[Candidate]
    tree: STRtree | None

    def query(self, cells: NDArray[np.object_]) -> list[tuple[int, ...]]:
        """Return per cell the sorted candidates whose outermost zone may touch it."""
        found: list[tuple[int, ...]] = [()] * len(cells)
        if self.tree is None or not len(cells):
            return found
        cell_i, cand_i = self.tree.query(cells)
        if not len(cell_i):
            return found
        order = np.lexsort((cand_i, cell_i))
        cell_i, cand_i = cell_i[order], cand_i[order]
        starts = np.flatnonzero(np.diff(cell_i)) + 1
        for i, group in zip(
            cell_i[np.r_[0, starts]].tolist(), np.split(cand_i, starts), strict=True
        ):
            found[i] = tuple(group.tolist())
        return found


def prepare_refinement_areas(
    refinement_areas: Sequence[tuple[BaseGeometry, int]],
    cell_size: float,
    transition_rows: Sequence[int],
    report: Reporter,
) -> Candidates:
    """Make polygons valid, simplify them and build their transition zones.

    Parameters
    ----------
    refinement_areas : sequence of (shapely geometry, int)
        Polygons with their refinement level.
    cell_size : float
        Base cell size.
    transition_rows : sequence of int
        Rows per level step from coarse to fine (index = level of the step).
    report : Reporter
        Progress.

    Returns
    -------
    Candidates
        The prepared areas.

    Notes
    -----
    The zone for level ``lvl`` around an area of level ``target`` is the area
    buffered by the transition rows of all steps from ``lvl`` to ``target``, each in
    the cell size of the finer level of the step. A cell touching that zone needs
    level ``lvl + 1``.
    """
    simplify_tol = cell_size / 8.0
    polys = [
        (_simplify(make_valid(geom, f"refinement area {i}"), simplify_tol), level)
        for i, (geom, level) in enumerate(refinement_areas)
    ]
    n_polys = len(polys)
    if n_polys:
        report(1, f"Quad Grid: transition zones for {n_polys} refinement areas...")

    items: list[Candidate] = []
    outer = []
    for poly_i, (poly, target_lvl) in enumerate(polys):
        if n_polys > 20 and poly_i % max(1, n_polys // 20) == 0:
            report(
                1 + int(3 * poly_i / n_polys),
                f"Quad Grid: transition zones {poly_i}/{n_polys}...",
            )
        zones: list[tuple[BaseGeometry, int]] = []
        for lvl in range(target_lvl):
            cum = 0.0
            for inner_lvl in range(lvl, target_lvl):
                n = transition_rows[inner_lvl]
                cum += n * (cell_size / (2 ** (inner_lvl + 1)))
            zone = poly.buffer(cum)
            shapely.prepare(zone)
            zones.append((zone, lvl))
        shapely.prepare(poly)
        items.append((target_lvl, poly, zones))
        outer.append(zones[0][0] if zones else poly)
    return Candidates(items, STRtree(outer) if items else None)


def make_valid(geom: BaseGeometry, what: str) -> BaseGeometry:
    """Return ``geom`` made valid, keeping only its polygonal parts.

    The structure method keeps every enclosed area (both loops of a bowtie, where
    ``buffer(0)`` keeps one) and drops collapsed parts such as spikes, which buffering
    would otherwise turn into areas.
    """
    if geom.is_valid:
        return geom
    _log.warning("%s is not valid (%s); made valid", what, shapely.is_valid_reason(geom))
    return shapely.make_valid(geom, method="structure", keep_collapsed=False)


def _simplify(geom: BaseGeometry, tolerance: float) -> BaseGeometry:
    """Simplify a polygon with many vertices.

    Detail smaller than the cell size cannot be resolved by the quadtree, so this
    tolerance is invisible in the result; it keeps a polygon with hundreds of
    thousands of vertices from stalling the zone buffering.
    """
    n_before = int(shapely.get_num_coordinates(geom))
    if n_before <= 2000:
        return geom
    simplified = geom.simplify(tolerance, preserve_topology=True)
    _log.debug(
        "[QuadRef] refinement geometry simplified (tolerance=%.2fm): %d -> ~%d vertices",
        tolerance,
        n_before,
        int(shapely.get_num_coordinates(simplified)),
    )
    return simplified


def required_levels(
    cells: NDArray[np.object_],
    base_lvl: NDArray[np.int64],
    cell_cands: Sequence[tuple[int, ...]],
    items: list[Candidate],
) -> tuple[NDArray[np.int64], NDArray[np.int64], list[tuple[int, ...]]]:
    """Return per cell the level it needs, the level fixed for its children and their candidates.

    Parameters
    ----------
    cells : ndarray of shapely geometry
        The cells.
    base_lvl : ndarray of int
        Per cell the level fixed by an ancestor that lies completely inside an area or
        zone.
    cell_cands : sequence of tuple of int
        Per cell the candidates that may still touch it.
    items : list of Candidate
        All candidates.

    Returns
    -------
    req : ndarray of int
        Required level per cell.
    child_base : ndarray of int
        Level fixed for all descendants (the cell lies completely inside the area or
        zone that gives it).
    keep : list of tuple of int
        Per cell the candidates its children still have to test, sorted. A candidate
        the cell does not touch at all (outermost zone included) cannot touch its
        children either.
    """
    req = base_lvl.copy()
    child_base = base_lvl.copy()
    keep: list[list[int]] = [[] for _ in cell_cands]
    cells_of: defaultdict[int, list[int]] = defaultdict(list)
    for i, cands in enumerate(cell_cands):
        for ci in cands:
            cells_of[ci].append(i)
    # Candidates in ascending order, so every cell's list of kept candidates is sorted.
    for ci in sorted(cells_of):
        level, area, zones = items[ci]
        idx = np.asarray(cells_of[ci], dtype=np.int64)
        hit = shapely.intersects(area, cells[idx])
        _touch(area, level, idx[hit], cells, ci, req, child_base, keep)
        # Outside the area: the finest touching zone decides. Finer zones are subsets
        # of coarser ones, so the first hit in reversed order is the finest, and
        # untouched finer zones stay out of reach for children.
        rest = idx[~hit]
        for zone, lvl in reversed(zones):
            if not rest.size:
                break
            hit = shapely.intersects(zone, cells[rest])
            _touch(zone, lvl + 1, rest[hit], cells, ci, req, child_base, keep)
            rest = rest[~hit]
    return req, child_base, [tuple(k) for k in keep]


def _touch(
    geom: BaseGeometry,
    level: int,
    idx: NDArray[np.int64],
    cells: NDArray[np.object_],
    ci: int,
    req: NDArray[np.int64],
    child_base: NDArray[np.int64],
    keep: list[list[int]],
) -> None:
    """Apply an area or zone of ``level`` to the cells ``idx`` that touch it."""
    if not idx.size:
        return
    req[idx] = np.maximum(req[idx], level)
    inside = shapely.contains_properly(geom, cells[idx])
    child_base[idx[inside]] = np.maximum(child_base[idx[inside]], level)
    for i in idx[~inside].tolist():
        keep[i].append(ci)
