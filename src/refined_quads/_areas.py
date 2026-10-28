"""Refinement areas and their transition zones, prepared for the quadtree."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import shapely
from shapely import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.prepared import PreparedGeometry, prep

from refined_quads._progress import Reporter

_log = logging.getLogger("refined_quads._engine")

if TYPE_CHECKING:
    Prepared = PreparedGeometry[BaseGeometry]
    #: A refinement area: its level, the prepared polygon and the prepared transition
    #: zones with the level of the step they belong to, coarsest (largest) zone first.
    Candidate = tuple[int, Prepared, list[tuple[Prepared, int]]]


@dataclass(frozen=True)
class Candidates:
    """The refinement areas of a run, with an STRtree on their outermost zones."""

    items: list[Candidate]
    tree: STRtree | None

    def query(self, cell: BaseGeometry) -> tuple[int, ...]:
        """Return the indices of the candidates whose outermost zone may touch ``cell``."""
        if self.tree is None:
            return ()
        return tuple(sorted(int(i) for i in self.tree.query(cell)))


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
        (_simplify(geom if geom.is_valid else geom.buffer(0), simplify_tol), level)
        for geom, level in refinement_areas
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
            zones.append((poly.buffer(cum), lvl))
        items.append((target_lvl, prep(poly), [(prep(z), lvl) for z, lvl in zones]))
        outer.append(zones[0][0] if zones else poly)
    return Candidates(items, STRtree(outer) if items else None)


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


def required_level(
    cell: BaseGeometry, base_lvl: int, cell_cands: tuple[int, ...], items: list[Candidate]
) -> tuple[int, int, list[int]]:
    """Return the level a cell needs, the level fixed for its children and their candidates.

    Parameters
    ----------
    cell : shapely geometry
        The cell.
    base_lvl : int
        Level fixed by an ancestor that lies completely inside an area or zone.
    cell_cands : tuple of int
        Candidates that may still touch the cell.
    items : list of Candidate
        All candidates.

    Returns
    -------
    req : int
        Required level of the cell.
    child_base : int
        Level fixed for all descendants (the cell lies completely inside the area or
        zone that gives it).
    keep : list of int
        Candidates the children still have to test. A candidate the cell does not
        touch at all (outermost zone included) cannot touch its children either.
    """
    req = base_lvl
    child_base = base_lvl
    keep = []
    for ci in cell_cands:
        level, p_main, zp = items[ci]
        if p_main.intersects(cell):
            req = max(req, level)
            if p_main.contains_properly(cell):
                child_base = max(child_base, level)
            else:
                keep.append(ci)
        else:
            # Outside the area: the finest touching zone decides. Finer zones are
            # subsets of coarser ones, so the first hit in reversed order is the
            # finest, and untouched finer zones stay out of reach for children.
            for p_buf, lvl_candidate in reversed(zp):
                if p_buf.intersects(cell):
                    contrib = lvl_candidate + 1
                    req = max(req, contrib)
                    if p_buf.contains_properly(cell):
                        child_base = max(child_base, contrib)
                    else:
                        keep.append(ci)
                    break
    return req, child_base, keep
