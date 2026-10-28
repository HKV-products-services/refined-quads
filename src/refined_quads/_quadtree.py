"""Build the quadtree: split base cells until every cell has its required level."""

from __future__ import annotations

import time

from shapely.geometry.base import BaseGeometry
from shapely.prepared import PreparedGeometry

from refined_quads._areas import Candidates, required_level
from refined_quads._grid import Grid, Leaf
from refined_quads._progress import Reporter, fmt_elapsed

# A queued cell: key, its level, the level fixed by an ancestor and its candidates.
_Queued = tuple[int, int, int, int, int, int, tuple[int, ...]]


def build_quadtree(
    domain: PreparedGeometry[BaseGeometry],
    grid: Grid,
    candidates: Candidates,
    report: Reporter,
) -> list[Leaf]:
    """Return the leaves of the quadtree over the base cells that touch the domain.

    Parameters
    ----------
    domain : prepared shapely geometry
        The domain.
    grid : Grid
        The base grid.
    candidates : Candidates
        The prepared refinement areas.
    report : Reporter
        Progress.

    Returns
    -------
    list of Leaf
        ``(ix0, iy0, ix1, iy1, level)`` per leaf, in the order they were completed.

    Notes
    -----
    Candidates are pruned along the tree: children inherit only the candidates that
    their parent touches, and a cell completely inside an area or zone fixes that
    level for its whole subtree, which then needs no geometry tests.
    """
    base = grid.base
    report(5, f"Quad Grid: base grid ({grid.nx}x{grid.ny} = {grid.nx * grid.ny} cells)...")

    queue: list[_Queued] = []
    for gx in range(grid.nx):
        for gy in range(grid.ny):
            ix0 = gx * base
            iy0 = gy * base
            cell = grid.box(ix0, iy0, ix0 + base, iy0 + base)
            if domain.intersects(cell):
                queue.append((ix0, iy0, ix0 + base, iy0 + base, 0, 0, candidates.query(cell)))

    report(10, f"Quad Grid: quadtree refinement ({len(queue)} base cells)...")

    max_levels = grid.levels
    leaves: list[Leaf] = []
    processed = 0
    report_every = max(1, min(10_000, len(queue) // 20))
    # Progress by covered area: a base cell weighs 4**max_levels units, a cell at
    # level L 4**(max_levels - L); an area is done when its cell becomes a leaf.
    weight_total = len(queue) * (4**max_levels)
    weight_done = 0
    started = time.perf_counter()

    while queue:
        ix0, iy0, ix1, iy1, cur_lvl, base_lvl, q_cands = queue.pop()
        if q_cands:
            req_lvl, child_base, child_cands_l = required_level(
                grid.box(ix0, iy0, ix1, iy1), base_lvl, q_cands, candidates.items
            )
            child_cands = tuple(child_cands_l)
        else:
            # No candidates left: the required level is fixed for the whole subtree.
            req_lvl = base_lvl
            child_base = base_lvl
            child_cands = ()
        if cur_lvl < req_lvl and cur_lvl < max_levels:
            ixm = (ix0 + ix1) >> 1
            iym = (iy0 + iy1) >> 1
            queue.append((ix0, iy0, ixm, iym, cur_lvl + 1, child_base, child_cands))
            queue.append((ixm, iy0, ix1, iym, cur_lvl + 1, child_base, child_cands))
            queue.append((ix0, iym, ixm, iy1, cur_lvl + 1, child_base, child_cands))
            queue.append((ixm, iym, ix1, iy1, cur_lvl + 1, child_base, child_cands))
        else:
            leaves.append((ix0, iy0, ix1, iy1, cur_lvl))
            weight_done += 4 ** (max_levels - cur_lvl)

        processed += 1
        if processed % report_every == 0:
            fraction = weight_done / weight_total
            elapsed = time.perf_counter() - started
            eta = ""
            if fraction > 0.02:
                eta = f"  ~{fmt_elapsed(max(0.0, elapsed / fraction - elapsed))} left"
            report(
                10 + int(30 * fraction),
                f"Quad Grid: refining... {processed} processed, {len(queue)} queued, "
                f"{len(leaves)} leaf cells{eta}",
            )
    return leaves
