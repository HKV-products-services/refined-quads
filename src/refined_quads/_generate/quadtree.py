"""Build the quadtree: split base cells until every cell has its required level."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from refined_quads._common.grid import Grid, Leaf
from refined_quads._common.progress import Reporter
from refined_quads._generate.areas import Candidates, required_levels

#: A queued cell: key, its level, the level fixed by an ancestor and its candidates.
Queued = tuple[int, int, int, int, int, int, tuple[int, ...]]


def base_cells(
    domain: BaseGeometry, grid: Grid, candidates: Candidates, report: Reporter
) -> list[Queued]:
    """Return the base cells that touch the domain, ready for :func:`build_quadtree`."""
    base = grid.base
    report(5, f"Quad Grid: base grid ({grid.nx}x{grid.ny} = {grid.nx * grid.ny} cells)...")
    gx, gy = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny), indexing="ij")
    ix0 = gx.ravel() * base
    iy0 = gy.ravel() * base
    keys = np.column_stack([ix0, iy0, ix0 + base, iy0 + base]).astype(np.int64)
    cells = grid.boxes(keys)
    shapely.prepare(domain)
    touching = shapely.intersects(domain, cells)
    keys, cells = keys[touching], cells[touching]
    return [
        (ix0, iy0, ix1, iy1, 0, 0, cands)
        for (ix0, iy0, ix1, iy1), cands in zip(keys.tolist(), candidates.query(cells), strict=True)
    ]


def existing_cells(leaves: list[Leaf], grid: Grid, candidates: Candidates) -> list[Queued]:
    """Return the lattice squares of an existing mesh, ready for :func:`build_quadtree`."""
    keys = np.asarray([leaf[:4] for leaf in leaves], dtype=np.int64).reshape(-1, 4)
    cands = candidates.query(grid.boxes(keys))
    return [
        (ix0, iy0, ix1, iy1, level, 0, c)
        for (ix0, iy0, ix1, iy1, level), c in zip(leaves, cands, strict=True)
    ]


def build_quadtree(
    queue: list[Queued], grid: Grid, candidates: Candidates, report: Reporter
) -> list[Leaf]:
    """Split the queued cells until every cell has its required level; return the leaves.

    Parameters
    ----------
    queue : list
        Start cells: the base cells that touch the domain (generate) or the lattice
        squares of an existing mesh (refine).
    grid : Grid
        The base grid.
    candidates : Candidates
        The prepared refinement areas.
    report : Reporter
        Progress.

    Returns
    -------
    list of Leaf
        ``(ix0, iy0, ix1, iy1, level)`` per leaf, in the order of a depth-first walk
        that takes the start cells from the back of the queue and the children of a
        cell from north-east to south-west. Node numbering follows this order, and the
        head-to-toe correction depends on it.

    Notes
    -----
    The tree is built one generation at a time, with the geometry tests of a
    generation in bulk. Candidates are pruned along the tree: children inherit only
    the candidates that their parent touches, and a cell completely inside an area or
    zone fixes that level for its whole subtree, which then needs no geometry tests.
    A start cell is never coarsened.
    """
    report(10, f"Quad Grid: quadtree refinement ({len(queue)} start cells)...")
    max_levels = grid.levels
    n = len(queue)
    keys = np.asarray([q[:4] for q in queue], dtype=np.int64).reshape(-1, 4)
    level = np.asarray([q[4] for q in queue], dtype=np.int64)
    base_lvl = np.asarray([q[5] for q in queue], dtype=np.int64)
    cands = [q[6] for q in queue]
    # Position in the depth-first walk: the rank of the start cell, then per level
    # below it a base-4 digit (3 for the south-west child down to 0 for the
    # north-east one). Leaves are never each other's ancestors, so sorting on this
    # number gives the walk.
    shift = 4**max_levels
    walk = [(n - 1 - i) * shift for i in range(n)]
    # Progress by covered area: a base cell weighs 4**max_levels units, a cell at
    # level L 4**(max_levels - L); an area is done when its cell becomes a leaf.
    weight_total = sum(4 ** (max_levels - int(lvl)) for lvl in level) or 1
    weight_done = 0
    found: list[tuple[int, Leaf]] = []

    while len(cands):
        req = base_lvl.copy()
        child_base = base_lvl.copy()
        child_cands: list[tuple[int, ...]] = [()] * len(cands)
        tested = np.flatnonzero([bool(c) for c in cands])
        if tested.size:
            r, b, k = required_levels(
                grid.boxes(keys[tested]),
                base_lvl[tested],
                [cands[i] for i in tested],
                candidates.items,
            )
            req[tested] = r
            child_base[tested] = b
            for i, c in zip(tested.tolist(), k, strict=True):
                child_cands[i] = c
        split = (level < req) & (level < max_levels)

        done = np.flatnonzero(~split)
        for i, key, lvl in zip(
            done.tolist(), keys[done].tolist(), level[done].tolist(), strict=True
        ):
            found.append((walk[i], (*key, lvl)))
            weight_done += 4 ** (max_levels - lvl)

        parents = np.flatnonzero(split)
        x0, y0, x1, y1 = keys[parents].T
        xm = (x0 + x1) >> 1
        ym = (y0 + y1) >> 1
        # Children SW, SE, NW, NE of every parent, child-major.
        keys = np.concatenate(
            [
                np.column_stack([x0, y0, xm, ym]),
                np.column_stack([xm, y0, x1, ym]),
                np.column_stack([x0, ym, xm, y1]),
                np.column_stack([xm, ym, x1, y1]),
            ]
        )
        level = np.tile(level[parents] + 1, 4)
        base_lvl = np.tile(child_base[parents], 4)
        parent_list = parents.tolist()
        cands = [child_cands[i] for i in parent_list] * 4
        digit = [shift >> (2 * lvl) for lvl in (level[: len(parent_list)]).tolist()]
        walk = [
            walk[i] + (3 - child) * d
            for child in range(4)
            for i, d in zip(parent_list, digit, strict=True)
        ]

        report(
            10 + int(30 * weight_done / weight_total),
            f"Quad Grid: refining... {len(found)} leaf cells, {len(cands)} queued",
        )

    found.sort(key=lambda item: item[0])
    return [leaf for _, leaf in found]
