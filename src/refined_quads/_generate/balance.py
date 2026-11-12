"""2:1 balance of lattice squares: making a quadtree balanced, and checking a mesh."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from numpy.typing import NDArray

from refined_quads._common.grid import Key, neighbours, split
from refined_quads._common.progress import Reporter
from refined_quads._common.recognise import LatticeSquares
from refined_quads._common.topology import PointLookup

_log = logging.getLogger("refined_quads._generate.engine")


def find_ancestor(
    cells: dict[Key, int], levels: int, ax0: int, ay0: int, ax1: int, ay1: int
) -> tuple[Key, int] | None:
    """Return the leaf (key and level) that contains the square ``(ax0, ay0, ax1, ay1)``.

    Looks at the enclosing aligned squares of each coarser level; ``None`` when no
    leaf contains it (outside the tree, or the square is split further).
    """
    iw = ax1 - ax0
    ih = ay1 - ay0
    cx = (ax0 + ax1) >> 1
    cy = (ay0 + ay1) >> 1
    for _ in range(levels + 1):
        iw <<= 1
        ih <<= 1
        px0 = (cx // iw) * iw
        py0 = (cy // ih) * ih
        pk = (px0, py0, px0 + iw, py0 + ih)
        if pk in cells:
            return pk, cells[pk]
    return None


def balance_2_to_1(cells: dict[Key, int], levels: int, report: Reporter) -> None:
    """Split leaves until no two neighbouring leaves differ more than one level.

    Parameters
    ----------
    cells : dict
        Leaves of the quadtree (key to level); changed in place.
    levels : int
        Highest refinement level.
    report : Reporter
        Progress.
    """
    queue = sorted(cells.keys(), key=lambda k: -cells[k])
    iteration = 0
    while queue:
        iteration += 1
        if iteration % 2000 == 0:
            report(msg=f"Quad Grid: 2:1 balancing... {len(queue)} left")
        k = queue.pop()
        if k not in cells:
            continue
        lvl = cells[k]
        for nk in neighbours(k):
            if nk in cells:
                n_lvl = cells[nk]
                if n_lvl < lvl - 1:
                    del cells[nk]
                    queue.extend(split(cells, nk, n_lvl + 1))
            else:
                found = find_ancestor(cells, levels, *nk)
                if found is not None:
                    pk, p_lvl = found
                    if p_lvl < lvl - 1:
                        del cells[pk]
                        queue.extend(split(cells, pk, p_lvl + 1))


def finer_exists(cells: dict[Key, int], ix0: int, iy0: int, ix1: int, iy1: int, side: str) -> bool:
    """Tell whether a leaf has a finer neighbour (a half-size leaf) on ``side`` (N, S, E, W)."""
    hw = (ix1 - ix0) >> 1
    hh = (iy1 - iy0) >> 1
    if side == "N":
        return (ix0, iy1, ix0 + hw, iy1 + hh) in cells or (ix0 + hw, iy1, ix1, iy1 + hh) in cells
    if side == "S":
        return (ix0, iy0 - hh, ix0 + hw, iy0) in cells or (ix0 + hw, iy0 - hh, ix1, iy0) in cells
    if side == "E":
        return (ix1, iy0, ix1 + hw, iy0 + hh) in cells or (ix1, iy0 + hh, ix1 + hw, iy1) in cells
    return (ix0 - hw, iy0, ix0, iy0 + hh) in cells or (ix0 - hw, iy0 + hh, ix0, iy1) in cells


def eliminate_t_junctions(cells: dict[Key, int], levels: int, report: Reporter) -> None:
    """Split leaves that have finer neighbours on three or four sides.

    Such a leaf would need a centre split with midpoints on every side anyway; with
    three finer sides it is split only when the fourth side has a neighbour at all.
    Splitting can make neighbours qualify, so they are queued again.

    Parameters
    ----------
    cells : dict
        Leaves of the quadtree (key to level); changed in place.
    levels : int
        Highest refinement level.
    report : Reporter
        Progress.
    """
    queue: deque[Key] = deque()
    in_queue: set[Key] = set()
    for k, lvl in cells.items():
        if lvl < levels:
            queue.append(k)
            in_queue.add(k)

    total = 0
    processed = 0
    while queue:
        k = queue.popleft()
        in_queue.discard(k)
        processed += 1
        if k not in cells:
            continue
        lvl = cells[k]
        if lvl >= levels:
            continue
        s_n = finer_exists(cells, *k, "N")
        s_s = finer_exists(cells, *k, "S")
        s_e = finer_exists(cells, *k, "E")
        s_w = finer_exists(cells, *k, "W")
        n_sp = s_n + s_s + s_e + s_w
        if n_sp < 3:
            continue
        if n_sp == 3:
            free = 0 if not s_n else 1 if not s_s else 2 if not s_e else 3
            side_key = neighbours(k)[free]
            if side_key not in cells and find_ancestor(cells, levels, *side_key) is None:
                continue

        del cells[k]
        split(cells, k, lvl + 1)
        total += 1
        for ncoords in neighbours(k):
            _enqueue(cells, levels, ncoords, queue, in_queue)
            if ncoords not in cells:
                found = find_ancestor(cells, levels, *ncoords)
                if found is not None:
                    _enqueue(cells, levels, found[0], queue, in_queue)

        if total % 200 == 0:
            report(msg=f"Quad Grid: T-junction elimination, {total} split, {len(queue)} queued...")

    _log.debug(
        "[QuadRef] T-junction elimination: %d split (%d checked), %d cells",
        total,
        processed,
        len(cells),
    )


def _enqueue(
    cells: dict[Key, int], levels: int, k: Key, queue: deque[Key], in_queue: set[Key]
) -> None:
    """Queue a leaf that can still be split, once."""
    if k in cells and k not in in_queue and cells[k] < levels:
        in_queue.add(k)
        queue.append(k)


def balance_violations(squares: LatticeSquares, levels: int) -> NDArray[np.int64]:
    """Find sides where a lattice square meets a square more than one level coarser.

    Parameters
    ----------
    squares : LatticeSquares
        The lattice squares of a mesh.
    levels : int
        Highest refinement level of the lattice.

    Returns
    -------
    numpy.ndarray of int, shape (v, 2)
        Midpoint of each violating side of the finer square, in units of half the
        finest level (so that midpoints of the finest squares are integers).

    Notes
    -----
    Every side of every square is probed just outside its midpoint. A coarser
    neighbour along that side always covers the midpoint, because lattice squares
    are aligned to their level; a finer neighbour is found from its own side.
    """
    side = 2 ** (levels - squares.level)
    # Work in half units so that a probe half a unit outside a side is an integer point.
    x0 = 2 * squares.x * side
    y0 = 2 * squares.y * side
    s2 = 2 * side
    mid_x, mid_y = x0 + side, y0 + side
    probes = np.concatenate(
        [
            np.stack([x0 + s2 + 1, mid_y], axis=1),
            np.stack([x0 - 1, mid_y], axis=1),
            np.stack([mid_x, y0 + s2 + 1], axis=1),
            np.stack([mid_x, y0 - 1], axis=1),
        ]
    )
    sides = np.concatenate(
        [
            np.stack([x0 + s2, mid_y], axis=1),
            np.stack([x0, mid_y], axis=1),
            np.stack([mid_x, y0 + s2], axis=1),
            np.stack([mid_x, y0], axis=1),
        ]
    )
    probe_level = np.tile(squares.level, 4)
    violating = np.zeros(len(probes), dtype=bool)
    for level in range(levels - 1):
        here = squares.level == level
        if not here.any():
            continue
        lookup = PointLookup(np.stack([squares.x[here], squares.y[here]], axis=1))
        candidates = probe_level >= level + 2
        cell = np.floor_divide(probes[candidates], 2 * 2 ** (levels - level))
        violating[np.flatnonzero(candidates)[lookup.find(cell) >= 0]] = True
    return np.asarray(sides[violating], dtype=np.int64)
