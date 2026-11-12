"""Conformity: head-to-toe correction and resolving hanging nodes."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from refined_quads._common.grid import Key
from refined_quads._common.progress import Reporter
from refined_quads._generate.faces import Nodes, square_cells

_log = logging.getLogger("refined_quads._generate.engine")


def head_to_toe(cells: list[list[int]], nodes: Nodes, report: Reporter) -> int:
    """Regenerate lattice squares whose transition triangles meet head to toe.

    Two triangles of neighbouring squares that share an axis-aligned edge, each with
    its apex beyond the middle of that edge on opposite ends, form a sliver pair.
    Both squares are emitted again with their current split sides. Repeats until no
    pair is left.

    Parameters
    ----------
    cells : list of list of int
        The cells; changed in place.
    nodes : Nodes
        The nodes; may get new midpoints and centres.
    report : Reporter
        Progress.

    Returns
    -------
    int
        Number of passes that changed cells.
    """
    passes = 0
    while _head_to_toe_pass(cells, nodes):
        passes += 1
        report(msg=f"Quad Grid: head-to-toe fix, iteration {passes}...")
    _log.debug("[H2T] done after %d iterations, %d cells", passes, len(cells))
    return passes


def _head_to_toe_pass(cells: list[list[int]], nodes: Nodes) -> bool:
    """Regenerate the two squares around each head-to-toe triangle pair; True if any."""
    tris_i = [i for i, c in enumerate(cells) if len(c) == 3]
    _log.debug("[H2T] pass start: %d cells, %d triangles", len(cells), len(tris_i))
    if len(tris_i) < 2:
        return False

    pts = np.array(nodes.points, dtype=np.int64)
    n_p = len(pts)
    tv = np.array([cells[i] for i in tris_i], dtype=np.int64)
    a_ = tv[:, [0, 1, 2]].ravel()
    b_ = tv[:, [1, 2, 0]].ravel()
    op = tv[:, [2, 0, 1]].ravel()
    ci = np.repeat(np.arange(len(tris_i)), 3)
    em = np.minimum(a_, b_)
    ex = np.maximum(a_, b_)
    ek = em.astype(np.int64) * (n_p + 1) + ex.astype(np.int64)
    order = np.argsort(ek, kind="stable")
    ek, ci, op, em, ex = ek[order], ci[order], op[order], em[order], ex[order]

    pp = np.where(ek[:-1] == ek[1:])[0]
    if not len(pp):
        return False
    ia, ib, ic, idd = em[pp], ex[pp], op[pp], op[pp + 1]
    ci1, ci2 = ci[pp], ci[pp + 1]
    ax_a, ay_a = pts[ia, 0], pts[ia, 1]
    bx_a, by_a = pts[ib, 0], pts[ib, 1]
    cx_a, cy_a = pts[ic, 0], pts[ic, 1]
    dx_a, dy_a = pts[idd, 0], pts[idd, 1]
    ab2 = (bx_a - ax_a) ** 2 + (by_a - ay_a) ** 2
    ac2 = (cx_a - ax_a) ** 2 + (cy_a - ay_a) ** 2
    bc2 = (cx_a - bx_a) ** 2 + (cy_a - by_a) ** 2
    ad2 = (dx_a - ax_a) ** 2 + (dy_a - ay_a) ** 2
    bd2 = (dx_a - bx_a) ** 2 + (dy_a - by_a) ** 2
    both_mid = (
        (ab2 > np.minimum(ac2, bc2))
        & (np.maximum(ac2, bc2) > ab2)
        & (ab2 > np.minimum(ad2, bd2))
        & (np.maximum(ad2, bd2) > ab2)
    )
    s = np.where(ab2 > 0, ab2, 1.0)
    t_c = ((cx_a - ax_a) * (bx_a - ax_a) + (cy_a - ay_a) * (by_a - ay_a)) / s
    t_d = ((dx_a - ax_a) * (bx_a - ax_a) + (dy_a - ay_a) * (by_a - ay_a)) / s
    h2t = ((t_c < 0.35) & (t_d > 0.65)) | ((t_c > 0.65) & (t_d < 0.35))
    bad = both_mid & h2t
    _log.debug(
        "[H2T] shared edges=%d, both_mid=%d, h2t=%d, bad=%d",
        len(pp),
        int(both_mid.sum()),
        int(h2t.sum()),
        int(bad.sum()),
    )
    if not bad.any():
        return False

    cells_of_node: dict[int, set[int]] = defaultdict(set)
    for i, cell in enumerate(cells):
        for v in cell:
            cells_of_node[v].add(i)

    to_delete: set[int] = set()
    new_cells: list[list[int]] = []
    done: set[Key] = set()
    for idx in np.where(bad)[0]:
        t1 = tris_i[int(ci1[idx])]
        t2 = tris_i[int(ci2[idx])]
        if t1 in to_delete or t2 in to_delete:
            continue
        ax, ay, bx, by = int(ax_a[idx]), int(ay_a[idx]), int(bx_a[idx]), int(by_a[idx])
        if ax != bx and ay != by:
            continue
        sq1 = _parent_square(ax, ay, bx, by, int(cx_a[idx]), int(cy_a[idx]))
        sq2 = _parent_square(ax, ay, bx, by, int(dx_a[idx]), int(dy_a[idx]))
        if sq1 in done or sq2 in done:
            continue
        done.add(sq1)
        done.add(sq2)
        to_delete.update(
            _cells_in_square(sq1, cells, nodes, cells_of_node)
            + _cells_in_square(sq2, cells, nodes, cells_of_node)
        )
        nodes.add((ax + bx) // 2, (ay + by) // 2)
        new_cells += _regenerate(sq1, nodes) + _regenerate(sq2, nodes)

    if not to_delete:
        _log.debug("[H2T] no cells found to remove")
        return False
    _log.debug("[H2T] removing %d cells, adding %d", len(to_delete), len(new_cells))
    for i in sorted(to_delete, reverse=True):
        cells.pop(i)
    cells.extend(new_cells)
    return True


def _parent_square(ax: int, ay: int, bx: int, by: int, gx: int, gy: int) -> Key:
    """Return the square on the side of point ``g`` that has edge ``a-b`` as side."""
    side = abs(bx - ax) + abs(by - ay)
    if ax == bx:
        y0, y1 = min(ay, by), max(ay, by)
        return (ax - side, y0, ax, y1) if gx <= ax else (ax, y0, ax + side, y1)
    x0, x1 = min(ax, bx), max(ax, bx)
    return (x0, ay - side, x1, ay) if gy <= ay else (x0, ay, x1, ay + side)


def _cells_in_square(
    square: Key, cells: list[list[int]], nodes: Nodes, cells_of_node: dict[int, set[int]]
) -> list[int]:
    """Cells whose nodes are all corners, midpoints or the centre of the square."""
    sx0, sy0, sx1, sy1 = square
    sxm, sym = (sx0 + sx1) // 2, (sy0 + sy1) // 2
    pattern_points = [
        (sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1), (sxm, sy0),
        (sx1, sym), (sxm, sy1), (sx0, sym), (sxm, sym),
    ]  # fmt: skip
    in_square = {v for v in (nodes.get(*p) for p in pattern_points) if v is not None}
    found: set[int] = set()
    for v in in_square:
        found.update(cells_of_node[v])
    return [c for c in found if all(v in in_square for v in cells[c])]


def _regenerate(square: Key, nodes: Nodes) -> list[list[int]]:
    """Emit a square again for the midpoints that exist on its sides now."""
    sx0, sy0, sx1, sy1 = square
    sxm, sym = (sx0 + sx1) // 2, (sy0 + sy1) // 2
    corners = (nodes.add(sx0, sy0), nodes.add(sx1, sy0), nodes.add(sx1, sy1), nodes.add(sx0, sy1))
    m_s, m_n = nodes.get(sxm, sy0), nodes.get(sxm, sy1)
    m_w, m_e = nodes.get(sx0, sym), nodes.get(sx1, sym)
    splits = sum(m is not None for m in (m_s, m_n, m_w, m_e))
    adjacent = splits == 2 and (m_s is None or m_n is None) and (m_e is None or m_w is None)
    if splits <= 1 or adjacent:
        return square_cells(corners, m_s, m_e, m_n, m_w)
    centre = nodes.add(sxm, sym)
    # Missing midpoints are created in the order S, N, W, E (node numbering).
    m_s = nodes.add(sxm, sy0) if m_s is None else m_s
    m_n = nodes.add(sxm, sy1) if m_n is None else m_n
    m_w = nodes.add(sx0, sym) if m_w is None else m_w
    m_e = nodes.add(sx1, sym) if m_e is None else m_e
    return square_cells(corners, m_s, m_e, m_n, m_w, centre)


def resolve_hanging_nodes(cells: list[list[int]], nodes: Nodes) -> None:
    """Split every cell that has a node in the middle of one of its edges.

    A quad becomes a fan from its centre over its boundary, midpoints included; a
    triangle is bisected from the hanging node to the opposite corner. Both reuse
    existing boundary nodes only, so no new hanging nodes appear; repeats until no
    cell changes (at most 16 rounds).

    Parameters
    ----------
    cells : list of list of int
        The cells; changed in place.
    nodes : Nodes
        The nodes; a fan adds its centre.
    """
    for _ in range(16):
        new: list[list[int]] = []
        changed = False
        for cell in cells:
            found = _hanging_edge(cell, nodes)
            if found is None:
                new.append(cell)
                continue
            changed = True
            j, m = found
            if len(cell) == 3:
                a, b, opp = cell[j], cell[(j + 1) % 3], cell[(j + 2) % 3]
                new.append([a, m, opp])
                new.append([m, b, opp])
            else:
                new.extend(_fan(cell, nodes))
        cells[:] = new
        if not changed:
            break


def _midpoint_node(nodes: Nodes, a: int, b: int) -> int | None:
    """Return the node in the middle of edge ``a-b``, if there is one."""
    ax, ay = nodes.points[a]
    bx, by = nodes.points[b]
    if (ax + bx) % 2 or (ay + by) % 2:
        return None
    m = nodes.get((ax + bx) // 2, (ay + by) // 2)
    return None if m is None or m in (a, b) else m


def _hanging_edge(cell: list[int], nodes: Nodes) -> tuple[int, int] | None:
    """First edge (position, midpoint node) of a cell with a node in its middle."""
    n = len(cell)
    for k in range(n):
        m = _midpoint_node(nodes, cell[k], cell[(k + 1) % n])
        if m is not None:
            return k, m
    return None


def _fan(cell: list[int], nodes: Nodes) -> list[list[int]]:
    """Triangles from the centre of a cell's bounding box over its boundary and midpoints."""
    xs = [nodes.points[v][0] for v in cell]
    ys = [nodes.points[v][1] for v in cell]
    centre = nodes.add((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)
    boundary: list[int] = []
    n = len(cell)
    for k in range(n):
        boundary.append(cell[k])
        m = _midpoint_node(nodes, cell[k], cell[(k + 1) % n])
        if m is not None:
            boundary.append(m)
    nb = len(boundary)
    return [[centre, boundary[i], boundary[(i + 1) % nb]] for i in range(nb)]


def count_hanging_nodes(cells: list[list[int]], nodes: Nodes) -> int:
    """Count the unique edges that have a node in their middle."""
    seen: set[tuple[int, int]] = set()
    hanging = 0
    for cell in cells:
        n = len(cell)
        for k in range(n):
            a, b = cell[k], cell[(k + 1) % n]
            edge = (a, b) if a < b else (b, a)
            if edge in seen:
                continue
            seen.add(edge)
            if _midpoint_node(nodes, a, b) is not None:
                hanging += 1
    return hanging
