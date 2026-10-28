"""Recognise the lattice squares (quadtree leaves) that the cells of a mesh fill."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LatticeSquares:
    """The lattice squares of a mesh and the level of every cell.

    Attributes
    ----------
    face_level : numpy.ndarray of int, shape (m,)
        Refinement level of the lattice square each cell lies in; ``-1`` for
        unrecognised cells.
    level, x, y : numpy.ndarray of int, shape (s,)
        Each lattice square: its level and position, in units of its own side,
        counted from the lattice origin.
    """

    face_level: NDArray[np.int64]
    level: NDArray[np.int64]
    x: NDArray[np.int64]
    y: NDArray[np.int64]


def lattice_squares(
    indices: NDArray[np.int64], faces: NDArray[np.int64], levels: int
) -> LatticeSquares:
    """Assign every cell to the lattice square it fills, alone or with other cells.

    Parameters
    ----------
    indices : numpy.ndarray of int, shape (n, 2)
        Lattice index of every node, in units of the finest level.
    faces : numpy.ndarray of int, shape (m, k)
        Node indices per face, padded with ``-1``.
    levels : int
        Highest refinement level of the lattice.

    Returns
    -------
    LatticeSquares
        The lattice squares and the level of each cell.

    Notes
    -----
    A square cell is its own lattice square when it is axis-aligned, its side is the
    side of a refinement level and it is aligned to that level. The other cells are
    transition cells: they are grouped, coarsest level first, into the lattice square
    that contains them, and a group is accepted when it fills the square exactly and
    its nodes are only corners, side midpoints or the centre of the square, the nodes
    the transition patterns use. Checking the nodes keeps a square that is tiled by
    four finer transition squares from being taken for one coarse transition square,
    because the finer squares have nodes at the quarter points of its sides.
    """
    m = len(faces)
    valid = faces >= 0
    sizes = valid.sum(axis=1)
    safe = np.where(valid, faces, 0)
    xs = np.where(valid, indices[safe, 0], 0)
    ys = np.where(valid, indices[safe, 1], 0)
    big = np.iinfo(np.int64).max
    x0 = np.where(valid, xs, big).min(axis=1)
    y0 = np.where(valid, ys, big).min(axis=1)
    x1 = np.where(valid, xs, -big).max(axis=1)
    y1 = np.where(valid, ys, -big).max(axis=1)
    area2 = np.abs(_twice_signed_area(xs, ys, sizes))

    face_level = np.full(m, -1, dtype=np.int64)
    side = x1 - x0
    is_square = (sizes == 4) & (side == y1 - y0) & (side > 0) & (area2 == 2 * side * side)
    for level in range(levels + 1):
        s = 2 ** (levels - level)
        hit = is_square & (side == s) & (x0 % s == 0) & (y0 % s == 0)
        face_level[hit] = level
    out_level = [face_level[face_level >= 0]]
    out_x = [x0[face_level >= 0] // 2 ** (levels - face_level[face_level >= 0])]
    out_y = [y0[face_level >= 0] // 2 ** (levels - face_level[face_level >= 0])]

    for level in range(levels):
        s = 2 ** (levels - level)
        free = np.flatnonzero(face_level < 0)
        if free.size == 0:
            break
        px, py = x0[free] // s, y0[free] // s
        fits = (x1[free] <= px * s + s) & (y1[free] <= py * s + s)
        rx = xs[free] - (px * s)[:, None]
        ry = ys[free] - (py * s)[:, None]
        allowed = _on_pattern_point(rx, s) & _on_pattern_point(ry, s)
        nodes_ok = (allowed | ~valid[free]).all(axis=1)
        free, px, py, nodes_ok = free[fits], px[fits], py[fits], nodes_ok[fits]
        if free.size == 0:
            continue
        keys = np.stack([px, py], axis=1)
        groups, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        group_area = np.bincount(inverse, weights=area2[free], minlength=len(groups))
        bad_nodes = (~nodes_ok).astype(np.float64)
        group_ok = np.bincount(inverse, weights=bad_nodes, minlength=len(groups)) == 0
        accept = group_ok & (group_area == 2 * s * s)
        face_level[free[accept[inverse]]] = level
        out_level.append(np.full(int(accept.sum()), level, dtype=np.int64))
        out_x.append(groups[accept, 0])
        out_y.append(groups[accept, 1])

    return LatticeSquares(
        face_level=face_level,
        level=np.concatenate(out_level).astype(np.int64),
        x=np.concatenate(out_x).astype(np.int64),
        y=np.concatenate(out_y).astype(np.int64),
    )


def twice_signed_areas(indices: NDArray[np.int64], faces: NDArray[np.int64]) -> NDArray[np.int64]:
    """Twice the signed area of every face in lattice units; positive is counterclockwise.

    Parameters
    ----------
    indices : numpy.ndarray of int, shape (n, 2)
        Lattice index of every node.
    faces : numpy.ndarray of int, shape (m, k)
        Node indices per face, padded with ``-1``.

    Returns
    -------
    numpy.ndarray of int, shape (m,)
        Exact twice-signed areas.
    """
    valid = faces >= 0
    safe = np.where(valid, faces, 0)
    xs = np.where(valid, indices[safe, 0], 0)
    ys = np.where(valid, indices[safe, 1], 0)
    return _twice_signed_area(xs, ys, valid.sum(axis=1))


def _twice_signed_area(
    xs: NDArray[np.int64], ys: NDArray[np.int64], sizes: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Shoelace formula on padded coordinate rows (padding entries are ignored)."""
    m, k = xs.shape
    cols = np.arange(k)
    nxt = np.where(cols[None, :] + 1 < sizes[:, None], cols[None, :] + 1, 0)
    rows = np.arange(m)[:, None]
    term = xs * ys[rows, nxt] - xs[rows, nxt] * ys
    return np.asarray(np.where(cols[None, :] < sizes[:, None], term, 0).sum(axis=1), dtype=np.int64)


def _on_pattern_point(r: NDArray[np.int64], s: int) -> NDArray[np.bool_]:
    """Tell where an offset within a square of side ``s`` is 0, ``s / 2`` or ``s``."""
    return np.asarray((r == 0) | (r == s) | (2 * r == s))
