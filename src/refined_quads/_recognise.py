"""Recognise the lattice squares (quadtree leaves) that the cells of a mesh fill."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from refined_quads.errors import InputError
from refined_quads.mesh import Lattice


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
    partial : numpy.ndarray of bool, shape (s,)
        True for a square that its cells fill only partly (cells removed after
        generation, such as clipping along water).
    """

    face_level: NDArray[np.int64]
    level: NDArray[np.int64]
    x: NDArray[np.int64]
    y: NDArray[np.int64]
    partial: NDArray[np.bool_]


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

    Cells left over are parts of squares from which cells were removed after
    generation (clipping along water removes single cells, triangles included). They
    are grouped, coarsest level first, into a square that contains them, whose
    pattern points hold all their nodes and that overlaps no square found so far;
    such a square is marked partial. Cells that fit no square stay unrecognised.
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

    n_full = sum(len(a) for a in out_level)
    level_arr = np.concatenate(out_level).astype(np.int64)
    x_arr = np.concatenate(out_x).astype(np.int64)
    y_arr = np.concatenate(out_y).astype(np.int64)
    partial = _partial_squares(
        face_level, (level_arr, x_arr, y_arr), (xs, ys, valid, x0, y0, x1, y1), levels
    )
    return LatticeSquares(
        face_level=face_level,
        level=np.concatenate([level_arr, partial[0]]).astype(np.int64),
        x=np.concatenate([x_arr, partial[1]]).astype(np.int64),
        y=np.concatenate([y_arr, partial[2]]).astype(np.int64),
        partial=np.arange(n_full + len(partial[0])) >= n_full,
    )


def _mark(
    occupied: list[set[tuple[int, int]]],
    inside: list[set[tuple[int, int]]],
    level: int,
    x: int,
    y: int,
) -> None:
    """Record a square, and record it as content of each of its ancestors."""
    occupied[level].add((x, y))
    for coarser in range(level - 1, -1, -1):
        shift = level - coarser
        inside[coarser].add((x >> shift, y >> shift))


def _partial_squares(
    face_level: NDArray[np.int64],
    squares: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]],
    cells: tuple[NDArray[Any], ...],
    levels: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Group the cells left over into partly filled squares; set their levels.

    Returns the level and position of every partial square; ``face_level`` is updated.
    """
    xs, ys, valid, x0, y0, x1, y1 = cells
    # Squares found so far, as keys per level: their own key and those of their ancestors.
    occupied: list[set[tuple[int, int]]] = [set() for _ in range(levels + 1)]
    inside: list[set[tuple[int, int]]] = [set() for _ in range(levels + 1)]

    for level, x, y in zip(*squares, strict=True):
        _mark(occupied, inside, int(level), int(x), int(y))

    out: tuple[list[int], list[int], list[int]] = ([], [], [])
    for level in range(levels + 1):
        free = np.flatnonzero(face_level < 0)
        if free.size == 0:
            break
        s = 2 ** (levels - level)
        groups: dict[tuple[int, int], list[int]] = {}
        for f in free:
            px, py = int(x0[f] // s), int(y0[f] // s)
            if x1[f] > px * s + s or y1[f] > py * s + s:
                continue
            rx = xs[f][valid[f]] - px * s
            ry = ys[f][valid[f]] - py * s
            if not (_on_pattern_point(rx, s) & _on_pattern_point(ry, s)).all():
                continue
            if (px, py) in occupied[level] or (px, py) in inside[level]:
                continue  # overlaps a square found before
            if any((px >> (level - c), py >> (level - c)) in occupied[c] for c in range(level)):
                continue  # lies inside a coarser square found before
            groups.setdefault((px, py), []).append(int(f))
        for (px, py), members in groups.items():
            face_level[members] = level
            _mark(occupied, inside, level, px, py)
            out[0].append(level)
            out[1].append(px)
            out[2].append(py)
    return (
        np.asarray(out[0], dtype=np.int64),
        np.asarray(out[1], dtype=np.int64),
        np.asarray(out[2], dtype=np.int64),
    )


def derive_lattice(nodes: NDArray[np.float64], faces: NDArray[np.int64]) -> Lattice:
    """Derive the lattice of a mesh from its cells.

    Parameters
    ----------
    nodes : numpy.ndarray of float, shape (n, 2)
        Node coordinates.
    faces : numpy.ndarray of int, shape (m, k)
        Node indices per face, padded with ``-1``.

    Returns
    -------
    Lattice
        The lattice: the smallest square cell gives the unit, the largest square
        cell the base cell and its alignment, and the origin is the lattice point at
        or below the lower-left node of the mesh.

    Raises
    ------
    InputError
        If the mesh has no axis-aligned square cells, the largest square is not the
        smallest times a power of 2, the largest squares are not aligned to one
        lattice, or a node is not a lattice point.

    Notes
    -----
    Only square cells are used: a triangle can span two lattice squares (after an
    edge flip of the orthogonality repair), so its extent and corner say nothing
    about the lattice. When no base cell is a plain square (every one is split into
    transition cells), the derived base cell is the largest square that is, and
    cells of coarser lattice squares are not recognised by :func:`check`. The base
    cell is rounded to ten significant digits, so that a cell size such as 10.1 m is
    recovered exactly from coordinates with rounding errors.
    """
    valid = faces >= 0
    used = np.unique(faces[valid])
    if used.size == 0:
        raise InputError("the mesh has no cells")
    safe = np.where(valid, faces, 0)
    xs = np.where(valid, nodes[safe, 0], np.nan)
    ys = np.where(valid, nodes[safe, 1], np.nan)
    x0, x1 = np.nanmin(xs, axis=1), np.nanmax(xs, axis=1)
    y0, y1 = np.nanmin(ys, axis=1), np.nanmax(ys, axis=1)
    width, height = x1 - x0, y1 - y0
    extent = np.maximum(width, height)
    tol = 1e-6 * extent
    on_corner = (
        np.isclose(xs, x0[:, None], rtol=0, atol=tol[:, None])
        | np.isclose(xs, x1[:, None], rtol=0, atol=tol[:, None])
    ) & (
        np.isclose(ys, y0[:, None], rtol=0, atol=tol[:, None])
        | np.isclose(ys, y1[:, None], rtol=0, atol=tol[:, None])
    )
    square = (
        (valid.sum(axis=1) == 4)
        & (np.abs(width - height) <= tol)
        & (on_corner | ~valid).all(axis=1)
    )
    if not square.any() or not (extent > 0).all():
        raise InputError(
            "the mesh has no axis-aligned square cells, so it has no lattice; meshes of "
            "rectangles or rotated cells are not supported"
        )
    sides = width[square]
    unit_estimate = float(sides.min())
    largest_side = float(sides.max())
    ratio = largest_side / unit_estimate
    levels = round(math.log2(ratio))
    if abs(ratio - 2**levels) > 1e-6 * ratio:
        raise InputError(
            f"the largest square cell ({largest_side:g}) is not the smallest one "
            f"({unit_estimate:g}) times a power of 2"
        )
    cell_size = float(f"{largest_side:.10g}")
    lattice_unit = cell_size / 2**levels
    # The largest squares are base cells: their corners lie on the base-cell lattice.
    largest = np.flatnonzero(square & (width >= largest_side * (1 - 1e-6)))
    corners = np.column_stack([x0[largest], y0[largest]])
    offsets = (corners - corners[0]) / cell_size
    if np.abs(offsets - np.rint(offsets)).max() > 1e-6:
        raise InputError(
            f"the largest square cells ({largest_side:g}) are not aligned to one lattice; "
            f"this is not a refined quad mesh"
        )
    corner = corners[0]
    low = nodes[used].min(axis=0)
    steps = np.ceil((corner - low) / cell_size - 1e-6)
    origin = corner - steps * cell_size
    lattice = Lattice((float(origin[0]), float(origin[1])), cell_size, levels)
    _, on_lattice = lattice.to_indices(nodes[used])
    if not on_lattice.all():
        bad = nodes[used[~on_lattice][0]]
        raise InputError(
            f"{int((~on_lattice).sum())} nodes are not on one lattice with unit "
            f"{lattice_unit:g} (for example ({bad[0]:.3f}, {bad[1]:.3f})); this is not a "
            f"refined quad mesh"
        )
    return lattice


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
