"""Compare meshes on their cells as integer lattice indices (ADR-0002).

Two meshes are the same when they contain the same set of cells, every node taken as
an integer lattice index normalised to the finest level of both meshes. Cell order,
node numbering, the node a cell starts with and its orientation are free; orientation
is checked separately by ``check``.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from refined_quads import Mesh


def _common_unit_and_origin(a: Mesh, b: Mesh):
    la, lb = a.lattice, b.lattice
    if not math.isclose(la.cell_size, lb.cell_size, rel_tol=1e-12):
        raise AssertionError(f"different base cells: {la.cell_size} and {lb.cell_size}")
    levels = max(la.levels, lb.levels)
    unit = la.cell_size / 2**levels
    shift = (np.asarray(lb.origin) - np.asarray(la.origin)) / unit
    if not np.allclose(shift, np.rint(shift), rtol=0, atol=1e-6):
        raise AssertionError(f"lattices do not coincide: origins {la.origin} and {lb.origin}")
    return levels, la.origin


def lattice_cells(mesh: Mesh, levels: int | None = None, origin=None) -> Counter:
    """Multiset of cells as canonical tuples of lattice indices at ``levels`` from ``origin``.

    Each cell is made counterclockwise and rotated to start at its smallest node.
    Raises AssertionError if a node is not a lattice point.
    """
    lattice = mesh.lattice
    levels = lattice.levels if levels is None else levels
    origin = lattice.origin if origin is None else origin
    unit = lattice.cell_size / 2**levels
    scaled = (mesh.nodes - np.asarray(origin)) / unit
    idx = np.rint(scaled).astype(np.int64)
    off = np.abs(scaled - idx).max(axis=1) > 1e-6
    used = np.unique(mesh.faces[mesh.faces >= 0])
    if off[used].any():
        raise AssertionError(f"{int(off[used].sum())} nodes are not lattice points")
    cells = Counter()
    for face in mesh.faces:
        pts = [tuple(int(v) for v in idx[n]) for n in face if n >= 0]
        cells[_canonical(pts)] += 1
    return cells


def _canonical(pts):
    area2 = sum(
        pts[k][0] * pts[(k + 1) % len(pts)][1] - pts[(k + 1) % len(pts)][0] * pts[k][1]
        for k in range(len(pts))
    )
    if area2 < 0:
        pts = pts[::-1]
    start = min(range(len(pts)), key=pts.__getitem__)
    return tuple(pts[start:] + pts[:start])


def assert_same_cells(actual: Mesh, expected: Mesh, max_examples: int = 5) -> None:
    """Assert that two meshes have exactly the same cells (order-independent)."""
    levels, origin = _common_unit_and_origin(actual, expected)
    a = lattice_cells(actual, levels, origin)
    b = lattice_cells(expected, levels, origin)
    if a == b:
        return
    unit = actual.lattice.cell_size / 2**levels

    def describe(cells):
        examples = sorted(cells.elements())[:max_examples]
        return "; ".join(
            "[" + ", ".join(f"({origin[0] + x * unit:.3f}, {origin[1] + y * unit:.3f})"
                            for x, y in cell) + "]"
            for cell in examples
        )  # fmt: skip

    only_a, only_b = a - b, b - a
    raise AssertionError(
        f"meshes differ: {only_a.total()} cells only in actual, {only_b.total()} only in expected "
        f"(of {a.total()} and {b.total()}).\n"
        f"only in actual: {describe(only_a)}\nonly in expected: {describe(only_b)}"
    )
