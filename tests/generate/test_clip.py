"""Cells MeshKernel would merge after the water clip keep their own cells (issue 20, ADR-0004)."""

import numpy as np

from golden.holes import closable_holes
from refined_quads._generate.clip import _keep_closable_holes
from refined_quads.mesh import Lattice, Mesh

# A 4 x 3 grid of 10 m squares; nodes numbered row by row (5 per row, 4 rows).
VERTICES = np.array([(x * 10.0, y * 10.0) for y in range(4) for x in range(5)])


def square(i, j):
    """The square in column i, row j, counterclockwise."""
    a = j * 5 + i
    return [a, a + 1, a + 6, a + 5]


GRID = [square(i, j) for j in range(3) for i in range(4)]
MIDDLE = [square(1, 1), square(2, 1)]  # two squares a small water area deletes
REST = [c for c in GRID if c not in MIDDLE]


def as_set(cells):
    return {tuple(sorted(c)) for c in cells}


def test_a_hole_meshkernel_would_close_gets_its_cells_back():
    # The two middle squares leave a ring of six nodes that MeshKernel makes a cell of.
    mesh = Mesh(VERTICES, _padded(REST), Lattice((0.0, 0.0), 10.0, 0))
    assert len(closable_holes(mesh)) == 1

    result = _keep_closable_holes(VERTICES, GRID, [list(c) for c in REST])

    assert as_set(result) == as_set(GRID)


def test_a_cell_meshkernel_merged_is_replaced_by_its_cells():
    # MeshKernel can also return the two deleted squares as one hexagon (Geul, 2026-11-12).
    hexagon = [6, 7, 8, 13, 12, 11]
    result = _keep_closable_holes(VERTICES, GRID, [list(c) for c in REST] + [hexagon])

    assert as_set(result) == as_set(GRID)


def test_a_hole_of_a_large_water_area_stays_open():
    # Six middle squares of a 6 x 4 grid: a ring of more than six nodes stays a hole.
    vertices = np.array([(x * 10.0, y * 10.0) for y in range(5) for x in range(7)])
    grid = [[j * 7 + i, j * 7 + i + 1, j * 7 + i + 8, j * 7 + i + 7]
            for j in range(4) for i in range(6)]  # fmt: skip
    water = [c for c in grid if 1 <= c[0] % 7 <= 3 and 1 <= c[0] // 7 <= 2]
    rest = [c for c in grid if c not in water]

    result = _keep_closable_holes(vertices, grid, [list(c) for c in rest])

    assert as_set(result) == as_set(rest)


def _padded(cells):
    return np.array(cells, dtype=np.int64)
