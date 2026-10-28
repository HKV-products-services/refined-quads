"""Small handmade meshes, written as cells of lattice points."""

import numpy as np

from refined_quads import Lattice, Mesh

ORIGIN = (1000.0, 2000.0)


def mesh_from_cells(cells, *, cell_size=4.0, levels=1, unit_coordinates=1.0):
    """Build a mesh from cells given as lists of (x, y) points relative to ORIGIN.

    Points are multiplied by ``unit_coordinates``; nodes are numbered in sorted order.
    """
    lattice = Lattice(ORIGIN, cell_size, levels)
    points = sorted({p for cell in cells for p in cell})
    number = {p: i for i, p in enumerate(points)}
    width = max(len(cell) for cell in cells)
    faces = [[number[p] for p in cell] + [-1] * (width - len(cell)) for cell in cells]
    nodes = np.asarray(ORIGIN) + np.asarray(points, dtype=float) * unit_coordinates
    return Mesh(nodes, faces, lattice)


# Base cell [0, 4]^2 next to four level-1 squares on [4, 8] x [0, 4]; the base cell is
# split by the transition pattern for one finer side (midpoint (4, 2)).
FINE_RIGHT = [
    [(4, 0), (6, 0), (6, 2), (4, 2)],
    [(6, 0), (8, 0), (8, 2), (6, 2)],
    [(4, 2), (6, 2), (6, 4), (4, 4)],
    [(6, 2), (8, 2), (8, 4), (6, 4)],
]
CLEAN = [[(0, 0), (4, 0), (4, 2)], [(0, 0), (4, 2), (0, 4)], [(4, 2), (4, 4), (0, 4)], *FINE_RIGHT]
