"""Refine meshes that this package did not make (issue 14)."""

import numpy as np
import pytest
import shapely
from shapely import Point, box
from shapely.affinity import rotate

from golden.compare import assert_same_cells
from handmade import RD, generate_rd
from refined_quads import InputError, Lattice, Mesh, check, read_ugrid, refine, write_ugrid

OX, OY = 187_013.3, 318_457.7  # an origin that is no multiple of the cell size


def mesh_of(cells, cell_size, levels, crs=RD):
    """A mesh from cells given as lists of (x, y) coordinates; lattice through OX, OY."""
    points = sorted({p for cell in cells for p in cell})
    number = {p: i for i, p in enumerate(points)}
    width = max(len(c) for c in cells)
    faces = [[number[p] for p in c] + [-1] * (width - len(c)) for c in cells]
    return Mesh(np.asarray(points), faces, Lattice((OX, OY), cell_size, levels), crs=crs)


def through_ugrid(mesh, tmp_path):
    """As another program's file: written, read back, lattice derived from the cells."""
    write_ugrid(mesh, tmp_path / "foreign_net.nc")
    return read_ugrid(tmp_path / "foreign_net.nc")


def square(x, y, s):
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]


def cell_set(mesh):
    return {frozenset(map(tuple, mesh.nodes[f[f >= 0]])) for f in mesh.faces}


def test_rectangular_grid_with_any_origin_is_refined(tmp_path):
    size = 10.1
    cells = [
        [(OX + i * size, OY + j * size), (OX + (i + 1) * size, OY + j * size),
         (OX + (i + 1) * size, OY + (j + 1) * size), (OX + i * size, OY + (j + 1) * size)]
        for i in range(12) for j in range(9)
    ]  # fmt: skip
    grid = through_ugrid(mesh_of(cells, size, 0), tmp_path)
    area = [(Point(OX + 60, OY + 45).buffer(15), 2)]

    result = refine(grid, area, crs=RD)

    assert check(result.mesh).is_valid
    assert result.mesh.lattice.levels == 2
    expected = generate_rd(
        box(OX, OY, OX + 12 * size, OY + 9 * size),
        cell_size=size,
        refinement_areas=area,
        origin=(OX, OY),
    )
    assert_same_cells(result.mesh, expected)
    rm = result.replacement_map
    for new in np.flatnonzero(rm.unchanged):
        old = grid.faces[rm.source[new]]
        new_face = result.mesh.faces[new]
        assert set(map(tuple, result.mesh.nodes[new_face[new_face >= 0]])) == set(
            map(tuple, grid.nodes[old[old >= 0]])
        )


def _pentagon_mesh():
    """Base cells of 8 m (10 x 4); a 2 x 2 block of base cells split in four. The base cells next
    to the block are pentagons (a square with the midpoint of the side facing the block),
    a transition other programs make instead of triangles."""
    s, h = 8.0, 4.0
    fine = {(2, 1), (3, 1), (2, 2), (3, 2)}
    cells = []
    for i in range(10):
        for j in range(4):
            x, y = OX + i * s, OY + j * s
            if (i, j) in fine:
                cells += [square(x + a * h, y + b * h, h) for a in (0, 1) for b in (0, 1)]
                continue
            corners = square(x, y, s)
            mids = []
            if (i + 1, j) in fine:
                mids.append((1, (x + s, y + h)))  # east side, after bottom-right corner
            if (i - 1, j) in fine:
                mids.append((3, (x, y + h)))  # west side, after top-left corner
            if (i, j + 1) in fine:
                mids.append((2, (x + h, y + s)))  # north side, after top-right corner
            if (i, j - 1) in fine:
                mids.append((0, (x + h, y)))  # south side, after bottom-left corner
            cell = []
            for k, corner in enumerate(corners):
                cell.append(corner)
                cell += [m for side, m in mids if side == k]
            cells.append(cell)
    return mesh_of(cells, s, 1)


def test_mesh_with_pentagon_transitions_is_recognised(tmp_path):
    mesh = through_ugrid(_pentagon_mesh(), tmp_path)

    report = check(mesh)

    assert 5 in set(mesh.face_sizes.tolist())
    assert len(report.unrecognised_cells) == 0
    assert report.cells_per_level == {0: 36, 1: 16}
    # Pentagons are not FM-orthogonal by this measure (their centroid is off-centre);
    # check reports that, refine leaves it alone where it does not change cells.
    assert len(report.non_orthogonal_edges) > 0


def test_untouched_pentagons_stay_exactly_and_touched_ones_get_our_patterns(tmp_path):
    mesh = through_ugrid(_pentagon_mesh(), tmp_path)
    far = [(box(OX + 73, OY + 1, OX + 79, OY + 7), 1)]  # in base cell (9, 0)
    near = [(box(OX + 33, OY + 1, OX + 39, OY + 7), 1)]  # in base cell (4, 0)

    far_result = refine(mesh, far, crs=RD)
    near_result = refine(mesh, near, crs=RD)

    old_pentagons = {c for c in cell_set(mesh) if len(c) == 5}
    assert old_pentagons <= cell_set(far_result.mesh)
    near_cells = cell_set(near_result.mesh)
    assert len(old_pentagons - near_cells) >= 1  # the pentagon west of base cell (4, 0)
    for result in (far_result, near_result):
        report = check(result.mesh)
        assert len(report.hanging_nodes) == 0
        assert len(report.unrecognised_cells) == 0
        assert len(report.balance_violations) == 0


def test_mesh_of_rectangles_raises_a_clear_error(tmp_path):
    cells = [
        [(OX + i * 10, OY + j * 5), (OX + (i + 1) * 10, OY + j * 5),
         (OX + (i + 1) * 10, OY + (j + 1) * 5), (OX + i * 10, OY + (j + 1) * 5)]
        for i in range(4) for j in range(4)
    ]  # fmt: skip
    mesh = mesh_of(cells, 10.0, 0)
    write_ugrid(mesh, tmp_path / "rectangles.nc")

    with pytest.raises(InputError, match="rectangles or rotated cells are not supported"):
        read_ugrid(tmp_path / "rectangles.nc")


def test_rotated_grid_raises_a_clear_error(tmp_path):
    squares = [
        shapely.Polygon(square(OX + i * 10, OY + j * 10, 10)) for i in range(4) for j in range(4)
    ]
    rotated = [rotate(sq, 30, origin=(OX, OY)) for sq in squares]
    cells = [[tuple(round(v, 9) for v in p) for p in sq.exterior.coords[:-1]] for sq in rotated]
    mesh = mesh_of(cells, 10.0, 0, crs=None)
    write_ugrid(mesh, tmp_path / "rotated.nc")

    with pytest.raises(InputError, match="rectangles or rotated cells are not supported"):
        read_ugrid(tmp_path / "rotated.nc")
