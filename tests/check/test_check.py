import numpy as np
import pytest

from golden.holes import closable_holes
from golden.masters import MASTERS_DIR, load_master
from golden.scenarios import SCENARIOS
from handmade import CLEAN, FINE_RIGHT, ORIGIN, mesh_from_cells
from refined_quads import Mesh, check


def test_clean_mesh_passes_every_check():
    report = check(mesh_from_cells(CLEAN))

    assert report.is_valid
    assert report.n_cells == 7
    assert report.n_nodes == 11
    assert report.max_orthogonality <= 0.30
    assert report.cells_per_level == {0: 3, 1: 4}


def test_hanging_node_is_found():
    # Plain base cell next to finer squares: their corner (4, 2) hangs on its edge.
    mesh = mesh_from_cells([[(0, 0), (4, 0), (4, 4), (0, 4)], *FINE_RIGHT])

    report = check(mesh)

    np.testing.assert_array_equal(report.hanging_nodes, [[ORIGIN[0] + 4, ORIGIN[1] + 2]])
    assert len(report.balance_violations) == 0
    assert not report.is_valid


def test_two_to_one_violation_is_found():
    # Level-0 cell (side 4) next to level-2 cells (side 1).
    fine = [[(4, y), (5, y), (5, y + 1), (4, y + 1)] for y in range(4)]
    mesh = mesh_from_cells([[(0, 0), (4, 0), (4, 4), (0, 4)], *fine], levels=2)

    report = check(mesh)

    expected = [[ORIGIN[0] + 4, ORIGIN[1] + y + 0.5] for y in range(4)]
    np.testing.assert_array_equal(np.sort(report.balance_violations, axis=0), expected)
    assert report.cells_per_level == {0: 1, 2: 4}


def test_level_difference_of_one_is_balanced():
    report = check(mesh_from_cells(CLEAN))

    assert len(report.balance_violations) == 0


def test_skewed_triangle_is_found():
    # The base cell split with the wrong diagonal: the obtuse sliver (0,0)-(4,2)-(4,4).
    skew = [[(0, 0), (4, 0), (4, 2)], [(0, 0), (4, 2), (4, 4)], [(0, 0), (4, 4), (0, 4)]]
    mesh = mesh_from_cells([*skew, *FINE_RIGHT])

    report = check(mesh)

    midpoints = {tuple(p) for p in report.non_orthogonal_edges - np.asarray(ORIGIN)}
    assert (2.0, 1.0) in midpoints  # edge (0,0)-(4,2) between the sliver and its neighbour
    assert report.max_orthogonality > 0.30
    assert not report.is_valid


def test_clockwise_cell_is_found():
    cells = [list(c) for c in CLEAN]
    cells[4] = cells[4][::-1]

    report = check(mesh_from_cells(cells))

    np.testing.assert_array_equal(report.clockwise_cells, [4])


def test_node_off_the_lattice_is_found():
    mesh = mesh_from_cells(CLEAN)
    nodes = mesh.nodes.copy()
    nodes[0] += [0.3, 0.0]

    report = check(Mesh(nodes, mesh.faces, mesh.lattice))

    assert len(report.off_lattice_nodes) == 1
    assert not report.is_valid


def test_cell_that_fills_no_lattice_square_is_unrecognised():
    # A triangle with a corner at a third of a side is no part of any pattern.
    mesh = mesh_from_cells([[(0, 0), (3, 0), (0, 3)]], levels=2)

    report = check(mesh)

    np.testing.assert_array_equal(report.unrecognised_cells, [0])
    assert report.cells_per_level == {}


def test_four_finer_transition_squares_are_not_taken_for_one_coarse_square():
    # A level-0 square tiled by four level-1 transition squares, each split towards
    # level-2 neighbours on its outer sides, has nodes at the quarter points of its
    # sides; recognising it as one level-0 square would report false 2:1 violations.
    def split_west(x, y):  # level-1 square at (x, y) with a midpoint on its west side
        return [
            [(x, y), (x + 2, y), (x, y + 1)],
            [(x, y + 1), (x + 2, y), (x + 2, y + 2)],
            [(x, y + 1), (x + 2, y + 2), (x, y + 2)],
        ]

    def split_east(x, y):  # level-1 square at (x, y) with a midpoint on its east side
        return [
            [(x, y), (x + 2, y), (x + 2, y + 1)],
            [(x, y), (x + 2, y + 1), (x, y + 2)],
            [(x + 2, y + 1), (x + 2, y + 2), (x, y + 2)],
        ]

    fine = [
        [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)] for x in (2, 3, 8, 9) for y in range(4)
    ]
    transition = [*split_west(4, 0), *split_west(4, 2), *split_east(6, 0), *split_east(6, 2)]

    report = check(mesh_from_cells([*transition, *fine], levels=2))

    assert report.cells_per_level == {1: 12, 2: 16}
    assert len(report.balance_violations) == 0
    assert report.is_valid


def test_summary_mentions_every_check():
    text = check(mesh_from_cells(CLEAN)).summary()

    for word in ("hanging", "2:1", "orthogonal", "clockwise", "unrecognised", "valid: yes"):
        assert word in text


@pytest.mark.parametrize("unit", [1.0, 0.1, 10.1])
def test_orthogonality_does_not_depend_on_scale(unit):
    mesh = mesh_from_cells(CLEAN, cell_size=4 * unit, unit_coordinates=unit)

    assert check(mesh).is_valid


def test_duplicate_node_is_found():
    # Two triangles of the base cell share a corner, but one gets its own copy of it.
    mesh = mesh_from_cells(CLEAN)
    shared = mesh.faces[2, 0]
    nodes = np.vstack([mesh.nodes, mesh.nodes[[shared]]])
    faces = mesh.faces.copy()
    faces[2, 0] = len(mesh.nodes)

    report = check(Mesh(nodes, faces, mesh.lattice))

    np.testing.assert_array_equal(report.duplicate_nodes, [mesh.nodes[shared]])
    assert "duplicate nodes (information): 1" in report.summary()


def test_edge_on_top_of_another_edge_with_other_nodes_is_a_split_edge():
    # Copy a corner used by only one cell: the cells around it touch in a point only.
    mesh = mesh_from_cells(CLEAN)
    corner = mesh.faces[3, 1]  # (6, 0), used by the two lower fine squares
    users = np.flatnonzero((mesh.faces == corner).any(axis=1))
    nodes = np.vstack([mesh.nodes, mesh.nodes[[corner]]])
    faces = mesh.faces.copy()
    faces[users[0]][faces[users[0]] == corner] = len(mesh.nodes)

    report = check(Mesh(nodes, faces, mesh.lattice))

    assert len(report.duplicate_nodes) == 1
    # The two fine squares shared the edge (6,0)-(6,2); now it is two edges on top of
    # each other with different nodes: a wall.
    np.testing.assert_array_equal(report.split_edges, [[1006.0, 2001.0]])
    assert not report.is_valid


def test_partly_filled_lattice_square_is_recognised():
    # Clipping along water removes single cells: the rest of the square is valid.
    mesh = mesh_from_cells([CLEAN[1], CLEAN[2], *FINE_RIGHT])

    report = check(mesh)

    assert report.cells_per_level == {0: 2, 1: 4}
    assert report.is_valid


@pytest.mark.parametrize("name", [s.name for s in SCENARIOS])
def test_check_reports_the_holes_meshkernel_closes(name):
    # Issue 20 / ADR-0004: compared with a full MeshKernel round trip (golden.holes).
    master = load_master(MASTERS_DIR / f"{name}.npz").to_mesh()

    report = check(master)

    expected = closable_holes(master)
    assert len(report.closable_holes) == len(expected)
    centres = {tuple(np.round(p.centroid.coords[0], 6)) for p in expected}
    assert {tuple(np.round(c, 6)) for c in report.closable_holes} == centres


def test_holes_meshkernel_closes_are_information_only():
    master = load_master(MASTERS_DIR / "many_small_water_areas.npz").to_mesh()

    report = check(master)

    assert len(report.closable_holes) == 17
    assert report.is_valid
    assert "holes MeshKernel closes (information): 17" in report.summary()
