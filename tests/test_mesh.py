import numpy as np
import pytest

from handmade import CLEAN, mesh_from_cells
from refined_quads import InputError, Lattice, Mesh


def test_lattice_unit_is_base_cell_halved_per_level():
    assert Lattice((0, 0), 25.0, 3).unit == 3.125


@pytest.mark.parametrize(
    ("origin", "cell_size", "levels"),
    [((0, 0), 0.0, 1), ((0, 0), -4.0, 1), ((0, 0), float("nan"), 1), ((0, 0), 4.0, -1),
     ((0, 0), 4.0, 1.5), ((float("inf"), 0), 4.0, 1)],
)  # fmt: skip
def test_invalid_lattice_raises_input_error(origin, cell_size, levels):
    with pytest.raises(InputError):
        Lattice(origin, cell_size, levels)


def test_coordinates_and_indices_round_trip():
    lattice = Lattice((187_013.0, 318_457.0), 10.1, 3)
    indices = np.array([[0, 0], [3, 5], [-8, 17]])

    back, on_lattice = lattice.to_indices(lattice.to_coordinates(indices))

    np.testing.assert_array_equal(back, indices)
    assert on_lattice.all()


def test_point_between_lattice_points_is_off_the_lattice():
    lattice = Lattice((0, 0), 4.0, 2)

    _, on_lattice = lattice.to_indices([[0.5, 0.0], [1.0 + 1e-9, 2.0]])

    np.testing.assert_array_equal(on_lattice, [False, True])


def test_mesh_arrays_are_read_only_copies():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mesh = Mesh(nodes, [[0, 1, 2]], Lattice((0, 0), 1.0, 0))
    nodes[0, 0] = 5.0

    assert mesh.nodes[0, 0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        mesh.nodes[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        mesh.faces[0, 0] = 1


def test_face_sizes_count_nodes_without_padding():
    mesh = mesh_from_cells(CLEAN)

    np.testing.assert_array_equal(mesh.face_sizes, [3, 3, 3, 4, 4, 4, 4])


@pytest.mark.parametrize(
    ("nodes", "faces"),
    [
        ([[0, 0, 0]], [[0, 0, 0]]),  # nodes not (n, 2)
        ([[0, 0], [1, 0], [np.nan, 1]], [[0, 1, 2]]),  # not finite
        ([[0, 0], [1, 0], [0, 1]], [[0, 1]]),  # fewer than three columns
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, 3]]),  # missing node
        ([[0, 0], [1, 0], [0, 1]], [[0, -1, 1, 2]]),  # padding before a node
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, -1, -1]]),  # fewer than three nodes
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, -2]]),  # invalid index
    ],
)
def test_invalid_mesh_raises_input_error(nodes, faces):
    with pytest.raises(InputError):
        Mesh(np.asarray(nodes, dtype=float), faces, Lattice((0, 0), 1.0, 0))
