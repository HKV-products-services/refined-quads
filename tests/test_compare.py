import numpy as np
import pytest

from golden.compare import assert_same_cells
from handmade import CLEAN, FINE_RIGHT, mesh_from_cells
from refined_quads import Lattice, Mesh


def renumbered(mesh, seed=1):
    """Same cells: shuffled cells and nodes, other start node, reversed orientation."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(mesh.nodes))
    new_index = np.empty_like(perm)
    new_index[perm] = np.arange(len(perm))
    faces = []
    for k, face in enumerate(mesh.faces[rng.permutation(len(mesh.faces))]):
        nodes = [int(new_index[n]) for n in face if n >= 0]
        nodes = nodes[k % len(nodes) :] + nodes[: k % len(nodes)]
        if k % 2:
            nodes = nodes[::-1]
        faces.append(nodes + [-1] * (mesh.faces.shape[1] - len(nodes)))
    return Mesh(mesh.nodes[perm], faces, mesh.lattice)


def test_order_of_cells_and_nodes_does_not_matter():
    mesh = mesh_from_cells(CLEAN)

    assert_same_cells(renumbered(mesh), mesh)


def test_a_different_cell_is_reported():
    split = [[(0, 0), (4, 0), (4, 4), (0, 4)], *FINE_RIGHT]

    with pytest.raises(AssertionError, match="1 cells only in actual, 3 only in expected"):
        assert_same_cells(mesh_from_cells(split), mesh_from_cells(CLEAN))


def test_a_duplicated_cell_is_reported():
    with pytest.raises(AssertionError, match="1 cells only in actual"):
        assert_same_cells(mesh_from_cells([*CLEAN, CLEAN[0]]), mesh_from_cells(CLEAN))


def test_meshes_are_compared_at_the_finest_level_of_both():
    mesh = mesh_from_cells(CLEAN)
    finer_lattice = Lattice(mesh.lattice.origin, mesh.lattice.cell_size, mesh.lattice.levels + 2)

    assert_same_cells(Mesh(mesh.nodes, mesh.faces, finer_lattice), mesh)


def test_origin_may_move_by_whole_base_cells():
    mesh = mesh_from_cells(CLEAN)
    moved = Lattice((mesh.lattice.origin[0] - 40.0, mesh.lattice.origin[1] + 8.0), 4.0, 1)

    assert_same_cells(Mesh(mesh.nodes, mesh.faces, moved), mesh)


def test_lattices_that_do_not_coincide_are_refused():
    mesh = mesh_from_cells(CLEAN)
    shifted = Lattice((mesh.lattice.origin[0] + 0.5, mesh.lattice.origin[1]), 4.0, 1)

    with pytest.raises(AssertionError, match="do not coincide"):
        assert_same_cells(Mesh(mesh.nodes + np.array([0.5, 0.0]), mesh.faces, shifted), mesh)


def test_comparison_is_exact_on_lattice_points():
    mesh = mesh_from_cells(CLEAN)
    nodes = mesh.nodes.copy()
    nodes[3] += [0.25, 0.0]  # a quarter unit off: not a lattice point

    with pytest.raises(AssertionError, match="not lattice points"):
        assert_same_cells(Mesh(nodes, mesh.faces, mesh.lattice), mesh)
