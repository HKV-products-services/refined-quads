"""Golden masters of the generator at qgis_dhydamo 9a324bc (ADR-0002)."""

import numpy as np
import pytest

from golden import record
from golden.masters import MASTERS_DIR, input_fingerprint, load_master, master_paths
from golden.scenarios import SCENARIOS
from refined_quads import check

NAMES = [s.name for s in SCENARIOS]


@pytest.fixture(scope="module", params=NAMES)
def master(request):
    return load_master(MASTERS_DIR / f"{request.param}.npz")


def test_every_scenario_has_exactly_one_master():
    assert [p.stem for p in master_paths()] == sorted(NAMES)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=NAMES)
def test_scenario_input_is_unchanged_since_recording(scenario):
    meta = load_master(MASTERS_DIR / f"{scenario.name}.npz").meta

    assert meta["input_fingerprint"] == input_fingerprint(scenario), (
        "scenario input changed: re-record its master (record.py --rerecord) and add a "
        "CHANGELOG entry"
    )


def test_master_is_conforming_balanced_and_orthogonal(master):
    report = check(master.to_mesh())

    assert len(report.off_lattice_nodes) == 0
    assert len(report.hanging_nodes) == 0
    assert len(report.balance_violations) == 0
    assert report.max_orthogonality <= 0.30
    assert len(report.non_orthogonal_edges) == 0
    assert len(report.unrecognised_cells) == 0
    assert sum(report.cells_per_level.values()) == report.n_cells == master.meta["n_cells"]


def unique_edges(faces):
    """Unique undirected edges and the number of faces that have each."""
    pairs = []
    for face in faces:
        nodes = face[face >= 0]
        pairs.append(np.stack([nodes, np.roll(nodes, -1)], axis=1))
    edges, count = np.unique(np.sort(np.concatenate(pairs), axis=1), axis=0, return_counts=True)
    return edges, count


def test_meshkernel_confirms_orthogonality(master):
    mk = pytest.importorskip("meshkernel")
    mesh = master.to_mesh()
    edges, face_count = unique_edges(mesh.faces)
    valid = mesh.faces >= 0
    kernel = mk.MeshKernel(projection=mk.ProjectionType.CARTESIAN)
    kernel.mesh2d_set(
        mk.Mesh2d(
            node_x=np.ascontiguousarray(mesh.nodes[:, 0]),
            node_y=np.ascontiguousarray(mesh.nodes[:, 1]),
            edge_nodes=edges.ravel().astype(np.int32),
            face_nodes=mesh.faces[valid].astype(np.int32),
            nodes_per_face=valid.sum(axis=1).astype(np.int32),
        )
    )

    values = np.asarray(kernel.mesh2d_get_orthogonality().values)
    internal = values[values != -999.0]

    assert kernel.mesh2d_get().face_x.size == len(mesh.faces)
    assert internal.size == int((face_count == 2).sum())
    assert internal.max() <= 0.30


def test_recording_refuses_to_overwrite_masters_without_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        record.main(["--qgis-dhydamo", "unused", "--only", NAMES[0]])

    assert "--rerecord" in str(exc.value.code)


def test_transition_rows_are_handed_to_the_old_generator_fine_to_coarse():
    params = record.legacy_params(32.0, 4, (1, 1, 2, 3), "inside")

    assert params["buffer_rows"] == [3, 2, 1, 1]
    assert params["max_refinement"] == 4
    assert params["water_delete_option"] == "INSIDE_NOT_INTERSECTED"


def test_levels_the_old_generator_would_cap_are_refused():
    with pytest.raises(ValueError, match="capped"):
        record.legacy_params(24.0, 5, (1, 1, 1, 1, 1), "centre")
