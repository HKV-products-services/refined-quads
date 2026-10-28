"""A repair that does not close: the error carries the raw mesh, the CLI dumps it.

The repair is made to fail through its own failure path: the real repair function,
run with no passes and a threshold every edge exceeds. There is no small input that
makes the repair fail on its own, so this patches the internal function (the testing
rules allow internal names when a failure cannot be reached otherwise).
"""

import numpy as np
import pyogrio
import pytest
from shapely import Point, box

import refined_quads._engine as engine
from golden.compare import assert_same_cells
from handmade import generate_rd, write_layer
from refined_quads import Mesh, OrthogonalityRepairError, read_ugrid
from refined_quads.cli import main

X0, Y0 = 187_013.0, 318_457.0
DOMAIN = box(X0, Y0, X0 + 200, Y0 + 150)
AREA = Point(X0 + 100, Y0 + 75).buffer(20)


@pytest.fixture
def failing_repair(monkeypatch):
    repair = engine.repair_orthogonality

    def fail(cells, nodes, origin, unit, **kwargs):
        return repair(cells, nodes, origin, unit, max_passes=0, max_fm_ortho=-1.0)

    monkeypatch.setattr(engine, "repair_orthogonality", fail)


def test_error_carries_the_raw_mesh_and_the_problem_points(failing_repair):
    with pytest.raises(OrthogonalityRepairError, match="did not close") as caught:
        generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)])

    exc = caught.value
    assert isinstance(exc.mesh, Mesh)
    assert exc.mesh.crs.to_epsg() == 28992
    assert exc.mesh.lattice.cell_size == 25.0
    assert len(exc.bad_points) > 0
    x, y, orth = exc.bad_points[0]
    assert DOMAIN.buffer(25).contains(Point(x, y))
    assert orth is None or orth >= 0.0


def test_raw_mesh_is_the_mesh_before_the_repair(failing_repair, monkeypatch):
    with pytest.raises(OrthogonalityRepairError) as caught:
        generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)])
    monkeypatch.undo()

    repaired = generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)])

    # This input needs no repair, so the raw mesh equals the repaired one.
    assert_same_cells(caught.value.mesh, repaired)


@pytest.fixture
def run_dir(tmp_path):
    write_layer(tmp_path / "model.gpkg", "domain", [DOMAIN])
    write_layer(tmp_path / "model.gpkg", "refine", [AREA], fields={"level": np.array([2])})
    (tmp_path / "run.toml").write_text(
        'cell_size = 25.0\n[domain]\npath = "model.gpkg"\nlayer = "domain"\n'
        '[[refinement_areas]]\npath = "model.gpkg"\nlayer = "refine"\nlevel = "level"\n',
        encoding="utf-8",
    )
    return tmp_path


def test_cli_writes_raw_mesh_and_problem_points_and_exits_1(failing_repair, run_dir, capsys):
    code = main(["generate", str(run_dir / "run.toml"), "-o", str(run_dir / "mesh_net.nc")])

    assert code == 1
    assert "did not close" in capsys.readouterr().err
    assert not (run_dir / "mesh_net.nc").exists()
    raw = read_ugrid(run_dir / "mesh_net_raw.nc")
    assert raw.crs.to_epsg() == 28992
    info = pyogrio.read_info(run_dir / "mesh_net_problems.gpkg")
    assert info["geometry_type"] == "Point"
    assert info["features"] > 0
    assert list(info["fields"]) == ["orth_e"]
    assert "28992" in info["crs"]


def test_failing_dump_does_not_mask_the_repair_error(failing_repair, run_dir, capsys, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("refined_quads.cli.write_ugrid", broken)

    code = main(["generate", str(run_dir / "run.toml"), "-o", str(run_dir / "mesh_net.nc")])

    err = capsys.readouterr().err
    assert code == 1
    assert "warning: could not write" in err
    assert "disk full" in err
    assert "failed: orthogonality repair did not close" in err
    assert (run_dir / "mesh_net_problems.gpkg").exists()
