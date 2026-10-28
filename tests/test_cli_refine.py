import numpy as np
import pyogrio
import pytest
from shapely import Point, box

from golden.compare import assert_same_cells
from golden.scenarios import SCENARIOS_BY_NAME, X0, Y0
from handmade import RD, generate_rd, write_layer
from refined_quads import (
    Lattice,
    Mesh,
    RefineSpec,
    check,
    read_refinement_areas,
    read_ugrid,
    refine,
    write_ugrid,
)
from refined_quads.cli import main

BASE = SCENARIOS_BY_NAME["cell_size_32"]
EXTRA = [Point(X0 + 450, Y0 + 300).buffer(40)]

SPEC = """\
transition_rows = [1, 1, 1]

[[refinement_areas]]
path = "extra.gpkg"
layer = "refine"
level = 3
"""


@pytest.fixture
def run_dir(tmp_path):
    mesh = generate_rd(
        BASE.domain,
        cell_size=BASE.cell_size,
        refinement_areas=BASE.refinement_areas,
        transition_rows=BASE.transition_rows,
    )
    write_ugrid(mesh, tmp_path / "mesh_net.nc")
    write_layer(tmp_path / "extra.gpkg", "refine", EXTRA)
    (tmp_path / "extra.toml").write_text(SPEC, encoding="utf-8")
    return tmp_path


def run(run_dir, *extra):
    return main(
        ["refine", str(run_dir / "mesh_net.nc"), str(run_dir / "extra.toml"),
         "-o", str(run_dir / "mesh_v2_net.nc"), *extra]
    )  # fmt: skip


def test_refine_writes_a_mesh_that_check_approves(run_dir, capsys):
    assert run(run_dir) == 0

    refined = read_ugrid(run_dir / "mesh_v2_net.nc")
    assert check(refined).is_valid
    assert refined.crs.to_epsg() == 28992
    out = capsys.readouterr().out
    assert "unchanged" in out
    assert "valid: yes" in out


def test_cli_equals_refine_from_python(run_dir):
    run(run_dir)

    mesh = read_ugrid(run_dir / "mesh_net.nc")
    areas = read_refinement_areas(run_dir / "extra.gpkg", "refine", level=3)
    expected = refine(mesh, areas, transition_rows=[1, 1, 1]).mesh
    assert_same_cells(read_ugrid(run_dir / "mesh_v2_net.nc"), expected)
    from_spec = RefineSpec.from_toml(run_dir / "extra.toml").refine(mesh).mesh
    assert_same_cells(from_spec, expected)


def test_transition_rows_option_overrides_the_toml(run_dir):
    run(run_dir)
    narrow = len(read_ugrid(run_dir / "mesh_v2_net.nc").faces)

    run(run_dir, "--transition-rows", "3")

    assert len(read_ugrid(run_dir / "mesh_v2_net.nc").faces) > narrow


def test_unrecognised_mesh_exits_2_and_writes_its_problem_locations(run_dir, capsys):
    # Two cells that fit no lattice square (a corner at 3/4 of a side), next to a base
    # cell of 32 m and a cell of 8 m that give the lattice.
    lattice = Lattice((X0, Y0), 32.0, 2)
    nodes = np.array(
        [[X0, Y0], [X0 + 32, Y0], [X0 + 32, Y0 + 32], [X0, Y0 + 32], [X0 + 24, Y0],
         [X0 + 64, Y0], [X0 + 64, Y0 + 32], [X0 + 72, Y0], [X0 + 72, Y0 + 8], [X0 + 64, Y0 + 8]]
    )  # fmt: skip
    faces = [[0, 4, 3, -1], [4, 1, 2, 3], [1, 5, 6, 2], [5, 7, 8, 9]]
    bad = Mesh(nodes, faces, lattice, crs=RD)
    write_ugrid(bad, run_dir / "mesh_net.nc")

    code = run(run_dir)

    assert code == 2
    assert "neither a lattice square" in capsys.readouterr().err
    path = run_dir / "mesh_v2_net_recognition.gpkg"
    info = pyogrio.read_info(path)
    assert info["features"] >= 1
    assert "28992" in info["crs"]
    assert not (run_dir / "mesh_v2_net.nc").exists()


def test_spec_without_refinement_areas_exits_2(run_dir, capsys):
    (run_dir / "extra.toml").write_text("transition_rows = 1\n", encoding="utf-8")

    assert run(run_dir) == 2
    assert "refinement_areas: missing" in capsys.readouterr().err


def test_missing_output_exits_2(run_dir, capsys):
    code = main(["refine", str(run_dir / "mesh_net.nc"), str(run_dir / "extra.toml")])

    assert code == 2
    assert "no output file" in capsys.readouterr().err


def test_domain_and_water_in_the_spec_clip_only_new_cells(run_dir):
    write_layer(run_dir / "extra.gpkg", "water", [box(X0 + 440, Y0 + 290, X0 + 460, Y0 + 310)])
    (run_dir / "extra.toml").write_text(
        SPEC + '\n[[water_areas]]\npath = "extra.gpkg"\nlayer = "water"\n', encoding="utf-8"
    )

    assert run(run_dir) == 0

    refined = read_ugrid(run_dir / "mesh_v2_net.nc")
    without = refine(
        read_ugrid(run_dir / "mesh_net.nc"),
        read_refinement_areas(run_dir / "extra.gpkg", "refine", level=3),
        transition_rows=[1, 1, 1],
    ).mesh
    assert len(refined.faces) < len(without.faces)
    assert check(refined).is_valid
