import numpy as np
import pytest
import shapely
from shapely import Point, box

from golden.compare import assert_same_cells
from handmade import write_layer
from refined_quads import InputError, MeshSpec, OrthogonalityRepairError, check, read_ugrid
from refined_quads.cli import main

X0, Y0 = 187_013.0, 318_457.0
DOMAIN_PARTS = [box(X0, Y0, X0 + 300, Y0 + 400), box(X0 + 300, Y0, X0 + 600, Y0 + 400)]
AREAS = [Point(X0 + 200, Y0 + 200).buffer(60), Point(X0 + 450, Y0 + 120).buffer(35)]
WATER = [Point(X0 + 420, Y0 + 300).buffer(25)]

SPEC = """\
cell_size = 25.0
transition_rows = [1, 2]
water_removal = "centre"

[domain]
path = "model.gpkg"
layer = "domain"

[[refinement_areas]]
path = "model.gpkg"
layer = "refine"
level = "level"

[[water_areas]]
path = "model.gpkg"
layer = "water"
"""


@pytest.fixture
def run_dir(tmp_path):
    gpkg = tmp_path / "model.gpkg"
    write_layer(gpkg, "domain", DOMAIN_PARTS)
    write_layer(gpkg, "refine", AREAS, fields={"level": np.array([2, 1], dtype=np.int32)})
    write_layer(gpkg, "water", WATER)
    (tmp_path / "run.toml").write_text(SPEC, encoding="utf-8")
    return tmp_path


def test_generate_writes_a_mesh_that_read_ugrid_and_check_approve(run_dir, capsys):
    out = run_dir / "mesh_net.nc"

    code = main(["generate", str(run_dir / "run.toml"), "-o", str(out)])

    assert code == 0
    mesh = read_ugrid(out)
    assert check(mesh).is_valid
    assert mesh.crs.to_epsg() == 28992
    assert f"wrote {out}" in capsys.readouterr().out


def test_mesh_spec_from_toml_gives_the_same_mesh_as_the_cli(run_dir):
    out = run_dir / "mesh_net.nc"
    main(["generate", str(run_dir / "run.toml"), "-o", str(out)])

    mesh = MeshSpec.from_toml(run_dir / "run.toml").generate()

    assert_same_cells(read_ugrid(out), mesh)


def test_option_overrides_the_toml(run_dir):
    out = run_dir / "mesh_net.nc"

    main(["generate", str(run_dir / "run.toml"), "-o", str(out), "--cell-size", "20"])

    assert read_ugrid(out).lattice.cell_size == 20.0


def test_transition_rows_option_overrides_the_toml(run_dir):
    plain, wide = run_dir / "plain.nc", run_dir / "wide.nc"
    spec = str(run_dir / "run.toml")

    main(["generate", spec, "-o", str(plain), "--transition-rows", "1"])
    main(["generate", spec, "-o", str(wide), "--transition-rows", "3", "3"])

    assert len(read_ugrid(wide).faces) > len(read_ugrid(plain).faces)


def test_output_from_the_toml_is_relative_to_the_toml(run_dir):
    (run_dir / "run.toml").write_text('output = "out/mesh_net.nc"\n' + SPEC, encoding="utf-8")
    (run_dir / "out").mkdir()

    assert main(["generate", str(run_dir / "run.toml")]) == 0
    assert (run_dir / "out" / "mesh_net.nc").exists()


def test_missing_output_is_an_input_error(run_dir, capsys):
    assert main(["generate", str(run_dir / "run.toml")]) == 2
    assert "no output file" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("cell_size = 25.0", "cell_size = -1"), "cell_size: must be a positive length"),
        (("cell_size = 25.0", "cellsize = 25.0"), "cellsize: unknown key"),
        (('level = "level"', "level = 0"), r"refinement_areas\[0\]\.level: must be a column name"),
        (('level = "level"', 'level = "niveau"'), "no column 'niveau'"),
        (
            ("transition_rows = [1, 2]", "transition_rows = [1, -1]"),
            r"transition_rows\[1\]: must be",
        ),
        (
            ('water_removal = "centre"', 'water_removal = "middelpunt"'),
            "water_removal: must be one of",
        ),
        (('layer = "domain"', 'layer = "domain"\nbuffer = 3'), "domain.buffer: unknown key"),
        (("[domain]", "[domain"), "not valid TOML"),
    ],
)
def test_invalid_toml_exits_2_with_key_and_reason(run_dir, capsys, change, message):
    old, new = change
    (run_dir / "run.toml").write_text(SPEC.replace(old, new), encoding="utf-8")

    code = main(["generate", str(run_dir / "run.toml"), "-o", str(run_dir / "mesh.nc")])

    assert code == 2
    err = capsys.readouterr().err
    assert "run.toml" in err
    assert pytest.importorskip("re").search(message, err), err


def test_missing_toml_file_exits_2(tmp_path, capsys):
    assert main(["generate", str(tmp_path / "missing.toml"), "-o", "x.nc"]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_failed_run_exits_1(run_dir, capsys, monkeypatch):
    def fail(self, progress=None):
        raise OrthogonalityRepairError("orthogonality repair did not close", [(1.0, 2.0, 0.5)])

    monkeypatch.setattr(MeshSpec, "generate", fail)

    code = main(["generate", str(run_dir / "run.toml"), "-o", str(run_dir / "mesh.nc")])

    assert code == 1
    assert "did not close" in capsys.readouterr().err


def test_info_shows_lattice_levels_cells_and_checks(run_dir, capsys):
    out = run_dir / "mesh_net.nc"
    main(["generate", str(run_dir / "run.toml"), "-o", str(out)])
    capsys.readouterr()

    code = main(["info", str(out)])

    text = capsys.readouterr().out
    assert code == 0
    for expected in ("crs: EPSG:28992", "lattice: origin", "base cell 25", "2 levels", "cells:",
                     "cells per level: 0 (25)", "hanging nodes: 0", "valid: yes"):  # fmt: skip
        assert expected in text, expected


def test_info_on_a_file_that_is_not_ugrid_exits_2(tmp_path, capsys):
    import netCDF4

    path = tmp_path / "empty.nc"
    netCDF4.Dataset(path, "w").close()

    assert main(["info", str(path)]) == 2
    assert "no 2D mesh topology" in capsys.readouterr().err


def test_spec_values_are_validated_when_replaced(run_dir):
    spec = MeshSpec.from_toml(run_dir / "run.toml")

    with pytest.raises(InputError, match="cell_size: must be a positive length"):
        spec.replace(cell_size=0)
    with pytest.raises(InputError, match="cannot replace"):
        spec.replace(domain=None)


def test_fixed_level_and_a_single_table_for_water(run_dir):
    toml = SPEC.replace('level = "level"', "level = 2").replace("[[water_areas]]", "[water_areas]")
    (run_dir / "run.toml").write_text(toml, encoding="utf-8")

    spec = MeshSpec.from_toml(run_dir / "run.toml")

    assert spec.refinement_areas[0].level == 2
    assert len(spec.water_areas) == 1
    assert spec.domain.path == run_dir / "model.gpkg"
    mesh = spec.generate()
    assert max(check(mesh).cells_per_level) == 2
    assert not shapely.contains(WATER[0], shapely.points(mesh.nodes)).all()


def test_origin_from_the_toml_and_the_option(run_dir):
    (run_dir / "run.toml").write_text("origin = [3.125, 12.5]\n" + SPEC, encoding="utf-8")
    from_toml, from_option = run_dir / "a.nc", run_dir / "b.nc"

    main(["generate", str(run_dir / "run.toml"), "-o", str(from_toml)])
    main(["generate", str(run_dir / "run.toml"), "-o", str(from_option), "--origin", "0", "0"])

    assert MeshSpec.from_toml(run_dir / "run.toml").origin == (3.125, 12.5)
    assert read_ugrid(from_toml).nodes[:, 0].min() % 25.0 == pytest.approx(3.125)
    assert read_ugrid(from_option).nodes[:, 0].min() % 25.0 == pytest.approx(0.0, abs=1e-6)


def test_invalid_origin_in_the_toml_exits_2(run_dir, capsys):
    (run_dir / "run.toml").write_text("origin = [1]\n" + SPEC, encoding="utf-8")

    assert main(["generate", str(run_dir / "run.toml"), "-o", str(run_dir / "x.nc")]) == 2
    assert "origin: must be two finite numbers" in capsys.readouterr().err


def test_verbose_prints_progress_to_stderr(run_dir, capsys):
    main(["generate", str(run_dir / "run.toml"), "-o", str(run_dir / "m.nc"), "-v"])

    err = capsys.readouterr().err
    assert "Quad Grid: base grid" in err
    assert "100% Quad Grid: done" in err


def test_toml_accepts_zero_transition_rows(run_dir):
    (run_dir / "run.toml").write_text(
        SPEC.replace("transition_rows = [1, 2]", "transition_rows = 0"), encoding="utf-8"
    )

    spec = MeshSpec.from_toml(run_dir / "run.toml")

    assert spec.transition_rows == 0
    assert check(spec.generate()).is_valid
