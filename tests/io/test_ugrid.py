import dataclasses
import shutil

import netCDF4
import numpy as np
import pytest
from shapely import Point, box

from golden.compare import assert_same_cells
from golden.masters import MASTERS_DIR, load_master
from golden.scenarios import SCENARIOS
from handmade import CLEAN, generate_rd, mesh_from_cells
from refined_quads import InputError, Mesh, check, read_ugrid, write_ugrid

# CLEAN with a plain base cell to its left, so that the base cell can be derived.
WITH_BASE_CELL = [[(-4, 0), (0, 0), (0, 4), (-4, 4)], *CLEAN]
LEGACY_FILE = MASTERS_DIR.parent / "ugrid" / "domain_with_hole_legacy.nc"
PROVENANCE = {"source", "history"}  # global attributes that name the writer and the time
# The old writer wrote -999 bed levels; write_ugrid leaves them out (ADR-0003).
BED_LEVELS = {"Mesh2d_node_z", "Mesh2d_face_z"}


def rd_master(name):
    return dataclasses.replace(load_master(MASTERS_DIR / f"{name}.npz").to_mesh(), crs="EPSG:28992")


@pytest.mark.parametrize("name", [s.name for s in SCENARIOS])
def test_write_then_read_gives_the_same_mesh(name, tmp_path):
    mesh = rd_master(name)
    path = tmp_path / "mesh.nc"

    write_ugrid(mesh, path)
    back = read_ugrid(path)

    assert_same_cells(back, mesh)
    assert back.crs.to_epsg() == 28992
    assert back.lattice.cell_size == mesh.lattice.cell_size
    np.testing.assert_array_equal(back.nodes, mesh.nodes)
    np.testing.assert_array_equal(back.faces, mesh.faces)


def test_rd_file_has_the_structure_and_attributes_of_the_old_writer(tmp_path):
    path = tmp_path / "mesh.nc"
    write_ugrid(rd_master("domain_with_hole"), path)

    with netCDF4.Dataset(LEGACY_FILE) as old, netCDF4.Dataset(path) as new:
        assert old.ncattrs() == new.ncattrs()
        for name in set(old.ncattrs()) - PROVENANCE:
            assert old.getncattr(name) == new.getncattr(name), name
        assert {k: len(d) for k, d in old.dimensions.items()} == {
            k: len(d) for k, d in new.dimensions.items()
        }
        assert [v for v in old.variables if v not in BED_LEVELS] == list(new.variables)
        for name, old_var in old.variables.items():
            if name in BED_LEVELS:
                continue
            new_var = new.variables[name]
            assert (old_var.dtype, old_var.dimensions) == (new_var.dtype, new_var.dimensions), name
            assert old_var.ncattrs() == new_var.ncattrs(), name
            for attr in old_var.ncattrs():
                old_value, new_value = old_var.getncattr(attr), new_var.getncattr(attr)
                assert type(old_value) is type(new_value), (name, attr)
                np.testing.assert_array_equal(old_value, new_value, err_msg=f"{name}.{attr}")
        assert int(new["projected_coordinate_system"][...]) == 28992


def test_file_of_the_old_writer_reads_as_its_golden_master():
    mesh = read_ugrid(LEGACY_FILE)

    assert_same_cells(mesh, load_master(MASTERS_DIR / "domain_with_hole.npz").to_mesh())
    assert mesh.crs.to_epsg() == 28992


def test_no_bed_level_is_written(tmp_path):
    # A -999 node_z outlives a MeshKernel clip in hydrolib and breaks saving the model, and
    # the D-HYDRO GUI rejects a per-cell face_z once it rebuilds the cells (ADR-0003).
    path = tmp_path / "mesh.nc"
    write_ugrid(rd_master("l_shaped_domain"), path)

    with netCDF4.Dataset(path) as ds:
        assert not [v for v in ds.variables if v.lower().endswith("_z")]


def test_only_the_2d_mesh_of_a_net_file_with_a_1d_network_is_read(tmp_path):
    path = tmp_path / "model_net.nc"
    shutil.copy(LEGACY_FILE, path)
    with netCDF4.Dataset(path, "a") as ds:  # add a 1D network and mesh as D-HYDRO writes them
        ds.createDimension("network1d_nNodes", 3)
        ds.createDimension("mesh1d_nNodes", 5)
        for name, dim in (("network1d", "network1d_nNodes"), ("mesh1d", "mesh1d_nNodes")):
            topology = ds.createVariable(name, "i4", ())
            topology.setncattr("cf_role", "mesh_topology")
            topology.setncattr("topology_dimension", np.int32(1))
            topology.setncattr("node_coordinates", f"{name}_node_x {name}_node_y")
            for axis in "xy":
                var = ds.createVariable(f"{name}_node_{axis}", "f8", (dim,))
                var[:] = np.arange(len(ds.dimensions[dim]), dtype=float)

    mesh = read_ugrid(path)

    assert_same_cells(mesh, load_master(MASTERS_DIR / "domain_with_hole.npz").to_mesh())


def test_mesh_without_crs_is_written_without_crs_variable(tmp_path):
    mesh = load_master(MASTERS_DIR / "l_shaped_domain.npz").to_mesh()
    path = tmp_path / "mesh.nc"

    write_ugrid(mesh, path)

    with netCDF4.Dataset(path) as ds:
        assert "projected_coordinate_system" not in ds.variables
    assert read_ugrid(path).crs is None


def test_other_crs_gets_attributes_derived_with_pyproj(tmp_path):
    mesh = dataclasses.replace(rd_master("l_shaped_domain"), crs="EPSG:32631")
    path = tmp_path / "mesh.nc"

    write_ugrid(mesh, path)

    with netCDF4.Dataset(path) as ds, netCDF4.Dataset(LEGACY_FILE) as old:
        var = ds["projected_coordinate_system"]
        old_var = old["projected_coordinate_system"]
        assert var.ncattrs() == old_var.ncattrs()
        for name in old_var.ncattrs():
            assert type(var.getncattr(name)) is type(old_var.getncattr(name)), name
        assert int(var[...]) == var.getncattr("epsg") == 32631
        assert var.getncattr("EPSG_code") == "EPSG:32631"
        assert var.getncattr("name") == "WGS 84 / UTM zone 31N"
        assert var.getncattr("grid_mapping_name") == "transverse_mercator"
        assert var.getncattr("semi_major_axis") == 6378137.0
    assert read_ugrid(path).crs.to_epsg() == 32631


def test_geographic_crs_is_refused():
    mesh = rd_master("l_shaped_domain")

    with pytest.raises(InputError, match="geographic"):
        dataclasses.replace(mesh, crs="EPSG:4326")


def test_base_cell_that_is_not_a_binary_fraction_is_recovered_exactly(tmp_path):
    mesh = generate_rd(
        box(187_013.0, 318_457.0, 187_400.0, 318_700.0),
        cell_size=10.1,
        refinement_areas=[(Point(187_200.0, 318_580.0).buffer(30), 2)],
    )
    path = tmp_path / "mesh.nc"

    write_ugrid(mesh, path)
    back = read_ugrid(path)

    assert back.lattice.cell_size == 10.1
    assert back.lattice.levels == 2
    assert_same_cells(back, mesh)
    assert check(back).is_valid


def _write_minimal_ugrid(path, nodes, faces, start_index=0, fill=None):
    """A bare UGRID file as other tools write it (0-based, padding below start_index)."""
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("n_node", len(nodes))
        ds.createDimension("n_face", len(faces))
        ds.createDimension("n_max", faces.shape[1])
        topology = ds.createVariable("mesh", "i4", ())
        topology.setncattr("cf_role", "mesh_topology")
        topology.setncattr("topology_dimension", np.int32(2))
        topology.setncattr("node_coordinates", "node_x node_y")
        topology.setncattr("face_node_connectivity", "face_nodes")
        for i, axis in enumerate("xy"):
            ds.createVariable(f"node_{axis}", "f8", ("n_node",))[:] = nodes[:, i]
        var = ds.createVariable("face_nodes", "i4", ("n_face", "n_max"), fill_value=fill)
        var.setncattr("start_index", np.int32(start_index))
        var[:] = np.where(faces >= 0, faces + start_index, -1 if fill is None else fill)


@pytest.mark.parametrize(
    ("start_index", "fill"), [(0, None), (1, None), (0, -999), (1, -2147483647)]
)
def test_zero_and_one_based_files_with_any_padding_are_read(tmp_path, start_index, fill):
    mesh = mesh_from_cells(WITH_BASE_CELL)
    path = tmp_path / "mesh.nc"
    _write_minimal_ugrid(path, mesh.nodes, mesh.faces, start_index, fill)

    back = read_ugrid(path)

    np.testing.assert_array_equal(back.faces, mesh.faces)
    assert back.crs is None
    assert_same_cells(back, mesh)


def test_file_without_2d_mesh_raises_input_error(tmp_path):
    path = tmp_path / "empty.nc"
    netCDF4.Dataset(path, "w").close()

    with pytest.raises(InputError, match="no 2D mesh topology"):
        read_ugrid(path)


def test_mesh_whose_nodes_are_not_on_one_lattice_raises_input_error(tmp_path):
    mesh = mesh_from_cells(WITH_BASE_CELL)
    nodes = mesh.nodes.copy()
    nodes[mesh.faces[5, 2]] += [0.3, 0.0]  # a corner of one fine square off the lattice
    path = tmp_path / "mesh.nc"
    _write_minimal_ugrid(path, nodes, mesh.faces)

    with pytest.raises(InputError, match="not a refined quad mesh"):
        read_ugrid(path)


def test_read_mesh_is_a_mesh(tmp_path):
    path = tmp_path / "mesh.nc"
    write_ugrid(rd_master("cell_size_25"), path)

    assert isinstance(read_ugrid(path), Mesh)


def test_without_a_plain_base_cell_the_largest_square_becomes_the_base_cell(tmp_path):
    # Documented limitation: the only base cell is split into transition cells, so the
    # derived lattice has the 2 m squares as base cell and does not recognise the
    # transition cells of the 4 m square.
    mesh = mesh_from_cells(CLEAN)
    path = tmp_path / "mesh.nc"
    _write_minimal_ugrid(path, mesh.nodes, mesh.faces)

    back = read_ugrid(path)

    assert (back.lattice.cell_size, back.lattice.levels) == (2.0, 0)
    assert len(check(back).unrecognised_cells) == 3
