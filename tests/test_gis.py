import sys

import numpy as np
import pytest
import shapely
from pyproj import CRS
from shapely import Point, box

from golden.compare import assert_same_cells
from handmade import RD, generate_rd, write_layer
from refined_quads import (
    InputError,
    check,
    generate,
    read_domain,
    read_refinement_areas,
    read_water_areas,
)

X0, Y0 = 187_013.0, 318_457.0
DOMAIN_PARTS = [box(X0, Y0, X0 + 300, Y0 + 400), box(X0 + 300, Y0, X0 + 600, Y0 + 400)]
DOMAIN = shapely.union_all(DOMAIN_PARTS)
AREAS = [Point(X0 + 200, Y0 + 200).buffer(60), Point(X0 + 450, Y0 + 120).buffer(35)]
LEVELS = [2, 1]
WATER = [Point(X0 + 420, Y0 + 300).buffer(25)]


@pytest.fixture
def gpkg(tmp_path):
    path = tmp_path / "input.gpkg"
    write_layer(path, "domain", DOMAIN_PARTS)
    write_layer(path, "refine", AREAS, fields={"level": np.array(LEVELS, dtype=np.int32)})
    write_layer(path, "water", WATER)
    return path


def reference_mesh():
    return generate_rd(
        DOMAIN,
        cell_size=25.0,
        refinement_areas=list(zip(AREAS, LEVELS, strict=True)),
        water_areas=WATER,
    )


def test_layers_of_one_geopackage_give_the_same_mesh_as_shapely_input(gpkg):
    mesh = generate(
        read_domain(gpkg, "domain"),
        cell_size=25.0,
        refinement_areas=read_refinement_areas(gpkg, "refine", level="level"),
        water_areas=read_water_areas(gpkg, "water"),
    )

    assert_same_cells(mesh, reference_mesh())
    assert mesh.crs.to_epsg() == 28992
    assert check(mesh).is_valid


def test_domain_features_are_merged(gpkg):
    domain = read_domain(gpkg, "domain")

    assert len(domain.geometries) == 1
    assert domain.geometries[0].equals(DOMAIN)


def test_layer_in_another_crs_is_transformed_to_the_crs_of_the_domain(gpkg, tmp_path):
    utm = tmp_path / "utm.gpkg"
    write_layer(utm, "refine", AREAS, crs="EPSG:32631", fields={"level": np.array(LEVELS)})

    areas = read_refinement_areas(utm, level="level")
    mesh = generate(read_domain(gpkg, "domain"), cell_size=25.0, refinement_areas=areas)

    assert areas.crs.to_epsg() == 32631
    assert mesh.crs.to_epsg() == 28992
    expected = generate_rd(
        DOMAIN, cell_size=25.0, refinement_areas=list(zip(AREAS, LEVELS, strict=True))
    )
    assert_same_cells(mesh, expected)


def test_fixed_level_for_the_whole_layer(gpkg):
    areas = read_refinement_areas(gpkg, "refine", level=3)

    assert areas.levels == (3, 3)


def test_layer_name_is_required_when_the_file_has_several_layers(gpkg):
    with pytest.raises(InputError, match=r"layers \['domain', 'refine', 'water'\]; name the layer"):
        read_domain(gpkg)


def test_unknown_layer_raises_input_error(gpkg):
    with pytest.raises(InputError, match="no layer 'gebied'"):
        read_domain(gpkg, "gebied")


def test_single_layer_file_needs_no_layer_name(tmp_path):
    path = tmp_path / "domain.gpkg"
    write_layer(path, "anything", DOMAIN_PARTS)

    assert read_domain(path).geometries[0].equals(DOMAIN)


@pytest.mark.filterwarnings("ignore:'crs' was not provided:UserWarning")
def test_layer_without_crs_raises_input_error_unless_crs_is_given(tmp_path):
    path = tmp_path / "no_crs.gpkg"
    write_layer(path, "domain", DOMAIN_PARTS, crs=None)

    with pytest.raises(InputError, match="has no CRS; pass crs="):
        read_domain(path)
    assert read_domain(path, crs=RD).crs.to_epsg() == 28992


def test_crs_that_contradicts_the_layer_raises_input_error(gpkg):
    with pytest.raises(InputError, match="has CRS EPSG:28992, but crs=EPSG:32631"):
        read_domain(gpkg, "domain", crs="EPSG:32631")


def test_geometry_without_crs_raises_input_error():
    with pytest.raises(InputError, match="the domain has no CRS"):
        generate(DOMAIN, cell_size=25.0)


def test_refinement_pairs_without_crs_raise_input_error(gpkg):
    with pytest.raises(InputError, match="the refinement areas have no CRS"):
        generate(
            read_domain(gpkg, "domain"),
            cell_size=25.0,
            refinement_areas=list(zip(AREAS, LEVELS, strict=True)),
        )


def test_geographic_crs_of_the_domain_raises_input_error(tmp_path):
    path = tmp_path / "wgs84.gpkg"
    write_layer(path, "domain", DOMAIN_PARTS, crs="EPSG:4326")

    with pytest.raises(InputError, match="geographic"):
        generate(read_domain(path), cell_size=25.0)


def test_missing_level_column_raises_input_error(gpkg):
    with pytest.raises(InputError, match=r"no column 'niveau'; columns are \['level'\]"):
        read_refinement_areas(gpkg, "refine", level="niveau")


@pytest.mark.parametrize("bad", [0.0, 2.5, np.nan, -1.0])
def test_invalid_level_value_names_the_feature(tmp_path, bad):
    path = tmp_path / "refine.gpkg"
    write_layer(path, "refine", AREAS, fields={"level": np.array([2.0, bad])})

    with pytest.raises(InputError, match="feature 2"):
        read_refinement_areas(path, level="level")


def test_layer_that_is_not_polygons_raises_input_error(tmp_path):
    path = tmp_path / "points.gpkg"
    write_layer(path, "points", [Point(X0, Y0), Point(X0 + 1, Y0)])

    with pytest.raises(InputError, match="feature 1 is a Point, not a polygon"):
        read_water_areas(path)


def test_missing_file_raises_input_error(tmp_path):
    with pytest.raises(InputError, match="cannot read"):
        read_domain(tmp_path / "missing.gpkg")


class FakeGeoDataFrame:
    """Just the attributes of a GeoDataFrame that refined-quads uses."""

    def __init__(self, geometries, crs, **columns):
        self.geometry = list(geometries)
        self.crs = CRS(crs) if crs else None
        self.index = list(range(10, 10 + len(self.geometry)))
        self._columns = columns
        self.columns = ["geometry", *columns]

    def __getitem__(self, name):
        return self._columns[name]


def test_geodataframe_input_works_without_geopandas():
    mesh = generate(
        FakeGeoDataFrame(DOMAIN_PARTS, RD),
        cell_size=25.0,
        refinement_areas=read_refinement_areas(
            FakeGeoDataFrame(AREAS, RD, level=LEVELS), level="level"
        ),
        water_areas=FakeGeoDataFrame(WATER, RD),
    )

    assert_same_cells(mesh, reference_mesh())
    assert "geopandas" not in sys.modules


def test_geodataframe_without_crs_raises_input_error():
    with pytest.raises(InputError, match="domain has no CRS"):
        generate(FakeGeoDataFrame(DOMAIN_PARTS, None), cell_size=25.0)


def test_geodataframe_feature_without_geometry_is_named():
    with pytest.raises(InputError, match="feature 11 has no geometry"):
        read_water_areas(FakeGeoDataFrame([WATER[0], None], RD))


def test_geodataframe_as_refinement_areas_needs_its_levels():
    with pytest.raises(InputError, match="read_refinement_areas"):
        generate_rd(DOMAIN, cell_size=25.0, refinement_areas=FakeGeoDataFrame(AREAS, RD))
