import math

import pytest
from shapely import LineString, MultiPolygon, Point, Polygon, box

from golden.compare import assert_same_cells
from golden.holes import assert_master_with_holes_filled, closable_holes
from golden.masters import MASTERS_DIR, load_master
from golden.scenarios import SCENARIOS
from handmade import generate_rd
from refined_quads import InputError, check

DOMAIN = box(0, 0, 400, 300)
AREA = Point(200, 150).buffer(40)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_generate_gives_the_cells_of_the_golden_master(scenario):
    mesh = generate_rd(
        scenario.domain,
        cell_size=scenario.cell_size,
        refinement_areas=scenario.refinement_areas,
        transition_rows=scenario.transition_rows,
        water_areas=scenario.water_areas,
        water_removal=scenario.water_removal,
    )

    master = load_master(MASTERS_DIR / f"{scenario.name}.npz").to_mesh()
    if closable_holes(master):
        # Issue 20 / ADR-0004: holes MeshKernel would close keep their cells.
        assert_master_with_holes_filled(mesh, master)
    else:
        assert_same_cells(mesh, master)
    report = check(mesh)
    assert report.is_valid, report.summary()
    assert mesh.lattice.levels == scenario.levels


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_meshkernel_finds_exactly_the_cells_of_the_mesh(scenario):
    # hydrolib and the D-HYDRO GUI rebuild the cells from nodes and edges with
    # MeshKernel; it must not find a cell the mesh does not have (issue 20).
    mesh = generate_rd(
        scenario.domain,
        cell_size=scenario.cell_size,
        refinement_areas=scenario.refinement_areas,
        transition_rows=scenario.transition_rows,
        water_areas=scenario.water_areas,
        water_removal=scenario.water_removal,
    )

    assert closable_holes(mesh) == []


def test_level_five_with_24_m_gives_cells_of_0_75_m():
    # The old generator capped this at four levels (smallest cell >= 1 m).
    mesh = generate_rd(
        box(0, 0, 96, 96), cell_size=24.0, refinement_areas=[(box(40, 40, 50, 50), 5)]
    )

    report = check(mesh)
    assert mesh.lattice.unit == 0.75
    assert max(report.cells_per_level) == 5
    assert report.is_valid, report.summary()


def test_single_transition_row_value_applies_to_every_step():
    one = generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)], transition_rows=2)
    per_step = generate_rd(
        DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)], transition_rows=[2, 2]
    )

    assert_same_cells(one, per_step)


def test_without_refinement_areas_the_mesh_is_the_base_grid():
    mesh = generate_rd(box(10, 10, 110, 60), cell_size=25.0)

    report = check(mesh)
    assert mesh.lattice.levels == 0
    assert mesh.lattice.origin == (0.0, 0.0)
    assert report.cells_per_level == {0: 5 * 3}
    assert report.is_valid


def test_level_given_as_integral_float_is_accepted():
    as_float = generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2.0)])
    as_int = generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)])

    assert_same_cells(as_float, as_int)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (0, "at least 1"),
        ([1, 0], "got 0 for the step from level 1 to 2"),
        ([0, 1], "got 0 for the step from level 0 to 1"),
        ([1, 1, 1], "so 2 values, got 3"),
        ([1], "so 2 values, got 1"),
        ([1, 1.5], "at least 1"),
        ("1", "at least 1"),
    ],
)
def test_invalid_transition_rows_raise_input_error(rows, message):
    with pytest.raises(InputError, match=message):
        generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)], transition_rows=rows)


@pytest.mark.parametrize("level", [0, -1, 2.5, "2", None, True, math.nan])
def test_invalid_level_raises_input_error(level):
    with pytest.raises(InputError, match="refinement area 0: the level must be an integer"):
        generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, level)])


@pytest.mark.parametrize("area", [(Polygon(), 1), (LineString([(0, 0), (1, 1)]), 1), (AREA,), AREA])
def test_invalid_refinement_area_raises_input_error(area):
    with pytest.raises(InputError, match="refinement area 0"):
        generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[area])


def test_bowtie_refinement_area_refines_both_loops_with_a_warning(caplog):
    bowtie = Polygon([(100, 100), (300, 200), (300, 100), (100, 200)])
    loops = MultiPolygon(
        [
            Polygon([(100, 100), (200, 150), (100, 200)]),
            Polygon([(200, 150), (300, 200), (300, 100)]),
        ]
    )

    mesh = generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(bowtie, 2)])

    assert_same_cells(mesh, generate_rd(DOMAIN, cell_size=25.0, refinement_areas=[(loops, 2)]))
    assert "refinement area 0 is not valid" in caplog.text


@pytest.mark.parametrize(
    ("domain", "message"),
    [
        (Polygon(), "domain is empty"),
        (LineString([(0, 0), (1, 1)]), "got LineString"),
        ("domain.gpkg", "got str"),
    ],
)
def test_invalid_domain_raises_input_error(domain, message):
    with pytest.raises(InputError, match=message):
        generate_rd(domain, cell_size=25.0)


@pytest.mark.parametrize("cell_size", [0, -25.0, math.nan, math.inf, "25", True])
def test_invalid_cell_size_raises_input_error(cell_size):
    with pytest.raises(InputError, match="cell size"):
        generate_rd(DOMAIN, cell_size=cell_size)
