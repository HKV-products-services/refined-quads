import math

import pytest
from shapely import LineString, Point, Polygon, box

from golden.compare import assert_same_cells
from golden.masters import MASTERS_DIR, load_master
from golden.scenarios import SCENARIOS
from refined_quads import InputError, check, generate

DOMAIN = box(0, 0, 400, 300)
AREA = Point(200, 150).buffer(40)


def _scenario_params():
    skip_water = pytest.mark.skip(reason="water areas are exposed in generate by issue 05")
    return [
        pytest.param(s, id=s.name, marks=[skip_water] if s.water_areas else []) for s in SCENARIOS
    ]


@pytest.mark.parametrize("scenario", _scenario_params())
def test_generate_gives_the_cells_of_the_golden_master(scenario):
    mesh = generate(
        scenario.domain,
        cell_size=scenario.cell_size,
        refinement_areas=scenario.refinement_areas,
        transition_rows=scenario.transition_rows,
    )

    assert_same_cells(mesh, load_master(MASTERS_DIR / f"{scenario.name}.npz").to_mesh())
    report = check(mesh)
    assert report.is_valid, report.summary()
    assert mesh.lattice.levels == scenario.levels


def test_level_five_with_24_m_gives_cells_of_0_75_m():
    # The old generator capped this at four levels (smallest cell >= 1 m).
    mesh = generate(box(0, 0, 96, 96), cell_size=24.0, refinement_areas=[(box(40, 40, 50, 50), 5)])

    report = check(mesh)
    assert mesh.lattice.unit == 0.75
    assert max(report.cells_per_level) == 5
    assert report.is_valid, report.summary()


def test_single_transition_row_value_applies_to_every_step():
    one = generate(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)], transition_rows=2)
    per_step = generate(
        DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)], transition_rows=[2, 2]
    )

    assert_same_cells(one, per_step)


def test_without_refinement_areas_the_mesh_is_the_base_grid():
    mesh = generate(box(10, 10, 110, 60), cell_size=25.0)

    report = check(mesh)
    assert mesh.lattice.levels == 0
    assert mesh.lattice.origin == (0.0, 0.0)
    assert report.cells_per_level == {0: 5 * 3}
    assert report.is_valid


def test_level_given_as_integral_float_is_accepted():
    as_float = generate(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2.0)])
    as_int = generate(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)])

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
        generate(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, 2)], transition_rows=rows)


@pytest.mark.parametrize("level", [0, -1, 2.5, "2", None, True, math.nan])
def test_invalid_level_raises_input_error(level):
    with pytest.raises(InputError, match="refinement area 0: the level must be an integer"):
        generate(DOMAIN, cell_size=25.0, refinement_areas=[(AREA, level)])


@pytest.mark.parametrize("area", [(Polygon(), 1), (LineString([(0, 0), (1, 1)]), 1), (AREA,), AREA])
def test_invalid_refinement_area_raises_input_error(area):
    with pytest.raises(InputError, match="refinement area 0"):
        generate(DOMAIN, cell_size=25.0, refinement_areas=[area])


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
        generate(domain, cell_size=25.0)


@pytest.mark.parametrize("cell_size", [0, -25.0, math.nan, math.inf, "25", True])
def test_invalid_cell_size_raises_input_error(cell_size):
    with pytest.raises(InputError, match="cell size"):
        generate(DOMAIN, cell_size=cell_size)
