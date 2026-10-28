import numpy as np
import pytest
import shapely
from shapely import LineString, Polygon, box

from golden.compare import assert_same_cells
from golden.scenarios import SCENARIOS_BY_NAME
from handmade import generate_rd
from refined_quads import InputError, check

WATER_SCENARIOS = [
    "water_removal_centre",
    "water_removal_inside",
    "water_removal_touches",
    "many_small_water_areas",
]


def lattice_squares(mesh, report):
    """Unique lattice squares (as boxes) that the cells of the mesh lie in."""
    lattice = mesh.lattice
    indices, _ = lattice.to_indices(mesh.nodes)
    keys = set()
    for face, level in zip(mesh.faces, report.cell_levels, strict=True):
        assert level >= 0
        side = 2 ** (lattice.levels - int(level))
        corner = indices[face[face >= 0]].min(axis=0) // side * side
        keys.add((int(corner[0]), int(corner[1]), side))
    x0, y0, side = np.array(sorted(keys)).T
    ox, oy = lattice.origin
    unit = lattice.unit
    return shapely.box(
        ox + x0 * unit, oy + y0 * unit, ox + (x0 + side) * unit, oy + (y0 + side) * unit
    )


@pytest.mark.parametrize("name", WATER_SCENARIOS)
def test_no_cell_breaks_the_water_removal_rule(name):
    scenario = SCENARIOS_BY_NAME[name]
    mesh = generate_rd(
        scenario.domain,
        cell_size=scenario.cell_size,
        refinement_areas=scenario.refinement_areas,
        transition_rows=scenario.transition_rows,
        water_areas=scenario.water_areas,
        water_removal=scenario.water_removal,
    )
    report = check(mesh)
    squares = lattice_squares(mesh, report)

    for water in scenario.water_areas:
        if scenario.water_removal == "centre":
            assert not shapely.contains(water, shapely.centroid(squares)).any()
        elif scenario.water_removal == "inside":
            assert not shapely.contains(water, squares).any()
        else:
            assert not shapely.intersects(water, squares).any()
    assert report.is_valid, report.summary()


def test_rules_remove_increasingly_many_cells():
    # inside removes least, centre more, touches most.
    counts = {
        rule: len(check(_water_mesh(rule)).cell_levels) for rule in ("inside", "centre", "touches")
    }

    assert counts["inside"] > counts["centre"] > counts["touches"]


def _water_mesh(rule):
    scenario = SCENARIOS_BY_NAME["water_removal_centre"]
    return generate_rd(
        scenario.domain,
        cell_size=scenario.cell_size,
        refinement_areas=scenario.refinement_areas,
        transition_rows=scenario.transition_rows,
        water_areas=scenario.water_areas,
        water_removal=rule,
    )


def test_many_small_water_areas_leave_no_hanging_nodes():
    scenario = SCENARIOS_BY_NAME["many_small_water_areas"]
    mesh = generate_rd(
        scenario.domain,
        cell_size=scenario.cell_size,
        refinement_areas=scenario.refinement_areas,
        transition_rows=scenario.transition_rows,
        water_areas=scenario.water_areas,
    )

    assert len(check(mesh).hanging_nodes) == 0


def test_default_rule_is_centre():
    scenario = SCENARIOS_BY_NAME["water_removal_centre"]
    with_default = generate_rd(
        scenario.domain,
        cell_size=scenario.cell_size,
        refinement_areas=scenario.refinement_areas,
        transition_rows=scenario.transition_rows,
        water_areas=scenario.water_areas,
    )

    assert len(with_default.faces) == len(_water_mesh("centre").faces)


@pytest.mark.parametrize(
    "rule", ["middelpunt", "CENTRE", "FACES_WITH_INCLUDED_CIRCUMCENTERS", None]
)
def test_unknown_rule_raises_input_error_with_allowed_values(rule):
    with pytest.raises(InputError, match="one of 'centre', 'inside', 'touches'"):
        generate_rd(
            box(0, 0, 100, 100), cell_size=25.0, water_areas=[box(0, 0, 10, 10)], water_removal=rule
        )


@pytest.mark.parametrize("water", [Polygon(), LineString([(0, 0), (50, 50)]), "water.gpkg"])
def test_invalid_water_area_raises_input_error(water):
    with pytest.raises(InputError, match="water area 0"):
        generate_rd(box(0, 0, 100, 100), cell_size=25.0, water_areas=[water])


def test_water_covering_the_whole_domain_raises_input_error():
    with pytest.raises(InputError, match="no cells are left"):
        generate_rd(box(0, 0, 100, 100), cell_size=25.0, water_areas=[box(-10, -10, 110, 110)])


@pytest.mark.parametrize("rule", ["centre", "inside", "touches"])
def test_holes_in_the_domain_are_treated_as_water_areas(rule):
    scenario = SCENARIOS_BY_NAME["domain_with_hole"]
    hole = shapely.Polygon(scenario.domain.interiors[0])
    mesh = generate_rd(
        scenario.domain,
        cell_size=scenario.cell_size,
        refinement_areas=scenario.refinement_areas,
        transition_rows=scenario.transition_rows,
        water_removal=rule,
    )
    as_water = generate_rd(
        shapely.Polygon(scenario.domain.exterior),
        cell_size=scenario.cell_size,
        refinement_areas=scenario.refinement_areas,
        transition_rows=scenario.transition_rows,
        water_areas=[hole],
        water_removal=rule,
    )
    squares = lattice_squares(mesh, check(mesh))

    assert_same_cells(mesh, as_water)
    if rule == "centre":
        assert not shapely.contains(hole, shapely.centroid(squares)).any()
    elif rule == "inside":
        assert not shapely.contains(hole, squares).any()
    else:
        assert not shapely.intersects(hole, squares).any()


def test_island_inside_a_hole_keeps_its_cells():
    outer = box(0, 0, 400, 400).difference(box(100, 100, 300, 300))
    island = box(160, 160, 240, 240)  # on the 20 m lattice: 4 x 4 cells
    mesh = generate_rd(shapely.union_all([outer, island]), cell_size=20.0)

    centres = shapely.centroid(
        np.array([shapely.Polygon(mesh.nodes[f[f >= 0]]) for f in mesh.faces], dtype=object)
    )
    assert shapely.contains(island, centres).sum() == 16
    assert not shapely.contains(box(101, 101, 159, 299), centres).any()  # the hole stays empty
