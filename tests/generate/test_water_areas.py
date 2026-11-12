import numpy as np
import pytest
import shapely
from shapely import LineString, Polygon, box

from golden.compare import assert_same_cells
from golden.scenarios import SCENARIOS_BY_NAME
from handmade import generate_rd
from refined_quads import InputError, check

# The rule holds cell by cell in these scenarios. MeshKernel, which does the clipping,
# keeps a single cell whose edges all stay (it derives cells from edges again); in
# "many_small_water_areas" that leaves some cells in small ponds, which is accepted.
WATER_SCENARIOS = ["water_removal_centre", "water_removal_inside", "water_removal_touches"]


def cells(mesh):
    """Cells as polygons and their MeshKernel centres (circumcentre of a triangle)."""
    polygons, centres = [], []
    for face in mesh.faces:
        pts = mesh.nodes[face[face >= 0]]
        polygons.append(shapely.Polygon(pts))
        if len(pts) == 3:
            (ax, ay), (bx, by), (cx, cy) = pts
            d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
            centres.append(
                ((a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d,
                 (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d)
            )  # fmt: skip
        else:
            centres.append(tuple(pts.mean(axis=0)))
    return np.array(polygons, dtype=object), shapely.points(np.array(centres))


def assert_rule_holds(mesh, water_areas, rule):
    polygons, centres = cells(mesh)
    for water in water_areas:
        if rule == "centre":
            assert not shapely.contains(water, centres).any()
        elif rule == "inside":
            assert not shapely.contains(water, polygons).any()
        else:
            assert not shapely.intersects(water, polygons).any()


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

    assert_rule_holds(mesh, scenario.water_areas, scenario.water_removal)
    assert check(mesh).is_valid, check(mesh).summary()


def test_water_is_clipped_cell_by_cell():
    # With the centre rule single triangles of a transition square can go while the
    # rest of the square stays: the mesh follows the water's edge.
    mesh = _water_mesh("centre")
    report = check(mesh)

    assert report.is_valid
    assert len(report.unrecognised_cells) == 0
    partial = check(mesh).cell_levels  # every remaining cell still has a lattice square
    assert (partial >= 0).all()


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

    assert_same_cells(mesh, as_water)
    assert_rule_holds(mesh, [hole], rule)


def test_island_inside_a_hole_keeps_its_cells():
    outer = box(0, 0, 400, 400).difference(box(100, 100, 300, 300))
    island = box(160, 160, 240, 240)  # on the 20 m lattice: 4 x 4 cells
    mesh = generate_rd(shapely.union_all([outer, island]), cell_size=20.0)

    centres = shapely.centroid(
        np.array([shapely.Polygon(mesh.nodes[f[f >= 0]]) for f in mesh.faces], dtype=object)
    )
    assert shapely.contains(island, centres).sum() == 16
    assert not shapely.contains(box(101, 101, 159, 299), centres).any()  # the hole stays empty
