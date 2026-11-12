import numpy as np
import pytest
import shapely
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely import Point, box

from golden.compare import assert_same_cells
from golden.holes import closable_holes
from golden.scenarios import SCENARIOS_BY_NAME, X0, Y0
from handmade import CLEAN, RD, generate_rd, mesh_from_cells
from refined_quads import (
    InputError,
    Mesh,
    MeshRecognitionError,
    check,
    read_ugrid,
    refine,
    write_ugrid,
)

BASE = SCENARIOS_BY_NAME["cell_size_32"]
KW = {"cell_size": BASE.cell_size, "transition_rows": BASE.transition_rows}


@pytest.fixture(scope="module")
def base_mesh():
    return generate_rd(BASE.domain, refinement_areas=BASE.refinement_areas, **KW)


def refine_rd(mesh, areas, **kwargs):
    kwargs.setdefault("transition_rows", BASE.transition_rows)
    return refine(mesh, areas, crs=RD, **kwargs)


def cell_points(mesh, i):
    face = mesh.faces[i]
    return {tuple(p) for p in mesh.nodes[face[face >= 0]]}


def test_refine_equals_generate_with_the_extra_area(base_mesh):
    extra = [(Point(X0 + 700, Y0 + 180).buffer(40), 3)]

    result = refine_rd(base_mesh, extra)

    expected = generate_rd(BASE.domain, refinement_areas=[*BASE.refinement_areas, *extra], **KW)
    assert_same_cells(result.mesh, expected)
    assert check(result.mesh).is_valid


# Y inside the domain of 900 x 700 m, well away from its edge (no domain effects).
centres_x = st.integers(min_value=200, max_value=700)
centres_y = st.integers(min_value=200, max_value=500)


@pytest.mark.timeout(120)
@settings(max_examples=10, deadline=None, derandomize=True)
@given(
    cx=centres_x,
    cy=centres_y,
    radius=st.integers(min_value=5, max_value=80),
    level=st.integers(min_value=1, max_value=3),
)
def test_refine_of_generate_equals_generate_of_both(base_mesh, cx, cy, radius, level):
    extra = [(Point(X0 + cx, Y0 + cy).buffer(radius), level)]

    result = refine_rd(base_mesh, extra)

    expected = generate_rd(BASE.domain, refinement_areas=[*BASE.refinement_areas, *extra], **KW)
    assert_same_cells(result.mesh, expected)


def test_cells_outside_the_affected_area_keep_their_exact_coordinates(base_mesh):
    result = refine_rd(base_mesh, [(Point(X0 + 450, Y0 + 300).buffer(30), 3)])
    rm = result.replacement_map

    assert rm.unchanged.sum() > 0.9 * len(base_mesh.faces)
    for new in np.flatnonzero(rm.unchanged):
        assert cell_points(result.mesh, new) == cell_points(base_mesh, rm.source[new])


def test_area_partly_outside_the_footprint_does_not_grow_the_mesh(base_mesh):
    outside = box(X0 + 850, Y0 + 300, X0 + 1100, Y0 + 400)  # domain ends at 900

    result = refine_rd(base_mesh, [(outside, 3)])

    def footprint(mesh):
        polys = [shapely.Polygon(mesh.nodes[f[f >= 0]]) for f in mesh.faces]
        return shapely.union_all(polys)

    old, new = footprint(base_mesh), footprint(result.mesh)
    assert new.symmetric_difference(old).area < 1e-6
    assert len(result.mesh.faces) > len(base_mesh.faces)


def _centres(mesh):
    polys = [shapely.Polygon(mesh.nodes[f[f >= 0]]) for f in mesh.faces]
    return shapely.centroid(np.array(polys, dtype=object))


def _footprint(mesh):
    return shapely.union_all([shapely.Polygon(mesh.nodes[f[f >= 0]]) for f in mesh.faces])


def test_hole_does_not_get_new_area_and_with_the_domain_no_new_cells():
    scenario = SCENARIOS_BY_NAME["domain_with_hole"]
    mesh = generate_rd(scenario.domain, cell_size=32.0, refinement_areas=scenario.refinement_areas,
                       transition_rows=scenario.transition_rows)  # fmt: skip
    hole = shapely.Polygon(scenario.domain.interiors[0])
    around_hole = [(hole.buffer(40), 2)]

    without_domain = refine_rd(mesh, around_hole, transition_rows=(1, 1))
    with_domain = refine_rd(mesh, around_hole, transition_rows=(1, 1), domain=scenario.domain)

    # Without the domain refine cannot know the hole: squares that were clipped along it
    # are refined within their remaining cells, so the mesh never grows. It can lose a
    # little where finer cells cannot fill a clipped triangle (here 320 of 471424 m2).
    old, new = _footprint(mesh), _footprint(without_domain.mesh)
    assert new.difference(old).area < 1e-6
    assert old.difference(new).area < 0.001 * old.area
    # With the domain its hole acts as water for new cells (rule "centre").
    rm = with_domain.replacement_map
    new_in_hole = shapely.contains(hole, _centres(with_domain.mesh)) & ~rm.unchanged
    assert not new_in_hole.any()
    assert check(with_domain.mesh).is_valid


def test_replacement_map_covers_every_new_and_every_removed_cell(base_mesh):
    result = refine_rd(base_mesh, [(Point(X0 + 450, Y0 + 300).buffer(60), 3)])
    rm = result.replacement_map
    n_old = len(base_mesh.faces)

    assert len(rm.source) == len(result.mesh.faces)
    assert (rm.source >= 0).all()
    assert (rm.source < n_old).all()
    kept = set(rm.source[rm.unchanged].tolist())
    removed = set(rm.removed.tolist())
    assert kept.isdisjoint(removed)
    assert kept | removed == set(range(n_old))
    # Every changed cell lies in the old cell it points to.
    for new in np.flatnonzero(~rm.unchanged):
        old = base_mesh.faces[rm.source[new]]
        old_cell = shapely.Polygon(base_mesh.nodes[old[old >= 0]])
        new_face = result.mesh.faces[new]
        centre = shapely.centroid(shapely.Polygon(result.mesh.nodes[new_face[new_face >= 0]]))
        assert old_cell.buffer(1e-6).contains(centre)


def test_refined_mesh_read_back_from_ugrid_refines_cleanly(base_mesh, tmp_path):
    first = refine_rd(base_mesh, [(Point(X0 + 450, Y0 + 300).buffer(30), 2)]).mesh
    write_ugrid(first, tmp_path / "first.nc")

    second = refine_rd(
        read_ugrid(tmp_path / "first.nc"), [(Point(X0 + 600, Y0 + 500).buffer(30), 3)]
    )

    assert check(second.mesh).is_valid
    expected = generate_rd(
        BASE.domain,
        refinement_areas=[
            *BASE.refinement_areas,
            (Point(X0 + 450, Y0 + 300).buffer(30), 2),
            (Point(X0 + 600, Y0 + 500).buffer(30), 3),
        ],
        **KW,
    )
    assert_same_cells(second.mesh, expected)


def test_finer_level_than_the_mesh_raises_its_levels(base_mesh):
    result = refine_rd(base_mesh, [(Point(X0 + 450, Y0 + 300).buffer(10), 5)],
                       transition_rows=(1, 1, 1, 1, 1))  # fmt: skip

    assert result.mesh.lattice.levels == 5
    assert result.mesh.lattice.unit == 1.0
    assert max(check(result.mesh).cells_per_level) == 5
    assert check(result.mesh).is_valid
    assert result.replacement_map.unchanged.sum() > 0.9 * len(base_mesh.faces)


def test_water_areas_and_domain_at_refine_only_remove_new_cells(base_mesh):
    water = Point(X0 + 450, Y0 + 300).buffer(25)
    area = [(Point(X0 + 450, Y0 + 300).buffer(60), 3)]

    with_water = refine_rd(base_mesh, area, water_areas=[water])
    without = refine_rd(base_mesh, area)

    assert len(with_water.mesh.faces) < len(without.mesh.faces)
    # An existing cell that overlaps the water but is not affected stays.
    old_centres = shapely.centroid(
        np.array([shapely.Polygon(base_mesh.nodes[f[f >= 0]]) for f in base_mesh.faces])
    )
    far_water = Point(X0 + 100, Y0 + 600).buffer(40)
    kept = refine_rd(base_mesh, area, water_areas=[far_water]).replacement_map
    in_far_water = set(np.flatnonzero(shapely.contains(far_water, old_centres)).tolist())
    assert in_far_water
    assert in_far_water <= set(kept.source[kept.unchanged].tolist())


def test_mesh_without_crs_takes_plain_geometries():
    mesh = mesh_from_cells([[(-4, 0), (0, 0), (0, 4), (-4, 4)], *CLEAN])

    result = refine(mesh, [(box(1000.5, 2000.5, 1001.5, 2001.5), 2)])

    assert result.mesh.crs is None
    assert result.mesh.lattice.levels == 2
    assert check(result.mesh).is_valid


def test_unrecognised_cell_raises_with_its_location():
    mesh = mesh_from_cells([[(0, 0), (3, 0), (0, 3)]], levels=2)

    with pytest.raises(MeshRecognitionError, match="neither a lattice square") as caught:
        refine(mesh, [(box(1000, 2000, 1001, 2001), 1)])
    assert caught.value.locations == [(1001.0, 2001.0)]


def test_two_to_one_violation_raises_with_its_location():
    fine = [[(4, y), (5, y), (5, y + 1), (4, y + 1)] for y in range(4)]
    mesh = mesh_from_cells([[(0, 0), (4, 0), (4, 4), (0, 4)], *fine], levels=2)

    with pytest.raises(MeshRecognitionError, match="2:1 balance") as caught:
        refine(mesh, [(box(1000, 2000, 1001, 2001), 1)])
    assert len(caught.value.locations) == 4


def test_node_off_the_lattice_raises_with_its_location():
    mesh = mesh_from_cells(CLEAN)
    nodes = mesh.nodes.copy()
    nodes[0] += [0.3, 0.0]

    with pytest.raises(MeshRecognitionError, match="not on the lattice"):
        refine(Mesh(nodes, mesh.faces, mesh.lattice), [(box(1000, 2000, 1001, 2001), 1)])


def test_areas_with_a_crs_need_a_mesh_with_a_crs():
    from refined_quads import Areas

    mesh = mesh_from_cells(CLEAN)

    with pytest.raises(InputError, match="has no CRS"):
        refine(mesh, Areas((box(1000, 2000, 1001, 2001),), RD, (1,)))


def test_refine_keeps_the_cells_of_a_small_water_area_meshkernel_would_close():
    # Issue 20 / ADR-0004: water over two adjacent 8 m cells would leave a hole of six
    # nodes that hydrolib and the D-HYDRO GUI make one (hexagonal) cell of.
    mesh = generate_rd(box(X0, Y0, X0 + 320, Y0 + 256), cell_size=32.0)
    ox, oy = mesh.lattice.origin
    area = box(ox + 96, oy + 64, ox + 224, oy + 192)
    water = box(ox + 127.5, oy + 103.5, ox + 144.5, oy + 112.5)

    result = refine_rd(mesh, [(area, 2)], transition_rows=1, water_areas=[water],
                       water_removal="inside")  # fmt: skip

    assert closable_holes(result.mesh) == []
    corners = {(ox + x, oy + y) for x in (128, 136, 144) for y in (104, 112)}
    kept = [i for i in range(len(result.mesh.faces)) if cell_points(result.mesh, i) <= corners]
    assert len(kept) == 2
    assert check(result.mesh).is_valid
