import math

import pytest
import shapely
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely import Polygon, box

from golden.compare import assert_same_cells
from handmade import generate_rd
from refined_quads import Cancelled, InputError, Lattice, Mesh, check

# Coordinates on 1/8 m, so that shifting by 1/8 m steps is exact in floating point.
X0, Y0 = 187_000.0, 318_000.0
DOMAIN = box(X0, Y0, X0 + 410.0, Y0 + 290.0)
AREAS = [
    (box(X0 + 120.0, Y0 + 80.0, X0 + 190.5, Y0 + 150.25), 2),
    (
        Polygon(
            [(X0 + 300, Y0 + 150), (X0 + 340, Y0 + 190), (X0 + 300, Y0 + 230), (X0 + 260, Y0 + 190)]
        ),
        1,
    ),
]
WATER = [box(X0 + 20.125, Y0 + 200.0, X0 + 90.0, Y0 + 230.5)]


def make(shift=(0.0, 0.0), **kwargs):
    dx, dy = shift
    move = lambda g: shapely.affinity.translate(g, dx, dy)  # noqa: E731
    return generate_rd(
        move(DOMAIN),
        cell_size=25.0,
        refinement_areas=[(move(g), level) for g, level in AREAS],
        water_areas=[move(g) for g in WATER],
        **kwargs,
    )


def test_progress_rises_to_one_with_english_messages():
    calls = []

    make(progress=lambda fraction, message: calls.append((fraction, message)))

    fractions = [f for f, _ in calls]
    assert len(calls) > 5
    assert fractions == sorted(fractions)
    assert fractions[0] >= 0.0
    assert fractions[-1] == 1.0
    assert all(isinstance(m, str) and m and m.isascii() for _, m in calls)
    assert calls[-1][1].startswith("Quad Grid: done")


@pytest.mark.parametrize("stop_at", [0.0, 0.4, 0.99])
def test_returning_true_cancels_the_run(stop_at):
    seen = []

    def progress(fraction, message):
        seen.append(fraction)
        return fraction >= stop_at

    with pytest.raises(Cancelled, match="cancelled"):
        make(progress=progress)
    assert seen[-1] >= stop_at
    assert seen[-1] < 1.0


def test_other_return_values_do_not_cancel():
    mesh = make(progress=lambda fraction, message: 1)

    assert check(mesh).is_valid


def test_default_origin_is_the_domain_corner_rounded_down_to_the_cell_size():
    default = make()
    explicit = make(origin=(175.0, 300.0))  # also a multiple of 25 m

    assert default.lattice.origin == (187_000.0, 318_000.0)
    assert explicit.lattice.origin == (175.0, 300.0)
    assert_same_cells(explicit, default)


def test_origin_moved_by_whole_cells_changes_nothing():
    assert_same_cells(make(origin=(X0 - 2 * 25.0, Y0 + 7 * 25.0)), make())


def test_origin_that_is_not_a_multiple_of_the_cell_size_moves_the_lattice():
    mesh = make(origin=(3.125, 12.5))

    assert mesh.lattice.origin == (3.125, 12.5)
    _, on_lattice = mesh.lattice.to_indices(mesh.nodes)
    assert on_lattice.all()
    assert check(mesh).is_valid
    assert math.isclose(mesh.nodes[:, 0].min() % 25.0, 3.125)


eighths = st.integers(min_value=-4000, max_value=4000).map(lambda i: i / 8)


@settings(max_examples=8, deadline=None, derandomize=True)
@given(dx=eighths, dy=eighths, ox=eighths, oy=eighths)
def test_shifting_input_and_origin_only_shifts_the_mesh(dx, dy, ox, oy):
    origin = (X0 + ox, Y0 + oy)
    mesh = make(origin=origin)
    shifted = make(shift=(dx, dy), origin=(origin[0] + dx, origin[1] + dy))

    back = Mesh(
        shifted.nodes - [dx, dy],
        shifted.faces,
        Lattice(origin, shifted.lattice.cell_size, shifted.lattice.levels),
    )
    assert_same_cells(back, mesh)
    assert check(mesh).is_valid


@pytest.mark.parametrize(
    "origin", [(1.0,), (1.0, 2.0, 3.0), ("a", 1.0), (math.nan, 0.0), "00", 5.0]
)
def test_invalid_origin_raises_input_error(origin):
    with pytest.raises(InputError, match="origin must be"):
        make(origin=origin)
