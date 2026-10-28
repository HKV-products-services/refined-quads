"""Synthetic golden-master scenarios (ADR-0002).

Each scenario is input in the terms of this package: transition rows per step from
coarse to fine and at least 1, the highest refinement level following from the
refinement areas. ``record.py`` translates them to the parameters of the generator at
qgis_dhydamo ``9a324bc``.

Changing a scenario changes its golden master: re-record it (``record.py --rerecord``)
and add a changelog entry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely import Polygon, box
from shapely.affinity import rotate
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

WATER_REMOVAL_RULES = ("centre", "inside", "touches")


@dataclass(frozen=True)
class Scenario:
    """Input for one golden-master mesh."""

    name: str
    description: str
    domain: BaseGeometry
    cell_size: float
    refinement_areas: tuple[tuple[BaseGeometry, int], ...]
    transition_rows: tuple[int, ...]
    water_areas: tuple[BaseGeometry, ...] = ()
    water_removal: str = "centre"

    def __post_init__(self):
        if self.water_removal not in WATER_REMOVAL_RULES:
            raise ValueError(f"{self.name}: unknown water removal rule {self.water_removal!r}")
        if len(self.transition_rows) != self.levels or min(self.transition_rows) < 1:
            raise ValueError(f"{self.name}: need one transition row count >= 1 per level")

    @property
    def levels(self) -> int:
        return max(level for _, level in self.refinement_areas)


# Coordinates near the RD origin used in practice, so that the origin is not 0 and
# coordinates are large floats as in real models.
X0, Y0 = 187_013.0, 318_457.0


def _circle(x, y, r):
    return Point(X0 + x, Y0 + y).buffer(r, quad_segs=16)


def _box(x0, y0, x1, y1):
    return box(X0 + x0, Y0 + y0, X0 + x1, Y0 + y1)


# The head-to-toe scenarios keep the coordinates of the random search they come from.
TX, TY = 187_000.0, 318_000.0


def _disc(x, y, r, quad_segs=16):
    return Point(TX + x, TY + y).buffer(r, quad_segs=quad_segs)


def _bar(x, y, w, h, angle):
    return rotate(box(TX + x, TY + y, TX + x + w, TY + y + h), angle, origin="centroid")


def _l_shape():
    return Polygon(
        [(X0, Y0), (X0 + 900, Y0), (X0 + 900, Y0 + 350), (X0 + 400, Y0 + 350),
         (X0 + 400, Y0 + 800), (X0, Y0 + 800)]
    )  # fmt: skip


def _domain_with_hole():
    outer = _box(0, 0, 800, 600)
    hole = _circle(420, 300, 140)
    return outer.difference(hole)


def _small_water_areas(n=60, seed=20261028):
    """Many small ponds and ditches, partly smaller than a base cell (Parkstad situation)."""
    rng = np.random.default_rng(seed)
    areas = []
    for _ in range(n):
        x, y = rng.uniform(20, 780), rng.uniform(20, 580)
        if rng.random() < 0.5:
            areas.append(_circle(x, y, rng.uniform(3, 25)))
        else:
            ditch = _box(x, y, x + rng.uniform(30, 120), y + rng.uniform(2, 6))
            areas.append(rotate(ditch, rng.uniform(0, 180), origin="centroid"))
    return tuple(areas)


def _water_rule_scenario(rule):
    return Scenario(
        name=f"water_removal_{rule}",
        description=f"Water areas removed with the '{rule}' water removal rule.",
        domain=_box(0, 0, 640, 480),
        cell_size=32.0,
        refinement_areas=((_box(120, 100, 420, 300), 2),),
        transition_rows=(1, 1),
        water_areas=(
            rotate(_box(60, 220, 600, 260), 17, origin="centroid"),
            _circle(500, 120, 45),
            _circle(150, 380, 12),
        ),
        water_removal=rule,
    )


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="refinement_at_domain_edge",
        description="Refinement areas that cross and touch the domain boundary.",
        domain=_box(0, 0, 800, 600),
        cell_size=32.0,
        refinement_areas=((_box(700, 180, 860, 420), 2), (_circle(0, 0, 120), 3)),
        transition_rows=(1, 1, 1),
    ),
    Scenario(
        name="cell_size_25",
        description="Base cell of 25 m, not a power of 2 (formerly rounded to 16 m).",
        domain=_box(0, 0, 700, 500),
        cell_size=25.0,
        refinement_areas=((_circle(300, 250, 90), 3), (_box(520, 60, 640, 160), 1)),
        transition_rows=(1, 1, 1),
    ),
    Scenario(
        name="cell_size_12_5",
        description="Base cell of 12.5 m with three levels (1.5625 m cells).",
        domain=_box(0, 0, 400, 300),
        cell_size=12.5,
        refinement_areas=((_circle(200, 150, 40), 3),),
        transition_rows=(1, 1, 1),
    ),
    Scenario(
        name="cell_size_24_four_levels",
        description="Base cell of 24 m with four levels (1.5 m cells).",
        domain=_box(0, 0, 600, 480),
        cell_size=24.0,
        refinement_areas=((rotate(_box(200, 200, 420, 240), 30, origin="centroid"), 4),),
        transition_rows=(1, 1, 1, 1),
    ),
    Scenario(
        name="cell_size_32",
        description="Base cell of 32 m with areas at levels 1, 2 and 3.",
        domain=_box(0, 0, 900, 700),
        cell_size=32.0,
        refinement_areas=(
            (_box(80, 80, 300, 260), 1),
            (_circle(560, 420, 110), 2),
            (_circle(250, 520, 50), 3),
        ),
        transition_rows=(1, 1, 1),
    ),
    Scenario(
        name="many_small_water_areas",
        description="Many small water areas, partly smaller than a base cell (pinch nodes).",
        domain=_box(0, 0, 800, 600),
        cell_size=32.0,
        refinement_areas=((_box(250, 150, 550, 450), 2),),
        transition_rows=(1, 1),
        water_areas=_small_water_areas(),
    ),
    Scenario(
        name="transition_rows_3_2_1_1",
        description=(
            "Non-uniform transition rows; '3 2 1 1' from fine to coarse in the old "
            "generator, so (1, 1, 2, 3) from coarse to fine (formerly slivers)."
        ),
        domain=_box(0, 0, 900, 700),
        cell_size=32.0,
        refinement_areas=((_circle(450, 350, 60), 4), (_box(100, 500, 220, 620), 3)),
        transition_rows=(1, 1, 2, 3),
    ),
    Scenario(
        name="l_shaped_domain",
        description="L-shaped domain with refinement in the inner corner.",
        domain=_l_shape(),
        cell_size=32.0,
        refinement_areas=((_circle(400, 350, 80), 2),),
        transition_rows=(1, 1),
    ),
    Scenario(
        name="domain_with_hole",
        description="Domain with a hole, refined along part of the hole's edge.",
        domain=_domain_with_hole(),
        cell_size=32.0,
        refinement_areas=((_box(250, 200, 420, 400), 2),),
        transition_rows=(1, 1),
    ),
    *(_water_rule_scenario(rule) for rule in WATER_REMOVAL_RULES),
    # Found by a random search (2026-10-28) as inputs where the head-to-toe correction
    # changes cells; no other scenario reaches that step.
    Scenario(
        name="head_to_toe_circles",
        description="Diamond-shaped areas at level 3 with transition rows (1, 2, 3); head-to-toe.",
        domain=box(TX, TY, TX + 429.9, TY + 296.9),
        cell_size=32.0,
        refinement_areas=(
            (_disc(316.4, 275.0, 44.2, 1), 3),
            (_disc(378.9, 51.7, 59.3, 6), 1),
            (_disc(88.9, 167.7, 46.3, 1), 3),
        ),
        transition_rows=(1, 2, 3),
        water_areas=(
            _disc(330.6, 20.1, 15.3),
            _disc(14.0, 93.2, 10.7),
            _disc(309.4, 135.1, 3.6),
            _disc(427.9, 263.8, 27.7),
        ),
    ),
    Scenario(
        name="head_to_toe_boxes",
        description="Two thin rotated boxes at level 3, transition rows (2, 1, 1); head-to-toe.",
        domain=box(TX, TY, TX + 261.7, TY + 302.0),
        cell_size=25.0,
        refinement_areas=(
            (_bar(161.9, 259.8, 23.5, 20.7, 88.5), 3),
            (_bar(39.3, 153.8, 50.9, 12.7, 100.1), 3),
        ),
        transition_rows=(2, 1, 1),
        water_areas=(
            _disc(22.2, 62.4, 4.3),
            _disc(186.3, 88.5, 27.2),
            _disc(233.9, 282.8, 9.2),
            _disc(0.2, 117.0, 29.0),
            _disc(199.9, 18.0, 21.7),
        ),
    ),
)

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}
