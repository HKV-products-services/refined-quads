"""Select the leaves that stay: those overlapping the domain and not removed for water."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from refined_quads._grid import Grid, Leaf


@dataclass(frozen=True)
class WaterAreas:
    """Water areas made valid, with an STRtree, and the water removal rule."""

    geometries: NDArray[np.object_]
    tree: STRtree | None
    rule: str


def prepare_water(water_areas: Sequence[BaseGeometry], rule: str) -> WaterAreas:
    """Make water areas valid and index them.

    Parameters
    ----------
    water_areas : sequence of shapely geometry
        The water areas (holes of the domain included).
    rule : {"centre", "inside", "touches"}
        Water removal rule.

    Returns
    -------
    WaterAreas
        The prepared water areas.
    """
    valid = [g if g.is_valid else g.buffer(0) for g in water_areas]
    return WaterAreas(np.asarray(valid, dtype=object), STRtree(valid) if valid else None, rule)


def select_leaves(
    leaves: list[Leaf], domain: BaseGeometry, grid: Grid, water: WaterAreas
) -> list[Leaf]:
    """Keep the leaves that overlap the domain and that the water removal rule keeps.

    Parameters
    ----------
    leaves : list of Leaf
        Leaves of the balanced quadtree.
    domain : shapely geometry
        The domain (valid).
    grid : Grid
        The base grid.
    water : WaterAreas
        Prepared water areas.

    Returns
    -------
    list of Leaf
        The kept leaves, in their original order. A leaf is kept whole when it lies
        in the domain or overlaps it by at least 1e-6 square units of area.
    """
    if not leaves:
        return []
    ox, oy, unit = grid.ox, grid.oy, grid.unit
    rx0 = np.array([ox + c[0] * unit for c in leaves])
    ry0 = np.array([oy + c[1] * unit for c in leaves])
    rx1 = np.array([ox + c[2] * unit for c in leaves])
    ry1 = np.array([oy + c[3] * unit for c in leaves])
    boxes = shapely.box(rx0, ry0, rx1, ry1)
    contained = shapely.contains(domain, boxes)
    intersects = ~contained & shapely.intersects(domain, boxes)
    removed = _water_removed(boxes, (rx0 + rx1) * 0.5, (ry0 + ry1) * 0.5, water)

    kept: list[Leaf] = []
    for i, leaf in enumerate(leaves):
        if removed[i]:
            continue
        if intersects[i]:
            clipped = boxes[i].intersection(domain)
            if clipped.is_empty or clipped.area < 1e-6:
                continue
        elif not contained[i]:
            continue
        kept.append(leaf)
    return kept


def _water_removed(
    boxes: NDArray[np.object_],
    centres_x: NDArray[np.float64],
    centres_y: NDArray[np.float64],
    water: WaterAreas,
) -> NDArray[np.bool_]:
    """Mark the leaves that the water removal rule removes."""
    removed = np.zeros(len(boxes), dtype=bool)
    if water.tree is None or len(boxes) == 0:
        return removed
    if water.rule == "centre":
        centres = cast("NDArray[np.object_]", shapely.points(centres_x, centres_y))
        q_idx, w_idx = water.tree.query(centres, predicate="intersects")
        if len(q_idx):
            contains = shapely.contains(water.geometries[w_idx], centres[q_idx])
            np.logical_or.at(removed, q_idx[contains], True)
        return removed
    # "intersects" as broad preselection (STRtree predicates were not reliably
    # direction-independent across versions), then the exact test.
    q_idx, w_idx = water.tree.query(boxes, predicate="intersects")
    if len(q_idx):
        if water.rule == "inside":
            hit = shapely.contains(water.geometries[w_idx], boxes[q_idx])
        else:  # touches
            hit = shapely.intersects(water.geometries[w_idx], boxes[q_idx])
        np.logical_or.at(removed, q_idx[hit], True)
    return removed
