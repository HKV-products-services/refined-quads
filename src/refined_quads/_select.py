"""Select what stays: lattice squares that overlap the domain, cells not removed for water."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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


def select_leaves(leaves: list[Leaf], domain: BaseGeometry | None, grid: Grid) -> list[Leaf]:
    """Keep the leaves that overlap the domain.

    Parameters
    ----------
    leaves : list of Leaf
        Leaves of the balanced quadtree.
    domain : shapely geometry or None
        The domain (valid); None keeps every leaf.
    grid : Grid
        The base grid.

    Returns
    -------
    list of Leaf
        The kept leaves, in their original order. A leaf is kept whole when it lies
        in the domain or overlaps it by at least 1e-6 square units of area.
    """
    if not leaves or domain is None:
        return list(leaves)
    ox, oy, unit = grid.ox, grid.oy, grid.unit
    rx0 = np.array([ox + c[0] * unit for c in leaves])
    ry0 = np.array([oy + c[1] * unit for c in leaves])
    rx1 = np.array([ox + c[2] * unit for c in leaves])
    ry1 = np.array([oy + c[3] * unit for c in leaves])
    boxes = shapely.box(rx0, ry0, rx1, ry1)
    contained = shapely.contains(domain, boxes)
    intersects = ~contained & shapely.intersects(domain, boxes)
    kept: list[Leaf] = []
    for i, leaf in enumerate(leaves):
        if intersects[i]:
            clipped = boxes[i].intersection(domain)
            if clipped.is_empty or clipped.area < 1e-6:
                continue
        elif not contained[i]:
            continue
        kept.append(leaf)
    return kept
