"""Select what stays: lattice squares that overlap the domain, cells not removed for water."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from refined_quads._areas import make_valid
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
    valid = [make_valid(g, f"water area {i}") for i, g in enumerate(water_areas)]
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
    boxes = grid.boxes(np.asarray([leaf[:4] for leaf in leaves], dtype=np.int64))
    shapely.prepare(domain)
    keep = shapely.contains(domain, boxes)
    partly = np.flatnonzero(~keep & shapely.intersects(domain, boxes))
    clipped = shapely.intersection(boxes[partly], domain)
    keep[partly] = ~shapely.is_empty(clipped) & (shapely.area(clipped) >= 1e-6)
    return [leaf for leaf, k in zip(leaves, keep.tolist(), strict=True) if k]
