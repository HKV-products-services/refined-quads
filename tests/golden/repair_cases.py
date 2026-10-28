"""Handmade meshes with slivers, for comparing the orthogonality repair with the original.

No golden-master scenario needs the repair (2026-10-28: 0 flips, 0 redecompositions),
so these small meshes exercise it directly. Each case is cells of lattice points (in
units) on a lattice; ``record.py --repair-reference`` stores what the repair of the
generator at qgis_dhydamo 9a324bc makes of them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepairCase:
    name: str
    description: str
    cells: tuple[tuple[tuple[int, int], ...], ...]
    origin: tuple[float, float] = (187_000.0, 318_000.0)
    unit: float = 1.0


def _square(x, y, s):
    return ((x, y), (x + s, y), (x + s, y + s), (x, y + s))


def _ring_of_fine_squares(x0, y0, size, fine):
    """Fine squares along the four sides of the square at (x0, y0) of side ``size``."""
    cells = []
    for k in range(0, size, fine):
        cells += [
            _square(x0 + k, y0 - fine, fine),  # south
            _square(x0 + k, y0 + size, fine),  # north
            _square(x0 - fine, y0 + k, fine),  # west
            _square(x0 + size, y0 + k, fine),  # east
        ]
    return cells


# A: a square with one split side (east) cut with the wrong diagonal.
_FINE_EAST = [_square(4, 0, 2), _square(4, 2, 2), _square(6, 0, 2), _square(6, 2, 2)]
WRONG_DIAGONAL = (
    ((0, 0), (4, 0), (4, 2)),
    ((0, 0), (4, 2), (4, 4)),
    ((0, 0), (4, 4), (0, 4)),
    *_FINE_EAST,
)

# B: a square with four split sides filled by a fan of six triangles from the south
# midpoint, without a centre node; fine squares all around.
FAN_FROM_SOUTH = (
    ((6, 4), (8, 4), (8, 6)),
    ((6, 4), (8, 6), (8, 8)),
    ((6, 4), (8, 8), (6, 8)),
    ((6, 4), (6, 8), (4, 8)),
    ((6, 4), (4, 8), (4, 6)),
    ((6, 4), (4, 6), (4, 4)),
    *_ring_of_fine_squares(4, 4, 4, 2),
)

# C: one split side (east) but the fan starts at the bottom-left corner.
FAN_FROM_CORNER = (
    ((0, 0), (4, 0), (4, 2)),
    ((0, 0), (4, 2), (4, 4)),
    ((0, 0), (4, 4), (0, 4)),
    *_FINE_EAST,
    _square(-4, 0, 4),
)

# D: two adjacent split sides (south and east) triangulated from the top-left corner.
_FINE_SE = [_square(4, -2, 2), _square(6, -2, 2), _square(8, 0, 2), _square(8, 2, 2)]
ADJACENT_WRONG = (
    ((4, 4), (4, 0), (6, 0)),
    ((4, 4), (6, 0), (8, 0)),
    ((4, 4), (8, 0), (8, 2)),
    ((4, 4), (8, 2), (8, 4)),
    *_FINE_SE,
)

CASES = (
    RepairCase("wrong_diagonal", "One split side, cut with the wrong diagonal.", WRONG_DIAGONAL),
    RepairCase("fan_from_south", "Four split sides as a fan without centre.", FAN_FROM_SOUTH),
    RepairCase("fan_from_corner", "One split side, fan from a corner.", FAN_FROM_CORNER),
    RepairCase("adjacent_wrong", "Two adjacent split sides, wrong fan.", ADJACENT_WRONG),
    RepairCase(
        "fan_from_south_10_1",
        "Case B on a 10.1 m lattice: the new centre node must be exactly on the lattice.",
        FAN_FROM_SOUTH,
        # At this origin the repair needs a redecomposition, so a centre node is added.
        origin=(155_000.0, 463_000.0),
        unit=10.1 / 4,
    ),
)
