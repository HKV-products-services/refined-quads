"""Turn lattice squares into cells: nodes, split sides and transition patterns."""

from __future__ import annotations

from collections import deque

from refined_quads._common.grid import Key, Leaf, neighbours
from refined_quads._common.progress import Reporter
from refined_quads._generate.balance import finer_exists

_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


class Nodes:
    """The nodes of a mesh as integer lattice points, numbered in order of creation."""

    def __init__(self, points: list[tuple[int, int]] | None = None) -> None:
        self.index: dict[tuple[int, int], int] = {}
        self.points: list[tuple[int, int]] = []
        for point in points or []:
            # A duplicate node (same lattice point) keeps its number; new cells use the first.
            self.index.setdefault(point, len(self.points))
            self.points.append(point)

    def add(self, ix: int, iy: int) -> int:
        """Return the number of the node at ``(ix, iy)``, creating it if needed."""
        key = (ix, iy)
        number = self.index.get(key)
        if number is None:
            number = len(self.points)
            self.index[key] = number
            self.points.append(key)
        return number

    def get(self, ix: int, iy: int) -> int | None:
        """Return the number of the node at ``(ix, iy)``, or None if there is none."""
        return self.index.get((ix, iy))


def square_cells(
    corners: tuple[int, int, int, int],
    m_s: int | None,
    m_e: int | None,
    m_n: int | None,
    m_w: int | None,
    centre: int | None = None,
) -> list[list[int]]:
    """Return the cells of a lattice square for its split sides (the transition patterns).

    Parameters
    ----------
    corners : tuple of int
        Nodes of the corners: bottom-left, bottom-right, top-right, top-left.
    m_s, m_e, m_n, m_w : int or None
        Node at the midpoint of the south, east, north and west side, if split.
    centre : int, optional
        Centre node; required, with all four midpoints, for two opposite or three
        or four split sides.

    Returns
    -------
    list of list of int
        No split: the square itself. One split: a fan from the midpoint to the two
        opposite corners (three triangles). Two adjacent splits: a fan joining the
        two midpoints and towards the opposite corner (four triangles). Otherwise:
        four squares around the centre.
    """
    bl, br, tr, tl = corners
    splits = (m_s is not None) + (m_e is not None) + (m_n is not None) + (m_w is not None)
    if splits == 0:
        return [[bl, br, tr, tl]]
    if splits == 1:
        if m_s is not None:
            return [[bl, m_s, tl], [m_s, br, tr], [m_s, tr, tl]]
        if m_e is not None:
            return [[br, m_e, bl], [m_e, tr, tl], [m_e, tl, bl]]
        if m_n is not None:
            return [[tl, m_n, bl], [m_n, tr, br], [m_n, br, bl]]
        assert m_w is not None
        return [[tl, m_w, tr], [m_w, bl, br], [m_w, br, tr]]
    if splits == 2 and (m_s is None or m_n is None) and (m_e is None or m_w is None):
        if m_s is not None and m_e is not None:
            return [[br, m_s, m_e], [bl, m_s, tl], [m_s, m_e, tl], [m_e, tr, tl]]
        if m_s is not None and m_w is not None:
            return [[bl, m_s, m_w], [m_s, br, tr], [m_s, tr, m_w], [m_w, tr, tl]]
        if m_n is not None and m_e is not None:
            return [[tr, m_n, m_e], [m_n, tl, bl], [m_n, bl, m_e], [m_e, bl, br]]
        assert m_n is not None
        assert m_w is not None
        return [[tl, m_w, m_n], [m_w, bl, br], [m_w, br, m_n], [m_n, br, tr]]
    assert centre is not None
    assert m_s is not None
    assert m_e is not None
    assert m_n is not None
    assert m_w is not None
    return [
        [bl, m_s, centre, m_w],
        [m_s, br, m_e, centre],
        [centre, m_e, tr, m_n],
        [m_w, centre, m_n, tl],
    ]


def centre_split(sides: set[str]) -> bool:
    """Tell whether a square with these split sides is split around its centre."""
    return len(sides) >= 3 or (len(sides) == 2 and sides in ({"N", "S"}, {"E", "W"}))


def split_sides(cells: dict[Key, int], leaves: list[Leaf]) -> dict[Key, set[str]]:
    """Determine which sides of each lattice square get a midpoint node.

    Parameters
    ----------
    cells : dict
        The kept lattice squares (key to level).
    leaves : list of Leaf
        The same squares, in emission order.

    Returns
    -------
    dict
        Split sides (subset of ``"NSEW"``) per square.

    Notes
    -----
    A side is split where the neighbour is finer. A centre split puts a midpoint on
    all four sides, which forces the neighbour on that side to split as well; that
    can make the neighbour a centre split too, so this iterates to a fixed point.
    """
    sides: dict[Key, set[str]] = {}
    for ix0, iy0, ix1, iy1, _lvl in leaves:
        sides[(ix0, iy0, ix1, iy1)] = {
            side for side in "NSEW" if finer_exists(cells, ix0, iy0, ix1, iy1, side)
        }
    work = deque(sides.keys())
    while work:
        key = work.popleft()
        if not centre_split(sides[key]):
            continue
        for side, neighbour in zip("NSEW", neighbours(key), strict=True):
            if neighbour in sides:
                need = _OPPOSITE[side]
                if need not in sides[neighbour]:
                    sides[neighbour].add(need)
                    work.append(neighbour)
    return sides


def emit_cells(
    leaves: list[Leaf],
    sides: dict[Key, set[str]],
    nodes: Nodes,
    report: Reporter,
    keep: dict[Key, list[list[int]]] | None = None,
) -> list[list[int]]:
    """Return the cells of all lattice squares, creating their nodes.

    A square in ``keep`` (refine: unchanged squares of the existing mesh) gets those
    cells instead of a pattern.

    Nodes are created per square in the order of the original generator: corners,
    then for a centre split the centre and the midpoints S, E, N, W, otherwise the
    midpoint of the vertical side (S or N) before that of the horizontal side. The
    head-to-toe step depends on the node numbering.
    """
    cells: list[list[int]] = []
    n_leaves = len(leaves)
    report_every = max(1, n_leaves // 20)
    for i, (ix0, iy0, ix1, iy1, _lvl) in enumerate(leaves):
        if i % report_every == 0:
            report(80 + int(18 * i / n_leaves))
        if keep and (ix0, iy0, ix1, iy1) in keep:
            cells += [list(c) for c in keep[(ix0, iy0, ix1, iy1)]]
            continue
        ixm = (ix0 + ix1) >> 1
        iym = (iy0 + iy1) >> 1
        corners = (
            nodes.add(ix0, iy0),
            nodes.add(ix1, iy0),
            nodes.add(ix1, iy1),
            nodes.add(ix0, iy1),
        )
        split = sides[(ix0, iy0, ix1, iy1)]
        mids: dict[str, int] = {}
        centre = None
        if centre_split(split):
            centre = nodes.add(ixm, iym)
            order: tuple[str, ...] = ("S", "E", "N", "W")
        else:
            order = tuple(side for side in "SNEW" if side in split)
        position = {"S": (ixm, iy0), "E": (ix1, iym), "N": (ixm, iy1), "W": (ix0, iym)}
        for side in order:
            mids[side] = nodes.add(*position[side])
        cells += square_cells(
            corners, mids.get("S"), mids.get("E"), mids.get("N"), mids.get("W"), centre
        )
    return cells
