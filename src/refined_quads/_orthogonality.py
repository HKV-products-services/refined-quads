"""FM orthogonality: the measure D-Flow FM uses to reject a network as not orthogonal.

The measure is taken from the D-HYDRO Mesh Editor (``compute_fm_orthogonality_circumcenter``),
as in the generator of qgis_dhydamo ``9a324bc``. For each internal edge

    orth_e = |cos(angle between the edge and the line joining the two face centres)|

with 0 perfectly orthogonal and 1 worst. The centre of a triangle is its circumcentre
clipped into the triangle (the RGFGRID centre), with the centroid as fallback for
(nearly) degenerate triangles; the centre of any other face is the mean of its nodes.
Clipping matters: the raw circumcentre of an obtuse sliver lies far outside it on the
perpendicular bisector of every edge, which would hide exactly the slivers D-Flow FM
rejects.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from refined_quads._faces import Nodes, square_cells
from refined_quads._topology import Edges, directed_edges, unique_edges
from refined_quads.errors import OrthogonalityRepairError

#: Highest FM orthogonality D-Flow FM accepts on an internal edge.
MAX_FM_ORTHOGONALITY = 0.30

_log = logging.getLogger("refined_quads._engine")

BadEdge = tuple[int, int, int, int]


def face_centres(
    nodes: NDArray[np.float64], faces: NDArray[np.int64], h: float | None = None
) -> NDArray[np.float64]:
    """Compute the FM centre of every face.

    Parameters
    ----------
    nodes : numpy.ndarray of float, shape (n, 2)
        Node coordinates.
    faces : numpy.ndarray of int, shape (m, k)
        Node indices per face, padded with ``-1``.
    h : float, optional
        Length scale for the degeneracy tests; by default the median edge length.

    Returns
    -------
    numpy.ndarray of float, shape (m, 2)
        Clipped circumcentre for triangles, mean of the nodes for other faces.
    """
    valid = faces >= 0
    sizes = valid.sum(axis=1)
    xy = np.where(valid[..., None], nodes[np.where(valid, faces, 0)], 0.0)
    centres = xy.sum(axis=1) / sizes[:, None]
    tri = np.flatnonzero(sizes == 3)
    if tri.size:
        if h is None:
            h = typical_edge_length(nodes, faces)
        corners = nodes[faces[tri, :3]]
        centres[tri] = _rgfgrid_centres(corners[:, 0], corners[:, 1], corners[:, 2], h)
    return centres


def edge_orthogonality(
    nodes: NDArray[np.float64], faces: NDArray[np.int64], edges: Edges, h: float | None = None
) -> NDArray[np.float64]:
    """Compute the FM orthogonality of every internal edge.

    Parameters
    ----------
    nodes : numpy.ndarray of float, shape (n, 2)
        Node coordinates.
    faces : numpy.ndarray of int, shape (m, k)
        Node indices per face, padded with ``-1``.
    edges : Edges
        Unique edges of ``faces``.
    h : float, optional
        Length scale for the triangle centres (see :func:`face_centres`).

    Returns
    -------
    numpy.ndarray of float, shape (e,)
        Orthogonality per edge; NaN for boundary edges and edges shared by more than
        two faces, ``inf`` for internal edges that are degenerate (zero length or
        coinciding face centres).
    """
    result = np.full(len(edges.nodes), np.nan)
    internal = np.flatnonzero(edges.face_count == 2)
    if internal.size == 0:
        return result
    centres = face_centres(nodes, faces, h)
    ev = nodes[edges.nodes[internal, 1]] - nodes[edges.nodes[internal, 0]]
    cv = centres[edges.faces[internal, 1]] - centres[edges.faces[internal, 0]]
    result[internal] = orthogonality(ev, cv)
    return result


def orthogonality(
    edge_vectors: NDArray[np.float64], centre_vectors: NDArray[np.float64]
) -> NDArray[np.float64]:
    """FM orthogonality of edges, given each edge vector and the vector between the centres.

    Parameters
    ----------
    edge_vectors, centre_vectors : numpy.ndarray of float, shape (e, 2)
        Edge vectors and centre-to-centre vectors.

    Returns
    -------
    numpy.ndarray of float, shape (e,)
        ``|cos|`` of the angle between them; ``inf`` where one has zero length.
    """
    el = np.hypot(edge_vectors[:, 0], edge_vectors[:, 1])
    cl = np.hypot(centre_vectors[:, 0], centre_vectors[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        orth = np.abs(
            (edge_vectors[:, 0] * centre_vectors[:, 0] + edge_vectors[:, 1] * centre_vectors[:, 1])
            / (el * cl)
        )
    orth[(el <= 0.0) | (cl <= 0.0) | ~np.isfinite(orth)] = np.inf
    return np.asarray(orth)


def typical_edge_length(nodes: NDArray[np.float64], faces: NDArray[np.int64]) -> float:
    """Median length of all face edges, the scale for the degeneracy thresholds."""
    _, edge_nodes = directed_edges(faces)
    d = nodes[edge_nodes[:, 1]] - nodes[edge_nodes[:, 0]]
    length = np.hypot(d[:, 0], d[:, 1])
    length = length[length > 0.0]
    h = float(np.median(length)) if length.size else 1.0
    return h if np.isfinite(h) and h > 0.0 else 1.0


def _rgfgrid_centres(
    a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64], h: float
) -> NDArray[np.float64]:
    """Circumcentres clipped into their triangles; centroids where degenerate."""
    centroid = (a + b + c) / 3.0
    ax, ay, bx, by, cx, cy = a[:, 0], a[:, 1], b[:, 0], b[:, 1], c[:, 0], c[:, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ok = np.abs(d) >= 1e-12 * (h * h)
    out = centroid.copy()
    if not ok.any():
        return out
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    dk = d[ok]
    ux = (a2[ok] * (by[ok] - cy[ok]) + b2[ok] * (cy[ok] - ay[ok]) + c2[ok] * (ay[ok] - by[ok])) / dk
    uy = (a2[ok] * (cx[ok] - bx[ok]) + b2[ok] * (ax[ok] - cx[ok]) + c2[ok] * (bx[ok] - ax[ok])) / dk
    cc = np.column_stack([ux, uy])
    # Nearly collinear: the circumcentre is unstable, keep the centroid.
    far = np.hypot(*(cc - centroid[ok]).T) > 50.0 * h
    clipped = _clip_into_triangles(cc, a[ok], b[ok], c[ok])
    clipped[far] = centroid[ok][far]
    out[ok] = clipped
    return out


def _clip_into_triangles(
    p: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Project each point that lies outside its triangle onto the nearest triangle edge."""

    def area2(
        u: NDArray[np.float64], v: NDArray[np.float64], w: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.asarray(
            (v[:, 0] - u[:, 0]) * (w[:, 1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (w[:, 0] - u[:, 0])
        )

    s = np.stack([area2(p, a, b), area2(p, b, c), area2(p, c, a)])
    outside = (s < -1e-12).any(axis=0) & (s > 1e-12).any(axis=0)
    out = p.copy()
    if not outside.any():
        return out
    po, ao, bo, co = p[outside], a[outside], b[outside], c[outside]
    candidates = np.stack(
        [
            _closest_on_segment(po, ao, bo),
            _closest_on_segment(po, bo, co),
            _closest_on_segment(po, co, ao),
        ]
    )
    dist = ((po - candidates) ** 2).sum(axis=2)
    # Ties go to the first edge (ab, then bc), as in the reference implementation.
    nearest = np.argmin(dist, axis=0)
    out[outside] = candidates[nearest, np.arange(len(po))]
    return out


def _closest_on_segment(
    p: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Closest point on segment ``[a, b]`` to each ``p``."""
    ab = b - a
    denom = (ab**2).sum(axis=1)
    safe = np.where(denom > 0, denom, 1.0)
    t = np.where(denom > 0, ((p - a) * ab).sum(axis=1) / safe, 0.0)
    return np.asarray(a + np.clip(t, 0.0, 1.0)[:, None] * ab)


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


class _Stats(NamedTuple):
    n_internal: int
    worst: float
    n_degenerate: int


class _Repair:
    """State of one orthogonality repair: the cells, the nodes and the lattice."""

    def __init__(
        self,
        cells: list[list[int]],
        nodes: Nodes,
        origin: tuple[float, float],
        unit: float,
        max_fm_ortho: float,
        log: Callable[[str], object] | None,
        fixed: set[tuple[int, ...]] | None = None,
    ) -> None:
        self.fixed = fixed or set()
        self.cells = cells
        self.nodes = nodes
        self.origin = origin
        self.unit = unit
        self.max_fm_ortho = max_fm_ortho
        self.log = log
        self.xy = self.coordinates()
        # Length scale for the triangle centres, from the cells as they come in.
        self.h = typical_edge_length(self.xy, _padded(cells))

    def emit(self, msg: str) -> None:
        _log.info(msg)
        if self.log is not None:
            self.log(msg)

    def coordinates(self) -> NDArray[np.float64]:
        """Coordinates of all nodes, ``origin + index * unit``."""
        pts = np.asarray(self.nodes.points, dtype=np.float64).reshape(-1, 2)
        xy = np.empty_like(pts)
        xy[:, 0] = self.origin[0] + pts[:, 0] * self.unit
        xy[:, 1] = self.origin[1] + pts[:, 1] * self.unit
        return xy

    def add_node(self, ix: int, iy: int) -> int:
        number = self.nodes.add(ix, iy)
        if number >= len(self.xy):
            self.xy = self.coordinates()
        return number

    def centre(self, cell: list[int]) -> NDArray[np.float64]:
        return np.asarray(face_centres(self.xy, _padded([cell]), self.h)[0])

    def orth(self, a: int, b: int, cell1: list[int], cell2: list[int]) -> float:
        """FM orthogonality of edge ``a-b`` between two cells; ``inf`` if degenerate."""
        ev = (self.xy[b] - self.xy[a])[None, :]
        cv = (self.centre(cell2) - self.centre(cell1))[None, :]
        return float(orthogonality(ev, cv)[0])

    def bad_edges(self) -> tuple[list[BadEdge], _Stats]:
        """Return the internal edges that are not FM-orthogonal or degenerate, in edge order."""
        faces = _padded(self.cells)
        edges = unique_edges(faces)
        orth = edge_orthogonality(self.xy, faces, edges, self.h)
        internal = edges.face_count == 2
        degenerate = internal & np.isinf(orth)
        bad = internal & (degenerate | (orth > self.max_fm_ortho))
        if self.fixed:
            # Edges between two cells of the existing mesh are not ours to judge.
            keys = [tuple(sorted(c)) for c in self.cells]
            for i in np.flatnonzero(bad):
                f1, f2 = edges.faces[i]
                if keys[f1] in self.fixed and keys[f2] in self.fixed:
                    bad[i] = False
        finite = orth[internal & np.isfinite(orth)]
        stats = _Stats(int(internal.sum()), float(finite.max()) if finite.size else 0.0,
                       int(degenerate.sum()))  # fmt: skip
        found = [
            (int(edges.nodes[i, 0]), int(edges.nodes[i, 1]), int(edges.faces[i, 0]),
             int(edges.faces[i, 1]))
            for i in np.flatnonzero(bad)
        ]  # fmt: skip
        return found, stats

    def point(self, v: int) -> tuple[int, int]:
        return self.nodes.points[v]

    def ccw(self, cell: list[int]) -> list[int] | None:
        """Return the cell counterclockwise; None if it has zero area."""
        area2 = _twice_area([self.point(v) for v in cell])
        if area2 == 0:
            return None
        return list(cell) if area2 > 0 else [cell[0], *cell[:0:-1]]

    def flip(self, bad: list[BadEdge]) -> int:
        """Flip the shared diagonal of triangle pairs where that makes the edge orthogonal."""
        flips = 0
        touched: set[int] = set()
        for a, b, f1, f2 in bad:
            if f1 in touched or f2 in touched:
                continue  # changed already; judged again in the next pass
            c1, c2 = self.cells[f1], self.cells[f2]
            if len(c1) != 3 or len(c2) != 3:
                continue  # only triangle-triangle can flip
            c = next(v for v in c1 if v not in (a, b))
            d = next(v for v in c2 if v not in (a, b))
            # Convex: a and b lie on either side of the new diagonal c-d.
            (cx, cy), (dx, dy) = self.point(c), self.point(d)
            (ax, ay), (bx, by) = self.point(a), self.point(b)
            side_a = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
            side_b = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
            if side_a * side_b >= 0:
                continue
            new1 = self.ccw([c, d, a])
            new2 = self.ccw([c, d, b])
            if new1 is None or new2 is None:
                continue
            if not self.orth(c, d, new1, new2) <= self.max_fm_ortho:
                continue
            self.cells[f1], self.cells[f2] = new1, new2
            touched.update((f1, f2))
            flips += 1
        return flips

    def redecompose(self, bad: list[BadEdge]) -> int:
        """Emit the lattice square around a bad edge again in its canonical pattern.

        Later steps (head-to-toe regeneration of a neighbour) can add a split to a side
        after emission; resolving the hanging node then splits a fan triangle and makes
        a sliver. This finds the square whose triangles contain the bad edge (its
        nodes only corners and side midpoints, no centre, the triangles filling it)
        and emits it again for its split sides: one side or two adjacent sides give
        their fan, four sides four squares around a new centre node. Two opposite or
        three sides are left alone. Returns the number of squares redone.
        """
        cells_of_node: dict[int, set[int]] = {}
        for fi, cell in enumerate(self.cells):
            for v in cell:
                cells_of_node.setdefault(v, set()).add(fi)
        done: set[int] = set()
        fixed = 0
        for _a, _b, f1, f2 in bad:
            if f1 in done or f2 in done:
                continue
            seed = [f for f in (f1, f2) if len(self.cells[f]) == 3]
            if not seed:
                continue
            found = self._square_of(seed, cells_of_node, done)
            if found is None:
                continue
            group, (x0, y0, size), corners, mids = found
            m_s, m_e, m_n, m_w = mids
            split = sum(m is not None for m in mids)
            adjacent = split == 2 and (m_s is None or m_n is None) and (m_e is None or m_w is None)
            if split == 4:
                centre = self.add_node(x0 + size // 2, y0 + size // 2)
                new_cells = square_cells(corners, m_s, m_e, m_n, m_w, centre)
                kind = "1 (four squares and centre)"
            elif split == 1 or adjacent:
                new_cells = square_cells(corners, m_s, m_e, m_n, m_w)
                kind = "2 (fan)" if split == 1 else "5 (corner fan)"
            else:
                continue  # two opposite or three sides: resolved at emission time
            if {tuple(sorted(self.cells[f])) for f in group} == {
                tuple(sorted(c)) for c in new_cells
            }:
                continue  # already canonical
            slots = sorted(group)
            for slot, cell in zip(slots, new_cells, strict=False):
                self.cells[slot] = list(cell)
            for slot in slots[len(new_cells) :]:
                self.cells[slot] = []
            self.cells.extend(list(c) for c in new_cells[len(slots) :])
            done.update(group)
            fixed += 1
            ox, oy = self.origin
            sides = "".join(
                s for s, m in zip("ENSW", (m_e, m_n, m_s, m_w), strict=True) if m is not None
            )
            self.emit(
                f"[QuadRef]   redecomposition @ ({ox + (x0 + size / 2) * self.unit:.1f},"
                f"{oy + (y0 + size / 2) * self.unit:.1f}): parent square "
                f"{size * self.unit:.0f}x{size * self.unit:.0f} m, splits={sides} -> "
                f"{len(group)} triangles replaced by canonical case {kind}"
            )
        if fixed:
            self.cells[:] = [c for c in self.cells if c]
        return fixed

    def _square_of(
        self, seed: list[int], cells_of_node: dict[int, set[int]], done: set[int]
    ) -> (
        tuple[set[int], tuple[int, int, int], tuple[int, int, int, int], tuple[int | None, ...]]
        | None
    ):
        """Find the lattice square that the seed triangles and their neighbours fill.

        The square's size follows from the bounding box of the seed triangles and its
        candidate origins lie against that box; each candidate is verified strictly.
        """
        pts = [self.point(v) for f in seed for v in self.cells[f]]
        bx0, bx1 = min(p[0] for p in pts), max(p[0] for p in pts)
        by0, by1 = min(p[1] for p in pts), max(p[1] for p in pts)
        size = max(bx1 - bx0, by1 - by0)
        if size <= 0:
            return None
        for x0, y0 in sorted({(cx, cy) for cx in (bx0, bx1 - size) for cy in (by0, by1 - size)}):
            box = (x0, y0, x0 + size, y0 + size)
            if not all(self._inside(f, box) for f in seed):
                continue
            group = self._grow(set(seed), box, cells_of_node, done)
            if group is None:
                continue
            nodes_in = {v for f in group for v in self.cells[f]}
            x1, y1 = x0 + size, y0 + size
            corners_or_none = [
                self._node_at(nodes_in, x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            ]
            if any(v is None for v in corners_or_none):
                continue
            corners = (
                corners_or_none[0],
                corners_or_none[1],
                corners_or_none[2],
                corners_or_none[3],
            )
            if size % 2:
                mids: tuple[int | None, ...] = (None, None, None, None)
                centre = None
            else:
                xm, ym = x0 + size // 2, y0 + size // 2
                mids = (
                    self._node_at(nodes_in, xm, y0),
                    self._node_at(nodes_in, x0 + size, ym),
                    self._node_at(nodes_in, xm, y0 + size),
                    self._node_at(nodes_in, x0, ym),
                )
                centre = self._node_at(nodes_in, xm, ym)
            expected = {*corners} | {m for m in mids if m is not None}
            if centre is not None or nodes_in != expected:
                continue  # extra or missing nodes: not a canonical square
            if (
                sum(abs(_twice_area([self.point(v) for v in self.cells[f]])) for f in group)
                != 2 * size * size
            ):
                continue  # the triangles do not fill the square exactly
            return group, (x0, y0, size), corners, mids  # type: ignore[return-value]
        return None

    def _inside(self, f: int, box: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = box
        return all(
            x0 <= x <= x1 and y0 <= y <= y1 for x, y in (self.point(v) for v in self.cells[f])
        )

    def _grow(
        self,
        group: set[int],
        box: tuple[int, int, int, int],
        cells_of_node: dict[int, set[int]],
        done: set[int],
    ) -> set[int] | None:
        """Add the triangles that share a node with the group and lie inside the box."""
        for _ in range(16):
            grown = False
            candidates: set[int] = set()
            for f in group:
                for v in self.cells[f]:
                    candidates |= cells_of_node.get(v, set())
            for f in candidates:
                if f in group or f in done or not self.cells[f] or not self._inside(f, box):
                    continue
                if len(self.cells[f]) != 3:
                    return None  # a quad inside the square: not a fan
                group.add(f)
                grown = True
            if not grown:
                break
        return group

    def _node_at(self, nodes_in: set[int], ix: int, iy: int) -> int | None:
        v = self.nodes.get(ix, iy)
        return v if v is not None and v in nodes_in else None

    def bad_points(self, bad: list[BadEdge]) -> list[tuple[float, float, float | None]]:
        points = []
        for a, b, f1, f2 in bad:
            o = self.orth(a, b, self.cells[f1], self.cells[f2])
            mx, my = (self.xy[a] + self.xy[b]) * 0.5
            points.append((float(mx), float(my), None if np.isinf(o) else o))
        return points


def repair_orthogonality(
    cells: list[list[int]],
    nodes: Nodes,
    origin: tuple[float, float],
    unit: float,
    max_passes: int = 10,
    max_fm_ortho: float = MAX_FM_ORTHOGONALITY,
    log: Callable[[str], object] | None = None,
    fixed: set[tuple[int, ...]] | None = None,
) -> list[list[int]]:
    """Make every internal edge FM-orthogonal, or raise.

    Parameters
    ----------
    cells : list of list of int
        The cells; changed in place.
    nodes : Nodes
        The nodes as lattice points; a redecomposed square may add its centre.
    origin : tuple of float
        Lattice point of index ``(0, 0)``.
    unit : float
        Lattice unit.
    max_passes : int
        Passes of flips and redecompositions before giving up.
    max_fm_ortho : float
        Highest accepted FM orthogonality.
    log : callable, optional
        Receives a line per pass and repair.
    fixed : set of tuple of int, optional
        Cells of an existing mesh (sorted node numbers): an edge between two of them
        is not judged, so refine does not fail on what it did not change.

    Returns
    -------
    list of list of int
        The repaired cells (the same list).

    Raises
    ------
    OrthogonalityRepairError
        When invalid internal edges remain; it carries their locations.

    Notes
    -----
    All topology works on integer lattice indices (exact convexity, areas and
    centre nodes); only the FM measure uses coordinates, ``origin + index * unit``.
    Per pass the invalid edges are flipped where that helps; when no flip helps, the
    lattice squares around them are redecomposed.
    """
    state = _Repair(cells, nodes, origin, unit, max_fm_ortho, log, fixed)
    state.emit(
        f"[QuadRef] orthogonality check start: {len(cells)} cells, threshold orth_e <= "
        f"{max_fm_ortho:.2f} (FM edge orthogonality, circumcentre in triangle)."
    )
    flips_total = blocks_total = 0
    bad: list[BadEdge] = []
    for pass_i in range(max_passes + 1):
        bad, stats = state.bad_edges()
        state.emit(
            f"[QuadRef] pass {pass_i + 1}: {stats.n_internal} internal edges, {len(bad)} invalid "
            f"(orth_e > {max_fm_ortho:.2f}), worst orth_e = {stats.worst:.3f}, "
            f"{stats.n_degenerate} degenerate."
        )
        if not bad:
            state.emit(
                f"[QuadRef] orthogonality OK: 0 invalid edges after {pass_i + 1} pass(es); "
                f"{flips_total} edge flip(s) and {blocks_total} redecomposition(s) in total."
            )
            return cells
        if pass_i == max_passes:
            state.emit(
                f"[QuadRef] max_passes ({max_passes}) reached with {len(bad)} invalid edge(s)."
            )
            break
        for x, y, o in state.bad_points(bad[:5]):
            state.emit(
                f"[QuadRef]   invalid edge @ ({x:.1f},{y:.1f}) "
                f"orth_e={'n/a' if o is None else f'{o:.3f}'}"
            )
        if len(bad) > 5:
            state.emit(f"[QuadRef]   ... and {len(bad) - 5} more invalid edge(s)")
        flips = state.flip(bad)
        flips_total += flips
        blocks = state.redecompose(bad) if flips == 0 else 0
        blocks_total += blocks
        state.emit(
            f"[QuadRef] pass {pass_i + 1} repair: {flips} edge flip(s) and {blocks} "
            f"redecomposition(s) applied to {len(bad)} invalid edge(s)."
        )
        if flips == 0 and blocks == 0:
            state.emit(f"[QuadRef] no further progress ({len(bad)} edges cannot be repaired).")
            break

    points = state.bad_points(bad)
    examples = ", ".join(
        f"({x:.1f},{y:.1f}) orth_e={'n/a' if o is None else f'{o:.2f}'}" for x, y, o in points[:5]
    )
    raise OrthogonalityRepairError(
        f"orthogonality repair did not close: {len(bad)} invalid internal edge(s) remain "
        f"(orth_e > {max_fm_ortho:.2f}; e.g. {examples}). They cannot be repaired with an "
        f"edge flip (e.g. triangle-quad or not convex); the mesh is not valid for D-Flow FM.",
        bad_points=points,
    )


def _padded(cells: list[list[int]]) -> NDArray[np.int64]:
    """Cells as a face array padded with -1."""
    width = max((len(c) for c in cells), default=3)
    faces = np.full((len(cells), width), -1, dtype=np.int64)
    for i, cell in enumerate(cells):
        faces[i, : len(cell)] = cell
    return faces


def _twice_area(points: list[tuple[int, int]]) -> int:
    """Twice the signed area of a polygon with integer corners."""
    n = len(points)
    return sum(
        points[k][0] * points[(k + 1) % n][1] - points[(k + 1) % n][0] * points[k][1]
        for k in range(n)
    )
