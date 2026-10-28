"""Quadtree engine, taken over from the generator of qgis_dhydamo ``9a324bc``.

The engine is a one-to-one port; issues 11 and 12 split it into one module per step.
The changes against the original are deliberate and listed in the changelog:

- transition rows arrive per step from coarse to fine and are validated by the caller
  (the original took them from fine to coarse and counted 0 as 1);
- the highest refinement level follows from the refinement areas, without a minimum
  cell size (the original capped the levels so that the smallest cell was >= 1 m);
- no code for shapely < 2 and no explicit refine zones without transition rows;
- errors are raised instead of being logged and skipped;
- messages are in English.

Strategy
--------
All cell coordinates are integers in units of the smallest cell
(``cell_size / 2**levels``). Integer keys make dictionary lookups exact and fast;
coordinates in metres are computed once, at the end, as ``origin + index * unit``.

- STRtree queries for refinement areas and water areas, prepared geometries for the
  repeated intersects tests.
- Candidate pruning along the quadtree: children inherit only the candidates that
  their parent touches. A cell completely inside an area or transition zone fixes
  that level as lower bound for its whole subtree, which needs no further geometry.
- 2:1 balancing, so neighbouring cells differ at most one level.
- Transition patterns for 0, 1, 2 adjacent, 2 opposite and 3-4 split sides, each with
  its own decomposition into quads and triangles.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from typing import NamedTuple, cast

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.prepared import PreparedGeometry, prep

from refined_quads._orthogonality import MAX_FM_ORTHOGONALITY, _rgfgrid_centres
from refined_quads.errors import Cancelled, InputError, OrthogonalityRepairError

_log = logging.getLogger(__name__)

Key = tuple[int, int, int, int]
Leaf = tuple[int, int, int, int, int]
BadEdge = tuple[int, int, int, int]
Progress = Callable[[int | None, str], object]
Float2 = NDArray[np.float64]

WATER_REMOVAL_RULES = ("centre", "inside", "touches")


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


# ---------------------------------------------------------------------------
# FM orthogonality, scalar helpers for the repair (see _orthogonality.py)
# ---------------------------------------------------------------------------


def _closest_point_on_segment(p: Float2, a: Float2, b: Float2) -> Float2:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0.0:
        return a
    t = float((p - a) @ ab) / denom
    t = max(0.0, min(1.0, t))
    return np.asarray(a + t * ab)


def _closest_point_in_triangle(p: Float2, a: Float2, b: Float2, c: Float2) -> Float2:
    """Project point p onto triangle abc (p itself when it lies inside)."""

    def area2(u: Float2, v: Float2, w: Float2) -> float:
        return float((v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]))

    s1 = area2(p, a, b)
    s2 = area2(p, b, c)
    s3 = area2(p, c, a)
    has_neg = (s1 < -1e-12) or (s2 < -1e-12) or (s3 < -1e-12)
    has_pos = (s1 > 1e-12) or (s2 > 1e-12) or (s3 > 1e-12)
    if not (has_neg and has_pos):
        return p
    p_ab = _closest_point_on_segment(p, a, b)
    p_bc = _closest_point_on_segment(p, b, c)
    p_ca = _closest_point_on_segment(p, c, a)
    d_ab = float((p - p_ab) @ (p - p_ab))
    d_bc = float((p - p_bc) @ (p - p_bc))
    d_ca = float((p - p_ca) @ (p - p_ca))
    if d_ab <= d_bc and d_ab <= d_ca:
        return p_ab
    if d_bc <= d_ca:
        return p_bc
    return p_ca


def _raw_circumcenter(a: Float2, b: Float2, c: Float2, eps_d: float) -> Float2 | None:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < eps_d:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.array([ux, uy], dtype=np.float64)


def _rgfgrid_center(a: Float2, b: Float2, c: Float2, h: float) -> Float2:
    """RGFGRID cell centre: circumcentre clipped into the triangle, centroid if degenerate."""
    centroid = np.asarray((a + b + c) / 3.0)
    cc = _raw_circumcenter(a, b, c, 1e-12 * (h * h))
    if cc is None:
        return centroid
    if np.linalg.norm(cc - centroid) > 50.0 * h:
        return centroid  # nearly collinear: unstable
    return _closest_point_in_triangle(cc, a, b, c)


class RawRepairError(OrthogonalityRepairError):
    """Repair failure with the raw nodes and cells, for the caller to build the raw mesh."""

    def __init__(
        self,
        message: str,
        bad_points: list[tuple[float, float, float | None]],
        vertices: NDArray[np.float64],
        cells: list[list[int]],
        grid_start: tuple[float, float],
    ) -> None:
        super().__init__(message, bad_points)
        self.vertices = vertices
        self.cells = cells
        self.grid_start = grid_start


class _BadEdgeStats(NamedTuple):
    n_internal: int
    worst: float
    n_degen: int


def orthogonality_repair(
    vertices: NDArray[np.float64],
    cells: list[list[int]],
    max_passes: int = 10,
    max_fm_ortho: float = MAX_FM_ORTHOGONALITY,
    log: Callable[[str], object] | None = None,
) -> tuple[NDArray[np.float64], list[list[int]]]:
    """Verify and repair the FM orthogonality of the mesh; iterate until valid.

    An internal edge (shared by exactly two faces) is invalid when its FM
    orthogonality exceeds ``max_fm_ortho`` (see ``_orthogonality.py``). Repairs, per
    invalid edge:

    1. Edge flip (triangle-triangle): replace the shared diagonal by the other one;
       accepted only if convex and the new edge is orthogonal enough.
    2. Square redecomposition: a transition square whose triangles were split after
       emission (so that a sliver appeared) is emitted again in its canonical pattern;
       with four split sides that is four squares and a new centre node.

    Passes repeat until no invalid edge is left.

    Raises
    ------
    OrthogonalityRepairError
        When invalid internal edges remain; it carries their locations.
    """
    vertices = np.asarray(vertices, dtype=np.float64)

    def _emit(msg: str) -> None:
        _log.info(msg)
        if log is not None:
            log(msg)

    # Centre nodes of redecomposed squares; during the passes they get an index
    # >= len(vertices) and are appended at the end.
    extra_vertices: list[Float2] = []

    def _vxy(idx: int) -> Float2:
        n0 = len(vertices)
        return vertices[idx] if idx < n0 else extra_vertices[idx - n0]

    def _add_vertex(xy: tuple[float, float]) -> int:
        extra_vertices.append(np.asarray(xy, dtype=np.float64))
        return len(vertices) + len(extra_vertices) - 1

    # Typical edge length (median) as scale for the RGFGRID centre.
    elen_all = [
        math.hypot(*(vertices[cell[(k + 1) % len(cell)]] - vertices[cell[k]]))
        for cell in cells
        for k in range(len(cell))
    ]
    elen = np.asarray([e for e in elen_all if e > 0.0], dtype=np.float64)
    h = float(np.median(elen)) if elen.size else 1.0
    if not math.isfinite(h) or h <= 0.0:
        h = 1.0

    def _face_cc(cell: list[int]) -> Float2:
        pts = np.array([_vxy(v) for v in cell])
        if len(cell) == 3:
            return _rgfgrid_center(pts[0], pts[1], pts[2], h)
        return np.asarray(pts.mean(axis=0))

    def _edge_orth(a: int, b: int, cc1: Float2, cc2: Float2) -> float | None:
        ev = _vxy(b) - _vxy(a)
        el = math.hypot(ev[0], ev[1])
        vv = np.asarray(cc2, float) - np.asarray(cc1, float)
        vl = math.hypot(vv[0], vv[1])
        if el <= 0.0 or vl <= 0.0:
            return None
        return abs(float((ev[0] * vv[0] + ev[1] * vv[1]) / (el * vl)))

    def _edge_ok(a: int, b: int, cc1: Float2, cc2: Float2) -> bool:
        o = _edge_orth(a, b, cc1, cc2)
        if o is None:
            return False  # degenerate or coinciding centres: invalid
        return o <= max_fm_ortho

    def _ccw(tri: list[int]) -> list[int] | None:
        (x1, y1), (x2, y2), (x3, y3) = (_vxy(v) for v in tri)
        area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if abs(area2) < 1e-9:
            return None
        return tri if area2 > 0 else [tri[0], tri[2], tri[1]]

    def _quad_ccw(quad: list[int]) -> list[int] | None:
        pts = [_vxy(v) for v in quad]
        area2 = sum(
            pts[k][0] * pts[(k + 1) % 4][1] - pts[(k + 1) % 4][0] * pts[k][1] for k in range(4)
        )
        if abs(area2) < 1e-9:
            return None
        return list(quad) if area2 > 0 else list(quad[::-1])

    def _inside(f: int, x0: float, y0: float, x1: float, y1: float) -> bool:
        return all(
            x0 - 1e-9 <= p[0] <= x1 + 1e-9 and y0 - 1e-9 <= p[1] <= y1 + 1e-9
            for p in (_vxy(v) for v in cells[f])
        )

    def _at(nodes: set[int], px: float, py: float) -> int | None:
        for v in nodes:
            p = _vxy(v)
            if abs(p[0] - px) < 1e-6 and abs(p[1] - py) < 1e-6:
                return v
        return None

    def _tri_area(cell: list[int]) -> float:
        p = [_vxy(v) for v in cell]
        return float(
            0.5
            * abs(
                (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])
            )
        )

    def _repair_square_redecompose(bad_edges: list[BadEdge]) -> int:
        """Emit the parent square around a bad edge again in its canonical pattern.

        Every cell is a square at heart; only its decomposition depends on which sides
        have a split (midpoint node), in five canonical cases: 4 sides split (four
        squares and a centre), 1 side (fan from the split to the opposite corners),
        2 opposite or 3 sides (resolved at emission time; skipped here), and 2
        adjacent sides (fan joining the two splits). Later steps (head-to-toe
        regeneration of a neighbour) can add a split to a side after emission; the
        hanging node is then resolved by splitting a fan triangle, which gives a
        sliver. This reconstructs the parent square, determines its split sides and
        emits the canonical decomposition again.

        Returns the number of repaired squares; mutates ``cells`` (an empty list marks
        a removed cell until compaction) and may register a centre node.
        """
        node_faces: dict[int, set[int]] = {}
        for fi, cell in enumerate(cells):
            for v in cell:
                node_faces.setdefault(v, set()).add(fi)

        done_faces: set[int] = set()
        n_fixed = 0
        for _a, _b, f1, f2 in bad_edges:
            if f1 in done_faces or f2 in done_faces:
                continue
            seed = [f for f in (f1, f2) if len(cells[f]) == 3]
            if not seed:
                continue
            # Hypothesis: the parent square. Its size follows from the bounding box of
            # the seed triangles and the candidate origins lie against that box. Each
            # candidate is verified strictly (node pattern and complete tiling).
            seed_pts = [_vxy(v) for f in seed for v in cells[f]]
            bx0 = min(p[0] for p in seed_pts)
            bx1 = max(p[0] for p in seed_pts)
            by0 = min(p[1] for p in seed_pts)
            by1 = max(p[1] for p in seed_pts)
            size = max(bx1 - bx0, by1 - by0)
            if size <= 0:
                continue
            cand_origins = {(cx, cy) for cx in (bx0, bx1 - size) for cy in (by0, by1 - size)}
            hit = None
            for x0, y0 in sorted(cand_origins):
                x1 = x0 + size
                y1 = y0 + size
                xm = (x0 + x1) / 2.0
                ym = (y0 + y1) / 2.0
                if not all(_inside(f, x0, y0, x1, y1) for f in seed):
                    continue
                # Grow: faces that share a node with the group and lie completely in
                # the candidate square (triangles only).
                group = set(seed)
                ok = True
                for _ in range(16):
                    grown = False
                    cand: set[int] = set()
                    for f in group:
                        for v in cells[f]:
                            cand |= node_faces.get(v, set())
                    for f in cand:
                        if f in group or f in done_faces or not cells[f]:
                            continue
                        if not _inside(f, x0, y0, x1, y1):
                            continue
                        if len(cells[f]) != 3:
                            ok = False  # a quad inside the square: not a fan
                            break
                        group.add(f)
                        grown = True
                    if not ok or not grown:
                        break
                if not ok:
                    continue
                nodes = {v for f in group for v in cells[f]}
                bl = _at(nodes, x0, y0)
                br = _at(nodes, x1, y0)
                tr = _at(nodes, x1, y1)
                tl = _at(nodes, x0, y1)
                if bl is None or br is None or tr is None or tl is None:
                    continue
                m_s = _at(nodes, xm, y0)
                m_e = _at(nodes, x1, ym)
                m_n = _at(nodes, xm, y1)
                m_w = _at(nodes, x0, ym)
                cen = _at(nodes, xm, ym)
                expected = {bl, br, tr, tl} | {m for m in (m_s, m_e, m_n, m_w) if m is not None}
                if cen is not None or nodes != expected:
                    continue  # extra or missing nodes: not a canonical square
                if abs(sum(_tri_area(cells[f]) for f in group) - size * size) > 1e-6 * size * size:
                    continue  # the triangles do not fill the square exactly
                hit = (group, bl, br, tr, tl, m_s, m_e, m_n, m_w, xm, ym)
                break
            if hit is None:
                continue
            group, bl, br, tr, tl, m_s, m_e, m_n, m_w, xm, ym = hit
            sides = {
                side
                for side, m in (("S", m_s), ("E", m_e), ("N", m_n), ("W", m_w))
                if m is not None
            }
            new_cells: list[list[int]]
            if len(sides) == 1:  # case 2: fan from the split
                if m_s is not None:
                    new_cells = [[bl, m_s, tl], [m_s, br, tr], [m_s, tr, tl]]
                elif m_e is not None:
                    new_cells = [[br, m_e, bl], [m_e, tr, tl], [m_e, tl, bl]]
                elif m_n is not None:
                    new_cells = [[tl, m_n, bl], [m_n, tr, br], [m_n, br, bl]]
                else:
                    assert m_w is not None
                    new_cells = [[tl, m_w, tr], [m_w, bl, br], [m_w, br, tr]]
            elif len(sides) == 2 and sides not in ({"N", "S"}, {"E", "W"}):
                # case 5: fan joining the two splits and towards the opposite corner
                if m_s is not None and m_e is not None:
                    new_cells = [[br, m_s, m_e], [bl, m_s, tl], [m_s, m_e, tl], [m_e, tr, tl]]
                elif m_s is not None and m_w is not None:
                    new_cells = [[bl, m_s, m_w], [m_s, br, tr], [m_s, tr, m_w], [m_w, tr, tl]]
                elif m_n is not None and m_e is not None:
                    new_cells = [[tr, m_n, m_e], [m_n, tl, bl], [m_n, bl, m_e], [m_e, bl, br]]
                else:
                    assert m_n is not None
                    assert m_w is not None
                    new_cells = [[tl, m_w, m_n], [m_w, bl, br], [m_w, br, m_n], [m_n, br, tr]]
            elif len(sides) == 4:  # case 1: four squares and a centre
                assert m_s is not None
                assert m_e is not None
                assert m_n is not None
                assert m_w is not None
                cnode = _add_vertex((xm, ym))
                quads = [
                    _quad_ccw([bl, m_s, cnode, m_w]),
                    _quad_ccw([m_s, br, m_e, cnode]),
                    _quad_ccw([cnode, m_e, tr, m_n]),
                    _quad_ccw([m_w, cnode, m_n, tl]),
                ]
                if any(q is None for q in quads):
                    continue
                new_cells = [q for q in quads if q is not None]
            else:
                continue  # case 3/4 (2 opposite or 3 sides): resolved at emission time
            # Already canonical: skip (avoids endless re-emission without effect).
            old_keys = {tuple(sorted(cells[f])) for f in group}
            new_keys = {tuple(sorted(c)) for c in new_cells}
            if old_keys == new_keys:
                continue
            # Replace: reuse slots; mark the surplus as removed or append.
            slots = sorted(group)
            for slot, ncell in zip(slots, new_cells, strict=False):
                cells[slot] = list(ncell)
            for slot in slots[len(new_cells) :]:
                cells[slot] = []
            cells.extend(list(ncell) for ncell in new_cells[len(slots) :])
            done_faces.update(group)
            n_fixed += 1
            kind = (
                "1 (four squares and centre)"
                if len(sides) == 4
                else ("2 (fan)" if len(sides) == 1 else "5 (corner fan)")
            )
            _emit(
                f"[QuadRef]   redecomposition @ ({xm:.1f},{ym:.1f}): parent square "
                f"{size:.0f}x{size:.0f} m, splits={''.join(sorted(sides))} -> {len(group)} "
                f"triangles replaced by canonical case {kind}"
            )
        if n_fixed:
            cells[:] = [c for c in cells if c]
        return n_fixed

    def _all_xy() -> NDArray[np.float64]:
        if extra_vertices:
            return np.vstack([vertices, *extra_vertices])
        return vertices

    def _compute_bad_edges() -> tuple[list[BadEdge], _BadEdgeStats]:
        """Find the internal edges that are not FM-orthogonal, vectorised over all cells."""
        xy = _all_xy()
        ncell = len(cells)
        lens = np.fromiter((len(c) for c in cells), dtype=np.int64, count=ncell)
        centers = np.empty((ncell, 2), dtype=np.float64)
        tri_mask = lens == 3
        quad_mask = lens == 4
        other_mask = ~(tri_mask | quad_mask)
        if tri_mask.any():
            tri_idx = np.where(tri_mask)[0]
            tri = np.array([cells[i] for i in tri_idx], dtype=np.int64)
            centers[tri_idx] = _rgfgrid_centres(xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]], h)
        if quad_mask.any():
            qi = np.where(quad_mask)[0]
            quad = np.array([cells[i] for i in qi], dtype=np.int64)
            centers[qi] = xy[quad].mean(axis=1)
        for i in np.where(other_mask)[0]:  # rare (> 4 nodes): centroid
            centers[i] = xy[cells[i]].mean(axis=0)
        fa_l: list[int] = []
        ea_l: list[int] = []
        eb_l: list[int] = []
        for fi, c in enumerate(cells):
            n = len(c)
            for k in range(n):
                a = c[k]
                b = c[(k + 1) % n]
                fa_l.append(fi)
                if a < b:
                    ea_l.append(a)
                    eb_l.append(b)
                else:
                    ea_l.append(b)
                    eb_l.append(a)
        fa = np.asarray(fa_l, np.int64)
        ea = np.asarray(ea_l, np.int64)
        eb = np.asarray(eb_l, np.int64)
        order = np.lexsort((eb, ea))
        ea_s = ea[order]
        eb_s = eb[order]
        fa_s = fa[order]
        # An internal edge is a group of exactly two faces; boundary (1) and
        # non-manifold (> 2) groups are skipped.
        n = ea_s.size
        if n == 0:
            return [], _BadEdgeStats(0, 0.0, 0)
        new_grp = np.empty(n, bool)
        new_grp[0] = True
        new_grp[1:] = (ea_s[1:] != ea_s[:-1]) | (eb_s[1:] != eb_s[:-1])
        grp_start = np.where(new_grp)[0]
        grp_size = np.diff(np.append(grp_start, n))
        pair_start = grp_start[grp_size == 2]
        if pair_start.size == 0:
            return [], _BadEdgeStats(0, 0.0, 0)
        a_ = ea_s[pair_start]
        b_ = eb_s[pair_start]
        f1 = fa_s[pair_start]
        f2 = fa_s[pair_start + 1]
        ev = xy[b_] - xy[a_]
        el = np.hypot(ev[:, 0], ev[:, 1])
        vv = centers[f2] - centers[f1]
        vl = np.hypot(vv[:, 0], vv[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            orth = np.abs((ev[:, 0] * vv[:, 0] + ev[:, 1] * vv[:, 1]) / (el * vl))
        degen = (el <= 0.0) | (vl <= 0.0) | ~np.isfinite(orth)
        bad = degen | (orth > max_fm_ortho)
        finite_orth = orth[np.isfinite(orth)]
        stats = _BadEdgeStats(
            n_internal=int(pair_start.size),
            worst=float(finite_orth.max()) if finite_orth.size else 0.0,
            n_degen=int(degen.sum()),
        )
        return [(int(a_[i]), int(b_[i]), int(f1[i]), int(f2[i])) for i in np.where(bad)[0]], stats

    n_flips_total = 0
    n_blocks_total = 0
    bad_edges: list[BadEdge] = []
    _emit(
        f"[QuadRef] orthogonality check start: {len(cells)} cells, threshold orth_e <= "
        f"{max_fm_ortho:.2f} (FM edge orthogonality, circumcentre in triangle)."
    )
    for pass_i in range(max_passes + 1):
        bad_edges, stats = _compute_bad_edges()
        n_bad = len(bad_edges)
        _emit(
            f"[QuadRef] pass {pass_i + 1}: {stats.n_internal} internal edges, {n_bad} invalid "
            f"(orth_e > {max_fm_ortho:.2f}), worst orth_e = {stats.worst:.3f}, "
            f"{stats.n_degen} degenerate."
        )
        if not bad_edges:
            _emit(
                f"[QuadRef] orthogonality OK: 0 invalid edges after {pass_i + 1} pass(es); "
                f"{n_flips_total} edge flip(s) and {n_blocks_total} redecomposition(s) in total."
            )
            if extra_vertices:
                return np.vstack([vertices, *extra_vertices]), cells
            return vertices, cells
        if pass_i == max_passes:
            _emit(f"[QuadRef] max_passes ({max_passes}) reached with {n_bad} invalid edge(s).")
            break

        for a, b, f1, f2 in bad_edges[:5]:
            o = _edge_orth(a, b, _face_cc(cells[f1]), _face_cc(cells[f2]))
            mx = (_vxy(a)[0] + _vxy(b)[0]) * 0.5
            my = (_vxy(a)[1] + _vxy(b)[1]) * 0.5
            kinds = f"{len(cells[f1])}n-{len(cells[f2])}n"
            orth_text = "n/a" if o is None else f"{o:.3f}"
            _emit(
                f"[QuadRef]   invalid edge @ ({mx:.1f},{my:.1f}) orth_e={orth_text} "
                f"between {kinds} faces"
            )
        if n_bad > 5:
            _emit(f"[QuadRef]   ... and {n_bad - 5} more invalid edge(s)")

        n_flips = 0
        touched: set[int] = set()
        for a, b, f1, f2 in bad_edges:
            if f1 in touched or f2 in touched:
                continue  # face already changed; judged again in the next pass
            c1 = cells[f1]
            c2 = cells[f2]
            if len(c1) != 3 or len(c2) != 3:
                continue  # only triangle-triangle can flip
            c = next(v for v in c1 if v not in (a, b))
            d = next(v for v in c2 if v not in (a, b))
            # Convexity: a and b must lie on either side of the new diagonal c-d.
            cx, cy = _vxy(c)
            dxv = _vxy(d) - _vxy(c)
            sa = dxv[0] * (_vxy(a)[1] - cy) - dxv[1] * (_vxy(a)[0] - cx)
            sb = dxv[0] * (_vxy(b)[1] - cy) - dxv[1] * (_vxy(b)[0] - cx)
            if sa * sb >= 0.0:
                continue
            new1 = _ccw([c, d, a])
            new2 = _ccw([c, d, b])
            if new1 is None or new2 is None:
                continue
            if not _edge_ok(c, d, _face_cc(new1), _face_cc(new2)):
                continue
            cells[f1] = new1
            cells[f2] = new2
            touched.add(f1)
            touched.add(f2)
            n_flips += 1
        n_flips_total += n_flips

        # When flips are exhausted, try the structural square redecomposition.
        n_blocks = 0
        if n_flips == 0:
            n_blocks = _repair_square_redecompose(bad_edges)
            n_blocks_total += n_blocks

        _emit(
            f"[QuadRef] pass {pass_i + 1} repair: {n_flips} edge flip(s) and {n_blocks} "
            f"redecomposition(s) applied to {n_bad} invalid edge(s)."
        )
        if n_flips == 0 and n_blocks == 0:
            _emit(f"[QuadRef] no further progress ({n_bad} edges cannot be repaired).")
            break

    bad_points = []
    for a, b, f1, f2 in bad_edges:
        o = _edge_orth(a, b, _face_cc(cells[f1]), _face_cc(cells[f2]))
        mx = (_vxy(a)[0] + _vxy(b)[0]) * 0.5
        my = (_vxy(a)[1] + _vxy(b)[1]) * 0.5
        bad_points.append((float(mx), float(my), (None if o is None else float(o))))
    examples = ", ".join(
        f"({x:.1f},{y:.1f}) orth_e={'n/a' if o is None else f'{o:.2f}'}"
        for x, y, o in bad_points[:5]
    )
    raise OrthogonalityRepairError(
        f"orthogonality repair did not close: {len(bad_edges)} invalid internal edge(s) "
        f"remain (orth_e > {max_fm_ortho:.2f}; e.g. {examples}). They cannot be repaired "
        f"with an edge flip (e.g. triangle-quad or not convex); the mesh is not valid for "
        f"D-Flow FM.",
        bad_points=bad_points,
    )


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------


def generate_cells(
    domain: BaseGeometry,
    refinement_areas: Sequence[tuple[BaseGeometry, int]],
    *,
    cell_size: float,
    levels: int,
    transition_rows: Sequence[int],
    water_areas: Sequence[BaseGeometry] = (),
    water_removal: str = "centre",
    origin: tuple[float, float] | None = None,
    progress: Progress | None = None,
) -> tuple[NDArray[np.float64], list[list[int]], tuple[float, float]]:
    """Generate a refined quad mesh; the input is validated by the caller.

    Parameters
    ----------
    domain : shapely geometry
        Polygonal area the mesh covers.
    refinement_areas : sequence of (shapely geometry, int)
        Polygonal areas with the refinement level (1 to ``levels``) their cells reach.
    cell_size : float
        Base cell size.
    levels : int
        Highest refinement level.
    transition_rows : sequence of int
        Per step from coarse to fine (index = level of the step), at least 1 each.
    water_areas : sequence of shapely geometry
        Areas where no cells are kept.
    water_removal : {"centre", "inside", "touches"}
        Remove a cell when its centre lies in water, when it lies completely in
        water, or as soon as it touches water.
    origin : tuple of float, optional
        A lattice point; the base grid starts at the largest lattice point at or
        below the lower-left corner of the domain. By default a multiple of the cell
        size.
    progress : callable, optional
        Called with ``(percent or None, message)``; a truthy result cancels.

    Returns
    -------
    vertices : numpy.ndarray of float, shape (n, 2)
        Node coordinates (some nodes may be unused).
    cells : list of list of int
        Node indices per cell.
    origin : tuple of float
        Start of the base grid, a lattice point: the lower-left corner of the domain
        rounded down to the lattice.
    """
    max_levels = levels
    # buffer_rows_by_level[lvl]: transition rows for the step from lvl to lvl + 1.
    buffer_rows_by_level = list(transition_rows)

    # Integer grid unit: the smallest cell.
    unit = cell_size / (2**max_levels)
    base_i = 2**max_levels

    t_start = time.perf_counter()

    def _check_cancel(pct: int | None = None, msg: str | None = None) -> None:
        suffix = ""
        if pct is not None or msg is not None:
            suffix = f"  [{_fmt_elapsed(time.perf_counter() - t_start)}]"
            _log.debug("[QuadRef] %s%s%s", f"{pct}% " if pct is not None else "", msg or "", suffix)
        if progress is not None:
            msg_out = (msg + suffix) if msg else (msg or "")
            if progress(pct, msg_out):
                raise Cancelled("mesh generation cancelled by the progress callback")

    # -- domain -----------------------------------------------------------------
    bnd_shp = domain if domain.is_valid else domain.buffer(0)
    bnd_prep = prep(bnd_shp)
    minx, miny, maxx, maxy = bnd_shp.bounds

    if origin is None:
        ox = math.floor(minx / cell_size) * cell_size
        oy = math.floor(miny / cell_size) * cell_size
    else:
        ox = origin[0] + math.floor((minx - origin[0]) / cell_size) * cell_size
        oy = origin[1] + math.floor((miny - origin[1]) / cell_size) * cell_size

    def _to_real(ix: int, iy: int) -> tuple[float, float]:
        return ox + ix * unit, oy + iy * unit

    def _shp_cell(ix0: int, iy0: int, ix1: int, iy1: int) -> BaseGeometry:
        rx0, ry0 = _to_real(ix0, iy0)
        rx1, ry1 = _to_real(ix1, iy1)
        return shapely.box(rx0, ry0, rx1, ry1)

    # -- water (exclusion) ------------------------------------------------------
    water_list = [wg if wg.is_valid else wg.buffer(0) for wg in water_areas]
    water_tree = STRtree(water_list) if water_list else None

    # -- refinement areas -------------------------------------------------------
    # Simplify first: detail smaller than the cell size cannot be resolved by the
    # quadtree, so such a tolerance is invisible in the result. Prevents a polygon
    # with hundreds of thousands of vertices from stalling the zone buffering.
    simplify_tol = cell_size / 8.0

    def _simplify_refine_geom(geom: BaseGeometry) -> BaseGeometry:
        n_before = int(shapely.get_num_coordinates(geom))
        if n_before > 2000:
            simplified = geom.simplify(simplify_tol, preserve_topology=True)
            _log.debug(
                "[QuadRef] refinement geometry simplified (tolerance=%.2fm): %d -> ~%d vertices",
                simplify_tol,
                n_before,
                int(shapely.get_num_coordinates(simplified)),
            )
            return simplified
        return geom

    refine_polys = []
    for geom, level in refinement_areas:
        s = geom if geom.is_valid else geom.buffer(0)
        refine_polys.append((_simplify_refine_geom(s), level))

    # -- transition zones ---------------------------------------------------------
    buffered_polys: list[tuple[BaseGeometry, int, list[tuple[BaseGeometry, int]]]] = []
    n_refine_polys = len(refine_polys)
    if n_refine_polys:
        _check_cancel(1, f"Quad Grid: transition zones for {n_refine_polys} refinement areas...")

    for poly_i, (poly, target_lvl) in enumerate(refine_polys):
        if n_refine_polys > 20 and poly_i % max(1, n_refine_polys // 20) == 0:
            _check_cancel(
                1 + int(3 * poly_i / n_refine_polys),
                f"Quad Grid: transition zones {poly_i}/{n_refine_polys}...",
            )
        zones: list[tuple[BaseGeometry, int]] = []
        for lvl in range(target_lvl):
            cum = 0.0
            for inner_lvl in range(lvl, target_lvl):
                n = buffer_rows_by_level[inner_lvl]
                cum += n * (cell_size / (2 ** (inner_lvl + 1)))
            zones.append((poly.buffer(cum), lvl))
        buffered_polys.append((poly, target_lvl, zones))

    # -- candidates: refinement areas with prepared geometries --------------------
    # Intersecting the area gives its level; otherwise the finest transition zone it
    # touches gives that zone's level + 1. The outermost zone feeds the STRtree.
    cands: list[
        tuple[int, PreparedGeometry[BaseGeometry], list[tuple[PreparedGeometry[BaseGeometry], int]]]
    ] = []
    cand_outer = []
    for poly, target_lvl, zones in buffered_polys:
        zp = [(prep(buf_poly), lvl) for buf_poly, lvl in zones]
        cands.append((target_lvl, prep(poly), zp))
        cand_outer.append(zones[0][0] if zones else poly)
    cand_tree = STRtree(cand_outer) if cands else None

    def _eval_cell(
        cell_shp: BaseGeometry, base_lvl: int, cell_cands: tuple[int, ...]
    ) -> tuple[int, int, list[int]]:
        """Return the required level, the level fixed for all children and their candidates.

        A candidate that the cell does not touch at all (outermost zone included)
        cannot touch its smaller children either and is dropped.
        """
        req = base_lvl
        child_base = base_lvl
        keep = []
        for ci in cell_cands:
            level, p_main, zp = cands[ci]
            if p_main.intersects(cell_shp):
                req = max(req, level)
                if p_main.contains_properly(cell_shp):
                    child_base = max(child_base, level)
                else:
                    keep.append(ci)
            else:
                # Outside the area: the finest touching zone decides. Finer zones are
                # subsets of coarser ones, so the first hit in reversed order is the
                # finest, and untouched finer zones stay out of reach for children.
                for p_buf, lvl_candidate in reversed(zp):
                    if p_buf.intersects(cell_shp):
                        contrib = lvl_candidate + 1
                        req = max(req, contrib)
                        if p_buf.contains_properly(cell_shp):
                            child_base = max(child_base, contrib)
                        else:
                            keep.append(ci)
                        break
        return req, child_base, keep

    # -- base grid ----------------------------------------------------------------
    nx = math.ceil((maxx - ox) / cell_size)
    ny = math.ceil((maxy - oy) / cell_size)
    _check_cancel(5, f"Quad Grid: base grid ({nx}x{ny} = {nx * ny} cells)...")

    queue: list[tuple[int, int, int, int, int, int, tuple[int, ...]]] = []
    for gx in range(nx):
        for gy in range(ny):
            ix0 = gx * base_i
            iy0 = gy * base_i
            ix1 = ix0 + base_i
            iy1 = iy0 + base_i
            cell_shp = _shp_cell(ix0, iy0, ix1, iy1)
            if bnd_prep.intersects(cell_shp):
                if cand_tree is not None:
                    cands0 = tuple(sorted(int(i) for i in cand_tree.query(cell_shp)))
                else:
                    cands0 = ()
                queue.append((ix0, iy0, ix1, iy1, 0, 0, cands0))

    _check_cancel(10, f"Quad Grid: quadtree refinement ({len(queue)} base cells)...")

    # -- quadtree refinement --------------------------------------------------------
    leaf_cells: list[Leaf] = []
    processed = 0
    report_every = max(1, min(10_000, len(queue) // 20))
    # Progress by covered area: a base cell weighs 4**max_levels units, a cell at
    # level L 4**(max_levels - L); an area is done when its cell becomes a leaf.
    weight_total = len(queue) * (4**max_levels)
    weight_done = 0
    t_refine_start = time.perf_counter()

    while queue:
        ix0, iy0, ix1, iy1, cur_lvl, base_lvl, q_cands = queue.pop()
        if q_cands:
            req_lvl, child_base, child_cands_l = _eval_cell(
                _shp_cell(ix0, iy0, ix1, iy1), base_lvl, q_cands
            )
            child_cands = tuple(child_cands_l)
        else:
            # No candidates left: the required level is fixed for the whole subtree.
            req_lvl = base_lvl
            child_base = base_lvl
            child_cands = ()
        if cur_lvl < req_lvl and cur_lvl < max_levels:
            ixm = (ix0 + ix1) >> 1
            iym = (iy0 + iy1) >> 1
            queue.append((ix0, iy0, ixm, iym, cur_lvl + 1, child_base, child_cands))
            queue.append((ixm, iy0, ix1, iym, cur_lvl + 1, child_base, child_cands))
            queue.append((ix0, iym, ixm, iy1, cur_lvl + 1, child_base, child_cands))
            queue.append((ixm, iym, ix1, iy1, cur_lvl + 1, child_base, child_cands))
        else:
            leaf_cells.append((ix0, iy0, ix1, iy1, cur_lvl))
            weight_done += 4 ** (max_levels - cur_lvl)

        processed += 1
        if processed % report_every == 0:
            fraction = weight_done / weight_total
            pct = 10 + int(30 * fraction)
            refine_elapsed = time.perf_counter() - t_refine_start
            eta = ""
            if fraction > 0.02:
                remaining = max(0.0, refine_elapsed / fraction - refine_elapsed)
                eta = f"  ~{_fmt_elapsed(remaining)} left"
            _check_cancel(
                pct,
                f"Quad Grid: refining... {processed} processed, {len(queue)} queued, "
                f"{len(leaf_cells)} leaf cells{eta}",
            )

    # -- 2:1 balancing ---------------------------------------------------------------
    _check_cancel(40, "Quad Grid: 2:1 balancing...")

    cell_dict: dict[Key, int] = {
        (ix0, iy0, ix1, iy1): lvl for ix0, iy0, ix1, iy1, lvl in leaf_cells
    }

    def _find_ancestor(
        ax0: int, ay0: int, ax1: int, ay1: int
    ) -> tuple[Key, int, int, int, int, int] | None:
        iw = ax1 - ax0
        ih = ay1 - ay0
        cx = (ax0 + ax1) >> 1
        cy = (ay0 + ay1) >> 1
        for _ in range(max_levels + 1):
            iw <<= 1
            ih <<= 1
            px0 = (cx // iw) * iw
            py0 = (cy // ih) * ih
            pk = (px0, py0, px0 + iw, py0 + ih)
            if pk in cell_dict:
                return pk, px0, py0, px0 + iw, py0 + ih, cell_dict[pk]
        return None

    def _split(key: Key, new_lvl: int) -> list[Key]:
        sx0, sy0, sx1, sy1 = key
        sxm = (sx0 + sx1) >> 1
        sym = (sy0 + sy1) >> 1
        subs = [
            (sx0, sy0, sxm, sym),
            (sxm, sy0, sx1, sym),
            (sx0, sym, sxm, sy1),
            (sxm, sym, sx1, sy1),
        ]
        for sub in subs:
            cell_dict[sub] = new_lvl
        return subs

    def _neighbours(key: Key) -> list[Key]:
        """Same-size neighbour keys in the order N, S, E, W."""
        ix0, iy0, ix1, iy1 = key
        iw = ix1 - ix0
        ih = iy1 - iy0
        return [
            (ix0, iy1, ix1, iy1 + ih),
            (ix0, iy0 - ih, ix1, iy0),
            (ix1, iy0, ix1 + iw, iy1),
            (ix0 - iw, iy0, ix0, iy1),
        ]

    balance_queue = sorted(cell_dict.keys(), key=lambda k: -cell_dict[k])
    bal_iter = 0
    while balance_queue:
        bal_iter += 1
        if bal_iter % 2000 == 0:
            _check_cancel(msg=f"Quad Grid: 2:1 balancing... {len(balance_queue)} left")

        k = balance_queue.pop()
        if k not in cell_dict:
            continue
        lvl = cell_dict[k]
        for nk in _neighbours(k):
            if nk in cell_dict:
                n_lvl = cell_dict[nk]
                if n_lvl < lvl - 1:
                    del cell_dict[nk]
                    balance_queue.extend(_split(nk, n_lvl + 1))
            else:
                result = _find_ancestor(*nk)
                if result is not None:
                    pk, *_, p_lvl = result
                    if p_lvl < lvl - 1:
                        del cell_dict[pk]
                        balance_queue.extend(_split(pk, p_lvl + 1))

    # -- T-junction detection -------------------------------------------------------
    def _finer_exists(ix0: int, iy0: int, ix1: int, iy1: int, side: str) -> bool:
        hw = (ix1 - ix0) >> 1
        hh = (iy1 - iy0) >> 1
        if side == "N":
            return (ix0, iy1, ix0 + hw, iy1 + hh) in cell_dict or (
                ix0 + hw,
                iy1,
                ix1,
                iy1 + hh,
            ) in cell_dict
        if side == "S":
            return (ix0, iy0 - hh, ix0 + hw, iy0) in cell_dict or (
                ix0 + hw,
                iy0 - hh,
                ix1,
                iy0,
            ) in cell_dict
        if side == "E":
            return (ix1, iy0, ix1 + hw, iy0 + hh) in cell_dict or (
                ix1,
                iy0 + hh,
                ix1 + hw,
                iy1,
            ) in cell_dict
        return (ix0 - hw, iy0, ix0, iy0 + hh) in cell_dict or (
            ix0 - hw,
            iy0 + hh,
            ix0,
            iy1,
        ) in cell_dict

    # -- T-junction elimination: split cells with >= 3 finer neighbours ------------
    _check_cancel(43, "Quad Grid: T-junction elimination (three-side halving)...")

    def _free_side(s_n: bool, s_s: bool, s_e: bool) -> str:
        if not s_n:
            return "N"
        if not s_s:
            return "S"
        if not s_e:
            return "E"
        return "W"

    def _has_any_neighbor(key: Key, side: str) -> bool:
        ncoords = _neighbours(key)["NSEW".index(side)]
        if ncoords in cell_dict:
            return True
        return _find_ancestor(*ncoords) is not None

    def _check_and_enqueue(k: Key, tj_queue: deque[Key], in_queue: set[Key]) -> None:
        if k in cell_dict and k not in in_queue and cell_dict[k] < max_levels:
            in_queue.add(k)
            tj_queue.append(k)

    tj_queue: deque[Key] = deque()
    tj_in_queue: set[Key] = set()
    for k, lvl in cell_dict.items():
        if lvl < max_levels:
            tj_queue.append(k)
            tj_in_queue.add(k)

    tj_total = 0
    tj_processed = 0
    while tj_queue:
        k = tj_queue.popleft()
        tj_in_queue.discard(k)
        tj_processed += 1
        if k not in cell_dict:
            continue
        ix0, iy0, ix1, iy1 = k
        lvl = cell_dict[k]
        if lvl >= max_levels:
            continue

        s_n = _finer_exists(ix0, iy0, ix1, iy1, "N")
        s_s = _finer_exists(ix0, iy0, ix1, iy1, "S")
        s_e = _finer_exists(ix0, iy0, ix1, iy1, "E")
        s_w = _finer_exists(ix0, iy0, ix1, iy1, "W")
        n_sp = s_n + s_s + s_e + s_w
        if n_sp < 3:
            continue
        if n_sp == 3 and not _has_any_neighbor(k, _free_side(s_n, s_s, s_e)):
            continue

        del cell_dict[k]
        _split(k, lvl + 1)
        tj_total += 1

        for ncoords in _neighbours(k):
            _check_and_enqueue(ncoords, tj_queue, tj_in_queue)
            if ncoords not in cell_dict:
                res = _find_ancestor(*ncoords)
                if res is not None:
                    _check_and_enqueue(res[0], tj_queue, tj_in_queue)

        if tj_total % 200 == 0:
            _check_cancel(
                msg=f"Quad Grid: T-junction elimination, {tj_total} split, "
                f"{len(tj_queue)} queued..."
            )

    _log.debug(
        "[QuadRef] T-junction elimination: %d split (%d checked), %d cells",
        tj_total,
        tj_processed,
        len(cell_dict),
    )

    leaf_cells = [(ix0, iy0, ix1, iy1, lvl) for (ix0, iy0, ix1, iy1), lvl in cell_dict.items()]

    # -- clip to the domain and remove water ------------------------------------------
    _check_cancel(50, f"Quad Grid: {len(leaf_cells)} cells after balancing, clipping...")

    def _water_exclusion_mask(
        boxes: NDArray[np.object_], centres_x: Float2, centres_y: Float2
    ) -> NDArray[np.bool_]:
        """Mark the cells that the water removal rule removes (True = removed)."""
        excluded = np.zeros(len(boxes), dtype=bool)
        if water_tree is None or len(boxes) == 0:
            return excluded
        water_arr = np.asarray(water_list, dtype=object)
        if water_removal == "centre":
            centres = cast("NDArray[np.object_]", shapely.points(centres_x, centres_y))
            q_idx, w_idx = water_tree.query(centres, predicate="intersects")
            if len(q_idx):
                contains = shapely.contains(water_arr[w_idx], centres[q_idx])
                np.logical_or.at(excluded, q_idx[contains], True)
            return excluded
        # "intersects" as broad preselection (STRtree predicates were not reliably
        # direction-independent across versions), then the exact test.
        q_idx, w_idx = water_tree.query(boxes, predicate="intersects")
        if len(q_idx):
            if water_removal == "inside":
                hit = shapely.contains(water_arr[w_idx], boxes[q_idx])
            else:  # touches
                hit = shapely.intersects(water_arr[w_idx], boxes[q_idx])
            np.logical_or.at(excluded, q_idx[hit], True)
        return excluded

    valid_cells: list[Leaf] = []
    if leaf_cells:
        rx0 = np.array([ox + c[0] * unit for c in leaf_cells])
        ry0 = np.array([oy + c[1] * unit for c in leaf_cells])
        rx1 = np.array([ox + c[2] * unit for c in leaf_cells])
        ry1 = np.array([oy + c[3] * unit for c in leaf_cells])
        boxes = shapely.box(rx0, ry0, rx1, ry1)
        contained = shapely.contains(bnd_shp, boxes)
        intersects = ~contained & shapely.intersects(bnd_shp, boxes)
        water_excl = _water_exclusion_mask(boxes, (rx0 + rx1) * 0.5, (ry0 + ry1) * 0.5)

        for i, leaf in enumerate(leaf_cells):
            if water_excl[i]:
                continue
            if intersects[i]:
                clipped = boxes[i].intersection(bnd_shp)
                if clipped.is_empty or clipped.area < 1e-6:
                    continue
            elif not contained[i]:
                continue
            valid_cells.append(leaf)

    if not valid_cells:
        raise InputError("no cells are left inside the domain; check the domain and water areas")

    # Rebuild cell_dict from the remaining cells: _finer_exists (used again below)
    # must not see neighbours that clipping or water removal just removed, or the
    # edge of every water hole gets an inconsistent midpoint; with many small water
    # areas this added up to thousands of pinch nodes (Parkstad).
    cell_dict = {(ix0, iy0, ix1, iy1): lvl for ix0, iy0, ix1, iy1, lvl in valid_cells}

    # -- vertex registry (integer keys) ------------------------------------------------
    _check_cancel(70, f"Quad Grid: {len(valid_cells)} valid cells, building nodes...")

    pt_map: dict[tuple[int, int], int] = {}
    pt_list: list[tuple[int, int]] = []

    def _vi(ix: int, iy: int) -> int:
        k = (ix, iy)
        if k not in pt_map:
            pt_map[k] = len(pt_list)
            pt_list.append(k)
        return pt_map[k]

    # -- prescan: which sides of each cell get a midpoint node --------------------------
    # A cell with a centre split (>= 3 splits, or 2 opposite) puts a midpoint on all
    # four sides, which forces the neighbour on that side to split as well; that can
    # turn the neighbour into a centre split too. Iterate to a fixed point.
    eff: dict[Key, set[str]] = {}
    for ix0, iy0, ix1, iy1, _lvl in valid_cells:
        eff[(ix0, iy0, ix1, iy1)] = {
            side for side in "NSEW" if _finer_exists(ix0, iy0, ix1, iy1, side)
        }

    valid_keys = set(eff.keys())
    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def _center_splits(sides: set[str]) -> bool:
        return len(sides) >= 3 or (len(sides) == 2 and sides in ({"N", "S"}, {"E", "W"}))

    work = deque(eff.keys())
    while work:
        ck = work.popleft()
        if not _center_splits(eff[ck]):
            continue
        for side, nk in zip("NSEW", _neighbours(ck), strict=True):
            if nk in valid_keys:
                need = opposite[side]
                if need not in eff[nk]:
                    eff[nk].add(need)
                    work.append(nk)

    # -- emit cells ----------------------------------------------------------------------
    _check_cancel(80, f"Quad Grid: converting {len(valid_cells)} cells to mesh elements...")

    out_cells: list[list[int]] = []
    n_valid = len(valid_cells)
    report_every3 = max(1, n_valid // 20)

    for i, (ix0, iy0, ix1, iy1, _lvl) in enumerate(valid_cells):
        if i % report_every3 == 0:
            _check_cancel(80 + int(18 * i / n_valid))
        ixm = (ix0 + ix1) >> 1
        iym = (iy0 + iy1) >> 1
        bl = _vi(ix0, iy0)
        br = _vi(ix1, iy0)
        tr = _vi(ix1, iy1)
        tl = _vi(ix0, iy1)
        sides = eff[(ix0, iy0, ix1, iy1)]
        s_n, s_s, s_e, s_w = ("N" in sides), ("S" in sides), ("E" in sides), ("W" in sides)
        n_splits = s_n + s_s + s_e + s_w

        if n_splits == 0:
            out_cells.append([bl, br, tr, tl])
        elif n_splits == 1:
            if s_s:
                m_s = _vi(ixm, iy0)
                out_cells += [[bl, m_s, tl], [m_s, br, tr], [m_s, tr, tl]]
            elif s_e:
                m_e = _vi(ix1, iym)
                out_cells += [[br, m_e, bl], [m_e, tr, tl], [m_e, tl, bl]]
            elif s_n:
                m_n = _vi(ixm, iy1)
                out_cells += [[tl, m_n, bl], [m_n, tr, br], [m_n, br, bl]]
            else:
                m_w = _vi(ix0, iym)
                out_cells += [[tl, m_w, tr], [m_w, bl, br], [m_w, br, tr]]
        elif n_splits == 2 and not (s_s and s_n) and not (s_e and s_w):
            if s_s and s_e:
                m_s = _vi(ixm, iy0)
                m_e = _vi(ix1, iym)
                out_cells += [[br, m_s, m_e], [bl, m_s, tl], [m_s, m_e, tl], [m_e, tr, tl]]
            elif s_s and s_w:
                m_s = _vi(ixm, iy0)
                m_w = _vi(ix0, iym)
                out_cells += [[bl, m_s, m_w], [m_s, br, tr], [m_s, tr, m_w], [m_w, tr, tl]]
            elif s_n and s_e:
                m_n = _vi(ixm, iy1)
                m_e = _vi(ix1, iym)
                out_cells += [[tr, m_n, m_e], [m_n, tl, bl], [m_n, bl, m_e], [m_e, bl, br]]
            else:
                m_n = _vi(ixm, iy1)
                m_w = _vi(ix0, iym)
                out_cells += [[tl, m_w, m_n], [m_w, bl, br], [m_w, br, m_n], [m_n, br, tr]]
        else:
            ctr = _vi(ixm, iym)
            m_s = _vi(ixm, iy0)
            m_e = _vi(ix1, iym)
            m_n = _vi(ixm, iy1)
            m_w = _vi(ix0, iym)
            out_cells += [
                [bl, m_s, ctr, m_w],
                [m_s, br, m_e, ctr],
                [ctr, m_e, tr, m_n],
                [m_w, ctr, m_n, tl],
            ]

    # -- head-to-toe fix -----------------------------------------------------------------
    _check_cancel(99, "Quad Grid: head-to-toe correction...")

    def _parent_sq(ax_: int, ay_: int, bx_: int, by_: int, gx_: int, gy_: int) -> Key:
        side_len = abs(bx_ - ax_) + abs(by_ - ay_)
        if ax_ == bx_:
            y0_ = min(ay_, by_)
            y1_ = max(ay_, by_)
            if gx_ <= ax_:
                return (ax_ - side_len, y0_, ax_, y1_)
            return (ax_, y0_, ax_ + side_len, y1_)
        x0_ = min(ax_, bx_)
        x1_ = max(ax_, bx_)
        if gy_ <= ay_:
            return (x0_, ay_ - side_len, x1_, ay_)
        return (x0_, ay_, x1_, ay_ + side_len)

    def _cells_in_sq(sq: Key, v2c: dict[int, set[int]]) -> list[int]:
        sx0, sy0, sx1, sy1 = sq
        sxm = (sx0 + sx1) // 2
        sym = (sy0 + sy1) // 2
        corners = [
            (sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1), (sxm, sy0),
            (sx1, sym), (sxm, sy1), (sx0, sym), (sxm, sym),
        ]  # fmt: skip
        sq_vs = {v for v in (pt_map.get(p) for p in corners) if v is not None}
        found: set[int] = set()
        for v in sq_vs:
            found.update(v2c[v])
        return [ci_ for ci_ in found if all(v in sq_vs for v in out_cells[ci_])]

    def _regen_sq(sq: Key) -> list[list[int]]:
        sx0, sy0, sx1, sy1 = sq
        sxm = (sx0 + sx1) // 2
        sym = (sy0 + sy1) // 2
        bl = _vi(sx0, sy0)
        br = _vi(sx1, sy0)
        tr = _vi(sx1, sy1)
        tl = _vi(sx0, sy1)
        m_s = pt_map.get((sxm, sy0))
        m_n = pt_map.get((sxm, sy1))
        m_w = pt_map.get((sx0, sym))
        m_e = pt_map.get((sx1, sym))
        n_ = sum(m is not None for m in (m_s, m_n, m_w, m_e))
        if n_ == 0:
            return [[bl, br, tr, tl]]
        if n_ == 1:
            if m_s is not None:
                return [[bl, m_s, tl], [m_s, br, tr], [m_s, tr, tl]]
            if m_e is not None:
                return [[br, m_e, bl], [m_e, tr, tl], [m_e, tl, bl]]
            if m_n is not None:
                return [[tl, m_n, bl], [m_n, tr, br], [m_n, br, bl]]
            assert m_w is not None
            return [[tl, m_w, tr], [m_w, bl, br], [m_w, br, tr]]
        if n_ == 2 and (m_s is None or m_n is None) and (m_e is None or m_w is None):
            if m_s is not None and m_e is not None:
                return [[br, m_s, m_e], [bl, m_s, tl], [m_s, m_e, tl], [m_e, tr, tl]]
            if m_s is not None and m_w is not None:
                return [[bl, m_s, m_w], [m_s, br, tr], [m_s, tr, m_w], [m_w, tr, tl]]
            if m_n is not None and m_e is not None:
                return [[tr, m_n, m_e], [m_n, tl, bl], [m_n, bl, m_e], [m_e, bl, br]]
            assert m_n is not None
            assert m_w is not None
            return [[tl, m_w, m_n], [m_w, bl, br], [m_w, br, m_n], [m_n, br, tr]]
        ctr = _vi(sxm, sym)
        if m_s is None:
            m_s = _vi(sxm, sy0)
        if m_n is None:
            m_n = _vi(sxm, sy1)
        if m_w is None:
            m_w = _vi(sx0, sym)
        if m_e is None:
            m_e = _vi(sx1, sym)
        return [[bl, m_s, ctr, m_w], [m_s, br, m_e, ctr], [ctr, m_e, tr, m_n], [m_w, ctr, m_n, tl]]

    def _h2t_pass() -> bool:
        """Regenerate the two squares around each head-to-toe triangle pair; True if any."""
        tris_i = [i for i, c in enumerate(out_cells) if len(c) == 3]
        _log.debug("[H2T] pass start: %d cells, %d triangles", len(out_cells), len(tris_i))
        if len(tris_i) < 2:
            return False

        pts = np.array(pt_list, dtype=np.int64)
        n_p = len(pts)
        tv = np.array([out_cells[i] for i in tris_i], dtype=np.int64)
        n_t = len(tris_i)

        a_ = tv[:, [0, 1, 2]].ravel()
        b_ = tv[:, [1, 2, 0]].ravel()
        op = tv[:, [2, 0, 1]].ravel()
        ci = np.repeat(np.arange(n_t), 3)
        em = np.minimum(a_, b_)
        ex = np.maximum(a_, b_)
        ek = em.astype(np.int64) * (n_p + 1) + ex.astype(np.int64)

        ord_ = np.argsort(ek, kind="stable")
        ek = ek[ord_]
        ci = ci[ord_]
        op = op[ord_]
        em = em[ord_]
        ex = ex[ord_]

        pp = np.where(ek[:-1] == ek[1:])[0]
        if not len(pp):
            return False

        ia = em[pp]
        ib = ex[pp]
        ic = op[pp]
        idd = op[pp + 1]
        ci1 = ci[pp]
        ci2 = ci[pp + 1]

        ax_a, ay_a = pts[ia, 0], pts[ia, 1]
        bx_a, by_a = pts[ib, 0], pts[ib, 1]
        cx_a, cy_a = pts[ic, 0], pts[ic, 1]
        dx_a, dy_a = pts[idd, 0], pts[idd, 1]
        ab2 = (bx_a - ax_a) ** 2 + (by_a - ay_a) ** 2
        ac2 = (cx_a - ax_a) ** 2 + (cy_a - ay_a) ** 2
        bc2 = (cx_a - bx_a) ** 2 + (cy_a - by_a) ** 2
        ad2 = (dx_a - ax_a) ** 2 + (dy_a - ay_a) ** 2
        bd2 = (dx_a - bx_a) ** 2 + (dy_a - by_a) ** 2
        both_mid = (
            (ab2 > np.minimum(ac2, bc2))
            & (np.maximum(ac2, bc2) > ab2)
            & (ab2 > np.minimum(ad2, bd2))
            & (np.maximum(ad2, bd2) > ab2)
        )
        s = np.where(ab2 > 0, ab2, 1.0)
        t_c = ((cx_a - ax_a) * (bx_a - ax_a) + (cy_a - ay_a) * (by_a - ay_a)) / s
        t_d = ((dx_a - ax_a) * (bx_a - ax_a) + (dy_a - ay_a) * (by_a - ay_a)) / s
        h2t = ((t_c < 0.35) & (t_d > 0.65)) | ((t_c > 0.65) & (t_d < 0.35))
        bad = both_mid & h2t

        _log.debug(
            "[H2T] shared edges=%d, both_mid=%d, h2t=%d, bad=%d",
            len(pp),
            int(both_mid.sum()),
            int(h2t.sum()),
            int(bad.sum()),
        )
        if not bad.any():
            return False

        v2c: dict[int, set[int]] = defaultdict(set)
        for i, cell in enumerate(out_cells):
            for v in cell:
                v2c[v].add(i)

        to_del: set[int] = set()
        new_out: list[list[int]] = []
        done_sqs: set[Key] = set()

        for idx in np.where(bad)[0]:
            t1 = tris_i[int(ci1[idx])]
            t2 = tris_i[int(ci2[idx])]
            if t1 in to_del or t2 in to_del:
                continue
            ax, ay, bx, by = int(ax_a[idx]), int(ay_a[idx]), int(bx_a[idx]), int(by_a[idx])
            if ax != bx and ay != by:
                continue
            sq1 = _parent_sq(ax, ay, bx, by, int(cx_a[idx]), int(cy_a[idx]))
            sq2 = _parent_sq(ax, ay, bx, by, int(dx_a[idx]), int(dy_a[idx]))
            if sq1 in done_sqs or sq2 in done_sqs:
                continue
            done_sqs.add(sq1)
            done_sqs.add(sq2)
            to_del.update(_cells_in_sq(sq1, v2c) + _cells_in_sq(sq2, v2c))
            _vi((ax + bx) // 2, (ay + by) // 2)
            new_out += _regen_sq(sq1) + _regen_sq(sq2)

        if not to_del:
            _log.debug("[H2T] no cells found to remove")
            return False
        _log.debug("[H2T] removing %d cells, adding %d", len(to_del), len(new_out))
        for i in sorted(to_del, reverse=True):
            out_cells.pop(i)
        out_cells.extend(new_out)
        return True

    h2t_n = 0
    while _h2t_pass():
        h2t_n += 1
        _check_cancel(msg=f"Quad Grid: head-to-toe fix, iteration {h2t_n}...")
    _log.debug("[H2T] done after %d iterations, %d cells", h2t_n, len(out_cells))

    # -- conformity repair: resolve every remaining hanging node ----------------------
    # The fixed-point prescan covers the centre-split cascade, but a single isolated
    # hanging node (on a transition or triangle edge) can remain. Split every cell
    # that still has a hanging node on an edge:
    #   * quad -> fan from the cell centre over its boundary including midpoints;
    #   * triangle -> bisect the edge with the hanging node towards the opposite corner.
    # Both reuse existing boundary nodes only (only interior nodes are new), so no
    # new hanging nodes appear and the iteration terminates.
    def _hanging_edge_index(cell: list[int]) -> tuple[int, int] | None:
        n = len(cell)
        for k in range(n):
            a = cell[k]
            b = cell[(k + 1) % n]
            ax, ay = pt_list[a]
            bx, by = pt_list[b]
            if (ax + bx) % 2 or (ay + by) % 2:
                continue
            m = pt_map.get(((ax + bx) // 2, (ay + by) // 2))
            if m is not None and m not in (a, b):
                return k, m
        return None

    for _rep_iter in range(16):
        new: list[list[int]] = []
        changed = False
        for cell in out_cells:
            n = len(cell)
            found = _hanging_edge_index(cell)
            if found is None:
                new.append(cell)
                continue
            changed = True
            j, m = found
            if n == 3:
                a = cell[j]
                b = cell[(j + 1) % 3]
                opp = cell[(j + 2) % 3]
                new.append([a, m, opp])
                new.append([m, b, opp])
            else:
                xs = [pt_list[v][0] for v in cell]
                ys = [pt_list[v][1] for v in cell]
                ctr = _vi((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)
                bnd = []
                for kk in range(n):
                    a = cell[kk]
                    b = cell[(kk + 1) % n]
                    bnd.append(a)
                    ax, ay = pt_list[a]
                    bx, by = pt_list[b]
                    if (ax + bx) % 2 == 0 and (ay + by) % 2 == 0:
                        mm = pt_map.get(((ax + bx) // 2, (ay + by) // 2))
                        if mm is not None and mm not in (a, b):
                            bnd.append(mm)
                nb = len(bnd)
                new.extend([ctr, bnd[ii], bnd[(ii + 1) % nb]] for ii in range(nb))
        out_cells[:] = new
        if not changed:
            break

    # -- conformity check: no hanging nodes ------------------------------------------
    edges_seen: set[tuple[int, int]] = set()
    hanging = 0
    for c in out_cells:
        n = len(c)
        for j in range(n):
            a = c[j]
            b = c[(j + 1) % n]
            e = (a, b) if a < b else (b, a)
            if e in edges_seen:
                continue
            edges_seen.add(e)
            ax, ay = pt_list[a]
            bx, by = pt_list[b]
            if (ax + bx) % 2 or (ay + by) % 2:
                continue
            mj = pt_map.get(((ax + bx) // 2, (ay + by) // 2))
            if mj is not None and mj not in (a, b):
                hanging += 1
    if hanging:
        _log.warning("[QuadRef] %d hanging nodes after the conformity pass", hanging)

    # -- integer -> coordinates (once) ------------------------------------------------
    pts_np = np.array(pt_list, dtype=np.float64)
    vertices = np.empty_like(pts_np)
    vertices[:, 0] = ox + pts_np[:, 0] * unit
    vertices[:, 1] = oy + pts_np[:, 1] * unit

    # -- orthogonality check and repair ------------------------------------------------
    # Last step, after balancing, centre splits and head-to-toe; may add centre nodes.
    _check_cancel(msg="Quad Grid: orthogonality check...")
    log = None if progress is None else (lambda m: progress(None, m))
    raw_cells = [list(c) for c in out_cells]  # the repair changes out_cells in place
    try:
        vertices, out_cells = orthogonality_repair(vertices, out_cells, log=log)
    except OrthogonalityRepairError as exc:
        raise RawRepairError(str(exc), exc.bad_points, vertices, raw_cells, (ox, oy)) from None

    _check_cancel(100, f"Quad Grid: done - {len(vertices)} nodes, {len(out_cells)} elements.")
    return vertices, out_cells, (ox, oy)
