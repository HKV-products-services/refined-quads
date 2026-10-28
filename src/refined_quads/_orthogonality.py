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

import numpy as np
from numpy.typing import NDArray

from refined_quads._topology import Edges, directed_edges

#: Highest FM orthogonality D-Flow FM accepts on an internal edge.
MAX_FM_ORTHOGONALITY = 0.30


def face_centres(nodes: NDArray[np.float64], faces: NDArray[np.int64]) -> NDArray[np.float64]:
    """Compute the FM centre of every face.

    Parameters
    ----------
    nodes : numpy.ndarray of float, shape (n, 2)
        Node coordinates.
    faces : numpy.ndarray of int, shape (m, k)
        Node indices per face, padded with ``-1``.

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
        h = _typical_edge_length(nodes, faces)
        corners = nodes[faces[tri, :3]]
        centres[tri] = _rgfgrid_centres(corners[:, 0], corners[:, 1], corners[:, 2], h)
    return centres


def edge_orthogonality(
    nodes: NDArray[np.float64], faces: NDArray[np.int64], edges: Edges
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
    centres = face_centres(nodes, faces)
    ev = nodes[edges.nodes[internal, 1]] - nodes[edges.nodes[internal, 0]]
    cv = centres[edges.faces[internal, 1]] - centres[edges.faces[internal, 0]]
    el = np.hypot(ev[:, 0], ev[:, 1])
    cl = np.hypot(cv[:, 0], cv[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        orth = np.abs((ev * cv).sum(axis=1) / (el * cl))
    orth[(el <= 0.0) | (cl <= 0.0) | ~np.isfinite(orth)] = np.inf
    result[internal] = orth
    return result


def _typical_edge_length(nodes: NDArray[np.float64], faces: NDArray[np.int64]) -> float:
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
    ok = np.abs(d) >= 1e-12 * h * h
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
