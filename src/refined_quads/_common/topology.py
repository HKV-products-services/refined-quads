"""Edges of padded face arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Edges:
    """Unique undirected edges of a face array and the faces on either side.

    Attributes
    ----------
    nodes : numpy.ndarray of int, shape (e, 2)
        Node indices of each edge, lowest first.
    face_count : numpy.ndarray of int, shape (e,)
        Number of faces that have the edge.
    faces : numpy.ndarray of int, shape (e, 2)
        The first two faces that have the edge; ``-1`` where there are fewer.
    """

    nodes: NDArray[np.int64]
    face_count: NDArray[np.int64]
    faces: NDArray[np.int64]


def directed_edges(faces: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """List the edges of every face in face order.

    Parameters
    ----------
    faces : numpy.ndarray of int, shape (m, k)
        Node indices per face, padded with ``-1``.

    Returns
    -------
    face : numpy.ndarray of int, shape (d,)
        Face of each edge.
    nodes : numpy.ndarray of int, shape (d, 2)
        Start and end node of each edge, following the face's node order.
    """
    valid = faces >= 0
    sizes = valid.sum(axis=1)
    rows, cols = np.nonzero(valid)
    nxt = np.where(cols + 1 < sizes[rows], cols + 1, 0)
    start = faces[rows, cols]
    end = faces[rows, nxt]
    return rows.astype(np.int64), np.stack([start, end], axis=1).astype(np.int64)


def unique_edges(faces: NDArray[np.int64]) -> Edges:
    """Find the unique undirected edges of a face array.

    Parameters
    ----------
    faces : numpy.ndarray of int, shape (m, k)
        Node indices per face, padded with ``-1``.

    Returns
    -------
    Edges
        The unique edges with the faces that share them.
    """
    face, nodes = directed_edges(faces)
    key = np.sort(nodes, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, face = key[order], face[order]
    if len(key) == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return Edges(empty, np.empty(0, dtype=np.int64), empty)
    new = np.ones(len(key), dtype=bool)
    new[1:] = (key[1:] != key[:-1]).any(axis=1)
    start = np.flatnonzero(new)
    count = np.diff(np.append(start, len(key)))
    second = np.where(count >= 2, face[np.minimum(start + 1, len(key) - 1)], -1)
    return Edges(key[start], count, np.stack([face[start], second], axis=1))


def hanging_nodes(indices: NDArray[np.int64], edges: NDArray[np.int64]) -> NDArray[np.int64]:
    """Find nodes that lie inside an edge instead of at its ends.

    Parameters
    ----------
    indices : numpy.ndarray of int, shape (n, 2)
        Lattice index of every node.
    edges : numpy.ndarray of int, shape (e, 2)
        Node indices of the edges to test.

    Returns
    -------
    numpy.ndarray of int
        Sorted indices of the nodes that lie strictly inside one of the edges.

    Notes
    -----
    Exact: only lattice points can be nodes, so the candidates are the lattice
    points strictly between the two ends of each edge.
    """
    used = np.unique(edges)
    if used.size == 0:
        return np.empty(0, dtype=np.int64)
    lookup = PointLookup(indices[used])
    a = indices[edges[:, 0]]
    d = indices[edges[:, 1]] - a
    steps = np.gcd(np.abs(d[:, 0]), np.abs(d[:, 1]))
    found: list[NDArray[np.int64]] = []
    for t in range(1, int(steps.max(initial=0))):
        sel = steps > t
        points = a[sel] + t * (d[sel] // steps[sel, None])
        hit = lookup.find(points)
        found.append(used[hit[hit >= 0]])
    if not found:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(found))


class PointLookup:
    """Exact lookup of integer points in a fixed set."""

    def __init__(self, points: NDArray[np.int64]) -> None:
        self._low = points.min(axis=0)
        self._span = points.max(axis=0) - self._low + 1
        keys = self._encode(points)
        self._order = np.argsort(keys)
        self._keys = keys[self._order]

    def _encode(self, points: NDArray[np.int64]) -> NDArray[np.int64]:
        shifted = points - self._low
        return np.asarray(shifted[:, 0] * self._span[1] + shifted[:, 1], dtype=np.int64)

    def find(self, points: NDArray[np.int64]) -> NDArray[np.int64]:
        """Return the position of each point in the set, or -1 if it is not there."""
        result = np.full(len(points), -1, dtype=np.int64)
        inside = ((points >= self._low) & (points < self._low + self._span)).all(axis=1)
        keys = self._encode(points[inside])
        pos = np.minimum(np.searchsorted(self._keys, keys), len(self._keys) - 1)
        hit = self._keys[pos] == keys
        result[np.flatnonzero(inside)[hit]] = self._order[pos[hit]]
        return result
