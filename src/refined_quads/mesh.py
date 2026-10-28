"""In-memory refined quad mesh and the lattice it lives on."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS
from pyproj.exceptions import CRSError

from refined_quads.errors import InputError

#: Distance, as a fraction of the lattice unit, within which a node counts as a lattice point.
LATTICE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Lattice:
    """The fixed grid of points on which every square of a refined quad mesh has its corners.

    Parameters
    ----------
    origin : tuple of float
        Lattice point ``(x, y)`` from which lattice indices are counted.
    cell_size : float
        Side length of the base cell; any positive length.
    levels : int
        Highest refinement level; the finest square has side ``cell_size / 2**levels``.

    Raises
    ------
    InputError
        If the origin is not finite, the cell size is not positive or the number of
        levels is negative.
    """

    origin: tuple[float, float]
    cell_size: float
    levels: int

    def __post_init__(self) -> None:
        """Validate and normalise the fields."""
        origin = (float(self.origin[0]), float(self.origin[1]))
        if not all(math.isfinite(v) for v in origin):
            raise InputError(f"lattice origin must be finite, got {self.origin!r}")
        cell_size = float(self.cell_size)
        if not (math.isfinite(cell_size) and cell_size > 0):
            raise InputError(f"cell size must be a positive length, got {self.cell_size!r}")
        if isinstance(self.levels, bool) or int(self.levels) != self.levels or self.levels < 0:
            raise InputError(f"levels must be a non-negative integer, got {self.levels!r}")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "cell_size", cell_size)
        object.__setattr__(self, "levels", int(self.levels))

    @property
    def unit(self) -> float:
        """Side length of the finest square, ``cell_size / 2**levels``."""
        return math.ldexp(self.cell_size, -self.levels)

    def to_coordinates(self, indices: ArrayLike) -> NDArray[np.float64]:
        """Convert integer lattice indices to coordinates.

        Parameters
        ----------
        indices : array_like of int, shape (n, 2)
            Lattice indices in units of :attr:`unit` from the origin.

        Returns
        -------
        numpy.ndarray of float, shape (n, 2)
            Coordinates ``origin + index * unit``.
        """
        idx = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
        return np.asarray(self.origin, dtype=np.float64) + idx * self.unit

    def to_indices(self, coordinates: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        """Convert coordinates to the nearest integer lattice indices.

        Parameters
        ----------
        coordinates : array_like of float, shape (n, 2)
            Point coordinates.

        Returns
        -------
        indices : numpy.ndarray of int, shape (n, 2)
            Nearest lattice indices in units of :attr:`unit` from the origin.
        on_lattice : numpy.ndarray of bool, shape (n,)
            True where the point lies within ``1e-6 * unit`` of that lattice point.
        """
        xy = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
        scaled = (xy - np.asarray(self.origin)) / self.unit
        indices = np.rint(scaled)
        on_lattice = np.all(np.abs(scaled - indices) <= LATTICE_TOLERANCE, axis=1)
        return indices.astype(np.int64), on_lattice


@dataclass(frozen=True, eq=False)
class Mesh:
    """An immutable refined quad mesh.

    Parameters
    ----------
    nodes : array_like of float, shape (n, 2)
        Node coordinates.
    faces : array_like of int, shape (m, k)
        Node indices of each face, counterclockwise, padded at the end with ``-1``.
    lattice : Lattice
        The lattice the mesh lives on.
    crs : pyproj.CRS or str or int, optional
        Projected coordinate reference system of the coordinates, or ``None`` when it
        is unknown; anything ``pyproj.CRS.from_user_input`` accepts.

    Raises
    ------
    InputError
        If the arrays have the wrong shape, coordinates are not finite, a face has
        fewer than three nodes, refers to a missing node or has padding before a node,
        or the CRS is invalid or geographic (degrees).

    Notes
    -----
    The arrays are stored as read-only copies. Two meshes are compared on their
    cells, not with ``==``; see ``docs/adr/0002-output-contract-same-cells.md``.
    """

    nodes: NDArray[np.float64]
    faces: NDArray[np.int64]
    lattice: Lattice = field(repr=False)
    crs: CRS | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate the arrays and the CRS and store the arrays as read-only copies."""
        if self.crs is not None:
            object.__setattr__(self, "crs", projected_crs(self.crs))
        nodes = np.array(self.nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise InputError(f"nodes must have shape (n, 2), got {nodes.shape}")
        if not np.isfinite(nodes).all():
            raise InputError("node coordinates must be finite")
        faces = np.array(self.faces, dtype=np.int64)
        if faces.ndim != 2 or (faces.size and faces.shape[1] < 3):
            raise InputError(f"faces must have shape (m, k) with k >= 3, got {faces.shape}")
        if faces.size:
            if ((faces < -1) | (faces >= len(nodes))).any():
                raise InputError("faces refer to nodes that do not exist")
            valid = faces >= 0
            if (valid[:, 1:] & ~valid[:, :-1]).any():
                raise InputError("faces may only be padded with -1 after their last node")
            if (valid.sum(axis=1) < 3).any():
                raise InputError("every face needs at least three nodes")
        nodes.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "faces", faces)

    @property
    def face_sizes(self) -> NDArray[np.int64]:
        """Number of nodes of each face."""
        return np.asarray((self.faces >= 0).sum(axis=1), dtype=np.int64)

    def __repr__(self) -> str:
        """Summarise the mesh without printing the arrays."""
        crs = "None" if self.crs is None else self.crs.to_string()
        return (
            f"Mesh(nodes={len(self.nodes)}, faces={len(self.faces)}, "
            f"cell_size={self.lattice.cell_size:g}, levels={self.lattice.levels}, crs={crs})"
        )


def projected_crs(value: object) -> CRS:
    """Interpret a CRS and require it to be projected.

    Parameters
    ----------
    value : pyproj.CRS or str or int
        Anything ``pyproj.CRS.from_user_input`` accepts.

    Returns
    -------
    pyproj.CRS
        The CRS.

    Raises
    ------
    InputError
        If the value is not a CRS, or the CRS is geographic (degrees).
    """
    try:
        crs = CRS.from_user_input(value)
    except CRSError as exc:
        raise InputError(f"invalid CRS {value!r}: {exc}") from exc
    if crs.is_geographic:
        raise InputError(f"CRS {crs.to_string()} is geographic; cell sizes need a projected CRS")
    return crs
