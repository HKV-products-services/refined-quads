"""Check a refined quad mesh for the properties D-Flow FM and refining rely on."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray

from refined_quads._common.recognise import lattice_squares, twice_signed_areas
from refined_quads._common.topology import hanging_nodes, unique_edges
from refined_quads._generate.balance import balance_violations
from refined_quads._generate.clip import closable_holes
from refined_quads._generate.orthogonality import MAX_FM_ORTHOGONALITY, edge_orthogonality
from refined_quads.mesh import Lattice, Mesh


@dataclass(frozen=True, eq=False)
class MeshReport:
    """Result of :func:`check`.

    Locations are coordinates ``(x, y)`` in the CRS of the mesh, so they can be shown
    on a map.

    Attributes
    ----------
    lattice : Lattice
        The lattice the mesh was checked against.
    n_nodes, n_cells : int
        Number of nodes and cells of the mesh.
    cells_per_level : dict of int to int
        Number of cells per refinement level; a transition cell counts at the level
        of the lattice square it lies in.
    cell_levels : numpy.ndarray of int, shape (m,)
        Refinement level of the lattice square each cell lies in; -1 for unrecognised
        cells.
    off_lattice_nodes : numpy.ndarray of float, shape (k, 2)
        Nodes that are not lattice points.
    duplicate_nodes : numpy.ndarray of float, shape (k, 2)
        Nodes at the same lattice point as another node (one row per extra node).
        Information only: cells that touch only in such a point lose nothing, as flow
        goes through edges (see ``split_edges``).
    split_edges : numpy.ndarray of float, shape (k, 2)
        Midpoints of edges that lie on top of another edge with different nodes: an
        internal wall that flow cannot cross.
    hanging_nodes : numpy.ndarray of float, shape (k, 2)
        Nodes that lie inside an edge of a cell instead of at its ends.
    balance_violations : numpy.ndarray of float, shape (k, 2)
        Midpoints of sides where a lattice square meets a square more than one
        level coarser (2:1 balance).
    orthogonality : numpy.ndarray of float, shape (e,)
        FM orthogonality of every internal edge (an edge of exactly two cells);
        ``inf`` for a degenerate edge.
    non_orthogonal_edges : numpy.ndarray of float, shape (k, 2)
        Midpoints of internal edges with an orthogonality above
        :attr:`max_allowed_orthogonality` or that are degenerate.
    max_allowed_orthogonality : float
        Threshold D-Flow FM applies, 0.30.
    clockwise_cells : numpy.ndarray of int
        Cells whose nodes are not counterclockwise (clockwise or zero area).
    unrecognised_cells : numpy.ndarray of int
        Cells that are neither a lattice square nor part of a group of transition
        cells that fills one; their level is unknown, so they are left out of
        ``cells_per_level`` and the 2:1 check.
    closable_holes : numpy.ndarray of float, shape (k, 2)
        Centres of holes that MeshKernel makes a cell of when it builds the cells from
        nodes and edges, as hydrolib (``mesh2d_from_netcdf``) and the D-HYDRO GUI do: a
        ring of at most six nodes around no cell. Information only: a hole of four
        nodes becomes an ordinary cell, but one of five or six nodes can be a cell
        D-Flow FM rejects as not orthogonal (ADR-0004).
    """

    lattice: Lattice
    n_nodes: int
    n_cells: int
    cells_per_level: dict[int, int]
    cell_levels: NDArray[np.int64]
    off_lattice_nodes: NDArray[np.float64]
    duplicate_nodes: NDArray[np.float64]
    split_edges: NDArray[np.float64]
    hanging_nodes: NDArray[np.float64]
    balance_violations: NDArray[np.float64]
    orthogonality: NDArray[np.float64]
    non_orthogonal_edges: NDArray[np.float64]
    max_allowed_orthogonality: float
    clockwise_cells: NDArray[np.int64]
    unrecognised_cells: NDArray[np.int64]
    closable_holes: NDArray[np.float64]

    @property
    def max_orthogonality(self) -> float:
        """Worst FM orthogonality over the internal edges; 0 if there are none."""
        return float(self.orthogonality.max()) if self.orthogonality.size else 0.0

    @property
    def is_valid(self) -> bool:
        """True when every check passes."""
        return not (
            len(self.off_lattice_nodes)
            or len(self.split_edges)
            or len(self.hanging_nodes)
            or len(self.balance_violations)
            or len(self.non_orthogonal_edges)
            or len(self.clockwise_cells)
            or len(self.unrecognised_cells)
        )

    def summary(self) -> str:
        """Describe the report in a few lines of text.

        Returns
        -------
        str
            Lattice, cells per level and the number of problems per check.
        """
        lat = self.lattice
        levels = ", ".join(
            f"{level} ({lat.cell_size / 2**level:g}): {n}"
            for level, n in sorted(self.cells_per_level.items())
        )
        lines = [
            f"lattice: origin ({lat.origin[0]:g}, {lat.origin[1]:g}), "
            f"base cell {lat.cell_size:g}, {lat.levels} levels, unit {lat.unit:g}",
            f"nodes: {self.n_nodes}, cells: {self.n_cells}",
            f"cells per level: {levels or '-'}",
            f"nodes off the lattice: {len(self.off_lattice_nodes)}",
            f"duplicate nodes (information): {len(self.duplicate_nodes)}",
            f"split edges: {len(self.split_edges)}",
            f"hanging nodes: {len(self.hanging_nodes)}",
            f"2:1 balance violations: {len(self.balance_violations)}",
            f"non-orthogonal edges (> {self.max_allowed_orthogonality:.2f}): "
            f"{len(self.non_orthogonal_edges)} (worst {self.max_orthogonality:.3f})",
            f"clockwise cells: {len(self.clockwise_cells)}",
            f"unrecognised cells: {len(self.unrecognised_cells)}",
            f"holes MeshKernel closes (information): {len(self.closable_holes)}",
            f"valid: {'yes' if self.is_valid else 'no'}",
        ]
        return "\n".join(lines)


def check(mesh: Mesh) -> MeshReport:
    """Check a mesh for hanging nodes, 2:1 balance, FM orthogonality and orientation.

    Parameters
    ----------
    mesh : Mesh
        The mesh to check.

    Returns
    -------
    MeshReport
        The findings per check; :attr:`MeshReport.is_valid` tells whether all pass.

    Notes
    -----
    Only nodes used by a cell are checked. Hanging nodes, 2:1 balance, orientation
    and levels are determined exactly on integer lattice indices; orthogonality is
    computed on the coordinates, as D-Flow FM does.
    """
    lattice = mesh.lattice
    faces = mesh.faces
    used = np.unique(faces[faces >= 0])
    indices, on_lattice = lattice.to_indices(mesh.nodes)
    off_lattice = used[~on_lattice[used]]

    _, first, counts = np.unique(indices[used], axis=0, return_index=True, return_counts=True)
    duplicates = np.setdiff1d(np.arange(len(used)), first).astype(np.int64)
    edges = unique_edges(faces)
    # An edge as the pair of its lattice points (each point encoded as one integer).
    low = indices[used].min(axis=0)
    span = indices[used].max(axis=0) - low + 1
    point_key = (indices[:, 0] - low[0]) * span[1] + (indices[:, 1] - low[1])
    edge_keys = np.sort(point_key[edges.nodes], axis=1)
    _, edge_first = np.unique(edge_keys, axis=0, return_index=True)
    split = np.setdiff1d(np.arange(len(edges.nodes)), edge_first).astype(np.int64)
    split_midpoints = (mesh.nodes[edges.nodes[split, 0]] + mesh.nodes[edges.nodes[split, 1]]) / 2
    hanging = hanging_nodes(indices, edges.nodes)

    squares = lattice_squares(indices, faces, lattice.levels)
    violations = balance_violations(squares, lattice.levels)
    levels, counts = np.unique(squares.face_level[squares.face_level >= 0], return_counts=True)

    orth = edge_orthogonality(mesh.nodes, faces, edges)
    internal = ~np.isnan(orth)
    bad = internal & (orth > MAX_FM_ORTHOGONALITY)
    bad_edges = edges.nodes[bad]
    bad_midpoints = (mesh.nodes[bad_edges[:, 0]] + mesh.nodes[bad_edges[:, 1]]) / 2.0

    return MeshReport(
        lattice=lattice,
        n_nodes=len(used),
        n_cells=len(faces),
        cells_per_level={int(lv): int(n) for lv, n in zip(levels, counts, strict=True)},
        cell_levels=squares.face_level,
        off_lattice_nodes=mesh.nodes[off_lattice],
        duplicate_nodes=mesh.nodes[used[duplicates]],
        split_edges=split_midpoints,
        hanging_nodes=mesh.nodes[hanging],
        balance_violations=np.asarray(lattice.origin) + violations * (lattice.unit / 2.0),
        orthogonality=orth[internal],
        non_orthogonal_edges=bad_midpoints,
        max_allowed_orthogonality=MAX_FM_ORTHOGONALITY,
        clockwise_cells=np.flatnonzero(twice_signed_areas(indices, faces) <= 0),
        unrecognised_cells=np.flatnonzero(squares.face_level < 0),
        closable_holes=_closable_hole_centres(mesh.nodes, faces),
    )


def _closable_hole_centres(
    nodes: NDArray[np.float64], faces: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Centres of the holes MeshKernel closes when it builds the cells from the edges.

    A hole is bounded by edges of one cell each, so only the cells with such an edge go
    to MeshKernel. A ring it finds counts when it overlaps no cell of the mesh (a ring
    around cells left out of that selection does).
    """
    cells = [[int(v) for v in f[f >= 0]] for f in faces]
    count: dict[tuple[int, int], int] = {}
    for c in cells:
        for k in range(len(c)):
            edge = (min(c[k], c[(k + 1) % len(c)]), max(c[k], c[(k + 1) % len(c)]))
            count[edge] = count.get(edge, 0) + 1
    border = [
        c for c in cells
        if any(count[(min(c[k], c[(k + 1) % len(c)]), max(c[k], c[(k + 1) % len(c)]))] == 1
               for k in range(len(c)))
    ]  # fmt: skip
    rings = closable_holes(nodes, border, known={tuple(sorted(c)) for c in cells})
    if not rings:
        return np.empty((0, 2))
    polygons = [shapely.Polygon(nodes[c]) for c in cells]
    tree = shapely.STRtree(polygons)
    holes = [
        ring for ring in rings
        if not any(polygons[i].intersection(ring).area > 1e-9 * ring.area
                   for i in tree.query(ring))
    ]  # fmt: skip
    return np.array([ring.centroid.coords[0] for ring in holes]).reshape(-1, 2)
