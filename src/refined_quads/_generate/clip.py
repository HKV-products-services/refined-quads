"""Clip a finished mesh cell by cell with MeshKernel, as the model-building workflow does.

The workflow (hydrolib-core's ``mesh2d_clip`` as patched in qgis_dhydamo) generates the
mesh without water and then deletes cells with MeshKernel: one ``mesh2d_delete`` per
part of the (merged) water polygon, holes as inner rings, and finally
``mesh2d_delete_hanging_edges``. MeshKernel keeps nodes and edges and derives the
cells from them again, so a single deleted cell whose edges all remain comes back, and
a hole of three or four nodes becomes a cell again; this module reproduces that by
calling MeshKernel itself.

A small hole can still remain: MeshKernel leaves it open here, but every program that
builds the cells from nodes and edges with MeshKernel (hydrolib's ``mesh2d_from_netcdf``,
the D-HYDRO GUI) makes it one cell, often a hexagon that D-Flow FM rejects as not
orthogonal. Such holes keep their cells (ADR-0004).
"""

from __future__ import annotations

from collections.abc import Sequence

import meshkernel as mk
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

#: MeshKernel's deletion option per water removal rule.
DELETE_OPTIONS = {
    "centre": mk.DeleteMeshOption.FACES_WITH_INCLUDED_CIRCUMCENTERS,
    "inside": mk.DeleteMeshOption.INSIDE_NOT_INTERSECTED,
    "touches": mk.DeleteMeshOption.INSIDE_AND_INTERSECTED,
}


def clip_cells(
    vertices: NDArray[np.float64],
    cells: list[list[int]],
    polygons: Sequence[BaseGeometry],
    rule: str,
) -> list[list[int]]:
    """Delete the cells that MeshKernel deletes for the polygons and the rule.

    Parameters
    ----------
    vertices : numpy.ndarray of float, shape (n, 2)
        Node coordinates.
    cells : list of list of int
        Node indices per cell.
    polygons : sequence of shapely geometry
        Water areas (and holes of the domain); merged before clipping.
    rule : {"centre", "inside", "touches"}
        Water removal rule.

    Returns
    -------
    list of list of int
        The cells MeshKernel keeps, in the node numbering of ``vertices``, plus the
        deleted cells of every hole that MeshKernel would make a cell of when it
        builds the cells from nodes and edges (see :func:`closable_holes`).
    """
    parts = _parts(polygons)
    if not parts or not cells:
        return cells
    # MeshKernel rejects nodes that no cell uses; give it only the used ones.
    used = np.unique(np.fromiter((v for c in cells for v in c), dtype=np.int64))
    renumber = {int(old): new for new, old in enumerate(used)}
    kernel = mk.MeshKernel(projection=mk.ProjectionType.CARTESIAN)
    kernel.mesh2d_set(_mesh2d(vertices[used], [[renumber[v] for v in c] for c in cells]))
    for part in parts:
        kernel.mesh2d_delete(_geometry_list(part), DELETE_OPTIONS[rule], False)
    kernel.mesh2d_delete_hanging_edges()
    clipped = _cells_of(kernel.mesh2d_get(), vertices, used)
    return _keep_closable_holes(vertices, cells, clipped)


def closable_holes(
    vertices: NDArray[np.float64],
    cells: list[list[int]],
    known: set[tuple[int, ...]] | None = None,
) -> list[shapely.Polygon]:
    """Holes that MeshKernel makes a cell of when it builds the cells from nodes and edges.

    hydrolib (``mesh2d_from_netcdf``) and the D-HYDRO GUI give MeshKernel only the nodes
    and edges of a mesh; MeshKernel then finds the cells itself, and also closes a
    small hole (a ring of at most six nodes) with one cell.

    Parameters
    ----------
    vertices : numpy.ndarray of float, shape (n, 2)
        Node coordinates.
    cells : list of list of int
        Node indices per cell.
    known : set of tuple of int, optional
        Cells (as sorted node indices) that are not holes; default ``cells``. Pass the
        cells of the whole mesh when ``cells`` is only a part of it.

    Returns
    -------
    list of shapely.Polygon
        One polygon per cell MeshKernel finds that is not a known cell.
    """
    if not cells:
        return []
    used = np.unique(np.fromiter((v for c in cells for v in c), dtype=np.int64))
    renumber = {int(old): new for new, old in enumerate(used)}
    kernel = mk.MeshKernel(projection=mk.ProjectionType.CARTESIAN)
    kernel.mesh2d_set(_mesh2d(vertices[used], [[renumber[v] for v in c] for c in cells],
                              with_faces=False))  # fmt: skip
    found = _cells_of(kernel.mesh2d_get(), vertices, used)
    if known is None:
        known = {tuple(sorted(c)) for c in cells}
    return [shapely.Polygon(vertices[c]) for c in found if tuple(sorted(c)) not in known]


def _keep_closable_holes(
    vertices: NDArray[np.float64], cells: list[list[int]], clipped: list[list[int]]
) -> list[list[int]]:
    """Give back the deleted cells MeshKernel makes one cell of (ADR-0004).

    Two cases, both a region of deleted cells inside a ring of at most six nodes:

    1. MeshKernel's clip itself returns the region as one cell (seen with real water
       areas): that cell is replaced by the original cells.
    2. MeshKernel's clip leaves the region open, but MeshKernel closes it when it
       builds the cells from nodes and edges (hydrolib, the D-HYDRO GUI): the original
       cells are put back. A hole can only be bounded by cells that share a node with
       a deleted cell, so only those go to MeshKernel.

    A region counts only when deleted cells fill it exactly; other rings (around cells
    left out of the selection, or a hole that was already there when refining) are
    not this clip's. Filling a region completely with original cells cannot open a
    new one.
    """
    # clipped is in the node numbering of vertices (looked up by coordinates), so
    # cells can be compared on their sorted node indices.
    originals = {tuple(sorted(c)) for c in cells}
    kept = {tuple(sorted(c)) for c in clipped}
    deleted = [c for c in cells if tuple(sorted(c)) not in kept]
    if not deleted:
        return clipped
    centres = shapely.points(np.array([vertices[c].mean(axis=0) for c in deleted]))
    areas = np.array([shapely.Polygon(vertices[c]).area for c in deleted])

    def filling(region: shapely.Polygon) -> NDArray[np.int64] | None:
        inside = np.flatnonzero(shapely.covers(region, centres))
        if abs(areas[inside].sum() - region.area) <= 1e-6 * max(region.area, 1.0):
            return inside
        return None

    result: list[list[int]] = []
    for cell in clipped:
        inside = (
            None if tuple(sorted(cell)) in originals else filling(shapely.Polygon(vertices[cell]))
        )
        if inside is None:
            result.append(cell)
        else:
            result.extend(deleted[i] for i in inside)

    kept = {tuple(sorted(c)) for c in result}
    deleted_nodes = {v for c in deleted if tuple(sorted(c)) not in kept for v in c}
    border = [c for c in result if any(v in deleted_nodes for v in c)]
    for hole in closable_holes(vertices, border, known=kept):
        inside = filling(hole)
        if inside is not None:
            result.extend(deleted[i] for i in inside)
    return result


def _cells_of(
    out: mk.Mesh2d, vertices: NDArray[np.float64], used: NDArray[np.int64]
) -> list[list[int]]:
    """Return the cells of MeshKernel's mesh in the node numbering of ``vertices``.

    MeshKernel got the nodes ``used``; its nodes are matched on their coordinates.
    """
    number = {(float(vertices[i, 0]), float(vertices[i, 1])): int(i) for i in used}
    lookup = np.array(
        [number[(float(x), float(y))] for x, y in zip(out.node_x, out.node_y, strict=True)],
        dtype=np.int64,
    )
    cells: list[list[int]] = []
    start = 0
    for n in out.nodes_per_face:
        cells.append([int(v) for v in lookup[out.face_nodes[start : start + n]]])
        start += int(n)
    return cells


def _parts(polygons: Sequence[BaseGeometry]) -> list[shapely.Polygon]:
    """Merge the polygons and return the parts."""
    if not polygons:
        return []
    merged = shapely.union_all(np.asarray(list(polygons), dtype=object))
    parts = []
    for part in shapely.get_parts(merged):
        if not isinstance(part, shapely.Polygon) or part.is_empty or part.area <= 0.0:
            continue  # an intersection can leave lines or points; MeshKernel needs areas
        holes = [ring for ring in part.interiors if shapely.Polygon(ring).area > 0.0]
        parts.append(shapely.Polygon(part.exterior, holes))
    return parts


def _geometry_list(polygon: shapely.Polygon) -> mk.GeometryList:
    """Encode a polygon for MeshKernel: exterior, then each hole after a separator."""
    rings = [np.asarray(polygon.exterior.coords)]
    rings += [np.asarray(ring.coords) for ring in polygon.interiors]
    separator = np.array([[-998.0, -998.0]])
    xy = rings[0]
    for ring in rings[1:]:
        xy = np.vstack([xy, separator, ring])
    return mk.GeometryList(
        np.ascontiguousarray(xy[:, 0]),
        np.ascontiguousarray(xy[:, 1]),
        geometry_separator=-999.0,
        inner_outer_separator=-998.0,
    )


def _mesh2d(
    vertices: NDArray[np.float64], cells: list[list[int]], with_faces: bool = True
) -> mk.Mesh2d:
    """Build MeshKernel's Mesh2d of the mesh: edges, and the faces unless ``with_faces`` is off.

    Without faces MeshKernel finds the cells itself from the edges, as it does when
    hydrolib or the D-HYDRO GUI reads a net file.
    """
    edges = sorted(
        {(min(c[k], c[(k + 1) % len(c)]), max(c[k], c[(k + 1) % len(c)])) for c in cells
         for k in range(len(c))}
    )  # fmt: skip
    if not with_faces:
        return mk.Mesh2d(
            node_x=np.ascontiguousarray(vertices[:, 0]),
            node_y=np.ascontiguousarray(vertices[:, 1]),
            edge_nodes=np.asarray(edges, dtype=np.int32).ravel(),
        )
    return mk.Mesh2d(
        node_x=np.ascontiguousarray(vertices[:, 0]),
        node_y=np.ascontiguousarray(vertices[:, 1]),
        edge_nodes=np.asarray(edges, dtype=np.int32).ravel(),
        face_nodes=np.asarray([v for c in cells for v in c], dtype=np.int32),
        nodes_per_face=np.asarray([len(c) for c in cells], dtype=np.int32),
    )
