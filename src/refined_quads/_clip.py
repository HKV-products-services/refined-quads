"""Clip a finished mesh cell by cell with MeshKernel, as the model-building workflow does.

The workflow (hydrolib-core's ``mesh2d_clip`` as patched in qgis_dhydamo) generates the
mesh without water and then deletes cells with MeshKernel: one ``mesh2d_delete`` per
part of the (merged) water polygon, holes as inner rings, and finally
``mesh2d_delete_hanging_edges``. MeshKernel keeps nodes and edges and derives the
cells from them again, so a single deleted cell whose edges all remain comes back, and
a hole of three or four nodes becomes a cell again; this module reproduces that by
calling MeshKernel itself.
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
        The cells MeshKernel keeps, in the node numbering of ``vertices``.
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
    out = kernel.mesh2d_get()

    number = {(float(x), float(y)): i for i, (x, y) in enumerate(vertices)}
    lookup = np.array(
        [number[(float(x), float(y))] for x, y in zip(out.node_x, out.node_y, strict=True)],
        dtype=np.int64,
    )
    clipped: list[list[int]] = []
    start = 0
    for n in out.nodes_per_face:
        clipped.append([int(v) for v in lookup[out.face_nodes[start : start + n]]])
        start += int(n)
    return clipped


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


def _mesh2d(vertices: NDArray[np.float64], cells: list[list[int]]) -> mk.Mesh2d:
    """Build MeshKernel's Mesh2d of the mesh, with edges and faces."""
    edges = sorted(
        {(min(c[k], c[(k + 1) % len(c)]), max(c[k], c[(k + 1) % len(c)])) for c in cells
         for k in range(len(c))}
    )  # fmt: skip
    return mk.Mesh2d(
        node_x=np.ascontiguousarray(vertices[:, 0]),
        node_y=np.ascontiguousarray(vertices[:, 1]),
        edge_nodes=np.asarray(edges, dtype=np.int32).ravel(),
        face_nodes=np.asarray([v for c in cells for v in c], dtype=np.int32),
        nodes_per_face=np.asarray([len(c) for c in cells], dtype=np.int32),
    )
