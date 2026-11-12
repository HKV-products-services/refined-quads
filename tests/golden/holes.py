"""Holes MeshKernel closes: rings of edges it turns into a cell when it builds the cells.

hydrolib (``mesh2d_from_netcdf``) and the D-HYDRO GUI give MeshKernel only nodes and
edges; MeshKernel then finds the cells itself and also makes a cell of a small hole
(issue 20, ADR-0004). This is a test-side implementation, independent of the package.
"""

import meshkernel as mk
import numpy as np
import shapely


def _key(points):
    return tuple(sorted((round(float(x), 6), round(float(y), 6)) for x, y in points))


def cell_polygons(mesh):
    """The cells of a mesh as shapely polygons."""
    return [shapely.Polygon(mesh.nodes[f[f >= 0]]) for f in mesh.faces]


def closable_holes(mesh):
    """Polygons of the cells MeshKernel finds that are not cells of the mesh."""
    faces = [f[f >= 0] for f in mesh.faces]
    used = np.unique(np.concatenate(faces))
    renumber = np.full(len(mesh.nodes), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))
    edges = {
        tuple(sorted((int(renumber[f[k]]), int(renumber[f[(k + 1) % len(f)]]))))
        for f in faces
        for k in range(len(f))
    }
    kernel = mk.MeshKernel(projection=mk.ProjectionType.CARTESIAN)
    kernel.mesh2d_set(
        mk.Mesh2d(
            node_x=np.ascontiguousarray(mesh.nodes[used, 0]),
            node_y=np.ascontiguousarray(mesh.nodes[used, 1]),
            edge_nodes=np.asarray(sorted(edges), dtype=np.int32).ravel(),
        )
    )
    out = kernel.mesh2d_get()
    known = {_key(mesh.nodes[f]) for f in faces}
    holes, start = [], 0
    for n in out.nodes_per_face:
        idx = out.face_nodes[start : start + n]
        start += int(n)
        points = np.column_stack([out.node_x[idx], out.node_y[idx]])
        if _key(points) not in known:
            holes.append(shapely.Polygon(points))
    return holes


def assert_master_with_holes_filled(mesh, master):
    """``mesh`` is ``master`` plus cells that exactly fill the holes MeshKernel closes in it."""
    master_cells = {_key(p.exterior.coords[:-1]) for p in cell_polygons(master)}
    mesh_polys = cell_polygons(mesh)
    mesh_cells = {_key(p.exterior.coords[:-1]) for p in mesh_polys}
    assert master_cells <= mesh_cells, f"{len(master_cells - mesh_cells)} master cells missing"
    extra = [p for p in mesh_polys if _key(p.exterior.coords[:-1]) not in master_cells]
    holes = closable_holes(master)
    filled = shapely.union_all(extra) if extra else shapely.Polygon()
    closed = shapely.union_all(holes) if holes else shapely.Polygon()
    assert filled.symmetric_difference(closed).area < 1e-6, (
        f"{len(extra)} extra cells ({filled.area:.1f} m2) do not fill the "
        f"{len(holes)} holes of the master ({closed.area:.1f} m2)"
    )
