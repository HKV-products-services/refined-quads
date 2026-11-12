"""Read and write the 2D mesh as UGRID-1.0 netCDF, as D-Flow FM and D-HYDRO use it."""

from __future__ import annotations

import datetime
import os
from typing import Any

import netCDF4
import numpy as np
from numpy.typing import NDArray
from pyproj import CRS

from refined_quads._common.recognise import derive_lattice
from refined_quads._io.gis import crs_cf_attributes
from refined_quads.errors import InputError
from refined_quads.mesh import Mesh

_FILL_INT = np.int32(-999)
_FILL_FLOAT = np.float64(-999.0)

# EPSG:28992 (Amersfoort / RD New, Bessel 1841) exactly as the D-Flow FM plugin writes it.
_RD_NEW_PROJ4 = (
    "+proj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 "
    "+k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +units=m +no_defs"
)
_RD_NEW_WKT = (
    'PROJCS["Amersfoort / RD New",\n'
    '    GEOGCS["Amersfoort",\n'
    '        DATUM["Amersfoort",\n'
    '            SPHEROID["Bessel 1841",6377397.155,299.1528128,\n'
    '                AUTHORITY["EPSG","7004"]],\n'
    '            AUTHORITY["EPSG","6289"]],\n'
    '        PRIMEM["Greenwich",0,\n'
    '            AUTHORITY["EPSG","8901"]],\n'
    '        UNIT["degree",0.0174532925199433,\n'
    '            AUTHORITY["EPSG","9122"]],\n'
    '        AUTHORITY["EPSG","4289"]],\n'
    '    PROJECTION["Oblique_Stereographic"],\n'
    '    PARAMETER["latitude_of_origin",52.1561605555556],\n'
    '    PARAMETER["central_meridian",5.38763888888889],\n'
    '    PARAMETER["scale_factor",0.9999079],\n'
    '    PARAMETER["false_easting",155000],\n'
    '    PARAMETER["false_northing",463000],\n'
    '    UNIT["metre",1,\n'
    '        AUTHORITY["EPSG","9001"]],\n'
    '    AXIS["Easting",EAST],\n'
    '    AXIS["Northing",NORTH],\n'
    '    AUTHORITY["EPSG","28992"]]'
)
_RD_NEW_ATTRIBUTES: dict[str, Any] = {
    "name": "Amersfoort / RD New",
    "epsg": np.int32(28992),
    "grid_mapping_name": "Unknown projected",
    "longitude_of_prime_meridian": np.float64(0.0),
    "semi_major_axis": np.float64(6377397.155),
    "semi_minor_axis": np.float64(6356078.962818189),
    "inverse_flattening": np.float64(299.1528128),
    "EPSG_code": "EPSG:28992",
    "value": "value is equal to EPSG code",
    "proj4_params": _RD_NEW_PROJ4,
    "projection_name": "unknown",
    "wkt": _RD_NEW_WKT,
}


def write_ugrid(mesh: Mesh, path: str | os.PathLike[str]) -> None:
    """Write a mesh as a UGRID-1.0 netCDF file that D-HYDRO and D-Flow FM read.

    Parameters
    ----------
    mesh : Mesh
        The mesh to write.
    path : str or path-like
        Output file; it is overwritten if it exists.

    Notes
    -----
    The structure is that of the D-HYDRO mesh generator export: the ``Mesh2d``
    topology with nodes, edges and faces, edge and face coordinates and face bounds.
    There is no bed level (``Mesh2d_node_z``, ``Mesh2d_face_z``): give it to D-Flow FM
    as samples in the field file. A placeholder bed level would outlive a MeshKernel
    clip in hydrolib and break saving the model, and the D-HYDRO GUI rejects a bed
    level per cell once it rebuilds the cells (ADR-0003). A mesh with CRS EPSG:28992
    gets exactly the CRS attributes of the D-Flow FM plugin; another CRS gets the same
    attributes derived with pyproj; a mesh without CRS gets no CRS variable.
    """
    crs_attributes = _crs_attributes(mesh.crs)
    vertices = mesh.nodes
    face_arr = np.where(mesh.faces >= 0, mesh.faces, _FILL_INT).astype(np.int32)
    n_faces, max_fn = face_arr.shape
    node_counts = (face_arr != _FILL_INT).sum(axis=1)
    edges = _unique_edges(face_arr, node_counts)

    edge_x = (vertices[edges[:, 0], 0] + vertices[edges[:, 1], 0]) * 0.5
    edge_y = (vertices[edges[:, 0], 1] + vertices[edges[:, 1], 1]) * 0.5
    face_x = np.zeros(n_faces, dtype=np.float64)
    face_y = np.zeros(n_faces, dtype=np.float64)
    for nk in range(3, max_fn + 1):
        mask = node_counts == nk
        if mask.any():
            fc = face_arr[mask, :nk]
            face_x[mask] = vertices[fc, 0].mean(axis=1)
            face_y[mask] = vertices[fc, 1].mean(axis=1)
    valid = face_arr != _FILL_INT
    face_x_bnd = np.full((n_faces, max_fn), _FILL_FLOAT)
    face_y_bnd = np.full((n_faces, max_fn), _FILL_FLOAT)
    face_x_bnd[valid] = vertices[face_arr[valid], 0]
    face_y_bnd[valid] = vertices[face_arr[valid], 1]

    from refined_quads import __version__

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("institution", "HKV / Svasek")
        ds.setncattr("references", "https://github.com/ugrid-conventions/ugrid-conventions")
        ds.setncattr("source", f"refined-quads {__version__}")
        ds.setncattr("history", datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        ds.setncattr("Conventions", "CF-1.8 UGRID-1.0 Deltares-0.10")

        if crs_attributes is not None:
            crs_var = ds.createVariable("projected_coordinate_system", "i4", ())
            for name, value in crs_attributes.items():
                crs_var.setncattr(name, value)
            crs_var[:] = crs_attributes["epsg"]

        ds.createDimension("Mesh2d_nNodes", len(vertices))
        ds.createDimension("Mesh2d_nEdges", len(edges))
        ds.createDimension("Mesh2d_nFaces", n_faces)
        ds.createDimension("Mesh2d_nMax_face_nodes", max_fn)
        ds.createDimension("Two", 2)

        topology = ds.createVariable("Mesh2d", "i4", ())
        for name, value in {
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D mesh",
            "topology_dimension": np.int32(2),
            "node_coordinates": "Mesh2d_node_x Mesh2d_node_y",
            "node_dimension": "Mesh2d_nNodes",
            "max_face_nodes_dimension": "Mesh2d_nMax_face_nodes",
            "edge_node_connectivity": "Mesh2d_edge_nodes",
            "edge_dimension": "Mesh2d_nEdges",
            "edge_coordinates": "Mesh2d_edge_x Mesh2d_edge_y",
            "face_node_connectivity": "Mesh2d_face_nodes",
            "face_dimension": "Mesh2d_nFaces",
            "face_coordinates": "Mesh2d_face_x Mesh2d_face_y",
        }.items():
            topology.setncattr(name, value)

        nodes_dim = ("Mesh2d_nNodes",)
        _float_var(ds, "Mesh2d_node_x", nodes_dim, vertices[:, 0],
                   "projection_x_coordinate", "x-coordinate of mesh nodes")  # fmt: skip
        _float_var(ds, "Mesh2d_node_y", nodes_dim, vertices[:, 1],
                   "projection_y_coordinate", "y-coordinate of mesh nodes")  # fmt: skip

        fn = ds.createVariable(
            "Mesh2d_face_nodes", "i4", ("Mesh2d_nFaces", "Mesh2d_nMax_face_nodes"),
            fill_value=_FILL_INT,
        )  # fmt: skip
        fn.setncattr("cf_role", "face_node_connectivity")
        fn.setncattr("long_name", "Vertex nodes of mesh faces (counterclockwise)")
        fn.setncattr("start_index", np.int32(1))
        fn[:] = np.where(face_arr == _FILL_INT, _FILL_INT, face_arr + 1)

        en = ds.createVariable("Mesh2d_edge_nodes", "i4", ("Mesh2d_nEdges", "Two"))
        en.setncattr("cf_role", "edge_node_connectivity")
        en.setncattr("long_name", "Start and end nodes of mesh edges")
        en.setncattr("start_index", np.int32(1))
        en[:] = edges + 1

        edges_dim = ("Mesh2d_nEdges",)
        edge_long_name = "characteristic {} of the mesh edge"
        _float_var(ds, "Mesh2d_edge_x", edges_dim, edge_x,
                   "projection_x_coordinate", edge_long_name.format("x-coordinate"))  # fmt: skip
        _float_var(ds, "Mesh2d_edge_y", edges_dim, edge_y,
                   "projection_y_coordinate", edge_long_name.format("y-coordinate"))  # fmt: skip

        faces_dim = ("Mesh2d_nFaces",)
        _float_var(ds, "Mesh2d_face_x", faces_dim, face_x,
                   "projection_x_coordinate", "Characteristic x-coordinate of mesh face",
                   bounds="Mesh2d_face_x_bnd")  # fmt: skip
        _float_var(ds, "Mesh2d_face_y", faces_dim, face_y,
                   "projection_y_coordinate", "Characteristic y-coordinate of mesh face",
                   bounds="Mesh2d_face_y_bnd")  # fmt: skip

        bnd_dims = ("Mesh2d_nFaces", "Mesh2d_nMax_face_nodes")
        for axis, data in (("x", face_x_bnd), ("y", face_y_bnd)):
            var = ds.createVariable(
                f"Mesh2d_face_{axis}_bnd", "f8", bnd_dims, fill_value=_FILL_FLOAT
            )
            var.setncattr("units", "m")
            var.setncattr("standard_name", f"projection_{axis}_coordinate")
            var.setncattr(
                "long_name", f"{axis}-coordinate bounds of mesh faces (i.e. corner coordinates)"
            )
            var[:] = data


def read_ugrid(path: str | os.PathLike[str]) -> Mesh:
    """Read the 2D mesh of a UGRID netCDF file.

    Parameters
    ----------
    path : str or path-like
        A UGRID file, such as a D-HYDRO net file; a 1D network or other variables in
        it are ignored.

    Returns
    -------
    Mesh
        The 2D mesh with its nodes and faces as stored (node order and orientation
        unchanged), the CRS of the file if it has one, and the lattice derived from
        the cells.

    Raises
    ------
    InputError
        If the file has no 2D mesh topology or more than one, or the mesh is not a
        refined quad mesh (its nodes do not lie on one lattice).
    """
    with netCDF4.Dataset(path, "r") as ds:
        topology = _mesh2d_topology(ds, path)
        x_name, y_name = str(topology.getncattr("node_coordinates")).split()
        nodes = np.column_stack(
            [np.ma.getdata(ds.variables[x_name][:]), np.ma.getdata(ds.variables[y_name][:])]
        ).astype(np.float64)
        face_var = ds.variables[str(topology.getncattr("face_node_connectivity"))]
        start = int(face_var.getncattr("start_index")) if "start_index" in face_var.ncattrs() else 0
        raw = face_var[:]
        data = np.ma.getdata(raw).astype(np.int64)
        padding = np.ma.getmaskarray(raw) | (data < start)
        faces = np.where(padding, -1, data - start)
        crs = _read_crs(ds)
    return Mesh(nodes, faces, derive_lattice(nodes, faces), crs=crs)


def _float_var(
    ds: netCDF4.Dataset,
    name: str,
    dims: tuple[str, ...],
    data: NDArray[np.float64],
    standard_name: str,
    long_name: str,
    bounds: str | None = None,
) -> None:
    var = ds.createVariable(name, "f8", dims)
    var.setncattr("units", "m")
    var.setncattr("standard_name", standard_name)
    var.setncattr("long_name", long_name)
    if bounds is not None:
        var.setncattr("bounds", bounds)
    var[:] = data


def _unique_edges(face_arr: NDArray[np.int32], node_counts: NDArray[np.int64]) -> NDArray[np.int32]:
    """List the unique edges in the order and orientation of the D-HYDRO export."""
    raw_parts: list[NDArray[np.int32]] = []
    for nk in range(3, face_arr.shape[1] + 1):
        mask = node_counts == nk
        if not mask.any():
            continue
        fc = face_arr[mask, :nk]
        raw_parts.extend(np.stack([fc[:, k], fc[:, (k + 1) % nk]], axis=1) for k in range(nk))
    edges_raw = np.concatenate(raw_parts, axis=0)
    _, first_idx = np.unique(np.sort(edges_raw, axis=1), axis=0, return_index=True)
    return np.asarray(edges_raw[first_idx])


def _crs_attributes(crs: CRS | None) -> dict[str, Any] | None:
    if crs is None:
        return None
    if crs.to_epsg() == 28992:
        return _RD_NEW_ATTRIBUTES
    return crs_cf_attributes(crs)


def _mesh2d_topology(ds: netCDF4.Dataset, path: object) -> netCDF4.Variable[Any]:
    found = [
        var
        for var in ds.variables.values()
        if "cf_role" in var.ncattrs()
        and var.getncattr("cf_role") == "mesh_topology"
        and int(var.getncattr("topology_dimension")) == 2
    ]
    if not found:
        raise InputError(f"{path}: no 2D mesh topology (cf_role mesh_topology, dimension 2)")
    if len(found) > 1:
        names = ", ".join(var.name for var in found)
        raise InputError(f"{path}: more than one 2D mesh topology ({names})")
    return found[0]


def _read_crs(ds: netCDF4.Dataset) -> CRS | None:
    """CRS from the grid mapping variable, by EPSG code or else WKT; None if absent."""
    for var in ds.variables.values():
        attrs = set(var.ncattrs())
        if "grid_mapping_name" not in attrs and var.name != "projected_coordinate_system":
            continue
        for key in ("epsg", "EPSG_code"):
            if key in attrs:
                value = var.getncattr(key)
                text = str(value)
                if text not in ("0", "", "EPSG:0"):
                    return CRS.from_user_input(int(value) if key == "epsg" else text)
        for key in ("wkt", "crs_wkt"):
            if key in attrs:
                return CRS.from_wkt(str(var.getncattr(key)))
    return None
