"""GIS input: polygons with a CRS, read from a GeoPackage or shapefile layer."""

from __future__ import annotations

import math
import numbers
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio
import shapely
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from shapely.geometry.base import BaseGeometry

from refined_quads.errors import InputError

_POLYGONAL = ("Polygon", "MultiPolygon")


@dataclass(frozen=True)
class Areas:
    """Polygons with their coordinate reference system.

    Parameters
    ----------
    geometries : sequence of shapely geometry
        The polygons.
    crs : pyproj.CRS or str or int
        CRS of the coordinates; anything ``pyproj.CRS.from_user_input`` accepts.
    levels : sequence of int, optional
        Refinement level per polygon, for refinement areas.

    Raises
    ------
    InputError
        If the CRS is invalid or the number of levels differs from the number of
        polygons.
    """

    geometries: tuple[BaseGeometry, ...]
    crs: CRS
    levels: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Normalise the fields."""
        object.__setattr__(self, "geometries", tuple(self.geometries))
        object.__setattr__(self, "crs", _crs(self.crs))
        if self.levels is not None:
            levels = tuple(self.levels)
            if len(levels) != len(self.geometries):
                raise InputError(
                    f"{len(levels)} levels for {len(self.geometries)} polygons; need one each"
                )
            object.__setattr__(self, "levels", levels)

    def to_crs(self, crs: CRS) -> Areas:
        """Return the polygons in another CRS.

        Parameters
        ----------
        crs : pyproj.CRS
            Target CRS.

        Returns
        -------
        Areas
            The same polygons and levels with transformed coordinates.
        """
        if self.crs == crs:
            return self
        transformer = Transformer.from_crs(self.crs, crs, always_xy=True)

        def transform(xy: NDArray[np.float64]) -> NDArray[np.float64]:
            x, y = transformer.transform(xy[:, 0], xy[:, 1])
            return np.column_stack([x, y])

        geometries = shapely.transform(np.asarray(self.geometries, dtype=object), transform)
        return Areas(tuple(geometries), crs, self.levels)


def read_domain(
    source: object,
    layer: str | None = None,
    *,
    crs: object = None,
) -> Areas:
    """Read the domain from a GIS layer; its polygons are merged into one.

    Parameters
    ----------
    source : str, path-like or GeoDataFrame
        A GeoPackage or shapefile, or an object with ``geometry`` and ``crs``
        attributes such as a GeoDataFrame (geopandas itself is not needed).
    layer : str, optional
        Layer name; required when the file has more than one layer.
    crs : pyproj.CRS or str or int, optional
        CRS of a layer that has none. A layer without CRS raises an error unless it
        is given; a layer with a different CRS raises an error too.

    Returns
    -------
    Areas
        One (multi)polygon with the CRS of the layer.

    Raises
    ------
    InputError
        If the file or layer cannot be read, the layer has no CRS and none is given,
        a feature has no geometry or is not a polygon, or the layer is empty.
    """
    areas = _read(source, layer, crs, "domain")
    return Areas((shapely.union_all(np.asarray(areas.geometries, dtype=object)),), areas.crs)


def read_refinement_areas(
    source: object,
    layer: str | None = None,
    *,
    level: str | int,
    crs: object = None,
) -> Areas:
    """Read refinement areas from a GIS layer.

    Parameters
    ----------
    source : str, path-like or GeoDataFrame
        A GeoPackage or shapefile, or an object with ``geometry`` and ``crs``
        attributes such as a GeoDataFrame.
    layer : str, optional
        Layer name; required when the file has more than one layer.
    level : str or int
        Name of the column with the refinement level of each polygon, or one level
        for all polygons.
    crs : pyproj.CRS or str or int, optional
        CRS of a layer that has none (see :func:`read_domain`).

    Returns
    -------
    Areas
        The polygons with their levels.

    Raises
    ------
    InputError
        As :func:`read_domain`, and if the level column is missing or a level is
        missing, not an integer or below 1; the message names the feature.
    """
    areas, fids, fields = _read_with_fields(source, layer, crs, "refinement area")
    if isinstance(level, str):
        if level not in fields:
            raise InputError(f"refinement areas: no column {level!r}; columns are {sorted(fields)}")
        values: Sequence[object] = fields[level]
    else:
        values = [level] * len(areas.geometries)
    levels = []
    for fid, value in zip(fids, values, strict=True):
        number = _integer(value)
        if number is None or number < 1:
            raise InputError(
                f"refinement area (feature {fid}): the level must be an integer of at least "
                f"1, got {value!r}"
            )
        levels.append(number)
    return Areas(areas.geometries, areas.crs, tuple(levels))


def read_water_areas(
    source: object,
    layer: str | None = None,
    *,
    crs: object = None,
) -> Areas:
    """Read water areas from a GIS layer.

    Parameters
    ----------
    source : str, path-like or GeoDataFrame
        A GeoPackage or shapefile, or an object with ``geometry`` and ``crs``
        attributes such as a GeoDataFrame.
    layer : str, optional
        Layer name; required when the file has more than one layer.
    crs : pyproj.CRS or str or int, optional
        CRS of a layer that has none (see :func:`read_domain`).

    Returns
    -------
    Areas
        The polygons.

    Raises
    ------
    InputError
        As :func:`read_domain`.
    """
    return _read(source, layer, crs, "water area")


def is_frame(value: object) -> bool:
    """Tell whether a value looks like a GeoDataFrame (``geometry`` and ``crs`` attributes)."""
    return (
        not isinstance(value, BaseGeometry | str | os.PathLike)
        and hasattr(value, "geometry")
        and hasattr(value, "crs")
    )


def frame_areas(frame: Any, crs: object, what: str) -> Areas:  # noqa: ANN401
    """Turn a GeoDataFrame-like object into areas, applying the CRS rules of the readers."""
    areas, _, _ = _from_frame(frame, crs, what)
    return areas


def _read(source: object, layer: str | None, crs: object, what: str) -> Areas:
    areas, _, _ = _read_with_fields(source, layer, crs, what)
    return areas


def _read_with_fields(
    source: object, layer: str | None, crs: object, what: str
) -> tuple[Areas, list[object], dict[str, list[object]]]:
    if is_frame(source):
        return _from_frame(source, crs, what)
    if not isinstance(source, str | os.PathLike):
        raise InputError(
            f"{what}: expected a file path or a GeoDataFrame, got {type(source).__name__}"
        )
    path = os.fspath(source)
    try:
        layers = [str(name) for name, _ in pyogrio.list_layers(path)]
    except Exception as exc:  # pyogrio raises its own error types per driver
        raise InputError(f"{what}: cannot read {path}: {exc}") from exc
    if layer is None:
        if len(layers) != 1:
            raise InputError(f"{what}: {path} has layers {layers}; name the layer to read")
        layer = layers[0]
    elif layer not in layers:
        raise InputError(f"{what}: {path} has no layer {layer!r}; layers are {layers}")
    meta, fids, wkb, field_data = pyogrio.raw.read(path, layer=layer, return_fids=True)
    source_name = f"{path}, layer {layer!r}"
    layer_crs = _layer_crs(meta["crs"], crs, f"{what} ({source_name})")
    geometries = _polygons(shapely.from_wkb(wkb), list(fids), f"{what} ({source_name})")
    fields = {
        str(name): list(values) for name, values in zip(meta["fields"], field_data, strict=True)
    }
    return Areas(geometries, layer_crs), list(fids), fields


def _from_frame(
    frame: Any,  # noqa: ANN401 (duck-typed GeoDataFrame)
    crs: object,
    what: str,
) -> tuple[Areas, list[object], dict[str, list[object]]]:
    layer_crs = _layer_crs(frame.crs, crs, what)
    raw = list(frame.geometry)
    index = list(frame.index) if hasattr(frame, "index") else list(range(len(raw)))
    geometries = _polygons(np.asarray(raw, dtype=object), index, what)
    geometry_name = getattr(frame.geometry, "name", None) or "geometry"
    columns = [c for c in getattr(frame, "columns", []) if c != geometry_name]
    fields = {str(c): list(frame[c]) for c in columns}
    return Areas(geometries, layer_crs), index, fields


def _layer_crs(own: object, given: object, what: str) -> CRS:
    if own is None or own == "":
        if given is None:
            raise InputError(f"{what} has no CRS; pass crs= to say which CRS it is in")
        return _crs(given)
    layer_crs = _crs(own)
    if given is not None and _crs(given) != layer_crs:
        raise InputError(
            f"{what} has CRS {layer_crs.to_string()}, but crs={_crs(given).to_string()} was "
            f"given; crs= is only for a layer without CRS"
        )
    return layer_crs


def _polygons(
    geometries: NDArray[np.object_], fids: Sequence[object], what: str
) -> tuple[BaseGeometry, ...]:
    if len(geometries) == 0:
        raise InputError(f"{what}: the layer has no features")
    for fid, geom in zip(fids, geometries, strict=True):
        if geom is None or geom.is_empty:
            raise InputError(f"{what}: feature {fid} has no geometry")
        if geom.geom_type not in _POLYGONAL:
            raise InputError(f"{what}: feature {fid} is a {geom.geom_type}, not a polygon")
    return tuple(geometries)


def _crs(value: object) -> CRS:
    try:
        return CRS.from_user_input(value)
    except CRSError as exc:
        raise InputError(f"invalid CRS {value!r}: {exc}") from exc


def _integer(value: object) -> int | None:
    """Return ``value`` as int if it is an integral number (not a bool), else None."""
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    return None


def crs_cf_attributes(crs: CRS) -> dict[str, Any]:
    """Derive the UGRID grid-mapping attributes of a projected CRS from pyproj.

    Parameters
    ----------
    crs : pyproj.CRS
        A projected CRS.

    Returns
    -------
    dict
        The attributes the D-Flow FM plugin writes, in the same order and types.
    """
    epsg = crs.to_epsg() or 0
    ellipsoid = crs.ellipsoid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # to_proj4 warns that PROJ strings lose information
        proj4 = crs.to_proj4()
        cf = crs.to_cf()
    operation = crs.coordinate_operation
    return {
        "name": crs.name,
        "epsg": np.int32(epsg),
        "grid_mapping_name": str(cf.get("grid_mapping_name", "Unknown projected")),
        "longitude_of_prime_meridian": np.float64(
            crs.prime_meridian.longitude if crs.prime_meridian else 0.0
        ),
        "semi_major_axis": np.float64(ellipsoid.semi_major_metre if ellipsoid else math.nan),
        "semi_minor_axis": np.float64(ellipsoid.semi_minor_metre if ellipsoid else math.nan),
        "inverse_flattening": np.float64(ellipsoid.inverse_flattening if ellipsoid else math.nan),
        "EPSG_code": f"EPSG:{epsg}" if epsg else "",
        "value": "value is equal to EPSG code",
        "proj4_params": proj4,
        "projection_name": operation.method_name if operation else "unknown",
        "wkt": crs.to_wkt("WKT1_GDAL", pretty=True),
    }


def write_points(
    path: str | os.PathLike[str],
    layer: str,
    xy: Sequence[tuple[float, float]],
    fields: dict[str, NDArray[Any]],
    crs: CRS | None,
) -> None:
    """Write points with attributes as a layer of a new GeoPackage.

    Parameters
    ----------
    path : str or path-like
        GeoPackage to write; an existing file is replaced.
    layer : str
        Layer name.
    xy : sequence of (float, float)
        The points.
    fields : dict of str to numpy.ndarray
        Attribute columns, one value per point.
    crs : pyproj.CRS or None
        CRS of the points.
    """
    target = Path(path)
    target.unlink(missing_ok=True)
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    with warnings.catch_warnings():
        # Without a CRS (a mesh of unknown CRS) pyogrio warns; that is intended here.
        warnings.filterwarnings("ignore", "'crs' was not provided", UserWarning)
        pyogrio.raw.write(
            target,
            shapely.to_wkb(shapely.points(points)),
            field_data=list(fields.values()),
            fields=list(fields),
            crs=None if crs is None else crs.to_wkt(),
            geometry_type="Point",
            layer=layer,
            driver="GPKG",
        )


def write_problem_points(
    path: str | os.PathLike[str],
    points: Sequence[tuple[float, float, float | None]],
    crs: CRS | None,
) -> None:
    """Write orthogonality problem locations as a point layer to a new GeoPackage.

    Parameters
    ----------
    path : str or path-like
        GeoPackage to write; an existing file is replaced.
    points : sequence of (x, y, orth_e)
        Location and FM orthogonality; ``None`` (a degenerate edge) is written as -1.
    crs : pyproj.CRS or None
        CRS of the points.
    """
    orth = np.array([-1.0 if o is None else o for _, _, o in points], dtype=np.float64)
    write_points(
        path, "orthogonality_problems", [(x, y) for x, y, _ in points], {"orth_e": orth}, crs
    )
