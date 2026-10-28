"""The input of a run as a TOML file: ``MeshSpec``."""

from __future__ import annotations

import math
import numbers
import os
import tomllib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

from refined_quads._engine import WATER_REMOVAL_RULES
from refined_quads._generate import WaterRemovalRule, generate
from refined_quads._gis import (
    Areas,
    read_domain,
    read_refinement_areas,
    read_water_areas,
)
from refined_quads.errors import InputError
from refined_quads.mesh import Mesh


@dataclass(frozen=True)
class LayerSource:
    """A layer of a GeoPackage or shapefile.

    Attributes
    ----------
    path : pathlib.Path
        The file.
    layer : str or None
        Layer name; may be omitted when the file has one layer.
    """

    path: Path
    layer: str | None = None


@dataclass(frozen=True)
class RefinementSource(LayerSource):
    """A layer with refinement areas.

    Attributes
    ----------
    level : str or int
        Column with the refinement level of each polygon, or one level for all.
    """

    level: str | int = "refinement"


@dataclass(frozen=True)
class MeshSpec:
    """Everything ``refined-quads generate`` needs; one-to-one with the TOML file.

    Attributes
    ----------
    cell_size : float
        Base cell size.
    domain : LayerSource
        Layer with the domain polygons.
    refinement_areas : tuple of RefinementSource
        Layers with refinement areas.
    transition_rows : int or tuple of int
        Transition rows per level step from coarse to fine, or one value for all.
    water_areas : tuple of LayerSource
        Layers with water areas.
    water_removal : str
        Water removal rule: ``"centre"``, ``"inside"`` or ``"touches"``.
    crs : str or None
        CRS of layers that have none.
    origin : tuple of float or None
        A lattice point, ``[x, y]`` in the TOML; by default a multiple of the cell
        size.
    output : pathlib.Path or None
        UGRID file to write.

    Examples
    --------
    A TOML file, with paths relative to the file::

        cell_size = 25.0
        transition_rows = [1, 1, 2]
        water_removal = "centre"
        output = "mesh_net.nc"

        [domain]
        path = "model.gpkg"
        layer = "modelgebied"

        [[refinement_areas]]
        path = "model.gpkg"
        layer = "verfijning"
        level = "refinement"   # column name, or one integer level

        [[water_areas]]
        path = "model.gpkg"
        layer = "water"
    """

    cell_size: float
    domain: LayerSource
    refinement_areas: tuple[RefinementSource, ...] = ()
    transition_rows: int | tuple[int, ...] = 1
    water_areas: tuple[LayerSource, ...] = ()
    water_removal: str = "centre"
    crs: str | None = None
    origin: tuple[float, float] | None = None
    output: Path | None = None

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> MeshSpec:
        """Read a spec from a TOML file.

        Parameters
        ----------
        path : str or path-like
            The TOML file. Relative paths in it are relative to its directory.

        Returns
        -------
        MeshSpec
            The validated spec.

        Raises
        ------
        InputError
            If the file cannot be read, is not valid TOML, or a key is unknown,
            missing or has an invalid value; the message names the key and the reason.
        """
        file = Path(path)
        try:
            data = tomllib.loads(file.read_text(encoding="utf-8"))
        except OSError as exc:
            raise InputError(f"{file}: cannot read: {exc.strerror or exc}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise InputError(f"{file}: not valid TOML: {exc}") from exc
        try:
            return cls.from_dict(data, base=file.parent)
        except InputError as exc:
            raise InputError(f"{file}: {exc}") from exc

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], base: str | os.PathLike[str] = ".") -> MeshSpec:
        """Build a spec from the contents of a TOML file.

        Parameters
        ----------
        data : mapping
            The parsed TOML.
        base : str or path-like
            Directory that relative paths are relative to.

        Returns
        -------
        MeshSpec
            The validated spec.

        Raises
        ------
        InputError
            If a key is unknown, missing or has an invalid value.
        """
        base_dir = Path(base)
        _known(data, {f.name for f in fields(cls)}, "")
        if "cell_size" not in data:
            raise InputError("cell_size: missing")
        if "domain" not in data:
            raise InputError("domain: missing")
        refinement = _list_of_tables(data.get("refinement_areas", []), "refinement_areas")
        water = _list_of_tables(data.get("water_areas", []), "water_areas")
        output = data.get("output")
        return cls(
            cell_size=_cell_size(data["cell_size"], "cell_size"),
            domain=_layer(data["domain"], "domain", base_dir),
            refinement_areas=tuple(
                _refinement(t, f"refinement_areas[{i}]", base_dir) for i, t in enumerate(refinement)
            ),
            transition_rows=_rows(data.get("transition_rows", 1), "transition_rows"),
            water_areas=tuple(
                _layer(t, f"water_areas[{i}]", base_dir) for i, t in enumerate(water)
            ),
            water_removal=_rule(data.get("water_removal", "centre"), "water_removal"),
            crs=_optional_str(data.get("crs"), "crs"),
            origin=_point(data.get("origin"), "origin"),
            output=None if output is None else base_dir / _string(output, "output"),
        )

    def replace(self, **changes: Any) -> MeshSpec:  # noqa: ANN401
        """Return a copy with some values replaced, validated like the TOML values.

        Parameters
        ----------
        **changes
            New values for ``cell_size``, ``transition_rows``, ``water_removal``,
            ``crs``, ``origin`` or ``output``.

        Returns
        -------
        MeshSpec
            The new spec.

        Raises
        ------
        InputError
            If a value is invalid.
        """
        checks: dict[str, Callable[[Any, str], Any]] = {
            "cell_size": _cell_size,
            "transition_rows": _rows,
            "water_removal": _rule,
            "crs": _optional_str,
            "origin": _point,
            "output": _output,
        }
        unknown = set(changes) - set(checks)
        if unknown:
            raise InputError(f"cannot replace {sorted(unknown)}; only {sorted(checks)}")
        new: dict[str, Any] = {k: checks[k](v, k) for k, v in changes.items()}
        return replace(self, **new)

    def generate(self, progress: Callable[[float, str], object] | None = None) -> Mesh:
        """Read the layers and generate the mesh.

        Parameters
        ----------
        progress : callable, optional
            Progress callback, as for :func:`generate`.

        Returns
        -------
        Mesh
            The mesh, in the CRS of the domain.

        Raises
        ------
        InputError
            If a layer is invalid (see :func:`read_domain` and :func:`generate`).
        OrthogonalityRepairError
            If the orthogonality repair does not close.
        """
        with _key("domain"):
            domain = read_domain(self.domain.path, self.domain.layer, crs=self.crs)
        refinement = []
        for i, src in enumerate(self.refinement_areas):
            with _key(f"refinement_areas[{i}]"):
                refinement.append(
                    read_refinement_areas(src.path, src.layer, level=src.level, crs=self.crs)
                )
        water = []
        for i, layer in enumerate(self.water_areas):
            with _key(f"water_areas[{i}]"):
                water.append(read_water_areas(layer.path, layer.layer, crs=self.crs))
        return generate(
            domain,
            cell_size=self.cell_size,
            refinement_areas=_combine(refinement, domain, with_levels=True),
            transition_rows=self.transition_rows,
            water_areas=_combine(water, domain, with_levels=False),
            water_removal=cast("WaterRemovalRule", self.water_removal),
            origin=self.origin,
            progress=progress,
        )


@contextmanager
def _key(key: str) -> Iterator[None]:
    """Prefix an input error with the TOML key it comes from."""
    try:
        yield
    except InputError as exc:
        raise InputError(f"{key}: {exc}") from exc


def _combine(layers: list[Areas], domain: Areas, *, with_levels: bool) -> Areas:
    """Merge layers into one set of areas in the CRS of the domain."""
    in_domain_crs = [areas.to_crs(domain.crs) for areas in layers]
    geometries = tuple(g for areas in in_domain_crs for g in areas.geometries)
    levels = (
        tuple(level for areas in in_domain_crs for level in (areas.levels or ()))
        if with_levels
        else None
    )
    return Areas(geometries, domain.crs, levels)


def _known(table: Mapping[str, Any], known: set[str], prefix: str) -> None:
    unknown = sorted(set(table) - known)
    if unknown:
        raise InputError(
            f"{prefix}{unknown[0]}: unknown key; known keys are {', '.join(sorted(known))}"
        )


def _table(value: object, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputError(f"{key}: must be a table with path (and layer), got {value!r}")
    return value


def _list_of_tables(value: object, key: str) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if not isinstance(value, list):
        raise InputError(f"{key}: must be a list of tables ([[{key}]]), got {value!r}")
    return [_table(v, f"{key}[{i}]") for i, v in enumerate(value)]


def _layer(value: object, key: str, base: Path) -> LayerSource:
    table = _table(value, key)
    _known(table, {"path", "layer"}, f"{key}.")
    return LayerSource(*_path_and_layer(table, key, base))


def _refinement(table: Mapping[str, Any], key: str, base: Path) -> RefinementSource:
    _known(table, {"path", "layer", "level"}, f"{key}.")
    if "level" not in table:
        raise InputError(f"{key}.level: missing (a column name or one integer level)")
    level = table["level"]
    if not isinstance(level, str):
        number = _integer(level)
        if number is None or number < 1:
            raise InputError(
                f"{key}.level: must be a column name or an integer of at least 1, got {level!r}"
            )
        level = number
    return RefinementSource(*_path_and_layer(table, key, base), level=level)


def _path_and_layer(table: Mapping[str, Any], key: str, base: Path) -> tuple[Path, str | None]:
    if "path" not in table:
        raise InputError(f"{key}.path: missing")
    return base / _string(table["path"], f"{key}.path"), _optional_str(
        table.get("layer"), f"{key}.layer"
    )


def _string(value: object, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(f"{key}: must be a non-empty string, got {value!r}")
    return value


def _optional_str(value: object, key: str) -> str | None:
    return None if value is None else _string(value, key)


def _point(value: object, key: str) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = list(value) if isinstance(value, list | tuple) else []
    if len(parts) != 2 or not all(
        isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v) for v in parts
    ):
        raise InputError(f"{key}: must be two finite numbers [x, y], got {value!r}")
    return float(parts[0]), float(parts[1])


def _output(value: object, key: str) -> Path | None:
    if value is None:
        return None
    return Path(value) if isinstance(value, os.PathLike) else Path(_string(value, key))


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    return None


def _cell_size(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InputError(f"{key}: must be a number, got {value!r}")
    size = float(value)
    if not (math.isfinite(size) and size > 0):
        raise InputError(f"{key}: must be a positive length, got {value!r}")
    return size


def _rows(value: object, key: str) -> int | tuple[int, ...]:
    values = list(value) if isinstance(value, list | tuple) else [value]
    rows = []
    for i, raw in enumerate(values):
        n = _integer(raw)
        if n is None or n < 1:
            where = key if not isinstance(value, list | tuple) else f"{key}[{i}]"
            raise InputError(f"{where}: must be an integer of at least 1, got {raw!r}")
        rows.append(n)
    return tuple(rows) if isinstance(value, list | tuple) else rows[0]


def _rule(value: object, key: str) -> str:
    if value not in WATER_REMOVAL_RULES:
        allowed = ", ".join(repr(rule) for rule in WATER_REMOVAL_RULES)
        raise InputError(f"{key}: must be one of {allowed}, got {value!r}")
    return str(value)
