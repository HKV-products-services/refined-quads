"""Record golden-master meshes with the generator of qgis_dhydamo 9a324bc (ADR-0002).

The generator is read straight from the Git objects of a qgis_dhydamo clone, so the
state of its working tree does not matter. It needs only numpy and shapely.

Synthetic scenarios (``scenarios.py``) are recorded into ``tests/golden/masters``,
together with a runtime baseline per scenario in ``runtime_baseline.json``::

    uv run python tests/golden/record.py --qgis-dhydamo D:/Python_Projects/qgis_dhydamo

``--ugrid-reference`` writes the UGRID file of the old writer for one scenario into
``tests/golden/ugrid``, the reference for the structure and attributes of ``write_ugrid``.

Existing masters are never overwritten unless ``--rerecord`` is given; an intended
change of a master needs a changelog entry. Real-data scenarios (Geul, Parkstad) stay
outside the repository: describe them in a TOML manifest and record them into the
directory that ``REFINED_QUADS_REALDATA`` points to::

    uv run python tests/golden/record.py --qgis-dhydamo ... --realdata manifest.toml

See ``realdata_manifest.example.toml`` for the manifest format.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
import types
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golden.masters import (
    MASTERS_DIR,
    REALDATA_ENV,
    RUNTIME_BASELINE,
    Master,
    input_fingerprint,
    save_master,
)
from golden.scenarios import SCENARIOS, Scenario
from refined_quads import Lattice

LEGACY_COMMIT = "9a324bcbf6c40199f5baea6397f095391d4cdc8b"
LEGACY_ENGINE = "tools/runtime/mesh/meshgen/generate_quad_refinement.py"
LEGACY_WRITER = "tools/runtime/mesh/meshgen/mesh_builder.py"
LEGACY_CRS = "tools/runtime/mesh/net_crs.py"
UGRID_DIR = Path(__file__).parent / "ugrid"
UGRID_SCENARIO = "domain_with_hole"
LEGACY_WATER_OPTIONS = {
    "centre": "FACES_WITH_INCLUDED_CIRCUMCENTERS",
    "inside": "INSIDE_NOT_INTERSECTED",
    "touches": "INSIDE_AND_INTERSECTED",
}


def _legacy_dir(repo: Path) -> Path:
    """Extract the legacy generator files from the Git objects into a temporary directory."""
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{LEGACY_COMMIT}^{{commit}}"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if commit.returncode != 0:
        sys.exit(f"error: commit {LEGACY_COMMIT[:7]} not found in {repo}")
    directory = Path(tempfile.mkdtemp(prefix="legacy_meshgen_"))
    for name in (LEGACY_ENGINE, LEGACY_WRITER, LEGACY_CRS):
        source = subprocess.run(
            ["git", "-C", str(repo), "show", f"{LEGACY_COMMIT}:{name}"],
            capture_output=True, check=True,
        ).stdout  # fmt: skip
        (directory / Path(name).name).write_bytes(source)
    return directory


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_legacy_engine(repo: Path):
    """Import ``generate_quad_refinement`` as it was at the legacy commit."""
    return _import(_legacy_dir(repo) / "generate_quad_refinement.py", "legacy_engine")


def load_legacy_writer(repo: Path):
    """Import ``mesh_builder`` (the UGRID writer) as it was at the legacy commit.

    ``mesh_builder`` imports geopandas at module level for its layer reading, which the
    writer does not use; a stub module stands in so that geopandas is not needed.
    """
    directory = _legacy_dir(repo)
    sys.path.insert(0, str(directory))  # its fallback imports: generate_quad_refinement, net_crs
    sys.modules.setdefault("geopandas", types.ModuleType("geopandas"))
    return _import(directory / "mesh_builder.py", "legacy_mesh_builder")


def record_ugrid_reference(repo: Path, rerecord: bool) -> None:
    """Write the legacy UGRID file of one scenario, the reference for the writer's structure."""
    target = UGRID_DIR / f"{UGRID_SCENARIO}_legacy.nc"
    if target.exists() and not rerecord:
        sys.exit(f"error: {target.name} exists; pass --rerecord to overwrite it")
    writer = load_legacy_writer(repo)
    scenario = next(s for s in SCENARIOS if s.name == UGRID_SCENARIO)
    vertices, cells, _, _ = run_legacy(writer, scenario, repeat=1)
    UGRID_DIR.mkdir(exist_ok=True)
    writer.write_netcdf_ugrid(str(target), vertices, cells)
    print(f"{target}: {len(cells)} cells")


def legacy_params(cell_size, levels, transition_rows, water_removal):
    """Translate input in this package's terms to the legacy parameters.

    The legacy generator takes the transition rows ("buffer_rows") from fine to coarse
    and caps the levels so that the smallest cell is at least 1 m; a capped scenario
    would not record what its input says, so it is refused.
    """
    max_possible = math.floor(math.log2(cell_size)) if cell_size > 1 else 0
    if levels > max_possible:
        raise ValueError(
            f"{levels} levels with a {cell_size:g} m base cell would be capped to "
            f"{max_possible} by the legacy generator"
        )
    return {
        "cell_size": cell_size,
        "max_refinement": levels,
        "buffer_rows": list(reversed(transition_rows)),
        "water_delete_option": LEGACY_WATER_OPTIONS[water_removal],
    }


def run_legacy(engine, scenario: Scenario, repeat: int):
    """Run the legacy generator; return its output and the best runtime of ``repeat`` runs."""
    params = legacy_params(
        scenario.cell_size, scenario.levels, scenario.transition_rows, scenario.water_removal
    )
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        vertices, cells = engine.generate_quad_refinement_mesh(
            scenario.domain,
            list(scenario.water_areas),
            [(g, level) for g, level in scenario.refinement_areas],
            dict(params),
        )
        times.append(time.perf_counter() - t0)
    return np.asarray(vertices, dtype=np.float64), cells, params, times


def to_master(name, scenario: Scenario, vertices, cells, params, meta) -> Master:
    """Convert legacy output to lattice indices, keeping only nodes used by a cell."""
    minx, miny = scenario.domain.bounds[:2]
    origin = (
        math.floor(minx / scenario.cell_size) * scenario.cell_size,
        math.floor(miny / scenario.cell_size) * scenario.cell_size,
    )
    lattice = Lattice(origin=origin, cell_size=scenario.cell_size, levels=scenario.levels)
    used = sorted({n for cell in cells for n in cell})
    renumber = {old: new for new, old in enumerate(used)}
    indices, on_lattice = lattice.to_indices(vertices[used])
    if not on_lattice.all():
        raise ValueError(f"{name}: {int((~on_lattice).sum())} nodes are not lattice points")
    width = max(len(cell) for cell in cells)
    faces = np.full((len(cells), width), -1, dtype=np.int64)
    for i, cell in enumerate(cells):
        faces[i, : len(cell)] = [renumber[n] for n in cell]
    meta = {
        **meta,
        "legacy_commit": LEGACY_COMMIT,
        "legacy_params": params,
        "recorded": date.today().isoformat(),
        "n_cells": len(cells),
    }
    return Master(name=name, lattice=lattice, nodes=indices, faces=faces, meta=meta)


def select_scenarios(names, rerecord) -> list[Scenario]:
    """Scenarios to record; refuses to overwrite existing masters without ``rerecord``."""
    unknown = set(names or ()) - {s.name for s in SCENARIOS}
    if unknown:
        sys.exit(f"error: unknown scenarios: {', '.join(sorted(unknown))}")
    selected = [s for s in SCENARIOS if not names or s.name in names]
    existing = [s.name for s in selected if (MASTERS_DIR / f"{s.name}.npz").exists()]
    if existing and not rerecord:
        sys.exit(
            "error: masters already exist for " + ", ".join(existing) + ". Re-recording "
            "changes the output contract: pass --rerecord and add a CHANGELOG entry."
        )
    return selected


def record_synthetic(engine, selected, repeat):
    baseline = json.loads(RUNTIME_BASELINE.read_text()) if RUNTIME_BASELINE.exists() else {}
    baseline.setdefault("scenarios", {})
    for scenario in selected:
        vertices, cells, params, times = run_legacy(engine, scenario, repeat)
        meta = {
            "scenario": scenario.name,
            "description": scenario.description,
            "transition_rows": list(scenario.transition_rows),
            "water_removal": scenario.water_removal,
            "input_fingerprint": input_fingerprint(scenario),
        }
        master = to_master(scenario.name, scenario, vertices, cells, params, meta)
        save_master(MASTERS_DIR / f"{scenario.name}.npz", master)
        baseline["scenarios"][scenario.name] = {
            "seconds": round(min(times), 4),
            "median_seconds": round(statistics.median(times), 4),
            "runs": repeat,
            "cells": len(cells),
        }
        print(f"{scenario.name}: {len(cells)} cells, best of {repeat}: {min(times):.3f} s")
    baseline["machine"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "shapely": __import__("shapely").__version__,
    }
    baseline["generator"] = f"qgis_dhydamo {LEGACY_COMMIT[:7]}"
    RUNTIME_BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")


# --- real data -------------------------------------------------------------------------


def read_layer(source: dict, base: Path, crs_out):
    """Read the geometries (and attribute rows) of a layer, transformed to ``crs_out``."""
    import pyogrio
    import pyproj
    import shapely

    path = base / source["path"]
    meta, _, wkb, field_data = pyogrio.raw.read(path, layer=source.get("layer"))
    if meta["crs"] is None:
        raise ValueError(f"{path}: layer has no CRS")
    geoms = shapely.from_wkb(wkb)
    crs_in = pyproj.CRS.from_user_input(meta["crs"])
    if crs_out is not None and not crs_in.equals(crs_out):
        transformer = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
        geoms = shapely.transform(geoms, lambda xy: np.column_stack(transformer.transform(*xy.T)))
    fields = dict(zip(meta["fields"], field_data, strict=True))
    return geoms, fields, crs_in


def realdata_scenario(entry: dict, base: Path) -> Scenario:
    import shapely

    domain_geoms, _, crs = read_layer(entry["domain"], base, None)
    domain = shapely.union_all([g for g in domain_geoms if g is not None and not g.is_empty])
    ref = entry["refinement_areas"]
    geoms, fields, _ = read_layer(ref, base, crs)
    level = ref["level"]
    areas = []
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        value = fields[level][i] if isinstance(level, str) else level
        if value is None or int(value) != value or int(value) < 1:
            raise ValueError(f"refinement area {i}: invalid level {value!r}")
        areas.append((g, int(value)))
    water = ()
    if "water_areas" in entry:
        wg, _, _ = read_layer(entry["water_areas"], base, crs)
        water = tuple(g for g in wg if g is not None and not g.is_empty)
    return Scenario(
        name=entry["name"],
        description=entry.get("description", ""),
        domain=domain,
        cell_size=float(entry["cell_size"]),
        refinement_areas=tuple(areas),
        transition_rows=tuple(entry["transition_rows"]),
        water_areas=water,
        water_removal=entry.get("water_removal", "centre"),
    )


def record_realdata(engine, manifest: Path, out: Path, rerecord, repeat):
    entries = tomllib.loads(manifest.read_text(encoding="utf-8"))["scenario"]
    for entry in entries:
        target = out / f"{entry['name']}.npz"
        if target.exists() and not rerecord:
            print(f"{entry['name']}: exists, skipped (pass --rerecord to overwrite)")
            continue
        scenario = realdata_scenario(entry, manifest.parent)
        vertices, cells, params, times = run_legacy(engine, scenario, repeat)
        meta = {
            "scenario": scenario.name,
            "manifest": str(manifest.resolve()),
            "input": entry,
            "seconds": round(min(times), 4),
        }
        save_master(target, to_master(scenario.name, scenario, vertices, cells, params, meta))
        print(f"{scenario.name}: {len(cells)} cells, best of {repeat}: {min(times):.1f} s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qgis-dhydamo", type=Path, required=True,
                        help="clone of qgis_dhydamo that contains commit 9a324bc")  # fmt: skip
    parser.add_argument("--rerecord", action="store_true",
                        help="overwrite existing masters (needs a CHANGELOG entry)")  # fmt: skip
    parser.add_argument("--only", nargs="+", metavar="NAME", help="record only these scenarios")
    parser.add_argument("--repeat", type=int, default=3, help="runs per scenario for the timing")
    parser.add_argument("--ugrid-reference", action="store_true",
                        help=f"write the old UGRID file of {UGRID_SCENARIO}")  # fmt: skip
    parser.add_argument("--realdata", type=Path, metavar="MANIFEST",
                        help="record the real-data scenarios of this TOML manifest")  # fmt: skip
    parser.add_argument("--out", type=Path, default=os.environ.get(REALDATA_ENV),
                        help=f"real-data output directory (default: ${REALDATA_ENV})")  # fmt: skip
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.ugrid_reference:
        record_ugrid_reference(args.qgis_dhydamo, args.rerecord)
    elif args.realdata:
        if args.out is None:
            parser.error(f"--realdata needs --out or ${REALDATA_ENV}")
        engine = load_legacy_engine(args.qgis_dhydamo)
        record_realdata(engine, args.realdata, Path(args.out), args.rerecord, args.repeat)
    else:
        selected = select_scenarios(args.only, args.rerecord)
        engine = load_legacy_engine(args.qgis_dhydamo)
        record_synthetic(engine, selected, args.repeat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
