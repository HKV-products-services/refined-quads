"""Storage of golden-master meshes as integer lattice indices (ADR-0002).

A master is a compact ``.npz`` with the lattice (``origin``, ``cell_size``,
``levels``), the nodes as integer indices in units of ``cell_size / 2**levels`` from
the origin, the faces padded with -1, and a JSON ``meta`` string with the input and
provenance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely

from refined_quads import Lattice, Mesh

MASTERS_DIR = Path(__file__).parent / "masters"
RUNTIME_BASELINE = Path(__file__).parent / "runtime_baseline.json"
#: Directory with real-data masters (Geul, Parkstad); they stay outside the repository.
REALDATA_ENV = "REFINED_QUADS_REALDATA"


@dataclass(frozen=True)
class Master:
    name: str
    lattice: Lattice
    nodes: np.ndarray  # (n, 2) int64 lattice indices
    faces: np.ndarray  # (m, k) int64, padded with -1
    meta: dict

    def to_mesh(self) -> Mesh:
        return Mesh(self.lattice.to_coordinates(self.nodes), self.faces, self.lattice)


def save_master(path: Path, master: Master) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        origin=np.asarray(master.lattice.origin, dtype=np.float64),
        cell_size=np.float64(master.lattice.cell_size),
        levels=np.int64(master.lattice.levels),
        nodes=master.nodes.astype(np.int64),
        faces=master.faces.astype(np.int32),
        meta=np.str_(json.dumps(master.meta, sort_keys=True)),
    )


def load_master(path: Path) -> Master:
    with np.load(path, allow_pickle=False) as data:
        lattice = Lattice(
            origin=(float(data["origin"][0]), float(data["origin"][1])),
            cell_size=float(data["cell_size"]),
            levels=int(data["levels"]),
        )
        return Master(
            name=path.stem,
            lattice=lattice,
            nodes=data["nodes"].astype(np.int64),
            faces=data["faces"].astype(np.int64),
            meta=json.loads(str(data["meta"])),
        )


def master_paths(directory: Path = MASTERS_DIR) -> list[Path]:
    return sorted(directory.glob("*.npz"))


def input_fingerprint(scenario) -> str:
    """Hash of a scenario's input, to notice a scenario edited without re-recording.

    Coordinates are rounded to 1e-6 so that the hash does not depend on the last bit
    of buffered geometries across GEOS versions.
    """

    def geom_key(g):
        return shapely.to_wkt(g, rounding_precision=6, trim=True)

    payload = {
        "domain": geom_key(scenario.domain),
        "cell_size": scenario.cell_size,
        "refinement_areas": [[geom_key(g), level] for g, level in scenario.refinement_areas],
        "transition_rows": list(scenario.transition_rows),
        "water_areas": [geom_key(g) for g in scenario.water_areas],
        "water_removal": scenario.water_removal,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
