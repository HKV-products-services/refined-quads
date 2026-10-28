"""Generate and refine refined quad meshes for D-Flow FM.

A refined quad mesh consists of axis-aligned squares on a fixed lattice, locally
refined by repeated halving, with transition cells only where refinement levels meet.
"""

from importlib.metadata import PackageNotFoundError, version

from refined_quads._check import MeshReport, check
from refined_quads._generate import generate
from refined_quads._gis import Areas, read_domain, read_refinement_areas, read_water_areas
from refined_quads._ugrid import read_ugrid, write_ugrid
from refined_quads.errors import InputError, OrthogonalityRepairError
from refined_quads.mesh import Lattice, Mesh
from refined_quads.spec import LayerSource, MeshSpec, RefinementSource

try:
    __version__ = version("refined-quads")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = [
    "Areas",
    "InputError",
    "Lattice",
    "LayerSource",
    "Mesh",
    "MeshReport",
    "MeshSpec",
    "OrthogonalityRepairError",
    "RefinementSource",
    "__version__",
    "check",
    "generate",
    "read_domain",
    "read_refinement_areas",
    "read_ugrid",
    "read_water_areas",
    "write_ugrid",
]
