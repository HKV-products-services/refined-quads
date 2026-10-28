"""Generate and refine refined quad meshes for D-Flow FM.

A refined quad mesh consists of axis-aligned squares on a fixed lattice, locally
refined by repeated halving, with transition cells only where refinement levels meet.
"""

from importlib.metadata import PackageNotFoundError, version

from refined_quads._check import MeshReport, check
from refined_quads._generate import generate
from refined_quads.errors import InputError, OrthogonalityRepairError
from refined_quads.mesh import Lattice, Mesh

try:
    __version__ = version("refined-quads")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = [
    "InputError",
    "Lattice",
    "Mesh",
    "MeshReport",
    "OrthogonalityRepairError",
    "__version__",
    "check",
    "generate",
]
