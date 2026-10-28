"""Generate and refine refined quad meshes for D-Flow FM.

A refined quad mesh consists of axis-aligned squares on a fixed lattice, locally
refined by repeated halving, with transition cells only where refinement levels meet.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("refined-quads")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
