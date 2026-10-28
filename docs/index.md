# refined-quads

`refined-quads` generates and refines *refined quad meshes* for D-Flow FM: axis-aligned
squares on a fixed lattice, locally refined by repeated halving, with transition cells
only where refinement levels meet.

```python exec="on" session="home"
import sys

sys.path.insert(0, "docs/snippets")
from plot_mesh import svg
```

```python exec="on" source="above" session="home"
from shapely import Point, box

from refined_quads import generate

mesh = generate(
    box(155_000, 463_000, 155_640, 463_480),
    cell_size=32.0,
    refinement_areas=[(Point(155_250, 463_240).buffer(70), 3)],
    transition_rows=[1, 1, 1],
    crs="EPSG:28992",
)
```

```python exec="on" html="on" session="home"
print(svg(mesh, areas=[Point(155_250, 463_240).buffer(70)]))
```

!!! warning "Pre-alpha"
    The package is under construction. The API described here is not stable yet.

## What it does

- **Generate** a mesh over a domain with refinement areas, transition rows and water
  areas, in Python or from a TOML file on the command line.
- **Refine** an existing mesh — also one made by another program — with new refinement
  areas, without its original input; cells that do not change stay exactly as they are.
- **Check** a mesh for the properties D-Flow FM and refining rely on: no hanging nodes,
  2:1 balance and FM orthogonality.
- **Read and write** UGRID net files that D-HYDRO and D-Flow FM use.

## Install

The package is not on PyPI yet. Install it from the repository with uv:

```bash
uv add "refined-quads @ git+https://github.com/<organisation>/refined-quads"
```

Its dependencies (numpy, shapely, netCDF4, pyogrio, pyproj and meshkernel) all come as
wheels; no separate GDAL installation is needed.

## Where to go next

- [Your first mesh](tutorial/first-mesh.md) walks through generating, checking and
  writing a mesh.
- The **how-to guides** cover GIS layers, TOML files and the command line, water areas,
  refining and D-HYDRO.
- The **explanation** pages describe the concepts and the decisions behind the design.
- The [API reference](reference/api.md) documents every public function.
