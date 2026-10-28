# Your first mesh

This tutorial generates a small refined quad mesh from shapely geometries, looks at it,
checks it and writes it as a UGRID net file.

```python exec="on" session="tutorial"
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "docs/snippets")
from plot_mesh import svg

workdir = Path(tempfile.mkdtemp())
```

## The input

A mesh needs a **domain**, the area it covers, and a **base cell**, the size of its
coarsest squares. **Refinement areas** say where the mesh must be finer: each is a
polygon paired with a **refinement level**, the number of times the base cell is halved
there. Coordinates are in RD New (EPSG:28992).

```python exec="on" source="above" session="tutorial"
from shapely import Point, box

domain = box(155_000, 463_000, 155_640, 463_480)
refinement_areas = [
    (Point(155_250, 463_240).buffer(60), 2),  # 32 m / 2**2 = 8 m cells
    (box(155_450, 463_080, 155_600, 463_160), 1),  # 16 m cells
]
```

## Generate

```python exec="on" source="above" result="text" session="tutorial"
from refined_quads import generate

mesh = generate(
    domain,
    cell_size=32.0,
    refinement_areas=refinement_areas,
    transition_rows=[1, 1],
    crs="EPSG:28992",
)
print(mesh)
```

`transition_rows` gives, per step from coarse to fine, how many rows of finer cells
surround a finer area before the mesh coarsens. Here one row for the step from 32 m to
16 m and one for 16 m to 8 m.

The mesh, coloured by refinement level, with the refinement areas in red:

```python exec="on" html="on" session="tutorial"
print(svg(mesh, areas=[g for g, _ in refinement_areas]))
```

Where levels meet, a lattice square is split into **transition cells**: triangles that
connect the finer cells on one side to the coarser square, without hanging nodes.

## Check

`check` reports everything D-Flow FM and refining rely on:

```python exec="on" source="above" result="text" session="tutorial"
from refined_quads import check

report = check(mesh)
print(report.summary())
```

## Write it for D-HYDRO

```python exec="on" source="above" result="text" session="tutorial"
from refined_quads import read_ugrid, write_ugrid

write_ugrid(mesh, workdir / "mesh_net.nc")
print(read_ugrid(workdir / "mesh_net.nc"))
```

The file has the structure of the D-HYDRO mesh export, with the RD New attributes the
D-Flow FM plugin writes; see [Use the mesh in D-HYDRO](../how-to/d-hydro.md).

## Next

- Read the input from GIS layers: [Read layers from a GeoPackage](../how-to/gis-layers.md).
- Leave out the water: [Water areas](../how-to/water-areas.md).
- Run the same thing from a file: [Run from a TOML file](../how-to/toml-cli.md).
