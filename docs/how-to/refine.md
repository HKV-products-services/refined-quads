# Refine an existing mesh

`refine` makes an existing mesh finer in new refinement areas. It needs only the mesh:
not the input it was generated with, and not even a mesh made by this package. Cells
outside the affected area stay exactly as they are, nodes included, so the rest of the
model is untouched.

```python exec="on" session="refine"
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "docs/snippets")
from plot_mesh import svg_row
from shapely import Point, box

from refined_quads import generate, write_ugrid

workdir = Path(tempfile.mkdtemp())
existing = generate(
    box(155_000, 463_000, 155_480, 463_320),
    cell_size=32.0,
    refinement_areas=[(Point(155_120, 463_160).buffer(50), 1)],
    transition_rows=1,
    crs="EPSG:28992",
)
write_ugrid(existing, workdir / "model_net.nc")
```

## In Python

```python exec="on" source="above" result="text" session="refine"
from refined_quads import check, read_ugrid, refine

mesh = read_ugrid(workdir / "model_net.nc")
new_area = Point(155_340, 463_170).buffer(45)

result = refine(mesh, [(new_area, 2)], transition_rows=1, crs="EPSG:28992")

print(result.mesh)
print("valid:", check(result.mesh).is_valid)
```

Before and after; the new or changed cells are highlighted:

```python exec="on" html="on" session="refine"
changed = ~result.replacement_map.unchanged
print(
    svg_row(
        [
            (mesh, {"title": "before"}),
            (result.mesh, {"areas": [new_area], "highlight": changed, "title": "after"}),
        ]
    )
)
```

`transition_rows` works as for `generate`: one value per level step, or one value for all.
A level finer than the mesh has adds levels to its lattice.

## Carry cell data over: the replacement map

`result.replacement_map` relates the new cells to the old ones:

- `source[i]`: the old cell that new cell `i` lies in (for an unchanged cell, itself);
- `unchanged[i]`: whether new cell `i` is an old cell with the same nodes;
- `removed`: the old cells that are not in the new mesh.

Per-cell values of the old mesh, such as an initial water level, carry over in one line:

```python exec="on" source="above" result="text" session="refine"
old_values = np.linspace(10.0, 12.0, len(mesh.faces))  # one value per old cell
rm = result.replacement_map

new_values = old_values[rm.source]

print(
    f"{rm.unchanged.sum()} cells unchanged, {(~rm.unchanged).sum()} new, "
    f"{len(rm.removed)} old cells replaced"
)
```

## On the command line

Put the new areas in a TOML file:

```toml
transition_rows = 1

[[refinement_areas]]
path = "extra.gpkg"
layer = "verfijning_nieuw"
level = 2
```

```bash
refined-quads refine model_net.nc extra.toml -o model_v2_net.nc
```

The options and exit codes are those of `generate`. A mesh that cannot be recognised
exits with 2 and writes `model_v2_net_recognition.gpkg` with the problem locations.

## Domain and water areas

You may pass a domain and water areas to `refine` (or put them in the TOML file). They
only remove **new** cells: existing cells stay. The domain's holes are clipped like water.
The mesh never grows: new cells only replace existing ones, inside the footprint of the
mesh. See [Why refine starts from the mesh](../explanation/refine.md).
