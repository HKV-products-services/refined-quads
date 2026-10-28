# Read layers from a GeoPackage

Most model input lives in GIS layers. `read_domain`, `read_refinement_areas` and
`read_water_areas` read a layer of a GeoPackage or shapefile (with pyogrio; geopandas is
not needed) and keep its CRS.

```python exec="on" session="gis"
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyogrio.raw
import shapely
from shapely import Point, box

sys.path.insert(0, "docs/snippets")
from plot_mesh import svg

workdir = Path(tempfile.mkdtemp())
gpkg = workdir / "model.gpkg"


def write(layer, geometries, fields=None):
    fields = fields or {}
    pyogrio.raw.write(
        gpkg,
        shapely.to_wkb(np.asarray(geometries, dtype=object)),
        field_data=list(fields.values()),
        fields=list(fields),
        crs="EPSG:28992",
        geometry_type="Polygon",
        layer=layer,
        driver="GPKG",
    )


write("modelgebied", [box(155_000, 463_000, 155_640, 463_480)])
write(
    "verfijning",
    [Point(155_250, 463_240).buffer(60), box(155_450, 463_080, 155_600, 463_160)],
    {"refinement": np.array([2, 1], dtype=np.int32)},
)
write("water", [Point(155_520, 463_330).buffer(35)])
```

The example GeoPackage `model.gpkg` has three layers:

```python exec="on" result="text" session="gis"
for name, kind in pyogrio.list_layers(gpkg):
    print(f"{name:12s} {kind}")
```

## Read and generate

```python exec="on" source="above" result="text" session="gis"
from refined_quads import (
    generate,
    read_domain,
    read_refinement_areas,
    read_water_areas,
)

domain = read_domain(gpkg, "modelgebied")
areas = read_refinement_areas(gpkg, "verfijning", level="refinement")
water = read_water_areas(gpkg, "water")

mesh = generate(
    domain,
    cell_size=32.0,
    refinement_areas=areas,
    transition_rows=1,
    water_areas=water,
)
print(mesh)
```

```python exec="on" html="on" session="gis"
print(svg(mesh, areas=list(areas.geometries), water=list(water.geometries)))
```

## Rules

- **Layer name.** A file with one layer needs no name; with several layers the name is
  required, otherwise you get an error that lists the layers.
- **Level.** `level` is the name of a column with the refinement level per polygon, or
  one integer for the whole layer (`level=3`). A missing or invalid level (0, a negative
  number, 2.5, empty) raises an error that names the feature.
- **Domain.** All polygons of the domain layer are merged into one (multi)polygon.
- **CRS.** The mesh gets the CRS of the domain. Other layers are transformed to it. A
  layer without CRS raises an error unless you pass `crs=` for it; a geographic CRS
  (degrees) for the domain raises an error, as cell sizes must be in metres.
- **GeoDataFrame.** Instead of a path you may pass a GeoDataFrame (or any object with
  `geometry` and `crs` attributes): `read_refinement_areas(frame, level="refinement")`.
- **Shapely geometries.** Plain shapely geometries have no CRS: pass `crs=` to
  `generate` (see the [tutorial](../tutorial/first-mesh.md)).
