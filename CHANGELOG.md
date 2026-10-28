# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Any intended change to the cells a given
input produces is at least a minor version (see `docs/adr/0002-output-contract-same-cells.md`).

## [Unreleased]

### Added

- Project skeleton: uv project, ruff, mypy, pytest, CI workflow and documentation site.
- `Mesh` and `Lattice`: an immutable mesh (nodes, faces padded with -1) on its lattice.
- `check(mesh) -> MeshReport`: hanging nodes, 2:1 balance, FM orthogonality per internal
  edge (the D-Flow FM measure), orientation, cells per refinement level and nodes off the
  lattice.
- `InputError` for invalid input.
- Golden masters of the generator at qgis_dhydamo `9a324bc` for twelve synthetic
  scenarios, with the recording script and a runtime baseline (`tests/golden`).
- `generate(domain, *, cell_size, refinement_areas, transition_rows) -> Mesh`, with the
  engine of the generator at qgis_dhydamo `9a324bc`; it gives the same cells for the
  same input (golden masters).
- `OrthogonalityRepairError` when the orthogonality repair does not close.
- Water areas in `generate` (`water_areas`, `water_removal`) with three water removal
  rules: `"centre"` (default), `"inside"` and `"touches"`.
- `MeshReport.cell_levels`: the refinement level of the lattice square of every cell.
- `write_ugrid(mesh, path)` and `read_ugrid(path)`: UGRID-1.0 netCDF with the structure
  of the old writer; for EPSG:28992 exactly the CRS attributes of the D-Flow FM plugin.
  `read_ugrid` reads only the 2D mesh of a net file (a 1D network is ignored) and
  derives the lattice from the square cells.
- `Mesh.crs`: the projected CRS of a mesh, or `None` when unknown; a geographic CRS
  raises `InputError`.
- `read_domain`, `read_refinement_areas` (level from a column or one fixed level) and
  `read_water_areas`: read a GeoPackage or shapefile layer with pyogrio, or take a
  GeoDataFrame (geopandas is not a dependency), into `Areas` (polygons with a CRS).
- `generate` takes `Areas` and GeoDataFrames and has a `crs` parameter for input
  without CRS. The mesh gets the CRS of the domain; other input is transformed to it.
- `write_ugrid` derives the CRS attributes of CRSs other than EPSG:28992 with pyproj.
- `MeshSpec`: the input of a run, one-to-one with a TOML file (`MeshSpec.from_toml`,
  `replace`, `generate`); an invalid TOML file gives an `InputError` naming the key and
  the reason.
- Command line: `refined-quads generate run.toml -o mesh_net.nc` (options override the
  TOML values) and `refined-quads info mesh_net.nc` (lattice, cells per level and the
  checks). Exit codes: 0 success, 1 the run failed, 2 invalid input or arguments.

### Changed compared with the generator at qgis_dhydamo `9a324bc`

- Intended output change: holes in the domain are treated as water areas, with the
  water removal rule. The old generator kept every lattice square that overlaps the
  domain, so cells reached into a hole (in the golden-master scenario 84 cells, 15% of
  the hole); with the default rule `"centre"` no square with its centre in a hole is
  kept. The golden master `domain_with_hole` is recorded with the hole as a water area.
- Intended output change: no minimum cell size. A 24 m base cell asked for level 5 now
  gives 0.75 m cells; the old generator capped the levels so that the smallest cell
  was at least 1 m (4 levels, 1.5 m).
- Every face is counterclockwise; the old generator wrote about 5% of its transition
  triangles clockwise. The cells are the same.
- Transition rows are given per step from coarse to fine and must be at least 1; 0
  raises `InputError` (the old generator took them from fine to coarse and counted 0
  as 1). Callers get the same mesh by reversing their list and replacing 0 by 1.
- The highest refinement level follows from the refinement areas; there is no
  `max_refinement`. A level that is not an integer of at least 1 raises `InputError`
  instead of being clamped.
- Explicit refine zones without transition rows and support for shapely < 2 are gone.
- Water areas that are given are always applied; there is no `clip_water` switch that
  silently ignored them. An unknown water removal rule raises `InputError` instead of
  falling back to the centre rule.
- `write_ugrid` writes the CRS of the mesh instead of always EPSG:28992: a mesh without
  CRS gets no CRS variable. The `source` attribute names refined-quads and its version.
- No CRS is ever assumed: input without CRS (a layer, a GeoDataFrame or shapely
  geometries) raises `InputError` unless `crs=` is given, and a geographic domain CRS
  raises `InputError` (the old generator read a layer without CRS as RD New and
  transformed everything else to RD New). A layer whose CRS contradicts `crs=` raises
  `InputError`, as do a missing or invalid level (naming the feature), a feature
  without geometry or that is not a polygon, and a file with several layers when no
  layer is named.
