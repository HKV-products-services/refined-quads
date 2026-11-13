# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Any intended change to the cells a given
input produces is at least a minor version (see `docs/adr/0002-output-contract-same-cells.qmd`).

## [Unreleased]

### Changed

- Transition rows may be 0, in `generate`, `refine`, the TOML file and the command
  line: only the cells touching a finer area get its level, and the 2:1 balance makes
  the transition. That gives 4–19% fewer cells on the golden-master scenarios, all
  valid. A negative number still raises `InputError`. The old generator counted
  `buffer_rows` 0 as 1, so its 0 is `transition_rows=1` here (the same cells as before).

## [0.3.0] - 2026-11-12

### Changed

- Intended output change (ADR-0004): `generate` and `refine` no longer leave a hole
  that MeshKernel makes one cell of (a ring of at most six nodes). A water area that
  would remove only one or two cells keeps them; a cell that MeshKernel's clip merges
  from removed cells is replaced by those cells. hydrolib's `mesh2d_from_netcdf` and the
  D-HYDRO GUI made such holes into (often hexagonal) cells that D-Flow FM rejected as
  not orthogonal. The golden-master scenarios `many_small_water_areas` (17 holes) and
  `head_to_toe_boxes` (1 hole) change; the others give the same cells.

### Added

- `MeshReport.closable_holes`: the holes MeshKernel would close, as information, also in
  meshes from other programs; shown in `summary()` and `refined-quads info`.

## [0.2.0] - 2026-11-12

### Changed

- `write_ugrid` no longer writes `Mesh2d_node_z` and `Mesh2d_face_z` (-999 for every node
  and face). After a MeshKernel clip in hydrolib the -999 `mesh2d_node_z` kept its old
  length and saving the model failed with a shape mismatch; the D-HYDRO GUI rejects a bed
  level per cell once it rebuilds the cells (ADR-0003). The cells are unchanged.
- Use the mesh in D-HYDRO: the bed level as samples (`bodem_2d.xyz` with two `[Initial]`
  blocks), with the steps for a D-HyDAMO model build; tested on the Geul model in D-Flow
  FM and the D-HYDRO GUI (D-HYDRO Suite 2026.02). Also the bed level on the nodes
  (`mesh2d_node_z`, `bedLevType` 3 to 6) via D-HyDAMO.

## [0.1.1] - 2026-11-12

### Changed

- The examples in the README and the documentation call the result `quad_mesh` instead
  of `mesh`, with a warning: in a D-HyDAMO script `mesh = generate(...)` replaces the
  module `hydrolib.dhydamo.geometry.mesh`.

## [0.1.0] - 2026-11-12

### Added

- Project skeleton: uv project, ruff, mypy, pytest, CI workflow and documentation site.
- `Mesh` and `Lattice`: an immutable mesh (nodes, faces padded with -1) on its lattice.
- `check(mesh) -> MeshReport`: hanging nodes, 2:1 balance, FM orthogonality per internal
  edge (the D-Flow FM measure), orientation, cells per refinement level and nodes off the
  lattice.
- `InputError` for invalid input.
- Golden masters of the generator at qgis_dhydamo `9a324bc` for fourteen synthetic
  scenarios, with the recording script and a runtime baseline (`tests/golden`); with
  water or holes they are the old generator without water plus the MeshKernel clip of
  the model workflow.
- `generate(domain, *, cell_size, refinement_areas, transition_rows) -> Mesh`, with the
  engine of the generator at qgis_dhydamo `9a324bc`; it gives the same cells for the
  same input (golden masters).
- `OrthogonalityRepairError` when the orthogonality repair does not close.
- Water areas in `generate` (`water_areas`, `water_removal`) with three water removal
  rules: `"centre"` (default), `"inside"` and `"touches"`. The finished mesh is clipped
  cell by cell with MeshKernel, exactly as the model workflow (hydrolib's
  `mesh2d_clip`) does, so the remaining small cells follow the water's edge; MeshKernel
  is a new dependency.
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
- `generate(..., origin=(x, y))` (also `origin` in the TOML and `--origin X Y`): a
  point of the lattice, so a mesh lines up with an existing mesh or raster; by
  default the lower-left corner of the domain rounded down to a multiple of the cell
  size, as before.
- `generate(..., progress=callback)`: `callback(fraction, message)` with a fraction
  rising from 0 to 1 and English messages; returning `True` cancels the run with
  `Cancelled` (the old generator raised a `RuntimeError`). `refined-quads generate -v`
  prints the progress.
- `OrthogonalityRepairError.mesh`: the raw mesh before the repair, next to
  `bad_points`. When the repair does not close, `refined-quads generate` writes
  `<output>_raw.nc` and `<output>_problems.gpkg` (layer `orthogonality_problems`,
  field `orth_e`, -1 for a degenerate edge) in the CRS of the mesh and exits with 1; a
  failure to write them is reported without hiding the repair error.
- `refine(mesh, refinement_areas, ...) -> RefineResult`: refine an existing mesh
  without its original input (ADR-0001). The lattice squares are recognised from the
  cells, the new areas applied, the 2:1 balance restored where needed; squares that do
  not change keep their cells and nodes exactly, the mesh never grows beyond its
  footprint, and a domain or water areas only remove new cells. `RefineResult` has the
  new mesh and a `ReplacementMap` (per new cell the old cell it lies in, which cells are
  unchanged, which old cells went). `generate` is the same route from the base grid.
- `MeshRecognitionError` (an `InputError` with locations) when a mesh cannot be
  refined: nodes off its lattice, cells that fill no lattice square, or a broken 2:1
  balance. Duplicate nodes are accepted, and so are lattice squares that their cells
  fill only partly (clipped along water); new cells in such a square stay only where
  they lie completely within its old cells.
- Refine works on meshes from other programs: any lattice origin and base cell (derived
  from the cells), and any transition cells (a pentagon, another triangulation) as long
  as they fill a lattice square; squares that do not change keep their cells. Refine
  judges the orthogonality only of edges next to new cells, so problems the mesh
  already had do not stop it. Meshes of rectangles or rotated cells raise `InputError`.
- `refined-quads refine mesh_net.nc extra.toml -o mesh_v2_net.nc` and `RefineSpec` (the
  TOML of a refine: new refinement areas, optional domain and water areas): same
  options and exit codes as `generate`; a mesh that cannot be recognised exits with 2
  and writes `<output>_recognition.gpkg` with the problem locations.
- `MeshReport.duplicate_nodes` (information only: cells that touch in a point lose
  nothing) and `MeshReport.split_edges` (an edge on top of another edge with other
  nodes: an internal wall; makes the mesh invalid).
- Benchmarks (`pytest -m benchmark`, pytest-benchmark) on three golden-master
  scenarios, and `tests/golden/benchmarks.py` to compare results with a baseline and
  report changes above 5%. CI runs slow tests and benchmarks on main and compares with
  the previous run on main; results are kept as artifact.
- Documentation: a tutorial (a first mesh), how-to guides (GIS layers, TOML and the
  command line, water areas, refining an existing mesh, D-HYDRO), explanations (refined
  quad meshes, FM orthogonality, why refine starts from the mesh) and the API reference.
  The site is built with Quarto, as in the HKV package template; the code on the pages
  runs during the build and draws the mesh figures, and quartodoc writes the API
  reference. CI publishes it to GitHub Pages from main once the repository is public.
- Licence: GNU GPL v3 or later (`LICENSE`), with `CITATION.cff`.
- pre-commit hooks (file hygiene, yamllint, pydoclint, ruff), also run in CI.
- Publishing to PyPI when a GitHub release is published (trusted publishing).

### Changed compared with the generator at qgis_dhydamo `9a324bc`

- Intended output change: water areas are clipped cell by cell with MeshKernel after
  generation, instead of removing whole lattice squares before the cells are formed
  (the old `clip_water` option). Holes in the domain are clipped the same way; the old
  generator kept every lattice square that overlaps the domain, so cells reached into
  a hole. MeshKernel derives cells from edges again: a single deleted cell whose edges
  all stay comes back, which is accepted.
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
- The orthogonality repair works on integer lattice indices: flips, convexity and the
  redecomposition of lattice squares are exact, and the centre nodes it adds lie exactly
  on the lattice (the old generator computed them as float midpoints). Only the FM
  measure uses coordinates, as D-Flow FM does.
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
- Intended output change for invalid polygons only: a domain, refinement area or water
  area that is not valid is made valid with shapely's `make_valid` (structure method,
  only polygonal parts) instead of `buffer(0)`, and a warning is logged. A
  self-intersecting polygon such as a bowtie keeps all its loops, where `buffer(0)` kept
  one. Valid input gives the same cells.
- The quadtree is built one generation at a time with the geometry tests in bulk
  (shapely's vectorised predicates on prepared geometries, one STRtree query for all
  start cells), and the selection of lattice squares in the domain clips in bulk. The
  cells, and their order, are unchanged; on the benchmarks generate is 3-14% faster.
