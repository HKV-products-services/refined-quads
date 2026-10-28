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

### Changed compared with the generator at qgis_dhydamo `9a324bc`

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
