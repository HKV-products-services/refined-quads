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
