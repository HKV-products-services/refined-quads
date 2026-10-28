# Refine starts from the existing mesh; generate is refine of an empty base grid

Refining an existing refined quad mesh rebuilds the quadtree from the mesh's own
cells (squares as leaves, transition triangles as parts of their parent square) on
a lattice derived from the mesh, adds the new refinement areas, re-balances only
where needed and leaves all other cells exactly in place. Generating a new mesh is
the same operation applied to the base grid covering the domain, so there is one
engine and the property `refine(generate(X), Y) == generate(X + Y)` can be tested.

## Considered Options

- **Re-run the generator with the original input** (the approach of
  `rooster_2d_verfijnen` in DHYDRO_Varianten: run twice in a window, without and
  with the new areas, and splice in the difference). Rejected: it needs the
  original generator input (`config.ini`) and only works for meshes this generator
  made with a lattice origin at a multiple of the base cell. It also fails for
  meshes made before 24-10-2026: the generator then rounded the base cell down to a
  power of 2 (25 m became 16 m), so re-running the recorded input now gives a
  different lattice.

## Consequences

- Refine never grows the mesh: new cells only replace existing ones within the
  footprint. A domain or water areas given to refine only remove new cells. The
  property above therefore holds exactly when the same domain and water areas are
  used and Y does not reach cells that `generate(X)` removed for water or domain.
- Meshes from other tools can be refined, as long as their cells are axis-aligned
  squares on one lattice (rectangles and rotated grids are out of scope for now).
- The original generator input is no longer needed to refine a mesh.
