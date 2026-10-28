# refined-quads

Generates 2D computational meshes of axis-aligned square cells that are locally
refined by halving, for use as the 2D grid of a D-Flow FM model.

## Language

**Refined quad mesh**:
A 2D mesh of axis-aligned squares on a fixed lattice, locally refined by
repeated halving, with triangles or irregular quads only where cell sizes change.
_Avoid_: quadtree grid, meshgen grid, Quad Grid

**Base cell**:
The coarsest square of a refined quad mesh; its size is the cell size the user
chooses, any positive length (not necessarily a power of 2).
_Avoid_: basiscel, coarse cell

**Refinement level**:
The number of halvings of the base cell; level 0 is the base cell, level N has
side length cell size / 2^N.
_Avoid_: halving level, refinement (as a number)

**Lattice**:
The fixed grid of points, given by an origin and the base cell size, on which
every square of a refined quad mesh has its corners.
_Avoid_: raster, grid origin

**Lattice square**:
A square of the quadtree: a base cell or one of the squares its halving gives,
aligned to the lattice at its refinement level. It becomes one square cell, or is
filled exactly by transition cells; after clipping along water it can be filled only
partly.
_Avoid_: parent square, quadtree cell, leaf

**Transition rows**:
For the step from refinement level k+1 down to level k, the number of rows of
level-(k+1) cells kept around the finer area before the mesh coarsens; at least 1.
Given per step from coarse to fine.
_Avoid_: buffer rows, bufferrijen, buffer

**Transition cell**:
A triangle or irregular quad that, together with the other transition cells in
the same lattice square, fills that square exactly where refinement levels meet.
_Avoid_: sliver, overgangsdriehoek, transition triangle

**Footprint**:
The area covered by the cells of an existing mesh; holes are not part of it.
Refining never adds cells outside it.
_Avoid_: extent, mesh outline

**Replacement map**:
The record of a refine: which existing cells were removed and, for every new
cell, the existing cell it lies in. Lets callers carry cell data over.
_Avoid_: diff, mapping, splice

## Input

**Domain**:
The (multi)polygon that encloses the mesh and forms its edges (the enclosure in
D-HYDRO terms); usually a polygon layer of a GeoPackage, whose features are merged. A
lattice square that overlaps it is kept whole, so the outer edge of the mesh follows
the lattice along the domain rather than the polygon exactly. Holes in the domain are
clipped like water areas, cell by cell.
_Avoid_: boundary, extent, model area

**Refinement area**:
A polygon paired with the refinement level its cells must reach; transition
rows are added around it automatically.
_Avoid_: refinement polygon, refine zone, refinement (as a shape)

**Water area**:
An area where no 2D cells are kept, because the 1D network carries the flow there.
_Avoid_: water polygon, exclusion area, clip area

**Water removal rule**:
The test that decides whether a cell touching a water area (or a hole in the domain)
is removed from the finished mesh: by its circumcentre, only when fully inside, or as
soon as it touches; applied cell by cell with MeshKernel, as in the model workflow, so
the remaining small cells follow the water's edge.
_Avoid_: delete option, clip method
