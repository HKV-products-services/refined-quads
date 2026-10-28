# Output contract: the same cells, compared on lattice indices

Two meshes are "the same" when they contain the same set of cells, with every node
compared as an integer lattice index (in units of base cell / 2^L from the lattice
origin, normalised to the finest level of both meshes). The order of cells and the
numbering of nodes are free. Golden-master meshes from the generator at
qgis_dhydamo `9a324bc` guard this contract; any intended change to the cells a given
input produces is a minor version or higher, with a changelog entry.

## Considered Options

- **Bit-identical arrays, order included.** Rejected: it blocks restructuring
  (vectorising, a different iteration order) without benefiting any caller.
  `rooster_2d_verfijnen` already compares cells independently of order.
- **Float coordinates with a tolerance.** Rejected: a tolerance makes "equal"
  fuzzy, and float coordinates can differ in the last bit between runs when the base
  cell is not an exact binary fraction (e.g. 10.1 m) and the lattice origin differs.

## Consequences

- Everything up to the final conversion works in integer lattice indices,
  including the centre nodes the orthogonality repair adds.
- Dropping the old 1 m minimum cell size is an intended output change: a 24 m base
  cell asked for 5 levels now gives 0.75 m cells instead of being capped at 1.5 m.
