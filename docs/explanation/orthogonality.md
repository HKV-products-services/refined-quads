# FM orthogonality

D-Flow FM computes the flow between two cells along the line that joins their centres.
That works well when the line crosses their shared edge at a right angle; D-Flow FM
rejects a network ("network is not orthogonal") where it does not.

## The measure

For every internal edge (an edge of exactly two cells):

```text
orth_e = | cos(angle between the edge and the line joining the two cell centres) |
```

0 is perfectly orthogonal and 1 the worst; an edge is acceptable when `orth_e <= 0.30`.
The centre of a triangle is its circumcentre **clipped into the triangle** (the RGFGRID
centre), the centre of any other cell the mean of its nodes. Clipping matters: the raw
circumcentre of an obtuse sliver lies far outside it, on the perpendicular bisector of
every edge, which would hide exactly the slivers D-Flow FM rejects. This is the measure
of the D-HYDRO mesh editor; `check` reports it per edge.

```python exec="on" source="above" result="text" session="orth"
from shapely import Point, box

from refined_quads import check, generate

mesh = generate(
    box(155_000, 463_000, 155_320, 463_240),
    cell_size=32.0,
    refinement_areas=[(Point(155_160, 463_120).buffer(40), 2)],
    transition_rows=[1, 1],
    crs="EPSG:28992",
)
report = check(mesh)
print(f"{report.orthogonality.size} internal edges, worst orth_e = {report.max_orthogonality:.3f}")
```

## Repair

The transition patterns are orthogonal by construction. Later steps (the head-to-toe
correction of neighbouring transition squares, resolving hanging nodes) can still leave
a sliver. The generator therefore checks every internal edge at the end and repairs bad
ones: it flips the shared diagonal of two triangles where that helps, and otherwise emits
the lattice square around the edge again in its canonical pattern. All of this works on
integer lattice indices; only the measure uses coordinates. If invalid edges remain,
`OrthogonalityRepairError` carries the raw mesh and their locations.

When refining, only edges next to new cells are judged: a mesh from another program may
have edges that are not orthogonal by this measure (for example pentagon transition
cells, whose centroid is off-centre), and refine does not fail on what it did not change.
