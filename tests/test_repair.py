"""The orthogonality repair gives the same cells as the original on handmade sliver meshes.

No golden-master scenario needs the repair, so it is tested directly (an internal
function: a wrong repair could not be localised otherwise). The references are what
the repair of the generator at qgis_dhydamo 9a324bc made of the same meshes
(``record.py --repair-reference``).
"""

import numpy as np
import pytest

from golden.record import REPAIR_DIR
from golden.repair_cases import CASES
from refined_quads import Lattice
from refined_quads._faces import Nodes
from refined_quads._orthogonality import repair_orthogonality


def canonical(cells, points):
    result = set()
    for cell in cells:
        p = [tuple(int(v) for v in points[n]) for n in cell if n >= 0]
        area2 = sum(
            p[k][0] * p[(k + 1) % len(p)][1] - p[(k + 1) % len(p)][0] * p[k][1]
            for k in range(len(p))
        )
        if area2 < 0:
            p = p[::-1]
        start = min(range(len(p)), key=p.__getitem__)
        result.add(tuple(p[start:] + p[:start]))
    return result


def repaired(case, **kwargs):
    nodes = Nodes()
    cells = [[nodes.add(*point) for point in cell] for cell in case.cells]
    messages = []
    repair_orthogonality(cells, nodes, case.origin, case.unit, log=messages.append, **kwargs)
    return cells, nodes, messages


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_repair_gives_the_cells_of_the_original(case):
    with np.load(REPAIR_DIR / f"{case.name}.npz") as reference:
        expected = canonical(reference["faces"], reference["points"])

    cells, nodes, messages = repaired(case)

    assert canonical(cells, nodes.points) == expected
    assert any("edge flip" in m and "OK" in m for m in messages)


def test_redecomposition_adds_a_centre_node_exactly_on_the_lattice():
    case = next(c for c in CASES if c.name == "fan_from_south_10_1")
    before = len({p for cell in case.cells for p in cell})

    _, nodes, messages = repaired(case)

    assert any("1 redecomposition(s)" in m for m in messages)
    centre = nodes.points[before:]
    assert centre == [(6, 6)]
    lattice = Lattice(case.origin, case.unit * 4, 2)
    coordinates = lattice.to_coordinates(np.asarray(centre))
    _, on_lattice = lattice.to_indices(coordinates)
    assert on_lattice.all()
    assert coordinates[0, 0] == case.origin[0] + 6 * case.unit


def test_repair_that_cannot_close_raises_with_the_problem_points():
    from refined_quads import OrthogonalityRepairError

    case = CASES[0]
    with pytest.raises(OrthogonalityRepairError) as caught:
        repaired(case, max_passes=0)
    assert len(caught.value.bad_points) > 0
