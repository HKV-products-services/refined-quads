"""Runtime benchmarks on fixed golden-master scenarios (``-m benchmark``).

Run them and compare with a baseline::

    uv run pytest -m benchmark --benchmark-json=benchmark.json
    uv run python tests/golden/benchmarks.py benchmark.json --baseline earlier.json

Without ``--baseline`` the runtime of the generator at qgis_dhydamo 9a324bc is the
baseline (``tests/golden/runtime_baseline.json``), which is only comparable on the
machine that recorded it.
"""

import json

import pytest

from golden.compare import assert_same_cells
from golden.holes import assert_master_with_holes_filled, closable_holes
from golden.masters import MASTERS_DIR, RUNTIME_BASELINE, load_master
from golden.scenarios import SCENARIOS_BY_NAME
from handmade import generate_rd

pytestmark = [pytest.mark.benchmark, pytest.mark.timeout(300)]

#: Small, middle and the scenario with many water areas.
BENCHMARKS = ["l_shaped_domain", "transition_rows_3_2_1_1", "many_small_water_areas"]


@pytest.mark.parametrize("name", BENCHMARKS)
def test_generate(benchmark, name):
    scenario = SCENARIOS_BY_NAME[name]
    legacy = json.loads(RUNTIME_BASELINE.read_text())["scenarios"][name]
    benchmark.extra_info.update(scenario=name, legacy_seconds=legacy["seconds"])

    mesh = benchmark.pedantic(
        generate_rd,
        args=(scenario.domain,),
        kwargs={
            "cell_size": scenario.cell_size,
            "refinement_areas": scenario.refinement_areas,
            "transition_rows": scenario.transition_rows,
            "water_areas": scenario.water_areas,
            "water_removal": scenario.water_removal,
        },
        rounds=5,
        warmup_rounds=1,
    )

    master = load_master(MASTERS_DIR / f"{name}.npz").to_mesh()
    if closable_holes(master):  # issue 20 / ADR-0004
        assert_master_with_holes_filled(mesh, master)
    else:
        assert_same_cells(mesh, master)
