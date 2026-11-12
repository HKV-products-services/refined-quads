"""Checks on real-data masters (Geul, Parkstad) recorded locally with ``record.py --realdata``.

They run with ``-m realdata`` and skip unless ``REFINED_QUADS_REALDATA`` points to the
directory with the recorded ``.npz`` files.
"""

import os
from pathlib import Path

import pytest

from golden.masters import REALDATA_ENV, load_master, master_paths
from refined_quads import check


def _realdata_masters():
    directory = os.environ.get(REALDATA_ENV)
    if not directory:
        reason = f"set {REALDATA_ENV} to the directory with real-data masters"
        return [pytest.param(None, marks=pytest.mark.skip(reason=reason), id="no-data")]
    paths = master_paths(Path(directory))
    if not paths:
        reason = f"no .npz masters in {directory}"
        return [pytest.param(None, marks=pytest.mark.skip(reason=reason), id="no-data")]
    return [pytest.param(p, id=p.stem) for p in paths]


@pytest.mark.realdata
@pytest.mark.timeout(600)
@pytest.mark.parametrize("path", _realdata_masters())
def test_realdata_master_is_conforming_balanced_and_orthogonal(path):
    report = check(load_master(path).to_mesh())
    print(f"\n{path.stem}\n{report.summary()}")

    assert len(report.off_lattice_nodes) == 0
    assert len(report.hanging_nodes) == 0
    assert len(report.balance_violations) == 0
    assert report.max_orthogonality <= 0.30
