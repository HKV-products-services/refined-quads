"""The real-data tests skip without data and run on whatever masters the directory holds."""

import shutil
from pathlib import Path

import pytest

from golden.masters import MASTERS_DIR, REALDATA_ENV

TEST_FILE = Path(__file__).with_name("test_realdata.py")


def _run(pytester, monkeypatch, directory):
    if directory is None:
        monkeypatch.delenv(REALDATA_ENV, raising=False)
    else:
        monkeypatch.setenv(REALDATA_ENV, str(directory))
    return pytester.runpytest_subprocess(
        "-m", "realdata", "-p", "no:cacheprovider", "--rootdir", str(TEST_FILE.parent),
        str(TEST_FILE),
    )  # fmt: skip


@pytest.mark.timeout(60)
def test_realdata_tests_skip_without_environment_variable(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, None)

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines([f"*set {REALDATA_ENV}*"])


@pytest.mark.timeout(60)
def test_realdata_tests_run_on_the_masters_in_the_directory(pytester, monkeypatch, tmp_path):
    shutil.copy(MASTERS_DIR / "l_shaped_domain.npz", tmp_path)

    result = _run(pytester, monkeypatch, tmp_path)

    result.assert_outcomes(passed=1)
