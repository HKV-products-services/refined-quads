from importlib.metadata import version

import pytest

import refined_quads
from refined_quads.cli import main


def test_version_matches_installed_metadata():
    assert refined_quads.__version__ == version("refined-quads")


def test_cli_version_prints_version_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert refined_quads.__version__ in capsys.readouterr().out


def test_cli_without_arguments_shows_help():
    assert main([]) == 0


def test_cli_unknown_argument_exits_with_input_error():
    with pytest.raises(SystemExit) as exc:
        main(["--no-such-option"])
    assert exc.value.code == 2
