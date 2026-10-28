"""Command-line interface of refined-quads.

Exit codes: 0 success, 1 the run failed, 2 invalid input or arguments.
"""

import argparse
from collections.abc import Sequence

from refined_quads import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the ``refined-quads`` command.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the global options; subcommands are added per feature.
    """
    parser = argparse.ArgumentParser(
        prog="refined-quads",
        description="Generate and refine refined quad meshes for D-Flow FM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name; ``None`` reads them from ``sys.argv``.

    Returns
    -------
    int
        Exit code: 0 success, 1 the run failed, 2 invalid input or arguments.
    """
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
