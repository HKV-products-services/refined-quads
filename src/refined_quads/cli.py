"""Command-line interface of refined-quads.

Exit codes: 0 success, 1 the run failed, 2 invalid input or arguments.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from refined_quads import __version__
from refined_quads._check import check
from refined_quads._engine import WATER_REMOVAL_RULES
from refined_quads._ugrid import read_ugrid, write_ugrid
from refined_quads.errors import InputError, OrthogonalityRepairError
from refined_quads.spec import MeshSpec

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the ``refined-quads`` command.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``generate`` and ``info`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="refined-quads",
        description="Generate and refine refined quad meshes for D-Flow FM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    gen = commands.add_parser(
        "generate",
        help="generate a mesh from a TOML file and write it as UGRID",
        description="Generate a mesh from a TOML file (see MeshSpec) and write it as UGRID. "
        "Options override the values in the TOML file.",
    )
    gen.add_argument("spec", type=Path, help="TOML file with the input of the run")
    gen.add_argument("-o", "--output", type=Path, help="UGRID file to write (overrides 'output')")
    gen.add_argument("--cell-size", type=float, help="base cell size (overrides 'cell_size')")
    gen.add_argument(
        "--transition-rows",
        type=int,
        nargs="+",
        metavar="N",
        help="transition rows per level step from coarse to fine, or one value for all",
    )
    gen.add_argument("--water-removal", choices=WATER_REMOVAL_RULES, help="water removal rule")
    gen.add_argument("--crs", help="CRS of layers without one (overrides 'crs')")
    gen.add_argument(
        "--origin",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="a lattice point (overrides 'origin')",
    )
    gen.add_argument("-v", "--verbose", action="store_true", help="show progress messages")

    info = commands.add_parser(
        "info",
        help="show the lattice, levels, cells and checks of a UGRID mesh",
        description="Read the 2D mesh of a UGRID file and report its lattice, cells per "
        "level and the results of check().",
    )
    info.add_argument("mesh", type=Path, help="UGRID file, such as a D-HYDRO net file")
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
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    try:
        if args.command == "generate":
            return _generate(args)
        return _info(args)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except OrthogonalityRepairError as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except OSError as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return EXIT_FAILED


def _generate(args: argparse.Namespace) -> int:
    spec = MeshSpec.from_toml(args.spec)
    overrides = {
        "cell_size": args.cell_size,
        "transition_rows": (
            None
            if args.transition_rows is None
            else (
                args.transition_rows[0]
                if len(args.transition_rows) == 1
                else tuple(args.transition_rows)
            )
        ),
        "water_removal": args.water_removal,
        "crs": args.crs,
        "origin": None if args.origin is None else tuple(args.origin),
        "output": args.output,
    }
    spec = spec.replace(**{k: v for k, v in overrides.items() if v is not None})
    if spec.output is None:
        raise InputError("no output file: give -o/--output or 'output' in the TOML file")
    try:
        mesh = spec.generate(progress=_print_progress if args.verbose else None)
    except InputError as exc:
        raise InputError(f"{args.spec}: {exc}") from exc
    write_ugrid(mesh, spec.output)
    report = check(mesh)
    print(f"wrote {spec.output}")
    print(report.summary())
    return EXIT_OK


def _print_progress(fraction: float, message: str) -> None:
    print(f"{fraction:4.0%} {message}", file=sys.stderr)


def _info(args: argparse.Namespace) -> int:
    mesh = read_ugrid(args.mesh)
    crs = "none" if mesh.crs is None else mesh.crs.to_string()
    print(f"{args.mesh}")
    print(f"crs: {crs}")
    print(check(mesh).summary())
    return EXIT_OK
