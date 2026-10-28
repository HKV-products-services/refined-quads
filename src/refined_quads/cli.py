"""Command-line interface of refined-quads.

Exit codes: 0 success, 1 the run failed, 2 invalid input or arguments.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from refined_quads import __version__
from refined_quads._check import check
from refined_quads._engine import WATER_REMOVAL_RULES
from refined_quads._gis import write_points, write_problem_points
from refined_quads._ugrid import read_ugrid, write_ugrid
from refined_quads.errors import InputError, MeshRecognitionError, OrthogonalityRepairError
from refined_quads.mesh import Mesh
from refined_quads.spec import MeshSpec, RefineSpec

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

    ref = commands.add_parser(
        "refine",
        help="refine an existing UGRID mesh with the areas of a TOML file",
        description="Refine the 2D mesh of a UGRID file with the new refinement areas of a "
        "TOML file (see RefineSpec) and write the result as UGRID. Options override the "
        "values in the TOML file.",
    )
    ref.add_argument("mesh", type=Path, help="UGRID file with the existing mesh")
    ref.add_argument("spec", type=Path, help="TOML file with the new refinement areas")
    ref.add_argument("-o", "--output", type=Path, help="UGRID file to write (overrides 'output')")
    ref.add_argument(
        "--transition-rows",
        type=int,
        nargs="+",
        metavar="N",
        help="transition rows per level step from coarse to fine, or one value for all",
    )
    ref.add_argument("--water-removal", choices=WATER_REMOVAL_RULES, help="water removal rule")
    ref.add_argument("--crs", help="CRS of layers without one (overrides 'crs')")
    ref.add_argument("-v", "--verbose", action="store_true", help="show progress messages")

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
        if args.command == "refine":
            return _refine(args)
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
        "transition_rows": _rows_option(args.transition_rows),
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
    except OrthogonalityRepairError as exc:
        _write_dump(exc, spec.output)
        raise
    write_ugrid(mesh, spec.output)
    report = check(mesh)
    print(f"wrote {spec.output}")
    print(report.summary())
    return EXIT_OK


def _rows_option(values: list[int] | None) -> int | tuple[int, ...] | None:
    if values is None:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _refine(args: argparse.Namespace) -> int:
    spec = RefineSpec.from_toml(args.spec)
    overrides = {
        "transition_rows": _rows_option(args.transition_rows),
        "water_removal": args.water_removal,
        "crs": args.crs,
        "output": args.output,
    }
    spec = spec.replace(**{k: v for k, v in overrides.items() if v is not None})
    if spec.output is None:
        raise InputError("no output file: give -o/--output or 'output' in the TOML file")
    mesh = read_ugrid(args.mesh)
    try:
        result = spec.refine(mesh, progress=_print_progress if args.verbose else None)
    except MeshRecognitionError as exc:
        _write_locations(exc, spec.output, mesh)
        raise InputError(f"{args.mesh}: {exc}") from exc
    except InputError as exc:
        raise InputError(f"{args.spec}: {exc}") from exc
    except OrthogonalityRepairError as exc:
        _write_dump(exc, spec.output)
        raise
    write_ugrid(result.mesh, spec.output)
    rm = result.replacement_map
    print(f"wrote {spec.output}")
    print(f"cells: {int(rm.unchanged.sum())} unchanged, {int((~rm.unchanged).sum())} new, "
          f"{len(rm.removed)} existing cells replaced or removed")  # fmt: skip
    print(check(result.mesh).summary())
    return EXIT_OK


def _write_locations(exc: MeshRecognitionError, output: Path, mesh: Mesh) -> None:
    """Write where a mesh could not be recognised next to the output (never masks the error)."""
    stem = output.with_suffix("")
    path = stem.with_name(f"{stem.name}_recognition.gpkg")
    try:
        problem = np.array([str(exc).split(";")[0][:254]] * len(exc.locations), dtype=object)
        write_points(path, "recognition_problems", exc.locations, {"problem": problem}, mesh.crs)
        print(f"wrote {len(exc.locations)} problem locations to {path}", file=sys.stderr)
    except Exception as dump_exc:  # any failure: it must not mask the recognition error
        print(f"warning: could not write {path}: {dump_exc}", file=sys.stderr)


def _write_dump(exc: OrthogonalityRepairError, output: Path) -> None:
    """Write the raw mesh and the problem locations next to the output.

    A failure here is reported but does not replace the original error.
    """
    stem = output.with_suffix("")
    raw_path = stem.with_name(f"{stem.name}_raw.nc")
    problems_path = stem.with_name(f"{stem.name}_problems.gpkg")
    if exc.mesh is not None:
        try:
            write_ugrid(exc.mesh, raw_path)
            print(f"wrote the raw (unrepaired) mesh to {raw_path}", file=sys.stderr)
        except Exception as dump_exc:  # any failure: it must not mask the repair error
            print(f"warning: could not write {raw_path}: {dump_exc}", file=sys.stderr)
    try:
        crs = None if exc.mesh is None else exc.mesh.crs
        write_problem_points(problems_path, exc.bad_points, crs)
        print(f"wrote {len(exc.bad_points)} problem locations to {problems_path}", file=sys.stderr)
    except Exception as dump_exc:  # idem
        print(f"warning: could not write {problems_path}: {dump_exc}", file=sys.stderr)


def _print_progress(fraction: float, message: str) -> None:
    print(f"{fraction:4.0%} {message}", file=sys.stderr)


def _info(args: argparse.Namespace) -> int:
    mesh = read_ugrid(args.mesh)
    crs = "none" if mesh.crs is None else mesh.crs.to_string()
    print(f"{args.mesh}")
    print(f"crs: {crs}")
    print(check(mesh).summary())
    return EXIT_OK
