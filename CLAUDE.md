# CLAUDE.md — refined-quads

Python package that generates and refines refined quad meshes for D-Flow FM. It is the
single source of the generator formerly vendored as `meshgen` in qgis_dhydamo
(`tools/runtime/mesh/meshgen`) and DHYDRO_Varianten (`tools/standalone/measures_cli/rooster/meshgen`).
Plan: `.scratch/issues/refined-quads/PRD.md` and the issues next to it.

## Rules

- Everything in English: code, docstrings (numpy style), error and log messages, docs.
  Conversation with the user may be in Dutch.
- Output contract (ADR-0002): the same input gives the same cells as the generator at
  qgis_dhydamo `9a324bc`, compared on integer lattice indices; cell and node order are
  free. Golden masters guard this; an intended output change is at least a minor
  version with a CHANGELOG entry.
- Replacing own code by shapely/numpy or vectorising: only if the golden masters still
  give the same cells and the benchmark is not more than 5% slower.
- Invalid input raises an error; never silently adjust (levels, transition rows, CRS).
  Exception: invalid polygons (domain, refinement areas, water areas) are made valid
  with `make_valid` and a logged warning, not an error.
- Public API only via the top-level package; internal modules start with an underscore.
- Use the terms from `CONTEXT.md`.

## Commands

| Task | Command |
| --- | --- |
| Install | `uv sync` |
| Fast tests | `uv run pytest` (slow/benchmark/realdata via `-m <marker>`) |
| Benchmarks | `uv run pytest -m benchmark --benchmark-json=after.json`, then `uv run python tests/golden/benchmarks.py after.json --baseline before.json` (5% rule: measure before and after back to back on one machine) |
| Lint / format | `uv run ruff check`, `uv run ruff format` |
| Types | `uv run mypy` |
| Docs | `uv run zensical build` / `uv run zensical serve` (Zensical reads `mkdocs.yml`; `strict: true` there) |

uv manages Python itself (`.python-version` = 3.12); the chocolatey Python shims on the
development PC are broken, so do not rely on a system Python.

## Agent skills

### Issue tracker

Local markdown in `.scratch/issues/<feature>/` (outside Git). See `docs/agents/issue-tracker.md`.

### Triage labels

Default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) as a `Status:` line in the issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the root, ADRs in `docs/adr/`. See `docs/agents/domain.md`.
