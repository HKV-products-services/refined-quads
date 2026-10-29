# refined-quads

Generate and refine *refined quad meshes* for D-Flow FM: axis-aligned squares on a
fixed lattice, locally refined by repeated halving, with transition cells only where
refinement levels meet.

> **Status:** pre-alpha, private. The licence still has to be agreed with HKV and
> Svasek; the package is not published yet.

## Development

The project uses [uv](https://docs.astral.sh/uv/).

```console
uv sync                      # create .venv with all dependency groups
uv run pytest                # fast test suite
uv run pytest -m slow        # slow scenarios (also: benchmark, realdata)
uv run ruff check            # lint
uv run ruff format --check   # formatting
uv run mypy                  # strict type check of src/
uv run zensical serve        # documentation at http://127.0.0.1:8000
```

The version comes from Git tags (`v0.1.0` → `0.1.0`).

Domain terms are defined in [CONTEXT.md](CONTEXT.md); architectural decisions are in
[docs/adr/](docs/adr/).
