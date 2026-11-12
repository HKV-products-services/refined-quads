# refined-quads

Generate and refine *refined quad meshes* for D-Flow FM: axis-aligned squares on a
fixed lattice, locally refined by repeated halving, with transition cells only where
refinement levels meet.

```python
from shapely import Point, box

from refined_quads import check, generate, write_ugrid

mesh = generate(
    box(155_000, 463_000, 155_640, 463_480),
    cell_size=32.0,
    refinement_areas=[(Point(155_250, 463_240).buffer(70), 3)],
    transition_rows=[1, 1, 1],
    crs="EPSG:28992",
)
print(check(mesh).summary())
write_ugrid(mesh, "mesh_net.nc")
```

The package also refines an existing mesh (also one made by another program), checks a
mesh for what D-Flow FM needs, reads its input from GIS layers and runs from a TOML file
on the command line. See the
[documentation](https://hkv-products-services.github.io/refined-quads).

> **Status:** pre-alpha. The API is not stable yet.

## Install

```console
pip install refined-quads
```

or, in a uv project, `uv add refined-quads`. All dependencies come as wheels; no separate
GDAL installation is needed. The package is tested on Windows and Linux; on macOS the
MeshKernel wheel needs OpenMP from Homebrew (`brew install libomp`).

## Development

The project uses [uv](https://docs.astral.sh/uv/), which also installs Python.

<details>
<summary>Install uv on Windows</summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

</details>

<details>
<summary>Install uv on Linux or macOS</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

```console
uv sync --locked              # create .venv with all dependency groups
uv run pre-commit install     # run the checks below on every commit
uv run pytest                 # fast test suite
uv run pytest -m benchmark    # also: slow, realdata
uv run pre-commit run --all-files   # file hygiene, yamllint, ruff check and format
uv run mypy                   # strict type check of src/
```

The source in `src/refined_quads/` is grouped by task: `_generate/`, `_refine/`,
`_check/`, `_io/` (GIS layers and UGRID files) and `_common/` (shared building blocks).
The public API is only what the top-level package exports. The tests in `tests/` use the
same folders; `tests/golden/` holds the golden masters that guard the output.

### Documentation

The site is built with [Quarto](https://quarto.org/), which is installed separately
(on Windows: `winget install Posit.Quarto`):

```console
cd docs
uv run quartodoc build           # API reference from the docstrings
uv run quarto preview            # live preview; `quarto render` builds docs/_site/
```

### Releases

The version comes from Git tags (`v0.1.0` → `0.1.0`). Publishing a GitHub release
uploads the package to PyPI.

Domain terms are defined in
[CONTEXT.md](https://github.com/HKV-products-services/refined-quads/blob/main/CONTEXT.md);
architectural decisions are in
[docs/adr/](https://github.com/HKV-products-services/refined-quads/tree/main/docs/adr).

## Licence

refined-quads is developed by HKV lijn in water and Svasek Hydraulics and published under
the [GNU General Public License v3 or later](https://github.com/HKV-products-services/refined-quads/blob/main/LICENSE).
