"""Draw meshes as inline SVG while the documentation is built.

``svg`` and ``svg_row`` return a raw HTML block; print it from a Quarto cell with
``#| output: asis``.
"""

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.collections import PolyCollection

LEVEL_COLOURS = ["#f4f1e8", "#dbe9f4", "#b7d3ea", "#8cb8dd", "#5e9bcf", "#3a7fbf"]


def _draw(ax, mesh, *, areas=(), water=(), title=None, highlight=None):
    from refined_quads import check

    levels = check(mesh).cell_levels
    polys = [mesh.nodes[f[f >= 0]] for f in mesh.faces]
    colours = [LEVEL_COLOURS[min(int(lv), len(LEVEL_COLOURS) - 1)] if lv >= 0 else "#f2b8b5"
               for lv in levels]  # fmt: skip
    if highlight is not None:
        colours = [("#f6c85f" if h else c) for c, h in zip(colours, highlight, strict=True)]
    ax.add_collection(PolyCollection(polys, facecolors=colours, edgecolors="#333333",
                                     linewidths=0.35))  # fmt: skip
    for geom in areas:
        for part in shapely.get_parts(geom):
            x, y = part.exterior.xy
            ax.plot(x, y, color="#c0392b", linewidth=1.2)
    for geom in water:
        for part in shapely.get_parts(geom):
            x, y = part.exterior.xy
            ax.fill(x, y, color="#2e86c1", alpha=0.35, linewidth=0)
    lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def svg(mesh, **kwargs):
    """Return an SVG picture of one mesh; cells coloured by refinement level."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _draw(ax, mesh, **kwargs)
    return _to_svg(fig)


def svg_row(panels):
    """Return an SVG with several meshes side by side; ``panels`` is a list of (mesh, kwargs)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0))
    for ax, (mesh, kwargs) in zip(np.atleast_1d(axes), panels, strict=True):
        _draw(ax, mesh, **kwargs)
    return _to_svg(fig)


def _to_svg(fig):
    fig.tight_layout()
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg")
    plt.close(fig)
    text = buffer.getvalue()
    return "```{=html}\n" + text[text.index("<svg") :] + "\n```\n"
