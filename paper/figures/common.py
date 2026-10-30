"""Shared category identities and fixed manuscript panel geometry."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import numpy as np

COLORS = {0.: "#08519C", -.1: "#4292C6", -.3: "#D94801",
          -.5: "#FD8D3C", -.7: "#238B45"}
MARKERS = ("o", "s", "^", "D", "v")
EXPORT_DPI = 1000
CANVAS = (210.66757202148438, 184.48095703125)
FRAME = (45.97911834716797, 15.816024780273438,
         196.33755493164062, 140.0849609375)
GUIDE_WIDTH = 89.53372192382812
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "mathtext.fontset": "dejavusans",
    "mathtext.default": "it", "axes.linewidth": .65,
    "axes.axisbelow": True, "grid.color": ".87", "grid.linewidth": .45,
    "grid.alpha": 1, "xtick.direction": "out", "ytick.direction": "out",
    "tick.len": 3, "tick.lenratio": 1.7 / 3, "tick.width": .65,
    "tick.widthratio": .5 / .65, "hatch.linewidth": .4,
    "pdf.fonttype": 42, "path.simplify": False,
    "legend.handleheight": .9, "legend.handletextpad": .8,
}


def validate_seeds(values):
    values = np.asarray(values)
    if (values.shape != (5,) or not np.issubdtype(values.dtype, np.integer)
            or len(np.unique(values)) != 5):
        raise ValueError("The paper layout requires five distinct initialization seeds")
    return values


def schedule_colors(values):
    values = np.asarray(values)
    if values.shape != (5,) or not np.isfinite(values).all() or len(np.unique(values)) != 5:
        raise ValueError("The paper layout requires five distinct finite schedule exponents")
    return dict(zip(map(float, values), COLORS.values()))


def save(fig, output, stem, formats=("pdf", "png")):
    import ultraplot as uplt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output / f"{stem}.{extension}", dpi=EXPORT_DPI, facecolor="white")
    uplt.close(fig)


def panel(letter):
    import ultraplot as uplt
    width, height = CANVAS
    left, top, right, bottom = FRAME
    # These panels share the paper's measured canvas and axes dimensions.
    fig, axes = uplt.subplots(figwidth=width / 72, figheight=height / 72,
        tight=False, left=f"{left / 72}in", right=f"{(width - right) / 72}in",
        top=f"{top / 72}in", bottom=f"{(height - bottom) / 72}in",
        share=False, span=False)
    ax = axes[0]
    ax.text(.025, .965, letter + ".", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold",
            zorder=10, rasterized=False)
    return fig, ax


def guide(handles, labels, title, height, stem, output, formats=("pdf", "png"),
          handler_map=None, handlelength=2):
    import ultraplot as uplt
    fig, axes = uplt.subplots(figwidth=GUIDE_WIDTH / 72, figheight=height / 72,
        tight=False, left="0in", right="0in", top="0in", bottom="0in")
    ax = axes[0]
    ax.axis("off")
    legend = ax.legend(handles, labels, title=title, loc="center", ncols=1,
        frame=True, fancybox=False, framealpha=1, edgecolor=".4", fontsize=8,
        title_fontsize=9, borderpad=.65, handlelength=handlelength,
        labelspacing=.5, handler_map=handler_map)
    legend.get_frame().set_linewidth(.65)
    save(fig, output, stem, formats)


def seed_legend(output, formats=("pdf", "png")):
    reference = Line2D([], [], color="black", marker="o", linestyle="none",
                       markerfacecolor="white", markersize=3.1, markeredgewidth=.75)
    others = tuple(Line2D([], [], color="black", marker=marker, linestyle="none",
                         markerfacecolor="white", markersize=3.1, markeredgewidth=.75)
                   for marker in MARKERS[1:])
    guide([reference, others], ["Reference", "Others"], "Seed",
          47.040000915527344, "legend_seeds", output, formats,
          handler_map={tuple: HandlerTuple(ndivide=None, pad=.3)}, handlelength=2.6)
