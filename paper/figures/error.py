"""Trailing 1,000-update epsilon error on a fixed five-seed loss clock."""
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, StrMethodFormatter
import numpy as np
import ultraplot as uplt

from .common import STYLE, save, validate_seeds, schedule_colors
from .data import load_section


def render(data_path, output, formats=("pdf", "png")):
    data = load_section(data_path, "error")
    if (not np.isfinite(float(data["epsilon"])) or float(data["epsilon"]) <= 0
            or int(data["window"]) < 1):
        raise ValueError("Relative-error summaries require positive epsilon and a nonempty trailing window")
    validate_seeds(data["seed"])
    colors = schedule_colors([float(data[f"g{i}_alpha"]) for i in range(5)])
    with uplt.rc.context(STYLE):
        fig, axes = uplt.subplots(ncols=2, wratios=(1, .40), figwidth=120 / 25.4,
                                  refaspect=1.5, share=False, span=False)
        ax, legend_axis = axes
        for i, alpha in enumerate(colors):
            x, mean, se = (data[f"g{i}_relative_{key}"] for key in ("x", "mean", "se"))
            if (float(data[f"g{i}_alpha"]) != alpha or int(data[f"g{i}_n"]) != 5
                    or x.ndim != 1 or mean.shape != x.shape or se.shape != x.shape
                    or not np.isfinite([x, mean, se]).all()
                    or np.any(np.diff(x) <= 0) or np.any(se < 0)):
                raise ValueError("Expected complete fixed-five-seed error summaries")
            ax.fill_between(x, mean - se, mean + se, color=colors[alpha], alpha=.16, lw=0)
            ax.plot(x, mean, color=colors[alpha], lw=1.2, linestyle="-")
        ax.format(abc=False, title="", xlabel=r"$-\log_{10}\mathcal{L}$",
                  ylabel=r"$\overline{\mathrm{err}}(t)$", xlim=(0, 10.12),
                  xticks=[0, 2, 4, 6, 8, 10], ylim=(0, 1.05),
                  yticks=[0, .2, .4, .6, .8, 1], grid=True,
                  xformatter=StrMethodFormatter("{x:g}"),
                  yformatter=StrMethodFormatter("{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        for spine in ax.spines.values():
            spine.set_visible(True)
        legend_axis.axis("off")
        handles = [Line2D([], [], color=color, lw=1.2) for color in colors.values()]
        legend = legend_axis.legend(handles, [rf"$\alpha={alpha:g}$" for alpha in colors],
            title=r"Schedule $\propto t^\alpha$", loc="c", ncols=1,
            frame=True, fancybox=False, framealpha=1, edgecolor=".4",
            fontsize=8, title_fontsize=9, borderpad=.65, handlelength=2, labelspacing=.5)
        legend.get_frame().set_linewidth(.65)
        save(fig, output, "err", formats)
