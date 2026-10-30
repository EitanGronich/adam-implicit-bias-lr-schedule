"""Five-by-five grids of trajectories in fixed endpoint reference metrics."""
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator, StrMethodFormatter
import numpy as np
import ultraplot as uplt

from .common import MARKERS, STYLE, save, validate_seeds, schedule_colors
from .data import load_section


def render_grid(data, output, stem, long_horizon=False, formats=("pdf", "png")):
    colors = schedule_colors(data["alpha"])
    if data["exponent"].shape != (5, 5):
        raise ValueError("Expected all five schedules and five endpoint reference metrics")
    validate_seeds(data["seed"])
    exponents = data["exponent"]
    if not np.isfinite(exponents).all() or not np.equal(exponents, np.floor(exponents)).all():
        raise ValueError("Display exponents must be finite integers")
    xmax, xticks = (20.4, [2, 10, 20]) if long_horizon else (10.12, [0, 5, 10])
    style = dict(STYLE, **{"abc.size": 10, "legend.handleheight": .7})
    with uplt.rc.context(style):
        fig, axes = uplt.subplots(nrows=5, ncols=5, journal="nat2", refaspect=.9,
                                  sharex=3, sharey=False, spanx=True, spany=False)
        for index, ax in enumerate(axes):
            row, reference = divmod(index, 5)
            curves = []
            for trajectory in range(5):
                prefix = f"s{row}_t{trajectory}_"
                x, values = data[prefix + "x"], data[prefix + "y"][reference]
                mark = data[prefix + "mark"] if long_horizon else np.ones(len(x), dtype=bool)
                if (x.ndim != 1 or not len(x) or values.shape != x.shape
                        or mark.shape != x.shape or mark.dtype != np.bool_
                        or not np.isfinite([x, values]).all()):
                    raise ValueError("Each panel requires complete signed trajectories")
                curves.append((x, values, mark))
            maximum = max(float(values.max()) for _, values, _ in curves)
            if maximum <= 0:
                raise ValueError("Every reference metric needs a positive displayed margin")
            # Mantissas and their multiplier preserve the weighted margins.
            shift = int(np.floor(np.log10(maximum)))
            factor, exponent = 10. ** -shift, int(exponents[row, reference]) + shift
            # Sparse curves stay vector; rasterizing each panel retains full-page buffers.
            for trajectory, (x, values, mark) in enumerate(curves):
                ax.plot(x, values * factor, color=list(colors.values())[row],
                        lw=1.1 if trajectory == reference else .9,
                        linestyle="-", marker=MARKERS[trajectory], markersize=3.1,
                        markerfacecolor="white", markeredgewidth=.75,
                        markevery=np.flatnonzero(mark).tolist(),
                        zorder=3 if trajectory == reference else 2)
            ax.format(abc=False, title="", xlim=(2 if long_horizon else 0, xmax),
                      xticks=xticks, ylim=(0, maximum * factor * 1.12), grid=True,
                      xlabel=r"$-\log_{10}\mathcal{L}$" if row == 4 else "",
                      ylabel=r"$\gamma_\omega$" if reference == 0 else "",
                      xformatter=StrMethodFormatter("{x:g}"),
                      yformatter=StrMethodFormatter("{x:g}"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            for spine in ax.spines.values():
                spine.set_visible(True)
            ax.text(.025, .965, chr(97 + index) + ".", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10, fontweight="bold", zorder=10)
            if exponent:
                ax.text(1, 1.015, rf"$\times10^{{{exponent}}}$", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8, clip_on=False)
        fig.format(collabels=[f"Ref. seed {i}" for i in range(1, 6)],
                   rowlabels=[rf"$\alpha={value:g}$" for value in data["alpha"]],
                   leftlabels_kw={"size": 9}, toplabels_kw={"size": 9}, abc=False)
        handles = [Line2D([], [], color="black", marker=marker, markerfacecolor="white",
                          markeredgewidth=.75, markersize=3.5, linestyle="none")
                   for marker in MARKERS]
        legend = fig.legend(handles, [str(i) for i in range(1, 6)], loc="b", ncols=5,
                            title="Trajectory seed", fontsize=8, title_fontsize=9,
                            handlelength=1.2, columnspacing=2, borderpad=.65,
                            labelspacing=.5, frame=True, fancybox=False,
                            edgecolor=".4", framealpha=1)
        legend.get_frame().set_linewidth(.65)
        save(fig, output, stem, formats)


def render(data_dir, output_dir, formats=("pdf", "png")):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    render_grid(load_section(data_dir / "long_horizon.npz", "endpoints"), output_dir / "eps0",
                "epsilon0_jumbo", long_horizon=True, formats=formats)
    for name in ("epsilon0", "epsilon1e8"):
        render_grid(load_section(data_dir / "comparison.npz", f"endpoints_{name}"),
                    output_dir / "comparison",
                    f"{name}_scale1_jumbo", formats=formats)
