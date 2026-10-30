"""Main MNIST figure and individual alpha=-0.3 trajectories."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import (LogFormatterMathtext, LogLocator, NullFormatter,
                              ScalarFormatter, StrMethodFormatter)
import numpy as np
import ultraplot as uplt

from .common import (COLORS, MARKERS, STYLE, panel, save, guide, validate_seeds,
                     schedule_colors)
from .data import load_section


def loss_rate(data, output, formats, beta2=.999):
    colors = schedule_colors(data["alphas"])
    window = int(data["window"])
    window_text = f"{window:,}".replace(",", "{,}")
    if window < 1 or not np.array_equal(data["x"], np.arange(1, window + 1)):
        raise ValueError("Expected consecutive secant updates within the configured window")
    if not 0 < beta2 < 1:
        raise ValueError("The loss-rate reference requires beta2 in (0, 1)")
    validate_seeds(data["seeds"])
    fig, ax = panel("a")
    lower, upper = np.inf, 0.
    for i, color in enumerate(colors.values()):
        mean, se = data[f"s{i}_mean"], data[f"s{i}_se"]
        if (mean.shape != data["x"].shape or se.shape != mean.shape
                or not np.isfinite([mean, se]).all()
                or np.any(mean - se <= 0) or np.any(se < 0)):
            raise ValueError("Log secant axes require positive uncertainty bounds")
        ax.fill_between(data["x"], mean - se, mean + se,
                        color=color, alpha=.16, linewidth=0)
        ax.plot(data["x"], mean, color=color, lw=1.2)
        lower = min(lower, float((mean - se).min()))
        upper = max(upper, float((mean + se).max()))
    ax.axhline(-np.log(beta2) / 2, color=".25", lw=.9, linestyle=(0, (4, 2)))
    ax.format(abc=False, title="", xlim=(0, window),
              xticks=np.linspace(0, window, 5),
              xformatter=StrMethodFormatter("{x:,.0f}"),
              yscale="log", ylim=(lower / 1.15, upper * 2), grid=True,
              xlabel=r"$t-T$ (updates)" + "\n" +
              rf"$T=t_{{\mathrm{{end}}}}-{window_text}$",
              ylabel=r"$\frac{-\ln\mathcal{L}(t)+\ln\mathcal{L}(T)}{t-T}$")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(NullFormatter())
    save(fig, output, "panel_loss_rate", formats)


def linf_margin(data, output, formats):
    colors = schedule_colors(data["alpha"])
    validate_seeds(data["seed"])
    fig, ax = panel("b")
    for i, color in enumerate(colors.values()):
        x, mean, se = (data[f"s{i}_{key}"] for key in ("x", "mean", "se"))
        if (x.ndim != 1 or not len(x) or x.shape != mean.shape or x.shape != se.shape
                or np.any(np.diff(x) <= 0)
                or not np.isfinite([x, mean, se]).all() or np.any(se < 0)):
            raise ValueError("Invalid first-passage margin summary")
        positive = mean > 0
        ax.fill_between(x, np.maximum(mean - se, .1), np.maximum(mean + se, .1),
                        where=positive, color=color, alpha=.16, linewidth=0)
        ax.plot(x, np.where(positive, mean, np.nan), color=color, lw=1.2)
    ax.format(abc=False, title="", xlim=(2, 20.4), xticks=[2, 4, 8, 12, 16, 20],
              xlabel=r"$-\log_{10}\mathcal{L}$", ylabel=r"$\gamma_\infty$",
              yscale="log", ylim=(.1, 1e6), grid=True)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(NullFormatter())
    save(fig, output, "panel_linf_margin", formats)


def weighted_margin(data, schedule, letter, name, output, formats):
    colors = schedule_colors(data["alpha"])
    fig, ax = panel(letter)
    maximum = 0.
    for trajectory, marker in enumerate(MARKERS):
        prefix = f"s{schedule}_t{trajectory}_"
        x, values, mark = (data[prefix + "x"], data[prefix + "y"][0],
                           data[prefix + "mark"])
        if (x.ndim != 1 or values.shape != x.shape or mark.shape != x.shape
                or mark.dtype != np.bool_ or not np.isfinite([x, values]).all()):
            raise ValueError("Invalid endpoint-weighted observations")
        # The main figure retains gaps at negative margins. The signed values
        # remain in the shared bundle used by the full appendix grids.
        displayed = np.where(values < 0, np.nan, values)
        maximum = max(maximum, float(np.nanmax(displayed)))
        ax.plot(x, displayed, color=list(colors.values())[schedule],
                lw=1.1 if trajectory == 0 else .9, marker=marker,
                markevery=np.flatnonzero(mark & np.isfinite(displayed)).tolist(),
                markersize=3.1, markerfacecolor="white", markeredgewidth=.75)
    ax.format(abc=False, title="", xlim=(2, 20.4), xticks=[2, 4, 8, 12, 16, 20],
              xlabel=r"$-\log_{10}\mathcal{L}$", ylabel=r"$\gamma_\omega$",
              ylim=(0, maximum * 1.12), grid=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    exponent = int(data["exponent"][schedule, 0])
    if exponent:
        ax.text(1., 1.015, rf"$\times10^{{{exponent}}}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, clip_on=False)
    save(fig, output, name, formats)


def appendix(data, output, formats, color=COLORS[-.3]):
    validate_seeds(data["seed"])
    alpha = float(data.get("alpha", -.3))
    if not np.isfinite(alpha):
        raise ValueError("The individual-seed figure requires a finite schedule exponent")
    stem = "alpha_03_all_seeds" if alpha == -.3 else "alpha_selected_all_seeds"
    with plt.rc_context({key: value for key, value in STYLE.items()
                         if key in plt.rcParams}):
        fig, ax = plt.subplots(figsize=(3.55, 2.75))
        fig.subplots_adjust(left=.185, right=.97, bottom=.19, top=.79)
        handles = []
        for i, marker in enumerate(MARKERS):
            x, y = data[f"x{i}"], data[f"y{i}"]
            if (x.ndim != 1 or y.shape != x.shape or not len(x)
                    or not np.isfinite([x, y]).all()
                    or np.any(np.diff(x) <= 0)):
                raise ValueError("Expected finite ordered individual margin curves")
            line, = ax.plot(x, y, color=color, lw=1.)
            ax.plot(data[f"marker_x{i}"], data[f"marker_y{i}"], linestyle="none",
                    color=color, marker=marker, markersize=3.1,
                    markerfacecolor="white", markeredgewidth=.75, zorder=3)
            line.set_marker(marker)
            line.set_markevery([])
            line.set_markerfacecolor("white")
            line.set_markeredgewidth(.75)
            line.set_markersize(3.1)
            handles.append(line)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color=".87", linewidth=.45)
        ax.tick_params(which="major", direction="out", length=3, width=.65)
        ax.tick_params(which="minor", direction="out", length=1.7, width=.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
        positive = np.concatenate([data[f"y{i}"][data[f"y{i}"] > 0] for i in range(5)])
        if not len(positive):
            raise ValueError("Log margin axes require at least one positive value")
        lower = float(positive.min())
        upper = max(float(data[f"y{i}"].max()) for i in range(5))
        ax.set(xlim=(4, 20.15), xticks=[4, 8, 12, 16, 20], yscale="log",
               ylim=(lower * .75, 10 ** np.ceil(np.log10(upper))),
               xlabel=r"$-\log_{10}\mathcal{L}$", ylabel=r"$\gamma_\infty$")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(NullFormatter())
        title = "Seed" if alpha == -.3 else rf"Seed ($\alpha={alpha:g}$)"
        ax.legend(handles=handles, labels=[str(i) for i in range(1, 6)], title=title,
                  loc="lower center", bbox_to_anchor=(.5, 1.025), ncol=5,
                  frameon=False, fontsize=7.5, title_fontsize=8,
                  handlelength=1.7, columnspacing=1., handletextpad=.4,
                  borderaxespad=0.)
        save(fig, output, stem, formats)


def render(data_dir, output, formats=("pdf", "png")):
    data_dir, output = Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with uplt.rc.context(STYLE):
        archive = data_dir / "long_horizon.npz"
        secants = load_section(archive, "secants")
        settings = load_section(archive, "settings", required=False)
        colors = schedule_colors(secants["alphas"])
        loss_rate(secants, output, formats, beta2=float(settings.get("beta2", .999)))
        linf_margin(load_section(archive, "linf"), output, formats)
        endpoints = load_section(archive, "endpoints")
        weighted_margin(endpoints, 0, "c", "panel_frozen_margin_a0", output, formats)
        weighted_margin(endpoints, 4, "d", "panel_frozen_margin_a07", output, formats)
        handles = [Line2D([], [], color=color, lw=1.2) for color in colors.values()]
        handles.append(Line2D([], [], color=".25", lw=.9, linestyle=(0, (4, 2))))
        guide(handles, [rf"$\alpha={alpha:g}$" for alpha in colors] + [r"$-\ln(\beta_2)/2$"],
              r"Schedule $\propto t^\alpha$", 101.76000213623047,
              "legend_schedules", output, formats)
        markers = [Line2D([], [], color="black", marker=marker, linestyle="none",
                          markerfacecolor="white", markersize=3.5, markeredgewidth=.75)
                   for marker in MARKERS]
        guide([markers[0], tuple(markers[1:])], ["Reference", "Others"], "Seed",
              47.040000915527344, "legend_seeds", output, formats,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=.15)}, handlelength=3.2)
    individual = load_section(archive, "alpha_m03")
    selected_alpha = float(individual.get("alpha", -.3))
    if selected_alpha not in colors:
        raise ValueError("The selected individual-seed exponent is absent from the experiment")
    appendix(individual, output, formats, color=colors[selected_alpha])
