"""Scale-1 epsilon comparison from fixed five-seed margin summaries."""
from pathlib import Path
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import (LogFormatterMathtext, LogFormatterSciNotation,
    LogLocator, MaxNLocator, NullFormatter, NullLocator, StrMethodFormatter)
import numpy as np
import ultraplot as uplt
from .common import (MARKERS, STYLE, panel, save, guide,
                     seed_legend, validate_seeds, schedule_colors)
from .data import load_section

FILL_ALPHA, EDGE_ALPHA, HATCH_ALPHA = .23, .65, .48



def load_margin(metric, data_dir):
    d = load_section(data_dir / "comparison.npz", metric)
    if (str(d['metric'].item()) != metric or int(d['count']) != 5
            or d['epsilons'].shape != (2,) or d['epsilons'][0] != 0
            or not np.isfinite(d['epsilons']).all() or d['epsilons'][1] <= 0):
        raise ValueError('Expected epsilon zero and one positive epsilon, with five-seed summaries')
    schedule_colors(d['alpha'])
    groups = [[tuple(d[f'e{e}_a{i}_{key}'] for key in ('x', 'mean', 'se'))
               for i in range(5)] for e in range(2)]
    for curves in groups:
        for x, mean, se in curves:
            if (x.ndim != 1 or not len(x) or mean.shape != x.shape or se.shape != x.shape
                    or not np.isfinite([x, mean, se]).all()
                    or (np.diff(x) <= 0).any() or (se < 0).any()):
                raise ValueError('Expected aligned finite five-seed mean/SE observations')
    return groups



def margin_panel(metric, letter, output, data_dir, formats):
    groups = load_margin(metric, data_dir)
    colors = schedule_colors(load_section(data_dir / "comparison.npz", metric)['alpha'])
    fig, ax = panel(letter)
    bands = [(e, curve, color) for e, curves in enumerate(groups)
             for curve, color in zip(curves, colors.values())]
    bands.sort(key=lambda item: float(np.mean(item[1][2])), reverse=True)
    for e, (x, mean, se), color in bands:
        ax.fill_between(x, mean-se, mean+se, facecolor=to_rgba(color, FILL_ALPHA),
                        edgecolor=to_rgba(color, EDGE_ALPHA), linewidth=.35,
                        rasterized=True, zorder=1)
        if e == 1:
            ax.fill_between(x, mean-se, mean+se, facecolor='none',
                            edgecolor=to_rgba(color, HATCH_ALPHA), linewidth=0,
                            hatch='/', rasterized=True, zorder=2)
    for e, curves in enumerate(groups):
        for (x, mean, _), color in zip(curves, colors.values()):
            ax.plot(x, mean, color=color, lw=.95,
                    linestyle='-' if e == 0 else '--', rasterized=True, zorder=3)
    maximum = max(float((mean+se).max()) for curves in groups for _, mean, se in curves)
    low, factor, ylabel = ((1., 1.8, r'$\gamma_\infty$') if metric == 'linf'
                          else (6e-4, 1.12, r'$\gamma_2$'))
    # Draw the complete signed data first; standard logarithmic limits clip it.
    ax.format(abc=False, title='', xlabel=r'$-\log_{10}\mathcal{L}$', ylabel=ylabel,
              xlim=(0, 10.12), xticks=[0, 2, 4, 6, 8, 10],
              yscale='log', ylim=(low, max(low*1.1, factor*maximum)),
              yformatter=LogFormatterMathtext(), grid=True)
    ax.xaxis.set_minor_locator(NullLocator())
    if metric == 'l2':
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 5, 6), numticks=8))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False,
            minor_thresholds=(np.inf, np.inf)))
        # The long bottom tick must not push this label outside the shared canvas.
        ax.yaxis.labelpad -= 6
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for spine in ax.spines.values():
        spine.set_visible(True)
    save(fig, output, f'panel_{letter}', formats)



def weighted_panels(output, data_dir, formats):
    d = load_section(data_dir / "comparison.npz", "weighted")
    colors = schedule_colors(load_section(data_dir / "comparison.npz", "linf")['alpha'])
    seeds = validate_seeds(d['seed'])
    if (not np.array_equal(d['alpha'], [next(iter(colors)), next(reversed(colors))])
            or not np.array_equal(d['reference_seed'], [seeds[0]]*2)):
        raise ValueError('Expected each schedule in its first-seed endpoint metric')
    groups = [[(d[f'a{i}_s{j}_x'], d[f'a{i}_s{j}_mantissa'])
               for j in range(5)] for i in range(2)]
    exponents = [int(d[f'a{i}_exponent']) for i in range(2)]
    for curves, alpha, letter, exponent in zip(groups, d['alpha'], 'cd', exponents):
        fig, ax = panel(letter)
        for j, (x, values) in enumerate(curves):
            if x.ndim != 1 or values.shape != x.shape or not np.isfinite([x, values]).all():
                raise ValueError('Expected finite signed endpoint-weighted observations')
            ax.plot(x, values, color=colors[float(alpha)], linestyle='-',
                    lw=1.1 if j == 0 else .9, marker=MARKERS[j], markersize=3.1,
                    markerfacecolor='white', markeredgewidth=.75,
                    rasterized=True, zorder=3 if j == 0 else 2)
        maximum = max(float(values.max()) for _, values in curves)
        ax.format(abc=False, title='', xlabel=r'$-\log_{10}\mathcal{L}$',
                  ylabel=r'$\gamma_\omega$', xlim=(0, 10.12), xticks=[0, 2, 4, 6, 8, 10],
                  ylim=(0, maximum*1.12), grid=True,
                  yformatter=StrMethodFormatter('{x:g}'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.text(.97, .965, rf'$\alpha={alpha:g}$', transform=ax.transAxes,
                ha='right', va='top', fontsize=9)
        if exponent:
            ax.text(1, 1.015, rf'$\mathrm{{\times10^{{{exponent}}}}}$',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, clip_on=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
        save(fig, output, f'panel_{letter}', formats)



class EpsilonBandHandler(HandlerBase):
    """Draw hatch strokes directly, avoiding viewer-dependent PDF pattern fills."""
    def create_artists(self, legend, original, xdescent, ydescent,
                       width, height, fontsize, transform):
        x0, y0 = -xdescent, -ydescent
        artists = [Rectangle((x0, y0), width, height, facecolor='.77',
                             edgecolor='.4', linewidth=.35, transform=transform)]
        if original.get_linestyle() == '--':
            for start in np.arange(-height, width, 4.):
                low, high = max(0., start), min(width, start+height)
                if high > low:
                    stroke = Line2D([x0+low, x0+high],
                        [y0+low-start, y0+high-start], color='black', linewidth=.65,
                        solid_capstyle='butt', transform=transform)
                    stroke.set_clip_path(artists[0])
                    artists.append(stroke)
        artists.append(Line2D([x0, x0+width], [y0+height/2]*2, color='black',
            linewidth=.95, linestyle=original.get_linestyle(), transform=transform))
        return artists



def epsilon_legend(output, formats, epsilon=1e-8):
    handles = [Line2D([], [], color='black', linestyle=style) for style in ('-', '--')]
    handlers = {handle: EpsilonBandHandler() for handle in handles}
    exponent = int(np.floor(np.log10(epsilon)))
    mantissa = epsilon / 10. ** exponent
    epsilon_text = (rf'$\varepsilon=10^{{{exponent}}}$' if np.isclose(mantissa, 1.)
                    else rf'$\varepsilon={mantissa:g}\times10^{{{exponent}}}$')
    guide(handles, [r'$\varepsilon=0$', epsilon_text], '', 60., 'legend_epsilon', output,
          formats=formats, handler_map=handlers)



def render(data_dir, output, formats=("pdf", "png")):
    data_dir, output = Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with uplt.rc.context(STYLE):
        summary = load_section(data_dir / "comparison.npz", "linf")
        colors = schedule_colors(summary['alpha'])
        margin_panel("linf", "a", output, data_dir, formats)
        margin_panel("l2", "b", output, data_dir, formats)
        weighted_panels(output, data_dir, formats)
        guide([Line2D([], [], color=color, lw=1.2) for color in colors.values()],
              [rf"$\alpha={alpha:g}$" for alpha in colors],
              r"Schedule $\propto t^\alpha$", 101.76000213623047,
              "legend_schedules", output, formats)
        epsilon_legend(output, formats, epsilon=float(summary['epsilons'][1]))
        seed_legend(output, formats)
