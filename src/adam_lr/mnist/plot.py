"""Plots from saved trajectories, anchors, and fixed-coordinate panels.

python -m adam_lr.mnist.plot runs/mnist_smoke --output-dir output/mnist_smoke
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np

DEFAULT_FIELDS = ("loss", "linf_margin", "parameter_linf", "euclidean_update_gradient_cosine", "a1em4_second_moment_cosine", "layer_linf_balance")
ANCHORS = ("a1em4", "a1em8", "a1em12")
COORDINATE_FIELDS = ("raw_second", "g2_over_v", "rho_over_c2")


def load_runs(root):
    """Load bounded scalar tables; reject ambiguous timing or inconsistent rows."""
    paths = sorted(Path(root).rglob("results.json"))
    if not paths:
        raise ValueError(f"no results.json files found in {root}")
    runs = []
    for path in paths:
        metadata = json.loads(path.read_text())
        series_path = path.parent / "trajectory_series.npz"
        with np.load(series_path, allow_pickle=False) as archive:
            series = {key: archive[key] for key in archive.files}
        if not len(series["step"]):
            continue
        if np.any(np.diff(series["step"]) <= 0):
            raise ValueError(f"non-increasing trajectory steps: {series_path}")
        if any(len(value) != len(series["step"]) for value in series.values()):
            raise ValueError(f"inconsistent scalar table lengths: {series_path}")
        runs.append((path, metadata, series))
    return runs


def x_values(series, kind):
    if kind == "step":
        return np.asarray(series["step"], dtype=np.float64)
    if kind == "loss-clock":
        loss = np.asarray(series["loss"], dtype=np.float64)
        result = np.full(loss.shape, np.nan)
        valid = np.isfinite(loss) & (loss > 0)
        result[valid] = -np.log(loss[valid])
        return result
    raise ValueError(f"unknown x-axis: {kind}")


def scalar_fields(runs, fields):
    """Require every requested field to be a scalar column in every run."""
    for path, _, series in runs:
        for key in fields:
            if key not in series or series[key].ndim != 1:
                raise ValueError(f"{key!r} is not a scalar field in {path.parent}")
    return fields


def read_incoming_for_display(path, max_points=2000):
    """Stream a trace into a bounded decimated display, preserving endpoints.

    Full-precision/full-resolution analysis belongs to incoming_analysis.py;
    this display sampler does not aggregate or overwrite the source trace.
    """
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    points, stride, final_line, final_index = [], 1, None, -1
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            final_line, final_index = line, index
            if index % stride:
                continue
            row = json.loads(line)
            points.append((row["step"], row["log_D_total"]))
            if len(points) >= max_points:
                points = points[::2]
                stride *= 2
    if final_line is not None:
        row = json.loads(final_line)
        if not points or row["step"] != points[-1][0]:
            points.append((row["step"], row["log_D_total"]))
    return (np.asarray([p[0] for p in points], dtype=np.int64),
            np.asarray([np.nan if p[1] is None else p[1] for p in points], dtype=np.float64),
            final_index + 1)


def _label(path, root):
    relative = str(path.parent.relative_to(root))
    return path.parent.name if relative == "." else relative


def _x_label(kind):
    return "Completed Adam updates t" if kind == "step" else "Loss clock: -ln(mean logistic loss)"


def _scale(key):
    return "log" if key in ("loss", "parameter_l1", "parameter_l2", "parameter_linf", "raw_second", "g2_over_v", "v_over_g2", "inverse_root_second") else "linear"


def _line(ax, x, y, *, key, label, **style):
    scale = _scale(key)
    valid = np.isfinite(x) & np.isfinite(y)
    if scale == "log":
        valid &= y > 0
    if valid.any():
        if np.count_nonzero(valid) == 1:
            style.setdefault("marker", "o")
            style.setdefault("markersize", 3)
        ax.plot(np.where(valid, x, np.nan), np.where(valid, y, np.nan), label=label, **style)
    ax.set_yscale(scale)


def _finish_axis(ax, key, kind):
    ax.set(xlabel=_x_label(kind), ylabel=key.replace("_", " "))
    ax.grid(alpha=.2)
    if not ax.lines:
        ax.text(.5, .5, "No defined observations\n(anchor may not be reached)", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=".45")
    if "cosine" in key:
        ax.set_ylim(-1.05, 1.05)


def _save(fig, directory, stem):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=180)


def plot_scalar_grid(runs, root, fields, kind, destination, stem):
    import matplotlib.pyplot as plt
    scalar_fields(runs, fields)
    cols = min(3, len(fields))
    rows = math.ceil(len(fields)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4*cols,3.7*rows), squeeze=False, layout="constrained")
    for ax, key in zip(axes.flat, fields):
        for path, _, series in runs:
            _line(ax, x_values(series, kind), series[key], key=key, label=_label(path,root), linewidth=1.3)
        _finish_axis(ax,key,kind)
    for ax in list(axes.flat)[len(fields):]:
        ax.set_visible(False)
    # An anchor panel may be empty; collect the first defined lines for a legend.
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles,labels,loc="outside lower center",ncol=min(4,len(labels)),fontsize=8)
            break
    _save(fig,destination,stem)
    plt.close(fig)


def load_coordinate_panels(path):
    with np.load(path, allow_pickle=False) as archive:
        series = {key: archive[key] for key in archive.files}
    layers = sorted(key[:-8] for key in series if key.endswith("_indices"))
    return series, layers


def _display_coordinates(sample_types, maximum):
    # Interleave persistent/random/inactive categories without changing the
    # saved fixed panel, so a small display includes both selection groups.
    groups = [np.flatnonzero(sample_types == kind).tolist() for kind in dict.fromkeys(sample_types.tolist())]
    result = []
    while len(result) < maximum and any(groups):
        for group in groups:
            if group and len(result) < maximum:
                result.append(group.pop(0))
    return result


def plot_coordinates(runs, root, fields, kind, destination, maximum):
    import matplotlib.pyplot as plt
    for path, _, _ in runs:
        series, layers = load_coordinate_panels(path.parent/"coordinate_series.npz")
        for layer in layers:
            x = x_values(series,kind)
            indices, types = series[f"{layer}_indices"], series[f"{layer}_sample_type"]
            selected = _display_coordinates(types,maximum)
            cols = min(2,len(fields))
            fig, axes = plt.subplots(math.ceil(len(fields)/cols),cols,figsize=(5*cols,3.4*math.ceil(len(fields)/cols)),squeeze=False,layout="constrained")
            for ax,key in zip(axes.flat,fields):
                full_key = f"{layer}_{key}"
                if full_key not in series or series[full_key].shape != (len(x),len(indices)):
                    plt.close(fig)
                    raise ValueError(f"invalid coordinate field {full_key}: {path.parent}")
                for column in selected:
                    label = f"{types[column]} {indices[column]}"
                    _line(ax,x,series[full_key][:,column],key=key,label=label,linewidth=.9,alpha=.8)
                _finish_axis(ax,key,kind)
            for ax in list(axes.flat)[len(fields):]:
                ax.set_visible(False)
            fig.suptitle(f"{_label(path,root)} / {layer} — {len(selected)} of {len(indices)} fixed coordinates",fontsize=11)
            handles,labels = axes.flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles,labels,loc="outside lower center",ncol=min(4,len(labels)),fontsize=7)
            safe_name = "__".join(path.parent.relative_to(root).parts) or "run"
            _save(fig,destination,f"coordinates__{safe_name}__{layer}")
            plt.close(fig)


def plot_norm_efficiency(runs,root,destination):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    plotted = False
    for path,metadata,series in runs:
        q,gamma = series["raw_margin"],series["a1em4_frozen_l2_margin"]
        valid = np.isfinite(q) & np.isfinite(gamma) & (q > 0) & (gamma > 0)
        if not valid.any():
            continue
        degree = float(metadata["signature"]["homogeneity_degree"])
        log_norm = (np.log(q[valid])-np.log(gamma[valid]))/degree
        relative_norm = np.exp(log_norm-log_norm[0])
        label = _label(path,root)
        axes[0].plot(series["step"][valid],relative_norm,label=label)
        axes[1].plot(q[valid]-q[valid][0],relative_norm,label=label)
        plotted = True
    if plotted:
        axes[0].set(xlabel="Completed Adam updates t",ylabel="Frozen metric norm / own 1e-4-anchor norm")
        axes[1].set(xlabel="Minimum signed margin growth since anchor",ylabel="Frozen metric norm / own 1e-4-anchor norm")
        for ax in axes:
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        _save(fig,destination,"frozen_norm_efficiency")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path("output/mnist"))
    parser.add_argument("--fields",help="comma-separated scalar NPZ columns; default six-panel overview")
    parser.add_argument("--x",choices=("step","loss-clock"),default="step")
    parser.add_argument("--anchor-cosines",action="store_true",help="additional momentum-action and raw-v panels for all three loss anchors")
    parser.add_argument("--coordinates",action="store_true",help="plot fixed-coordinate panels, one figure per run and layer")
    parser.add_argument("--coordinate-fields",default=",".join(COORDINATE_FIELDS))
    parser.add_argument("--max-coordinates",type=int,default=12,help="display cap per saved layer panel; source data stay intact")
    parser.add_argument("--max-incoming-points",type=int,default=2000,help="bounded streaming display size; use incoming_analysis for full data")
    cli = parser.parse_args(argv)
    if cli.max_coordinates < 1 or cli.max_incoming_points < 2:
        parser.error("display caps must be positive (incoming at least 2)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = load_runs(cli.run_dir)
    if not runs:
        raise ValueError("no finite-step trajectories to plot")
    cli.output_dir.mkdir(parents=True,exist_ok=True)
    fields = tuple(key.strip() for key in cli.fields.split(",") if key.strip()) if cli.fields else DEFAULT_FIELDS
    if not fields:
        parser.error("--fields must contain at least one scalar column")
    plot_scalar_grid(runs,cli.run_dir,fields,cli.x,cli.output_dir,"trajectories")
    if cli.anchor_cosines:
        fields = tuple(f"{anchor}_{metric}" for metric in ("momentum_action_cosine","second_moment_cosine") for anchor in ANCHORS)
        plot_scalar_grid(runs,cli.run_dir,fields,cli.x,cli.output_dir,"anchor_cosines")
    if cli.coordinates:
        fields = tuple(key.strip() for key in cli.coordinate_fields.split(",") if key.strip())
        if not fields:
            parser.error("--coordinate-fields must contain at least one field")
        plot_coordinates(runs,cli.run_dir,fields,cli.x,cli.output_dir,cli.max_coordinates)
    plot_norm_efficiency(runs,cli.run_dir,cli.output_dir)
    incoming = [p for p,_,_ in runs if (p.parent/"incoming_second_moment.jsonl").exists()]
    if incoming:
        fig,ax = plt.subplots(figsize=(7,4.5),layout="constrained")
        for path in incoming:
            x,y,total = read_incoming_for_display(path.parent/"incoming_second_moment.jsonl",cli.max_incoming_points)
            ax.plot(x,y,label=f"{_label(path,cli.run_dir)} ({len(x)}/{total} rows)")
        ax.set(xlabel="Completed Adam updates t",ylabel="Natural log prospective transformed increment D_t")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        _save(fig,cli.output_dir,"incoming_second_moment")
        plt.close(fig)
    print(f"Saved figures in {cli.output_dir.resolve()}")

if __name__ == "__main__":
    main()
