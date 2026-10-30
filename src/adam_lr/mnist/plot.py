"""Plot loss and normalized margins from saved experiment traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .traces import load_trace


FIELDS = {"loss": "Mean logistic loss", "margin_linf": r"$\gamma_\infty$",
          "margin_l2": r"$\gamma_2$"}


def load_runs(directory: Path) -> list[tuple[dict, dict]]:
    paths = sorted(Path(directory).rglob("results.json"))
    if not paths:
        raise ValueError(f"No completed experiment runs found in {directory}")
    runs = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        trace = load_trace(path.parent)
        series = {"step": trace["step"], "loss": trace["loss"]}
        series.update({name: trace["values"][:, index]
                       for index, name in enumerate(trace["metric_columns"])})
        runs.append((result["signature"], series))
    return runs


def plot(directory: Path, output: Path, *, fields=("loss", "margin_linf"), x="step") -> None:
    if x not in ("step", "loss-clock") or not fields or set(fields) - FIELDS.keys():
        raise ValueError("Select loss, margin_linf or margin_l2, and step or loss-clock")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(directory)
    output.mkdir(parents=True, exist_ok=True)
    groups = sorted({(r["alpha"], r["epsilon"], r["beta2"]) for r, _ in runs})
    colors = {group: f"C{i % 10}" for i, group in enumerate(groups)}
    for field in fields:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        labeled = set()
        for settings, series in runs:
            group = (settings["alpha"], settings["epsilon"], settings["beta2"])
            label = (f"alpha={group[0]:g}, epsilon={group[1]:g}, beta2={group[2]:g}"
                     if group not in labeled else "_nolegend_")
            labeled.add(group)
            axis = series["step"] if x == "step" else -np.log10(series["loss"])
            ax.plot(axis, series[field], color=colors[group], label=label,
                    linewidth=1, alpha=.8)
        ax.set(xlabel="Update" if x == "step" else r"$-\log_{10}\mathcal{L}$",
               ylabel=FIELDS[field])
        if field == "loss":
            ax.set_yscale("log")
        ax.legend(fontsize=7)
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(output / f"{field}.{extension}", dpi=200)
        plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fields", default="loss,margin_linf")
    parser.add_argument("--x", choices=("step", "loss-clock"), default="step")
    args = parser.parse_args(argv)
    plot(args.runs, args.output_dir, fields=tuple(args.fields.split(",")), x=args.x)


if __name__ == "__main__":
    main()
