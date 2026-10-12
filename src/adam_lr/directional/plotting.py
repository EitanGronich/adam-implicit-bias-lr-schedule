"""Plots of saved Adam states, with no predicted or extrapolated limit curves."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def plot_run(directory: str | Path, *, max_trajectories: int = 12) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    with np.load(directory / "history.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if max_trajectories < 1:
        raise ValueError("max_trajectories must be positive")
    steps = data["step"]
    count, dimension = data["theta"].shape[1:]
    select = np.unique(np.linspace(0, count-1, min(max_trajectories, count)).astype(int))
    coords = ["u1", "u2"] + [f"b{j}" for j in range(1, dimension-1)]
    colors = plt.get_cmap("viridis")(np.linspace(.05, .9, len(select)))
    written = []
    style = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
             "figure.dpi": 130, "savefig.dpi": 180, "axes.grid": True,
             "grid.alpha": .2, "axes.titlepad": 8}

    def save(fig, stem):
        for suffix in ("png", "pdf"):
            path = directory / f"{stem}.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
        for offset, index in enumerate(select):
            color = colors[offset]
            axes[0, 0].plot(steps, data["log_loss"][:, index], color=color, alpha=.8)
            axes[0, 1].plot(steps, data["margin"][:, index, 0], color=color, alpha=.8)
            axes[0, 2].plot(steps, data["theta_norm"][:, index], color=color, alpha=.8)
        axes[0, 0].set(title="Observed exponential objective", ylabel="log(exp(-z_S) + exp(-z_T))")
        axes[0, 1].set(title="Observed support margin", ylabel="z_S")
        axes[0, 2].set(title="Parameter growth", ylabel="Euclidean norm of theta")
        for ax, key, label in ((axes[1, 0], "gradient", "Absolute gradient coordinates"),
                               (axes[1, 1], "second_moment", "Raw second-moment coordinates"),
                               (axes[1, 2], "last_update", "Absolute last Adam update")):
            values = np.abs(data[key]).reshape(len(steps), -1)
            # Zeros are masked, not replaced by an arbitrary artificial positive floor.
            lo, med, hi = np.quantile(values, [.1, .5, .9], axis=1)
            ax.fill_between(steps, np.where(lo > 0, lo, np.nan), np.where(hi > 0, hi, np.nan), alpha=.2, color="#187f83")
            ax.plot(steps, np.where(med > 0, med, np.nan), color="#187f83", label="median; band 10-90%")
            if np.any(values > 0):
                ax.set_yscale("log")
            ax.set(title=label, ylabel="Magnitude")
            ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.set_xlabel("Accepted Adam updates")
            if steps[-1] > 10:
                ax.set_xscale("symlog", linthresh=10)
        fig.suptitle(f"Actual discrete Adam | {summary['accepted_steps']:,} accepted steps | {summary['stop_reason']}")
        save(fig, "diagnostics")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for coordinate, label in enumerate(coords):
            values = data["unit"][:, select, coordinate]
            axes[0].plot(steps, np.median(values, axis=1), label=label)
        axes[0].set(title="Parameter direction coordinates", ylabel="Median theta_i / ||theta|| (displayed runs)")
        axes[0].legend(fontsize=8, ncols=2)
        for ax, key, title in ((axes[1], "checkpoint_drift_deg", "Drift since previous saved state"),
                               (axes[2], "initial_drift_deg", "Drift from initialization")):
            values = data[key]
            lo, med, hi = np.quantile(values, [.1, .5, .9], axis=1)
            ax.fill_between(steps, lo, hi, color="#a55532", alpha=.18)
            ax.plot(steps, med, color="#a55532", label="median; band 10-90%")
            ax.set(title=title, ylabel="Angle (degrees)")
            ax.legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("Accepted Adam updates")
            if steps[-1] > 10:
                ax.set_xscale("symlog", linthresh=10)
        fig.suptitle("Finite-run observed directions; a small drift does not establish a limit")
        save(fig, "directions")

        if dimension == 4:
            q = data["unit"][-1] ** 2
            # An affine display of the positive orthant of S^3: no inferred directions.
            vertices = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)
            points = q @ vertices
            fig = plt.figure(figsize=(7.4, 6.8), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
            for i in range(4):
                for j in range(i+1, 4):
                    ax.plot(*vertices[[i, j]].T, color="#6a737b", alpha=.6, linewidth=1)
                ax.text(*((1.12 * vertices)[i]), coords[i], weight="bold")
            scatter = ax.scatter(*points.T, c=data["controls"][:, 0], cmap="viridis", s=9, alpha=.75, linewidths=0)
            fig.colorbar(scatter, ax=ax, shrink=.65, pad=.02, label="Initialization control s")
            ax.set_box_aspect([1, 1, 1])
            ax.set_axis_off()
            ax.view_init(elev=18, azim=-58)
            paired = bool(data["sector_swapped"].any() and (~data["sector_swapped"]).any())
            sector_caption = "Two coordinate-permuted datasets" if paired else "One fixed dataset"
            ax.set_title(f"{sector_caption}\nObserved endpoint directions after {summary['accepted_steps']:,} updates\nq_i = theta_i^2 / ||theta||^2; each point is an actual run")
            save(fig, "endpoint_tetrahedron")
    provenance = {"plotting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "history_sha256": hashlib.sha256((directory / "history.npz").read_bytes()).hexdigest(),
                  "summary_sha256": hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(),
                  "matplotlib_version": matplotlib.__version__, "max_trajectories": max_trajectories}
    (directory / "plot_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return written
