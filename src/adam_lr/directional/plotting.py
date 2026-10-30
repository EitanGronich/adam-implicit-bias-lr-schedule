"""Loss and parameter-direction plots from width-1 experiment outputs."""
from pathlib import Path

import numpy as np


def plot_run(directory: str | Path, *, max_trajectories: int = 12) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    with np.load(directory / "history.npz", allow_pickle=False) as source:
        steps, loss = source["step"], source["log_loss"]
        unit, swapped = source["unit"], source["sector_swapped"]
    if max_trajectories < 1:
        raise ValueError("max_trajectories must be positive")
    selected = np.unique(np.linspace(0, unit.shape[1] - 1, min(max_trajectories, unit.shape[1])).astype(int))
    written = []

    def save(fig, name):
        for suffix in ("pdf", "png"):
            path = directory / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight", pad_inches=.3, dpi=180)
            written.append(path)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(steps, loss[:, selected], lw=.8)
    ax.set(xlabel="Adam updates", ylabel="Log exponential loss")
    save(fig, "loss")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    names = ["u1", "u2"] + [f"b{j}" for j in range(1, unit.shape[2] - 1)]
    for j, name in enumerate(names):
        ax.plot(steps, np.median(unit[:, selected, j], axis=1), label=name)
    ax.set(xlabel="Adam updates", ylabel="Median unit parameter coordinate")
    ax.legend()
    save(fig, "directions")
    if unit.shape[2] == 3:
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(projection="3d")
        for label, color in ((False, "#0072B2"), (True, "#D55E00")):
            points = unit[-1, swapped == label]
            if len(points):
                ax.scatter(*points.T, color=color, s=15, depthshade=False)
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="u1", ylabel="u2", zlabel="b1")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=45)
        save(fig, "endpoint_sphere")
    return written
