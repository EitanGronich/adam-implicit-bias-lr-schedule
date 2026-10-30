"""Render the width-1 experiment from its saved parameter directions."""
from __future__ import annotations

import argparse
from pathlib import Path
from shutil import which

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pymupdf


DATA = Path(__file__).resolve().parents[1] / "data" / "sphere.csv"


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if (values.ndim != 2 or values.shape[1] != 5 or not len(values)
            or not np.isfinite(values).all()):
        raise ValueError("Expected a nonempty table of finite endpoint directions")
    if not np.array_equal(values[:, 0], np.arange(len(values))):
        raise ValueError("Endpoint rows must retain their trajectory order")
    sectors = values[:, 1]
    points = values[:, 2:]
    if (not np.isin(sectors, (0, 1)).all()
            or (points <= 0).any()
            or not np.allclose(np.linalg.norm(points, axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Expected positive unit directions and dataset labels 0 or 1")
    return points, sectors.astype(bool)


def font_style() -> tuple[dict, FontProperties]:
    executable = which("pdflatex")
    relative_font = Path("fonts/type1/public/amsfonts/cm/cmmib10.pfb")
    native_font = next((parent / relative_font for parent in Path(executable).resolve().parents
                        if (parent / relative_font).is_file()), None) if executable else None
    native = native_font is not None
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 9, "mathtext.fontset": "cm", "pdf.fonttype": 3 if native else 42,
             "axes.linewidth": .75}
    if native:
        style.update({"mathtext.it": FontProperties(fname=native_font).get_fontconfig_pattern(),
                      "mathtext.rm": "cmr10"})
    return style, FontProperties(math_fontfamily="custom" if native else "cm")


def render(data: Path, output: Path) -> None:
    points, swapped = load_points(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    style, font = font_style()
    with plt.rc_context(style):
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
        angle = np.linspace(0, np.pi / 2, 101)
        for fixed in np.linspace(0, np.pi / 2, 7):
            ax.plot(np.cos(angle) * np.cos(fixed), np.sin(angle) * np.cos(fixed),
                    np.full_like(angle, np.sin(fixed)), color="#9da9b2", lw=.65, alpha=.38,
                    zorder=1)
            ax.plot(np.cos(fixed) * np.cos(angle), np.sin(fixed) * np.cos(angle),
                    np.sin(angle), color="#9da9b2", lw=.65, alpha=.38, zorder=1)
        for coordinates in ((np.cos(angle), np.sin(angle), np.zeros_like(angle)),
                            (np.cos(angle), np.zeros_like(angle), np.sin(angle)),
                            (np.zeros_like(angle), np.cos(angle), np.sin(angle))):
            ax.plot(*coordinates, color="#77858f", lw=.91, alpha=.65, zorder=2)
        for mask, color in ((~swapped, "#0072B2"), (swapped, "#D55E00")):
            ax.scatter(*points[mask].T, color=color, marker="o", s=25,
                       edgecolors="white", linewidths=.26, alpha=.9, depthshade=False,
                       zorder=5)
        for coordinate, label in enumerate((r"$u_1$", r"$u_2$", r"$b_1$")):
            vector = np.zeros(3)
            vector[coordinate] = 1.08
            ax.plot(*np.stack([np.zeros(3), vector]).T, color="#4d5963", lw=.975, zorder=3)
            position = vector.copy()
            position[coordinate] = 1.18 if coordinate < 2 else 1.15
            ax.text(*position, label, ha="center", va="center", fontsize=20,
                    fontproperties=font, color="#24292f")
            for tick in (.5, 1.):
                point = np.zeros(3)
                point[coordinate] = tick
                displacement = (np.array([0., 0., -.045]) if coordinate < 2
                                else np.array([-.065, .065, 0.]))
                ax.plot(*np.stack([point, point + .35 * displacement]).T,
                        color="#4d5963", lw=.715, zorder=3)
        ax.set(xlim=(-.08, 1.17), ylim=(-.08, 1.17), zlim=(-.08, 1.17))
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        ax.view_init(elev=23, azim=45)
        ax.set_axis_off()
        fig.savefig(output, transparent=True, dpi=1000)
        plt.close(fig)
    with pymupdf.open(output) as document:
        page = document[0]
        bounds = [item["rect"] for item in page.get_drawings()]
        bounds += [pymupdf.Rect(block[:4]) for block in page.get_text("blocks")]
        box = pymupdf.Rect(bounds[0])
        for bound in bounds[1:]:
            box |= bound
        page.set_cropbox(pymupdf.Rect(box.x0 - 6, box.y0 - 6, box.x1 + 6, box.y1 + 6) & page.rect)
        document.save(output.with_suffix(".cropped.pdf"), garbage=4, deflate=True)
    output.with_suffix(".cropped.pdf").replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=Path("output/paper/figures/sphere.pdf"))
    args = parser.parse_args()
    render(args.data, args.output)


if __name__ == "__main__":
    main()
