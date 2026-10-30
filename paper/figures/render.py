"""Render the manuscript figures from the numerical bundles in paper/data."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FAMILIES = ("mnist", "epsilon", "endpoints", "error", "construction")
ASSETS = {
    "mnist": tuple("mnist/" + name + ".pdf" for name in (
        "panel_loss_rate", "panel_linf_margin", "panel_frozen_margin",
        "panel_frozen_margin_am07", "legend_schedules", "legend_seeds",
        "alpha_03_all_seeds")),
    "epsilon": tuple("epsilon_1e8/" + name + ".pdf" for name in (
        "panel_a", "panel_b", "panel_c", "panel_d", "legend_schedules",
        "legend_epsilon", "legend_seeds")),
    "endpoints": ("mnist/epsilon0_jumbo.pdf",
                  "epsilon_1e8/epsilon0_scale1_jumbo.pdf",
                  "epsilon_1e8/epsilon1e8_scale1_jumbo.pdf"),
    "error": ("epsilon_1e8/err.pdf",),
    "construction": ("sphere.pdf", "adam_construction_tikz.pdf"),
}


def construction(data_dir, output, skip_tex=False, tex_engine="xelatex", formats=("pdf", "png")):
    from paper.figures.sphere import render
    render(data_dir / "sphere.csv", output / "sphere.pdf")
    if "png" in formats:
        import pymupdf
        with pymupdf.open(output / "sphere.pdf") as document:
            document[0].get_pixmap(dpi=1000, alpha=True).save(output / "sphere.png")
    if skip_tex:
        return
    engine = shutil.which(tex_engine)
    if engine is None:
        raise RuntimeError("XeLaTeX is required for the schematic; use --skip-tex to render numerical figures only")
    source = REPO / "paper/diagrams/adam_construction.tex"
    with tempfile.TemporaryDirectory(prefix="paper-tex-", dir=output.parent) as directory:
        result = subprocess.run([engine, "-no-shell-escape", "-interaction=nonstopmode",
                                 "-halt-on-error", "-jobname=adam_construction_tikz",
                                 f"-output-directory={directory}", str(source)],
                                cwd=source.parent, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        if result.returncode:
            raise RuntimeError("XeLaTeX failed:\n" + result.stdout.decode(errors="replace")[-6000:])
        shutil.copy2(Path(directory) / "adam_construction_tikz.pdf",
                     output / "adam_construction_tikz.pdf")
    if "png" in formats:
        import pymupdf
        with pymupdf.open(output / "adam_construction_tikz.pdf") as document:
            document[0].get_pixmap(dpi=1000, alpha=False).save(output / "adam_construction_tikz.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("all", *FAMILIES), default="all")
    parser.add_argument("--data-dir", type=Path, default=REPO / "paper/data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "output/paper")
    parser.add_argument("--format", choices=("pdf", "both"), default="pdf",
                        help="PDF only (default), or PDF and PNG")
    parser.add_argument("--skip-tex", action="store_true",
                        help="omit the TikZ schematic when XeLaTeX is unavailable")
    parser.add_argument("--tex-engine", default="xelatex")
    parser.add_argument("--list", action="store_true", help="list manuscript PDF paths without rendering")
    args = parser.parse_args(argv)
    selected = FAMILIES if args.family == "all" else (args.family,)
    if args.list:
        for family in selected:
            for asset in ASSETS[family]:
                if not (args.skip_tex and asset == "adam_construction_tikz.pdf"):
                    print("figures/" + asset)
        return
    data_dir = args.data_dir.resolve()
    output = args.output_dir.resolve() / "figures"
    output.mkdir(parents=True, exist_ok=True)
    formats = ("pdf", "png") if args.format == "both" else ("pdf",)
    for family in selected:
        if family == "construction":
            construction(data_dir, output, args.skip_tex, args.tex_engine, formats)
        elif family == "mnist":
            from paper.figures.mnist import render
            render(data_dir, output / "mnist", formats)
        elif family == "epsilon":
            from paper.figures.epsilon import render
            render(data_dir, output / "epsilon_1e8", formats)
        elif family == "endpoints":
            from paper.figures.endpoints import render
            render(data_dir, output, formats)
        elif family == "error":
            from paper.figures.error import render
            render(data_dir / "comparison.npz", output / "epsilon_1e8", formats)
        print(f"Rendered {family}: {output}", flush=True)


if __name__ == "__main__":
    main()
