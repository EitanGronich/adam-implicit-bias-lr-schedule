"""Prepare figure data from one completed MNIST experiment sweep."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from .data import write_sections
from .prepare_traces import load_sweep, summarize_traces
from .prepare_endpoints import summarize_endpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"),
                        help="directory containing MNIST/raw; synthetic runs recreate their data")
    parser.add_argument("--beta2", type=float, help="select one decay rate from a beta2 sweep")
    parser.add_argument("--secant-window", type=int, default=10000)
    parser.add_argument("--error-window", type=int, default=1000)
    parser.add_argument("--selected-alpha", type=float,
                        help="schedule for the individual-seed panel; defaults to -0.3 if present")
    args = parser.parse_args(argv)
    if args.secant_window < 1 or args.error_window < 1:
        parser.error("Window lengths must be positive")
    sweep = load_sweep(args.runs_dir, beta2=args.beta2)
    sections = summarize_traces(sweep, secant_window=args.secant_window,
                                error_window=args.error_window,
                                selected_alpha=args.selected_alpha)
    sections.update(summarize_endpoints(sweep, args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=args.output.parent) as directory:
        completed = Path(directory) / "experiment.npz"
        write_sections(completed, **sections)
        completed.replace(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
