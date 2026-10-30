"""CLI: python -m adam_lr.directional --config configs/directional/sphere.toml --steps 1000."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import load_config
from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental validation of the width-1 construction from the paper using Adam.")
    parser.add_argument("--config", type=Path, help="Readable TOML experiment configuration")
    parser.add_argument("--output", type=Path, help="Override run.output_dir; must be empty or new")
    parser.add_argument("--steps", type=int, help="Override run.steps")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", type=Path, help="Plot an existing run directory")
    args = parser.parse_args()
    if args.plot_only:
        from .plotting import plot_run
        plot_run(args.plot_only)
        return
    if args.config is None:
        parser.error("--config is required unless --plot-only is supplied")
    config = load_config(args.config)
    if args.steps is not None:
        config = replace(config, run=replace(config.run, steps=args.steps))
    if args.output is not None:
        config = replace(config, run=replace(config.run, output_dir=str(args.output)))
    if args.no_plots:
        config = replace(config, logging=replace(config.logging, plots=False))
    summary = run(config, config_source=args.config.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, allow_nan=False))
    if summary["stop_reason"] in {"initialization_guard", "numerical_guard", "parameter_stagnation_observed"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
