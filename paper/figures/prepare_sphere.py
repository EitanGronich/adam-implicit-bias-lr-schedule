"""Extract final unit parameter directions from a completed width-1 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def prepare(run: Path, output: Path) -> None:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    accepted = summary.get("accepted_steps")
    requested = config.get("run", {}).get("steps", summary.get("requested_steps"))
    reason = summary.get("stop_reason", "step_limit")
    if (config.get("model", {}).get("layers") != 2
            or not isinstance(accepted, int) or accepted < 0
            or not isinstance(requested, int) or not 0 <= accepted <= requested
            or reason != "step_limit" or accepted != requested
            or "guard_event" not in summary or summary["guard_event"] is not None):
        raise ValueError("Expected a completed, unguarded depth-2 sphere run")
    with np.load(run / "history.npz", allow_pickle=False) as history:
        steps = history["step"]
        units = history["unit"]
        if (steps.ndim != 1 or not len(steps) or not np.issubdtype(steps.dtype, np.integer)
                or steps[0] < 0 or (np.diff(steps) <= 0).any() or steps[-1] != accepted):
            raise ValueError("Sphere history must reach its reported final update")
        if (units.ndim != 3 or units.shape[0] != len(steps)
                or units.shape[2] != 3 or units.shape[1] < 1):
            raise ValueError("Expected saved three-coordinate parameter directions")
        points = units[-1]
        swapped = history["sector_swapped"]
    if swapped.shape != (len(points),):
        raise ValueError("Expected one dataset label per endpoint direction")
    if (not np.isfinite(points).all() or (points <= 0).any()
            or not np.allclose(np.linalg.norm(points, axis=1), 1, atol=1e-6, rtol=0)
            or not np.isin(swapped, (0, 1)).all()):
        raise ValueError("Expected positive unit directions and dataset labels 0 or 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, np.column_stack([np.arange(len(points)), swapped.astype(int), points]),
               delimiter=",", header="trajectory,sector_swapped,q_u1,q_u2,q_b1",
               comments="", fmt="%.17g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.run_dir, args.output)


if __name__ == "__main__":
    main()
