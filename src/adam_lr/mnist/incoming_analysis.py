"""Analyze actual adjacent-step incoming transformed second moments.

At state t, W_t=beta2**(-t)*v_t and D_t=||W_(t+1)-W_t||_1.
For x_t=-log10(loss_t)>0, Fbar_t=D_t/(ln x_(t+1)-ln x_t) is
a transition-average mass per unit ln(x), not a numerical derivative fitted
to sparsely sampled loss. The terminal prospective D has no successor and
is never used. Undefined transitions stay gaps; no seed bands are inferred.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load_trace(run_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a complete contiguous trace using compact arrays, checking identity."""
    root = Path(run_dir)
    result = json.loads((root / "results.json").read_text(encoding="utf-8"))
    if result.get("completion_reason") not in {"target_loss", "max_steps", "signal", "checkpoint_finalize"}:
        raise ValueError(f"refusing a failed or unrecognized run: {root}")
    meta_path = root / "incoming_second_moment.meta.json"
    meta_bytes = meta_path.read_bytes()
    metadata = json.loads(meta_bytes)
    if metadata.get("version") != 1 or metadata.get("signature") != result.get("signature"):
        raise ValueError(f"trace/result signature mismatch: {root}")
    layers = metadata.get("layer_names")
    if (not isinstance(layers, list) or not layers or len(set(layers)) != len(layers)
            or any(not isinstance(name, str) for name in layers)):
        raise ValueError("invalid trace layer names")
    columns = ["step", "log_loss", "log_D_total", *(f"log_D_{name}" for name in layers)]
    if metadata.get("columns") != columns:
        raise ValueError("unsupported trace columns")
    losses, increments = array("d"), array("d")
    trace_path = root / "incoming_second_moment.jsonl"
    digest = hashlib.sha256()
    with trace_path.open("rb") as stream:
        for step, line in enumerate(stream):
            digest.update(line)
            row = json.loads(line)
            if (not isinstance(row, dict) or set(row) != set(columns)
                    or type(row.get("step")) is not int or row["step"] != step
                    or not line.endswith(b"\n")):
                raise ValueError(f"trace is not contiguous at step {step}: {root}")
            loss = row["log_loss"]
            value = row["log_D_total"]
            if type(loss) not in (int, float) or not math.isfinite(loss):
                raise ValueError("log loss must be finite")
            for key in columns[2:]:
                if row[key] is not None and (type(row[key]) not in (int, float) or not math.isfinite(row[key])):
                    raise ValueError("log incoming mass must be finite or null")
            losses.append(loss)
            increments.append(-math.inf if value is None else value)
    if (len(losses) < 2 or len(losses) != result["final_step"] + 1
            or metadata.get("rows_written") != len(losses)
            or metadata.get("last_step_written") != result["final_step"]):
        raise ValueError("trace is incomplete or has no completed transition")
    if not (result.get("final_loss", 0) > 0 and math.isclose(
            losses[-1], math.log(result["final_loss"]), rel_tol=0, abs_tol=1e-10)):
        raise ValueError("trace endpoint loss does not match result")
    for file, sha in ((trace_path, digest.hexdigest()),
                      (meta_path, hashlib.sha256(meta_bytes).hexdigest())):
        identity = result.get("artifacts", {}).get(file.name, {})
        if identity.get("sha256") != sha or identity.get("size_bytes") != file.stat().st_size:
            raise ValueError(f"artifact identity mismatch: {file}")
    return np.frombuffer(losses, dtype=np.float64), np.frombuffer(increments, dtype=np.float64), result


def transition_values(log_losses, log_increments, *, reference_loss: float = 1e-8) -> dict:
    """Return every transition after the threshold, preserving invalid gaps.

    Each run is normalized by its FIRST valid transition starting at or after
    the first state reaching reference_loss. No interpolation or branch
    selection is performed, including if the loss is nonmonotone.
    """
    if not math.isfinite(reference_loss) or not 0 < reference_loss < 1:
        raise ValueError("reference_loss must lie strictly between zero and one")
    losses = np.asarray(log_losses, dtype=np.float64)
    incoming = np.asarray(log_increments, dtype=np.float64)
    if losses.ndim != 1 or incoming.shape != losses.shape or len(losses) < 2:
        raise ValueError("aligned one-dimensional traces of at least two states are required")
    if not np.isfinite(losses).all() or np.any(np.isnan(incoming) | np.isposinf(incoming)):
        raise ValueError("nonfinite loss or invalid incoming logarithm")
    anchors = np.flatnonzero(losses <= math.log(reference_loss))
    if not len(anchors) or anchors[0] == len(losses)-1:
        raise ValueError("no transition remains after the requested reference loss")
    start = int(anchors[0])
    # ln(x_next/x) = ln(z_next/z), z=-ln(loss). Scaling z by ln(10)
    # first can round two distinct adjacent clocks to the same float.
    z = -losses[start:]
    valid_pair = (z[:-1] > 0) & (z[1:] > 0)
    delta = np.full(len(z)-1, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta[valid_pair] = np.log1p(np.diff(z)[valid_pair] / z[:-1][valid_pair])
        abscissa = (np.log10(z[:-1]) + np.log10(z[1:])) / 2 - math.log10(math.log(10))
    log_d = incoming[start:-1]  # terminal row is prospective, not applied
    valid = np.isfinite(delta) & (delta > 0) & np.isfinite(log_d)
    if not valid.any():
        raise ValueError("no positive-mass, loss-decreasing transition after reference")
    log_f = np.full(delta.shape, np.nan)
    log_f[valid] = log_d[valid] - np.log(delta[valid])
    reference = int(np.flatnonzero(valid)[0])
    return {"step": np.arange(start, len(losses)-1), "log10_x": abscissa,
            "log_F": log_f,
            "log10_relative_F": (log_f-log_f[reference])/math.log(10),
            "valid": valid, "reference_step": start+reference,
            "invalid_transitions": int((~valid).sum())}


def display_indices(values: np.ndarray, max_points: int) -> np.ndarray:
    """Thin display points, with a soft cap to preserve all gap boundaries."""
    if max_points < 2:
        raise ValueError("max_points must be at least two")
    valid = np.isfinite(values)
    boundaries = np.flatnonzero(valid[1:] != valid[:-1])
    stride = max(1, math.ceil(len(values)/max_points))
    return np.unique(np.r_[np.arange(0, len(values), stride), boundaries,
                           boundaries+1, 0, len(values)-1]).astype(int)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run or suite root; finds every-step traces recursively")
    parser.add_argument("--output-dir", type=Path, default=Path("output/incoming_analysis"))
    parser.add_argument("--reference-loss", type=float, default=1e-8)
    parser.add_argument("--max-plot-points", type=int, default=4000,
                        help="soft display budget; gap boundaries are always retained")
    args = parser.parse_args(argv)
    sources = sorted(args.run_dir.rglob("incoming_second_moment.meta.json"))
    if not sources:
        raise ValueError("no every-step incoming traces found")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    manifest = []
    for index, source in enumerate(sources):
        loss, incoming, result = load_trace(source.parent)
        data = transition_values(loss, incoming, reference_loss=args.reference_loss)
        output = args.output_dir / f"run_{index:03d}.npz"
        np.savez_compressed(output, **data)
        label = source.parent.relative_to(args.run_dir).as_posix()
        label = source.parent.name if label == "." else label
        chosen = display_indices(data["log10_relative_F"], args.max_plot_points)
        ax.plot(data["log10_x"][chosen], data["log10_relative_F"][chosen], label=label, linewidth=1)
        manifest.append({"run": label, "signature": result["signature"],
                         "trace_artifacts": result["artifacts"],
                         "reference_step": data["reference_step"],
                         "invalid_transitions": data["invalid_transitions"],
                         "processed_file": output.name, "final_step": result["final_step"]})
    ax.set(xlabel=r"$\log_{10}x$, $x=-\log_{10}\mathcal{L}$",
           ylabel=r"$\log_{10}(\overline{F}_t/F_{\mathrm{ref}})$",
           title="Adjacent-step incoming mass; each line is a run, gaps are undefined transitions")
    ax.axhline(0, color="0.5", linestyle=":", linewidth=1)
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, loc="best")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir/f"incoming_normalized.{suffix}", dpi=180)
    plt.close(fig)
    (args.output_dir/"manifest.json").write_text(json.dumps({"reference_loss": args.reference_loss,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "transition averages; per-run normalization; no pooled seed statistics",
        "display_thinning": "full processed values retained; gap boundaries preserved", "runs": manifest},
        indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(sources)} processed runs and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
