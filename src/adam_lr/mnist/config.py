"""TOML settings for full-batch MNIST experiments."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import tomllib


DEFAULTS = {
    "experiment": {"seed": 2026100901, "output_dir": "runs/mnist", "dtype": "float64", "device": "auto", "threads": 2},
    "dataset": {"kind": "mnist", "data_dir": "data", "per_class": 512, "center": True, "download": False, "input_dim": 784, "seed": 709},
    "model": {"hidden_widths": [512, 512], "bias": False, "initialization": "torch_default", "initialization_std": 1.0, "layer_scale": 1e-4},
    "optimizer": {"learning_rate": 0.0003, "beta1": 0.9, "beta2": 0.999, "epsilon": 0.0},
    "run": {"max_steps": 1000000, "target_loss": 1e-20, "stop_at_target": True, "resume": True},
    "logging": {"trace_buffer_rows": 1024, "checkpoint_every": 10000, "checkpoint_every_seconds": 600.0, "progress_every": 2500, "full_checkpoint_losses": []},
}


def _number(value, label, *, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if positive and value <= 0 or minimum is not None and value < minimum:
        raise ValueError(f"{label} is outside its allowed range")


def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def from_mapping(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS) - {"schedules", "sweep"}:
        raise ValueError("unknown configuration sections")
    sections = {}
    for name, defaults in DEFAULTS.items():
        supplied = raw.get(name, {})
        if not isinstance(supplied, dict) or set(supplied) - set(defaults):
            raise ValueError(f"unknown or invalid keys in [{name}]")
        sections[name] = {**defaults, **supplied}
        for key, value in sections[name].items():
            if isinstance(defaults[key], bool) and not isinstance(value, bool):
                raise ValueError(f"{name}.{key} must be true or false")
    e, d, m, o, r, log = (sections[name] for name in DEFAULTS)
    if e["dtype"] not in ("float32", "float64") or e["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("select dtype float32/float64 and device auto/cpu/cuda")
    _integer(e["seed"], "experiment.seed", 0)
    _integer(e["threads"], "experiment.threads")
    if d["kind"] not in ("mnist", "synthetic"):
        raise ValueError("dataset.kind must be mnist or synthetic")
    for key in ("per_class", "input_dim", "seed"):
        _integer(d[key], f"dataset.{key}", 0 if key == "seed" else 1)
    if not isinstance(m["hidden_widths"], list):
        raise ValueError("model.hidden_widths must be an array")
    for width in m["hidden_widths"]:
        _integer(width, "model.hidden_widths entry")
    if m["initialization"] not in ("torch_default", "normal"):
        raise ValueError("model.initialization must be torch_default or normal")
    _number(m["initialization_std"], "model.initialization_std", positive=True)
    _number(m["layer_scale"], "model.layer_scale", positive=True)
    _number(o["learning_rate"], "optimizer.learning_rate", positive=True)
    _number(o["epsilon"], "optimizer.epsilon", minimum=0)
    for key in ("beta1", "beta2"):
        _number(o[key], f"optimizer.{key}")
        if not 0 <= o[key] < 1 or key == "beta2" and o[key] == 0:
            raise ValueError("beta1 must be in [0,1); beta2 must be in (0,1)")
    _integer(r["max_steps"], "run.max_steps")
    _number(r["target_loss"], "run.target_loss", positive=True)
    for key in ("trace_buffer_rows", "checkpoint_every", "progress_every"):
        _integer(log[key], f"logging.{key}", 1 if key == "trace_buffer_rows" else 0)
    _number(log["checkpoint_every_seconds"], "logging.checkpoint_every_seconds", minimum=0)
    losses = log["full_checkpoint_losses"]
    if not isinstance(losses, list):
        raise ValueError("logging.full_checkpoint_losses must be an array")
    for loss in losses:
        _number(loss, "logging.full_checkpoint_losses entry", positive=True)
        if r["stop_at_target"] and loss < r["target_loss"]:
            raise ValueError("loss checkpoints below target cannot be reached")
    if len(set(losses)) != len(losses):
        raise ValueError("loss checkpoint thresholds must be distinct")
    sweep = raw.get("sweep", {})
    if not isinstance(sweep, dict) or set(sweep) - {"seeds", "beta2_values", "epsilon_values"}:
        raise ValueError("invalid [sweep] settings")
    axes = {"seeds": sweep.get("seeds", [e["seed"]]),
            "beta2_values": sweep.get("beta2_values", [o["beta2"]]),
            "epsilon_values": sweep.get("epsilon_values", [o["epsilon"]])}
    for key, values in axes.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.{key} must be a nonempty array")
        for value in values:
            if key == "seeds":
                _integer(value, "sweep seed", 0)
            else:
                _number(value, f"sweep.{key} entry", minimum=0)
                if key == "beta2_values" and not 0 < value < 1:
                    raise ValueError("sweep beta2 values must be in (0,1)")
        if len(set(values)) != len(values):
            raise ValueError("sweep entries must be unique")
    schedules = raw.get("schedules", [{"name": "constant", "alpha": 0.0}])
    if not isinstance(schedules, list) or not schedules:
        raise ValueError("at least one [[schedules]] table is required")
    specs = []
    for item in schedules:
        if not isinstance(item, dict) or set(item) - {"name", "alpha", "learning_rate", "max_steps"}:
            raise ValueError("invalid schedule keys; use alpha for eta_k=eta_0*k**alpha")
        name, alpha = item.get("name"), item.get("alpha")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("schedule name must be a simple directory name")
        _number(alpha, f"schedule {name}.alpha")
        rate, steps = item.get("learning_rate", o["learning_rate"]), item.get("max_steps", r["max_steps"])
        _number(rate, f"schedule {name}.learning_rate", positive=True)
        _integer(steps, f"schedule {name}.max_steps")
        specs.append(dict(name=name, alpha=float(alpha), learning_rate=float(rate), max_steps=steps))
    if len({item["name"].casefold() for item in specs}) != len(specs):
        raise ValueError("schedule names must be unique")
    return argparse.Namespace(
        **e, **r, **log, data_dir=Path(d["data_dir"]), dataset=d["kind"],
        per_class=d["per_class"], center=d["center"], download=d["download"],
        input_dim=d["input_dim"], data_seed=d["seed"], hidden_widths=tuple(m["hidden_widths"]),
        bias=m["bias"], initialization_kind=m["initialization"], initialization_std=m["initialization_std"],
        layer_scale=m["layer_scale"], beta1=o["beta1"], beta2=o["beta2"], epsilon=float(o["epsilon"]),
        specs=specs, sweep_seeds=axes["seeds"], sweep_beta2_values=axes["beta2_values"],
        sweep_epsilon_values=axes["epsilon_values"], epsilon_sweep="epsilon_values" in sweep,
        sweep_enabled=bool(sweep), dry_run=False,
    )


def default_args():
    args = from_mapping({})
    args.output_dir = Path(args.output_dir)
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Full-batch Adam on configurable MNIST ReLU networks")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--schedule")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true", help="list runs without loading data")
    cli = parser.parse_args(argv)
    with cli.config.open("rb") as stream:
        args = from_mapping(tomllib.load(stream))
    args.output_dir = cli.output_dir or Path(args.output_dir)
    for key in ("data_dir", "device"):
        if getattr(cli, key) is not None:
            setattr(args, key, getattr(cli, key))
    if cli.seed is not None:
        _integer(cli.seed, "--seed", 0)
        args.seed, args.sweep_seeds = cli.seed, [cli.seed]
    if cli.beta2 is not None:
        _number(cli.beta2, "--beta2")
        if not 0 < cli.beta2 < 1:
            parser.error("--beta2 must be in (0,1)")
        args.beta2, args.sweep_beta2_values = cli.beta2, [cli.beta2]
    if cli.epsilon is not None:
        _number(cli.epsilon, "--epsilon", minimum=0)
        args.epsilon, args.sweep_epsilon_values = cli.epsilon, [cli.epsilon]
    if cli.max_steps is not None:
        _integer(cli.max_steps, "--max-steps")
        args.max_steps = cli.max_steps
        args.specs = [{**spec, "max_steps": cli.max_steps} for spec in args.specs]
    if cli.schedule:
        args.specs = [spec for spec in args.specs if spec["name"] == cli.schedule]
        if not args.specs:
            parser.error(f"unknown schedule {cli.schedule}")
    args.dry_run = cli.dry_run
    return args
