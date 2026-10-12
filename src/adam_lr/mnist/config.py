"""Validated TOML configuration for the MNIST experiment.

Relative data/output paths resolve against the working directory. The chosen
configuration is supplied explicitly; no cluster paths or environment-specific
locations are embedded in the package.
"""
from __future__ import annotations
import argparse
import math
import re
import tomllib
from pathlib import Path

DEFAULTS = {
    "experiment": {"seed": 20260828, "output_dir": "runs/mnist", "dtype": "float64", "device": "auto", "threads": 4},
    "dataset": {"kind": "mnist", "data_dir": "data", "per_class": 512, "center": True, "download": False, "input_dim": 784, "seed": 709},
    "model": {"hidden_widths": [1024, 1024], "initialization": "torch_default", "initialization_std": 1.0, "layer_scale": 0.01},
    "optimizer": {"learning_rate": 0.0003, "beta1": 0.9, "beta2": 0.999, "beta_policy": "fixed", "beta_reference_rate": 0.005},
    "run": {"max_steps": 200000, "target_loss": 1e-20, "stop_at_target": True, "resume": True, "allow_target_extension": False, "allow_dtype_promotion": False, "finalize_checkpoint": False},
    "logging": {"record_every": 100, "coordinate_every": 1000, "expensive_every": 5000, "coordinate_count": 100, "selection_loss": 0.1, "spectral_norm_steps": 20, "nuclear_norm_steps": 12, "checkpoint_every": 10000, "checkpoint_every_seconds": 600.0, "progress_every": 2500, "sketch_dimension": 256, "log_second_moment_shadow": False, "log_incoming_every_step": False, "full_checkpoint_losses": []},
}

def _finite(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")

def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")

def from_mapping(raw: dict) -> argparse.Namespace:
    unknown = set(raw) - set(DEFAULTS) - {"schedules", "sweep"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    sections = {}
    for name, defaults in DEFAULTS.items():
        supplied = raw.get(name, {})
        if not isinstance(supplied, dict) or set(supplied) - set(defaults):
            raise ValueError(f"unknown or invalid keys in [{name}]")
        sections[name] = {**defaults, **supplied}
    e,d,m,o,r,l = (sections[k] for k in DEFAULTS)
    for section_name, section in sections.items():
        for key, value in section.items():
            default = DEFAULTS[section_name][key]
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"{section_name}.{key} must be true or false")
    if e["dtype"] not in ("float32", "float64") or e["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("dtype must be float32/float64 and device auto/cpu/cuda")
    for key in ("seed", "threads"):
        _integer(e[key], f"experiment.{key}", 0 if key == "seed" else 1)
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
    _finite(m["layer_scale"], "model.layer_scale", positive=True)
    _finite(m["initialization_std"], "model.initialization_std", positive=True)
    for key in ("learning_rate", "beta_reference_rate"):
        _finite(o[key], f"optimizer.{key}", positive=True)
    if o["beta_policy"] not in ("fixed", "rescaled"):
        raise ValueError("optimizer.beta_policy must be fixed or rescaled")
    for key in ("beta1", "beta2"):
        _finite(o[key], f"optimizer.{key}")
        if not 0 <= o[key] < 1 or key == "beta2" and o[key] == 0:
            raise ValueError("beta1 must lie in [0,1); beta2 in (0,1) for transformed-moment diagnostics")
    _integer(r["max_steps"], "run.max_steps")
    _finite(r["target_loss"], "run.target_loss", positive=True)
    for key in ("record_every", "coordinate_every", "expensive_every", "coordinate_count", "spectral_norm_steps", "nuclear_norm_steps", "sketch_dimension", "checkpoint_every", "progress_every"):
        _integer(l[key], f"logging.{key}", 0 if key in ("checkpoint_every", "progress_every") else 1)
    _finite(l["selection_loss"], "logging.selection_loss", positive=True)
    _finite(l["checkpoint_every_seconds"], "logging.checkpoint_every_seconds", nonnegative=True)
    if not isinstance(l["full_checkpoint_losses"], list):
        raise ValueError("logging.full_checkpoint_losses must be an array")
    for loss in l["full_checkpoint_losses"]:
        _finite(loss, "logging.full_checkpoint_losses entry", positive=True)
        if r["stop_at_target"] and loss < r["target_loss"]:
            raise ValueError("loss checkpoints below target cannot be reached with stop_at_target=true")
    if len(set(l["full_checkpoint_losses"])) != len(l["full_checkpoint_losses"]):
        raise ValueError("full checkpoint losses must be distinct")
    if l["log_incoming_every_step"] and not l["log_second_moment_shadow"]:
        raise ValueError("incoming trace requires log_second_moment_shadow=true")
    if l["log_second_moment_shadow"] and e["dtype"] != "float64":
        raise ValueError("log shadow currently requires float64")
    sweep = raw.get("sweep", {})
    if not isinstance(sweep, dict) or set(sweep) - {"seeds", "beta2_values"}:
        raise ValueError("[sweep] supports only seeds and beta2_values")
    seeds, betas = sweep.get("seeds", [e["seed"]]), sweep.get("beta2_values", [o["beta2"]])
    if not isinstance(seeds, list) or not seeds or not isinstance(betas, list) or not betas:
        raise ValueError("sweep seeds and beta2_values must be nonempty arrays")
    for seed in seeds:
        _integer(seed, "sweep.seeds entry", 0)
    for beta in betas:
        _finite(beta, "sweep.beta2_values entry")
        if not 0 < beta < 1:
            raise ValueError("sweep beta2 values must lie in (0,1)")
    if len(set(seeds)) != len(seeds) or len(set(betas)) != len(betas):
        raise ValueError("sweep entries must be unique")
    schedules = raw.get("schedules", [{"name": "p025", "power": 0.25}])
    if not isinstance(schedules, list) or not schedules:
        raise ValueError("at least one [[schedules]] table is required")
    specs = []
    for s in schedules:
        if not isinstance(s, dict) or set(s) - {"name", "power", "learning_rate", "max_steps"}:
            raise ValueError("invalid schedule keys")
        name, power = s.get("name"), s.get("power")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("schedule name must be a simple directory name")
        _finite(power, f"schedule {name}.power")
        rate, steps = s.get("learning_rate", o["learning_rate"]), s.get("max_steps", r["max_steps"])
        _finite(rate, f"schedule {name}.learning_rate", positive=True)
        _integer(steps, f"schedule {name}.max_steps")
        specs.append({"name": name, "activation_power": 1.0, "schedule_power": float(power), "learning_rate": float(rate), "max_steps": steps})
    if len({s["name"].casefold() for s in specs}) != len(specs):
        raise ValueError("schedule names must be unique")
    return argparse.Namespace(
        **e, **r, **{k:v for k,v in l.items() if k != "full_checkpoint_losses"},
        data_dir=Path(d["data_dir"]), dataset=d["kind"], per_class=d["per_class"], center=d["center"], download=d["download"], input_dim=d["input_dim"], data_seed=d["seed"],
        hidden_widths=tuple(m["hidden_widths"]), initialization_kind=m["initialization"], initialization_std=m["initialization_std"], layer_scale=m["layer_scale"],
        beta_policy=o["beta_policy"], beta_reference_rate=o["beta_reference_rate"], beta1_reference=o["beta1"], beta2_reference=o["beta2"],
        specs=specs, sweep_seeds=seeds, sweep_beta2_values=betas, sweep_enabled=bool(sweep), full_checkpoint_losses=",".join(str(v) for v in l["full_checkpoint_losses"]), allow_legacy_diagnostic_upgrade=False,
    )

def default_args():
    args = from_mapping({})
    args.output_dir = Path(args.output_dir)
    return args

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Full-batch epsilon-zero Adam on configurable MNIST ReLU networks")
    parser.add_argument("--config", type=Path, required=True, help="TOML configuration")
    parser.add_argument("--output-dir", type=Path, help="override experiment.output_dir")
    parser.add_argument("--data-dir", type=Path, help="override dataset.data_dir")
    parser.add_argument("--seed", type=int, help="select one initialization seed")
    parser.add_argument("--beta2", type=float, help="select one beta2 value")
    parser.add_argument("--schedule", help="run only one named schedule")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-steps", type=int, help="override every schedule's horizon")
    cli = parser.parse_args(argv)
    with cli.config.open("rb") as stream:
        raw = tomllib.load(stream)
    args = from_mapping(raw)
    args.output_dir = cli.output_dir or Path(args.output_dir)
    if cli.data_dir is not None:
        args.data_dir = cli.data_dir
    if cli.device is not None:
        args.device = cli.device
    if cli.max_steps is not None:
        _integer(cli.max_steps, "--max-steps")
        args.max_steps = cli.max_steps
        args.specs = [{**s, "max_steps": cli.max_steps} for s in args.specs]
    if cli.seed is not None:
        _integer(cli.seed, "--seed", 0)
        args.seed = cli.seed
        args.sweep_seeds = [cli.seed]
    if cli.beta2 is not None:
        if not 0 < cli.beta2 < 1:
            parser.error("--beta2 must lie in (0,1)")
        args.beta2_reference = cli.beta2
        args.sweep_beta2_values = [cli.beta2]
    if cli.schedule:
        args.specs = [s for s in args.specs if s["name"] == cli.schedule]
        if not args.specs:
            parser.error(f"unknown schedule {cli.schedule}")
    return args
