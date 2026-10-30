"""Aggregate statistics from completed experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adam_lr.mnist.traces import load_trace


class Run:
    def __init__(self, folder, config, result, history=None):
        self.folder = Path(folder)
        self.config = config
        self.result = result
        self.history = history

    @property
    def key(self):
        return (float(self.config["epsilon"]), float(self.config["alpha"]),
                int(self.config["seed"]))

    @property
    def step(self):
        return int(self.result["final_step"])

    @property
    def endpoint(self):
        return self.folder / "checkpoint.pt"

    def read_history(self):
        return self.history if self.history is not None else load_history(self.folder)[1]


def moving_average(values, valid, window=1000):
    """Require a full trailing window of finite observations."""
    values, valid = np.asarray(values, dtype=float), np.asarray(valid, dtype=bool)
    if values.ndim != 1 or values.shape != valid.shape or window < 1:
        raise ValueError("Moving averages require matching vectors and a positive window")
    valid = valid & np.isfinite(values)
    sums = np.r_[np.longdouble(0), np.cumsum(np.where(valid, values, 0.), dtype=np.longdouble)]
    bad = np.r_[0, np.cumsum(~valid)]
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        average = np.asarray((sums[window:] - sums[:-window]) / window, dtype=float)
        average[(bad[window:] - bad[:-window]) != 0] = np.nan
        result[window - 1:] = average
    return result


def interpolate_first_passage(x, y, grid):
    """After a loss reversal, use the actual preceding chronological sample."""
    x, y, grid = np.asarray(x), np.asarray(y), np.asarray(grid)
    if (x.ndim != 1 or x.shape != y.shape or not len(x) or not len(grid)
            or not np.isfinite(x).all() or np.isinf(y).any()):
        raise ValueError("Invalid first-passage observations")
    records = np.flatnonzero(np.r_[True, x[1:] > np.maximum.accumulate(x[:-1])])
    if grid[0] < x[0] or grid[-1] > x[records[-1]]:
        raise ValueError("First-passage interpolation would extrapolate")
    right = records[np.searchsorted(x[records], grid, side="left")]
    left = np.maximum(right - 1, 0)
    result = y[right].copy()
    between = grid != x[right]
    fraction = ((grid[between] - x[left[between]]) /
                (x[right[between]] - x[left[between]]))
    if np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("First-passage interpolation would extrapolate")
    result[between] = y[left[between]] + fraction * (y[right[between]] - y[left[between]])
    return result


def common_grid_summary(curves, spacing=.001, lower=None):
    """Use every seed on shared first-passage support and report sample SEM."""
    if len(curves) < 2:
        raise ValueError("A sample standard error requires at least two seeds")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Loss-clock spacing must be positive")
    domains = []
    for x, y in curves:
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 1 or not len(x) or x.shape != y.shape or not np.isfinite(x).all():
            raise ValueError("Invalid first-passage observations")
        records = np.flatnonzero(np.r_[True, x[1:] > np.maximum.accumulate(x[:-1])])
        eligible = records[np.isfinite(y[records])]
        if len(eligible) < 2:
            raise ValueError("A seed has no complete first-passage support")
        domains.append((x[eligible[0]], x[eligible[-1]]))
    low, high = max(a for a, _ in domains), min(b for _, b in domains)
    if lower is not None:
        low = max(low, lower)
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("The seeds have no common observed support")
    interior = np.arange(np.ceil(low / spacing), np.floor(high / spacing) + 1) * spacing
    grid = np.unique(np.r_[low, interior[(interior > low) & (interior < high)], high])
    matrix = np.stack([interpolate_first_passage(x, y, grid) for x, y in curves])
    if not np.isfinite(matrix).all():
        raise ValueError("An incomplete interpolation window lies inside the common support")
    return grid, matrix.mean(axis=0), matrix.std(axis=0, ddof=1) / np.sqrt(len(curves))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_metadata(folder):
    result = read_json(Path(folder) / "results.json")
    config = dict(result["signature"])
    if result["completion_reason"] not in ("target_loss", "max_steps"):
        raise ValueError(f"Experiment did not finish normally: {folder}")
    if int(result["final_step"]) < 1 or not np.isfinite(result["final_loss"]) or result["final_loss"] <= 0:
        raise ValueError(f"Invalid terminal observation: {folder}")
    return config, result


def load_history(folder):
    """Read the runner's integrity-checked per-update log."""
    folder = Path(folder)
    config, result = run_metadata(folder)
    saved = load_trace(folder, signature=result["signature"],
                       final_step=result["final_step"],
                       completion_reason=result["completion_reason"])
    trace = {key: saved[key] for key in ("step", "loss", "update_applied")}
    trace["loss_clock"] = -np.log10(trace["loss"])
    for index, name in enumerate(saved["metric_columns"]):
        trace[name] = saved["values"][:, index]
    if (not np.array_equal(trace["step"], np.arange(result["final_step"] + 1))
            or not np.isfinite(trace["loss"]).all() or np.any(trace["loss"] <= 0)
            or trace["loss"][-1] != result["final_loss"]
            or trace["update_applied"][-1] != 0
            or not np.all(trace["update_applied"][:-1] == 1)):
        raise ValueError(f"History and terminal observation differ: {folder}")
    trace["linf_margin"] = trace["margin_linf"]
    return config, trace


def load_sweep(root, *, beta2=None):
    """Discover one rectangular seed/schedule/epsilon sweep."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    runs, first, initializations = {}, None, {}
    invariants = ("architecture", "sample_count", "dataset_kind", "centered",
                  "dataset_sha256", "hidden_widths", "layer_scale", "biases",
                  "beta1", "beta2", "dtype", "target_loss", "learning_rate")
    for result_path in sorted(root.rglob("results.json")):
        folder = result_path.parent
        config, result = run_metadata(folder)
        if beta2 is not None and float(config["beta2"]) != beta2:
            continue
        if first is None:
            first = config
        elif any(config.get(name) != first.get(name) for name in invariants):
            raise ValueError("Runs have incompatible experiment settings; select one beta2 value if needed")
        run = Run(folder, config, result)
        if run.key in runs:
            raise ValueError(f"Duplicate trajectory: {run.key}")
        seed = int(config["seed"])
        digest = config["initialization_sha256"]
        if seed in initializations and initializations[seed] != digest:
            raise ValueError("Initial parameters differ across schedules or epsilon values")
        initializations[seed] = digest
        runs[run.key] = run
    if not runs:
        raise ValueError(f"No completed experiments found in {root}")
    seeds = tuple(sorted({key[2] for key in runs}))
    alphas = tuple(sorted({key[1] for key in runs}, reverse=True))
    epsilons = tuple(sorted({key[0] for key in runs}))
    if (root / "experiment.json").is_file():
        planned = read_json(root / "experiment.json")["sweep"]
        seeds = tuple(sorted(planned["seeds"]))
        alphas = tuple(sorted({float(item["alpha"]) for item in planned["schedules"]}, reverse=True))
        epsilons = tuple(sorted(planned["epsilon_values"]))
    expected = {(epsilon, alpha, seed) for epsilon in epsilons for alpha in alphas for seed in seeds}
    if set(runs) != expected:
        raise ValueError(f"Missing or unexpected seed/schedule/epsilon trajectories: "
                         f"{len(expected - set(runs))} missing, {len(set(runs) - expected)} unexpected")
    if len(seeds) < 2:
        raise ValueError("A sample standard error requires at least two seeds")
    kind = "long_horizon" if epsilons == (0.,) and not first["biases"] else "comparison"
    return dict(runs=runs, seeds=seeds, alphas=alphas, epsilons=epsilons,
                config=first, kind=kind)


def metadata_section(sweep, secant_window, error_window):
    config = sweep["config"]
    return dict(seed=np.asarray(sweep["seeds"]), alpha=np.asarray(sweep["alphas"]),
                epsilon=np.asarray(sweep["epsilons"]), biases=np.asarray(config["biases"]),
                hidden_widths=np.asarray(config["hidden_widths"]),
                layer_scale=np.asarray(config["layer_scale"]), beta1=np.asarray(config["beta1"]),
                beta2=np.asarray(config["beta2"]), learning_rate=np.asarray(config["learning_rate"]),
                target_loss=np.asarray(config["target_loss"]),
                normalization_degree=np.asarray(config.get("normalization_degree", len(config["hidden_widths"]) + 1)),
                secant_window=np.asarray(secant_window), error_window=np.asarray(error_window))


def summarize_traces(sweep, *, secant_window=10000, error_window=1000, selected_alpha=None):
    """Compute plot summaries without changing the experiment sample."""
    runs, seeds, alphas, epsilons = (sweep[key] for key in ("runs", "seeds", "alphas", "epsilons"))
    sections = {"settings": metadata_section(sweep, secant_window, error_window)}
    if sweep["kind"] == "long_horizon":
        if secant_window < 1:
            raise ValueError("The secant window must be positive")
        secants = dict(x=np.arange(1, secant_window + 1), alphas=np.asarray(alphas),
                       seeds=np.asarray(seeds), window=np.asarray(secant_window))
        linf = dict(alpha=np.asarray(alphas), seed=np.asarray(seeds))
        for index, alpha in enumerate(alphas):
            histories = [runs[0., alpha, seed].read_history() for seed in seeds]
            slopes = []
            for trace in histories:
                loss = trace["loss"]
                if len(loss) <= secant_window:
                    raise ValueError("The run is shorter than --secant-window")
                slopes.append((np.log(loss[-secant_window - 1]) -
                               np.log(loss[-secant_window:])) / secants["x"])
            rows = np.stack(slopes)
            secants[f"s{index}_mean"] = rows.mean(axis=0)
            secants[f"s{index}_se"] = rows.std(axis=0, ddof=1) / np.sqrt(len(seeds))
            x, mean, se = common_grid_summary([
                (trace["loss_clock"], trace["linf_margin"])
                for trace in histories])
            linf.update({f"s{index}_x": x, f"s{index}_mean": mean, f"s{index}_se": se})
        if selected_alpha is None:
            selected_alpha = -.3 if -.3 in alphas else alphas[len(alphas) // 2]
        if selected_alpha not in alphas:
            raise ValueError("The selected alpha is absent from the experiment")
        appendix = dict(seed=np.asarray(seeds), alpha=np.asarray(selected_alpha))
        for index, seed in enumerate(seeds):
            trace = runs[0., selected_alpha, seed].read_history()
            x, y = trace["loss_clock"], trace["linf_margin"]
            records = np.flatnonzero(np.r_[True, x[1:] > np.maximum.accumulate(x[:-1])])
            x, y = x[records], y[records]
            positions = np.unique(np.linspace(0, len(x) - 1, min(9, len(x))).astype(int))
            appendix.update({f"x{index}": x, f"y{index}": y,
                             f"marker_x{index}": x[positions], f"marker_y{index}": y[positions]})
        sections.update(secants=secants, linf=linf, alpha_m03=appendix)
        return sections
    summaries = {metric: dict(epsilons=np.asarray(epsilons), alpha=np.asarray(alphas),
                              count=np.asarray(len(seeds)), metric=np.asarray(metric))
                 for metric in ("linf", "l2")}
    selected_epsilon = max(epsilons)
    errors = dict(epsilon=np.asarray(selected_epsilon), window=np.asarray(error_window),
                  initialization_scale=np.asarray(sweep["config"]["layer_scale"]),
                  bias=np.asarray(sweep["config"]["biases"]), seed=np.asarray(seeds))
    for e, epsilon in enumerate(epsilons):
        for i, alpha in enumerate(alphas):
            histories = [runs[epsilon, alpha, seed].read_history() for seed in seeds]
            for metric, field in (("linf", "linf_margin"), ("l2", "margin_l2")):
                x, mean, se = common_grid_summary([
                    (trace["loss_clock"], trace[field]) for trace in histories])
                summaries[metric].update({f"e{e}_a{i}_x": x, f"e{e}_a{i}_mean": mean,
                                           f"e{e}_a{i}_se": se})
            if epsilon == selected_epsilon:
                averaged = []
                for trace in histories:
                    squared = trace["epsilon_update_change_squared"]
                    valid = (trace["epsilon_update_change_squared_valid"] == 1) & (squared >= 0)
                    average = moving_average(np.sqrt(np.where(valid, squared, 0.)), valid, error_window)
                    averaged.append((trace["loss_clock"], average))
                x, mean, se = common_grid_summary(averaged)
                errors.update({f"g{i}_relative_x": x, f"g{i}_relative_mean": mean,
                               f"g{i}_relative_se": se, f"g{i}_alpha": np.asarray(alpha),
                               f"g{i}_n": np.asarray(len(seeds))})
    sections.update(summaries, error=errors)
    return sections
