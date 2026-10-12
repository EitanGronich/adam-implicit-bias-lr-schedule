"""Run discrete Adam and record only directly observed trajectory diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch

from adam_lr.optim.adam import InspectableAdam, schedule_learning_rate
from adam_lr.provenance import source_metadata
from .config import DirectionalConfig
from .model import angle_degrees, gradient_and_margins, initialize, normalize
from .numerics import numerical_issue


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _quantiles(value: np.ndarray) -> dict[str, float | None]:
    flat = value[np.isfinite(value)]
    names = ("min", "median", "p90", "max")
    if not len(flat):
        return dict.fromkeys(names)
    return dict(zip(names, map(float, np.quantile(flat, [0, .5, .9, 1]))))


def checkpoint_steps(config: DirectionalConfig) -> set[int]:
    steps = config.run.steps
    result = {0, steps, steps // 2}
    result.update(k for k in (1, 2, 3, 5, 10, 20, 50, 100) if k <= steps)
    if config.logging.every:
        result.update(range(config.logging.every, steps + 1, config.logging.every))
    if steps and config.logging.geometric_checkpoints:
        result.update(map(int, np.geomspace(1, steps, config.logging.geometric_checkpoints)))
    return result


def run(config: DirectionalConfig, *, output: str | Path | None = None,
        config_source: str | None = None) -> dict[str, Any]:
    """Execute independent trajectories in a batch; stop before an unsafe proposal.

    Every saved step is the number of ACCEPTED Adam updates. Rejected proposals
    restore theta, both moments, and the bias-correction clock. A single failing
    trajectory stops the whole batch so that all endpoints share a true horizon.
    """
    config.validate()
    provenance = source_metadata()
    torch.set_num_threads(config.run.threads)
    destination = Path(output or config.run.output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}; choose another --output")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    if config_source is not None:
        (destination / "config.toml").write_text(config_source, encoding="utf-8")
    initial = initialize(config)
    theta = torch.nn.Parameter(initial.theta.clone())
    optimizer = InspectableAdam([theta], config.optimizer.beta1, config.optimizer.beta2, config.optimizer.epsilon)
    first, second = optimizer.first[0], optimizer.second[0]
    checkpoints = checkpoint_steps(config)
    records: list[dict[str, np.ndarray]] = []
    aggregate: list[dict[str, Any]] = []
    previous_unit = normalize(theta.detach())
    zero_update_streak = torch.zeros(len(theta), dtype=torch.int64, device=theta.device)
    last_update = torch.zeros_like(theta)
    accepted = 0
    stop_reason = "step_limit"
    issue = None
    started = time.perf_counter()
    csv_path = destination / "checkpoints.csv"

    @torch.no_grad()
    def record(step: int) -> None:
        nonlocal previous_unit
        gradient, margin = gradient_and_margins(theta, initial.swapped, config.model)
        unit = normalize(theta)
        drift = angle_degrees(unit, previous_unit)
        previous_unit = unit.clone()
        log_loss = torch.logsumexp(-margin, dim=-1)
        # Norm computation is stable after scaling; norms themselves can still overflow,
        # which is a recorded diagnostic failure, not a silently normalized state.
        scale = theta.abs().amax(dim=-1)
        norm = scale * torch.linalg.vector_norm(theta / scale[:, None], dim=-1)
        record = {"step": np.array(step, dtype=np.int64), "theta": _numpy(theta),
                  "unit": _numpy(unit), "predictor_unit": _numpy(normalize(theta[:, :2])),
                  "margin": _numpy(margin), "loss": _numpy(torch.exp(log_loss)),
                  "log_loss": _numpy(log_loss), "gradient": _numpy(gradient),
                  "first_moment": _numpy(first), "second_moment": _numpy(second),
                  "last_update": _numpy(last_update), "theta_norm": _numpy(norm),
                  "checkpoint_drift_deg": _numpy(drift),
                  "initial_drift_deg": _numpy(angle_degrees(theta, initial.theta))}
        records.append(record)
        rate = schedule_learning_rate(config.schedule.initial_rate, config.schedule.power, max(1, step))
        next_rate = None
        if step < config.run.steps:
            try:
                next_rate = schedule_learning_rate(config.schedule.initial_rate, config.schedule.power, step + 1)
            except (ValueError, OverflowError):
                pass  # The actual proposal reports this failure; logging must not discard a valid state.
        row: dict[str, Any] = {"step": step, "rate_at_last_update": rate if step > 0 else None,
                               "rate_at_next_update": next_rate,
                               "zero_gradient_fraction": float((gradient == 0).float().mean().item()),
                               "zero_gradient_square_fraction": float((gradient.square() == 0).float().mean().item()),
                               "zero_second_moment_fraction": float((second == 0).float().mean().item()),
                               "subnormal_second_moment_fraction": float(((second > 0) & (second < torch.finfo(theta.dtype).tiny)).float().mean().item()),
                               "underflowed_loss_term_fraction": float((torch.exp(-margin) == 0).float().mean().item()),
                               "unchanged_trajectory_fraction": float((last_update == 0).all(dim=-1).float().mean().item())}
        quantities = {"loss": record["loss"], "log_loss": record["log_loss"],
                      "support_margin": record["margin"][:, 0], "transient_margin": record["margin"][:, 1],
                      "margin_gap": record["margin"][:, 1] - record["margin"][:, 0],
                      "theta_norm": record["theta_norm"], "checkpoint_drift_deg": record["checkpoint_drift_deg"],
                      "initial_drift_deg": record["initial_drift_deg"],
                      "abs_gradient": np.abs(record["gradient"]), "abs_first_moment": np.abs(record["first_moment"]),
                      "second_moment": record["second_moment"], "abs_last_update": np.abs(record["last_update"])}
        for name, values in quantities.items():
            row.update({f"{name}_{stat}": value for stat, value in _quantiles(values).items()})
        aggregate.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if len(aggregate) == 1:
                writer.writeheader()
            writer.writerow(row)

    with torch.no_grad():
        record(0)
        gradient, margin = gradient_and_margins(theta, initial.swapped, config.model)
        issue = numerical_issue(theta, gradient, margin, first, second, step=0,
                                beta2=config.optimizer.beta2, config=config.numerics)
        if issue:
            issue = {"proposed_step": 0, "at_initialization": True, **issue}
            stop_reason = "initialization_guard"
        else:
            for step in range(1, config.run.steps + 1):
                # Keep a full proposal checkpoint, including Adam's step counter.
                before, before_first, before_second = theta.clone(), first.clone(), second.clone()
                theta.grad = gradient
                try:
                    rate = schedule_learning_rate(config.schedule.initial_rate, config.schedule.power, step)
                    optimizer.step(rate)
                    next_gradient, next_margin = gradient_and_margins(theta, initial.swapped, config.model)
                    issue = numerical_issue(theta, next_gradient, next_margin, first, second, step=step,
                                            beta2=config.optimizer.beta2, config=config.numerics)
                except (FloatingPointError, OverflowError, RuntimeError, ValueError) as error:
                    issue = {"kind": "optimizer_or_arithmetic_failure", "detail": str(error)}
                if issue is not None:
                    theta.copy_(before)
                    first.copy_(before_first)
                    second.copy_(before_second)
                    optimizer.step_count = accepted
                    stop_reason = "support_margin_cap" if issue["kind"] == "support_margin_cap" else "numerical_guard"
                    issue = {"proposed_step": step, "at_initialization": False, **issue}
                    break
                accepted = step
                last_update.copy_(theta - before)
                gradient, margin = next_gradient, next_margin
                unchanged = (last_update == 0).all(dim=-1)
                zero_update_streak = torch.where(unchanged, zero_update_streak + 1, 0)
                if step in checkpoints:
                    record(step)
                if config.logging.progress_every and step % config.logging.progress_every == 0:
                    print(json.dumps({"step": step, "trajectories": len(theta),
                                      "seconds": round(time.perf_counter()-started, 2),
                                      "max_support_margin": float(margin[:, 0].max().item())}), flush=True)
                if config.numerics.stagnation_patience and (zero_update_streak >= config.numerics.stagnation_patience).any():
                    stop_reason = "parameter_stagnation_observed"
                    issue = {"kind": "rounded_zero_updates", "proposed_step": step,
                             "at_initialization": False,
                             "trajectories": int((zero_update_streak >= config.numerics.stagnation_patience).sum().item())}
                    break
        if int(records[-1]["step"]) != accepted:
            record(accepted)

    arrays = {key: np.stack([entry[key] for entry in records]) for key in records[0]}
    arrays.update(controls=_numpy(initial.controls), sector_swapped=_numpy(initial.swapped), beta=_numpy(initial.beta))
    np.savez_compressed(destination / "history.npz", **arrays)
    # Save the exact accepted state; no unaccepted/global checkpoint numbers.
    torch.save({"step": accepted, "theta": theta.detach().cpu(),
                "first_moment": first.cpu(), "second_moment": second.cpu(),
                "config": config.as_dict()}, destination / "final_state.pt")
    unit = records[-1]["unit"]
    half_target = accepted // 2
    reference = max(i for i, entry in enumerate(records) if int(entry["step"]) <= half_target)
    half_drift = _numpy(angle_degrees(theta, torch.tensor(records[reference]["theta"], device=theta.device, dtype=theta.dtype)))
    permutation_error = None
    if config.initialization.sectors == "both":
        half = len(theta) // 2
        permuted = torch.cat([theta[:half, [1, 0]], theta[:half, 2:]], dim=-1)
        permutation_error = float((theta[half:] - permuted).abs().max().item())
    summary = {"experiment": "discrete_adam_product_model", "requested_steps": config.run.steps,
               "accepted_steps": accepted, "stop_reason": stop_reason, "guard_event": issue,
               "trajectories": len(theta), "wall_seconds": time.perf_counter()-started,
               "dtype": config.run.dtype, "device": config.run.device,
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "python_version": platform.python_version(), "provenance": provenance,
               "endpoint_interpretation": "Observed finite-run directions; no limit is inferred or proved.",
               "half_reference_step": int(records[reference]["step"]),
               "half_to_endpoint_drift_deg": _quantiles(half_drift),
               "initial_to_endpoint_drift_deg": _quantiles(records[-1]["initial_drift_deg"]),
               "permutation_max_absolute_error": permutation_error,
               "final_support_margin": _quantiles(records[-1]["margin"][:, 0]),
               "final_log_loss": _quantiles(records[-1]["log_loss"]),
               "max_stagnation_streak": int(zero_update_streak.max().item()),
               "endpoint_squared_unit_coordinate_min": (unit**2).min(axis=0).tolist(),
               "endpoint_squared_unit_coordinate_max": (unit**2).max(axis=0).tolist(),
               "endpoint_squared_unit_covariance_eigenvalues": np.linalg.eigvalsh(
                   np.cov(unit**2, rowvar=False) if len(unit) > 1 else np.zeros((unit.shape[1], unit.shape[1]))
               ).tolist()}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    names = ["trajectory", "sector_swapped", "s"] + [f"ell_{j}" for j in range(1, config.model.layers)]
    coords = ["u1", "u2"] + [f"b{j}" for j in range(1, config.model.layers)]
    names += [f"theta_{c}" for c in coords] + [f"unit_{c}" for c in coords] + [f"q_{c}" for c in coords]
    with (destination / "endpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for j in range(len(theta)):
            writer.writerow([j, bool(arrays["sector_swapped"][j]), *arrays["controls"][j],
                             *records[-1]["theta"][j], *unit[j], *(unit[j] ** 2)])
    if config.logging.plots:
        from .plotting import plot_run
        plot_run(destination, max_trajectories=config.logging.plot_trajectories)
    return summary
