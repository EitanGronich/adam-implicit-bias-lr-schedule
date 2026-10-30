"""Guarded Adam trajectories for the width-1 product model."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from adam_lr.optim.adam import InspectableAdam, schedule_learning_rate
from adam_lr.provenance import source_metadata
from .config import DirectionalConfig
from .model import gradient_and_margins, initialize, normalize
from .numerics import numerical_issue


def run(config: DirectionalConfig, *, output: str | Path | None = None,
        config_source: str | None = None) -> dict:
    """Stop before an unusable proposal, retaining the last accepted state."""
    config.validate()
    torch.set_num_threads(config.run.threads)
    destination = Path(output or config.run.output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    if config_source is not None:
        (destination / "config.toml").write_text(config_source, encoding="utf-8")
    initial = initialize(config)
    theta = torch.nn.Parameter(initial.theta.clone())
    optimizer = InspectableAdam([theta], config.optimizer.beta1, config.optimizer.beta2, config.optimizer.epsilon)
    first, second = optimizer.first[0], optimizer.second[0]
    records = []
    accepted, stop_reason, issue = 0, "step_limit", None
    zero_streak = torch.zeros(len(theta), dtype=torch.int64, device=theta.device)
    started = time.perf_counter()

    def array(value):
        return value.detach().cpu().numpy().copy()

    @torch.no_grad()
    def record(step):
        _, margin = gradient_and_margins(theta, initial.swapped, config.model)
        log_loss = torch.logsumexp(-margin, dim=-1)
        records.append({"step": np.asarray(step, dtype=np.int64), "theta": array(theta),
                        "unit": array(normalize(theta)), "margin": array(margin),
                        "loss": array(torch.exp(log_loss)), "log_loss": array(log_loss)})

    with torch.no_grad():
        record(0)
        gradient, margin = gradient_and_margins(theta, initial.swapped, config.model)
        issue = numerical_issue(theta, gradient, margin, first, second, step=0,
                                beta2=config.optimizer.beta2, config=config.numerics)
        if issue is not None:
            stop_reason = "initialization_guard"
            issue = {"proposed_step": 0, "at_initialization": True, **issue}
        else:
            for step in range(1, config.run.steps + 1):
                before, before_first, before_second = theta.clone(), first.clone(), second.clone()
                theta.grad = gradient
                try:
                    rate = schedule_learning_rate(config.schedule.initial_rate, config.schedule.alpha, step)
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
                gradient, margin = next_gradient, next_margin
                zero_streak = torch.where((theta == before).all(dim=-1), zero_streak + 1, 0)
                if step % config.logging.every == 0:
                    record(step)
                if config.logging.progress_every and step % config.logging.progress_every == 0:
                    print(json.dumps({"step": step, "trajectories": len(theta),
                                      "max_support_margin": float(margin[:, 0].max())}), flush=True)
                if config.numerics.stagnation_patience and (zero_streak >= config.numerics.stagnation_patience).any():
                    stop_reason = "parameter_stagnation_observed"
                    issue = {"kind": "rounded_zero_updates", "proposed_step": step,
                             "at_initialization": False}
                    break
        if int(records[-1]["step"]) != accepted:
            record(accepted)

    history = {key: np.stack([row[key] for row in records]) for key in records[0]}
    history.update(controls=array(initial.controls), pi=array(initial.pi),
                   sector_swapped=array(initial.swapped))
    np.savez_compressed(destination / "history.npz", **history)
    torch.save({"step": accepted, "theta": theta.detach().cpu(),
                "first_moment": first.detach().cpu(), "second_moment": second.detach().cpu(),
                "config": config.as_dict()}, destination / "final_state.pt")
    summary = {"requested_steps": config.run.steps, "accepted_steps": accepted,
               "stop_reason": stop_reason, "guard_event": issue,
               "trajectories": len(theta), "seconds": time.perf_counter() - started,
               "source": source_metadata()}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    coordinates = ["u1", "u2"] + [f"b{j}" for j in range(1, config.model.layers)]
    with (destination / "endpoints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trajectory", "sector_swapped"] + [f"unit_{name}" for name in coordinates])
        for j, unit in enumerate(history["unit"][-1]):
            writer.writerow([j, bool(history["sector_swapped"][j]), *unit])
    if config.logging.plots:
        from .plotting import plot_run
        plot_run(destination, max_trajectories=config.logging.plot_trajectories)
    return summary
