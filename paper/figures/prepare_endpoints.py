"""Compute endpoint-weighted margins from saved experiment states."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from adam_lr.mnist.data import balanced_mnist, synthetic_data
from adam_lr.mnist.models import ReLUNetwork


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        array = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def experiment_data(config, directory):
    """Recreate the selected dataset and verify it matches the experiment."""
    if config["dataset_kind"] == "synthetic":
        images, labels = synthetic_data(config["per_class"], config["architecture"][0],
                                        config.get("data_seed", 709))
    else:
        images, labels = balanced_mnist(directory, train=True,
                                        per_class=config["per_class"], download=False)
    if tensor_hash({"images": images, "labels01": labels}) != config["dataset_sha256"]:
        raise ValueError("The selected dataset differs from the training data")
    if config["centered"]:
        images = images - images.mean(0, keepdim=True)
    return images, labels * 2 - 1


def load_checkpoint(path):
    path = Path(path)
    before = path.stat()
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"Checkpoint changed while reading: {path}")
    return state


def checkpoint_parts(state):
    """Return parameters and committed moments after t applied updates."""
    model = state["model_state"]
    optimizer = state["optimizer_state"]
    second = dict(zip(model, optimizer["second"], strict=True))
    step = int(optimizer["step_count"])
    if "step" in state and state["step"] != step:
        raise ValueError("Parameter and optimizer update counts differ")
    for name, theta in model.items():
        moment = second[name]
        if (theta.shape != moment.shape or theta.dtype != torch.float64
                or moment.dtype != torch.float64 or not torch.isfinite(theta).all()
                or not torch.isfinite(moment).all() or (moment < 0).any()):
            raise ValueError(f"Invalid parameter or moment tensor: {name}")
    return model, second, step, state["signature"]


def model_for(run):
    config = run.config
    return ReLUNetwork(input_dim=config["architecture"][0],
                       hidden_widths=config["hidden_widths"],
                       bias=config["biases"]).double()


def evaluate(model, state, x, y, *, gradient=False):
    """Evaluate a saved state without changing checkpoint tensors."""
    parameters, _, _, _ = checkpoint_parts(state)
    if tuple(parameters) != tuple(model.state_dict()):
        raise ValueError("Checkpoint parameter names differ from the model")
    model.load_state_dict(parameters)
    model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(gradient):
        signed = y * model(x)
        loss = F.softplus(-signed).mean()
        raw = float(signed.detach().min())
        theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().copy()
    value = float(loss.detach())
    if not math.isfinite(value) or value <= 0 or not math.isfinite(raw):
        raise ValueError("Checkpoint loss or raw margin is not finite")
    if gradient:
        loss.backward()
    return theta, value, raw


def prospective_metric(model, state, x, y):
    """Return omega^{1/2} with unit l2 norm, including the terminal gradient."""
    _, second, step, config = checkpoint_parts(state)
    beta2 = float(config["beta2"])
    if not 0 < beta2 < 1:
        raise ValueError("The endpoint metric requires beta2 in (0, 1)")
    theta, loss, raw = evaluate(model, state, x, y, gradient=True)
    following = torch.cat([
        second[name].detach().clone().mul_(beta2)
        .addcmul_(parameter.grad, parameter.grad, value=1 - beta2).reshape(-1)
        for name, parameter in model.named_parameters()
    ]).numpy()
    if not np.isfinite(following).all() or (following < 0).any() or not (following > 0).any():
        raise ValueError("Invalid prospective endpoint second moment")
    # The common beta2**(-(step + 1)/2) factor cancels upon normalization.
    weights = np.sqrt(following)
    weights /= weights.max()
    weights /= np.linalg.norm(weights)
    return weights, step, theta, loss, raw


def signed_log_margins(theta, raw_margin, weights, degree=3.):
    """Compute signed margins in fixed normalized endpoint metrics."""
    quadratic = np.asarray(weights) @ np.square(theta)
    if not np.isfinite(quadratic).all() or (quadratic <= 0).any() or degree <= 0:
        raise ValueError("The endpoint metric has zero or nonfinite parameter norm")
    sign = float(np.sign(raw_margin))
    log_raw = -np.inf if raw_margin == 0 else math.log(abs(raw_margin))
    logarithms = (log_raw - degree / 2 * np.log(quadratic)) / math.log(10)
    if raw_margin != 0 and not np.isfinite(logarithms).all():
        raise ValueError("The signed weighted margin is not finite")
    return sign, logarithms


def display_series(logs_by_trajectory, signs_by_trajectory):
    """Keep a labeled power of ten separate from each reference's mantissas."""
    references = logs_by_trajectory[0].shape[0]
    exponents = np.zeros(references, dtype=np.int64)
    for reference in range(references):
        positive = [logs[reference, signs > 0] for logs, signs in
                    zip(logs_by_trajectory, signs_by_trajectory) if (signs > 0).any()]
        finite = positive or [logs[reference, signs != 0] for logs, signs in
                              zip(logs_by_trajectory, signs_by_trajectory) if (signs != 0).any()]
        if finite:
            maximum = max(float(values.max()) for values in finite)
            exponents[reference] = 0 if -3 <= maximum < 4 else math.floor(maximum)
    curves = []
    for logs, signs in zip(logs_by_trajectory, signs_by_trajectory):
        values = signs[None, :] * np.power(10., logs - exponents[:, None])
        if not np.isfinite(values).all():
            raise ValueError("Signed display margins are not representable")
        curves.append(values)
    return exponents, curves


def saved_states(run):
    states = {run.step: run.endpoint}
    identity = ("seed", "alpha", "layer_scale", "epsilon", "biases",
                "initialization_sha256", "dataset_sha256")
    for path in sorted((run.folder / "loss_checkpoints").glob("*.pt")):
        state = load_checkpoint(path)
        _, _, step, config = checkpoint_parts(state)
        if step > run.step:
            raise ValueError("A saved state lies beyond the terminal observation")
        if any(config.get(key) != run.config.get(key) for key in identity):
            raise ValueError("Saved states have different experiment settings")
        if step in states:
            old = checkpoint_parts(load_checkpoint(states[step]))[0]
            if tensor_hash(old) != tensor_hash(checkpoint_parts(state)[0]):
                raise ValueError("Saved parameter states at the same step disagree")
        else:
            states[step] = path
    return states


def check_observation(history, step, loss, raw):
    if not math.isclose(loss, float(history["loss"][step]), rel_tol=5e-11, abs_tol=0):
        raise ValueError("Recomputed checkpoint loss differs from the saved history")
    if "raw_margin" in history and not math.isclose(
            raw, float(history["raw_margin"][step]), rel_tol=5e-11, abs_tol=0):
        raise ValueError("Recomputed raw margin differs from the saved history")


def prepare_grid(sweep, x, y, *, epsilon):
    runs, alphas, seeds = (sweep[key] for key in ("runs", "alphas", "seeds"))
    arrays = dict(alpha=np.asarray(alphas), seed=np.asarray(seeds), epsilon=np.asarray(epsilon),
                  exponent=np.zeros((len(alphas), len(seeds)), dtype=np.int64),
                  reference_step=np.zeros((len(alphas), len(seeds)), dtype=np.int64))
    degree = float(sweep["config"].get("normalization_degree", len(sweep["config"]["hidden_widths"]) + 1))
    for schedule, alpha in enumerate(alphas):
        row = [runs[epsilon, alpha, seed] for seed in seeds]
        model = model_for(row[0])
        metrics = []
        for reference, run in enumerate(row):
            history = run.read_history()
            if int(history["step"][-1]) != run.step:
                raise ValueError("The checkpoint is not the terminal history state")
            state = load_checkpoint(run.endpoint)
            weights, step, _, loss, raw = prospective_metric(model, state, x, y)
            check_observation(history, step, loss, raw)
            arrays["reference_step"][schedule, reference] = step
            metrics.append(weights)
        weights = np.stack(metrics)
        logs_by_trajectory, signs_by_trajectory = [], []
        for trajectory, run in enumerate(row):
            history = run.read_history()
            clocks, logs, signs = [], [], []
            for step, path in sorted(saved_states(run).items()):
                state = load_checkpoint(path)
                theta, loss, raw = evaluate(model, state, x, y)
                check_observation(history, step, loss, raw)
                sign, logarithms = signed_log_margins(theta, raw, weights, degree)
                clocks.append(-math.log10(loss))
                logs.append(logarithms)
                signs.append(sign)
            arrays[f"s{schedule}_t{trajectory}_x"] = np.asarray(clocks)
            arrays[f"s{schedule}_t{trajectory}_mark"] = np.ones(len(clocks), dtype=bool)
            logs_by_trajectory.append(np.asarray(logs).T)
            signs_by_trajectory.append(np.asarray(signs))
        exponents, curves = display_series(logs_by_trajectory, signs_by_trajectory)
        arrays["exponent"][schedule] = exponents
        for trajectory, values in enumerate(curves):
            arrays[f"s{schedule}_t{trajectory}_y"] = values
    return arrays


def weighted_panels(grid):
    """Select the first reference seed at the fastest and slowest schedules."""
    alphas, seeds = grid["alpha"], grid["seed"]
    indices = (0, len(alphas) - 1)
    arrays = dict(alpha=alphas[list(indices)], seed=seeds,
                  reference_seed=np.asarray([seeds[0], seeds[0]]))
    for panel, schedule in enumerate(indices):
        arrays[f"a{panel}_exponent"] = grid["exponent"][schedule, 0]
        for trajectory in range(len(seeds)):
            arrays[f"a{panel}_s{trajectory}_x"] = grid[f"s{schedule}_t{trajectory}_x"]
            arrays[f"a{panel}_s{trajectory}_mantissa"] = grid[f"s{schedule}_t{trajectory}_y"][0]
    return arrays


def summarize_endpoints(sweep, data_dir):
    x, y = experiment_data(sweep["config"], data_dir)
    if sweep["kind"] == "long_horizon":
        return {"endpoints": prepare_grid(sweep, x, y, epsilon=0.)}
    sections = {}
    for epsilon in sweep["epsilons"]:
        label = "endpoints_epsilon0" if epsilon == 0 else "endpoints_epsilon1e8"
        if label in sections:
            raise ValueError("Prepare the epsilon comparison with one zero and one positive epsilon")
        sections[label] = prepare_grid(sweep, x, y, epsilon=epsilon)
    selected = "endpoints_epsilon1e8" if "endpoints_epsilon1e8" in sections else "endpoints_epsilon0"
    sections["weighted"] = weighted_panels(sections[selected])
    return sections
