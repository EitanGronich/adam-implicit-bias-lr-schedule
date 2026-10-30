"""Full-batch Adam experiments with resumable checkpoints and paper data logs."""
from __future__ import annotations

import copy
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from adam_lr.optim.adam import InspectableAdam, NumericalError, logistic_loss_and_logits, schedule_learning_rate
from adam_lr.provenance import source_metadata
from .config import default_args, parse_args
from .data import balanced_mnist, synthetic_data
from .models import ReLUNetwork, make_scaled_initialization
from .traces import TraceWriter


STOP_REQUESTED = False
CHECKPOINT_FORMAT = "adam_mnist_checkpoint_v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def tensor_state_sha256(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)


def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        if state["cuda"] is None:
            raise RuntimeError("checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def validate_resume_signature(previous, current):
    if previous != current:
        changed = sorted(key for key in set(previous) | set(current) if previous.get(key) != current.get(key))
        raise RuntimeError(f"checkpoint signature mismatch: {', '.join(changed)}")


def _state(model, optimizer, signature, trace, reached_losses):
    return dict(format=CHECKPOINT_FORMAT, signature=signature, step=optimizer.step_count,
                model_state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                optimizer_state=dict(step_count=optimizer.step_count,
                    first=[value.detach().cpu().clone() for value in optimizer.first],
                    second=[value.detach().cpu().clone() for value in optimizer.second]),
                rng_state=_rng_state(optimizer.parameters[0].device), trace_state=trace.state(),
                reached_losses=sorted(reached_losses, reverse=True))


def _restore(model, optimizer, state):
    model.load_state_dict(state["model_state"])
    saved = state["optimizer_state"]
    if state["step"] != saved["step_count"] or type(state["step"]) is not int or state["step"] < 0:
        raise ValueError("checkpoint parameter and optimizer steps disagree")
    for name, targets in (("first", optimizer.first), ("second", optimizer.second)):
        values = saved[name]
        if len(values) != len(targets):
            raise ValueError("checkpoint has a different parameter count")
        for source, target in zip(values, targets):
            if source.shape != target.shape or source.dtype != target.dtype or not torch.isfinite(source).all():
                raise ValueError("checkpoint has invalid moment tensors")
            if name == "second" and (source < 0).any():
                raise ValueError("checkpoint second moments must be nonnegative")
            target.copy_(source.to(target.device))
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("checkpoint parameters are not finite")
    optimizer.step_count = state["step"]
    _restore_rng(state["rng_state"], optimizer.parameters[0].device)


def _normalized_margin(raw, norm, degree):
    if norm <= 0 or not math.isfinite(raw) or not math.isfinite(norm):
        return math.nan
    if raw == 0:
        return 0.
    exponent = math.log(abs(raw)) - degree * math.log(norm)
    if not math.log(np.nextafter(0., 1.)) <= exponent <= math.log(np.finfo(float).max):
        return math.nan
    return math.copysign(math.exp(exponent), raw)


@torch.no_grad()
def _observation(model, optimizer, raw_margin):
    parameters = torch.cat([value.detach().reshape(-1) for value in model.parameters()])
    maximum = parameters.abs().max()
    norm = maximum * (parameters / torch.where(maximum > 0, maximum, torch.ones_like(maximum))).square().sum().sqrt()
    linf, l2 = float(maximum), float(norm)
    step = optimizer.step_count + 1
    first = torch.cat([(m.clone().mul_(optimizer.beta1).add_(p.grad, alpha=1 - optimizer.beta1)
                        / (1 - optimizer.beta1**step)).reshape(-1)
                       for p, m in zip(model.parameters(), optimizer.first)])
    root = torch.cat([((v.clone().mul_(optimizer.beta2).addcmul_(p.grad, p.grad, value=1 - optimizer.beta2)
                        / (1 - optimizer.beta2**step)).sqrt()).reshape(-1)
                      for p, v in zip(model.parameters(), optimizer.second)])
    gradient = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    active = root > 0
    ordinary = torch.where(active, first / torch.where(active, root, torch.ones_like(root)), torch.zeros_like(first))
    denominator = root + optimizer.epsilon
    following = torch.where(denominator > 0, first / torch.where(denominator > 0, denominator,
                                                                torch.ones_like(denominator)), torch.zeros_like(first))
    valid = (torch.isfinite(ordinary).all() & torch.isfinite(following).all()
             & ~((root == 0) & ((first != 0) | (gradient != 0))).any())
    scale = ordinary.abs().max()
    valid = valid & (scale > 0)
    if bool(valid):
        reference = ordinary / scale
        error = float((reference - following / scale).square().sum() / reference.square().sum())
    else:
        error = math.nan
    degree = model.homogeneity_degree
    return (raw_margin, linf, l2, _normalized_margin(raw_margin, linf, degree),
            _normalized_margin(raw_margin, l2, degree), error, float(valid))


def run_one(*, args, spec, x, y):
    """Train one seed/schedule/epsilon combination without changing its data."""
    if (x.ndim < 2 or not len(x) or len(x) != y.numel() or x.device != y.device
            or x.dtype != y.dtype or x.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(x).all() or not torch.all((y == -1) | (y == 1))):
        raise ValueError("finite nonempty inputs and matching +/-1 labels are required")
    y = y.reshape(-1)
    if any(key in spec for key in ("power", "schedule_power")):
        raise ValueError("use alpha for eta_k=eta_0*k**alpha")
    architecture = (x.flatten(1).shape[1], *args.hidden_widths, 1)
    factory = lambda: ReLUNetwork(input_dim=architecture[0], hidden_widths=args.hidden_widths, bias=args.bias)
    initial, initialization = make_scaled_initialization(factory, x[:8], layer_scale=args.layer_scale,
        seed=args.seed, initialization_dtype=torch.float64, kind=args.initialization_kind, std=args.initialization_std)
    model = factory().to(device=x.device, dtype=x.dtype)
    converted = {name: value.to(device=x.device, dtype=x.dtype) for name, value in initial.items()}
    for name, value in converted.items():
        if not torch.isfinite(value).all() or ((initial[name].to(x.device) != 0) & (value == 0)).any():
            raise ValueError("initialization is not representable in the configured precision")
    model.load_state_dict(converted)
    optimizer = InspectableAdam(model.parameters(), args.beta1, args.beta2, args.epsilon)
    provenance = source_metadata()
    source_files = ("mnist/runner.py", "mnist/models.py", "mnist/data.py", "mnist/traces.py", "optim/adam.py")
    signature = dict(architecture=list(architecture), hidden_widths=list(args.hidden_widths), biases=args.bias,
        normalization_degree=float(len(architecture) - 1), exact_homogeneous=not args.bias,
        homogeneity_degree=None if args.bias else float(len(architecture) - 1), seed=int(args.seed),
        alpha=float(spec["alpha"]), learning_rate=float(spec["learning_rate"]), beta1=float(args.beta1),
        beta2=float(args.beta2), epsilon=float(args.epsilon), layer_scale=float(args.layer_scale),
        initialization_kind=args.initialization_kind, initialization_std=float(args.initialization_std),
        initialization_sha256=tensor_state_sha256(initial), dataset_sha256=str(args.dataset_sha256),
        dataset_kind=args.dataset, data_seed=int(args.data_seed), per_class=int(args.per_class),
        sample_count=len(x), centered=args.center, target_loss=float(args.target_loss),
        stop_at_target=args.stop_at_target, full_checkpoint_losses=list(args.full_checkpoint_losses),
        dtype=str(x.dtype).replace("torch.", ""), device_type=x.device.type,
        torch_version=str(torch.__version__), threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), source_sha256={name: provenance["module_sha256"][name] for name in source_files})
    max_steps = int(spec.get("max_steps", args.max_steps))
    if max_steps < 1:
        raise ValueError("the update budget must be positive")
    directory = Path(args.output_dir) / spec["name"]
    checkpoint = directory / "checkpoint.pt"
    saved = None
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError(f"run already exists: {directory}; enable resume or choose another output directory")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported checkpoint format")
        validate_resume_signature(saved["signature"], signature)
        _restore(model, optimizer, saved)
    elif directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"run directory contains data without a checkpoint: {directory}")
    if optimizer.step_count > max_steps:
        raise ValueError("the requested update budget is below the saved checkpoint step")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", signature)
    trace = TraceWriter(directory, signature, buffer_rows=args.trace_buffer_rows,
                        step=optimizer.step_count, saved_state=saved["trace_state"] if saved else None)
    reached = set(saved["reached_losses"] if saved else ())
    snapshot_dir = directory / "loss_checkpoints"
    if args.full_checkpoint_losses:
        snapshot_dir.mkdir(exist_ok=True)
        if saved is None:
            _save_torch(snapshot_dir / "initial.pt", _state(model, optimizer, signature, trace, reached))
    if saved is None:
        _save_torch(checkpoint, _state(model, optimizer, signature, trace, reached))
        trace.commit(0, "running")
    last_save = time.monotonic()
    failure = None
    while True:
        step = optimizer.step_count
        model.zero_grad(set_to_none=True)
        loss, logits = logistic_loss_and_logits(model, x, y)
        loss.backward()
        loss_value = float(loss.detach())
        raw_margin = float((y * logits).detach().min())
        finite = (math.isfinite(loss_value) and loss_value > 0 and math.isfinite(raw_margin)
                  and all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        values = _observation(model, optimizer, raw_margin) if finite else (raw_margin, *([math.nan] * 5), 0.)
        target = finite and loss_value <= args.target_loss
        reason = ("numerical_failure" if not finite else "target_loss" if target and args.stop_at_target
                  else "max_steps" if step >= max_steps else "signal" if STOP_REQUESTED else None)
        if finite and args.full_checkpoint_losses:
            crossed = {threshold for threshold in args.full_checkpoint_losses if loss_value <= threshold} - reached
            if crossed:
                reached.update(crossed)
                _save_torch(snapshot_dir / f"step_{step:09d}.pt", _state(model, optimizer, signature, trace, reached))
        if reason is None:
            try:
                rate = schedule_learning_rate(spec["learning_rate"], spec["alpha"], step + 1)
                optimizer.step(rate)
            except (NumericalError, ValueError) as error:
                reason, failure = "numerical_failure", str(error)
        trace.append(step, loss_value, reason is None, values)
        if reason is not None:
            state = _state(model, optimizer, signature, trace, reached)
            _save_torch(checkpoint, state)
            trace.commit(step, reason)
            result = dict(signature=signature, initialization=initialization, final_step=step,
                          final_loss=loss_value if math.isfinite(loss_value) else None,
                          completion_reason=reason, target_reached=bool(target), max_steps=max_steps, numerical_error=failure)
            write_json(directory / "results.json", result)
            return result
        if args.progress_every and optimizer.step_count % args.progress_every == 0:
            print(f"{spec['name']}: step {optimizer.step_count}, loss {loss_value:.6g}", flush=True)
        now = time.monotonic()
        if (args.checkpoint_every and optimizer.step_count % args.checkpoint_every == 0
                or args.checkpoint_every_seconds and now - last_save >= args.checkpoint_every_seconds):
            _save_torch(checkpoint, _state(model, optimizer, signature, trace, reached))
            trace.commit(optimizer.step_count, "running")
            last_save = now


def experiment_cases(args):
    for seed, beta2, epsilon, spec in product(args.sweep_seeds, args.sweep_beta2_values,
                                           args.sweep_epsilon_values, args.specs):
        selected = copy.copy(args)
        selected.seed, selected.beta2, selected.epsilon = seed, beta2, epsilon
        selected.max_steps = spec["max_steps"]
        if args.sweep_enabled:
            selected.output_dir = Path(args.output_dir) / f"seed_{seed}" / f"beta2_{beta2:g}"
        if args.epsilon_sweep or epsilon != 0:
            selected.output_dir = Path(selected.output_dir) / f"epsilon_{epsilon:g}"
        yield selected, spec, f"seed_{seed}/beta2_{beta2:g}/epsilon_{epsilon:g}/{spec['name']}"


def _experiment_record(args):
    return dict(format="adam_mnist_experiment_v1",
        sweep=dict(seeds=args.sweep_seeds, beta2_values=args.sweep_beta2_values,
                   epsilon_values=args.sweep_epsilon_values,
                   schedules=[{key: spec[key] for key in ("name", "alpha", "learning_rate")} for spec in args.specs]),
        dataset=dict(kind=args.dataset, per_class=args.per_class, input_dim=args.input_dim, seed=args.data_seed, center=args.center),
        model=dict(hidden_widths=list(args.hidden_widths), bias=args.bias, layer_scale=args.layer_scale,
                   initialization=args.initialization_kind, initialization_std=args.initialization_std),
        optimizer=dict(beta1=args.beta1))


def _merge_experiment(path, current):
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for name in ("model", "dataset", "optimizer"):
            if previous.get(name) != current[name]:
                raise ValueError("output directory belongs to a different experiment")
        for name in ("seeds", "beta2_values", "epsilon_values"):
            current["sweep"][name] = sorted(set(previous["sweep"][name]) | set(current["sweep"][name]))
        specs = {spec["name"]: spec for spec in previous["sweep"]["schedules"]}
        for spec in current["sweep"]["schedules"]:
            if spec["name"] in specs and spec != specs[spec["name"]]:
                raise ValueError("schedule name already has different parameters")
            specs[spec["name"]] = spec
        current["sweep"]["schedules"] = list(specs.values())
    write_json(path, current)


def main(argv=None):
    global STOP_REQUESTED
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps([dict(seed=selected.seed, beta2=selected.beta2, epsilon=selected.epsilon,
                               schedule=spec["name"], alpha=spec["alpha"], output=str(selected.output_dir / spec["name"]))
                          for selected, spec, _ in experiment_cases(args)], indent=2))
        return
    STOP_REQUESTED = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    if args.dataset == "synthetic":
        images, labels = synthetic_data(args.per_class, args.input_dim, args.data_seed)
    else:
        images, labels = balanced_mnist(args.data_dir, train=True, per_class=args.per_class, download=args.download)
    args.dataset_sha256 = tensor_state_sha256({"images": images, "labels01": labels})
    if args.center:
        images = images - images.mean(0, keepdim=True)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    x, y = images.to(device=device, dtype=dtype), labels.mul(2).sub(1).to(device=device, dtype=dtype)
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _merge_experiment(args.output_dir / "experiment.json", _experiment_record(args))
    handlers = {}
    def request_stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, request_stop)
        results = {}
        for selected, spec, key in experiment_cases(args):
            results[key] = run_one(args=selected, spec=spec, x=x, y=y)
            if STOP_REQUESTED or results[key]["completion_reason"] == "numerical_failure":
                break
        write_json(args.output_dir / "experiment_summary.json", dict(results=results))
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    if any(result["completion_reason"] == "numerical_failure" for result in results.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
