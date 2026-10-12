"""Full-batch Adam on a configurable bias-free ReLU network.

Row t describes theta_t and g_t. Diagnostic moments include g_t prospectively;
rolling checkpoints store moments after t updates. See docs/mnist.md.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adam_lr.optim.adam import (
    InspectableAdam,
    NumericalError,
    logistic_loss_and_logits,
    schedule_learning_rate,
)
from .log_second_moment import LogSecondMomentShadow
from .incoming_trace import BufferedIncomingTrace
from .data import balanced_mnist, synthetic_data
from .models import HomogeneousReLU, HomogeneousThreeLayerReLU, make_scaled_initialization
from .matrix_norms import nuclear_norm_iterative
from .coordinate_diagnostics import (
    second_moment_tracking_ratios,
    stable_normalized_rho_from_updated_ratio,
    theory_prediction,
)


from adam_lr.provenance import source_metadata
from .config import parse_args, default_args

SCHEDULES = {
    "p000": 0.0,
    "p010": 0.10,
    "p025": 0.25,
    "p050": 0.50,
    "p075": 0.75,
}
ANCHORS = (1e-4, 1e-8, 1e-12)
LAYER_NAMES = ("hidden1", "hidden2", "output")
ARCHITECTURE = (28 * 28, 1024, 1024, 1)
LOSS_THRESHOLDS = (
    1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12,
    1e-14, 1e-16, 1e-18, 1e-20,
)
# Version 5 preserves raw-v anchors, makes moment_step explicit, aligns all
# g2/v ratios to raw v, and includes numerical-resolution indicators.
DIAGNOSTIC_VERSION = 5
STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    """Ask the training loop to checkpoint at its next evaluated boundary."""

    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; checkpointing after this evaluation", flush=True)


def install_signal_handlers() -> tuple[str, ...]:
    installed = []
    for name in ("SIGTERM", "SIGINT", "SIGUSR1", "SIGUSR2"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        signal.signal(signum, request_stop)
        installed.append(name)
    return tuple(installed)


def parse_full_checkpoint_losses(raw: str) -> tuple[float, ...]:
    """Parse distinct positive thresholds in descending crossing order."""

    if not raw.strip():
        return ()
    try:
        values = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("invalid --full_checkpoint_losses value") from error
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("full checkpoint losses must be finite and positive")
    if len(set(values)) != len(values):
        raise ValueError("full checkpoint losses must be distinct")
    return tuple(sorted(values, reverse=True))


def loss_checkpoint_filename(index: int, threshold: float) -> str:
    exponent = -math.log10(threshold)
    exponent_tag = f"{exponent:.6f}".replace(".", "p")
    return f"loss_checkpoint_{index + 1:02d}_log10_{exponent_tag}.pt"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor together with its dtype and shape."""

    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(state[name]).encode("ascii"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(root: Path) -> dict[str, object]:
    def query(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = query("rev-parse", "HEAD")
    status = query("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "tracked_worktree_dirty": None if status is None else bool(status),
    }


def hardware_metadata(device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update({
            "cuda_device_name": properties.name,
            "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_total_memory_bytes": int(properties.total_memory),
            "cudnn_version": torch.backends.cudnn.version(),
        })
    return result


def peak_host_rss_bytes() -> int | None:
    """Return the process high-water RSS when the host exposes getrusage."""

    try:
        import resource
    except ImportError:
        return None
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; Darwin and the BSD-derived macOS API report bytes.
    return maximum if platform.system() == "Darwin" else maximum * 1024


def vector_norms(vector: torch.Tensor) -> dict[str, float | int]:
    detached = vector.detach()
    maximum = detached.abs().max() if detached.numel() else detached.new_tensor(0.0)
    # Squaring raw moments can underflow even when each moment is normal.
    # Scaling first preserves every representable norm of a finite vector.
    l2 = maximum * (detached / maximum).norm() if maximum > 0 and torch.isfinite(maximum) else maximum
    return {
        "l1": float(detached.abs().sum().item()),
        "l2": float(l2.item()),
        "linf": float(maximum.item()),
        "zero_count": int((detached == 0).sum().item()),
    }


def distribution_summary(values: torch.Tensor) -> dict[str, float | int]:
    finite = values.detach().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {
            "count": 0, "mean": math.nan, "std": math.nan,
            "minimum": math.nan, "q33": math.nan, "median": math.nan,
            "q66": math.nan, "maximum": math.nan,
        }
    quantiles = torch.quantile(
        finite,
        torch.tensor([1 / 3, 0.5, 2 / 3], dtype=finite.dtype, device=finite.device),
    )
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "minimum": float(finite.min().item()),
        "q33": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "q66": float(quantiles[2].item()),
        "maximum": float(finite.max().item()),
    }


def prefixed_values(prefix: str, values: dict[str, float | int]) -> dict:
    return {f"{prefix}_{name}": value for name, value in values.items()}


@torch.no_grad()
def deterministic_spectral_norm(
    matrix: torch.Tensor, *, steps: int = 20, epsilon: float = 1e-30,
) -> float:
    """Scale-invariant power-iteration estimate, exact for row/column vectors."""

    if matrix.ndim != 2:
        raise ValueError("spectral norm requires a matrix")
    if not matrix.numel() or steps <= 0:
        raise ValueError("spectral norm requires a nonempty matrix and positive steps")
    scale = matrix.abs().max()
    if not torch.isfinite(scale):
        return math.nan
    if scale == 0:
        return 0.0
    # The regularizer belongs to the dimensionless power iteration, not to
    # the raw matrix. Otherwise a small gradient/action can be reported as 0.
    normalized = matrix / scale
    if min(matrix.shape) == 1:
        return float((scale * normalized.norm()).item())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(104729 * matrix.shape[0] + 130363 * matrix.shape[1])
    right = torch.randn(
        matrix.shape[1], dtype=matrix.dtype, generator=generator,
    ).to(matrix.device)
    right /= right.norm().clamp_min(epsilon)
    for _ in range(int(steps)):
        left = normalized @ right
        left /= left.norm().clamp_min(epsilon)
        right = normalized.T @ left
        right /= right.norm().clamp_min(epsilon)
    return float((scale * torch.dot(left, normalized @ right).abs()).item())


def affine_matrix(
    weight: torch.Tensor, bias: torch.Tensor | None,
) -> torch.Tensor:
    if bias is None:
        return weight
    return torch.cat((weight, bias.reshape(-1, 1)), dim=1)


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if not left.numel() or not right.numel():
        return math.nan
    left_scale, right_scale = left.abs().max(), right.abs().max()
    if not (left_scale > 0 and right_scale > 0 and torch.isfinite(left_scale) and torch.isfinite(right_scale)):
        return math.nan
    left_unit = left / left_scale
    right_unit = right / right_scale
    left_unit = left_unit / left_unit.norm()
    right_unit = right_unit / right_unit.norm()
    return float(torch.dot(left_unit, right_unit).clamp(-1, 1).item())


def split_flat_like(
    vector: torch.Tensor, references: list[torch.Tensor],
) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    offset = 0
    for reference in references:
        count = reference.numel()
        result.append(vector[offset:offset + count].reshape_as(reference))
        offset += count
    if offset != vector.numel():
        raise ValueError("flat vector does not match parameter layout")
    return result


def validate_resume_signature(
    saved: object,
    requested: dict,
    *,
    allow_target_extension: bool,
    allow_dtype_promotion: bool = False,
    allow_legacy_diagnostic_upgrade: bool = False,
) -> tuple[str, ...]:
    """Validate an exact resume, optionally permitting only a harder target."""

    if saved == requested:
        return ()
    if not isinstance(saved, dict):
        raise RuntimeError("checkpoint signature mismatch")
    saved_remaining = dict(saved)
    requested_remaining = dict(requested)
    migrations: list[str] = []

    # Git also changes for documentation and plots. Exact nonempty hashes of
    # every computational module, when available, are the resume authority.
    saved_sources = saved_remaining.get("source_sha256")
    requested_sources = requested_remaining.get("source_sha256")
    if (
        saved_remaining.get("git_commit") != requested_remaining.get("git_commit")
        and isinstance(saved_sources, dict) and bool(saved_sources)
        and saved_sources == requested_sources
    ):
        migrations.append(
            f"git_commit:{saved_remaining.get('git_commit')}->"
            f"{requested_remaining.get('git_commit')}:identical_computational_sources"
        )
        saved_remaining.pop("git_commit", None)
        requested_remaining.pop("git_commit", None)

    if saved_remaining.get("target_loss") != requested_remaining.get("target_loss"):
        if not allow_target_extension:
            raise RuntimeError("checkpoint signature mismatch")
        try:
            saved_target = float(saved_remaining["target_loss"])
            requested_target = float(requested_remaining["target_loss"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("checkpoint signature mismatch") from error
        if not (
            math.isfinite(saved_target)
            and math.isfinite(requested_target)
            and 0.0 < requested_target < saved_target
        ):
            raise RuntimeError(
                "target extension must request a strictly smaller positive target_loss"
            )
        migrations.append(f"target_loss:{saved_target:g}->{requested_target:g}")
        saved_remaining.pop("target_loss")
        requested_remaining.pop("target_loss")

    if saved_remaining.get("max_steps") != requested_remaining.get("max_steps"):
        try:
            saved_max_steps = int(saved_remaining["max_steps"])
            requested_max_steps = int(requested_remaining["max_steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("checkpoint signature mismatch") from error
        if not (0 < saved_max_steps < requested_max_steps):
            raise RuntimeError(
                "max_steps migration must strictly extend a positive horizon"
            )
        migrations.append(
            f"max_steps:{saved_max_steps}->{requested_max_steps}"
        )
        saved_remaining.pop("max_steps")
        requested_remaining.pop("max_steps")

    if saved_remaining.get("dtype") != requested_remaining.get("dtype"):
        if not (
            allow_dtype_promotion
            and saved_remaining.get("dtype") == "float32"
            and requested_remaining.get("dtype") == "float64"
        ):
            raise RuntimeError("checkpoint signature mismatch")
        migrations.append("dtype:float32->float64")
        saved_remaining.pop("dtype")
        requested_remaining.pop("dtype")

    saved_version = int(saved_remaining.get("diagnostic_version", -1))
    requested_version = int(requested_remaining.get("diagnostic_version", -1))
    if saved_version != requested_version:
        if not (
            allow_legacy_diagnostic_upgrade
            and saved_version == 1
            and requested_version == 2
        ):
            raise RuntimeError("checkpoint signature mismatch")
        if "beta_policy" in saved_remaining:
            raise RuntimeError("legacy diagnostic signature unexpectedly has beta_policy")
        if requested_remaining.get("beta_policy") != "fixed":
            raise RuntimeError("legacy diagnostic upgrade requires fixed beta policy")
        migrations.append("diagnostic_version:1->2")
        migrations.append("beta_policy:implicit_fixed->fixed")
        saved_remaining.pop("diagnostic_version")
        requested_remaining.pop("diagnostic_version")
        requested_remaining.pop("beta_policy")

    if saved_remaining != requested_remaining:
        raise RuntimeError("checkpoint signature mismatch")
    return tuple(migrations)


def completed_result_matches_request(
    result: object,
    requested_signature: dict,
    *,
    allow_target_extension: bool,
    allow_dtype_promotion: bool = False,
    allow_legacy_diagnostic_upgrade: bool = False,
) -> bool:
    """Return whether a prior result already satisfies this resume request."""

    if not isinstance(result, dict):
        return False
    validate_resume_signature(
        result.get("signature"),
        requested_signature,
        allow_target_extension=allow_target_extension,
        allow_dtype_promotion=allow_dtype_promotion,
        allow_legacy_diagnostic_upgrade=allow_legacy_diagnostic_upgrade,
    )
    try:
        final_loss = float(result["final_loss"])
        target_loss = float(requested_signature["target_loss"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        result.get("target_reached", False)
        and result.get("completion_reason") == "target_loss"
        and math.isfinite(final_loss)
        and final_loss <= target_loss
    )


def upgrade_legacy_scalar_records(
    records: dict[str, list], migrations: tuple[str, ...],
) -> None:
    """Make a copied v1 record table append-compatible with v2 diagnostics."""

    if "diagnostic_version:1->2" not in migrations:
        return
    count = len(records.get("step", ()))
    for anchor in ANCHORS:
        tag = anchor_tag(anchor)
        for suffix in ("metric_scale_aligned_error", "metric_cosine"):
            key = f"{tag}_{suffix}"
            if key in records:
                raise RuntimeError(f"legacy scalar records unexpectedly contain {key}")
            records[key] = [math.nan] * count


def beta_at_rate(reference: float, reference_rate: float, rate: float) -> float:
    return float(reference ** (reference_rate / rate))


def optimizer_betas(args: argparse.Namespace, rate: float) -> tuple[float, float]:
    """Resolve Adam betas under the explicitly selected comparison policy."""

    if args.beta_policy == "fixed":
        return float(args.beta1_reference), float(args.beta2_reference)
    if args.beta_policy == "rescaled":
        return (
            beta_at_rate(args.beta1_reference, args.beta_reference_rate, rate),
            beta_at_rate(args.beta2_reference, args.beta_reference_rate, rate),
        )
    raise ValueError(f"unknown beta policy: {args.beta_policy}")


def optimizer_payload(optimizer: InspectableAdam) -> dict:
    payload = {
        "step_count": int(optimizer.step_count),
        "first": [value.detach().cpu().clone() for value in optimizer.first],
        "second": [value.detach().cpu().clone() for value in optimizer.second],
    }
    shadow = getattr(optimizer, "log_second_shadow", None)
    if shadow is not None:
        if shadow.step_count != optimizer.step_count:
            raise RuntimeError("shadow and ordinary optimizer steps disagree")
        payload["log_second_shadow"] = shadow.checkpoint_state()
    return payload


def restore_optimizer(optimizer: InspectableAdam, payload: dict, *, allow_dtype_promotion=False) -> None:
    """Reject truncated/corrupt state before changing any ordinary buffer."""
    if not isinstance(payload, dict):
        raise RuntimeError("invalid optimizer checkpoint")
    step = payload.get("step_count")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError("invalid optimizer step_count")
    for key, destinations in (("first", optimizer.first), ("second", optimizer.second)):
        sources = payload.get(key)
        if not isinstance(sources, list) or len(sources) != len(destinations):
            raise RuntimeError(f"checkpoint {key} tensor count mismatch")
        for source, destination in zip(sources, destinations):
            if not torch.is_tensor(source) or source.shape != destination.shape:
                raise RuntimeError(f"checkpoint {key} tensor shape mismatch")
            if source.dtype != destination.dtype and not (
                allow_dtype_promotion and source.dtype == torch.float32 and destination.dtype == torch.float64
            ):
                raise RuntimeError(f"checkpoint {key} dtype mismatch")
            if not bool(torch.isfinite(source).all()) or key == "second" and bool((source < 0).any()):
                raise RuntimeError(f"checkpoint {key} has nonfinite or invalid entries")
    shadow = getattr(optimizer, "log_second_shadow", None)
    if shadow is not None:
        if "log_second_shadow" not in payload:
            raise RuntimeError("cannot reconstruct a log shadow from a raw-only checkpoint")
        if payload["log_second_shadow"].get("step_count") != step:
            raise RuntimeError("restored shadow and ordinary optimizer steps disagree")
        shadow.restore_checkpoint_state(payload["log_second_shadow"])
    elif "log_second_shadow" in payload:
        raise RuntimeError("shadow checkpoint requires --log_second_moment_shadow")
    with torch.no_grad():
        for key, destinations in (("first", optimizer.first), ("second", optimizer.second)):
            for source, destination in zip(payload[key], destinations):
                destination.copy_(source)
    optimizer.step_count = step


def write_shadow_record(path: Path, record: dict, *, mode: str = "a") -> None:
    """Stream live diagnostics; NPZ retains nonfinite values represented here by null."""
    clean = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in record.items()
    }
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(json.dumps(clean, allow_nan=False) + "\n")


def cpu_nested(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_nested(item) for item in value]
    return value


def device_nested(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: device_nested(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [device_nested(item, device) for item in value]
    return value


def rng_state_payload() -> dict[str, object]:
    return {
        "torch_cpu": torch.random.get_rng_state().cpu(),
        "torch_cuda": (
            [value.cpu() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(payload: dict[str, object]) -> None:
    torch.random.set_rng_state(payload["torch_cpu"])
    cuda_states = payload.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(payload["numpy"])
    random.setstate(payload["python"])


def layer_parameter_indices(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    names = [name for name, _ in model.named_parameters()]
    groups = {
        layer: tuple(
            index for index, name in enumerate(names)
            if name.startswith(f"{layer}.")
        )
        for layer in model.layer_names
    }
    if any(not indices for indices in groups.values()):
        raise ValueError(f"model parameters do not match layer schema: {names}")
    covered = sorted(index for indices in groups.values() for index in indices)
    if covered != list(range(len(names))):
        raise ValueError(f"unassigned model parameters in layer schema: {names}")
    return groups


def concatenate_parameter_values(
    values: list[torch.Tensor], indices: tuple[int, ...],
) -> torch.Tensor:
    return torch.cat([values[index].reshape(-1) for index in indices])


def _named_values(
    model: torch.nn.Module, values: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    names = [name for name, _ in model.named_parameters()]
    if len(names) != len(values):
        raise ValueError("named parameter/value count mismatch")
    return dict(zip(names, values))


def _layer_affine_values(
    named: dict[str, torch.Tensor], layer: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    weight = named[f"{layer}.weight"]
    bias = named.get(f"{layer}.bias")
    block_values = [weight.reshape(-1)]
    if bias is not None:
        block_values.append(bias.reshape(-1))
    return weight, bias, torch.cat(block_values)


@torch.no_grad()
def layer_scalar_diagnostics(
    *,
    model: torch.nn.Module,
    gradients: list[torch.Tensor],
    first_raw: torch.Tensor,
    second_raw: torch.Tensor,
    first_hat: torch.Tensor,
    root_second: torch.Tensor,
    descent_action: torch.Tensor,
    learning_rate: float,
    raw_margin: float,
    degree: float,
    spectral_norm_steps: int,
    nuclear_norm_steps: int,
    expensive_due: bool,
) -> dict[str, float | int]:
    parameters_list = [parameter.detach() for parameter in model.parameters()]
    first_raw_list = split_flat_like(first_raw, parameters_list)
    second_raw_list = split_flat_like(second_raw, parameters_list)
    first_list = split_flat_like(first_hat, parameters_list)
    root_list = split_flat_like(root_second, parameters_list)
    action_list = split_flat_like(descent_action, parameters_list)
    named_parameter = _named_values(model, parameters_list)
    named_gradient = _named_values(model, gradients)
    named_first_raw = _named_values(model, first_raw_list)
    named_second_raw = _named_values(model, second_raw_list)
    named_first = _named_values(model, first_list)
    named_root = _named_values(model, root_list)
    named_action = _named_values(model, action_list)

    result: dict[str, float | int] = {}
    layer_linf: dict[str, float] = {}
    layer_spectral: dict[str, float] = {}
    layer_weight_spectral: dict[str, float] = {}
    layer_gradient_nuclear: dict[str, float] = {}
    layer_action_spectral: dict[str, float] = {}
    for layer in model.layer_names:
        parameter_weight, parameter_bias, parameter_block = _layer_affine_values(
            named_parameter, layer,
        )
        gradient_weight, gradient_bias, gradient_block = _layer_affine_values(
            named_gradient, layer,
        )
        _, _, first_raw_block = _layer_affine_values(named_first_raw, layer)
        _, _, second_raw_block = _layer_affine_values(named_second_raw, layer)
        _, _, first_block = _layer_affine_values(named_first, layer)
        _, _, root_block = _layer_affine_values(named_root, layer)
        action_weight, action_bias, action_block = _layer_affine_values(
            named_action, layer,
        )
        applied_block = learning_rate * action_block

        for name, vector in (
            ("parameter", parameter_block),
            ("gradient", gradient_block),
            ("first_moment_raw", first_raw_block),
            ("second_moment_raw", second_raw_block),
            ("first_moment_hat", first_block),
            ("root_second_moment_hat", root_block),
            ("update", action_block),
            ("applied_update", applied_block),
        ):
            result.update(prefixed_values(
                f"{layer}_{name}", vector_norms(vector),
            ))
        dynamic = parameter_dynamic_range(parameter_block)
        result.update({
            f"{layer}_parameter_max_abs": dynamic["parameter_max_abs"],
            f"{layer}_parameter_nonzero_min_abs": dynamic[
                "parameter_nonzero_min_abs"
            ],
            f"{layer}_parameter_nonzero_count": dynamic[
                "parameter_nonzero_count"
            ],
            f"{layer}_parameter_coordinate_ratio": dynamic[
                "parameter_coordinate_ratio"
            ],
            f"{layer}_log_parameter_coordinate_ratio": dynamic[
                "log_parameter_coordinate_ratio"
            ],
            f"{layer}_weight_linf": float(parameter_weight.abs().max().item()),
            f"{layer}_bias_linf": (
                math.nan
                if parameter_bias is None
                else float(parameter_bias.abs().max().item())
            ),
        })
        for diagnostic_name in (
            "g2_over_v", "v_over_g2", "inverse_root_second",
        ):
            if not expensive_due:
                summary = distribution_summary(torch.empty(
                    0, dtype=gradient_block.dtype, device=gradient_block.device,
                ))
            elif diagnostic_name in ("g2_over_v", "v_over_g2"):
                # Both layer summaries and coordinate panels use RAW v_(t+1),
                # not bias-corrected v_hat. Undefined/inactive ratios stay NaN.
                summary_values = torch.full_like(gradient_block, math.nan)
                gradient_squared = gradient_block.square()
                active_values = (second_raw_block > 0) & (gradient_squared > 0)
                numerator, denominator = (
                    (gradient_squared, second_raw_block) if diagnostic_name == "g2_over_v"
                    else (second_raw_block, gradient_squared)
                )
                summary_values[active_values] = numerator[active_values] / denominator[active_values]
                summary = distribution_summary(summary_values)
            else:
                summary_values = torch.full_like(root_block, math.nan)
                active_values = root_block > 0
                summary_values[active_values] = root_block[active_values].reciprocal()
                summary = distribution_summary(summary_values)
            result.update(prefixed_values(
                f"{layer}_{diagnostic_name}", summary,
            ))
        layer_linf[layer] = float(parameter_block.abs().max().item())

        parameter_matrix = affine_matrix(parameter_weight, parameter_bias)
        gradient_matrix = affine_matrix(gradient_weight, gradient_bias)
        action_matrix = affine_matrix(action_weight, action_bias)
        layer_spectral[layer] = deterministic_spectral_norm(
            parameter_matrix, steps=spectral_norm_steps,
        )
        layer_weight_spectral[layer] = deterministic_spectral_norm(
            parameter_weight, steps=spectral_norm_steps,
        )
        layer_action_spectral[layer] = deterministic_spectral_norm(
            action_matrix, steps=spectral_norm_steps,
        )
        result[f"{layer}_affine_spectral_norm"] = layer_spectral[layer]
        result[f"{layer}_weight_spectral_norm"] = layer_weight_spectral[layer]
        result[f"{layer}_update_affine_spectral_norm"] = (
            layer_action_spectral[layer]
        )
        if expensive_due:
            layer_gradient_nuclear[layer] = float(nuclear_norm_iterative(
                gradient_matrix,
                polar_steps=nuclear_norm_steps,
                epsilon=1e-30,
            ).item())
        else:
            layer_gradient_nuclear[layer] = math.nan
        result[f"{layer}_gradient_nuclear_norm"] = (
            layer_gradient_nuclear[layer]
        )

    for layer, value in layer_linf.items():
        result[f"{layer}_layer_linf"] = value
    minimum = min(layer_linf.values())
    balance = (
        math.nan if minimum <= 0.0 else max(layer_linf.values()) / minimum
    )
    result.update({
        "layer_linf_balance": balance,
        "layer_linf_imbalance": (
            math.nan if not math.isfinite(balance) else balance - 1.0
        ),
        "log_layer_linf_balance": (
            math.nan if not math.isfinite(balance) else math.log(balance)
        ),
        "max_affine_spectral_norm": max(layer_spectral.values()),
        "max_weight_spectral_norm": max(layer_weight_spectral.values()),
    })
    maximum_spectral = max(layer_spectral.values())
    result["msp_margin"] = (
        math.nan
        if maximum_spectral <= 0.0
        else float(raw_margin / maximum_spectral**degree)
    )

    if expensive_due:
        gradient_dual = sum(layer_gradient_nuclear.values())
        result["msp_gradient_dual_norm"] = gradient_dual
        update_primal = max(layer_action_spectral.values())
        parameter_primal = maximum_spectral
        gradient_flat = torch.cat([value.reshape(-1) for value in gradients])
        parameter_flat = torch.cat([
            value.reshape(-1) for value in parameters_list
        ])
        result["msp_update_gradient_alignment"] = (
            math.nan
            if update_primal <= 0.0 or gradient_dual <= 0.0
            else float(
                -torch.dot(descent_action, gradient_flat).item()
                / (update_primal * gradient_dual)
            )
        )
        result["msp_parameter_gradient_alignment"] = (
            math.nan
            if parameter_primal <= 0.0 or gradient_dual <= 0.0
            else float(
                -torch.dot(parameter_flat, gradient_flat).item()
                / (parameter_primal * gradient_dual)
            )
        )
    else:
        result["msp_gradient_dual_norm"] = math.nan
        result["msp_update_gradient_alignment"] = math.nan
        result["msp_parameter_gradient_alignment"] = math.nan

    for layer in model.layer_names:
        _, _, root_block = _layer_affine_values(named_root, layer)
        active = root_block > 0
        if bool(active.any()):
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            local = torch.argmin(root_block[active])
            index = int(active_indices[local].item())
            result[f"{layer}_inverse_metric_max"] = float(
                root_block[index].reciprocal().item()
            )
            result[f"{layer}_inverse_metric_argmax"] = index
        else:
            result[f"{layer}_inverse_metric_max"] = math.nan
            result[f"{layer}_inverse_metric_argmax"] = -1
    return result


def current_gradients(model: torch.nn.Module) -> list[torch.Tensor]:
    return [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach()
        for parameter in model.parameters()
    ]


@torch.no_grad()
def next_moment_vectors(
    optimizer: InspectableAdam,
    gradients: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    next_step = optimizer.step_count + 1
    correction1 = 1.0 - optimizer.beta1 ** next_step
    correction2 = 1.0 - optimizer.beta2 ** next_step
    first_next = [
        first.clone().mul_(optimizer.beta1).add_(gradient, alpha=1.0 - optimizer.beta1)
        for first, gradient in zip(optimizer.first, gradients)
    ]
    second_next = [
        second.clone().mul_(optimizer.beta2).addcmul_(gradient, gradient, value=1.0 - optimizer.beta2)
        for second, gradient in zip(optimizer.second, gradients)
    ]
    first_hat = torch.cat([
        value.reshape(-1) / correction1 for value in first_next
    ])
    root_second = torch.cat([
        (value / correction2).clamp_min(0).sqrt().reshape(-1)
        for value in second_next
    ])
    return first_next, second_next, first_hat, root_second


def masked_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(numerator)
    active = denominator > 0
    result[active] = numerator[active] / denominator[active]
    return result


def frozen_inverse_action(numerator: torch.Tensor, metric: torch.Tensor) -> torch.Tensor | None:
    """Apply a frozen diagonal inverse only on its mathematically defined domain.

    A zero frozen entry with a nonzero current numerator has no finite inverse
    action. Returning None prevents newly active ReLU coordinates from being
    silently projected away in frozen-action comparisons. Compatible zero
    entries receive zero, as in the optimizer's genuinely inactive coordinates.
    """
    if numerator.shape != metric.shape:
        raise ValueError("frozen metric and numerator shapes must agree")
    if not bool((torch.isfinite(numerator) & torch.isfinite(metric) & (metric >= 0)).all()):
        return None
    active = metric > 0
    if bool((numerator[~active] != 0).any()):
        return None
    result = masked_ratio(numerator, metric)
    return result if bool(torch.isfinite(result).all()) else None


def frozen_metric_alignment(action: torch.Tensor, gradient: torch.Tensor,
                            metric: torch.Tensor) -> float:
    """Dual alignment on compatible frozen support; undefined otherwise.

    Evaluate cos(sqrt(metric)*action, -gradient/sqrt(metric)) using log-scaled
    weighted vectors. This avoids squaring tiny gradients or multiplying tiny
    frozen moments before normalization. Both vectors must be supported on the
    positive frozen coordinates; zero vectors have no alignment.
    """
    if action.shape != gradient.shape or action.shape != metric.shape:
        raise ValueError("frozen alignment tensor shapes must agree")
    if not bool((torch.isfinite(action) & torch.isfinite(gradient)
                 & torch.isfinite(metric) & (metric >= 0)).all()):
        return math.nan
    active = metric > 0
    if not bool(active.any()) or bool(((action[~active] != 0) | (gradient[~active] != 0)).any()):
        return math.nan
    log_metric = metric[active].double().log()

    def weighted_unit_scale(vector: torch.Tensor, exponent: float) -> torch.Tensor | None:
        vector = vector[active].double()
        logarithm = vector.abs().log() + exponent * log_metric
        maximum = logarithm.max()
        if not bool(torch.isfinite(maximum)):
            return None
        return vector.sign() * (logarithm - maximum).exp()

    weighted_action = weighted_unit_scale(action, 0.5)
    weighted_negative_gradient = weighted_unit_scale(-gradient, -0.5)
    if weighted_action is None or weighted_negative_gradient is None:
        return math.nan
    return safe_cosine(weighted_action, weighted_negative_gradient)


def scale_aligned_error(
    target: torch.Tensor, reference: torch.Tensor,
) -> tuple[float, float]:
    if not target.numel() or not reference.numel():
        return math.nan, math.nan
    target_scale, reference_scale = target.abs().max(), reference.abs().max()
    if not (target_scale > 0 and reference_scale > 0 and torch.isfinite(target_scale) and torch.isfinite(reference_scale)):
        return math.nan, math.nan
    target_unit, reference_unit = target / target_scale, reference / reference_scale
    target_unit = target_unit / target_unit.norm()
    reference_unit = reference_unit / reference_unit.norm()
    cosine = torch.dot(target_unit, reference_unit)
    # Compute the residual directly: sqrt(1-cosine**2) loses small errors.
    residual = target_unit - cosine * reference_unit
    residual_scale = residual.abs().max()
    error = residual_scale * (residual / residual_scale).norm() if residual_scale > 0 else residual_scale
    return float(error.clamp(0, 1).item()), float(cosine.clamp(-1, 1).item())


def normalized_margin(
    parameters: torch.Tensor,
    raw_margin: float,
    degree: float,
    metric: torch.Tensor | None,
    norm_kind: str,
) -> float:
    if norm_kind == "l2":
        norm = parameters.norm()
    elif norm_kind == "linf":
        norm = parameters.abs().max()
    elif norm_kind == "metric":
        if metric is None:
            return math.nan
        metric_scale = metric.abs().max()
        if not (metric_scale > 0 and torch.isfinite(metric_scale)) or (metric < 0).any():
            return math.nan
        relative_metric = metric / metric_scale
        normalized_metric = relative_metric / relative_metric.square().mean().sqrt()
        norm = torch.dot(normalized_metric, parameters.square()).clamp_min(0).sqrt()
    else:
        raise ValueError(norm_kind)
    if norm <= 0:
        return math.nan
    return float(raw_margin / float(norm.item()) ** degree)


def parameter_dynamic_range(parameters: torch.Tensor) -> dict[str, float | int]:
    """Summarize absolute parameter magnitudes, excluding exact zeros.

    The log ratio is evaluated as a difference of logarithms so it remains
    useful even when the corresponding ordinary ratio overflows float64.
    """

    magnitudes = parameters.detach().abs()
    maximum = float(magnitudes.max().item())
    nonzero = magnitudes[magnitudes > 0]
    count = int(nonzero.numel())
    if count == 0:
        return {
            "parameter_max_abs": maximum,
            "parameter_nonzero_min_abs": math.nan,
            "parameter_nonzero_count": count,
            "parameter_coordinate_ratio": math.nan,
            "log_parameter_coordinate_ratio": math.nan,
        }
    minimum = float(nonzero.min().item())
    log_ratio = math.log(maximum) - math.log(minimum)
    ratio = (
        maximum / minimum
        if log_ratio <= math.log(np.finfo(np.float64).max)
        else math.inf
    )
    return {
        "parameter_max_abs": maximum,
        "parameter_nonzero_min_abs": minimum,
        "parameter_nonzero_count": count,
        "parameter_coordinate_ratio": ratio,
        "log_parameter_coordinate_ratio": log_ratio,
    }


class CoordinateSketch:
    def __init__(self, size: int, dimension: int, seed: int, device: torch.device):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        dimension = min(int(dimension), int(size))
        self.indices = torch.randperm(size, generator=generator)[:dimension].to(device)
        signs = torch.randint(0, 2, (dimension,), generator=generator, dtype=torch.int8)
        self.sign = signs.to(device=device).mul_(2).sub_(1)
        self.dimension = int(dimension)

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        result = vector[self.indices] * self.sign.to(vector.dtype)
        norm = result.norm()
        return result if norm <= 0 else result / norm


def initialize_scalar_records(
    *, include_parameter_dynamic_range: bool = False,
    layer_names: tuple[str, ...] = LAYER_NAMES,
) -> dict[str, list[float]]:
    # ``include_parameter_dynamic_range`` remains in the public signature for
    # callers of diagnostic-v2 helpers.  Canonical diagnostic-v3 records the
    # range for every network rather than only for the linear experiment.
    del include_parameter_dynamic_range
    records = {
        "step": [], "moment_step": [], "loss": [], "classification_error": [],
        "negative_log10_loss": [], "learning_rate": [],
        "cumulative_learning_rate": [], "elapsed_seconds_total": [],
        "raw_margin": [], "q_min": [], "q_min_index": [],
        "near_active_count_1pct": [], "near_active_count_5pct": [],
        "near_active_count_10pct": [],
        "parameter_l1": [], "parameter_l2": [], "parameter_linf": [],
        "parameter_zero_count": [],
        "gradient_l1": [], "gradient_l2": [], "gradient_linf": [],
        "gradient_zero_count": [],
        "gradient_square_underflow_count": [],
        "second_moment_subnormal_count": [],
        "first_moment_raw_l1": [], "first_moment_raw_l2": [],
        "first_moment_raw_linf": [], "first_moment_raw_zero_count": [],
        "second_moment_raw_l1": [], "second_moment_raw_l2": [],
        "second_moment_raw_linf": [], "second_moment_raw_zero_count": [],
        "first_moment_hat_l1": [], "first_moment_hat_l2": [],
        "first_moment_hat_linf": [], "first_moment_hat_zero_count": [],
        "root_second_moment_hat_l1": [], "root_second_moment_hat_l2": [],
        "root_second_moment_hat_linf": [],
        "root_second_moment_hat_zero_count": [],
        "update_l1": [], "update_l2": [], "update_linf": [],
        "update_zero_count": [],
        "applied_update_l1": [], "applied_update_l2": [],
        "applied_update_linf": [], "applied_update_zero_count": [],
        "dynamic_sign_error": [], "actual_sign_error": [],
        "corrected_linf_pairing": [],
        "euclidean_update_gradient_cosine": [],
        "linf_dual_alignment": [],
        "transformed_first_moment_log_l2": [],
        "euclidean_margin": [], "linf_margin": [],
        "msp_margin": [],
        "parameter_sketch": [], "gradient_sketch": [],
        "parameter_max_abs": [],
        "parameter_nonzero_min_abs": [],
        "parameter_nonzero_count": [],
        "parameter_coordinate_ratio": [],
        "log_parameter_coordinate_ratio": [],
        "max_affine_spectral_norm": [],
        "max_weight_spectral_norm": [],
        "layer_linf_balance": [], "layer_linf_imbalance": [],
        "log_layer_linf_balance": [],
        "msp_gradient_dual_norm": [],
        "msp_update_gradient_alignment": [],
        "msp_parameter_gradient_alignment": [],
        "peak_cuda_memory_allocated_bytes": [],
    }
    for layer in layer_names:
        for name in (
            "weight_linf", "bias_linf", "affine_spectral_norm",
            "weight_spectral_norm", "update_affine_spectral_norm",
            "gradient_nuclear_norm", "layer_linf",
        ):
            records[f"{layer}_{name}"] = []
        for quantity in (
            "parameter", "gradient", "first_moment_raw", "second_moment_raw",
            "first_moment_hat",
            "root_second_moment_hat", "update", "applied_update",
        ):
            for norm in ("l1", "l2", "linf", "zero_count"):
                records[f"{layer}_{quantity}_{norm}"] = []
        for name in (
            "parameter_max_abs", "parameter_nonzero_min_abs",
            "parameter_nonzero_count", "parameter_coordinate_ratio",
            "log_parameter_coordinate_ratio",
            "inverse_metric_max", "inverse_metric_argmax",
        ):
            records[f"{layer}_{name}"] = []
        for diagnostic_name in (
            "g2_over_v", "v_over_g2", "inverse_root_second",
        ):
            for statistic in (
                "count", "mean", "std", "minimum", "q33", "median",
                "q66", "maximum",
            ):
                records[f"{layer}_{diagnostic_name}_{statistic}"] = []
    for anchor in ANCHORS:
        tag = anchor_tag(anchor)
        for key in (
            "dynamic_frozen_error", "actual_frozen_error",
            "gradient_action_cosine", "momentum_action_cosine",
            "frozen_l2_margin", "frozen_l2_alignment",
            "metric_scale_aligned_error", "metric_cosine",
            "update_frozen_cosine", "momentum_frozen_cosine",
            "second_moment_cosine",
        ):
            records[f"{tag}_{key}"] = []
    return records


def initialize_coordinate_records(layer_names: tuple[str, ...] = LAYER_NAMES) -> dict:
    result = {"step": [], "loss": []}
    for layer in layer_names:
        result[layer] = {
            "gradient": [], "parameter": [], "raw_second": [],
            "first_moment_hat": [], "descent_action": [],
            "inverse_root_second": [],
            "g2_over_v": [], "v_over_g2": [], "rho_over_c2": [],
        }
    return result


def anchor_tag(anchor: float) -> str:
    return f"a1e{int(round(math.log10(anchor)))}".replace("-", "m")


def update_selection_scores(
    scores: dict[str, torch.Tensor],
    nonzero_counts: dict[str, torch.Tensor],
    observations: int,
    gradients: list[torch.Tensor],
    groups: dict[str, tuple[int, ...]],
    decay: float = 0.9,
) -> int:
    observations += 1
    for layer, indices in groups.items():
        gradient = concatenate_parameter_values(gradients, indices)
        normalized = gradient.abs() / gradient.abs().sum().clamp_min(1e-30)
        scores[layer].mul_(decay).add_(normalized, alpha=1.0 - decay)
        nonzero_counts[layer].add_(gradient != 0)
    return observations


def choose_coordinates(
    *,
    scores: dict[str, torch.Tensor],
    nonzero_counts: dict[str, torch.Tensor],
    observations: int,
    count: int,
    seed: int,
) -> dict:
    selection = {}
    for layer_index, layer in enumerate(scores):
        eligible = nonzero_counts[layer] >= math.ceil(0.9 * observations)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if eligible_indices.numel() == 0:
            eligible_indices = torch.nonzero(
                nonzero_counts[layer] > 0, as_tuple=False,
            ).flatten()
        if eligible_indices.numel() == 0:
            # A dead ReLU layer is a valid observed state; preserve a reproducible
            # inactive panel whose undefined ratios remain NaN.
            generator = torch.Generator().manual_seed(seed + 97 * layer_index)
            indices = torch.randperm(scores[layer].numel(), generator=generator)[:count]
            selection[layer] = {"indices": indices, "sample_type": np.asarray(["inactive"] * len(indices), dtype="U12"),
                                "persistent_count": 0, "random_count": int(len(indices))}
            continue
        local_scores = scores[layer][eligible_indices]
        top_count = min(count, int(eligible_indices.numel()))
        top_local = torch.topk(local_scores, top_count).indices
        top = eligible_indices[top_local]
        remaining_mask = torch.ones(
            eligible_indices.numel(), dtype=torch.bool, device=eligible_indices.device,
        )
        remaining_mask[top_local] = False
        remaining = eligible_indices[remaining_mask].cpu()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 97 * layer_index)
        random_count = min(count, int(remaining.numel()))
        random = remaining[
            torch.randperm(remaining.numel(), generator=generator)[:random_count]
        ].to(top.device)
        selection[layer] = {
            "indices": torch.cat((top, random)).detach().cpu(),
            "sample_type": np.asarray(
                ["persistent"] * len(top) + ["random"] * len(random),
                dtype="U12",
            ),
            "persistent_count": int(len(top)),
            "random_count": int(len(random)),
        }
    return selection


def coordinate_metrics(
    *,
    gradient: torch.Tensor,
    parameter: torch.Tensor,
    first_hat: torch.Tensor,
    root_second: torch.Tensor,
    second_previous: torch.Tensor,
    second_next: torch.Tensor,
    beta2: float,
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    indices_device = indices.to(gradient.device)
    g = gradient[indices_device].detach().double().cpu()
    theta = parameter[indices_device].detach().double().cpu()
    momentum = first_hat[indices_device].detach().double().cpu()
    root = root_second[indices_device].detach().double().cpu()
    action = -masked_ratio(momentum, root)
    inverse_root = torch.zeros_like(root)
    active_root = root > 0
    inverse_root[active_root] = root[active_root].reciprocal()
    previous = second_previous[indices_device].detach().double().cpu()
    updated = second_next[indices_device].detach().double().cpu()
    ratios = second_moment_tracking_ratios(updated.numpy(), g.numpy())
    # Use the previous raw buffer directly. Reconstructing the increment
    # from g^2/v_next is singular when v_previous=0 and cancellation-prone
    # near that boundary. This log-domain recurrence also handles g=0.
    rho_over_c2 = np.full(previous.shape, np.nan, dtype=np.float64)
    active_previous = (previous > 0) & (updated > 0) & torch.isfinite(previous) & torch.isfinite(updated) & torch.isfinite(g)
    log_relative_increment = (
        math.log1p(-beta2) - math.log(beta2)
        + 2 * torch.log(g[active_previous].abs()) - torch.log(previous[active_previous])
    )
    rho_over_c2[active_previous.numpy()] = (
        torch.logaddexp(torch.zeros_like(log_relative_increment), log_relative_increment) / (-math.log(beta2))
    ).numpy()
    return {
        "gradient": g,
        "parameter": theta,
        "first_moment_hat": momentum,
        "descent_action": action,
        "inverse_root_second": inverse_root,
        "raw_second": updated,
        "g2_over_v": torch.from_numpy(
            ratios.gradient_squared_over_second,
        ),
        "v_over_g2": torch.from_numpy(
            ratios.second_over_gradient_squared,
        ),
        "rho_over_c2": torch.from_numpy(rho_over_c2),
    }


def append_coordinate_record(
    records: dict,
    *,
    step: int,
    loss: float,
    selection: dict,
    model: torch.nn.Module,
    optimizer: InspectableAdam,
    first_hat: torch.Tensor,
    root_second: torch.Tensor,
    second_next: list[torch.Tensor],
    gradients: list[torch.Tensor],
    groups: dict[str, tuple[int, ...]],
) -> None:
    records["step"].append(int(step))
    records["loss"].append(float(loss))
    parameters = [parameter.detach() for parameter in model.parameters()]
    first_values = split_flat_like(first_hat, parameters)
    root_values = split_flat_like(root_second, parameters)
    for layer, group_indices in groups.items():
        metrics = coordinate_metrics(
            gradient=concatenate_parameter_values(gradients, group_indices),
            parameter=concatenate_parameter_values(parameters, group_indices),
            first_hat=concatenate_parameter_values(first_values, group_indices),
            root_second=concatenate_parameter_values(root_values, group_indices),
            second_previous=concatenate_parameter_values(
                optimizer.second, group_indices,
            ),
            second_next=concatenate_parameter_values(
                second_next, group_indices,
            ),
            beta2=optimizer.beta2,
            indices=selection[layer]["indices"],
        )
        for key, value in metrics.items():
            records[layer][key].append(value)


def append_scalar_record(
    records: dict[str, list],
    *,
    step: int,
    loss: float,
    error: float,
    learning_rate: float,
    raw_margin: float,
    parameters: torch.Tensor,
    gradient: torch.Tensor,
    root_second: torch.Tensor,
    first_hat: torch.Tensor,
    anchors: dict[str, torch.Tensor],
    degree: float,
    parameter_sketch: torch.Tensor,
    gradient_sketch: torch.Tensor,
    first_raw: torch.Tensor | None = None,
    second_raw: torch.Tensor | None = None,
    include_parameter_dynamic_range: bool = False,
    model: torch.nn.Module | None = None,
    gradients: list[torch.Tensor] | None = None,
    signed_margins: torch.Tensor | None = None,
    cumulative_learning_rate: float = math.nan,
    elapsed_seconds_total: float = math.nan,
    beta1: float = 0.9,
    beta2: float = 0.999,
    anchor_momenta: dict[str, torch.Tensor] | None = None,
    anchor_updates: dict[str, torch.Tensor] | None = None,
    anchor_second_moments: dict[str, torch.Tensor] | None = None,
    spectral_norm_steps: int = 20,
    nuclear_norm_steps: int = 12,
    expensive_due: bool = False,
) -> None:
    del include_parameter_dynamic_range
    anchor_momenta = {} if anchor_momenta is None else anchor_momenta
    anchor_updates = {} if anchor_updates is None else anchor_updates
    anchor_second_moments = (
        {} if anchor_second_moments is None else anchor_second_moments
    )
    dynamic = masked_ratio(gradient, root_second)
    actual = -masked_ratio(first_hat, root_second)
    applied = learning_rate * actual
    moment_step = step + 1
    first_correction = (
        1.0 if beta1 == 0.0
        else -math.expm1(moment_step * math.log(beta1))
    )
    second_correction = (
        1.0 if beta2 == 0.0
        else -math.expm1(moment_step * math.log(beta2))
    )
    if first_raw is None:
        first_raw = first_hat * first_correction
    if second_raw is None:
        second_raw = root_second.square() * second_correction
    if first_raw.shape != first_hat.shape or second_raw.shape != root_second.shape:
        raise ValueError("raw and bias-corrected moment vectors must match")
    sign = gradient.sign()
    dynamic_sign_error, _ = scale_aligned_error(dynamic, sign)
    actual_sign_error, _ = scale_aligned_error(actual, -sign)
    gradient_l1 = gradient.abs().sum()
    update_linf = actual.abs().max()
    corrected_pairing = (
        math.nan if gradient_l1 <= 0
        else float((-torch.dot(actual, gradient) / gradient_l1).item())
    )
    integer_fields = {
        key for key in records
        if key in ("step", "moment_step") or key.endswith("_count") or key.endswith("_index")
        or key.endswith("_argmax")
    }
    values: dict[str, object] = {
        key: (-1 if key in integer_fields else math.nan)
        for key in records
        if key not in {"parameter_sketch", "gradient_sketch"}
    }
    values.update({
        "step": int(step), "moment_step": int(step + 1), "loss": float(loss),
        "classification_error": float(error),
        "gradient_square_underflow_count": int(((gradient != 0) & (gradient.square() == 0)).sum().item()),
        "second_moment_subnormal_count": int(((second_raw > 0) & (second_raw < torch.finfo(second_raw.dtype).tiny)).sum().item()),
        "negative_log10_loss": -math.log10(loss) if loss > 0 else math.nan,
        "learning_rate": float(learning_rate), "raw_margin": float(raw_margin),
        "q_min": float(raw_margin),
        "cumulative_learning_rate": float(cumulative_learning_rate),
        "elapsed_seconds_total": float(elapsed_seconds_total),
        "dynamic_sign_error": dynamic_sign_error,
        "actual_sign_error": actual_sign_error,
        "corrected_linf_pairing": corrected_pairing,
        "euclidean_update_gradient_cosine": safe_cosine(actual, -gradient),
        "linf_dual_alignment": (
            math.nan
            if update_linf <= 0 or gradient_l1 <= 0
            else float(
                -torch.dot(actual, gradient).item()
                / (update_linf.item() * gradient_l1.item())
            )
        ),
        "transformed_first_moment_log_l2": (
            math.nan
            if vector_norms(first_raw)["l2"] <= 0 or beta1 <= 0.0
            else float(
                math.log(vector_norms(first_raw)["l2"])
                - moment_step * math.log(beta1)
            )
        ),
        "euclidean_margin": normalized_margin(
            parameters, raw_margin, degree, None, "l2",
        ),
        "linf_margin": normalized_margin(
            parameters, raw_margin, degree, None, "linf",
        ),
        "parameter_sketch": parameter_sketch.detach().cpu().float(),
        "gradient_sketch": gradient_sketch.detach().cpu().float(),
    })
    values.update(prefixed_values("parameter", vector_norms(parameters)))
    values.update(prefixed_values("gradient", vector_norms(gradient)))
    values.update(prefixed_values("first_moment_raw", vector_norms(first_raw)))
    values.update(prefixed_values("second_moment_raw", vector_norms(second_raw)))
    values.update(prefixed_values("first_moment_hat", vector_norms(first_hat)))
    values.update(prefixed_values(
        "root_second_moment_hat", vector_norms(root_second),
    ))
    values.update(prefixed_values("update", vector_norms(actual)))
    values.update(prefixed_values("applied_update", vector_norms(applied)))
    values.update(parameter_dynamic_range(parameters))
    if signed_margins is not None:
        signed_values = signed_margins.detach()
        values["q_min_index"] = int(torch.argmin(signed_values).item())
        if raw_margin > 0:
            for percent in (1, 5, 10):
                values[f"near_active_count_{percent}pct"] = int(
                    (signed_values <= (1.0 + percent / 100.0) * raw_margin)
                    .sum().item()
                )
    if model is not None and gradients is not None:
        values.update(layer_scalar_diagnostics(
            model=model,
            gradients=gradients,
            first_raw=first_raw,
            second_raw=second_raw,
            first_hat=first_hat,
            root_second=root_second,
            descent_action=actual,
            learning_rate=learning_rate,
            raw_margin=raw_margin,
            degree=degree,
            spectral_norm_steps=spectral_norm_steps,
            nuclear_norm_steps=nuclear_norm_steps,
            expensive_due=expensive_due,
        ))
    values["peak_cuda_memory_allocated_bytes"] = (
        int(torch.cuda.max_memory_allocated())
        if parameters.device.type == "cuda" else 0
    )
    for anchor in ANCHORS:
        tag = anchor_tag(anchor)
        frozen = anchors.get(tag)
        if frozen is None:
            values[f"{tag}_dynamic_frozen_error"] = math.nan
            values[f"{tag}_actual_frozen_error"] = math.nan
            values[f"{tag}_frozen_l2_margin"] = math.nan
            values[f"{tag}_frozen_l2_alignment"] = math.nan
            values[f"{tag}_metric_scale_aligned_error"] = math.nan
            values[f"{tag}_metric_cosine"] = math.nan
            values[f"{tag}_gradient_action_cosine"] = math.nan
            values[f"{tag}_momentum_action_cosine"] = math.nan
            values[f"{tag}_update_frozen_cosine"] = math.nan
            values[f"{tag}_momentum_frozen_cosine"] = math.nan
            values[f"{tag}_second_moment_cosine"] = math.nan
            continue
        frozen_action = frozen_inverse_action(gradient, frozen)
        frozen_momentum_action = frozen_inverse_action(first_hat, frozen)
        (
            values[f"{tag}_dynamic_frozen_error"],
            values[f"{tag}_gradient_action_cosine"],
        ) = ((math.nan, math.nan) if frozen_action is None
             else scale_aligned_error(dynamic, frozen_action))
        (
            values[f"{tag}_actual_frozen_error"],
            values[f"{tag}_momentum_action_cosine"],
        ) = ((math.nan, math.nan) if frozen_momentum_action is None
             else scale_aligned_error(actual, -frozen_momentum_action))
        values[f"{tag}_update_frozen_cosine"] = (
            math.nan
            if tag not in anchor_updates
            else safe_cosine(actual, anchor_updates[tag])
        )
        values[f"{tag}_momentum_frozen_cosine"] = (
            math.nan
            if tag not in anchor_momenta
            else safe_cosine(first_hat, anchor_momenta[tag])
        )
        values[f"{tag}_second_moment_cosine"] = (
            math.nan
            if tag not in anchor_second_moments
            else safe_cosine(second_raw, anchor_second_moments[tag])
        )
        values[f"{tag}_frozen_l2_margin"] = normalized_margin(
            parameters, raw_margin, degree, frozen, "metric",
        )
        (
            values[f"{tag}_metric_scale_aligned_error"],
            values[f"{tag}_metric_cosine"],
        ) = scale_aligned_error(root_second, frozen)
        values[f"{tag}_frozen_l2_alignment"] = frozen_metric_alignment(
            actual, gradient, frozen,
        )
    for key, value in values.items():
        if key not in records:
            raise RuntimeError(f"undeclared scalar diagnostic field: {key}")
        records[key].append(value)
    missing = set(records).difference(values)
    if missing:
        raise RuntimeError(f"scalar diagnostics omitted fields: {sorted(missing)}")


def save_checkpoint(
    path: Path,
    *,
    signature: dict,
    model: torch.nn.Module,
    optimizer: InspectableAdam,
    scalar_records: dict,
    coordinate_records: dict,
    selection: dict | None,
    selection_step: int | None,
    selection_actual_loss: float | None,
    scores: dict[str, torch.Tensor],
    nonzero_counts: dict[str, torch.Tensor],
    score_observations: int,
    anchors: dict[str, torch.Tensor],
    anchor_momenta: dict[str, torch.Tensor],
    anchor_updates: dict[str, torch.Tensor],
    anchor_second_moments: dict[str, torch.Tensor],
    anchor_steps: dict[str, int],
    anchor_losses: dict[str, float],
    cumulative_learning_rate: float,
    elapsed_seconds_total: float,
    loss_checkpoint_records: list[dict] | None = None,
    provenance: dict | None = None,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "signature": signature,
        "provenance": provenance,
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer_state": optimizer_payload(optimizer),
        "scalar_records": cpu_nested(scalar_records),
        "coordinate_records": cpu_nested(coordinate_records),
        "selection": cpu_nested(selection),
        "selection_step": selection_step,
        "selection_actual_loss": selection_actual_loss,
        "scores": cpu_nested(scores),
        "nonzero_counts": cpu_nested(nonzero_counts),
        "score_observations": int(score_observations),
        "anchors": cpu_nested(anchors),
        "anchor_momenta": cpu_nested(anchor_momenta),
        "anchor_updates": cpu_nested(anchor_updates),
        "anchor_second_moments": cpu_nested(anchor_second_moments),
        "anchor_steps": anchor_steps,
        "anchor_losses": anchor_losses,
        "cumulative_learning_rate": float(cumulative_learning_rate),
        "elapsed_seconds_total": float(elapsed_seconds_total),
        "loss_checkpoint_records": cpu_nested(loss_checkpoint_records or []),
        "rng_state": rng_state_payload(),
    }, temporary)
    temporary.replace(path)


def save_analysis_snapshot(
    path: Path,
    *,
    signature: dict,
    threshold: float,
    step: int,
    loss: float,
    raw_margin: float,
    learning_rate: float,
    cumulative_learning_rate: float,
    model: torch.nn.Module,
    optimizer: InspectableAdam,
    gradients: list[torch.Tensor],
    first_raw: list[torch.Tensor],
    second_raw: list[torch.Tensor],
    first_hat: torch.Tensor,
    root_second: torch.Tensor,
    signed_margins: torch.Tensor,
    scalar_record: dict[str, object],
    provenance: dict | None = None,
) -> None:
    """Atomically save a compact KKT- and dynamics-ready evaluated state."""

    parameters = [parameter.detach() for parameter in model.parameters()]
    names = [name for name, _ in model.named_parameters()]
    first_values = split_flat_like(first_hat, parameters)
    root_values = split_flat_like(root_second, parameters)
    action = -masked_ratio(first_hat, root_second)
    action_values = split_flat_like(action, parameters)
    payload = {
        "schema_version": DIAGNOSTIC_VERSION,
        "purpose": "canonical loss-spaced analysis snapshot",
        "signature": signature,
        "provenance": provenance,
        "threshold": float(threshold),
        "step": int(step),
        "moment_step": int(step + 1),
        "actual_loss": float(loss),
        "q_min": float(raw_margin),
        "learning_rate": float(learning_rate),
        "cumulative_learning_rate": float(cumulative_learning_rate),
        "timing": (
            "pre-update theta_t and gradient g_t; bias-corrected moments include "
            "g_t; applied_update_t is eta_(t+1) times descent_action_t"
        ),
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer_state_before_current_gradient": optimizer_payload(optimizer),
        "gradient": {
            name: value.detach().cpu().clone()
            for name, value in zip(names, gradients)
        },
        "first_moment_raw": {
            name: value.detach().cpu().clone()
            for name, value in zip(names, first_raw)
        },
        "second_moment_raw": {
            name: value.detach().cpu().clone()
            for name, value in zip(names, second_raw)
        },
        "first_moment_hat": {
            name: value.detach().cpu().clone()
            for name, value in zip(names, first_values)
        },
        "root_second_moment_hat": {
            name: value.detach().cpu().clone()
            for name, value in zip(names, root_values)
        },
        "descent_action": {
            name: value.detach().cpu().clone()
            for name, value in zip(names, action_values)
        },
        "applied_update": {
            name: (learning_rate * value).detach().cpu().clone()
            for name, value in zip(names, action_values)
        },
        "signed_margins": signed_margins.detach().cpu().clone(),
        "scalar_record": cpu_nested(scalar_record),
        "rng_state": rng_state_payload(),
    }
    shadow = getattr(optimizer, "log_second_shadow", None)
    if shadow is not None:
        payload["second_moment_log_raw"] = {
            name: value.detach().cpu().clone()
            for name, value in zip(names, shadow.preview(gradients))
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_loss_checkpoint_manifest(
    path: Path,
    *,
    signature: dict,
    requested_losses: tuple[float, ...],
    records: list[dict],
) -> None:
    write_json(path, {
        "format_version": 2,
        "signature": signature,
        "requested_losses": list(requested_losses),
        "checkpoint_count": len(records),
        "expected_checkpoint_count": len(requested_losses),
        "complete": len(records) == len(requested_losses),
        "checkpoints": records,
    })


def save_npz_records(
    scalar_path: Path,
    coordinate_path: Path,
    scalar_records: dict,
    coordinate_records: dict,
    selection: dict | None,
) -> None:
    scalar_arrays = {}
    for key, values in scalar_records.items():
        if key in {"parameter_sketch", "gradient_sketch"}:
            if not values:
                scalar_arrays[key] = np.empty((0, 0), dtype=np.float64)
                continue
            scalar_arrays[key] = np.stack([
                value.detach().cpu().numpy() for value in values
            ])
        else:
            scalar_arrays[key] = np.asarray(
                values,
                dtype=(
                    np.int64
                    if (
                        key in ("step", "moment_step") or key.endswith("_count")
                        or key.endswith("_index") or key.endswith("_argmax")
                    )
                    else np.float64
                ),
            )
    np.savez_compressed(scalar_path, **scalar_arrays)
    coordinate_arrays = {
        "step": np.asarray(coordinate_records["step"], dtype=np.int64),
        "loss": np.asarray(coordinate_records["loss"], dtype=np.float64),
    }
    if selection is not None:
        for layer in selection:
            coordinate_arrays[f"{layer}_indices"] = (
                selection[layer]["indices"].detach().cpu().numpy()
            )
            coordinate_arrays[f"{layer}_sample_type"] = selection[layer]["sample_type"]
            for key, values in coordinate_records[layer].items():
                coordinate_arrays[f"{layer}_{key}"] = np.stack([
                    value.detach().cpu().numpy() for value in values
                ])
    np.savez_compressed(coordinate_path, **coordinate_arrays)


def crossing_steps(records: dict[str, list]) -> dict[str, int | None]:
    losses = np.asarray(records["loss"], dtype=float)
    steps = np.asarray(records["step"], dtype=int)
    return {
        f"{threshold:g}": (
            None if not np.any(losses <= threshold)
            else int(steps[np.flatnonzero(losses <= threshold)[0]])
        )
        for threshold in LOSS_THRESHOLDS
    }


def theory_metadata(activation_power: float, schedule_power: float, affine_layer_count: int = 3) -> dict:
    if not -1 <= schedule_power <= 1:
        return {"affine_layer_count": affine_layer_count,
                "homogeneity_degree": float(affine_layer_count),
                "activation_power": activation_power, "schedule_power": schedule_power,
                "regime": "unavailable", "prediction_kind": "theory guide implemented only for powers in [-1,1]"}
    return theory_prediction(
        activation_power,
        schedule_power,
        affine_layer_count=affine_layer_count,
    )


def run_one(
    *,
    args: argparse.Namespace,
    spec: dict,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    if x.ndim < 2 or len(x) != len(y) or not len(x):
        raise ValueError("nonempty inputs and matching labels are required")
    if not torch.all((y == -1) | (y == 1)):
        raise ValueError("labels must be -1 or +1")
    provenance = getattr(args, "source_provenance", None) or source_metadata()
    architecture = (x.flatten(1).shape[1], *args.hidden_widths, 1)
    layer_names = tuple(f"hidden{i+1}" for i in range(len(args.hidden_widths))) + ("output",)
    depth = len(architecture) - 1
    run_dir = args.output_dir / spec["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.json"
    scalar_path = run_dir / "trajectory_series.npz"
    coordinate_path = run_dir / "coordinate_series.npz"
    checkpoint_path = run_dir / "checkpoint.pt"
    anchor_path = run_dir / "frozen_metrics.pt"
    if not args.resume and any(p.exists() for p in (checkpoint_path, result_path)):
        raise FileExistsError(f"run artifacts already exist: {run_dir}; enable resume or choose another output directory")
    loss_checkpoint_losses = parse_full_checkpoint_losses(
        str(getattr(args, "full_checkpoint_losses", "")),
    )
    loss_checkpoint_dir = run_dir / "loss_checkpoints"
    loss_checkpoint_manifest_path = run_dir / "loss_checkpoints_manifest.json"
    if loss_checkpoint_losses:
        loss_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_checkpoint and (not args.resume or not checkpoint_path.exists()):
        raise RuntimeError(
            "--finalize_checkpoint requires --resume and an existing checkpoint at "
            f"{checkpoint_path}"
        )
    activation_power = float(spec["activation_power"])
    if not math.isclose(activation_power, 1.0):
        raise ValueError("the dedicated ReLU runner requires activation_power=1.0")
    schedule_power = float(spec["schedule_power"])
    learning_rate = float(spec["learning_rate"])
    theory = theory_metadata(activation_power, schedule_power, depth)
    beta1, beta2 = optimizer_betas(args, learning_rate)
    if not 0 <= beta1 < 1 or not 0 < beta2 < 1:
        raise ValueError("effective betas must satisfy 0<=beta1<1 and 0<beta2<1; rescaling may underflow or round to one")
    factory = lambda: HomogeneousReLU(
        input_dim=architecture[0], hidden_widths=args.hidden_widths,
    )
    state, initialization = make_scaled_initialization(
        factory,
        x[: min(8, len(x))],
        layer_scale=args.layer_scale,
        seed=args.seed,
        initialization_dtype=torch.float64,
        kind=args.initialization_kind, std=args.initialization_std,
    )
    model = factory().to(device=x.device, dtype=x.dtype)
    cast_state = {}
    for name, value in state.items():
        converted = value.to(device=x.device, dtype=x.dtype)
        original_nonzero = (value != 0).to(device=x.device)
        if not torch.isfinite(converted).all() or torch.any(original_nonzero & (converted == 0)):
            raise ValueError(
                f"initialization of {name} is not representable in {x.dtype}: "
                "casting produced nonfinite values or erased nonzero weights; "
                "change initialization scale or use float64"
            )
        cast_state[name] = converted
    model.load_state_dict(cast_state)
    optimizer = InspectableAdam(model.parameters(), beta1, beta2, 0.0)
    shadow = None
    incoming_requested = bool(getattr(args, "log_incoming_every_step", False))
    if incoming_requested and not getattr(args, "log_second_moment_shadow", False):
        raise ValueError("--log_incoming_every_step requires --log_second_moment_shadow")
    shadow_path = run_dir / "log_second_moment_shadow.jsonl"
    if getattr(args, "log_second_moment_shadow", False):
        if x.dtype != torch.float64 or not 0.0 < beta2 < 1.0:
            raise ValueError("log second-moment comparison requires float64 and 0 < beta2 < 1")
        shadow = LogSecondMomentShadow(model.parameters(), beta2)
        optimizer.log_second_shadow = shadow
    parameter_names = tuple(name for name, _ in model.named_parameters())
    expected_names = tuple(f"{layer}.weight" for layer in layer_names)
    if parameter_names != expected_names or any(
        module.bias is not None
        for module in model.children()
    ):
        raise RuntimeError("dedicated model violated the bias-free layer schema")
    if optimizer.epsilon != 0.0:
        raise RuntimeError("dedicated experiment requires Adam epsilon exactly zero")
    groups = layer_parameter_indices(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    sketcher = CoordinateSketch(
        total_parameters, args.sketch_dimension, args.seed + 311, x.device,
    )
    repository = git_metadata(Path(__file__).resolve().parent)
    signature = {
        "run_name": str(spec["name"]),
        "seed": int(args.seed), "sample_count": int(len(x)),
        "per_class": int(args.per_class),
        "architecture": list(architecture),
        "affine_layer_count": depth, "homogeneity_degree": float(depth),
        "biases": False,
        "initialization_kind": args.initialization_kind,
        "initialization_std": args.initialization_std,
        "dataset_kind": args.dataset,
        "centered": args.center,
        "hidden_widths": list(args.hidden_widths), "layer_scale": float(args.layer_scale),
        "activation_power": activation_power,
        "schedule_power": schedule_power, "learning_rate": learning_rate,
        "beta1": beta1, "beta2": beta2, "epsilon": 0.0,
        "beta_policy": str(args.beta_policy),
        "dtype": args.dtype,
        "device": str(x.device),
        "device_type": x.device.type,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "target_loss": float(args.target_loss),
        "max_steps": int(args.max_steps),
        "stop_at_target": bool(args.stop_at_target),
        "selection_loss": float(args.selection_loss),
        "coordinate_count": int(args.coordinate_count),
        "record_every": int(args.record_every),
        "coordinate_every": int(args.coordinate_every),
        "expensive_every": int(args.expensive_every),
        "checkpoint_every": int(args.checkpoint_every),
        "checkpoint_every_seconds": float(args.checkpoint_every_seconds),
        "progress_every": int(args.progress_every),
        "sketch_dimension": int(args.sketch_dimension),
        "threads": int(args.threads),
        "spectral_norm_steps": int(args.spectral_norm_steps),
        "nuclear_norm_steps": int(args.nuclear_norm_steps),
        "metric_anchors": list(ANCHORS),
        "dataset_sha256": str(args.dataset_sha256),
        "initialization_sha256": tensor_state_sha256(state),
        "git_commit": repository["commit"],
        "source_sha256": {name: provenance["module_sha256"][name]
                          for name in ("mnist/runner.py", "mnist/models.py", "mnist/data.py",
                                       "mnist/matrix_norms.py", "mnist/coordinate_diagnostics.py",
                                       "mnist/log_second_moment.py", "mnist/incoming_trace.py",
                                       "optim/adam.py")},
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "full_checkpoint_losses": list(loss_checkpoint_losses),
    }
    if shadow is not None:
        signature["log_second_moment_shadow"] = {
            "version": 1,
            "primary_optimizer": "unchanged ordinary float64 epsilon-zero Adam",
            "ordinary_moment_preview": "same mul_/add_/addcmul_ operation sequence as actual optimizer",
            "shadow_recurrence": "logaddexp(log(beta2)+log_v, log(1-beta2)+2*log(abs(g)))",
            "transition": "raw optimizer v after t updates -> v after t+1 updates using gradient at theta_t",
            "transformed_relative_l1_increment": "norm(v_next/beta2-v_old,1)/norm(v_old,1)",
            "update_reference": "log-v denominator with the same ordinary first moment and bias corrections",
            "logging": "every scalar row; endpoint transition is prospective; initial zero denominator undefined",
            "source_sha256": {
                name: provenance["module_sha256"][f"mnist/{name}"]
                for name in ("runner.py", "log_second_moment.py")
            },
        }
    if incoming_requested:
        signature["incoming_every_step"] = {
            "version": 1, "buffer_rows": 1024,
            "logarithm": "natural; null denotes exact zero magnitude",
            "alignment": "loss(theta_t) and prospective D_t for t -> t+1; terminal row included",
            "formula": "log(1-beta2)-(t+1)*log(beta2)+logsumexp(2*log(abs(g_t)))",
            "layers": list(layer_names),
            "source_sha256": {
                name: provenance["module_sha256"][f"mnist/{name}"]
                for name in ("runner.py",
                             "log_second_moment.py", "incoming_trace.py")
            },
        }
    if args.resume and result_path.exists():
        prior_result = json.loads(result_path.read_text(encoding="utf-8"))
        if completed_result_matches_request(
            prior_result,
            signature,
            allow_target_extension=bool(args.allow_target_extension),
            allow_dtype_promotion=bool(args.allow_dtype_promotion),
            allow_legacy_diagnostic_upgrade=bool(
                args.allow_legacy_diagnostic_upgrade
            ),
        ):
            required_artifacts = (
                checkpoint_path, scalar_path, coordinate_path, anchor_path,
            )
            if loss_checkpoint_losses:
                required_artifacts += (loss_checkpoint_manifest_path,)
            if incoming_requested:
                required_artifacts += (
                    run_dir / "incoming_second_moment.jsonl",
                    run_dir / "incoming_second_moment.meta.json",
                )
            missing = [str(path) for path in required_artifacts if not path.exists()]
            if missing:
                raise RuntimeError(
                    "completed result is missing required artifacts: " + ", ".join(missing)
                )
            print(
                f"Already complete {spec['name']}: "
                f"step={int(prior_result['final_step']):,}, "
                f"loss={float(prior_result['final_loss']):.4e}",
                flush=True,
            )
            return prior_result
    scalar_records = initialize_scalar_records(layer_names=layer_names)
    coordinate_records = initialize_coordinate_records(layer_names)
    selection = None
    selection_step = None
    selection_actual_loss = None
    parameters_list = list(model.parameters())
    scores = {
        layer: torch.zeros(
            sum(parameters_list[index].numel() for index in indices),
            dtype=x.dtype, device=x.device,
        )
        for layer, indices in groups.items()
    }
    nonzero_counts = {
        layer: torch.zeros_like(score, dtype=torch.int32)
        for layer, score in scores.items()
    }
    score_observations = 0
    resume_migrations: tuple[str, ...] = ()
    anchors: dict[str, torch.Tensor] = {}
    anchor_momenta: dict[str, torch.Tensor] = {}
    anchor_updates: dict[str, torch.Tensor] = {}
    anchor_second_moments: dict[str, torch.Tensor] = {}
    anchor_steps: dict[str, int] = {}
    anchor_losses: dict[str, float] = {}
    cumulative_learning_rate = 0.0
    elapsed_seconds_prior = 0.0
    loss_checkpoint_records: list[dict] = []
    incoming_saved_signature = None
    if args.resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        try:
            resume_migrations = validate_resume_signature(
                payload.get("signature"),
                signature,
                allow_target_extension=bool(args.allow_target_extension),
                allow_dtype_promotion=bool(args.allow_dtype_promotion),
                allow_legacy_diagnostic_upgrade=bool(
                    args.allow_legacy_diagnostic_upgrade
                ),
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"checkpoint signature mismatch at {checkpoint_path}: {error}"
            ) from error
        model.load_state_dict(payload["model_state"])
        incoming_saved_signature = payload["signature"]
        restore_optimizer(optimizer, payload["optimizer_state"], allow_dtype_promotion=args.allow_dtype_promotion)
        scalar_records = payload["scalar_records"]
        upgrade_legacy_scalar_records(scalar_records, resume_migrations)
        coordinate_records = payload["coordinate_records"]
        selection = payload["selection"]
        selection_step = payload["selection_step"]
        selection_actual_loss = payload["selection_actual_loss"]
        scores = device_nested(payload["scores"], x.device)
        nonzero_counts = device_nested(payload["nonzero_counts"], x.device)
        score_observations = int(payload["score_observations"])
        anchors = {
            key: value.to(device=x.device, dtype=x.dtype)
            for key, value in payload["anchors"].items()
        }
        anchor_momenta = {
            key: value.to(device=x.device, dtype=x.dtype)
            for key, value in payload.get("anchor_momenta", {}).items()
        }
        anchor_updates = {
            key: value.to(device=x.device, dtype=x.dtype)
            for key, value in payload.get("anchor_updates", {}).items()
        }
        anchor_second_moments = {
            key: value.to(device=x.device, dtype=x.dtype)
            for key, value in payload.get("anchor_second_moments", {}).items()
        }
        if set(anchor_second_moments) != set(anchors):
            raise RuntimeError(
                "checkpoint is missing raw second-moment anchor buffers"
            )
        anchor_steps = payload["anchor_steps"]
        anchor_losses = payload["anchor_losses"]
        cumulative_learning_rate = float(
            payload.get("cumulative_learning_rate", 0.0)
        )
        elapsed_seconds_prior = float(payload.get("elapsed_seconds_total", 0.0))
        loss_checkpoint_records = list(payload.get("loss_checkpoint_records", []))
        recorded_indices: set[int] = set()
        for record in loss_checkpoint_records:
            index = int(record["index"])
            if index in recorded_indices or not 0 <= index < len(loss_checkpoint_losses):
                raise RuntimeError("invalid loss-checkpoint record in rolling checkpoint")
            recorded_indices.add(index)
            expected = loss_checkpoint_losses[index]
            if not math.isclose(
                float(record["requested_loss"]), expected, rel_tol=1e-12,
            ):
                raise RuntimeError("loss-checkpoint threshold mismatch on resume")
            snapshot_path = loss_checkpoint_dir / str(record["filename"])
            if not snapshot_path.is_file() or snapshot_path.stat().st_size <= 0:
                raise RuntimeError(f"missing recorded loss checkpoint: {snapshot_path}")
            if int(record.get("size_bytes", -1)) != snapshot_path.stat().st_size:
                raise RuntimeError(f"loss checkpoint size mismatch: {snapshot_path}")
            if str(record.get("sha256", "")) != file_sha256(snapshot_path):
                raise RuntimeError(f"loss checkpoint hash mismatch: {snapshot_path}")
        if "rng_state" not in payload:
            raise RuntimeError("checkpoint is missing exact-continuation RNG state")
        restore_rng_state(payload["rng_state"])
        print(f"Resuming {spec['name']} at step {optimizer.step_count:,}", flush=True)
    if (
        not args.finalize_checkpoint
        and
        optimizer.step_count >= args.max_steps
        and result_path.exists() and scalar_path.exists() and coordinate_path.exists()
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))

    incoming_trace = None
    if incoming_requested:
        incoming_trace = BufferedIncomingTrace(
            run_dir, signature=signature, layer_names=layer_names,
            resume_step=(optimizer.step_count if incoming_saved_signature is not None else None),
            saved_signature=incoming_saved_signature,
        )
    start_step = optimizer.step_count
    start_time = time.perf_counter()
    last_print = start_time
    last_checkpoint_time = start_time
    # Reconstruct against the current target, including a permitted smaller
    # target on resume. Every crossing is recorded even between log intervals.
    target_reached = any(
        math.isfinite(float(value)) and float(value) <= args.target_loss
        for value in scalar_records["loss"]
    )
    final_loss = math.nan
    completion_reason = "max_steps"
    numerical_failure = None
    recorded_dense_thresholds = {
        threshold for threshold in LOSS_THRESHOLDS
        if scalar_records["loss"]
        and min(float(value) for value in scalar_records["loss"]) <= threshold
    }

    def persist_checkpoint(
        destination: Path,
        records: list[dict] | None = None,
    ) -> None:
        nonlocal last_checkpoint_time
        if incoming_trace is not None:
            incoming_trace.flush()
        save_checkpoint(
            destination,
            signature=signature,
            provenance=provenance,
            model=model,
            optimizer=optimizer,
            scalar_records=scalar_records,
            coordinate_records=coordinate_records,
            selection=selection,
            selection_step=selection_step,
            selection_actual_loss=selection_actual_loss,
            scores=scores,
            nonzero_counts=nonzero_counts,
            score_observations=score_observations,
            anchors=anchors,
            anchor_momenta=anchor_momenta,
            anchor_updates=anchor_updates,
            anchor_second_moments=anchor_second_moments,
            anchor_steps=anchor_steps,
            anchor_losses=anchor_losses,
            cumulative_learning_rate=cumulative_learning_rate,
            elapsed_seconds_total=(
                elapsed_seconds_prior + time.perf_counter() - start_time
            ),
            loss_checkpoint_records=(
                loss_checkpoint_records if records is None else records
            ),
        )
        last_checkpoint_time = time.perf_counter()

    if shadow is not None:
        # Rebuild the live stream from the durable recovery point on resume.
        shadow_path.write_text("", encoding="utf-8")
        shadow_keys = [key for key in scalar_records if key.startswith("shadow_v_")]
        for index in range(len(scalar_records["step"])):
            write_shadow_record(shadow_path, {
                key: scalar_records[key][index]
                for key in ("step", "loss", "learning_rate", *shadow_keys)
            })
    while optimizer.step_count <= args.max_steps:
        step = optimizer.step_count
        model.zero_grad(set_to_none=True)
        loss_tensor, logits = logistic_loss_and_logits(model, x, y)
        if not bool(torch.isfinite(loss_tensor)):
            final_loss = None
            completion_reason = "numerical_failure"
            numerical_failure = f"non-finite loss at step {step}"
            break
        loss_value = float(loss_tensor.detach().item())
        signed = y * logits.detach()
        error = float((signed <= 0).to(torch.float64).mean().item())
        raw_margin = float(signed.min().item())
        loss_tensor.backward()
        gradients = current_gradients(model)
        if not math.isfinite(loss_value) or not all(bool(torch.isfinite(g).all()) for g in gradients):
            final_loss = loss_value if math.isfinite(loss_value) else None
            completion_reason = "numerical_failure"
            numerical_failure = f"non-finite loss/gradient at step {step}"
            break
        if loss_value <= 0.0:
            final_loss = loss_value
            completion_reason = "numerical_failure"
            numerical_failure = f"positive logistic loss underflowed to zero at step {step}"
            break
        if incoming_trace is not None:
            incoming_trace.append(
                step, math.log(loss_value) if loss_value > 0.0 else -math.inf,
                shadow.prepare_incoming(gradients, step=step),
            )
        first_next, second_next, first_hat, root_second = next_moment_vectors(
            optimizer, gradients,
        )
        first_raw = torch.cat([value.reshape(-1) for value in first_next])
        second_raw = torch.cat([value.reshape(-1) for value in second_next])
        terminal_evaluation = (
            (loss_value <= args.target_loss and args.stop_at_target)
            or step == args.max_steps
            or (args.finalize_checkpoint and step == start_step)
            or STOP_REQUESTED
        )
        try:
            rate = schedule_learning_rate(learning_rate, schedule_power, step + 1)
        except (ValueError, OverflowError) as schedule_error:
            if terminal_evaluation:
                # No further update is requested. The loss and prospective
                # moments remain defined; an unrepresentable UNUSED eta is
                # NaN, as are diagnostics depending on that applied action.
                rate = math.nan
            else:
                final_loss = loss_value
                completion_reason = "numerical_failure"
                numerical_failure = str(schedule_error)
                break
        recorded_indices = {
            int(record["index"]) for record in loss_checkpoint_records
        }
        checkpoint_crossings = [
            index for index, threshold in enumerate(loss_checkpoint_losses)
            if index not in recorded_indices and loss_value <= threshold
        ]
        dense_crossings = [
            threshold for threshold in LOSS_THRESHOLDS
            if threshold not in recorded_dense_thresholds and loss_value <= threshold
        ]
        diagnostic_due = (
            step % args.record_every == 0 or step == args.max_steps
            or loss_value <= args.target_loss or bool(checkpoint_crossings)
            or bool(dense_crossings) or STOP_REQUESTED
        )
        scalar_already_recorded = bool(
            scalar_records["step"]
            and int(scalar_records["step"][-1]) == step
        )
        if selection is None and diagnostic_due and not scalar_already_recorded:
            score_observations = update_selection_scores(
                scores, nonzero_counts, score_observations, gradients, groups,
            )
            if loss_value <= args.selection_loss:
                selection = choose_coordinates(
                    scores=scores,
                    nonzero_counts=nonzero_counts,
                    observations=score_observations,
                    count=args.coordinate_count,
                    seed=args.seed + 701,
                )
                selection_step = int(step)
                selection_actual_loss = loss_value
                print(
                    f"[{spec['name']}] selected coordinates at step={step:,}, "
                    f"loss={loss_value:.3e}",
                    flush=True,
                )
        newly_frozen = False
        for anchor in ANCHORS:
            tag = anchor_tag(anchor)
            if tag not in anchors and loss_value <= anchor:
                anchors[tag] = root_second.detach().clone()
                anchor_momenta[tag] = first_raw.detach().clone()
                anchor_updates[tag] = -masked_ratio(first_hat, root_second).clone()
                anchor_second_moments[tag] = second_raw.detach().clone()
                anchor_steps[tag] = int(step)
                anchor_losses[tag] = loss_value
                newly_frozen = True
                print(
                    f"[{spec['name']}] froze {tag} at step={step:,}, "
                    f"loss={loss_value:.3e}",
                    flush=True,
                )
        diagnostic_due = diagnostic_due or newly_frozen
        if diagnostic_due and not scalar_already_recorded:
            gradient = torch.cat([value.reshape(-1) for value in gradients])
            parameters = torch.cat([
                parameter.detach().reshape(-1) for parameter in model.parameters()
            ])
            expensive_due = (
                step % args.expensive_every == 0 or newly_frozen
                or bool(checkpoint_crossings) or bool(dense_crossings)
                or loss_value <= args.target_loss or STOP_REQUESTED
            )
            append_scalar_record(
                scalar_records if shadow is None else {
                    key: values for key, values in scalar_records.items()
                    if not key.startswith("shadow_v_")
                },
                step=step,
                loss=loss_value,
                error=error,
                learning_rate=rate,
                raw_margin=raw_margin,
                parameters=parameters,
                gradient=gradient,
                root_second=root_second,
                first_hat=first_hat,
                first_raw=first_raw,
                second_raw=second_raw,
                anchors=anchors,
                degree=model.homogeneity_degree,
                parameter_sketch=sketcher(parameters),
                gradient_sketch=sketcher(gradient),
                model=model,
                gradients=gradients,
                signed_margins=signed,
                cumulative_learning_rate=cumulative_learning_rate,
                elapsed_seconds_total=(
                    elapsed_seconds_prior + time.perf_counter() - start_time
                ),
                beta1=beta1,
                beta2=beta2,
                anchor_momenta=anchor_momenta,
                anchor_updates=anchor_updates,
                anchor_second_moments=anchor_second_moments,
                spectral_norm_steps=args.spectral_norm_steps,
                nuclear_norm_steps=args.nuclear_norm_steps,
                expensive_due=expensive_due,
            )
            if shadow is not None:
                shadow_record = shadow.diagnostics(optimizer, gradients)
                for key, value in shadow_record.items():
                    if not key.startswith("shadow_v_"):
                        raise RuntimeError(f"invalid shadow diagnostic name: {key}")
                    values = scalar_records.setdefault(key, [])
                    if len(values) != len(scalar_records["step"]) - 1:
                        raise RuntimeError(f"shadow diagnostic length mismatch: {key}")
                    values.append(value)
                write_shadow_record(shadow_path, {
                    "step": step, "loss": loss_value, "learning_rate": rate,
                    **shadow_record,
                })
            coordinate_due = (
                step % args.coordinate_every == 0 or newly_frozen
                or bool(checkpoint_crossings) or bool(dense_crossings)
                or loss_value <= args.target_loss or STOP_REQUESTED
            )
            coordinate_already_recorded = bool(
                coordinate_records["step"]
                and int(coordinate_records["step"][-1]) == step
            )
            if (
                selection is not None and coordinate_due
                and not coordinate_already_recorded
            ):
                append_coordinate_record(
                    coordinate_records,
                    step=step,
                    loss=loss_value,
                    selection=selection,
                    model=model,
                    optimizer=optimizer,
                    first_hat=first_hat,
                    root_second=root_second,
                    second_next=second_next,
                    gradients=gradients,
                    groups=groups,
                )
            recorded_dense_thresholds.update(dense_crossings)
        for index, threshold in enumerate(loss_checkpoint_losses):
            if index in recorded_indices or loss_value > threshold:
                continue
            filename = loss_checkpoint_filename(index, threshold)
            snapshot_path = loss_checkpoint_dir / filename
            record = {
                "index": index,
                "requested_loss": float(threshold),
                "actual_loss": loss_value,
                "step": int(step),
                "filename": filename,
            }
            scalar_record = {
                key: values[-1] for key, values in scalar_records.items()
            }
            save_analysis_snapshot(
                snapshot_path,
                signature=signature,
                threshold=threshold,
                step=step,
                loss=loss_value,
                raw_margin=raw_margin,
                learning_rate=rate,
                cumulative_learning_rate=cumulative_learning_rate,
                model=model,
                optimizer=optimizer,
                gradients=gradients,
                first_raw=first_next,
                second_raw=second_next,
                first_hat=first_hat,
                root_second=root_second,
                signed_margins=signed,
                scalar_record=scalar_record,
                provenance=provenance,
            )
            record["size_bytes"] = int(snapshot_path.stat().st_size)
            record["sha256"] = file_sha256(snapshot_path)
            loss_checkpoint_records = sorted(
                [*loss_checkpoint_records, record], key=lambda item: int(item["index"]),
            )
            # Immediately advance the rolling recovery point so a preemption
            # cannot separate a durable snapshot from its bookkeeping.
            persist_checkpoint(checkpoint_path)
            write_loss_checkpoint_manifest(
                loss_checkpoint_manifest_path,
                signature=signature,
                requested_losses=loss_checkpoint_losses,
                records=loss_checkpoint_records,
            )
            print(
                f"[{spec['name']}] saved full loss checkpoint {index + 1}/"
                f"{len(loss_checkpoint_losses)} at step={step:,}, "
                f"loss={loss_value:.3e}",
                flush=True,
            )
            recorded_indices.add(index)
        reached_now = loss_value <= args.target_loss
        target_reached = target_reached or reached_now
        if (
            (reached_now and args.stop_at_target)
            or step == args.max_steps
            or (args.finalize_checkpoint and step == start_step)
            or STOP_REQUESTED
        ):
            final_loss = loss_value
            completion_reason = (
                "target_loss" if reached_now and args.stop_at_target
                else "checkpoint_finalize" if args.finalize_checkpoint
                else "signal" if STOP_REQUESTED
                else "max_steps"
            )
            break
        try:
            optimizer.step(rate)
        except NumericalError as error:
            final_loss = loss_value
            completion_reason = "numerical_failure"
            numerical_failure = str(error)
            break
        if shadow is not None:
            shadow.advance(gradients)
        cumulative_learning_rate += rate
        checkpoint_step_due = (
            args.checkpoint_every > 0
            and optimizer.step_count % args.checkpoint_every == 0
        )
        checkpoint_time_due = (
            args.checkpoint_every_seconds > 0.0
            and time.perf_counter() - last_checkpoint_time
            >= args.checkpoint_every_seconds
        )
        if checkpoint_step_due or checkpoint_time_due:
            persist_checkpoint(checkpoint_path)
        if args.progress_every > 0 and optimizer.step_count % args.progress_every == 0:
            now = time.perf_counter()
            print(
                f"[{spec['name']}] step={optimizer.step_count:7d} "
                f"loss={loss_value:.4e} err={error:.3f} "
                f"lr={rate:.3e} block={now-last_print:.1f}s",
                flush=True,
            )
            last_print = now

    elapsed = time.perf_counter() - start_time
    persist_checkpoint(checkpoint_path)
    if loss_checkpoint_losses:
        write_loss_checkpoint_manifest(
            loss_checkpoint_manifest_path,
            signature=signature,
            requested_losses=loss_checkpoint_losses,
            records=loss_checkpoint_records,
        )
    save_npz_records(
        scalar_path, coordinate_path, scalar_records, coordinate_records, selection,
    )
    frozen_payload = {
        "metrics": cpu_nested(anchors),
        "momenta": cpu_nested(anchor_momenta),
        "updates": cpu_nested(anchor_updates),
        "second_moments_raw": cpu_nested(anchor_second_moments),
        "steps": anchor_steps,
        "losses": anchor_losses,
        "timing": (
            "metrics are bias-corrected roots, momenta are raw first moments, "
            "and second_moments_raw are raw Adam v vectors; all include the "
            "current gradient and precede the parameter update"
        ),
    }
    anchor_temporary = anchor_path.with_suffix(anchor_path.suffix + ".tmp")
    torch.save(frozen_payload, anchor_temporary)
    anchor_temporary.replace(anchor_path)
    result = {
        "signature": signature,
        "initialization": {
            **initialization,
            "sha256": signature["initialization_sha256"],
            "fingerprint": signature["initialization_sha256"],
        },
        "repository": repository,
        "provenance": provenance,
        "hardware": hardware_metadata(x.device),
        "theory": theory,
        "selection_step": selection_step,
        "selection_actual_loss": selection_actual_loss,
        "coordinate_selection": None if selection is None else {
            layer: {
                "persistent_count": selection[layer]["persistent_count"],
                "random_count": selection[layer]["random_count"],
            }
            for layer in layer_names
        },
        "anchor_steps": anchor_steps,
        "anchor_losses": anchor_losses,
        "final_step": int(optimizer.step_count),
        "final_loss": None if final_loss is None else float(final_loss),
        "target_reached": bool(target_reached),
        "completion_reason": completion_reason,
        "numerical_failure": numerical_failure,
        "numerical_observations": {
            "maximum_recorded_gradient_square_underflow_count": max(scalar_records["gradient_square_underflow_count"], default=0),
            "maximum_recorded_second_moment_subnormal_count": max(scalar_records["second_moment_subnormal_count"], default=0),
            "scope": "sampled scalar rows; zero does not certify unrecorded steps",
        },
        "resume_migrations": list(resume_migrations),
        "recorded_crossing_steps": crossing_steps(scalar_records),
        "elapsed_seconds_this_invocation": float(elapsed),
        "elapsed_seconds_total": float(elapsed_seconds_prior + elapsed),
        "steps_per_second_this_invocation": float(
            (optimizer.step_count - start_step) / elapsed if elapsed > 0 else 0.0
        ),
        "trajectory_series_file": str(scalar_path.resolve()),
        "coordinate_series_file": str(coordinate_path.resolve()),
        "frozen_metrics_file": str(anchor_path.resolve()),
        "resource_usage": {
            "peak_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if x.device.type == "cuda" else 0
            ),
            "peak_cuda_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if x.device.type == "cuda" else 0
            ),
            "peak_host_rss_bytes": peak_host_rss_bytes(),
            "filesystem_free_bytes": int(
                shutil.disk_usage(args.output_dir).free
            ),
        },
    }
    if loss_checkpoint_losses:
        result.update({
            "loss_checkpoint_manifest_file": str(
                loss_checkpoint_manifest_path.resolve()
            ),
            "loss_checkpoint_count": len(loss_checkpoint_records),
            "expected_loss_checkpoint_count": len(loss_checkpoint_losses),
        })
    artifact_paths = (scalar_path, coordinate_path, anchor_path, checkpoint_path)
    if incoming_trace is not None:
        result["incoming_second_moment_file"] = str(incoming_trace.path.resolve())
        result["incoming_second_moment_metadata_file"] = str(incoming_trace.metadata_path.resolve())
        result["incoming_second_moment_rows"] = int(incoming_trace.rows_written)
        artifact_paths += (incoming_trace.path, incoming_trace.metadata_path)
    result["artifacts"] = {
        path.name: {
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in artifact_paths
    }
    write_json(result_path, result)
    print(
        f"Completed {spec['name']}: step={optimizer.step_count:,}, "
        f"loss={final_loss}, reason={completion_reason}, elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    args = parse_args(argv)
    provenance = source_metadata()
    args.source_provenance = provenance
    installed_signals = install_signal_handlers()
    full_checkpoint_losses = parse_full_checkpoint_losses(
        args.full_checkpoint_losses,
    )
    if (
        full_checkpoint_losses
        and args.stop_at_target
        and min(full_checkpoint_losses) < args.target_loss
    ):
        raise ValueError(
            "full checkpoint losses below target_loss cannot be reached when "
            "--stop_at_target is enabled"
        )
    if (
        args.max_steps <= 0 or args.record_every <= 0
        or args.coordinate_every <= 0 or args.expensive_every <= 0
        or args.coordinate_count <= 0 or args.spectral_norm_steps <= 0
        or args.nuclear_norm_steps <= 0 or args.checkpoint_every < 0
        or args.checkpoint_every_seconds < 0.0 or args.progress_every < 0
        or args.sketch_dimension <= 0 or args.threads <= 0
    ):
        raise ValueError("invalid horizon or diagnostic interval")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_default_dtype(dtype)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "synthetic":
        images, labels01 = synthetic_data(args.per_class, args.input_dim, args.data_seed)
    else:
        images, labels01 = balanced_mnist(args.data_dir, train=True, per_class=args.per_class, download=args.download)
    architecture = (images[0].numel(), *args.hidden_widths, 1)
    depth = len(architecture) - 1
    args.dataset_sha256 = tensor_state_sha256({
        "images": images,
        "labels01": labels01,
    })
    if args.center:
        images = images - images.mean(dim=0, keepdim=True)
    x = images.to(device=device, dtype=dtype)
    y = labels01.mul(2).sub(1).to(device=device, dtype=dtype)
    specs = args.specs
    if any(not math.isclose(spec["activation_power"], 1.0) for spec in specs):
        raise ValueError("the dedicated ReLU runner requires activation_power=1.0")
    results = {}
    for seed in args.sweep_seeds:
        for beta2 in args.sweep_beta2_values:
            for spec in specs:
                run_args = copy.copy(args)
                run_args.seed = seed
                run_args.beta2_reference = beta2
                run_args.max_steps = int(spec.get("max_steps", args.max_steps))
                prefix = f"seed_{seed}/beta2_{beta2!r}" if args.sweep_enabled else ""
                run_args.output_dir = args.output_dir / prefix
                key = f"{prefix}/{spec['name']}" if prefix else spec["name"]
                results[key] = run_one(args=run_args, spec=spec, x=x, y=y)
                if STOP_REQUESTED:
                    break
            if STOP_REQUESTED:
                break
        if STOP_REQUESTED:
            break
    write_json(args.output_dir / "experiment_summary.json", {
        "design": {
            "dataset": args.dataset,
            "labels": "even (-1), odd (+1) for MNIST; paired synthetic labels otherwise",
            "sample_count": int(len(x)),
            "centered_by_private_training_mean": args.center,
            "hidden_widths": list(args.hidden_widths),
            "architecture": list(architecture),
            "affine_layer_count": depth,
            "homogeneity_degree": float(depth),
            "layer_scale": float(args.layer_scale),
            "biases": False,
            "network": "bias_free_relu",
            "hidden_activation": "relu",
            "dtype": args.dtype,
            "device": str(device),
            "target_loss": float(args.target_loss),
            "max_steps": int(args.max_steps),
            "metric_anchors": list(ANCHORS),
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "dense_record_every": int(args.record_every),
            "incoming_every_step": bool(args.log_incoming_every_step),
            "coordinate_record_every": int(args.coordinate_every),
            "expensive_record_every": int(args.expensive_every),
            "checkpoint_every_steps": int(args.checkpoint_every),
            "checkpoint_every_seconds": float(args.checkpoint_every_seconds),
            "installed_signal_handlers": list(installed_signals),
            "layer_linf_balance_definition": (
                "max complete weight-block linf across all layers "
                "/ min complete weight-block linf across all layers"
            ),
            "linf_dual_alignment_definition": (
                "<-g/||g||_1, descent_action/||descent_action||_inf>"
            ),
            "msp_dual_norm_definition": (
                "sum of approximate nuclear norms of augmented affine-layer "
                "gradient matrices"
            ),
            "snapshot_timing": (
                "pre-update theta_t and g_t; moment/action include g_t; applied "
                "update uses eta_(t+1)"
            ),
            "beta_policy": str(args.beta_policy),
            "beta_values": [
                float(args.beta1_reference), float(args.beta2_reference),
            ],
            "beta_scaling": (
                "fixed beta references for every learning rate"
                if args.beta_policy == "fixed"
                else "beta_i(eta)=beta_i(reference)^(reference_eta/eta)"
            ),
            "coordinate_diagnostic": (
                "rho=Delta log(beta2^(-t)V_t), raw V; rho/(-log beta2)"
            ),
            "second_moment_anchor_diagnostic": (
                "cosine of the current raw Adam v vector with the raw v vector "
                "frozen at each loss anchor"
            ),
        },
        "provenance": provenance,
        "sweep": {"seeds": args.sweep_seeds, "beta2_values": args.sweep_beta2_values},
        "results": results,
    })
    if any(result["completion_reason"] == "numerical_failure" for result in results.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
