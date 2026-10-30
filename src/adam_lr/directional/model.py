"""Width-1 product model, canonical initialization, and analytic gradients."""
from dataclasses import dataclass
import numpy as np
import torch

from .config import DirectionalConfig, ModelConfig


@dataclass(frozen=True)
class InitialState:
    theta: torch.Tensor
    swapped: torch.Tensor
    controls: torch.Tensor
    pi: torch.Tensor


def initialize(config: DirectionalConfig) -> InitialState:
    config.validate()
    model, initial = config.model, config.initialization
    z_min, z_max, pi_min, pi_max = initial.control_bounds(model)
    lower = np.array([z_min] + [pi_min] * (model.layers - 1))
    upper = np.array([z_max] + [pi_max] * (model.layers - 1))
    if initial.sampling == "grid":
        values = [initial.z_values] + [initial.pi_values] * (model.layers - 1)
        axes = [np.asarray(explicit, dtype=float) if explicit is not None else
                np.linspace(a, b, initial.grid_size) if initial.grid_size > 1 else np.array([(a+b)/2])
                for a, b, explicit in zip(lower, upper, values)]
        controls = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, model.layers)
    elif initial.sampling == "explicit":
        controls = np.asarray(initial.control_points, dtype=float)
    else:
        controls = np.random.default_rng(initial.seed).uniform(lower, upper, size=(initial.sample_count, model.layers))
    control = torch.tensor(controls, dtype=getattr(torch, config.run.dtype), device=config.run.device)
    factors = control[:, 1:]
    pi0 = model.Lambda / factors.prod(dim=-1)
    pi = torch.cat([pi0[:, None], factors], dim=-1)
    a = (1 + control[:, 0] / initial.rho ** model.layers) / model.M
    u = initial.rho * pi0[:, None] * torch.stack([a, 1 - model.B * a], dim=-1)
    theta = torch.cat([u, initial.rho * pi[:, 1:]], dim=-1)
    if not torch.isfinite(theta).all() or not (theta > 0).all():
        raise ValueError("initial parameters are not representable: narrow pi bounds or change dtype")
    count = len(theta)
    if initial.sectors == "both":
        theta = torch.cat([theta, torch.cat([theta[:, [1, 0]], theta[:, 2:]], dim=-1)])
        control, pi = torch.cat([control, control]), torch.cat([pi, pi])
        swapped = torch.arange(2 * count, device=theta.device) >= count
    elif initial.sectors == "swapped":
        theta = torch.cat([theta[:, [1, 0]], theta[:, 2:]], dim=-1)
        swapped = torch.ones(count, dtype=torch.bool, device=theta.device)
    else:
        swapped = torch.zeros(count, dtype=torch.bool, device=theta.device)
    return InitialState(theta, swapped, control, pi)


def dataset(swapped: torch.Tensor, config: ModelConfig, dtype: torch.dtype) -> torch.Tensor:
    """N independent pairs of examples, shape (N,2,2); swaps inputs with u."""
    base = torch.tensor([[config.B, 1.0], [config.M, 0.0]], dtype=dtype, device=swapped.device) / config.Lambda
    expanded = base.expand(len(swapped), -1, -1)
    return torch.where(swapped[:, None, None], expanded.flip(-1), expanded)


def margins(theta: torch.Tensor, swapped: torch.Tensor, config: ModelConfig) -> torch.Tensor:
    x = dataset(swapped, config, theta.dtype)
    linear = (theta[:, None, :2] * x).sum(dim=-1)
    return theta[:, 2:].prod(dim=-1, keepdim=True) * linear


def gradient_and_margins(theta: torch.Tensor, swapped: torch.Tensor,
                         config: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic gradient of each row's objective; valid even at zero b factors.

    Exclusion products avoid the incorrect/undefined shortcut product / b_j.
    No loss, gradient, parameter, or moment rescaling/clipping is used.
    """
    x = dataset(swapped, config, theta.dtype)
    linear = (theta[:, None, :2] * x).sum(dim=-1)
    b = theta[:, 2:]
    product = b.prod(dim=-1, keepdim=True)
    z = product * linear
    losses = torch.exp(-z)
    g_u = -product * (losses[:, :, None] * x).sum(dim=1)
    ones = torch.ones_like(b[:, :1])
    prefix = torch.cat([ones, b[:, :-1].cumprod(dim=-1)], dim=-1)
    suffix = torch.cat([b[:, 1:].flip(-1).cumprod(dim=-1).flip(-1), ones], dim=-1)
    g_b = -(losses * linear).sum(dim=-1, keepdim=True) * prefix * suffix
    return torch.cat([g_u, g_b], dim=-1), z


def normalize(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize safely without squaring huge or tiny coordinates first."""
    scale = vectors.abs().amax(dim=-1, keepdim=True)
    if not (scale > 0).all() or not torch.isfinite(scale).all():
        raise ValueError("direction is undefined for zero/nonfinite vectors")
    scaled = vectors / scale
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
