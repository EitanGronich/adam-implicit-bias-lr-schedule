"""Homogeneous model and the initialization family, without limit formulas.

Each row is an independent run with theta=(u1,u2,b1,...,b_{L-1}).
Its objective is the SUM exp(-z_S)+exp(-z_T), with both labels +1.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .config import DirectionalConfig, ModelConfig


@dataclass(frozen=True)
class InitialState:
    theta: torch.Tensor
    swapped: torch.Tensor
    controls: torch.Tensor  # columns s, ell_1, ..., ell_{L-1}
    beta: torch.Tensor  # beta_0,...,beta_{L-1}, product exactly C in real arithmetic


def initialize(config: DirectionalConfig) -> InitialState:
    config.validate()
    cfg, ini = config.model, config.initialization
    lower = np.array([ini.s_min] + [ini.ell_min] * (cfg.layers - 1))
    upper = np.array([ini.s_max] + [ini.ell_max] * (cfg.layers - 1))
    if ini.sampling == "grid":
        # One grid point means the midpoint, not an arbitrary lower corner.
        axes = [np.linspace(a, b, ini.grid_size) if ini.grid_size > 1 else np.array([(a+b)/2])
                for a, b in zip(lower, upper)]
        controls = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, cfg.layers)
    else:
        controls = np.random.default_rng(ini.seed).uniform(lower, upper, size=(ini.sample_count, cfg.layers))
    dtype = getattr(torch, config.run.dtype)
    control = torch.tensor(controls, dtype=dtype, device=config.run.device)
    beta0 = cfg.C ** (1 / cfg.layers) * torch.exp(-2 * control[:, 1:].sum(dim=-1) / cfg.layers)
    beta = torch.cat([beta0[:, None], beta0[:, None] * torch.exp(2 * control[:, 1:])], dim=-1)
    a = (1 + control[:, 0] / ini.rho ** cfg.layers) / cfg.M
    u = ini.rho * beta0[:, None] * torch.stack([a, 1 - cfg.B * a], dim=-1)
    theta = torch.cat([u, ini.rho * beta[:, 1:]], dim=-1)
    if not torch.isfinite(theta).all() or not (theta > 0).all():
        raise ValueError("initial parameters overflowed/underflowed: narrow ell bounds or change dtype")
    count = len(theta)
    if ini.sectors == "both":
        theta = torch.cat([theta, torch.cat([theta[:, [1, 0]], theta[:, 2:]], dim=-1)])
        control = torch.cat([control, control])
        beta = torch.cat([beta, beta])
        swapped = torch.arange(2 * count, device=theta.device) >= count
    elif ini.sectors == "swapped":
        theta = torch.cat([theta[:, [1, 0]], theta[:, 2:]], dim=-1)
        swapped = torch.ones(count, dtype=torch.bool, device=theta.device)
    else:
        swapped = torch.zeros(count, dtype=torch.bool, device=theta.device)
    return InitialState(theta, swapped, control, beta)


def dataset(swapped: torch.Tensor, config: ModelConfig, dtype: torch.dtype) -> torch.Tensor:
    """N independent pairs of examples, shape (N,2,2); swaps inputs with u."""
    base = torch.tensor([[config.B, 1.0], [config.M, 0.0]], dtype=dtype, device=swapped.device) / config.C
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


def angle_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Stable small-angle distance; arguments need not already be unit vectors."""
    a, b = normalize(first), normalize(second)
    return torch.rad2deg(2 * torch.atan2(torch.linalg.vector_norm(a-b, dim=-1),
                                        torch.linalg.vector_norm(a+b, dim=-1)))
