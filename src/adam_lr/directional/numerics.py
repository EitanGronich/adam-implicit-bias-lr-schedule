"""Conservative numerical health checks for positive product-model trajectories."""
from __future__ import annotations

import torch

from .config import NumericsConfig


def numerical_issue(theta: torch.Tensor, gradient: torch.Tensor, margins: torch.Tensor,
                    first: torch.Tensor, second: torch.Tensor, *, step: int,
                    beta2: float, config: NumericsConfig) -> dict | None:
    """Return a concrete first failure; zero initial moment buffers are expected.

    All exact gradient coordinates are strictly negative in this positive sector.
    A represented zero therefore means loss/gradient underflow, unlike a dead ReLU.
    Individual exp(-z_T) terms may round to zero while total gradients remain well
    resolved; their frequency is separately logged and not called convergence.
    """
    for name, value in (("parameters", theta), ("gradient", gradient),
                        ("margins", margins), ("first_moment", first), ("second_moment", second)):
        mask = ~torch.isfinite(value)
        if mask.any():
            return {"kind": "nonfinite_" + name, "coordinates": int(mask.sum().item())}
    if (theta <= 0).any():
        return {"kind": "left_positive_parameter_sector", "coordinates": int((theta <= 0).sum().item())}
    if (gradient >= 0).any():
        return {"kind": "zero_or_nonnegative_gradient", "coordinates": int((gradient >= 0).sum().item())}
    square = gradient.square()
    if not torch.isfinite(square).all():
        return {"kind": "gradient_square_overflow", "coordinates": int((~torch.isfinite(square)).sum().item())}
    if (square == 0).any():
        return {"kind": "gradient_square_underflow", "coordinates": int((square == 0).sum().item())}
    increment = square * (1 - beta2)
    if (increment == 0).any():
        return {"kind": "second_moment_increment_underflow", "coordinates": int((increment == 0).sum().item())}
    if step > 0 and (second <= 0).any():
        return {"kind": "nonpositive_second_moment", "coordinates": int((second <= 0).sum().item())}
    if config.stop_on_subnormal:
        tiny = torch.finfo(theta.dtype).tiny
        for name, values in (("gradient_square", square), ("second_moment_increment", increment)):
            mask = (values > 0) & (values < tiny)
            if mask.any():
                return {"kind": "subnormal_" + name, "coordinates": int(mask.sum().item())}
        if step > 0:
            mask = (second > 0) & (second < tiny)
            if mask.any():
                return {"kind": "subnormal_second_moment", "coordinates": int(mask.sum().item())}
    if config.support_margin_cap > 0 and (margins[:, 0] > config.support_margin_cap).any():
        return {"kind": "support_margin_cap", "trajectories": int((margins[:, 0] > config.support_margin_cap).sum().item())}
    return None
