"""Inspectable, bias-corrected, discrete Adam shared by both experiments.

At update t>=1: g_t=grad(loss)(theta_{t-1}); m_t=beta1*m_{t-1}
+ (1-beta1)*g_t; v_t=beta2*v_{t-1}+(1-beta2)*g_t**2; and
theta_t=theta_{t-1}-eta_t*(m_t/(1-beta1**t))/(sqrt(v_t/(1-beta2**t))+eps).
There is no weight decay, gradient rescaling, or continuous-time integration.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class NumericalError(FloatingPointError):
    """The represented update can no longer be trusted as finite Adam."""


def logistic_loss_and_logits(model: nn.Module, x: torch.Tensor,
                             y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean binary logistic loss for one scalar logit and +/-1 label per row.

    A column-shaped model output is flattened deliberately: multiplying an
    (N,1) output by an (N,) label vector would otherwise form an (N,N) loss.
    """
    logits = model(x)
    if logits.numel() != x.shape[0] or y.numel() != x.shape[0]:
        raise ValueError("expected exactly one scalar logit and label per example")
    logits = logits.reshape(-1)
    labels = y.reshape(-1)
    return F.softplus(-labels * logits).mean(), logits


def schedule_learning_rate(initial_rate: float, power: float, step: int) -> float:
    """eta_t=initial_rate*t**(-power), with the first update numbered t=1.

    A positive power decays, zero is constant, and a negative power grows.
    The schedule multiplies the Adam update; it does not change the betas.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("Adam step must be a positive integer")
    if not math.isfinite(initial_rate) or initial_rate < 0:
        raise ValueError("initial_rate must be finite and nonnegative")
    if not math.isfinite(power):
        raise ValueError("power must be finite")
    try:
        rate = initial_rate * step ** (-power)
    except OverflowError as error:
        raise ValueError("learning-rate schedule overflowed") from error
    if not math.isfinite(rate) or (initial_rate > 0 and rate == 0):
        raise ValueError("learning-rate schedule overflowed or underflowed")
    return rate


class InspectableAdam:
    """Full-batch Adam with explicit moment buffers and numerical guards.

    Every parameter must receive a dense gradient every step. Exactly zero
    histories have update zero when epsilon=0. A nonzero gradient or momentum
    with zero denominator is a numerical failure, never a stationary point.
    Guards reject a bad proposal before committing any parameter or buffer.
    They cannot recover information already lost upstream in the loss or
    gradient computation; experiment runners also monitor those quantities.
    """

    def __init__(self, parameters: Iterable[nn.Parameter], beta1: float,
                 beta2: float, epsilon: float) -> None:
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta parameters must lie in [0, 1)")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("Adam epsilon must be finite and nonnegative")
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("InspectableAdam requires at least one parameter")
        if len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError("duplicate parameters would be updated twice")
        if any(not p.is_floating_point() for p in self.parameters):
            raise ValueError("Adam parameters must have real floating-point dtype")
        self.beta1, self.beta2, self.epsilon = float(beta1), float(beta2), float(epsilon)
        self.step_count = 0
        self.first = [torch.zeros_like(p) for p in self.parameters]
        self.second = [torch.zeros_like(p) for p in self.parameters]

    @torch.no_grad()
    def step(self, learning_rate: float) -> None:
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        step = self.step_count + 1
        correction1, correction2 = 1 - self.beta1 ** step, 1 - self.beta2 ** step
        proposals = []
        for index, (parameter, first, second) in enumerate(zip(
                self.parameters, self.first, self.second)):
            gradient = parameter.grad
            if gradient is None:
                raise ValueError("full-batch Adam requires a gradient for every parameter")
            if gradient.is_sparse:
                raise RuntimeError("InspectableAdam does not support sparse gradients")
            if not bool((torch.isfinite(parameter) & torch.isfinite(gradient)
                         & torch.isfinite(first) & torch.isfinite(second)
                         & (second >= 0)).all()):
                raise NumericalError(f"nonfinite parameter/gradient/moment or negative v at update {step}, tensor {index}")
            next_first = first.clone().mul_(self.beta1).add_(gradient, alpha=1-self.beta1)
            next_second = second.clone().mul_(self.beta2).addcmul_(gradient, gradient, value=1-self.beta2)
            numerator = next_first / correction1
            denominator = (next_second / correction2).sqrt().add_(self.epsilon)
            failed_zero = (denominator == 0) & ((gradient != 0) | (next_first != 0))
            if bool(failed_zero.any()):
                raise NumericalError(f"nonzero gradient/momentum has zero Adam denominator at update {step}, tensor {index}; second-moment underflow")
            update = torch.zeros_like(parameter)
            active = denominator > 0
            update[active] = numerator[active] / denominator[active]
            candidate = parameter.add(update, alpha=-learning_rate)
            if not bool((torch.isfinite(candidate) & torch.isfinite(next_first)
                         & torch.isfinite(next_second) & torch.isfinite(denominator)
                         & torch.isfinite(update)).all()):
                raise NumericalError(f"nonfinite Adam proposal at update {step}, tensor {index}")
            proposals.append((candidate, next_first, next_second))
        for parameter, first, second, (candidate, next_first, next_second) in zip(
                self.parameters, self.first, self.second, proposals):
            parameter.copy_(candidate)
            first.copy_(next_first)
            second.copy_(next_second)
        self.step_count = step

    @torch.no_grad()
    def flattened_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return m_hat_t, sqrt(v_hat_t), and the Adam denominator at theta_t."""
        if self.step_count == 0:
            zero = torch.cat([torch.zeros_like(p).reshape(-1) for p in self.parameters])
            return zero, zero.clone(), zero + self.epsilon
        c1, c2 = 1-self.beta1**self.step_count, 1-self.beta2**self.step_count
        first_hat = torch.cat([m.reshape(-1)/c1 for m in self.first])
        root_second = torch.cat([(v/c2).sqrt().reshape(-1) for v in self.second])
        return first_hat, root_second, root_second + self.epsilon

    def serializable_state(self, model: nn.Module) -> dict:
        named = list(model.named_parameters())
        if [id(p) for _, p in named] != [id(p) for p in self.parameters]:
            raise ValueError("model parameter order differs from optimizer parameter order")
        return {"step": self.step_count, "beta1": self.beta1, "beta2": self.beta2,
                "epsilon": self.epsilon,
                "first_moment": {name: m.detach().clone() for (name, _), m in zip(named, self.first)},
                "second_moment": {name: v.detach().clone() for (name, _), v in zip(named, self.second)}}
