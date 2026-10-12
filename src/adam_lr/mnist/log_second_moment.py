"""Non-invasive log-domain second-moment shadow for epsilon-free Adam.

Rows describe the prospective transition from optimizer step t to t+1, using
the gradient at the current parameters.  All second moments here are raw,
without bias correction.  With c2=-log(beta2), the requested transformed
increment is ||v_next / beta2 - v_old||_1 / ||v_old||_1.

``*_direct_*`` measures represented states (and may show a cancellation
floor). ``*_recurrence_*`` uses the positive-increment identity
((1-beta2)/beta2) * sum(g**2) / sum(v_old), evaluated without forming g**2.
The latter is the exact-arithmetic recurrence quantity, rather than the
literal difference of rounded stored states.  ``raw`` and ``log`` identify
which second-moment history supplies the denominator.  Log scalar fields
use natural logarithms, retaining values outside float64's linear range.

Primary update errors compare the actual float64 ordinary update with
sign(m_hat)*exp(log|m_hat|-log(v_hat)/2), subtracting these computed updates
before calculating norms.  Secondary ``update_extended`` errors retain the
reference update in log magnitude, including when its explicit exponential
underflows or overflows.  Both use the SAME ordinary first moment numerator,
isolating second-moment arithmetic.  They do not certify the first moment,
backpropagation, or a rerun trajectory.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch


def _logsumexp(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values.reshape(-1), dim=0)


def _log_abs_difference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """log|exp(left)-exp(right)|, including two exact zero magnitudes."""
    maximum = torch.maximum(left, right)
    result = maximum + torch.log(-torch.expm1(-torch.abs(left - right)))
    both_zero = torch.isneginf(left) & torch.isneginf(right)
    return torch.where(both_zero, -torch.inf, result)


def _log_norm(log_magnitudes: torch.Tensor, order: int | float) -> torch.Tensor:
    if order == math.inf:
        return log_magnitudes.max()
    return _logsumexp(order * log_magnitudes) / order


def _ratio(log_numerator: torch.Tensor, log_denominator: torch.Tensor) -> tuple[float, float]:
    denominator = float(log_denominator.item())
    if not math.isfinite(denominator):
        return math.nan, math.nan
    logarithm = float((log_numerator - log_denominator).item())
    try:
        value = math.exp(logarithm)
    except OverflowError:
        value = math.inf
    return value, logarithm


def _flat(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values])


class LogSecondMomentShadow:
    """Track log(v) each step without changing Adam's actual update."""

    def __init__(self, parameters: Iterable[torch.nn.Parameter], beta2: float):
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("the shadow requires parameters")
        if not 0.0 < beta2 < 1.0:
            raise ValueError("log second-moment tracking requires 0 < beta2 < 1")
        if any(not value.is_floating_point() for value in self.parameters):
            raise ValueError("shadow parameters must have floating-point dtype")
        self.beta2 = float(beta2)
        self.log_beta2 = math.log(self.beta2)
        self.log_one_minus_beta2 = math.log1p(-self.beta2)
        self.step_count = 0
        self.log_second = [torch.full_like(value, -torch.inf) for value in self.parameters]
        self._incoming_cache = None

    def _validate_gradients(self, gradients: Sequence[torch.Tensor]) -> None:
        if len(gradients) != len(self.log_second):
            raise ValueError("gradient count does not match the shadow")
        for gradient, state in zip(gradients, self.log_second):
            if gradient.is_sparse:
                raise ValueError("the shadow requires dense gradients")
            if gradient.shape != state.shape or gradient.dtype != state.dtype or gradient.device != state.device:
                raise ValueError("gradient shape, dtype, or device does not match the shadow")

    @torch.no_grad()
    def preview(self, gradients: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Compute the next log-v without advancing or mutating any tensor."""
        self._validate_gradients(gradients)
        return [
            torch.logaddexp(
                old + self.log_beta2,
                2.0 * torch.log(gradient.abs()) + self.log_one_minus_beta2,
            )
            for old, gradient in zip(self.log_second, gradients)
        ]

    @torch.no_grad()
    def advance(self, gradients: Sequence[torch.Tensor]) -> None:
        if self._incoming_cache is None:
            self.log_second = self.preview(gradients)
        else:
            self._validate_incoming_cache(gradients)
            self.log_second = self._incoming_cache[2]
            self._incoming_cache = None
        self.step_count += 1

    def _validate_incoming_cache(self, gradients: Sequence[torch.Tensor]) -> None:
        step, sources, _, _ = self._incoming_cache
        if step != self.step_count or len(sources) != len(gradients) or any(
            source is not gradient or version != gradient._version
            for (source, version), gradient in zip(sources, gradients)
        ):
            raise ValueError("prepared incoming gradients changed before shadow advance")

    @torch.no_grad()
    def prepare_incoming(
        self, gradients: Sequence[torch.Tensor], *, step: int | None = None,
    ) -> torch.Tensor:
        """Return total/per-parameter log D_t, reusing log(g^2) for next log-v.

        This does not advance either optimizer.  A terminal preparation is
        prospective; a later advance must receive the same unmodified gradients.
        The returned tiny tensor can be buffered on-device without synchronization.
        """
        if step is not None and step != self.step_count:
            raise ValueError("incoming step does not match the shadow")
        self._validate_gradients(gradients)
        if self._incoming_cache is not None:
            self._validate_incoming_cache(gradients)
            return self._incoming_cache[3]
        next_logs, energies = [], []
        for old, gradient in zip(self.log_second, gradients):
            log_square = 2.0 * torch.log(gradient.abs())
            next_logs.append(torch.logaddexp(
                old + self.log_beta2, log_square + self.log_one_minus_beta2,
            ))
            energies.append(_logsumexp(log_square))
        per_parameter = torch.stack(energies)
        incoming = torch.cat((_logsumexp(per_parameter).reshape(1), per_parameter))
        incoming = incoming + self.log_one_minus_beta2 - (self.step_count + 1) * self.log_beta2
        self._incoming_cache = (
            self.step_count, [(gradient, gradient._version) for gradient in gradients],
            next_logs, incoming,
        )
        return incoming

    @torch.no_grad()
    def diagnostics(self, optimizer, gradients: Sequence[torch.Tensor]) -> dict[str, float | int]:
        """Preview the next transition; the actual optimizer remains untouched."""
        if optimizer.step_count != self.step_count:
            raise ValueError("ordinary and shadow optimizer steps are not aligned")
        if optimizer.beta2 != self.beta2:
            raise ValueError("ordinary and shadow beta2 do not match")
        if optimizer.epsilon != 0.0:
            raise ValueError("these shadow diagnostics require epsilon=0")
        next_log_list = self.preview(gradients)
        if len(optimizer.second) != len(self.log_second):
            raise ValueError("ordinary and shadow state lengths differ")
        # Match the actual InspectableAdam.step operation sequence, including
        # addcmul_, rather than the runner's algebraically equivalent preview.
        next_first = [
            old.clone().mul_(optimizer.beta1).add_(gradient, alpha=1.0 - optimizer.beta1)
            for old, gradient in zip(optimizer.first, gradients)
        ]
        next_second = [
            old.clone().mul_(self.beta2).addcmul_(gradient, gradient, value=1.0 - self.beta2)
            for old, gradient in zip(optimizer.second, gradients)
        ]
        old_raw = _flat(optimizer.second)
        next_raw = _flat(next_second)
        old_log = _flat(self.log_second)
        next_log = _flat(next_log_list)
        gradient = _flat(gradients)
        log_old_raw = torch.log(old_raw)
        log_raw_denominator = _logsumexp(log_old_raw)
        log_shadow_denominator = _logsumexp(old_log)
        # Keep the gradient-square sum in the log domain for BOTH recurrence
        # quantities.  Their difference then isolates the moment histories.
        log_increment = (
            self.log_one_minus_beta2 - self.log_beta2
            + _logsumexp(2.0 * torch.log(gradient.abs()))
        )
        log_raw_direct = _logsumexp(torch.log((next_raw / self.beta2 - old_raw).abs()))
        log_shadow_direct = _logsumexp(
            _log_abs_difference(next_log - self.log_beta2, old_log)
        )
        result: dict[str, float | int] = {
            "shadow_v_step_before": int(self.step_count),
            "shadow_v_step_after": int(self.step_count + 1),
            "shadow_v_raw_denominator_positive": int(torch.isfinite(log_raw_denominator).item()),
            "shadow_v_log_denominator_positive": int(torch.isfinite(log_shadow_denominator).item()),
        }
        for name, numerator, denominator in (
            ("raw_direct", log_raw_direct, log_raw_denominator),
            ("raw_recurrence", log_increment, log_raw_denominator),
            ("log_direct", log_shadow_direct, log_shadow_denominator),
            ("log_recurrence", log_increment, log_shadow_denominator),
        ):
            value, logarithm = _ratio(numerator, denominator)
            result[f"shadow_v_{name}_relative_l1"] = value
            result[f"shadow_v_{name}_log_relative_l1"] = logarithm

        correction1 = 1.0 - optimizer.beta1 ** (self.step_count + 1)
        correction2 = 1.0 - self.beta2 ** (self.step_count + 1)
        numerator = _flat(next_first) / correction1
        denominator = (next_raw / correction2).clamp_min(0.0).sqrt()
        ordinary_update = torch.zeros_like(numerator)
        active = denominator > 0.0
        ordinary_update[active] = numerator[active] / denominator[active]
        # The sign is the same for both updates, since the numerator is shared.
        # Keep reference updates in log-magnitude form: exp(log-v) could lose
        # history even when the resulting Adam update is representable.
        log_update_reference = torch.full_like(numerator, -torch.inf)
        nonzero_numerator = numerator != 0.0
        log_update_reference[nonzero_numerator] = (
            torch.log(numerator[nonzero_numerator].abs())
            - 0.5 * (next_log[nonzero_numerator] - math.log(correction2))
        )
        log_update_ordinary = torch.log(ordinary_update.abs())
        log_update_extended_difference = _log_abs_difference(log_update_ordinary, log_update_reference)
        shadow_update = numerator.sign() * torch.exp(log_update_reference)
        # Subtract the computed updates BEFORE taking logs.  Subtracting their
        # rounded logarithms instead can conceal one-ulp update differences or
        # introduce a discrepancy when explicit exp rounds both updates equal.
        log_computed_difference = torch.log((ordinary_update - shadow_update).abs())
        log_computed_reference = torch.log(shadow_update.abs())
        result["shadow_v_update_reference_nonzero"] = int((shadow_update != 0).any().item())
        result["shadow_v_update_extended_reference_nonzero"] = int(torch.isfinite(log_update_reference).any().item())
        for name, order in (("l1", 1), ("l2", 2), ("linf", math.inf)):
            value, logarithm = _ratio(
                _log_norm(log_computed_difference, order),
                _log_norm(log_computed_reference, order),
            )
            result[f"shadow_v_update_relative_error_{name}"] = value
            result[f"shadow_v_update_log_relative_error_{name}"] = logarithm
            extended_value, extended_logarithm = _ratio(
                _log_norm(log_update_extended_difference, order),
                _log_norm(log_update_reference, order),
            )
            result[f"shadow_v_update_extended_relative_error_{name}"] = extended_value
            result[f"shadow_v_update_extended_log_relative_error_{name}"] = extended_logarithm

        value, logarithm = _ratio(
            _logsumexp(_log_abs_difference(torch.log(next_raw), next_log)),
            _logsumexp(next_log),
        )
        result["shadow_v_next_relative_error_l1"] = value
        result["shadow_v_next_log_relative_error_l1"] = logarithm
        tiny = torch.finfo(next_raw.dtype).tiny
        next_first_flat = _flat(next_first)
        result.update({
            "shadow_v_raw_zero_log_positive_count": int(((next_raw == 0) & torch.isfinite(next_log)).sum().item()),
            "shadow_v_raw_subnormal_count": int(((next_raw > 0) & (next_raw < tiny)).sum().item()),
            "shadow_v_first_subnormal_count": int(((next_first_flat != 0) & (next_first_flat.abs() < tiny)).sum().item()),
            "shadow_v_gradient_square_underflow_count": int(((gradient != 0) & (gradient.square() == 0)).sum().item()),
            "shadow_v_ordinary_update_nonfinite_count": int((~torch.isfinite(ordinary_update)).sum().item()),
            "shadow_v_reference_update_underflow_count": int((torch.isfinite(log_update_reference) & (shadow_update == 0)).sum().item()),
            "shadow_v_reference_update_nonfinite_count": int((~torch.isfinite(shadow_update)).sum().item()),
            "shadow_v_reference_update_log_invalid_count": int((torch.isnan(log_update_reference) | torch.isposinf(log_update_reference)).sum().item()),
        })
        return result

    @torch.no_grad()
    def checkpoint_state(self) -> dict:
        return {
            "version": 1,
            "beta2": self.beta2,
            "step_count": int(self.step_count),
            "log_second": [value.detach().cpu().clone() for value in self.log_second],
        }

    @torch.no_grad()
    def restore_checkpoint_state(self, payload: dict) -> None:
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported log-v checkpoint format")
        if payload.get("beta2") != self.beta2:
            raise ValueError("log-v checkpoint beta2 mismatch")
        step = payload.get("step_count")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("invalid log-v checkpoint step")
        values = payload.get("log_second")
        if not isinstance(values, (list, tuple)) or len(values) != len(self.log_second):
            raise ValueError("log-v checkpoint tensor count mismatch")
        for source, destination in zip(values, self.log_second):
            if not torch.is_tensor(source) or source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError("log-v checkpoint tensor shape or dtype mismatch")
            if (torch.isnan(source) | torch.isposinf(source)).any().item():
                raise ValueError("log-v checkpoint contains NaN or positive infinity")
        # Validate the entire payload before changing any state.
        self.log_second = [source.to(device=destination.device).clone() for source, destination in zip(values, self.log_second)]
        self._incoming_cache = None
        self.step_count = step
