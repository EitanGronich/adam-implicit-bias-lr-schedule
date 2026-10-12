"""Coordinatewise diagnostics for Adam's raw second-moment buffer.

The raw (not bias-corrected) Adam buffer obeys

    V_t = beta2 * V_{t-1} + (1 - beta2) * g_t**2.

The continuous-time theory uses an exponentially transformed integral.  Its
exact discrete counterpart, up to an irrelevant constant multiplier, is

    I_t = beta2**(-t) * V_t.

This module evaluates changes of ``log(I_t)`` without ever constructing the
potentially enormous factor ``beta2**(-t)``.  Invalid or inactive coordinate
ratios are represented by ``NaN`` so they cannot silently contaminate layer
summaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackingRatios:
    """Coordinatewise second-moment/gradient tracking ratios."""

    gradient_squared_over_second: np.ndarray
    second_over_gradient_squared: np.ndarray


@dataclass(frozen=True)
class PowerLawFit:
    """OLS fit of ``log(value) = intercept + slope * log(time)``."""

    slope: float
    intercept: float
    r_squared: float
    point_count: int


def _validated_beta2(beta2: float) -> float:
    beta2 = float(beta2)
    if not math.isfinite(beta2) or not 0.0 < beta2 < 1.0:
        raise ValueError("beta2 must be finite and lie strictly between zero and one")
    return beta2


def second_moment_decay_rate(beta2: float) -> float:
    """Return the continuous-time analogue ``c2 = -log(beta2)``."""

    return -math.log(_validated_beta2(beta2))


def exact_discrete_rho(
    v_previous: Any,
    v_next: Any,
    gradient: Any,
    beta2: float,
    *,
    verify_recurrence: bool = True,
    recurrence_rtol: float = 1e-7,
    recurrence_atol: float = 0.0,
) -> np.ndarray:
    """Compute the exact one-step change of ``log(beta2**(-t) * V_t)``.

    For coordinates with a positive previous and next buffer, the stable
    recurrence form is

    ``rho_t = log1p((1-beta2) * g_t**2 / (beta2 * V_{t-1}))``.

    This equals ``log(V_t) - log(V_{t-1}) - log(beta2)``.  The former avoids
    subtracting nearly equal logarithms.  Coordinates for which either log is
    undefined are returned as ``NaN``.  A zero current gradient is not itself
    inactive: if the previous buffer is positive, its exact transformed-log
    increment is zero.

    By default, ``v_next`` is checked against the supplied gradient and the
    Adam recurrence, after which the stable recurrence expression is used.
    When verification is disabled, the function instead returns the exact
    raw-buffer log difference; this is useful for rounded external records.
    """

    beta2 = _validated_beta2(beta2)
    previous, following, grad = np.broadcast_arrays(
        np.asarray(v_previous, dtype=np.float64),
        np.asarray(v_next, dtype=np.float64),
        np.asarray(gradient, dtype=np.float64),
    )
    if recurrence_rtol < 0.0 or recurrence_atol < 0.0:
        raise ValueError("recurrence tolerances must be nonnegative")

    finite_inputs = (
        np.isfinite(previous) & np.isfinite(following) & np.isfinite(grad)
    )
    expected = beta2 * previous + (1.0 - beta2) * np.square(grad)
    recurrence_domain = finite_inputs & (previous >= 0.0) & (following >= 0.0)
    if verify_recurrence and np.any(recurrence_domain):
        matches = np.isclose(
            following[recurrence_domain],
            expected[recurrence_domain],
            rtol=recurrence_rtol,
            atol=recurrence_atol,
        )
        if not np.all(matches):
            maximum_error = float(np.max(
                np.abs(following[recurrence_domain] - expected[recurrence_domain])
            ))
            raise ValueError(
                "v_next does not satisfy the supplied Adam recurrence; "
                f"maximum absolute discrepancy is {maximum_error:.6g}"
            )

    rho = np.full(previous.shape, np.nan, dtype=np.float64)
    active = finite_inputs & (previous > 0.0) & (following > 0.0)
    if np.any(active):
        if verify_recurrence:
            # Form the logarithm of the positive recurrence increment first.
            # Its ratio may overflow although the requested log increment is
            # representable. Zero gradients give log_increment=-inf and rho=0.
            with np.errstate(divide="ignore"):
                log_increment = (
                    math.log1p(-beta2) - math.log(beta2)
                    + 2.0 * np.log(np.abs(grad[active]))
                    - np.log(previous[active])
                )
            rho[active] = np.logaddexp(0.0, log_increment)
        else:
            rho[active] = (
                np.log(following[active])
                - np.log(previous[active])
                - math.log(beta2)
            )
    return rho


def normalized_rho(rho: Any, beta2: float) -> np.ndarray:
    """Normalize a transformed-log rate by ``c2 = -log(beta2)``."""

    return np.asarray(rho, dtype=np.float64) / second_moment_decay_rate(beta2)


def exact_normalized_discrete_rho(
    v_previous: Any,
    v_next: Any,
    gradient: Any,
    beta2: float,
    **kwargs: Any,
) -> np.ndarray:
    """Convenience wrapper returning the exact discrete ``rho / c2``."""

    return normalized_rho(
        exact_discrete_rho(
            v_previous, v_next, gradient, beta2, **kwargs,
        ),
        beta2,
    )


def second_moment_tracking_ratios(
    v_next: Any,
    gradient: Any,
) -> TrackingRatios:
    """Return ``g_t**2 / V_t`` and its reciprocal on active coordinates.

    A coordinate is active here only when ``V_t`` and ``g_t**2`` are both
    finite and strictly positive.  All other entries are ``NaN``.
    """

    following, grad = np.broadcast_arrays(
        np.asarray(v_next, dtype=np.float64),
        np.asarray(gradient, dtype=np.float64),
    )
    gradient_squared = np.square(grad)
    active = (
        np.isfinite(following)
        & np.isfinite(gradient_squared)
        & (following > 0.0)
        & (gradient_squared > 0.0)
    )
    gradient_over_second = np.full(following.shape, np.nan, dtype=np.float64)
    second_over_gradient = np.full(following.shape, np.nan, dtype=np.float64)
    gradient_over_second[active] = gradient_squared[active] / following[active]
    second_over_gradient[active] = following[active] / gradient_squared[active]
    return TrackingRatios(
        gradient_squared_over_second=gradient_over_second,
        second_over_gradient_squared=second_over_gradient,
    )


def stable_normalized_rho_from_updated_ratio(
    gradient_squared_over_second: Any,
    beta2: float,
) -> np.ndarray:
    """Recover the normalized transformed-log increment without cancellation.

    With ``q_t = g_t**2 / V_t`` for the updated raw Adam buffer, the
    recurrence implies

    ``Delta log(beta2**(-t) V_t) = -log(1 - (1-beta2) q_t)``.

    This avoids subtracting two rounded, nearly equal float32 logarithms.
    Invalid coordinates are represented by ``NaN``.
    """

    beta2 = _validated_beta2(beta2)
    ratio = np.asarray(gradient_squared_over_second, dtype=np.float64)
    scaled = (1.0 - beta2) * ratio
    result = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(ratio) & (ratio >= 0.0)
        & np.isfinite(scaled) & (scaled >= 0.0) & (scaled < 1.0)
    )
    result[valid] = -np.log1p(-scaled[valid]) / second_moment_decay_rate(beta2)
    return result


def centered_window_rho(
    steps: Any,
    raw_second_moment: Any,
    beta2: float,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a stable centered-window transformed-log rate.

    ``steps`` must be uniformly spaced and index the first axis of
    ``raw_second_moment``.  ``window`` is the full raw-step width.  The result
    is

    ``(log(V[t+W/2]) - log(V[t-W/2])) / W - log(beta2)``.

    This is exactly the centered change rate of ``log(beta2**(-t) V_t)`` and
    avoids materializing the exponential transform.  Coordinates with a
    nonpositive or nonfinite endpoint buffer are returned as ``NaN``.
    """

    beta2 = _validated_beta2(beta2)
    step_values = np.asarray(steps)
    values = np.asarray(raw_second_moment, dtype=np.float64)
    if step_values.ndim != 1:
        raise ValueError("steps must be one-dimensional")
    if values.ndim < 1 or values.shape[0] != step_values.size:
        raise ValueError("raw_second_moment's first axis must match steps")
    if step_values.size < 3:
        raise ValueError("at least three time records are required")
    if not isinstance(window, (int, np.integer)) or int(window) <= 0:
        raise ValueError("window must be a positive integer number of raw steps")
    window = int(window)

    differences = np.diff(step_values.astype(np.float64))
    if np.any(~np.isfinite(differences)) or np.any(differences <= 0.0):
        raise ValueError("steps must be finite and strictly increasing")
    spacing = float(differences[0])
    if not np.allclose(differences, spacing, rtol=0.0, atol=1e-12):
        raise ValueError("steps must be uniformly spaced")
    half_records_float = window / (2.0 * spacing)
    half_records = int(round(half_records_float))
    if not math.isclose(half_records_float, half_records, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("window must be divisible by twice the record spacing")
    if half_records < 1 or 2 * half_records >= step_values.size:
        raise ValueError("window is too large for the supplied time series")

    before = values[:-2 * half_records]
    after = values[2 * half_records:]
    valid = (
        np.isfinite(before) & np.isfinite(after)
        & (before > 0.0) & (after > 0.0)
    )
    rates = np.full(before.shape, np.nan, dtype=np.float64)
    rates[valid] = (
        (np.log(after[valid]) - np.log(before[valid])) / float(window)
        - math.log(beta2)
    )
    centers = step_values[half_records:-half_records].copy()
    return centers, rates


def fit_loglog_power_law(
    times: Any,
    values: Any,
    *,
    start: float | None = None,
    end: float | None = None,
    minimum_points: int = 3,
) -> PowerLawFit:
    """Fit ``value ~= C * time**slope`` using ordinary least squares.

    Nonfinite and nonpositive samples are omitted.  Optional ``start`` and
    ``end`` bounds are applied on the raw time clock before taking logarithms.
    """

    time_values = np.asarray(times, dtype=np.float64)
    rate_values = np.asarray(values, dtype=np.float64)
    if time_values.ndim != 1 or rate_values.ndim != 1:
        raise ValueError("times and values must both be one-dimensional")
    if time_values.shape != rate_values.shape:
        raise ValueError("times and values must have the same shape")
    if minimum_points < 3:
        raise ValueError("minimum_points must be at least three")
    valid = (
        np.isfinite(time_values) & np.isfinite(rate_values)
        & (time_values > 0.0) & (rate_values > 0.0)
    )
    if start is not None:
        valid &= time_values >= float(start)
    if end is not None:
        valid &= time_values <= float(end)
    x = np.log(time_values[valid])
    y = np.log(rate_values[valid])
    if x.size < minimum_points:
        raise ValueError("too few finite positive points for a log-log fit")
    if np.ptp(x) <= 0.0:
        raise ValueError("log-times must span more than one distinct value")

    design = np.column_stack((np.ones_like(x), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    intercept, slope = coefficients
    residual = y - design @ coefficients
    residual_sum_squares = float(np.dot(residual, residual))
    centered = y - y.mean()
    total_sum_squares = float(np.dot(centered, centered))
    if total_sum_squares <= np.finfo(np.float64).eps:
        r_squared = (
            1.0
            if residual_sum_squares <= np.finfo(np.float64).eps
            else float("nan")
        )
    else:
        r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return PowerLawFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        point_count=int(x.size),
    )


def theory_prediction(
    activation_power: float,
    schedule_power: float,
    *,
    affine_layer_count: int = 2,
    homogeneity_degree: float | None = None,
) -> dict[str, float | str | bool | None]:
    """Return theory-facing metadata for a homogeneous feed-forward network.

    The historical two-affine-layer default is preserved: for
    ``ReLU(z)**a``, scaling both affine layers jointly gives exact degree
    ``L = 1 + a``.  Deeper callers pass their affine-layer count; absent an
    explicit degree, a network with this activation after each non-output
    affine layer has ``L = sum(a**r for r in range(K))``.

    The runner uses ``eta(t) = eta0 * t**(-p)``, so a
    negative ``schedule_power=p`` denotes an increasing learning rate.  The
    manuscript's slow tracking theorem requires ``p > 1 - 1/L``.  Below that
    boundary we report the conditional critical-memory heuristic
    ``rho(t) ~ t**(-2*zeta)``, where
    ``zeta = 1 - p - 1/L``; this heuristic is not a theorem.
    """

    activation_power = float(activation_power)
    schedule_power = float(schedule_power)
    if not isinstance(affine_layer_count, int) or affine_layer_count < 1:
        raise ValueError("affine_layer_count must be a positive integer")
    if not any(math.isclose(activation_power, allowed) for allowed in (1.0, 1.1)):
        raise ValueError("activation_power must be either 1 or 1.1")
    if not math.isfinite(schedule_power) or not -1.0 <= schedule_power <= 1.0:
        raise ValueError("schedule_power must be finite and lie in [-1, 1]")

    if homogeneity_degree is None:
        degree = sum(
            activation_power ** exponent
            for exponent in range(affine_layer_count)
        )
    else:
        degree = float(homogeneity_degree)
        if not math.isfinite(degree) or degree <= 0.0:
            raise ValueError("homogeneity_degree must be finite and positive")
    slow_boundary = 1.0 - 1.0 / degree
    fast_boundary = 0.5 - 1.0 / degree
    zeta = 1.0 - schedule_power - 1.0 / degree
    boundary_tolerance = 1e-12

    if schedule_power > slow_boundary + boundary_tolerance:
        regime = "slow"
        predicted_slope = 0.0
        predicted_decay_exponent = 0.0
        prediction_kind = "conditional manuscript tracking plateau"
        normalized_plateau = 1.0
    elif math.isclose(
        schedule_power, slow_boundary, rel_tol=0.0, abs_tol=boundary_tolerance,
    ):
        regime = "slow-boundary"
        predicted_slope = None
        predicted_decay_exponent = None
        prediction_kind = "boundary not covered by the strict little-o condition"
        normalized_plateau = None
    else:
        if schedule_power < fast_boundary - boundary_tolerance:
            regime = "fast"
            prediction_kind = "fast/freezing power guide"
        elif math.isclose(
            schedule_power, fast_boundary,
            rel_tol=0.0,
            abs_tol=boundary_tolerance,
        ):
            regime = "fast-intermediate-boundary"
            prediction_kind = "critical-boundary power guide"
        else:
            regime = "intermediate"
            prediction_kind = "conditional critical-memory heuristic"
        predicted_decay_exponent = 2.0 * zeta
        predicted_slope = -predicted_decay_exponent
        normalized_plateau = None

    return {
        "activation_power": activation_power,
        "affine_layer_count": affine_layer_count,
        "homogeneity_degree": degree,
        "schedule_power": schedule_power,
        "learning_rate_time_exponent": -schedule_power,
        "regime": regime,
        "slow_tracking_threshold": slow_boundary,
        "fast_freezing_threshold": fast_boundary,
        "zeta": zeta,
        "predicted_decay_exponent": predicted_decay_exponent,
        "predicted_loglog_slope": predicted_slope,
        "normalized_rho_plateau": normalized_plateau,
        "prediction_kind": prediction_kind,
        "requires_parameter_direction_convergence": True,
        "requires_positive_limiting_margin": True,
        "requires_nonsmooth_t3": math.isclose(activation_power, 1.0),
        "smooth_model_m1": activation_power > 1.0,
    }
