"""Regression checks for singular frozen metrics and newly active coordinates."""
import math

import pytest
import torch

from adam_lr.mnist.runner import (
    append_scalar_record, frozen_inverse_action, frozen_metric_alignment,
    initialize_scalar_records,
)


def _vector(values):
    return torch.tensor(values, dtype=torch.float64)


def test_frozen_inverse_rejects_newly_active_coordinate():
    assert frozen_inverse_action(_vector([1, 1]), _vector([1, 0])) is None
    action = frozen_inverse_action(_vector([2, 0]), _vector([4, 0]))
    torch.testing.assert_close(action, _vector([.5, 0]))


def test_frozen_alignment_does_not_clamp_invalid_support_to_perfect_alignment():
    # Previously denominator used coordinate 0 only, numerator both: ratio 2
    # was then silently clamped to 1 despite an undefined frozen dual norm.
    assert math.isnan(frozen_metric_alignment(_vector([-1, -1]), _vector([1, 1]), _vector([1, 0])))
    assert math.isnan(frozen_metric_alignment(_vector([-1, -1]), _vector([1, 0]), _vector([1, 0])))
    assert math.isnan(frozen_metric_alignment(_vector([-1, 0]), _vector([1, 1]), _vector([1, 0])))


def test_supported_singular_frozen_alignment_is_defined():
    assert frozen_metric_alignment(_vector([-2, 0]), _vector([3, 0]), _vector([4, 0])) == pytest.approx(1)
    assert math.isnan(frozen_metric_alignment(_vector([0, 0]), _vector([3, 0]), _vector([4, 0])))


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_frozen_alignment_matches_weighted_inner_product_and_scale_invariance(scale):
    action, gradient, metric = _vector([-2, 3]), _vector([4, -5]), _vector([2, 7])
    reference = -torch.dot(action, gradient) / (
        torch.sqrt(torch.dot(metric, action.square()))
        * torch.sqrt(torch.dot(metric.reciprocal(), gradient.square())))
    assert frozen_metric_alignment(action, gradient, scale * metric) == pytest.approx(float(reference), rel=1e-13)
    # Tiny gradients must not become an undefined norm merely through squaring.
    assert frozen_metric_alignment(action, gradient * 1e-200, scale * metric) == pytest.approx(float(reference), rel=1e-13)


def _record(gradient, first_hat):
    records = initialize_scalar_records(layer_names=())
    append_scalar_record(
        records, step=1, loss=.1, error=0, learning_rate=.01, raw_margin=1,
        parameters=_vector([1, 1]), gradient=_vector(gradient),
        root_second=_vector([1, 1]), first_hat=_vector(first_hat),
        anchors={"a1em4": _vector([1, 0])}, degree=2,
        parameter_sketch=_vector([1, 1]), gradient_sketch=_vector(gradient),
    )
    return records


def test_exported_inverse_fields_are_nan_when_gradient_and_momentum_leave_support():
    record = _record([1, 1], [1, 1])
    for name in ("dynamic_frozen_error", "gradient_action_cosine", "actual_frozen_error",
                 "momentum_action_cosine", "frozen_l2_alignment"):
        assert math.isnan(record[f"a1em4_{name}"][0]), name
    # Metric shape and observed own-anchor margins remain separate diagnostics.
    assert math.isfinite(record["a1em4_frozen_l2_margin"][0])
    assert math.isfinite(record["a1em4_metric_cosine"][0])


def test_current_gradient_and_momentum_support_are_checked_independently():
    gradient_only = _record([1, 1], [1, 0])
    assert math.isnan(gradient_only["a1em4_gradient_action_cosine"][0])
    assert gradient_only["a1em4_momentum_action_cosine"][0] == pytest.approx(1)
    momentum_only = _record([1, 0], [1, 1])
    assert momentum_only["a1em4_gradient_action_cosine"][0] == pytest.approx(1)
    assert math.isnan(momentum_only["a1em4_momentum_action_cosine"][0])
    supported = _record([1, 0], [1, 0])
    for name in ("gradient_action_cosine", "momentum_action_cosine", "frozen_l2_alignment"):
        assert supported[f"a1em4_{name}"][0] == pytest.approx(1)
