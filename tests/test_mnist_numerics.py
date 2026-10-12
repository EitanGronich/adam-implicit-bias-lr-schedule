"""Regression tests for moment diagnostics at extreme representable scales."""

import math

import pytest
import torch

from adam_lr.mnist.runner import deterministic_spectral_norm, normalized_margin, safe_cosine, scale_aligned_error, vector_norms
from adam_lr.mnist.matrix_norms import nuclear_norm_iterative
from adam_lr.mnist.coordinate_diagnostics import exact_discrete_rho


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_norm_is_stable_without_squaring_unscaled_entries(scale):
    vector = torch.tensor([scale, 2 * scale], dtype=torch.float64)
    observed = vector_norms(vector)
    assert observed["l2"] == pytest.approx(math.sqrt(5) * scale, rel=2e-15, abs=0)
    assert observed["zero_count"] == 0


@pytest.mark.parametrize("left_scale,right_scale", [(1e-200, 1e200), (1e200, 1e-200)])
def test_cosine_and_projection_error_ignore_vector_magnitudes(left_scale, right_scale):
    left = torch.tensor([1, 2], dtype=torch.float64) * left_scale
    right = torch.tensor([2, 1], dtype=torch.float64) * right_scale
    assert safe_cosine(left, right) == pytest.approx(4 / 5, abs=2e-15)
    error, cosine = scale_aligned_error(left, right)
    assert cosine == pytest.approx(4 / 5, abs=2e-15)
    assert error == pytest.approx(3 / 5, abs=2e-15)


@pytest.mark.parametrize("residual", [1e-10, 1e-200])
def test_small_projection_residual_is_not_lost_to_one_minus_cosine_squared(residual):
    target = torch.tensor([1.0, residual], dtype=torch.float64)
    reference = torch.tensor([1.0, 0.0], dtype=torch.float64)
    error, _ = scale_aligned_error(target, reference)
    assert error == pytest.approx(residual, rel=1e-14, abs=0)


@pytest.mark.parametrize("invalid", [[0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0]])
def test_undefined_direction_is_explicitly_nan(invalid):
    left = torch.tensor(invalid, dtype=torch.float64)
    right = torch.tensor([1.0, 1.0], dtype=torch.float64)
    assert math.isnan(safe_cosine(left, right))
    assert all(math.isnan(value) for value in scale_aligned_error(left, right))


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_scalar_output_nuclear_norm_is_exact_at_all_scales(scale):
    row = torch.tensor([[scale, 2 * scale]], dtype=torch.float64)
    assert nuclear_norm_iterative(row).item() == pytest.approx(math.sqrt(5) * scale, rel=2e-15, abs=0)


def test_matrix_nuclear_approximation_is_scale_invariant():
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    ordinary = nuclear_norm_iterative(matrix).item()
    tiny = nuclear_norm_iterative(matrix * 1e-100).item()
    assert tiny / ordinary == pytest.approx(1e-100, rel=2e-13, abs=0)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_frozen_metric_margin_ignores_common_metric_scale(scale):
    parameters = torch.tensor([2.0, 3.0], dtype=torch.float64)
    metric = torch.tensor([1.0, 4.0], dtype=torch.float64)
    expected = normalized_margin(parameters, 3.0, 3.0, metric, "metric")
    observed = normalized_margin(parameters, 3.0, 3.0, scale * metric, "metric")
    assert observed == pytest.approx(expected, rel=2e-15, abs=0)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_spectral_norm_is_scale_invariant_and_matches_diagonal_oracle(scale):
    matrix = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64)) * scale
    assert deterministic_spectral_norm(matrix) == pytest.approx(4 * scale, rel=2e-14, abs=0)
    row = torch.tensor([[scale, 2 * scale]], dtype=torch.float64)
    assert deterministic_spectral_norm(row) == pytest.approx(math.sqrt(5) * scale, rel=2e-15, abs=0)


def test_spectral_start_does_not_miss_a_zero_row_sum_matrix():
    # An all-ones start lies in this matrix's nullspace although its norm is 2.
    matrix = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    assert deterministic_spectral_norm(matrix) == pytest.approx(2.0, abs=1e-14)


def test_transformed_moment_log_increment_does_not_overflow_with_its_ratio():
    beta = 0.999
    previous, gradient = 1e-100, 1e150
    following = beta * previous + (1-beta) * gradient**2
    rho = exact_discrete_rho(previous, following, gradient, beta)
    expected = math.log(following) - math.log(previous) - math.log(beta)
    assert math.isfinite(float(rho))
    assert float(rho) == pytest.approx(expected, rel=2e-15)
