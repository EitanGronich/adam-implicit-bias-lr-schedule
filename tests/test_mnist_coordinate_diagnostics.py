import math
import unittest

import numpy as np

from adam_lr.mnist.coordinate_diagnostics import (
    centered_window_rho,
    exact_discrete_rho,
    exact_normalized_discrete_rho,
    fit_loglog_power_law,
    normalized_rho,
    second_moment_decay_rate,
    second_moment_tracking_ratios,
    stable_normalized_rho_from_updated_ratio,
    theory_prediction,
)


class SecondMomentCoordinateDiagnosticTests(unittest.TestCase):
    def test_theory_prediction_supports_deeper_models_without_changing_default(self):
        legacy = theory_prediction(1.0, 0.25)
        self.assertEqual(legacy["affine_layer_count"], 2)
        self.assertEqual(legacy["homogeneity_degree"], 2.0)

        three_layer = theory_prediction(1.0, 0.25, affine_layer_count=3)
        self.assertEqual(three_layer["affine_layer_count"], 3)
        self.assertEqual(three_layer["homogeneity_degree"], 3.0)
        self.assertAlmostEqual(three_layer["slow_tracking_threshold"], 2.0 / 3.0)

        powered_three_layer = theory_prediction(
            1.1, 0.25, affine_layer_count=3,
        )
        self.assertAlmostEqual(powered_three_layer["homogeneity_degree"], 3.31)

        explicit = theory_prediction(
            1.1, 0.25, affine_layer_count=3, homogeneity_degree=3.2,
        )
        self.assertEqual(explicit["homogeneity_degree"], 3.2)

    def test_exact_discrete_rho_uses_stable_recurrence_identity(self):
        beta2 = 0.9
        previous = np.array([2.0, 0.0, 2.0])
        gradient = np.array([3.0, 1.0, 0.0])
        following = beta2 * previous + (1.0 - beta2) * gradient ** 2

        rho = exact_discrete_rho(previous, following, gradient, beta2)

        self.assertAlmostEqual(rho[0], math.log(1.5))
        self.assertTrue(math.isnan(rho[1]))
        self.assertEqual(rho[2], 0.0)
        self.assertAlmostEqual(
            rho[0], math.log(following[0]) - math.log(previous[0]) - math.log(beta2),
        )

    def test_exact_discrete_rho_rejects_inconsistent_buffer(self):
        with self.assertRaisesRegex(ValueError, "does not satisfy"):
            exact_discrete_rho(2.0, 7.0, 3.0, 0.9)

        unchecked = exact_discrete_rho(
            2.0, 7.0, 3.0, 0.9, verify_recurrence=False,
        )
        self.assertAlmostEqual(
            unchecked.item(), math.log(7.0) - math.log(2.0) - math.log(0.9),
        )

    def test_normalized_rho_uses_negative_log_beta2(self):
        beta2 = 0.999
        c2 = -math.log(beta2)
        self.assertAlmostEqual(second_moment_decay_rate(beta2), c2)
        self.assertAlmostEqual(normalized_rho(c2, beta2).item(), 1.0)

        previous = np.array([2.0])
        gradient = np.array([3.0])
        following = beta2 * previous + (1.0 - beta2) * gradient ** 2
        direct = exact_discrete_rho(previous, following, gradient, beta2)
        wrapped = exact_normalized_discrete_rho(
            previous, following, gradient, beta2,
        )
        np.testing.assert_allclose(wrapped, direct / c2)

    def test_tracking_ratios_mark_inactive_coordinates_nan(self):
        ratios = second_moment_tracking_ratios(
            np.array([2.0, 2.0, 0.0, np.nan]),
            np.array([2.0, 0.0, 1.0, 1.0]),
        )
        self.assertEqual(ratios.gradient_squared_over_second[0], 2.0)
        self.assertEqual(ratios.second_over_gradient_squared[0], 0.5)
        self.assertTrue(np.all(np.isnan(
            ratios.gradient_squared_over_second[1:]
        )))
        self.assertTrue(np.all(np.isnan(
            ratios.second_over_gradient_squared[1:]
        )))

    def test_stable_rho_from_updated_ratio_avoids_log_subtraction(self):
        beta2 = 0.999
        previous = np.asarray([1.0, 3.0], dtype=np.float32)
        gradient_squared = np.asarray([1e-7, 0.25], dtype=np.float32)
        updated = beta2 * previous + (1.0 - beta2) * gradient_squared
        ratio = gradient_squared.astype(np.float64) / updated.astype(np.float64)
        actual = stable_normalized_rho_from_updated_ratio(ratio, beta2)
        expected = -np.log1p(-(1.0 - beta2) * ratio) / (-math.log(beta2))
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
        self.assertTrue(np.all(actual > 0.0))

    def test_centered_window_rate_does_not_construct_exponential_integral(self):
        beta2 = 0.9
        steps = np.arange(0, 18, 2)
        target_rates = np.array([0.2, -0.05])
        values = np.stack([
            beta2 ** steps * np.exp(rate * steps) for rate in target_rates
        ], axis=1)

        centers, rates = centered_window_rho(
            steps, values, beta2=beta2, window=4,
        )

        np.testing.assert_array_equal(centers, steps[1:-1])
        np.testing.assert_allclose(
            rates,
            np.broadcast_to(target_rates, rates.shape),
            atol=1e-14,
        )

    def test_centered_window_rate_marks_invalid_endpoint_nan(self):
        values = np.ones((5, 2))
        values[0, 1] = 0.0
        _, rates = centered_window_rho(
            np.arange(5), values, beta2=0.9, window=2,
        )
        self.assertTrue(math.isnan(rates[0, 1]))
        self.assertTrue(np.isfinite(rates[1:, 1]).all())

    def test_loglog_fit_recovers_polynomial_slope_and_r_squared(self):
        times = np.arange(1.0, 101.0)
        values = 3.0 * times ** -0.5
        values[10] = np.nan
        fit = fit_loglog_power_law(times, values, start=5.0, end=95.0)

        self.assertAlmostEqual(fit.slope, -0.5, places=12)
        self.assertAlmostEqual(fit.intercept, math.log(3.0), places=12)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=12)
        self.assertEqual(fit.point_count, 90)

    def test_relu_theory_predictions_for_requested_schedules(self):
        expected = {
            0.25: ("intermediate", -0.5),
            0.4: ("intermediate", -0.2),
            0.6: ("slow", 0.0),
            0.75: ("slow", 0.0),
        }
        for schedule_power, (regime, slope) in expected.items():
            with self.subTest(schedule_power=schedule_power):
                prediction = theory_prediction(1.0, schedule_power)
                self.assertEqual(prediction["homogeneity_degree"], 2.0)
                self.assertEqual(prediction["regime"], regime)
                self.assertAlmostEqual(prediction["predicted_loglog_slope"], slope)
                self.assertTrue(prediction["requires_nonsmooth_t3"])

    def test_increasing_learning_rates_are_fast(self):
        relu_010 = theory_prediction(1.0, -0.10)
        relu_025 = theory_prediction(1.0, -0.25)
        power_010 = theory_prediction(1.1, -0.10)
        power_025 = theory_prediction(1.1, -0.25)

        for prediction in (relu_010, relu_025, power_010, power_025):
            self.assertEqual(prediction["regime"], "fast")
            self.assertGreater(prediction["predicted_decay_exponent"], 1.0)

        self.assertAlmostEqual(relu_010["predicted_loglog_slope"], -1.2)
        self.assertAlmostEqual(relu_025["predicted_loglog_slope"], -1.5)
        self.assertAlmostEqual(
            power_010["predicted_loglog_slope"], -1.2476190476190476,
        )
        self.assertAlmostEqual(
            power_025["predicted_loglog_slope"], -1.5476190476190477,
        )
        self.assertAlmostEqual(relu_010["learning_rate_time_exponent"], 0.10)

    def test_power_relu_theory_uses_exact_degree_two_point_one(self):
        prediction_025 = theory_prediction(1.1, 0.25)
        prediction_040 = theory_prediction(1.1, 0.4)
        prediction_060 = theory_prediction(1.1, 0.6)

        self.assertAlmostEqual(prediction_025["homogeneity_degree"], 2.1)
        self.assertAlmostEqual(
            prediction_025["predicted_decay_exponent"], 23.0 / 42.0,
        )
        self.assertAlmostEqual(
            prediction_040["predicted_decay_exponent"], 26.0 / 105.0,
        )
        self.assertEqual(prediction_060["regime"], "slow")
        self.assertEqual(prediction_060["normalized_rho_plateau"], 1.0)
        self.assertFalse(prediction_060["requires_nonsmooth_t3"])
        self.assertTrue(prediction_060["smooth_model_m1"])


if __name__ == "__main__":
    unittest.main()
