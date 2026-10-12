import copy
import math
import unittest
from decimal import Decimal, localcontext

import torch

from adam_lr.optim.adam import InspectableAdam, NumericalError
from adam_lr.mnist.log_second_moment import LogSecondMomentShadow


def make_pair(values=(1.0, -2.0, 3.0), beta2=0.999):
    parameter = torch.nn.Parameter(torch.tensor(values, dtype=torch.float64))
    optimizer = InspectableAdam([parameter], beta1=0.9, beta2=beta2, epsilon=0.0)
    shadow = LogSecondMomentShadow([parameter], beta2=beta2)
    return parameter, optimizer, shadow


class LogSecondMomentShadowTest(unittest.TestCase):
    def test_zero_and_signed_zero_histories_have_explicit_undefined_ratio(self):
        parameter, optimizer, shadow = make_pair((1.0, 2.0))
        gradient = torch.tensor([0.0, -0.0], dtype=torch.float64)
        record = shadow.diagnostics(optimizer, [gradient])
        self.assertEqual(record["shadow_v_raw_denominator_positive"], 0)
        self.assertEqual(record["shadow_v_log_denominator_positive"], 0)
        for name in ("raw_direct", "raw_recurrence", "log_direct", "log_recurrence"):
            self.assertTrue(math.isnan(record[f"shadow_v_{name}_relative_l1"]))
        self.assertEqual(record["shadow_v_update_reference_nonzero"], 0)
        shadow.advance([gradient])
        self.assertTrue(torch.isneginf(shadow.log_second[0]).all())
        self.assertEqual(shadow.step_count, 1)

    def test_log_recurrence_and_moment_against_decimal_reference(self):
        parameter, optimizer, shadow = make_pair()
        with localcontext() as context:
            context.prec = 100
            beta = Decimal.from_float(optimizer.beta2)
            one_minus = Decimal.from_float(1.0 - optimizer.beta2)
            moments = [Decimal(0)] * parameter.numel()
            for index in range(80):
                values = [0.031 / (index + 1), (-1.0) ** index * 0.017, 1e-130 / (index + 1)]
                gradient = torch.tensor(values, dtype=torch.float64)
                record = shadow.diagnostics(optimizer, [gradient])
                squared = [Decimal.from_float(value) ** 2 for value in values]
                if index:
                    expected = one_minus / beta * sum(squared) / sum(moments)
                    self.assertAlmostEqual(
                        record["shadow_v_log_recurrence_log_relative_l1"],
                        float(expected.ln()), delta=2e-12,
                    )
                moments = [beta * old + one_minus * square for old, square in zip(moments, squared)]
                shadow.advance([gradient])
                parameter.grad = gradient
                optimizer.step(0.001)
            expected_logs = torch.tensor([float(value.ln()) for value in moments], dtype=torch.float64)
            self.assertTrue(torch.allclose(shadow.log_second[0], expected_logs, rtol=0, atol=5e-12))

    def test_squaring_underflow_does_not_erase_shadow_or_update_reference(self):
        parameter, optimizer, shadow = make_pair((1.0,))
        gradient = torch.tensor([1e-200], dtype=torch.float64)
        first = shadow.diagnostics(optimizer, [gradient])
        self.assertEqual(first["shadow_v_gradient_square_underflow_count"], 1)
        self.assertEqual(first["shadow_v_raw_zero_log_positive_count"], 1)
        self.assertAlmostEqual(first["shadow_v_update_relative_error_l2"], 1.0)
        shadow.advance([gradient])
        parameter.grad = gradient
        with self.assertRaises(NumericalError):
            optimizer.step(0.001)
        self.assertEqual(optimizer.step_count, 0)
        self.assertEqual(optimizer.second[0].item(), 0.0)
        self.assertTrue(math.isfinite(shadow.log_second[0].item()))

    def test_pure_decay_has_zero_mathematical_increment_but_detects_sticky_v(self):
        parameter, optimizer, shadow = make_pair((1.0,))
        smallest = math.ulp(0.0)
        optimizer.second[0].fill_(smallest)
        optimizer.first[0].fill_(1e-160)
        optimizer.step_count = shadow.step_count = 1
        shadow.log_second[0].fill_(math.log(smallest))
        gradient = torch.zeros_like(parameter)
        record = shadow.diagnostics(optimizer, [gradient])
        self.assertEqual(record["shadow_v_raw_recurrence_relative_l1"], 0.0)
        self.assertEqual(record["shadow_v_log_recurrence_relative_l1"], 0.0)
        self.assertEqual(record["shadow_v_log_recurrence_log_relative_l1"], -math.inf)
        self.assertEqual(record["shadow_v_raw_subnormal_count"], 1)
        self.assertAlmostEqual(record["shadow_v_next_relative_error_l1"], 1.0 / optimizer.beta2 - 1.0, delta=2e-13)
        shadow.advance([gradient])
        parameter.grad = gradient
        optimizer.step(0.0)
        self.assertEqual(optimizer.second[0].item(), smallest)
        self.assertAlmostEqual(shadow.log_second[0].item(), math.log(smallest) + math.log(optimizer.beta2), delta=1e-13)

    def test_tiny_increment_survives_recurrence_metric_below_direct_roundoff(self):
        parameter, optimizer, shadow = make_pair((1.0,))
        optimizer.second[0].fill_(1.0)
        optimizer.step_count = shadow.step_count = 1
        shadow.log_second[0].zero_()
        gradient = torch.tensor([1e-100], dtype=torch.float64)
        record = shadow.diagnostics(optimizer, [gradient])
        expected = (1.0 - optimizer.beta2) / optimizer.beta2 * 1e-200
        self.assertAlmostEqual(record["shadow_v_log_recurrence_log_relative_l1"], math.log(expected), delta=1e-12)
        self.assertGreater(record["shadow_v_log_recurrence_relative_l1"], 0)
        self.assertEqual(record["shadow_v_log_direct_relative_l1"], 0)

    def test_preview_matches_actual_optimizer_sequence_and_is_noninvasive(self):
        parameter, optimizer, shadow = make_pair()
        baseline_parameter = torch.nn.Parameter(parameter.detach().clone())
        baseline = InspectableAdam([baseline_parameter], beta1=0.9, beta2=0.999, epsilon=0.0)
        for values in ((0.3, -0.01, 0.0), (1e-160, 0.2, -0.07), (-0.03, 0.0, 0.09)):
            gradient = torch.tensor(values, dtype=torch.float64)
            parameter.grad = gradient.clone()
            baseline_parameter.grad = gradient.clone()
            before_parameter = parameter.detach().clone()
            before_first = optimizer.first[0].clone()
            before_second = optimizer.second[0].clone()
            before_shadow = shadow.log_second[0].clone()
            record = shadow.diagnostics(optimizer, [parameter.grad])
            self.assertTrue(torch.equal(parameter, before_parameter))
            self.assertTrue(torch.equal(parameter.grad, gradient))
            self.assertTrue(torch.equal(optimizer.first[0], before_first))
            self.assertTrue(torch.equal(optimizer.second[0], before_second))
            self.assertTrue(torch.equal(shadow.log_second[0], before_shadow))
            next_log = shadow.preview([parameter.grad])[0]
            shadow.advance([parameter.grad])
            optimizer.step(0.002)
            baseline.step(0.002)
            self.assertTrue(torch.equal(parameter, baseline_parameter))
            self.assertTrue(torch.equal(optimizer.first[0], baseline.first[0]))
            self.assertTrue(torch.equal(optimizer.second[0], baseline.second[0]))
            self.assertTrue(torch.equal(shadow.log_second[0], next_log))
            # Independently materialize both computed update vectors, then
            # subtract.  Their primary error must match even at the ulp scale.
            m_hat = optimizer.first[0] / (1.0 - optimizer.beta1 ** optimizer.step_count)
            correction2 = 1.0 - optimizer.beta2 ** optimizer.step_count
            ordinary_root = (optimizer.second[0] / correction2).sqrt()
            ordinary_action = torch.zeros_like(m_hat)
            mask = ordinary_root > 0
            ordinary_action[mask] = m_hat[mask] / ordinary_root[mask]
            shadow_action = torch.zeros_like(m_hat)
            mask = m_hat != 0
            shadow_action[mask] = m_hat[mask].sign() * torch.exp(
                torch.log(m_hat[mask].abs()) - 0.5 * (next_log[mask] - math.log(correction2))
            )
            for name, order in (("l1", 1), ("l2", 2), ("linf", math.inf)):
                expected = (torch.linalg.vector_norm(ordinary_action - shadow_action, ord=order)
                            / torch.linalg.vector_norm(shadow_action, ord=order)).item()
                self.assertTrue(math.isclose(
                    record[f"shadow_v_update_relative_error_{name}"], expected,
                    rel_tol=5e-14, abs_tol=0,
                ))
            if before_second.sum().item() > 0:
                direct = ((optimizer.second[0] / optimizer.beta2 - before_second).abs().sum() / before_second.sum()).item()
                self.assertAlmostEqual(record["shadow_v_raw_direct_relative_l1"], direct, delta=1e-12 * max(1, direct))
            # In ordinary range only ordinary log/exp rounding should separate
            # the update calculations; this does not change actual training.
            self.assertLess(record["shadow_v_update_relative_error_l2"], 2e-12)

    def test_extended_update_metric_retains_underflowed_computed_reference(self):
        parameter, optimizer, shadow = make_pair((1.0,))
        optimizer.step_count = shadow.step_count = 1
        optimizer.first[0].fill_(1e-300)
        optimizer.second[0].fill_(1e200)
        shadow.log_second[0].fill_(math.log(1e200))
        record = shadow.diagnostics(optimizer, [torch.zeros_like(parameter)])
        self.assertEqual(record["shadow_v_update_reference_nonzero"], 0)
        self.assertEqual(record["shadow_v_reference_update_underflow_count"], 1)
        self.assertTrue(math.isnan(record["shadow_v_update_relative_error_l2"]))
        self.assertEqual(record["shadow_v_update_extended_reference_nonzero"], 1)
        self.assertAlmostEqual(record["shadow_v_update_extended_relative_error_l2"], 1.0)

    def test_checkpoint_resume_and_atomic_validation(self):
        parameter, optimizer, shadow = make_pair()
        gradients = [torch.tensor([0.1, 1e-200, 0.0], dtype=torch.float64)]
        shadow.advance(gradients)
        payload = shadow.checkpoint_state()
        restored = LogSecondMomentShadow([parameter], optimizer.beta2)
        restored.restore_checkpoint_state(payload)
        self.assertEqual(restored.step_count, shadow.step_count)
        self.assertTrue(torch.equal(restored.log_second[0], shadow.log_second[0]))
        shadow.advance(gradients)
        restored.advance(gradients)
        self.assertTrue(torch.equal(restored.log_second[0], shadow.log_second[0]))
        for mutation in (
            {"beta2": 0.99}, {"step_count": -1},
            {"log_second": [torch.zeros(3, dtype=torch.float32)]},
            {"log_second": [torch.zeros(2, dtype=torch.float64)]},
            {"log_second": [torch.tensor([0.0, math.inf, -math.inf], dtype=torch.float64)]},
            {"log_second": [torch.tensor([0.0, math.nan, -math.inf], dtype=torch.float64)]},
        ):
            invalid = copy.deepcopy(payload)
            invalid.update(mutation)
            before = restored.log_second[0].clone()
            with self.assertRaises(ValueError):
                restored.restore_checkpoint_state(invalid)
            self.assertTrue(torch.equal(restored.log_second[0], before))
            self.assertEqual(restored.step_count, 2)

    def test_step_mismatch_is_rejected(self):
        parameter, optimizer, shadow = make_pair()
        shadow.step_count = 1
        with self.assertRaisesRegex(ValueError, "not aligned"):
            shadow.diagnostics(optimizer, [torch.zeros_like(parameter)])


if __name__ == "__main__":
    unittest.main()
