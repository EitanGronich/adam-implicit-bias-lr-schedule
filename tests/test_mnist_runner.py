from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from adam_lr.mnist import runner
from adam_lr.optim.adam import InspectableAdam


class Mnist1024ThreeLayerAdamExperimentTest(unittest.TestCase):
    def test_model_is_bias_free_cubic_and_has_exact_production_shape(self) -> None:
        model = runner.HomogeneousThreeLayerReLU()
        self.assertEqual(
            tuple(name for name, _ in model.named_parameters()),
            ("hidden1.weight", "hidden2.weight", "output.weight"),
        )
        self.assertEqual(tuple(model.hidden1.weight.shape), (1024, 784))
        self.assertEqual(tuple(model.hidden2.weight.shape), (1024, 1024))
        self.assertEqual(tuple(model.output.weight.shape), (1, 1024))
        self.assertIsNone(model.hidden1.bias)
        self.assertIsNone(model.hidden2.bias)
        self.assertIsNone(model.output.bias)
        self.assertEqual(model.affine_layer_count, 3)
        self.assertEqual(model.homogeneity_degree, 3.0)

        toy = runner.HomogeneousThreeLayerReLU(input_dim=3, hidden_width=4).double()
        x = torch.randn(5, 3, dtype=torch.float64)
        with torch.no_grad():
            original = toy(x)
            for parameter in toy.parameters():
                parameter.mul_(2.0)
            scaled = toy(x)
        torch.testing.assert_close(scaled, 8.0 * original)

    def test_parser_defaults_to_production_contract(self) -> None:
        with mock.patch("sys.argv", ["runner"]):
            args = runner.default_args()
        self.assertEqual(args.per_class, 512)
        self.assertEqual(args.hidden_widths, (1024, 1024))
        self.assertEqual(args.dtype, "float64")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.checkpoint_every, 10_000)
        self.assertEqual(args.checkpoint_every_seconds, 600.0)

    def test_theory_metadata_uses_three_layer_homogeneity(self) -> None:
        theory = runner.theory_metadata(1.0, 0.25)
        self.assertEqual(theory["affine_layer_count"], 3)
        self.assertEqual(theory["homogeneity_degree"], 3.0)
        self.assertAlmostEqual(theory["slow_tracking_threshold"], 2.0 / 3.0)
        self.assertAlmostEqual(theory["fast_freezing_threshold"], 1.0 / 6.0)
        expected_regimes = {
            0.0: "fast", 0.1: "fast", 0.25: "intermediate",
            0.5: "intermediate", 0.7: "slow",
        }
        for power, regime in expected_regimes.items():
            with self.subTest(schedule_power=power):
                self.assertEqual(
                    runner.theory_metadata(1.0, power)["regime"], regime,
                )

    def test_three_layer_groups_and_regular_schema_are_complete(self) -> None:
        model = runner.HomogeneousThreeLayerReLU(input_dim=3, hidden_width=4)
        self.assertEqual(
            runner.layer_parameter_indices(model),
            {"hidden1": (0,), "hidden2": (1,), "output": (2,)},
        )
        records = runner.initialize_scalar_records()
        for layer in runner.LAYER_NAMES:
            for suffix in (
                "parameter_l1", "gradient_l2", "first_moment_raw_linf",
                "second_moment_raw_l2", "root_second_moment_hat_l2",
                "update_linf", "applied_update_l2", "weight_spectral_norm",
                "gradient_nuclear_norm", "layer_linf",
            ):
                self.assertIn(f"{layer}_{suffix}", records)
        for tag in ("a1em4", "a1em8", "a1em12"):
            self.assertIn(f"{tag}_second_moment_cosine", records)
        self.assertEqual(runner.DIAGNOSTIC_VERSION, 5)

    def test_raw_second_moment_anchor_cosine_is_logged(self) -> None:
        records = runner.initialize_scalar_records()
        second = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        frozen_second = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64)
        runner.append_scalar_record(
            records,
            step=7,
            loss=1e-5,
            error=0.0,
            learning_rate=1e-3,
            raw_margin=2.0,
            parameters=torch.ones(3, dtype=torch.float64),
            gradient=torch.ones(3, dtype=torch.float64),
            root_second=torch.ones(3, dtype=torch.float64),
            first_hat=torch.ones(3, dtype=torch.float64),
            second_raw=second,
            anchors={"a1em4": torch.ones(3, dtype=torch.float64)},
            anchor_second_moments={"a1em4": frozen_second},
            degree=3.0,
            parameter_sketch=torch.ones(1),
            gradient_sketch=torch.ones(1),
        )
        expected = torch.dot(second, frozen_second) / (
            second.norm() * frozen_second.norm()
        )
        self.assertAlmostEqual(
            records["a1em4_second_moment_cosine"][0], expected.item(),
        )
        self.assertTrue(math.isnan(records["a1em8_second_moment_cosine"][0]))
        self.assertEqual(records["q_min"][0], records["raw_margin"][0])

    def test_rolling_checkpoint_contains_raw_anchor_and_all_rng_states(self) -> None:
        model = runner.HomogeneousThreeLayerReLU(input_dim=3, hidden_width=4)
        optimizer = InspectableAdam(model.parameters(), 0.9, 0.99, 0.0)
        groups = runner.layer_parameter_indices(model)
        scores = {
            layer: torch.zeros(sum(
                list(model.parameters())[index].numel() for index in indices
            ))
            for layer, indices in groups.items()
        }
        counts = {
            layer: torch.zeros_like(score, dtype=torch.int32)
            for layer, score in scores.items()
        }
        frozen = torch.tensor([0.1, 0.2], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            runner.save_checkpoint(
                path,
                signature={"diagnostic_version": 4},
                model=model,
                optimizer=optimizer,
                scalar_records=runner.initialize_scalar_records(),
                coordinate_records=runner.initialize_coordinate_records(),
                selection=None,
                selection_step=None,
                selection_actual_loss=None,
                scores=scores,
                nonzero_counts=counts,
                score_observations=0,
                anchors={"a1em4": frozen.sqrt()},
                anchor_momenta={"a1em4": frozen.clone()},
                anchor_updates={"a1em4": frozen.clone()},
                anchor_second_moments={"a1em4": frozen.clone()},
                anchor_steps={"a1em4": 10},
                anchor_losses={"a1em4": 1e-4},
                cumulative_learning_rate=0.1,
                elapsed_seconds_total=1.0,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
        torch.testing.assert_close(
            payload["anchor_second_moments"]["a1em4"], frozen,
        )
        self.assertEqual(
            set(payload["rng_state"]),
            {"torch_cpu", "torch_cuda", "numpy", "python"},
        )

    def test_rng_round_trip_and_signal_flag(self) -> None:
        torch.manual_seed(11)
        np.random.seed(12)
        random.seed(13)
        state = runner.rng_state_payload()
        expected = (torch.rand(2), np.random.rand(2), random.random())
        runner.restore_rng_state(state)
        actual = (torch.rand(2), np.random.rand(2), random.random())
        torch.testing.assert_close(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])

        runner.STOP_REQUESTED = False
        with mock.patch("builtins.print"):
            runner.request_stop(15, None)
        self.assertTrue(runner.STOP_REQUESTED)
        runner.STOP_REQUESTED = False

    def test_resume_allows_only_monotone_max_steps_extension(self) -> None:
        saved = {"max_steps": 20_000, "target_loss": 1e-20}
        requested = {"max_steps": 1_000_000, "target_loss": 1e-20}
        migrations = runner.validate_resume_signature(
            saved, requested, allow_target_extension=False,
        )
        self.assertEqual(migrations, ("max_steps:20000->1000000",))
        self.assertEqual(runner.validate_resume_signature(
            saved, dict(saved), allow_target_extension=False,
        ), ())
        with self.assertRaisesRegex(RuntimeError, "strictly extend"):
            runner.validate_resume_signature(
                saved,
                {"max_steps": 19_999, "target_loss": 1e-20},
                allow_target_extension=False,
            )


if __name__ == "__main__":
    unittest.main()
