"""End-to-end coverage of the optional shadow in the canonical runner."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from adam_lr.mnist import runner


class ShadowRunnerIntegrationTest(unittest.TestCase):
    def test_baseline_and_shadow_are_identical_and_resume_is_exact(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(709)
        x = torch.randn(1024, 3, generator=generator, dtype=torch.float64)
        y = torch.where(x[:, 0] > 0, 1.0, -1.0)
        spec = {"name": "toy", "activation_power": 1.0,
                "schedule_power": .25, "learning_rate": 3e-4}

        def run(root, steps, shadow, resume=False):
            with mock.patch("sys.argv", ["runner"]):
                args = runner.default_args()
            args.output_dir = root
            args.hidden_widths = (4, 4)
            args.max_steps = steps
            args.record_every = 2
            args.coordinate_every = 2
            args.expensive_every = 4
            args.checkpoint_every = 2
            args.checkpoint_every_seconds = 0
            args.progress_every = 0
            args.spectral_norm_steps = 2
            args.nuclear_norm_steps = 2
            args.sketch_dimension = 4
            args.coordinate_count = 2
            args.full_checkpoint_losses = "1.0"
            args.dataset_sha256 = runner.tensor_sha256(x)
            args.log_second_moment_shadow = shadow
            args.beta_policy = "fixed"
            args.beta2_reference = .999
            args.resume = resume
            with mock.patch.object(runner, "ARCHITECTURE", (3, 4, 4, 1)):
                result = runner.run_one(args=args, spec=spec, x=x, y=y)
            return torch.load(root / "toy/checkpoint.pt", weights_only=False), result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, _ = run(root / "baseline", 6, False)
            full, result = run(root / "full", 6, True)
            run(root / "resume", 3, True)
            resumed, _ = run(root / "resume", 6, True, resume=True)
            for name, value in baseline["model_state"].items():
                self.assertTrue(torch.equal(value, full["model_state"][name]))
                self.assertTrue(torch.equal(value, resumed["model_state"][name]))
            for field in ("first", "second"):
                for a, b, c in zip(baseline["optimizer_state"][field],
                                   full["optimizer_state"][field],
                                   resumed["optimizer_state"][field]):
                    self.assertTrue(torch.equal(a, b))
                    self.assertTrue(torch.equal(a, c))
            a = full["optimizer_state"]["log_second_shadow"]
            b = resumed["optimizer_state"]["log_second_shadow"]
            self.assertEqual(a["step_count"], 6)
            self.assertEqual(b["step_count"], 6)
            for left, right in zip(a["log_second"], b["log_second"]):
                self.assertTrue(torch.equal(left, right))
            self.assertIn("log_second_moment_shadow", result["signature"])
            rows = [json.loads(line) for line in
                    (root / "resume/toy/log_second_moment_shadow.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [0, 2, 3, 4, 6])
            self.assertIsNone(rows[0]["shadow_v_log_recurrence_relative_l1"])
            with np.load(root / "resume/toy/trajectory_series.npz") as series:
                for key in series.files:
                    if key.startswith("shadow_v_"):
                        self.assertEqual(len(series[key]), len(series["step"]))
                np.testing.assert_array_equal(series["shadow_v_step_before"], series["step"])
                np.testing.assert_array_equal(series["shadow_v_step_after"], series["step"] + 1)
            manifest = json.loads((root / "full/toy/loss_checkpoints_manifest.json").read_text())
            snapshot = torch.load(root / "full/toy/loss_checkpoints" /
                                  manifest["checkpoints"][0]["filename"], weights_only=False)
            self.assertIn("second_moment_log_raw", snapshot)
            self.assertIn("log_second_shadow", snapshot["optimizer_state_before_current_gradient"])


if __name__ == "__main__":
    unittest.main()
