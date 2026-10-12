"""Focused synthetic coverage of the per-step incoming-moment trace."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from adam_lr.mnist import runner
from adam_lr.optim.adam import InspectableAdam
from adam_lr.mnist.incoming_trace import BufferedIncomingTrace
from adam_lr.mnist.log_second_moment import LogSecondMomentShadow


LAYERS = ("hidden1", "hidden2", "output")
FIELDS = ("log_D_total", *(f"log_D_{layer}" for layer in LAYERS))
SIGNATURE = {"beta2": 0.999, "epsilon": 0.0, "seed": 709}


def read_rows(root: Path) -> list[dict]:
    return [json.loads(line) for line in
            (root / "incoming_second_moment.jsonl").read_text().splitlines()]


def make_shadow():
    parameters = [torch.nn.Parameter(torch.tensor(values, dtype=torch.float64))
                  for values in ((1.0, -2.0), (0.25, 0.75, -1.0), (0.5,))]
    return parameters, LogSecondMomentShadow(parameters, beta2=0.999)


class IncomingContributionTest(unittest.TestCase):
    def test_total_and_layer_contributions_match_direct_gradient_energy(self):
        parameters, shadow = make_shadow()
        gradients = [torch.tensor(values, dtype=torch.float64)
                     for values in ((0.3, -0.4), (0.0, -0.2, 0.1), (0.7,))]
        for step in range(4):
            before = [value.clone() for value in shadow.log_second]
            logs = shadow.prepare_incoming(gradients, step=step)
            factor = (1.0 - shadow.beta2) * shadow.beta2 ** (-(step + 1))
            energies = [float(gradient.square().sum()) for gradient in gradients]
            expected = torch.tensor([math.log(factor * energy)
                                     for energy in (sum(energies), *energies)],
                                    dtype=torch.float64)
            torch.testing.assert_close(logs, expected, rtol=0, atol=2e-13)
            self.assertEqual(shadow.step_count, step)
            for old, current in zip(before, shadow.log_second):
                self.assertTrue(torch.equal(old, current))
            self.assertTrue(torch.equal(logs, shadow.prepare_incoming(gradients, step=step)))
            shadow.advance(gradients)

    def test_log_contributions_survive_square_underflow_and_overflow(self):
        for magnitude in (1e-200, 1e200):
            with self.subTest(magnitude=magnitude):
                parameters, shadow = make_shadow()
                shadow.step_count = 1234
                gradients = [torch.tensor([magnitude, -2 * magnitude], dtype=torch.float64),
                             torch.zeros_like(parameters[1]), torch.zeros_like(parameters[2])]
                energy = gradients[0].square().sum().item()
                self.assertTrue(energy == 0.0 if magnitude < 1 else math.isinf(energy))
                logs = shadow.prepare_incoming(gradients, step=1234)
                expected = (math.log1p(-0.999) - 1235 * math.log(0.999)
                            + math.log(5.0) + 2 * math.log(magnitude))
                self.assertAlmostEqual(logs[0].item(), expected, delta=3e-12)
                self.assertAlmostEqual(logs[1].item(), expected, delta=3e-12)
                self.assertTrue(torch.isneginf(logs[2:]).all())

    def test_all_zero_gradients_have_exact_zero_incoming_energy(self):
        parameters, shadow = make_shadow()
        logs = shadow.prepare_incoming([torch.zeros_like(value) for value in parameters])
        self.assertTrue(torch.isneginf(logs).all())

    def test_tracing_does_not_change_adam_or_shadow_trajectory(self):
        parameters, traced = make_shadow()
        baseline_parameters = [torch.nn.Parameter(value.detach().clone()) for value in parameters]
        baseline = LogSecondMomentShadow(baseline_parameters, beta2=0.999)
        optimizer = InspectableAdam(parameters, beta1=0.9, beta2=0.999, epsilon=0.0)
        control = InspectableAdam(baseline_parameters, beta1=0.9, beta2=0.999, epsilon=0.0)
        for step in range(7):
            gradients = [torch.cos(value.detach() * (step + 1)) / (step + 2)
                         for value in baseline_parameters]
            for left, right, gradient in zip(parameters, baseline_parameters, gradients):
                left.grad = gradient.clone()
                right.grad = gradient.clone()
            before_parameters = [value.detach().clone() for value in parameters]
            traced.prepare_incoming([value.grad for value in parameters], step=step)
            for old, value in zip(before_parameters, parameters):
                self.assertTrue(torch.equal(old, value))
            traced.advance([value.grad for value in parameters])
            baseline.advance([value.grad for value in baseline_parameters])
            optimizer.step(0.002)
            control.step(0.002)
            for left, right in zip(parameters, baseline_parameters):
                self.assertTrue(torch.equal(left, right))
            for field in ("first", "second"):
                for left, right in zip(getattr(optimizer, field), getattr(control, field)):
                    self.assertTrue(torch.equal(left, right))
            for left, right in zip(traced.log_second, baseline.log_second):
                self.assertTrue(torch.equal(left, right))

    def test_misaligned_step_and_changed_pending_gradient_are_rejected(self):
        parameters, shadow = make_shadow()
        gradients = [torch.ones_like(value) for value in parameters]
        with self.assertRaises(ValueError):
            shadow.prepare_incoming(gradients, step=1)
        shadow.prepare_incoming(gradients, step=0)
        gradients[0].mul_(2.0)
        with self.assertRaises(ValueError):
            shadow.prepare_incoming(gradients, step=0)
        with self.assertRaises(ValueError):
            shadow.advance(gradients)
        self.assertEqual(shadow.step_count, 0)


class BufferedIncomingTraceTest(unittest.TestCase):
    def make_writer(self, root, **kwargs):
        return BufferedIncomingTrace(root, signature=SIGNATURE, layer_names=LAYERS,
                                     buffer_rows=2, **kwargs)

    def test_chunk_flush_and_exact_zero_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = self.make_writer(root)
            logs = torch.tensor([-2.0, -3.0, -4.0, -math.inf], dtype=torch.float64)
            writer.append(step=0, log_loss=-1.0, log_increments=logs)
            writer.append(step=1, log_loss=-2.0, log_increments=logs)
            self.assertEqual([row["step"] for row in read_rows(root)], [0, 1])
            writer.append(step=2, log_loss=-3.0, log_increments=logs.tolist())
            self.assertEqual(len(read_rows(root)), 2)
            writer.flush()
            writer.flush()
            rows = read_rows(root)
            self.assertEqual([row["step"] for row in rows], [0, 1, 2])
            self.assertEqual(rows[-1]["log_loss"], -3.0)
            self.assertEqual([rows[-1][key] for key in FIELDS], [-2.0, -3.0, -4.0, None])
            self.assertEqual(set(rows[-1]), {"step", "log_loss", *FIELDS})
            with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
                self.make_writer(root)

    def test_resume_discards_checkpoint_row_and_later_rows_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = self.make_writer(root)
            for step in range(6):
                writer.append(step=step, log_loss=-float(step),
                              log_increments=[float(step)] * 4)
            writer.flush()
            resumed = self.make_writer(root, resume_step=3, saved_signature=SIGNATURE)
            self.assertEqual([row["step"] for row in read_rows(root)], [0, 1, 2])
            for step in range(3, 5):
                resumed.append(step=step, log_loss=-float(step),
                               log_increments=[100.0 + step] * 4)
            resumed.flush()
            rows = read_rows(root)
            self.assertEqual([row["step"] for row in rows], list(range(5)))
            self.assertEqual(rows[2]["log_D_total"], 2.0)
            self.assertEqual(rows[3]["log_D_total"], 103.0)

    def test_duplicate_and_gap_are_rejected(self):
        for bad_step in (0, 2):
            with self.subTest(bad_step=bad_step), tempfile.TemporaryDirectory() as temporary:
                writer = self.make_writer(Path(temporary))
                writer.append(step=0, log_loss=-1.0, log_increments=[-2.0] * 4)
                with self.assertRaises((ValueError, RuntimeError)):
                    writer.append(step=bad_step, log_loss=-2.0, log_increments=[-3.0] * 4)

    def test_missing_durable_prefix_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = self.make_writer(root)
            writer.append(step=0, log_loss=-1.0, log_increments=[-2.0] * 4)
            writer.flush()
            with self.assertRaises((ValueError, RuntimeError)):
                self.make_writer(root, resume_step=3, saved_signature=SIGNATURE)

    def test_resume_discards_torn_tail_but_rejects_a_duplicate_required_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = self.make_writer(root)
            for step in range(3):
                writer.append(step=step, log_loss=-1.0, log_increments=[-2.0] * 4)
            writer.flush()
            path = root / "incoming_second_moment.jsonl"
            with path.open("ab") as stream:
                stream.write(b'{"step":3,"log_loss":')
            resumed = self.make_writer(root, resume_step=3, saved_signature=SIGNATURE)
            resumed.append(step=3, log_loss=-2.0, log_increments=[-3.0] * 4)
            resumed.flush()
            self.assertEqual([row["step"] for row in read_rows(root)], [0, 1, 2, 3])
            lines = path.read_text().splitlines(keepends=True)
            path.write_text(lines[0] + lines[0] + "".join(lines[2:]))
            with self.assertRaises((ValueError, RuntimeError)):
                self.make_writer(root, resume_step=3, saved_signature=SIGNATURE)

    def test_resume_requires_matching_signature_or_explicit_saved_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = self.make_writer(root)
            writer.append(step=0, log_loss=-1.0, log_increments=[-2.0] * 4)
            writer.flush()
            changed = {**SIGNATURE, "target_loss": 1e-30}
            with self.assertRaises((ValueError, RuntimeError)):
                BufferedIncomingTrace(root, signature=changed, layer_names=LAYERS, resume_step=1)
            migrated = BufferedIncomingTrace(root, signature=changed, layer_names=LAYERS,
                                             resume_step=1, saved_signature=SIGNATURE)
            migrated.append(step=1, log_loss=-2.0, log_increments=[-3.0] * 4)
            migrated.flush()
            metadata = json.loads((root / "incoming_second_moment.meta.json").read_text())
            self.assertEqual(metadata["signature"], changed)
            self.assertEqual(metadata["rows_written"], 2)
            self.assertEqual(metadata["last_step_written"], 1)


class IncomingRunnerIntegrationTest(unittest.TestCase):
    def test_trace_every_step_including_terminal_matches_loss_and_resume(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(709)
        x = torch.randn(1024, 3, generator=generator, dtype=torch.float64)
        y = torch.where(x[:, 0] > 0, 1.0, -1.0)
        spec = {"name": "toy", "activation_power": 1.0,
                "schedule_power": .25, "learning_rate": 3e-4}

        def run(root, steps, trace, resume=False, stop_immediately=False):
            with mock.patch("sys.argv", ["runner"]):
                args = runner.default_args()
            args.output_dir = root
            args.hidden_widths = (4, 4)
            args.max_steps = steps
            args.record_every = args.coordinate_every = 2
            args.expensive_every = 4
            args.checkpoint_every = 2
            args.checkpoint_every_seconds = args.progress_every = 0
            args.spectral_norm_steps = args.nuclear_norm_steps = 2
            args.sketch_dimension = 4
            args.coordinate_count = 2
            args.full_checkpoint_losses = "1.0"
            args.dataset_sha256 = runner.tensor_sha256(x)
            args.log_second_moment_shadow = True
            args.log_incoming_every_step = trace
            args.beta_policy = "fixed"
            args.beta2_reference = .999
            args.resume = resume
            if stop_immediately:
                args.target_loss = 1.0
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
            for payload in (full, resumed):
                logs = payload["optimizer_state"]["log_second_shadow"]["log_second"]
                for expected, actual in zip(
                        baseline["optimizer_state"]["log_second_shadow"]["log_second"], logs):
                    self.assertTrue(torch.equal(expected, actual))
            rows = read_rows(root / "full/toy")
            self.assertEqual([row["step"] for row in rows], list(range(7)))
            self.assertEqual(rows, read_rows(root / "resume/toy"))
            self.assertEqual(rows[-1]["step"], result["final_step"])
            self.assertAlmostEqual(rows[-1]["log_loss"], math.log(result["final_loss"]), delta=2e-14)
            # D_terminal is prospective.  Exactly rows 0,...,N-1 reconstruct
            # beta^-N v_N; including the final row would be an off-by-one error.
            moment_sums = [float(value.sum()) for value in full["optimizer_state"]["second"]]
            for field, value in zip(FIELDS, (sum(moment_sums), *moment_sums)):
                cumulative_log = np.logaddexp.reduce([row[field] for row in rows[:-1]])
                expected = math.log(value) - result["final_step"] * math.log(.999)
                self.assertAlmostEqual(cumulative_log, expected, delta=3e-12)
            with np.load(root / "full/toy/trajectory_series.npz") as series:
                for i, step in enumerate(series["step"]):
                    row = rows[int(step)]
                    self.assertAlmostEqual(row["log_loss"], math.log(series["loss"][i]), delta=2e-14)
                    factor = math.log1p(-.999) - (int(step) + 1) * math.log(.999)
                    for field, prefix in zip(FIELDS, ("", *(layer + "_" for layer in LAYERS))):
                        expected = factor + 2 * math.log(series[prefix + "gradient_l2"][i])
                        self.assertAlmostEqual(row[field], expected, delta=3e-12)
            stopped, stopped_result = run(root / "stopped", 6, True, stop_immediately=True)
            self.assertEqual(stopped_result["final_step"], 0)
            self.assertEqual(stopped["optimizer_state"]["step_count"], 0)
            self.assertEqual([row["step"] for row in read_rows(root / "stopped/toy")], [0])


if __name__ == "__main__":
    unittest.main()
