"""Training matches Adam's recurrence and resumes without changing a trajectory."""
from pathlib import Path
import json
import os
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from adam_lr.mnist import runner
from adam_lr.mnist.config import parse_args
from adam_lr.mnist.data import synthetic_data
from adam_lr.mnist.models import ReLUNetwork, make_scaled_initialization
from adam_lr.mnist.traces import load_trace


ROOT = Path(__file__).resolve().parents[1]
SPEC = dict(name="trial", alpha=-.3, learning_rate=.003)


def arguments(path, *, bias=False, epsilon=0., steps=5, resume=False, widths=()):
    args = runner.default_args()
    args.output_dir, args.dataset, args.dataset_sha256 = path, "synthetic", "test-data"
    args.seed, args.hidden_widths, args.bias, args.epsilon = 29, widths, bias, epsilon
    args.layer_scale, args.max_steps, args.target_loss = .1, steps, 1e-30
    args.trace_buffer_rows, args.checkpoint_every = 2, 2
    args.checkpoint_every_seconds = args.progress_every = 0
    args.resume, args.device = resume, "cpu"
    return args


def dataset():
    images, labels = synthetic_data(8, 3, 42)
    return images - images.mean(0), labels * 2 - 1


@pytest.mark.parametrize("bias,epsilon,widths", [
    (False, 0., ()), (True, 0., ()), (False, 1e-8, ()),
    (True, 1e-8, ()), (False, 1e-8, (4, 3)), (True, 1e-8, (4, 3)),
])
def test_run_matches_pytorch(tmp_path, bias, epsilon, widths):
    torch.set_num_threads(1)
    runner.STOP_REQUESTED = False
    x, y = dataset()
    args = arguments(tmp_path, bias=bias, epsilon=epsilon, widths=widths)
    factory = lambda: ReLUNetwork(input_dim=3, hidden_widths=widths, bias=bias)
    state, _ = make_scaled_initialization(factory, x[:8], layer_scale=.1,
                                          seed=args.seed, initialization_dtype=torch.float64)
    reference = factory().double()
    reference.load_state_dict(state)
    optimizer = torch.optim.Adam(reference.parameters(), lr=.003, betas=(.9, .999),
                                 eps=epsilon, foreach=False)
    for step in range(1, 6):
        reference.zero_grad(set_to_none=True)
        F.softplus(-y * reference(x)).mean().backward()
        optimizer.param_groups[0]["lr"] = .003 * step ** -.3
        optimizer.step()
    result = runner.run_one(args=args, spec=SPEC, x=x, y=y)
    actual = torch.load(tmp_path / "trial/checkpoint.pt", weights_only=False)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(actual["model_state"][name], value, rtol=3e-12, atol=3e-15)
    for i, parameter in enumerate(reference.parameters()):
        for key, reference_key in (("first", "exp_avg"), ("second", "exp_avg_sq")):
            torch.testing.assert_close(actual["optimizer_state"][key][i],
                                       optimizer.state[parameter][reference_key], rtol=3e-12, atol=3e-15)
    assert result["signature"]["epsilon"] == epsilon and result["signature"]["biases"] is bias
    assert result["signature"]["exact_homogeneous"] is (not bias)
    trace = load_trace(tmp_path / "trial")
    assert trace["step"].tolist() == list(range(6))
    assert trace["update_applied"].tolist() == [True] * 5 + [False]
    fields = dict(zip(trace["metric_columns"], trace["values"].T))
    np.testing.assert_allclose(fields["margin_linf"], fields["raw_margin"] /
                               fields["theta_linf"] ** (len(widths) + 1), rtol=3e-14)
    if epsilon == 0:
        np.testing.assert_array_equal(fields["epsilon_update_change_squared"], np.zeros(6))
    else:
        assert np.all(fields["epsilon_update_change_squared"] >= 0)
        assert fields["epsilon_update_change_squared_valid"].tolist() == [1.] * 6


def test_resume_is_bitwise_identical_and_restores_rng(tmp_path):
    x, y = dataset()
    runner.STOP_REQUESTED = False
    random.seed(8)
    np.random.seed(8)
    runner.run_one(args=arguments(tmp_path / "full", bias=True, epsilon=1e-8, steps=7), spec=SPEC, x=x, y=y)
    random.seed(8)
    np.random.seed(8)
    runner.run_one(args=arguments(tmp_path / "resumed", bias=True, epsilon=1e-8, steps=3), spec=SPEC, x=x, y=y)
    random.random()
    np.random.rand(10)
    torch.rand(10)
    runner.run_one(args=arguments(tmp_path / "resumed", bias=True, epsilon=1e-8, steps=7, resume=True), spec=SPEC, x=x, y=y)
    full = torch.load(tmp_path / "full/trial/checkpoint.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/trial/checkpoint.pt", weights_only=False)
    for name in full["model_state"]:
        assert torch.equal(full["model_state"][name], resumed["model_state"][name])
    for key in ("first", "second"):
        assert all(torch.equal(a, b) for a, b in zip(full["optimizer_state"][key], resumed["optimizer_state"][key]))
    assert full["rng_state"]["python"] == resumed["rng_state"]["python"]
    np.testing.assert_array_equal(full["rng_state"]["numpy"][1], resumed["rng_state"]["numpy"][1])
    assert torch.equal(full["rng_state"]["torch"], resumed["rng_state"]["torch"])
    left, right = load_trace(tmp_path / "full/trial"), load_trace(tmp_path / "resumed/trial")
    for key in ("step", "loss", "update_applied", "values"):
        np.testing.assert_array_equal(left[key], right[key])


def test_interrupted_checkpoint_replays_uncommitted_observation(tmp_path, monkeypatch):
    x, y = dataset()
    runner.STOP_REQUESTED = False
    runner.run_one(args=arguments(tmp_path / "full", steps=6), spec=SPEC, x=x, y=y)
    original = runner.InspectableAdam.step
    def interrupt(optimizer, rate):
        if optimizer.step_count == 3:
            raise RuntimeError("interrupted")
        return original(optimizer, rate)
    with monkeypatch.context() as patch:
        patch.setattr(runner.InspectableAdam, "step", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            runner.run_one(args=arguments(tmp_path / "partial", steps=6), spec=SPEC, x=x, y=y)
    checkpoint = torch.load(tmp_path / "partial/trial/checkpoint.pt", weights_only=False)
    assert checkpoint["step"] == 2
    runner.run_one(args=arguments(tmp_path / "partial", steps=6, resume=True), spec=SPEC, x=x, y=y)
    left, right = load_trace(tmp_path / "full/trial"), load_trace(tmp_path / "partial/trial")
    for key in ("loss", "values", "update_applied"):
        np.testing.assert_array_equal(left[key], right[key])


def test_zero_gradient_history_remains_stationary(tmp_path):
    x, y = dataset()
    x[:, 2] = 0
    args = arguments(tmp_path)
    factory = lambda: ReLUNetwork(input_dim=3, hidden_widths=())
    initial, _ = make_scaled_initialization(factory, x[:8], layer_scale=.1,
                                            seed=args.seed, initialization_dtype=torch.float64)
    runner.STOP_REQUESTED = False
    runner.run_one(args=args, spec=SPEC, x=x, y=y)
    state = torch.load(tmp_path / "trial/checkpoint.pt", weights_only=False)
    assert state["model_state"]["output.weight"][0, 2] == initial["output.weight"][0, 2]
    assert state["optimizer_state"]["first"][0][0, 2] == state["optimizer_state"]["second"][0][0, 2] == 0


def test_cli_runs_epsilon_sweep_and_resumes_one_cell(tmp_path):
    config = tmp_path / "experiment.toml"
    config.write_text('''
[experiment]
seed = 31
dtype = "float64"
device = "cpu"
threads = 1
[dataset]
kind = "synthetic"
per_class = 4
input_dim = 3
seed = 42
[model]
hidden_widths = [4, 3]
bias = true
layer_scale = 0.1
[run]
max_steps = 3
target_loss = 1e-30
resume = true
[logging]
trace_buffer_rows = 2
checkpoint_every = 2
checkpoint_every_seconds = 0.0
progress_every = 0
[sweep]
seeds = [31]
epsilon_values = [0.0, 1e-8]
[[schedules]]
name = "trial"
alpha = -0.3
''', encoding="utf-8")
    output = tmp_path / "runs"
    cli = ["--config", str(config), "--output-dir", str(output)]
    runner.main(cli + ["--max-steps", "2"])
    runner.main(cli + ["--epsilon", "0", "--max-steps", "3"])
    experiment = json.loads((output / "experiment.json").read_text())
    assert experiment["sweep"]["epsilon_values"] == [0., 1e-8]
    assert experiment["sweep"]["seeds"] == [31]
    for epsilon, expected in ((0., 3), (1e-8, 2)):
        folder = output / f"seed_31/beta2_0.999/epsilon_{epsilon:g}/trial"
        state = torch.load(folder / "checkpoint.pt", weights_only=False)
        assert state["step"] == state["optimizer_state"]["step_count"] == expected
        assert load_trace(folder)["step"].tolist() == list(range(expected + 1))


def test_cli_dry_run_has_no_data_dependency(tmp_path, capsys):
    output = tmp_path / "not-created"
    runner.main(["--config", str(ROOT / "configs/mnist/comparison.toml"),
                 "--output-dir", str(output), "--data-dir", str(tmp_path / "missing"), "--dry-run"])
    assert len(json.loads(capsys.readouterr().out)) == 50
    assert not output.exists()


def test_first_update_interruption_resumes_from_initial_checkpoint(tmp_path, monkeypatch):
    x, y = dataset()
    runner.STOP_REQUESTED = False
    runner.run_one(args=arguments(tmp_path / "full", steps=3), spec=SPEC, x=x, y=y)
    def interrupt(*args):
        raise RuntimeError("interrupted before first update")
    with monkeypatch.context() as patch:
        patch.setattr(runner.InspectableAdam, "step", interrupt)
        with pytest.raises(RuntimeError, match="before first update"):
            runner.run_one(args=arguments(tmp_path / "partial", steps=3), spec=SPEC, x=x, y=y)
    state = torch.load(tmp_path / "partial/trial/checkpoint.pt", weights_only=False)
    assert state["step"] == state["trace_state"]["rows"] == 0
    runner.run_one(args=arguments(tmp_path / "partial", steps=3, resume=True), spec=SPEC, x=x, y=y)
    np.testing.assert_array_equal(load_trace(tmp_path / "full/trial")["values"],
                                  load_trace(tmp_path / "partial/trial")["values"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_resume_restores_cpu_rng_state_tensors(tmp_path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    x, y = (value.cuda() for value in dataset())
    runner.STOP_REQUESTED = False
    for name, steps, resume in (("full", 3, False), ("resumed", 1, False), ("resumed", 3, True)):
        args = arguments(tmp_path / name, bias=True, epsilon=1e-8, steps=steps, resume=resume)
        args.device = "cuda"
        runner.run_one(args=args, spec=SPEC, x=x, y=y)
    full = torch.load(tmp_path / "full/trial/checkpoint.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/trial/checkpoint.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(full["model_state"][name], resumed["model_state"][name]) for name in full["model_state"])
    assert all(value.device.type == "cpu" for value in resumed["rng_state"]["cuda"])
