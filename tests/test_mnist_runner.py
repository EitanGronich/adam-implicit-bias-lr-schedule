"""Terminal observations and loss snapshots describe the evaluated model state."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from adam_lr.mnist import runner
from adam_lr.mnist.models import ReLUNetwork
from adam_lr.mnist.traces import load_trace
from test_mnist_training import arguments, dataset, SPEC


def test_initial_target_stops_without_an_update(tmp_path):
    x, y = dataset()
    args = arguments(tmp_path)
    args.target_loss = 2.
    runner.STOP_REQUESTED = False
    result = runner.run_one(args=args, spec=SPEC, x=x, y=y)
    assert result["completion_reason"] == "target_loss" and result["final_step"] == 0
    trace = load_trace(tmp_path / "trial")
    assert trace["step"].tolist() == [0] and trace["update_applied"].tolist() == [False]
    state = torch.load(tmp_path / "trial/checkpoint.pt", weights_only=False)
    assert state["step"] == 0
    assert all(value.count_nonzero() == 0 for value in state["optimizer_state"]["second"])


def test_saved_states_have_the_logged_loss_and_committed_moment_count(tmp_path):
    x, y = dataset()
    args = arguments(tmp_path)
    args.full_checkpoint_losses = [1., .65]
    runner.STOP_REQUESTED = False
    runner.run_one(args=args, spec=SPEC, x=x, y=y)
    trace = load_trace(tmp_path / "trial")
    states = list((tmp_path / "trial/loss_checkpoints").glob("*.pt"))
    assert any(path.name == "initial.pt" for path in states)
    assert any(path.name.startswith("step_") for path in states)
    for path in states:
        state = torch.load(path, weights_only=False)
        step = state["step"]
        assert step == state["optimizer_state"]["step_count"]
        model = ReLUNetwork(input_dim=3, hidden_widths=()).double()
        model.load_state_dict(state["model_state"])
        with torch.no_grad():
            loss = float(F.softplus(-y * model(x)).mean())
        assert loss == trace["loss"][step]
    final = torch.load(tmp_path / "trial/checkpoint.pt", weights_only=False)
    # The terminal gradient observation does not advance Adam's committed moments.
    assert final["step"] == args.max_steps
    assert len(trace["step"]) == args.max_steps + 1


def test_rejected_proposal_never_counts_as_applied(tmp_path, monkeypatch):
    x, y = dataset()
    args = arguments(tmp_path)
    def reject(*args, **kwargs):
        raise runner.NumericalError("injected nonfinite proposal")
    monkeypatch.setattr(runner.InspectableAdam, "step", reject)
    runner.STOP_REQUESTED = False
    result = runner.run_one(args=args, spec=SPEC, x=x, y=y)
    assert result["completion_reason"] == "numerical_failure" and result["final_step"] == 0
    state = torch.load(tmp_path / "trial/checkpoint.pt", weights_only=False)
    assert state["step"] == 0
    assert all(value.count_nonzero() == 0 for value in state["optimizer_state"]["first"])
    rows = np.fromfile(tmp_path / "trial/trace.bin", dtype="<f8").reshape(1, -1)
    assert rows[0, 2] == 0
    with pytest.raises(ValueError, match="numerically failed"):
        load_trace(tmp_path / "trial")


def test_nonfinite_data_is_rejected_before_files_are_created(tmp_path):
    x, y = dataset()
    x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite nonempty"):
        runner.run_one(args=arguments(tmp_path), spec=SPEC, x=x, y=y)
    assert not (tmp_path / "trial").exists()


def test_unrepresentable_learning_rate_stops_at_last_committed_state(tmp_path):
    x, y = dataset()
    runner.STOP_REQUESTED = False
    result = runner.run_one(args=arguments(tmp_path, steps=3),
                            spec={**SPEC, "alpha": -2000.}, x=x, y=y)
    assert result["completion_reason"] == "numerical_failure"
    assert result["final_step"] == 1 and "underflowed" in result["numerical_error"]
    state = torch.load(tmp_path / "trial/checkpoint.pt", weights_only=False)
    assert state["step"] == 1


def test_column_labels_preserve_loss_raw_margin_and_training(tmp_path):
    x, y = dataset()
    runner.STOP_REQUESTED = False
    runner.run_one(args=arguments(tmp_path / "flat", steps=3), spec=SPEC, x=x, y=y)
    runner.run_one(args=arguments(tmp_path / "column", steps=3), spec=SPEC, x=x, y=y[:, None])
    flat, column = load_trace(tmp_path / "flat/trial"), load_trace(tmp_path / "column/trial")
    np.testing.assert_array_equal(flat["loss"], column["loss"])
    np.testing.assert_array_equal(flat["values"], column["values"])
    left = torch.load(tmp_path / "flat/trial/checkpoint.pt", weights_only=False)
    right = torch.load(tmp_path / "column/trial/checkpoint.pt", weights_only=False)
    assert all(torch.equal(left["model_state"][name], right["model_state"][name]) for name in left["model_state"])
