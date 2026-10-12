"""Configuration and arbitrary-depth end-to-end regression tests."""
import copy
import json
from pathlib import Path
from unittest import mock
import numpy as np
import pytest
import torch
from adam_lr.mnist import runner
from adam_lr.mnist.config import from_mapping, parse_args
from adam_lr.mnist.data import synthetic_data
from adam_lr.mnist.models import HomogeneousReLU
from adam_lr.optim.adam import NumericalError

@pytest.mark.parametrize("widths", [(), (4,), (5, 4), (5, 4, 3)])
def test_homogeneity_for_every_configured_depth(widths):
    torch.manual_seed(12)
    model = HomogeneousReLU(input_dim=3, hidden_widths=widths).double()
    x = torch.randn(8, 3, dtype=torch.float64)
    before = model(x).detach()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(2)
    torch.testing.assert_close(model(x), before * 2 ** (len(widths)+1))
    assert all(module.bias is None for module in model.children())

def test_config_validation_and_per_schedule_horizon():
    args = from_mapping({"model": {"hidden_widths": [3, 2, 2]}, "schedules": [{"name":"fast", "power": 0.1, "max_steps": 13}]})
    assert args.hidden_widths == (3, 2, 2)
    assert args.specs[0]["max_steps"] == 13
    for raw in [{"modle": {}}, {"optimizer": {"beta2": 1.0}}, {"model": {"hidden_widths": [0]}}, {"schedules": [{"name":"../escape", "power": 0.1}]}]:
        with pytest.raises(ValueError):
            from_mapping(raw)

def test_production_sweep_preserves_cells():
    args = parse_args(["--config", "configs/mnist/three_layer_eta3e4.toml"])
    assert len(args.sweep_seeds) == 3
    assert args.sweep_beta2_values == [0.9, 0.99, 0.999]
    assert [(s["schedule_power"], s["max_steps"]) for s in args.specs] == [(0.0,100000), (0.1,120000), (0.25,150000), (0.5,350000), (0.7,1200000)]
    selected = parse_args(["--config", "configs/mnist/three_layer_eta3e4.toml", "--seed", "19", "--beta2", "0.99", "--schedule", "p025_eta3e4", "--max-steps", "3"])
    assert selected.sweep_seeds == [19]
    assert selected.sweep_beta2_values == [0.99]
    assert len(selected.specs) == 1 and selected.specs[0]["max_steps"] == 3

def tiny_args(root, widths):
    args = runner.default_args()
    args.output_dir = root
    args.hidden_widths = widths
    args.max_steps = 3
    args.record_every = args.coordinate_every = args.expensive_every = 1
    args.coordinate_count = 2
    args.selection_loss = 1.0
    args.spectral_norm_steps = args.nuclear_norm_steps = 2
    args.sketch_dimension = 4
    args.checkpoint_every = 1
    args.checkpoint_every_seconds = args.progress_every = 0
    args.resume = False
    args.dataset = "synthetic"
    args.dataset_sha256 = "test"
    return args

@pytest.mark.parametrize("widths", [(), (4,), (5, 4, 3)])
def test_arbitrary_depth_run_and_fresh_checkpoint_gradients(tmp_path, widths):
    torch.set_num_threads(1)
    args = tiny_args(tmp_path, widths)
    x, labels = synthetic_data(4, 3, 709)
    y = 2 * labels - 1
    spec = {"name":"toy", "activation_power":1.0, "schedule_power":0.25, "learning_rate":0.003}
    result = runner.run_one(args=args, spec=spec, x=x, y=y)
    assert result["signature"]["affine_layer_count"] == len(widths)+1
    assert result["final_step"] == 3
    checkpoint = torch.load(tmp_path / "toy/checkpoint.pt", weights_only=False)
    model = HomogeneousReLU(input_dim=3, hidden_widths=widths).double()
    model.load_state_dict(checkpoint["model_state"])
    loss, _ = runner.logistic_loss_and_logits(model, x, y)
    assert loss.item() == result["final_loss"]
    with np.load(tmp_path / "toy/trajectory_series.npz") as data:
        assert data["step"].tolist() == [0, 1, 2, 3]
        assert data["loss"][-1] == result["final_loss"]
        assert all(len(data[key]) == 4 for key in data.files)

def test_numerical_failure_preserves_checkpoint_and_shadow(tmp_path):
    args = tiny_args(tmp_path, (4,4))
    args.log_second_moment_shadow = True
    args.log_incoming_every_step = True
    x, labels = synthetic_data(4, 3, 709)
    spec = {"name":"toy", "activation_power":1.0, "schedule_power":0.25, "learning_rate":0.003}
    with mock.patch.object(runner.InspectableAdam, "step", side_effect=NumericalError("injected atomic failure")):
        result = runner.run_one(args=args, spec=spec, x=x, y=2*labels-1)
    assert result["completion_reason"] == "numerical_failure"
    assert result["final_step"] == 0
    checkpoint = torch.load(tmp_path / "toy/checkpoint.pt", weights_only=False)
    assert checkpoint["optimizer_state"]["step_count"] == 0
    assert checkpoint["optimizer_state"]["log_second_shadow"]["step_count"] == 0
    assert json.loads((tmp_path / "toy/results.json").read_text())["numerical_failure"] == "injected atomic failure"

def test_zero_gradient_coordinate_panel_is_defined():
    selection = runner.choose_coordinates(scores={"output":torch.zeros(4)}, nonzero_counts={"output":torch.zeros(4,dtype=torch.int32)}, observations=1, count=2, seed=1)
    assert selection["output"]["persistent_count"] == 0
    assert selection["output"]["sample_type"].tolist() == ["inactive","inactive"]


def test_checkpoint_rejects_truncation_before_mutating():
    model = HomogeneousReLU(input_dim=3, hidden_widths=(4,)).double()
    optimizer = runner.InspectableAdam(model.parameters(), .9, .99, 0.)
    payload = runner.optimizer_payload(optimizer)
    payload["first"] = payload["first"][:-1]
    payload["step_count"] = 4
    with pytest.raises(RuntimeError, match="count"):
        runner.restore_optimizer(optimizer, payload)
    assert optimizer.step_count == 0
    assert all(torch.count_nonzero(v) == 0 for v in optimizer.first)

def test_nonfinite_loss_writes_failure_result(tmp_path):
    args = tiny_args(tmp_path, (4,))
    x, labels = synthetic_data(4, 3, 709)
    spec = {"name":"toy", "activation_power":1.0, "schedule_power":0.25, "learning_rate":0.003}
    with mock.patch.object(runner, "logistic_loss_and_logits", return_value=(torch.tensor(float("nan")), torch.zeros(8))):
        result = runner.run_one(args=args, spec=spec, x=x, y=2*labels-1)
    assert result["completion_reason"] == "numerical_failure"
    assert result["final_loss"] is None
    assert (tmp_path / "toy/checkpoint.pt").is_file()


def test_mnist_pixels_independent_of_default_dtype(monkeypatch):
    from adam_lr.mnist.data import balanced_mnist
    class FakeMNIST:
        data = torch.tensor([[[127,255]], [[1,128]]], dtype=torch.uint8)
        targets = torch.tensor([0,1])
        def __init__(self, *args, **kwargs):
            pass
    monkeypatch.setattr("torchvision.datasets.MNIST", FakeMNIST)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        a, _ = balanced_mnist("unused", True, 1)
        torch.set_default_dtype(torch.float64)
        b, _ = balanced_mnist("unused", True, 1)
    finally:
        torch.set_default_dtype(previous)
    assert torch.equal(a,b)
    assert a.dtype == torch.float64

def test_effective_beta_underflow_is_rejected(tmp_path):
    args = tiny_args(tmp_path, (4,))
    args.beta_policy = "rescaled"
    x, labels = synthetic_data(4,3,709)
    spec = {"name":"toy", "activation_power":1., "schedule_power":.25, "learning_rate":1e-300}
    with pytest.raises(ValueError, match="effective betas"):
        runner.run_one(args=args, spec=spec, x=x, y=2*labels-1)


def test_layer_and_coordinate_ratios_both_use_raw_second_moment():
    model = HomogeneousReLU(input_dim=2, hidden_widths=()).double()
    with torch.no_grad():
        model.output.weight.fill_(1)
    gradient = torch.ones(1,2,dtype=torch.float64)
    result = runner.layer_scalar_diagnostics(model=model, gradients=[gradient], first_raw=torch.full((2,),.1,dtype=torch.float64), second_raw=torch.full((2,),.001,dtype=torch.float64), first_hat=torch.ones(2,dtype=torch.float64), root_second=torch.ones(2,dtype=torch.float64), descent_action=-torch.ones(2,dtype=torch.float64), learning_rate=.001, raw_margin=1., degree=1., spectral_norm_steps=2, nuclear_norm_steps=2, expensive_due=True)
    assert result["output_g2_over_v_mean"] == pytest.approx(1000)
    assert result["output_v_over_g2_mean"] == pytest.approx(.001)


def test_schedule_power_outside_theory_domain_is_still_configurable():
    args = from_mapping({"schedules": [{"name":"steep", "power": 2.0}]})
    assert args.specs[0]["schedule_power"] == 2.0
    assert runner.theory_metadata(1., 2., 3)["regime"] == "unavailable"


def test_coordinate_rho_has_undefined_zero_history_and_zero_for_pure_decay():
    def values(previous,updated,gradient):
        tensors = [torch.tensor([v],dtype=torch.float64) for v in (previous,updated,gradient)]
        prior,next_v,g=tensors
        return runner.coordinate_metrics(gradient=g,parameter=torch.ones(1,dtype=torch.float64),first_hat=g,root_second=torch.ones(1,dtype=torch.float64),second_previous=prior,second_next=next_v,beta2=.99,indices=torch.tensor([0]))["rho_over_c2"].item()
    assert np.isnan(values(0.,.01,1.))
    assert values(1.,.99,0.) == 0.
    expected = np.log1p(.01/.99)/(-np.log(.99))
    assert values(1.,1.,1.) == pytest.approx(expected)
    # g^2 underflows in float64; the log recurrence still resolves its increment.
    assert values(1e-300,.99e-300,1e-200) > 0


@pytest.mark.parametrize("steps, expected_reason", [(1,"max_steps"),(2,"numerical_failure")])
def test_unused_terminal_schedule_overflow_is_not_a_failure(tmp_path,steps,expected_reason):
    args=tiny_args(tmp_path,(4,4))
    args.max_steps=steps
    x,labels=synthetic_data(4,3,709)
    spec={"name":"extreme", "activation_power":1., "schedule_power":-2000., "learning_rate":.003}
    result=runner.run_one(args=args,spec=spec,x=x,y=2*labels-1)
    assert result["final_step"] == 1
    assert result["completion_reason"] == expected_reason
    required={"mnist/runner.py","mnist/models.py","mnist/data.py","mnist/matrix_norms.py","mnist/coordinate_diagnostics.py","mnist/log_second_moment.py","mnist/incoming_trace.py","optim/adam.py"}
    assert set(result["signature"]["source_sha256"]) == required
    assert all(len(value)==64 for value in result["signature"]["source_sha256"].values())
    if steps==1:
        with np.load(tmp_path/"extreme/trajectory_series.npz") as data:
            assert data["step"].tolist()==[0,1]
            assert np.isnan(data["learning_rate"][-1])
            assert np.isnan(data["applied_update_l2"][-1])
            assert np.isfinite(data["loss"][-1])


@pytest.mark.parametrize("scale,widths", [(1e-50,(4,)),(1e40,())])
def test_initialization_cast_cannot_erase_or_overflow_weights(tmp_path,scale,widths):
    args=tiny_args(tmp_path,widths)
    args.layer_scale=scale
    x,labels=synthetic_data(4,3,709)
    spec={"name":"cast", "activation_power":1., "schedule_power":0., "learning_rate":.003}
    with pytest.raises(ValueError,match="initialization.*not representable"):
        runner.run_one(args=args,spec=spec,x=x.float(),y=(2*labels-1).float())


@pytest.mark.parametrize("resumed_target,expected", [(.5,True),(.3,False)])
def test_resume_remembers_prior_crossing_for_current_target(tmp_path,resumed_target,expected):
    args=tiny_args(tmp_path,(4,))
    args.max_steps=1
    args.target_loss=.5
    args.stop_at_target=False
    x,labels=synthetic_data(4,3,709)
    spec={"name":"resume", "activation_power":1., "schedule_power":0., "learning_rate":.003}
    original_loss=runner.logistic_loss_and_logits
    values=iter([.4,.8,.8])
    def controlled_loss(*arguments):
        loss,logits=original_loss(*arguments)
        return loss-loss.detach()+next(values),logits
    with mock.patch.object(runner,"logistic_loss_and_logits",side_effect=controlled_loss):
        first=runner.run_one(args=args,spec=spec,x=x,y=2*labels-1)
        assert first["target_reached"] is True and first["final_loss"] == .8
        args.resume=True
        args.finalize_checkpoint=True
        args.max_steps=2
        args.target_loss=resumed_target
        args.allow_target_extension=True
        resumed=runner.run_one(args=args,spec=spec,x=x,y=2*labels-1)
    assert resumed["target_reached"] is expected
    assert resumed["final_step"] == 1
