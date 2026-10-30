"""Initialization, analytic gradients, Adam parity, and guarded run outputs."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from adam_lr.directional.config import DirectionalConfig, ModelConfig, load_config, rho_lower_bound
from adam_lr.directional.model import gradient_and_margins, initialize, margins, normalize
from adam_lr.directional.numerics import numerical_issue
from adam_lr.directional.runner import run


ROOT = Path(__file__).resolve().parents[1]


def small_config(layers=2):
    base = DirectionalConfig()
    return replace(base, model=replace(base.model, layers=layers),
                   initialization=replace(base.initialization, sampling="random", sample_count=4,
                                          z_max=2., pi_min=.5, pi_max=2.),
                   run=replace(base.run, steps=8),
                   logging=replace(base.logging, plots=False, progress_every=0, every=2))


@pytest.mark.parametrize("layers", [2, 3, 5])
def test_initialization_identities_at_nonunit_normalization(layers):
    config = small_config(layers)
    config = replace(config, model=replace(config.model, Lambda=2.5),
                     initialization=replace(config.initialization, rho=1.2))
    initial = initialize(config)
    z = margins(initial.theta, initial.swapped, config.model)
    torch.testing.assert_close(z[:, 0], torch.full_like(z[:, 0], 1.2**layers), rtol=4e-14, atol=0)
    torch.testing.assert_close(z[:, 1], 1.2**layers + initial.controls[:, 0], rtol=4e-14, atol=0)
    torch.testing.assert_close(initial.pi.prod(-1), torch.full_like(z[:, 0], 2.5), rtol=4e-14, atol=0)
    half = len(initial.theta) // 2
    torch.testing.assert_close(initial.theta[half:, :2], initial.theta[:half, [1, 0]], rtol=0, atol=0)
    torch.testing.assert_close(initial.theta[half:, 2:], initial.theta[:half, 2:], rtol=0, atol=0)


@pytest.mark.parametrize("layers", [2, 3, 5])
@pytest.mark.parametrize("zero_factor", [False, True])
def test_analytic_gradient_matches_autograd_including_zero_factors(layers, zero_factor):
    config = small_config(layers)
    theta = (.3 + .1 * torch.randn(6, layers + 1, dtype=torch.float64,
                                  generator=torch.Generator().manual_seed(127))).requires_grad_()
    if zero_factor:
        with torch.no_grad():
            theta[0, 2] = 0
            if layers > 2:
                theta[1, 2:4] = 0
    swapped = torch.tensor([False, True, False, True, False, True])
    gradient, z = gradient_and_margins(theta, swapped, config.model)
    automatic = torch.autograd.grad(torch.exp(-margins(theta, swapped, config.model)).sum(), theta)[0]
    torch.testing.assert_close(gradient, automatic, rtol=5e-14, atol=1e-14)
    torch.testing.assert_close(z, margins(theta, swapped, config.model), rtol=0, atol=0)


@pytest.mark.parametrize("epsilon", [0., 1e-8])
def test_runner_matches_pytorch_and_autograd(tmp_path, epsilon):
    config = small_config(3)
    config = replace(config, optimizer=replace(config.optimizer, epsilon=epsilon),
                     schedule=replace(config.schedule, initial_rate=.01, alpha=-.25))
    initial = initialize(config)
    theta = torch.nn.Parameter(initial.theta.clone())
    optimizer = torch.optim.Adam([theta], lr=.01, betas=(.9, .999), eps=epsilon, foreach=False)
    for step in range(1, config.run.steps + 1):
        optimizer.zero_grad()
        torch.exp(-margins(theta, initial.swapped, config.model)).sum().backward()
        optimizer.param_groups[0]["lr"] = .01 * step**-.25
        optimizer.step()
    result = run(config, output=tmp_path / "run")
    saved = torch.load(tmp_path / "run/final_state.pt", weights_only=True)
    assert result["accepted_steps"] == config.run.steps and result["stop_reason"] == "step_limit"
    torch.testing.assert_close(saved["theta"], theta, rtol=3e-14, atol=3e-15)
    torch.testing.assert_close(saved["first_moment"], optimizer.state[theta]["exp_avg"], rtol=3e-13, atol=1e-16)
    torch.testing.assert_close(saved["second_moment"], optimizer.state[theta]["exp_avg_sq"], rtol=3e-13, atol=1e-16)


def test_history_records_actual_states_and_symmetric_endpoints(tmp_path):
    config = small_config()
    run(config, output=tmp_path / "run")
    with np.load(tmp_path / "run/history.npz", allow_pickle=False) as history:
        assert set(history.files) == {"step", "theta", "unit", "margin", "loss", "log_loss", "controls", "pi", "sector_swapped"}
        np.testing.assert_array_equal(history["step"], [0, 2, 4, 6, 8])
        np.testing.assert_allclose(np.linalg.norm(history["unit"], axis=-1), 1, atol=2e-15)
        half = history["theta"].shape[1] // 2
        np.testing.assert_array_equal(history["theta"][-1, half:, :2], history["theta"][-1, :half, [1, 0]].T)
        final = torch.tensor(history["theta"][-1])
        z = margins(final, torch.tensor(history["sector_swapped"]), config.model)
        np.testing.assert_array_equal(z.numpy(), history["margin"][-1])
    with pytest.raises(FileExistsError):
        run(config, output=tmp_path / "run")


def test_guard_restores_parameter_moments_and_update_clock(tmp_path, monkeypatch):
    config = small_config()
    reference = replace(config, run=replace(config.run, steps=2))
    run(reference, output=tmp_path / "reference")

    def reject_third(*args, **kwargs):
        if kwargs["step"] == 3:
            return {"kind": "injected_failure"}
        return numerical_issue(*args, **kwargs)

    monkeypatch.setattr("adam_lr.directional.runner.numerical_issue", reject_third)
    result = run(config, output=tmp_path / "guarded")
    assert result["accepted_steps"] == 2 and result["guard_event"]["proposed_step"] == 3
    expected = torch.load(tmp_path / "reference/final_state.pt", weights_only=True)
    actual = torch.load(tmp_path / "guarded/final_state.pt", weights_only=True)
    assert actual["step"] == expected["step"] == 2
    for name in ("theta", "first_moment", "second_moment"):
        assert torch.equal(actual[name], expected[name])


def test_underflow_and_margin_cap_are_failures(tmp_path):
    config = small_config()
    value = torch.ones(1, 3, dtype=torch.float64)
    issue = numerical_issue(value, torch.full_like(value, -1e-200), value[:, :2],
                            value, value, step=1, beta2=.999, config=config.numerics)
    assert issue["kind"] == "gradient_square_underflow"
    config = replace(config, schedule=replace(config.schedule, initial_rate=1.),
                     numerics=replace(config.numerics, support_margin_cap=1.1))
    result = run(config, output=tmp_path / "guard")
    assert result["accepted_steps"] == 0 and result["stop_reason"] == "support_margin_cap"
    config = replace(config, initialization=replace(config.initialization, rho=100.),
                     numerics=replace(config.numerics, support_margin_cap=0.))
    result = run(config, output=tmp_path / "underflow")
    assert result["stop_reason"] == "initialization_guard"


def test_sphere_preset_contains_all_fixed_initializations():
    config = load_config(ROOT / "configs/directional/sphere.toml")
    initial = initialize(config)
    assert len(config.initialization.control_points) == 142
    assert initial.theta.shape == (284, 3)
    assert config.model == ModelConfig(layers=2, B=2., M=35., Lambda=1.)
    assert config.initialization.rho == 1.01
    assert config.run.steps == 600000
    assert config.schedule.initial_rate == .001 and config.schedule.alpha == .15
    assert initial.controls.shape == (284, 2)
    assert hashlib.sha256(initial.theta.numpy().tobytes()).hexdigest() == (
        "1a4e122601f021d36603fded0d2a8149aa97a5ebf1d22be62a456166825c5eb4")


@pytest.mark.parametrize("sampling", ["grid", "random", "explicit"])
def test_canonical_sampling_is_deterministic_and_has_expected_size(sampling):
    config = small_config()
    points = [[0.2, .7], [.8, 1.2]] if sampling == "explicit" else None
    config = replace(config, initialization=replace(config.initialization, sampling=sampling,
                     grid_size=2, sample_count=2, control_points=points))
    first, second = initialize(config), initialize(config)
    assert torch.equal(first.theta, second.theta)
    assert len(first.theta) == (8 if sampling == "grid" else 4)


def test_strict_rho_bound_and_invalid_configuration(tmp_path):
    config = small_config()
    with pytest.raises(ValueError, match="strictly greater"):
        replace(config, initialization=replace(config.initialization, rho=1.)).validate()
    assert rho_lower_bound(ModelConfig(M=8.)) == math.sqrt(9 * math.log(8) / 5)
    with pytest.raises(ValueError, match="max_trajectories"):
        replace(config, initialization=replace(config.initialization, max_trajectories=1)).validate()
    for text in ('[model]\nC=2\n', '[initialization]\ns_min=0\n', '[stopping]\npatience=2\n'):
        path = tmp_path / "bad.toml"
        path.write_text(text)
        with pytest.raises(ValueError, match="unknown"):
            load_config(path)
    stable = normalize(torch.tensor([[1e200, 2e200], [1e-200, 2e-200]], dtype=torch.float64))
    torch.testing.assert_close(stable[0], stable[1], rtol=0, atol=0)


def test_plain_plots_use_saved_training_outputs(tmp_path):
    config = small_config()
    config = replace(config, logging=replace(config.logging, plots=True))
    run(config, output=tmp_path / "plots")
    for name in ("loss", "directions", "endpoint_sphere"):
        for suffix in ("pdf", "png"):
            assert (tmp_path / "plots" / f"{name}.{suffix}").stat().st_size > 1000
    assert not list((tmp_path / "plots").glob('*.html'))
