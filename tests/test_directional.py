from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from adam_lr.directional.config import DirectionalConfig, load_config
from adam_lr.directional.model import angle_degrees, gradient_and_margins, initialize, margins, normalize
from adam_lr.directional.numerics import numerical_issue
from adam_lr.directional.runner import run


def small_config(layers=3, **run_changes):
    base = DirectionalConfig()
    return replace(base, model=replace(base.model, layers=layers),
                   initialization=replace(base.initialization, sampling="random", sample_count=4),
                   run=replace(base.run, steps=8, **run_changes),
                   logging=replace(base.logging, plots=False, progress_every=0, every=2, geometric_checkpoints=0))


@pytest.mark.parametrize("layers", [2, 3, 4, 6])
def test_initial_margins_and_factor_product(layers):
    cfg = small_config(layers)
    cfg = replace(cfg, model=replace(cfg.model, C=2.5), initialization=replace(cfg.initialization, rho=1.2))
    initial = initialize(cfg)
    z = margins(initial.theta, initial.swapped, cfg.model)
    torch.testing.assert_close(z[:, 0], torch.full_like(z[:, 0], 1.2**layers), rtol=4e-14, atol=0)
    torch.testing.assert_close(z[:, 1], 1.2**layers + initial.controls[:, 0], rtol=4e-14, atol=0)
    torch.testing.assert_close(initial.beta.prod(dim=-1), torch.full_like(z[:, 0], cfg.model.C), rtol=4e-14, atol=0)


@pytest.mark.parametrize("layers", [2, 3, 5])
@pytest.mark.parametrize("zero_factor", [False, True])
def test_gradient_matches_autograd_including_zero_factors(layers, zero_factor):
    cfg = small_config(layers)
    generator = torch.Generator().manual_seed(127)
    theta = (.3 + .1 * torch.randn(6, layers + 1, dtype=torch.float64, generator=generator)).requires_grad_()
    if zero_factor:
        with torch.no_grad():
            theta[0, 2] = 0
            if layers >= 3:
                theta[1, 2:4] = 0
    swapped = torch.tensor([False, True, False, True, False, True])
    analytic, z = gradient_and_margins(theta, swapped, cfg.model)
    automatic = torch.autograd.grad(torch.exp(-margins(theta, swapped, cfg.model)).sum(), theta)[0]
    torch.testing.assert_close(analytic, automatic, rtol=5e-14, atol=1e-14)
    torch.testing.assert_close(z, margins(theta, swapped, cfg.model), rtol=0, atol=0)


def test_gradient_central_difference_independent_loss():
    cfg = small_config()
    theta = torch.tensor([[.25, .6, .8, 1.1]], dtype=torch.float64)
    swapped = torch.tensor([False])
    analytic, _ = gradient_and_margins(theta, swapped, cfg.model)
    def objective(t):
        a, b, c, d = t[0].tolist()
        return np.exp(-c*d*(2*a+b)) + np.exp(-c*d*128*a)
    numerical = torch.empty_like(theta)
    for j in range(4):
        plus, minus = theta.clone(), theta.clone()
        plus[0, j] += 1e-6
        minus[0, j] -= 1e-6
        numerical[0, j] = (objective(plus) - objective(minus)) / 2e-6
    torch.testing.assert_close(analytic, numerical, rtol=2e-8, atol=2e-10)


def test_runner_matches_pytorch_adam_and_autograd(tmp_path):
    cfg = small_config()
    cfg = replace(cfg, optimizer=replace(cfg.optimizer, epsilon=1e-8),
                  schedule=replace(cfg.schedule, power=.25))
    init = initialize(cfg)
    parameter = torch.nn.Parameter(init.theta.clone())
    optimizer = torch.optim.Adam([parameter], lr=.01, betas=(.9, .999), eps=1e-8, foreach=False)
    for step in range(1, cfg.run.steps + 1):
        optimizer.zero_grad()
        loss = torch.exp(-margins(parameter, init.swapped, cfg.model)).sum()
        loss.backward()
        optimizer.param_groups[0]["lr"] = .01 * step**(-.25)
        optimizer.step()
    summary = run(cfg, output=tmp_path / "adam")
    state = torch.load(tmp_path / "adam/final_state.pt", weights_only=True)
    assert summary["accepted_steps"] == cfg.run.steps
    assert summary["stop_reason"] == "step_limit"
    torch.testing.assert_close(state["theta"], parameter, rtol=3e-14, atol=3e-15)
    torch.testing.assert_close(state["first_moment"], optimizer.state[parameter]["exp_avg"], rtol=3e-13, atol=1e-16)
    torch.testing.assert_close(state["second_moment"], optimizer.state[parameter]["exp_avg_sq"], rtol=3e-13, atol=1e-16)


def test_sector_permutation_and_finite_diagnostics(tmp_path):
    summary = run(small_config(), output=tmp_path / "symmetric")
    assert summary["permutation_max_absolute_error"] < 1e-14
    with np.load(tmp_path / "symmetric/history.npz") as data:
        for name in data.files:
            assert np.isfinite(data[name]).all(), name
        assert np.all(np.diff(data["step"]) > 0)
        np.testing.assert_allclose(np.sum(data["unit"]**2, axis=-1), 1, rtol=1e-14)
        np.testing.assert_allclose(np.sum(data["predictor_unit"]**2, axis=-1), 1, rtol=1e-14)
        theta = torch.tensor(data["theta"][-1])
        swapped = torch.tensor(data["sector_swapped"])
        current_gradient, z = gradient_and_margins(theta, swapped, small_config().model)
        # Checkpoint gradients are recomputed at theta_k, never stale g(theta_{k-1}).
        np.testing.assert_array_equal(data["gradient"][-1], current_gradient.numpy())
        np.testing.assert_array_equal(data["margin"][-1], z.numpy())


def test_proposal_guard_restores_state_and_step(tmp_path):
    cfg = small_config()
    cfg = replace(cfg, numerics=replace(cfg.numerics, support_margin_cap=1.1),
                  schedule=replace(cfg.schedule, initial_rate=1.0))
    init = initialize(cfg)
    summary = run(cfg, output=tmp_path / "guard")
    assert summary["accepted_steps"] == 0
    assert summary["stop_reason"] == "support_margin_cap"
    assert summary["guard_event"]["proposed_step"] == 1
    state = torch.load(tmp_path / "guard/final_state.pt", weights_only=True)
    assert state["step"] == 0
    torch.testing.assert_close(state["theta"], init.theta, rtol=0, atol=0)
    assert (state["first_moment"] == 0).all()
    assert (state["second_moment"] == 0).all()
    with np.load(tmp_path / "guard/history.npz") as data:
        np.testing.assert_array_equal(data["step"], [0])


def test_gradient_square_underflow_is_failure():
    cfg = small_config()
    theta = torch.ones(1, 4, dtype=torch.float64)
    gradient = torch.full_like(theta, -1e-200)
    issue = numerical_issue(theta, gradient, torch.ones(1, 2, dtype=torch.float64),
                            theta, theta, step=1, beta2=.999, config=cfg.numerics)
    assert issue["kind"] == "gradient_square_underflow"


def test_zero_gradient_cannot_be_convergence(tmp_path):
    cfg = small_config()
    cfg = replace(cfg, initialization=replace(cfg.initialization, rho=10.0),
                  numerics=replace(cfg.numerics, support_margin_cap=0))
    summary = run(cfg, output=tmp_path / "zero_gradient")
    assert summary["accepted_steps"] == 0
    assert summary["stop_reason"] == "initialization_guard"
    assert summary["guard_event"]["kind"] == "zero_or_nonnegative_gradient"
    assert "limit" in summary["endpoint_interpretation"]


def test_l3_high_rate_numerical_failure_reproduced(tmp_path):
    cfg = small_config()
    cfg = replace(cfg, initialization=replace(cfg.initialization, rho=3.0, sampling="grid", grid_size=3),
                  schedule=replace(cfg.schedule, initial_rate=1.0),
                  numerics=replace(cfg.numerics, support_margin_cap=0))
    summary = run(cfg, output=tmp_path / "historical_failure")
    assert summary["accepted_steps"] < cfg.run.steps
    assert summary["stop_reason"] == "numerical_guard"
    assert summary["guard_event"]["kind"] in {"gradient_square_underflow", "subnormal_gradient_square", "zero_or_nonnegative_gradient"}


def test_normalization_and_angle_are_scale_stable():
    a = torch.tensor([[1e200, 2e200], [1e-200, 2e-200]], dtype=torch.float64)
    expected = torch.tensor([[1/np.sqrt(5), 2/np.sqrt(5)]]*2, dtype=torch.float64)
    torch.testing.assert_close(normalize(a), expected)
    torch.testing.assert_close(angle_degrees(a, -a), torch.full((2,), 180., dtype=torch.float64))
    with pytest.raises(ValueError):
        normalize(torch.zeros(1, 2))


def test_config_validation_and_presets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs/directional").glob("*.toml"):
        load_config(path)
    path = tmp_path / "bad.toml"
    path.write_text("[optimizer]\nbeta_2=0.9\n")
    with pytest.raises(ValueError, match="unknown optimizer"):
        load_config(path)
    cfg = small_config()
    with pytest.raises(ValueError, match="controls"):
        replace(cfg, initialization=replace(cfg.initialization, rho=.01)).validate()
    with pytest.raises(ValueError, match="max_trajectories"):
        replace(cfg, initialization=replace(cfg.initialization, sampling="grid", grid_size=100)).validate()


def test_plots_render_from_real_history(tmp_path):
    cfg = small_config()
    cfg = replace(cfg, logging=replace(cfg.logging, plots=True))
    run(cfg, output=tmp_path / "plot")
    for stem in ("diagnostics", "directions", "endpoint_tetrahedron"):
        for suffix in ("png", "pdf"):
            assert (tmp_path / "plot" / f"{stem}.{suffix}").stat().st_size > 1000


def test_nonempty_output_is_protected(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("existing result")
    with pytest.raises(FileExistsError):
        run(small_config(), output=output)
    assert (output / "keep.txt").read_text() == "existing result"

def test_cli_returns_failure_exit_after_writing_guard_results(tmp_path, monkeypatch):
    from adam_lr.directional.__main__ import main
    path = tmp_path / "underflow.toml"
    path.write_text('[initialization]\nrho=10.0\ngrid_size=1\n[run]\nsteps=1\n[logging]\nplots=false\nprogress_every=0\n[numerics]\nsupport_margin_cap=0.0\n', encoding="utf-8")
    output = tmp_path / "guard_cli"
    monkeypatch.setattr("sys.argv", ["adam_lr.directional", "--config", str(path), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stop_reason"] == "initialization_guard"
    assert (output / "history.npz").exists()

@pytest.mark.parametrize("requested_steps", [1, 2])
def test_future_schedule_overflow_does_not_destroy_accepted_state(tmp_path, requested_steps):
    cfg = small_config()
    cfg = replace(cfg, run=replace(cfg.run, steps=requested_steps),
                  schedule=replace(cfg.schedule, power=-2000.0))
    summary = run(cfg, output=tmp_path / f"overflow{requested_steps}")
    assert summary["accepted_steps"] == 1
    assert summary["stop_reason"] == ("step_limit" if requested_steps == 1 else "numerical_guard")
    with np.load(tmp_path / f"overflow{requested_steps}" / "history.npz") as history:
        np.testing.assert_array_equal(history["step"], [0, 1])
        assert np.isfinite(history["theta"]).all()
