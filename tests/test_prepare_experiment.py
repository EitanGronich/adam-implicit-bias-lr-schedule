"""Numerical and integration checks for preparing a new experiment."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from paper.figures import prepare
from paper.figures import prepare_endpoints as endpoint
from paper.figures import prepare_traces as traces
from paper.figures.data import load_section, write_sections


def metadata(seed, alpha, epsilon=0., *, bias=False):
    config = dict(seed=seed, alpha=alpha, epsilon=epsilon, biases=bias,
                  architecture=[2, 1], hidden_widths=[], sample_count=8, per_class=4,
                  dataset_kind="synthetic", data_seed=709, centered=True,
                  dataset_sha256="same-data", initialization_sha256=str(seed),
                  layer_scale=1., beta1=.9, beta2=.95, dtype="float64",
                  learning_rate=.01, target_loss=1e-12, max_steps=20,
                  normalization_degree=1.)
    return dict(signature=config, final_step=20, final_loss=.1,
                completion_reason="max_steps", target_reached=False)


def write_results(folder, record):
    folder.mkdir(parents=True)
    (folder / "results.json").write_text(json.dumps(record), encoding="utf-8")


def test_sweep_uses_actual_axes_and_rejects_missing_runs(tmp_path):
    for seed in (11, 29, 83):
        for alpha in (0., -.4):
            write_results(tmp_path / f"seed_{seed}" / f"a{alpha}", metadata(seed, alpha))
    result = traces.load_sweep(tmp_path)
    assert result["seeds"] == (11, 29, 83)
    assert result["alphas"] == (0., -.4)
    assert result["epsilons"] == (0.,)
    assert len(result["runs"]) == 6
    (tmp_path / "seed_29/a-0.4/results.json").unlink()
    with pytest.raises(ValueError, match="Missing"):
        traces.load_sweep(tmp_path)


def test_sweep_rejects_unpaired_initializations_and_incompatible_settings(tmp_path):
    left, right = metadata(11, 0.), metadata(11, -.4)
    right["signature"]["initialization_sha256"] = "different"
    write_results(tmp_path / "left", left)
    write_results(tmp_path / "right", right)
    with pytest.raises(ValueError, match="Initial parameters differ"):
        traces.load_sweep(tmp_path)
    right["signature"]["initialization_sha256"] = "11"
    right["signature"]["layer_scale"] = .1
    (tmp_path / "right/results.json").write_text(json.dumps(right))
    with pytest.raises(ValueError, match="incompatible"):
        traces.load_sweep(tmp_path)


def test_configured_sweep_detects_a_wholly_missing_seed(tmp_path):
    for seed in (11, 29):
        write_results(tmp_path / f"seed_{seed}", metadata(seed, 0.))
    planned = {"sweep": {"seeds": [11, 29, 83], "epsilon_values": [0.],
                         "schedules": [{"name": "constant", "alpha": 0.}]}}
    (tmp_path / "experiment.json").write_text(json.dumps(planned))
    with pytest.raises(ValueError, match="1 missing"):
        traces.load_sweep(tmp_path)


def test_failed_run_and_single_seed_are_not_summarized(tmp_path):
    record = metadata(4, 0.)
    write_results(tmp_path / "single", record)
    with pytest.raises(ValueError, match="at least two"):
        traces.load_sweep(tmp_path)
    record["completion_reason"] = "numerical_failure"
    (tmp_path / "single/results.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="did not finish normally"):
        traces.load_sweep(tmp_path)


def test_terminal_metric_uses_actual_beta2_and_unapplied_gradient():
    class Scalar(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x).reshape(-1)
    model = Scalar(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.2, -.1]], dtype=torch.float64))
    state = dict(signature={"beta2": .92},
                 model_state={name: value.detach().clone() for name, value in model.state_dict().items()},
                 optimizer_state={"step_count": 8, "second": [torch.full_like(model.weight, .004)]})
    original = copy.deepcopy(state)
    x = torch.tensor([[1., 2.], [.5, -1.], [-.2, .4]], dtype=torch.float64)
    y = torch.tensor([1., -1., 1.], dtype=torch.float64)
    F.softplus(-y * model(x)).mean().backward()
    g = model.weight.grad.numpy().reshape(-1).copy()
    expected = np.sqrt((.92 * .004 + .08 * g**2) / .92**9)
    expected /= np.linalg.norm(expected)
    weights, step, theta, _, raw = endpoint.prospective_metric(model, state, x, y)
    np.testing.assert_allclose(weights, expected, rtol=3e-15)
    np.testing.assert_allclose(np.linalg.norm(weights), 1., rtol=3e-15)
    assert step == 8
    sign, logarithms = endpoint.signed_log_margins(theta, raw, weights[None], degree=1.)
    np.testing.assert_allclose(sign * 10**logarithms, [raw / np.sqrt(np.dot(expected, theta**2))],
                               rtol=3e-15)
    assert torch.equal(state["model_state"]["weight"], original["model_state"]["weight"])
    assert torch.equal(state["optimizer_state"]["second"][0], original["optimizer_state"]["second"][0])


@pytest.mark.parametrize("factor,step", [(1e-150, 8), (1., 1000000000), (1e150, 8)])
def test_normalized_metric_cancels_global_scale_without_overflow(monkeypatch, factor, step):
    model = torch.nn.Linear(2, 1, bias=False).double()
    state = dict(signature={"beta2": .92}, model_state=model.state_dict(),
                 optimizer_state={"step_count": step,
                                  "second": [torch.tensor([[.004, .008]], dtype=torch.float64) * factor**2]})

    def evaluate(*args, **kwargs):
        model.weight.grad = torch.tensor([[.2, .3]], dtype=torch.float64) * factor
        return np.array([2., 3.]), .1, -2.

    monkeypatch.setattr(endpoint, "evaluate", evaluate)
    weights, actual_step, theta, _, raw = endpoint.prospective_metric(model, state, None, None)
    expected = np.sqrt(.92 * np.array([.004, .008]) + .08 * np.array([.2, .3])**2)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(weights, expected, rtol=3e-15)
    assert actual_step == step
    sign, logs = endpoint.signed_log_margins(theta, raw, weights[None])
    np.testing.assert_allclose(sign * 10**logs, [raw / (expected @ theta**2)**1.5], rtol=3e-15)


def test_display_series_keeps_signed_margins_below_float_range():
    sign, logs = -1., np.array([-400.])
    exponents, values = endpoint.display_series([logs[:, None]], [np.array([sign])])
    assert exponents[0] < -308 and values[0][0, 0] < 0
    np.testing.assert_allclose(np.log10(abs(values[0][0, 0])) + exponents[0], logs[0],
                               rtol=0, atol=2e-13)


def test_cli_rejects_invalid_window_before_loading(tmp_path):
    with pytest.raises(SystemExit):
        prepare.main(["--runs-dir", str(tmp_path), "--output", str(tmp_path / "out.npz"),
                      "--error-window", "0"])


def test_individual_seed_panel_records_its_actual_selected_alpha():
    class SavedRun:
        def read_history(self):
            return dict(loss=np.exp(-np.arange(6) * .1),
                        loss_clock=np.arange(6) * .1 / np.log(10),
                        linf_margin=np.arange(6, dtype=float))
    seeds, alphas = (11, 29), (0., -.4, -.9)
    sweep = dict(seeds=seeds, alphas=alphas, epsilons=(0.,), kind="long_horizon",
                 config=metadata(11, 0.)["signature"],
                 runs={(0., alpha, seed): SavedRun() for alpha in alphas for seed in seeds})
    default = traces.summarize_traces(sweep, secant_window=2)
    assert float(default["alpha_m03"]["alpha"]) == -.4
    selected = traces.summarize_traces(sweep, secant_window=2, selected_alpha=0.)
    assert float(selected["alpha_m03"]["alpha"]) == 0.
    with pytest.raises(ValueError, match="selected alpha is absent"):
        traces.summarize_traces(sweep, secant_window=2, selected_alpha=-.3)


def test_preparing_again_replaces_old_sections_instead_of_mixing_experiments(tmp_path, monkeypatch):
    output = tmp_path / "experiment.npz"
    write_sections(output, obsolete={"value": [1.]})
    monkeypatch.setattr(prepare, "load_sweep", lambda *args, **kwargs: {})
    monkeypatch.setattr(prepare, "summarize_traces", lambda *args, **kwargs:
                        {"settings": {"seed": np.array([11, 29])}})
    monkeypatch.setattr(prepare, "summarize_endpoints", lambda *args: {})
    prepare.main(["--runs-dir", str(tmp_path), "--output", str(output)])
    np.testing.assert_array_equal(load_section(output, "settings")["seed"], [11, 29])
    with pytest.raises(ValueError, match="no 'obsolete' section"):
        load_section(output, "obsolete")


@pytest.mark.parametrize("bias,epsilons", [(False, (0.,)), (True, (0., 1e-8))])
def test_real_new_experiment_prepares_one_archive(tmp_path, monkeypatch, bias, epsilons):
    from adam_lr.mnist import runner
    from adam_lr.mnist.data import synthetic_data
    images, labels = synthetic_data(4, 2, 709)
    x, y = images - images.mean(0, keepdim=True), labels * 2 - 1
    torch.set_num_threads(1)
    monkeypatch.setattr(runner, "STOP_REQUESTED", False)
    for seed in (11, 29):
        for epsilon in epsilons:
            args = runner.default_args()
            args.output_dir = tmp_path / f"seed_{seed}" / f"epsilon_{epsilon:g}"
            args.seed, args.bias, args.epsilon = seed, bias, epsilon
            args.dataset, args.data_seed, args.input_dim = "synthetic", 709, 2
            args.per_class, args.hidden_widths, args.layer_scale = 4, (), 1.
            args.max_steps, args.target_loss = 60, 1e-12
            args.checkpoint_every = 10
            args.checkpoint_every_seconds = args.progress_every = 0
            args.full_checkpoint_losses = [0.7, 0.5]
            args.trace_buffer_rows = 4
            args.beta1, args.beta2 = .9, .95
            args.dataset_sha256 = endpoint.tensor_hash({"images": images, "labels01": labels})
            args.resume, args.device = False, "cpu"
            runner.run_one(args=args,
                spec={"name": "constant", "alpha": 0., "learning_rate": .03},
                x=x, y=y)
    output = tmp_path / "prepared.npz"
    assert prepare.main(["--runs-dir", str(tmp_path), "--output", str(output),
                         "--secant-window", "5", "--error-window", "3"]) == 0
    settings = load_section(output, "settings")
    np.testing.assert_array_equal(settings["seed"], [11, 29])
    assert bool(settings["biases"]) == bias
    if bias:
        for section in ("linf", "l2", "error", "endpoints_epsilon0", "endpoints_epsilon1e8", "weighted"):
            assert load_section(output, section)
        np.testing.assert_array_equal(load_section(output, "linf")["epsilons"], [0., 1e-8])
        assert int(load_section(output, "error")["window"]) == 3
    else:
        for section in ("linf", "secants", "alpha_m03", "endpoints"):
            assert load_section(output, section)
        assert load_section(output, "endpoints")["exponent"].shape == (1, 2)
        np.testing.assert_array_equal(load_section(output, "secants")["x"], np.arange(1, 6))
        assert float(load_section(output, "alpha_m03")["alpha"]) == 0.
