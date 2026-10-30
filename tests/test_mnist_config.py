"""Validate the public experiment configuration and reject removed settings."""
from pathlib import Path

import pytest

from adam_lr.mnist.config import from_mapping, parse_args
from adam_lr.mnist.runner import experiment_cases

ROOT = Path(__file__).resolve().parents[1]


def test_presets_and_cli_select_the_same_run_directory():
    path = ROOT / "configs/mnist/comparison.toml"
    complete = list(experiment_cases(parse_args(["--config", str(path)])))
    assert len(complete) == 50
    paths = {str(args.output_dir / spec["name"]) for args, spec, _ in complete}
    assert len(paths) == 50
    selected = parse_args(["--config", str(path), "--seed", "2026100901", "--epsilon", "0",
                           "--schedule", "alpha_neg0p30_eta3e4"])
    [(args, spec, _)] = list(experiment_cases(selected))
    assert str(args.output_dir / spec["name"]) in paths
    long = parse_args(["--config", str(ROOT / "configs/mnist/long_horizon.toml")])
    assert len(list(experiment_cases(long))) == 25
    assert long.hidden_widths == (512, 512) and not long.bias and long.epsilon == 0


@pytest.mark.parametrize("raw", [
    {"model": {"bias": "yes"}}, {"optimizer": {"epsilon": -1}},
    {"optimizer": {"beta2": 1}}, {"sweep": {"epsilon_values": []}},
    {"sweep": {"epsilon_values": [0, 0]}}, {"sweep": {"seeds": [False]}},
    {"logging": {"trace_buffer_rows": 0}}, {"logging": {"coordinate_every": 10}},
    {"logging": {"log_second_moment_shadow": True}}, {"run": {"allow_dtype_promotion": True}},
    {"optimizer": {"beta_policy": "rescaled"}}, {"schedules": [{"name": "bad", "power": -.3}]},
])
def test_invalid_or_removed_settings_are_rejected(raw):
    with pytest.raises(ValueError):
        from_mapping(raw)


def test_new_experiment_axes_and_model_are_configurable():
    args = from_mapping({"model": {"hidden_widths": [7, 5, 3], "bias": True},
        "sweep": {"seeds": [17, 31], "beta2_values": [.95], "epsilon_values": [0., 1e-5]},
        "schedules": [{"name": "steady", "alpha": 0.}, {"name": "decay", "alpha": -.2, "max_steps": 13}]})
    assert args.hidden_widths == (7, 5, 3) and args.bias
    assert len(list(experiment_cases(args))) == 8
    assert args.specs[1]["max_steps"] == 13
