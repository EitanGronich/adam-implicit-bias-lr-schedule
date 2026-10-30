"""Saved direction geometry and configurable depth-2 experiments."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PAPER = Path(__file__).resolve().parents[1] / "paper"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PAPER / "figures" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sphere = load_script("sphere")
prepare_sphere = load_script("prepare_sphere")


@pytest.fixture
def synthetic_run(tmp_path):
    directory = tmp_path / "experiment"
    directory.mkdir()
    points = np.arange(1., 19.).reshape(6, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    sectors = np.array([False, False, False, False, True, True])
    config = {
        "model": {"layers": 2, "B": 3., "M": 17., "Lambda": 2.},
        "initialization": {"rho": 2.},
        "schedule": {"initial_rate": .002, "alpha": -.2},
        "optimizer": {"beta1": .8, "beta2": .99, "epsilon": 1e-8},
        "run": {"dtype": "float64", "steps": 12},
    }
    summary = {"accepted_steps": 12, "requested_steps": 12,
               "stop_reason": "step_limit", "guard_event": None}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    history = {"step": np.array([0, 7, 12]), "unit": np.stack([points]*3),
               "sector_swapped": sectors}
    np.savez(directory / "history.npz", **history)
    return directory, points, sectors, history


def write_csv(path, values):
    np.savetxt(path, values, delimiter=",", fmt="%.17g",
               header="trajectory,sector_swapped,q_u1,q_u2,q_b1", comments="")


def test_saved_paper_directions_are_unit_vectors_on_symmetric_datasets():
    points, sectors = sphere.load_points(sphere.DATA)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1, rtol=0, atol=2e-14)
    np.testing.assert_allclose(points[:142, [1, 0, 2]], points[142:], rtol=0, atol=2e-14)
    np.testing.assert_array_equal(sectors, np.repeat([False, True], 142))


def test_prepare_preserves_arbitrary_trajectory_count_settings_and_sectors(synthetic_run, tmp_path):
    directory, points, sectors, _ = synthetic_run
    output = tmp_path / "sphere.csv"
    prepare_sphere.prepare(directory, output)
    actual, actual_sectors = sphere.load_points(output)
    np.testing.assert_array_equal(actual, points)
    np.testing.assert_array_equal(actual_sectors, sectors)


def test_loader_accepts_a_single_direction(tmp_path):
    point = np.ones(3) / np.sqrt(3)
    path = tmp_path / "sphere.csv"
    write_csv(path, np.r_[[0, 1], point][None])
    values, labels = sphere.load_points(path)
    np.testing.assert_array_equal(values, point[None])
    assert labels.tolist() == [True]


@pytest.mark.parametrize("summary", [
    {"accepted_steps": 11, "guard_event": None},
    {"accepted_steps": 12, "guard_event": {"kind": "nonfinite"}},
    {"accepted_steps": 12},
    {"accepted_steps": 12, "guard_event": None, "stop_reason": "parameter_stagnation_observed"},
])
def test_prepare_rejects_incomplete_or_guarded_runs(synthetic_run, tmp_path, summary):
    directory, _, _, _ = synthetic_run
    (directory / "summary.json").write_text(json.dumps(summary))
    output = tmp_path / "sphere.csv"
    with pytest.raises(ValueError, match="completed, unguarded depth-2"):
        prepare_sphere.prepare(directory, output)
    assert not output.exists()


def test_prepare_rejects_other_parameter_dimensions(synthetic_run, tmp_path):
    directory, _, _, _ = synthetic_run
    config = json.loads((directory / "config.json").read_text())
    config["model"]["layers"] = 3
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="depth-2"):
        prepare_sphere.prepare(directory, tmp_path / "sphere.csv")


@pytest.mark.parametrize("case", ["shape", "order", "nan", "negative", "nonunit", "sector"])
def test_csv_loader_rejects_invalid_geometry_or_labels(tmp_path, case):
    values = np.loadtxt(sphere.DATA, delimiter=",", skiprows=1)
    if case == "shape":
        values = values[:, :-1]
    elif case == "order":
        values[[0, 1]] = values[[1, 0]]
    elif case == "nan":
        values[0, 2] = np.nan
    elif case == "negative":
        values[0, 2] *= -1
    elif case == "nonunit":
        values[0, 2:] *= 2
    elif case == "sector":
        values[0, 1] = 2
    path = tmp_path / "changed.csv"
    write_csv(path, values)
    with pytest.raises(ValueError):
        sphere.load_points(path)


@pytest.mark.parametrize("case", ["horizon", "reversed_steps", "shape", "nan", "nonunit", "sector"])
def test_prepare_rejects_invalid_history_before_writing(synthetic_run, tmp_path, case):
    directory, _, _, history = synthetic_run
    if case == "horizon":
        history["step"][-1] = 11
    elif case == "reversed_steps":
        history["step"] = np.array([12, 7, 0])
    elif case == "shape":
        history["unit"] = history["unit"][:, :-1]
    elif case == "nan":
        history["unit"][-1, 0, 0] = np.nan
    elif case == "nonunit":
        history["unit"][-1, 0] *= 2
    elif case == "sector":
        history["sector_swapped"] = history["sector_swapped"].astype(float)
        history["sector_swapped"][0] = .1
    np.savez(directory / "history.npz", **history)
    output = tmp_path / "sphere.csv"
    with pytest.raises(ValueError):
        prepare_sphere.prepare(directory, output)
    assert not output.exists()
