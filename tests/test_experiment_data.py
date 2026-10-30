"""Lossless experiment archives shared by preprocessing and plotting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment_data", ROOT / "paper/figures/data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def test_updating_section_preserves_other_metrics_and_display_scales(tmp_path):
    path = tmp_path / "experiment.npz"
    original = {"exponent": np.array([-334, 5], dtype=np.int64),
                "mantissa": np.array([[-2., 0., 4.], [np.nan, 1., np.inf]])}
    data.write_sections(path, endpoints=original, error={"mean": np.array([.1, .9])})
    data.write_sections(path, error={"mean": np.array([.2, .8]), "window": 1000})
    restored = data.load_section(path, "endpoints")
    for key, expected in original.items():
        assert restored[key].dtype == expected.dtype
        np.testing.assert_array_equal(restored[key], expected)
    np.testing.assert_array_equal(data.load_section(path, "error")["mean"], [.2, .8])


def test_archive_rejects_pickle_arrays_and_keeps_valid_file_unchanged(tmp_path):
    path = tmp_path / "experiment.npz"
    data.write_sections(path, margins={"seed": np.array([11, 12])})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="object arrays"):
        data.write_sections(path, margins={"unsafe": np.asarray([object()], dtype=object)})
    assert path.read_bytes() == before


def test_missing_section_and_wrong_format_fail_clearly(tmp_path):
    path = tmp_path / "experiment.npz"
    data.write_sections(path, margins={"seed": [11, 12]})
    with pytest.raises(ValueError, match="no 'endpoints' section"):
        data.load_section(path, "endpoints")
    np.savez(path, unrelated=np.arange(3))
    with pytest.raises(ValueError, match="Unsupported experiment archive"):
        data.load_section(path, "margins")


def test_paper_archives_hold_consistent_scientific_axes():
    long = ROOT / "paper/data/long_horizon.npz"
    comparison = ROOT / "paper/data/comparison.npz"
    long_grid = data.load_section(long, "endpoints")
    np.testing.assert_array_equal(data.load_section(long, "secants")["seeds"], long_grid["seed"])
    np.testing.assert_array_equal(data.load_section(long, "linf")["alpha"], long_grid["alpha"])
    np.testing.assert_array_equal(data.load_section(long, "alpha_m03")["seed"], long_grid["seed"])
    for section in ("endpoints_epsilon0", "endpoints_epsilon1e8"):
        grid = data.load_section(comparison, section)
        np.testing.assert_array_equal(grid["seed"], data.load_section(comparison, "error")["seed"])
        np.testing.assert_array_equal(grid["alpha"], data.load_section(comparison, "linf")["alpha"])


def test_weighted_panels_share_the_normalized_endpoint_metrics():
    path = ROOT / "paper/data/comparison.npz"
    grid = data.load_section(path, "endpoints_epsilon1e8")
    panels = data.load_section(path, "weighted")
    np.testing.assert_array_equal(panels["alpha"], grid["alpha"][[0, -1]])
    for panel, schedule in enumerate((0, 4)):
        assert panels[f"a{panel}_exponent"] == grid["exponent"][schedule, 0]
        for trajectory in range(5):
            np.testing.assert_array_equal(panels[f"a{panel}_s{trajectory}_x"],
                                          grid[f"s{schedule}_t{trajectory}_x"])
            np.testing.assert_array_equal(panels[f"a{panel}_s{trajectory}_mantissa"],
                                          grid[f"s{schedule}_t{trajectory}_y"][0])
