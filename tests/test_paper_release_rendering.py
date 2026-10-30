"""Scientific invariants and entry points of the manuscript reproduction path."""
from pathlib import Path
import importlib.util
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / "paper/figures" / (name + ".py")
    spec = importlib.util.spec_from_file_location("paper_release_" + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_first_passage_uses_chronological_bracket_after_reversal():
    prepare = module("prepare_traces")
    actual = prepare.interpolate_first_passage(
        np.array([0., 2., 1., 3.]), np.array([0., 2., 100., 3.]),
        np.array([2., 2.5, 3.]))
    np.testing.assert_array_equal(actual, [2., 27.25, 3.])


def test_trailing_average_requires_complete_valid_window():
    prepare = module("prepare_traces")
    values = np.array([1., 2., np.nan, 4., 5., 6.])
    actual = prepare.moving_average(values, np.ones(6, dtype=bool), window=3)
    np.testing.assert_array_equal(np.isnan(actual), [True, True, True, True, True, False])
    assert actual[-1] == 5.


def test_fixed_five_seed_summary_and_sample_standard_error():
    prepare = module("prepare_traces")
    curves = [(np.array([0., 1., 2.]), np.array([0., 1., 2.]) + i) for i in range(5)]
    grid, mean, se = prepare.common_grid_summary(curves, spacing=.5)
    np.testing.assert_allclose(mean, grid + 2)
    np.testing.assert_allclose(se, np.sqrt(2.5) / np.sqrt(5))
    with pytest.raises(ValueError, match="at least two"):
        prepare.common_grid_summary(curves[:1])


def test_summaries_accept_new_seed_sets():
    prepare = module("prepare_traces")
    curves = [(np.array([0., 1., 2.]), np.array([0., 1., 2.]) + i) for i in range(3)]
    grid, mean, se = prepare.common_grid_summary(curves, spacing=.5)
    np.testing.assert_allclose(mean, grid + 1)
    np.testing.assert_allclose(se, 1 / np.sqrt(3))


def test_invalid_interior_samples_do_not_change_seed_cohort():
    prepare = module("prepare_traces")
    curves = [(np.array([0., 1., 2., 3.]), np.array([0., 1., 2., 3.])) for _ in range(5)]
    curves[-1] = (curves[-1][0], np.array([0., 1., np.nan, 3.]))
    with pytest.raises(ValueError, match="incomplete interpolation"):
        prepare.common_grid_summary(curves, spacing=.5)


def test_renderer_lists_exact_manuscript_assets(capsys):
    render = module("render")
    render.main(["--list"])
    paths = capsys.readouterr().out.splitlines()
    assert len(paths) == len(set(paths)) == 20
    assert "figures/mnist/alpha_03_all_seeds.pdf" in paths
    assert "figures/epsilon_1e8/epsilon1e8_scale1_jumbo.pdf" in paths
    assert "figures/adam_construction_tikz.pdf" in paths
    render.main(["--list", "--skip-tex"])
    assert len(capsys.readouterr().out.splitlines()) == 19


def test_error_bundle_uses_scale_one_and_five_seeds():
    data = module("data").load_section(ROOT / "paper/data/comparison.npz", "error")
    assert float(data["initialization_scale"]) == 1.
    assert bool(data["bias"])
    assert float(data["epsilon"]) == 1e-8
    assert int(data["window"]) == 1000
    assert len(np.unique(data["seed"])) == 5
    for i, alpha in enumerate((0., -.1, -.3, -.5, -.7)):
        assert float(data[f"g{i}_alpha"]) == alpha
        assert int(data[f"g{i}_n"]) == 5
        x, mean, se = (data[f"g{i}_relative_{key}"] for key in ("x", "mean", "se"))
        assert x.shape == mean.shape == se.shape
        assert np.isfinite([x, mean, se]).all()
        assert np.all(np.diff(x) > 0) and np.all(se >= 0)


@pytest.mark.parametrize("filename,section", [
    ("long_horizon.npz", "endpoints"),
    ("comparison.npz", "endpoints_epsilon0"),
    ("comparison.npz", "endpoints_epsilon1e8"),
])
def test_bundled_endpoint_margins_use_normalized_metric_scale(filename, section):
    data = module("data").load_section(ROOT / "paper/data" / filename, section)
    assert data["exponent"].shape == (5, 5)
    np.testing.assert_array_equal(data["exponent"], np.zeros((5, 5), dtype=np.int64))
    for schedule in range(5):
        for trajectory in range(5):
            assert np.isfinite(data[f"s{schedule}_t{trajectory}_y"]).all()
    assert np.any(data["s0_t0_y"] < 0)


def test_error_export_uses_shared_data_and_paper_size(tmp_path):
    pytest.importorskip("ultraplot")
    pymupdf = pytest.importorskip("pymupdf")
    module("render")  # Make the repository root importable from any test CWD.
    from paper.figures.error import render
    render(ROOT / "paper/data/comparison.npz", tmp_path, formats=("pdf",))
    with pymupdf.open(tmp_path / "err.pdf") as document:
        assert len(document) == 1
        assert document[0].rect.width * 25.4 / 72 == pytest.approx(120, abs=.2)
        assert "Seed" not in document[0].get_text()


def test_custom_individual_exponent_has_truthful_filename_and_legend(tmp_path):
    pytest.importorskip("ultraplot")
    pymupdf = pytest.importorskip("pymupdf")
    module("render")
    from paper.figures.data import load_section
    from paper.figures.mnist import appendix
    data = load_section(ROOT / "paper/data/long_horizon.npz", "alpha_m03")
    data["alpha"] = np.asarray(-.4)
    data["seed"] = np.arange(11, 16)
    appendix(data, tmp_path, ("pdf",))
    assert not (tmp_path / "alpha_03_all_seeds.pdf").exists()
    with pymupdf.open(tmp_path / "alpha_selected_all_seeds.pdf") as document:
        assert "0.4" in document[0].get_text()


@pytest.mark.parametrize("kind", ["linf", "appendix"])
def test_newly_prepared_signed_margins_render_on_log_axes(tmp_path, monkeypatch, kind):
    pytest.importorskip("ultraplot")
    pymupdf = pytest.importorskip("pymupdf")
    module("render")
    from paper.figures import mnist

    clock = np.array([.16, 1., 2., 4., 6., 8.])
    margins = np.array([-2., 0., 1., 10., 100., 1000.])

    class SavedRun:
        def read_history(self):
            return dict(loss=10.**-clock, loss_clock=clock, linf_margin=margins)

    seeds, alphas = tuple(range(11, 16)), (0., -.1, -.3, -.5, -.7)
    config = dict(biases=False, hidden_widths=[], layer_scale=1., beta1=.9,
                  beta2=.999, learning_rate=.01, target_loss=1e-8)
    sweep = dict(seeds=seeds, alphas=alphas, epsilons=(0.,), kind="long_horizon",
                 config=config,
                 runs={(0., alpha, seed): SavedRun() for alpha in alphas for seed in seeds})
    sections = module("prepare_traces").summarize_traces(sweep, secant_window=2)
    original_save = mnist.save

    def check_and_save(fig, *args):
        ax = fig.axes[0]
        assert ax.get_ylim()[0] > 0
        if kind == "appendix":
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), margins)
            assert all(line.get_clip_on() for line in ax.lines)
        original_save(fig, *args)

    monkeypatch.setattr(mnist, "save", check_and_save)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Attempt to set non-positive ylim.*")
        if kind == "linf":
            assert sections["linf"]["s0_x"][0] < 2
            mnist.linf_margin(sections["linf"], tmp_path, ("pdf",))
            stem = "panel_linf_margin"
        else:
            mnist.appendix(sections["alpha_m03"], tmp_path, ("pdf",))
            stem = "alpha_03_all_seeds"
    with pymupdf.open(tmp_path / f"{stem}.pdf") as document:
        assert len(document) == 1
