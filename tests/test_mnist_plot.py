"""Basic experiment plots consume the training trace without modifying it."""
from adam_lr.mnist import runner
from adam_lr.mnist.plot import main as plot


def test_plot_a_new_experiment_without_changing_saved_state(tmp_path):
    config = tmp_path / "experiment.toml"
    config.write_text('''
[experiment]
device = "cpu"
threads = 1
[dataset]
kind = "synthetic"
per_class = 4
input_dim = 3
seed = 42
[model]
hidden_widths = [4]
bias = true
layer_scale = 0.1
[optimizer]
epsilon = 1e-8
[run]
max_steps = 3
target_loss = 1e-30
[logging]
trace_buffer_rows = 2
checkpoint_every = 2
progress_every = 0
''', encoding="utf-8")
    runs, output = tmp_path / "runs", tmp_path / "plots"
    runner.main(["--config", str(config), "--output-dir", str(runs)])
    inputs = {p: p.read_bytes() for p in runs.rglob("*") if p.is_file()}
    plot([str(runs), "--output-dir", str(output), "--fields",
          "loss,margin_linf,margin_l2", "--x", "loss-clock"])
    for field in ("loss", "margin_linf", "margin_l2"):
        for extension in ("pdf", "png"):
            assert (output / f"{field}.{extension}").stat().st_size > 1000
    assert all(path.read_bytes() == data for path, data in inputs.items())
