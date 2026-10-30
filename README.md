# Watch Your Step: Adam's Margin Maximization Bias Depends on the Learning Rate Schedule

Experiments and figure code for the paper by Eitan Gronich and Gal Vardi.
The experiments compare Adam learning-rate schedules
`eta_k = eta_0 * k**alpha`, starting at update `k = 1`.

## Installation

Use Python 3.12 and a PyTorch installation suitable for your CPU or GPU:

```sh
git clone https://github.com/EitanGronich/adam-implicit-bias-lr-schedule.git
cd adam-implicit-bias-lr-schedule
python -m pip install -e ".[test,paper]"
```

A GPU is recommended for full MNIST runs. Figure rendering runs on CPU.

## Run a new MNIST experiment

Choose a preset, copy it, and edit its TOML settings:

| Preset | Model | Initialization scale | Adam epsilon |
|---|---|---|---|
| `configs/mnist/long_horizon.toml` | Bias-free ReLU network | `1e-4` | `0` |
| `configs/mnist/comparison.toml` | ReLU network with biases | `1` | `0`, `1e-8` |

```sh
python -m adam_lr.mnist --config configs/mnist/comparison.toml
```

The configuration controls the dataset, network widths and biases, initialization,
Adam parameters, schedule exponents, seeds, stopping rules, and logging.
The supplied presets use 1,024 MNIST images and a `784 -> 512 -> 512 -> 1`
network. Set `dataset.download = true` to download MNIST, or point
`dataset.data_dir` to a directory containing `MNIST/raw/`.

For a short check of one configuration:

```sh
python -m adam_lr.mnist --config configs/mnist/comparison.toml --seed 2026100901 --schedule alpha_neg0p30_eta3e4 --epsilon 1e-8 --max-steps 100 --output-dir runs/check
python -m adam_lr.mnist.plot runs/check --output-dir output/check
```

The runner saves configuration, the training trace, and checkpoints for each run.
With `run.resume = true`, running the same command continues an existing run.
See [MNIST experiments](docs/mnist.md) for configuration and output details.

## Run a width-1 experiment

```sh
python -m adam_lr.directional --config configs/directional/sphere.toml
```

Edit the model, initialization controls, optimizer, and schedule in the TOML
file to run a different experiment. See [width-1 experiments](docs/directional.md).

## Render figures

The bundled data contain one archive for each MNIST experiment and a table of
width-1 endpoint directions. Render the paper figures with:

```sh
python paper/figures/render.py --output-dir output/paper
```

The schematic requires XeLaTeX with TikZ and the `standalone` class. Add
`--skip-tex` to render only numerical figures, or `--format both` for PNGs too.
[Figure preparation](docs/paper.md) explains how to prepare the same plots from
a new experiment.

## Development

```sh
python -m pytest -q
```

```text
configs/        Experiment settings
src/adam_lr/    Optimizers, training, and trace storage
paper/data/     Figure inputs, grouped by experiment
paper/figures/  Data preparation and plotting
paper/diagrams/ TikZ schematic
tests/          Numerical and integration tests
docs/           Experiment instructions
```
