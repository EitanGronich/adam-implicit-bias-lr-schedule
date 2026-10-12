# Adam implicit bias and learning-rate schedules

Two experiment families using **actual, bias-corrected discrete Adam**:

1. Full-batch odd/even MNIST classification with bias-free homogeneous ReLU networks. The main presets use three weight layers, widths `784 → 1024 → 1024 → 1`, and 1,024 training examples. Logging includes loss, margins, parameter/gradient/moment geometry, fixed coordinate panels, loss-triggered snapshots, and optional log-space second-moment traces.
2. Low-dimensional homogeneous product models for observing directional behavior. Directions and their changes are measured from trained parameter vectors. This package contains no ODE solver and no frozen-weight formula for predicting a limiting direction.

The original source folders are preserved separately. This repository contains source, configurations, tests, and documentation; datasets, checkpoints, and generated figures are excluded from Git.

## Install

Use Python 3.11 or newer and a suitable PyTorch installation for your CPU/GPU:

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e ".[test]"
python -m pytest
```

All commands below run from the repository root. Relative data/output paths are interpreted from the current working directory. Configuration files are ordinary TOML with named sections; unknown keys are rejected to catch misspellings.

## Start with small runs

```bash
python -m adam_lr.mnist --config configs/mnist/smoke.toml
python -m adam_lr.directional --config configs/directional/smoke.toml
```

The first command uses synthetic data to exercise the MNIST pipeline without a download. For a real-MNIST execution check, use `configs/mnist/real_smoke.toml` and point `--data-dir` at a directory containing `MNIST/raw/`. To download MNIST, explicitly set `dataset.download = true` in the MNIST configuration.

```bash
python -m adam_lr.mnist --config configs/mnist/real_smoke.toml --data-dir data
python -m adam_lr.mnist --config configs/mnist/three_layer_eta3e4.toml
```

Production presets are long experiments, not smoke tests. Their per-schedule iteration budgets are visible in the TOML. See [MNIST experiments](docs/mnist.md) and [directional simulations](docs/directional.md) for commands, plotting, field definitions, outputs, and resume behavior.

Plot any saved scalar fields, anchor cosines, and coordinate traces without rerunning training. The [incoming-moment analysis](docs/incoming_analysis.md) also preserves the original adjacent-transition calculation, with undefined transitions kept as gaps.

## What is configurable?

| Choice | MNIST | Directional model |
|---|---|---|
| Depth | `model.hidden_widths`: number of weight layers is length + 1 | `model.layers` |
| Width/geometry | Any positive hidden widths, including unequal widths | Two input coordinates and depth-dependent scalar factors |
| Training points | `dataset.per_class` points in each parity class | Two explicitly specified model examples |
| Initialization | Method, scale, standard deviation, seed | Initialization scale and control-grid bounds |
| Optimizer | Beta values; fixed or explicitly rescaled beta policy | Beta values and epsilon |
| Learning rate | Base rate and power per named schedule | Base rate and schedule power |
| Run budget | Global and per-schedule steps, target loss | Iteration budget |
| Observation | Logging intervals, coordinate count, loss snapshots, checkpoints | Sampling interval and empirical direction diagnostics |
| Computation | CPU/GPU, float32/float64, thread count | CPU/GPU, float32/float64, thread count |

For both families the schedule is

\[
\eta_t=\eta_0 t^{-p},\qquad t=1,2,\ldots.
\]

The first update therefore uses `eta_0`. The learning rate multiplies the Adam update. Fixed betas remain fixed throughout a trajectory; changing the learning-rate schedule does not implicitly rescale them. MNIST's optional legacy beta policy is documented separately and recorded in run metadata.

## Numerical interpretation

Epsilon-free Adam has a defined zero update for coordinates with an exactly zero gradient history. A nonzero gradient/momentum with a zero represented denominator is a failure, not convergence. The shared optimizer rejects nonfinite inputs and invalid proposals before committing an update. The directional runner additionally checks exponential-loss and second-moment representability.

Finite-time angular drift and endpoint clouds are observations. They do not certify an asymptotic directional limit or coverage of an entire open set. A coordinate-permuted directional sector also permutes the data; combined sectors must not be interpreted as one fixed dataset.

See [audit findings](docs/audit_findings.md) for confirmed source bugs, their consequences, and limitations, and [validation](docs/validation.md) for the checks actually performed. Full production sweeps must be rerun before using changed diagnostics in a paper.

## Layout

```text
configs/                readable small-run and research presets
src/adam_lr/optim/      shared discrete Adam and schedule
src/adam_lr/mnist/      model, data, runner, diagnostics, logging, plotting
src/adam_lr/directional/ product model, runner, empirical diagnostics, plotting
tests/                  mathematical, numerical, configuration, and resume tests
docs/                   experiment definitions, provenance, audit, validation
```
