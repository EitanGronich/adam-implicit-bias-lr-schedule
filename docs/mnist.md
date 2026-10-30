# MNIST experiments

Run all seeds, schedules, and epsilon values in a TOML configuration:

```sh
python -m adam_lr.mnist --config configs/mnist/comparison.toml
```

To run one cell of the sweep:

```sh
python -m adam_lr.mnist --config configs/mnist/comparison.toml --seed 2026100901 --schedule alpha_neg0p30_eta3e4 --epsilon 1e-8
```

`--data-dir`, `--output-dir`, `--device`, and `--max-steps` override the
corresponding settings. Relative paths are resolved from the working directory.

## Configuration

| Section | Settings |
|---|---|
| `experiment` | Output directory, device, precision, CPU threads |
| `dataset` | Data directory, sample count, centering, download |
| `model` | Hidden widths, biases, initialization method and scale |
| `optimizer` | Initial learning rate, beta1, beta2, epsilon |
| `sweep` | Initialization seeds, beta2 values, epsilon values |
| `schedules` | Named schedule exponents, rates, and step budgets |
| `run` | Target loss, step budget, checkpoint resume |
| `logging` | Trace buffering, progress and checkpoint frequency, loss checkpoints |

Use `model.hidden_widths = [512, 512]` for two hidden ReLU layers;
`model.bias = true` adds a bias to every layer. Weights and biases use the
selected initialization and common `model.layer_scale`.

For update `k >= 1`, the learning rate is `eta_0 * k**alpha`. All schedules
for a seed start from the same parameters. Changing epsilon also preserves
that pairing. Adam uses bias correction and no weight decay. For epsilon zero,
coordinates with an identically zero gradient history receive zero updates.

The supplied dataset is binary MNIST: the first 512 even and first 512 odd
training examples, labeled -1 and +1. Inputs are divided by 255, flattened, and
centered by the selected training set's per-pixel mean. Training minimizes the
full-batch mean logistic loss. Adjust `dataset.per_class` to change sample size.

## Outputs and resume

Each run has its own directory containing settings, results, a training trace, and a
checkpoint. `run.resume = true` resumes that checkpoint on the next invocation;
incompatible model or optimizer settings are rejected. Use a new output directory
when starting a different experiment.

The trace records every evaluated state's loss, raw margin, parameter norms,
normalized margins, and epsilon update error. `logging.trace_buffer_rows`
controls buffered writes. `logging.full_checkpoint_losses` selects losses at
which to save the parameter states used by endpoint-metric analysis. The terminal
observation is recorded before any further update is applied.

## Plot and analyze

```sh
python -m adam_lr.mnist.plot runs/check --output-dir output/check
python -m adam_lr.mnist.plot runs/check --output-dir output/check --fields loss,margin_linf --x loss-clock
```

The loss clock is `-log10(loss)`. The plotter displays loss and normalized
margins directly from the recorded states. For aggregate margins and
endpoint-reference plots, follow
[figure preparation](paper.md).

Networks with biases are not exactly homogeneous. Their parameter-normalized
margins are empirical diagnostics; the bias-free network has homogeneity degree
equal to its number of linear layers.
