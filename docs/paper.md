# Figure preparation

## Render the bundled results

```sh
python paper/figures/render.py --output-dir output/paper
```

The renderer writes the manuscript's 20 PDF assets under `output/paper/figures/`.
Use `--family mnist`, `epsilon`, `error`, `endpoints`, or `construction` to select
a group. Add `--format both` for PNGs. The TikZ schematic requires XeLaTeX;
`--skip-tex` renders the numerical figures without it.

The inputs are grouped by experiment:

| Input | Contents |
|---|---|
| `paper/data/long_horizon.npz` | Loss-rate summaries, margins, individual-seed curves, endpoint metrics |
| `paper/data/comparison.npz` | Both epsilon values, norm margins, update error, endpoint metrics |
| `paper/data/sphere.csv` | Unit parameter directions and dataset labels |

## Prepare a new experiment

Set the loss checkpoints in the training configuration. After the sweep
completes, prepare its statistics and endpoint-reference margins:

```sh
python -m paper.figures.prepare --runs-dir runs/long_horizon --data data --output output/figure-data/long_horizon.npz
python -m paper.figures.prepare --runs-dir runs/comparison --data data --output output/figure-data/comparison.npz
```

The preprocessor reads the run configurations to identify seeds, schedules,
model settings, and epsilon values. It requires a complete sweep and checks
that schedules share their seed's initialization. The MNIST data are needed
to evaluate saved parameter checkpoints. Use float64 for the paper's
endpoint-metric preparation; both supplied presets already do so.

For a width-1 experiment, extract its final directions:

```sh
python paper/figures/prepare_sphere.py runs/sphere --output output/figure-data/sphere.csv
```

Then render the new figure data:

```sh
python paper/figures/render.py --data-dir output/figure-data --output-dir output/new-figures
```

## Quantities and statistics

Loss-aligned curves use first chronological passage through each loss level.
Means use the common observed support of the seeds; shading is one sample
standard error. The loss-rate panel uses secants over the final 10,000 updates.
The relative-error panel averages the difference between Adam actions with and
without epsilon over trailing windows of 1,000 updates.

An endpoint metric includes the terminal gradient:
`v_next = beta2*v_T + (1-beta2)*g_T**2` and
`H = sqrt(beta2**(-(T+1)) * v_next)`. This prospective update is used for
analysis and is not applied to the parameters. Normalize the vector
`H = omega**(1/2)` by its l2 norm: `H_hat = H / ||H||_2`. Weighted margins
divide the minimum signed model output by
`(sum_j H_hat_j*theta_j**2)**(L/2)`. The common exponential factor in `H`
cancels during normalization. Small margins use labeled powers of ten.

The paper layouts use five schedules and five seeds. Use the general
[MNIST plotter](mnist.md#plot-and-analyze) for other sweep sizes.
