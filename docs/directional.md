# Width-1 experiments

Run a configured experiment:

```sh
python -m adam_lr.directional --config configs/directional/sphere.toml
```

For a short run in a new output directory:

```sh
python -m adam_lr.directional --config configs/directional/sphere.toml --steps 1000 --output runs/sphere-check
python -m adam_lr.directional --plot-only runs/sphere-check
```

## Model and objective

The depth-L model has vector factor `u` and `L-1` scalar factors:

```text
theta = (u1, u2, b1, ..., b_(L-1))
f(x; theta) = (product_j b_j) * (u1*x1 + u2*x2)
x_S = (B, 1) / Lambda
x_T = (M, 0) / Lambda
loss(theta) = exp(-f(x_S; theta)) + exp(-f(x_T; theta))
```

Both examples have label +1. Each initialization defines an independent
optimization trajectory. Training uses full-batch, bias-corrected Adam.

## Configure a new experiment

The TOML file controls model depth and dataset geometry, initialization scale
and controls, optimizer parameters, learning-rate schedule, run length, and
plotting. `initialization.sectors = "both"` also runs the coordinate-permuted
dataset; `"original"` selects one dataset.

The controls are `(z, pi_1, ..., pi_(L-1))`, with positive `pi_j`:

```text
pi_0 = Lambda / product_j pi_j
a = (1 + z/rho**L) / M
u = rho * pi_0 * (a, 1-B*a)
b_j = rho * pi_j
```

Use `sampling = "grid"` with `grid_size` and control bounds, `"random"` with
`sample_count` and `seed`, or `"explicit"` with a `control_points` array.
The sphere preset supplies 142 control pairs per dataset at depth 2, with
`M=35` and `rho=1.01`. `require_admissible_rho = true` enforces the strict
initialization bound from the paper.

The schedule is `eta_k = initial_rate * k**alpha` for update `k >= 1`.
Computation defaults to float64. Configure numerical guards in `[numerics]`
and history/progress frequency in `[logging]`. Output directories must be new
or empty.

## Outputs

Each run saves its configuration, trajectory arrays, endpoint table, stopping
reason, and plots. Directions are observations at the recorded training horizon.
Regenerate plots with `--plot-only`. Depth-2 parameter directions can also be
exported for the paper renderer; see [figure preparation](paper.md).
