# Discrete Adam directional experiments

Run from the repository root after installing the package:

```sh
python -m adam_lr.directional --config configs/directional/smoke.toml
python -m adam_lr.directional --config configs/directional/final_rho1_eta001.toml
python -m adam_lr.directional --config configs/directional/final_rho1_eta001.toml --steps 1000 --output runs/directional/check
python -m adam_lr.directional --plot-only runs/directional/check
```

Each output directory must be new or empty. All trajectories take actual discrete
Adam steps using unscaled, analytically evaluated gradients. This package contains
no ODE solver, frozen-metric direction formula, predicted asymptotic direction,
or theorem-based replacement for a measured parameter vector. The model is small
and fully specified; the experiment does not use MNIST.

## Model, data, and objective

Depth `L = model.layers >= 2` counts the vector factor `u` and `L-1` scalar factors:

```text
theta = (u1, u2, b1, ..., b_(L-1))
f(x; theta) = (product_j b_j) * (u1*x1 + u2*x2)
x_S = (B, 1) / C, x_T = (M, 0) / C
both labels = +1
loss(theta) = exp(-f(x_S; theta)) + exp(-f(x_T; theta))
```

This is a positive, homogeneous product model with two training examples. There
is no hidden-layer width: its parameter count is `L+1`. The sum, rather than the
mean, is intentional and matches the historical simulation. Changing that
normalization changes an Adam trajectory when `epsilon > 0`.

A batch row is a separate optimization problem, not an additional training
example. No gradient is averaged across trajectories. The analytic gradient
uses products excluding each scalar factor, rather than dividing by that factor;
it remains valid at zero factors and is tested against autograd and finite
differences. The runner restricts accepted trajectories to positive parameters,
where every exact gradient coordinate is negative.

## Initialization and controls

There are `L` controls `(s, ell_1, ..., ell_(L-1))`. Define

```text
beta0 = C^(1/L) * exp(-(2/L) * sum_j ell_j)
beta_j = beta0 * exp(2*ell_j)
a = (1 + s/rho^L) / M
u = rho * beta0 * (a, 1-B*a)
b_j = rho * beta_j
```

These are an initialization parameterization, not a limiting-direction formula.
Their product satisfies `product_(j=0)^(L-1) beta_j = C`. Direct substitution
therefore gives initial margins `z_S = rho^L` and `z_T = rho^L + s`, which are
explicitly tested for several depths. The config requires
`0 <= s_min <= s_max` and `(1+s_max/rho^L)/M < 1/(B+1)`, ensuring `0<u1<u2`
in the original sector. Controls that violate this condition are rejected.

`sampling="grid"` uses `grid_size^L` control combinations in each sector;
`grid_size=1` uses the midpoint of every control interval. `sampling="random"`
uses `sample_count` uniform control vectors with a saved seed. The random seed
selects controls only; there is no minibatching. Increasing depth makes a tensor
grid exponentially large, so `max_trajectories` prevents accidental allocations.
`rho`, `s_min`, `s_max`, and the shared `ell_min`/`ell_max` bounds are explicit.
Changing `M` does not silently rewrite `s_max`.

**Sector semantics matter.** `sectors="swapped"` swaps `u1/u2` in the initialization
AND the coordinates of both examples. `sectors="both"` runs paired original and
swapped problems, as the historical code did. Their union is two
coordinate-equivalent datasets, not an experiment varying initialization on one
fixed dataset. Use `sectors="original"` for the latter. The pairing is useful for
checking permutation symmetry, and the final maximum permutation error is saved.
The words "original" and "swapped" describe initialization/data construction;
they do not assert that an ordering of `u` remains invariant during optimization.

## Adam and learning rate

At accepted update `k>=1`, the gradient is evaluated at `theta_(k-1)`:

```text
g_k = gradient(loss)(theta_(k-1))
m_k = beta1*m_(k-1) + (1-beta1)*g_k
v_k = beta2*v_(k-1) + (1-beta2)*g_k^2
mhat_k = m_k/(1-beta1^k), vhat_k = v_k/(1-beta2^k)
eta_k = initial_rate * k^(-power)
theta_k = theta_(k-1) - eta_k*mhat_k/(sqrt(vhat_k)+epsilon)
```

Both moments start at zero. The epsilon is outside the square root. A positive
`power` decays the rate, zero keeps it constant, and a negative power grows it.
There is no weight decay, gradient clipping, gradient renormalization, or state
rescaling. `float64` is the default; `float32` is available but can encounter
numerical limits much earlier. Set `device="cuda"` only on a working CUDA device.

The selected historical preset is `L=3`, `B=2`, `C=1`, `M=128`, `rho=1`, an `11^3`
grid in each sector, `beta1=0.9`, `beta2=0.999`, `epsilon=0`, constant `eta=0.01`,
and `400000` requested updates. It corresponds to the source
`repository_adam_tuned_20261006/final_rho1_eta001` metadata. The `l4_power025.toml`
file illustrates changing depth, sampling, and schedule together; it is an
example configuration, not a historical result.

## Numerical stopping is not convergence

The experiment monitors gradient squares, the new second-moment contribution,
raw moments, parameters, and margins. Zero/subnormal squared gradients can erase
information long before the gradient itself becomes zero. The default guard
rejects such a proposal and restores the preceding parameters, both moments,
and optimizer step count. `stop_on_subnormal=false` allows positive subnormals
but still rejects exact underflow. This switch trades numerical conservatism for
a longer representable horizon; it does not repair lost precision.

Because gradients are strictly negative in the configured sector, a represented
zero gradient is also a numerical failure. A nonzero gradient or momentum with a
zero Adam denominator raises a failure instead of silently producing a zero
update. The runner does not claim convergence from a stopped or unchanged state.

The optional `support_margin_cap` ends the accepted trajectory before its support
margin exceeds that value; zero disables the cap. One failing trajectory stops
the entire batch at its last common accepted update. There are no cohorts with
misleading later global checkpoint numbers. `stagnation_patience=0` logs rounded
zero-update streaks but does not stop: a finite-precision plateau with a nonzero
gradient can restart as the old second moment decays. A positive patience stops
when any trajectory reaches that streak and labels the event as observed
stagnation, never as a directional limit.

An individual `exp(-z_T)` term can underflow after becoming negligible relative
to the support term. Its frequency is logged as `underflowed_loss_term_fraction`;
it is not silently used as evidence of convergence. The guard acts on the total
gradient and its moment contributions. Nonfinite states are rejected. After writing results, the CLI exits with status 2
on an initialization/numerical guard or configured stagnation stop. A normal
step limit or requested support-margin horizon exits with status 0.

## Saved observations and plots

- `config.toml` copies the submitted configuration; `config.json` records the
  resolved configuration including CLI overrides.
- `checkpoints.csv` is flushed at every saved checkpoint and contains aggregate
  losses, margins, gradient/moment/update magnitudes, numerical-health fractions,
  parameter norms, and observed angular drifts.
- `history.npz` contains the complete saved parameter and unit-direction arrays,
  predictor directions, current-state gradients, moments, margins, log losses,
  actual last parameter updates, initialization controls, and sector flags.
- `final_state.pt` contains the exact accepted final parameters, both raw Adam
  moments, step count, and config. This is an inspection artifact; the CLI does
  not currently implement resuming.
- `endpoints.csv` reports actual parameters, unit directions, and squared unit
  coordinates. These are finite-run endpoints.
- `summary.json` identifies the accepted horizon, stop reason, any guard event,
  permutation error, initial/half-horizon-to-endpoint angular drift, and endpoint
  squared-coordinate ranges/covariance eigenvalues (empirical summaries only).
- `plot_provenance.json` records the plotting-source hash, input history/summary
  hashes, and Matplotlib version, including when an existing run is replotted.
- `diagnostics` and `directions` PNG/PDF pairs show the observed paths. For `L=3`,
  `endpoint_tetrahedron` maps each observed positive unit direction to
  `q_i = theta_i^2 / ||theta||^2`. This is an affine display of simplex coordinates,
  not evidence of full reachability or asymptotic convergence.

At saved state `k`, the `gradient` array is freshly evaluated at `theta_k`,
whereas `first_moment`, `second_moment`, and `last_update` belong to update `k`
(which used the preceding state's gradient). At state zero all moments and the
last update are exactly zero. `rate_at_last_update` is blank at state zero and thereafter reports the rate
used for update `k`; `rate_at_next_update` reports the proposed rate for `k+1`
when another update is requested and its rate is representable (otherwise blank).

Checkpoint angular drift is measured since the previous saved state. Its
interval changes with logging cadence, so shrinking checkpoint drift by itself
is not a convergence criterion. Half-horizon drift names its actual reference
step; an early guarded stop may use the closest saved state at or before half
the accepted horizon. The angular calculation uses a stable `atan2` formula,
not `acos` of a rounded dot product. Zero/nonfinite vectors have no direction.

## Consequential issue found in historical results

The earlier `rho=3`, `eta=1` discrete rerun had gradient-square underflow in every
trajectory, first appearing at update 2. Its saved summary reports that all
trajectories eventually stagnated; for `beta=(0.9,0.999)`, all gradients became
zero, with the first event at update 3. Those endpoints are numerically frozen
states and cannot support a claim about an actual Adam directional limit.
The source's later `rho=1`, `eta=0.01` selected run used guards and stayed in a
resolved range in its saved metadata. The new runner preserves that preset,
checks every proposal, and omits all historical formula-based reference arrows.
