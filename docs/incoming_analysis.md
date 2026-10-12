# Adjacent-step incoming-moment analysis

Enable both `logging.log_second_moment_shadow` and `logging.log_incoming_every_step` before running MNIST. Then analyze one run or a suite:

```sh
python -m adam_lr.mnist.incoming_analysis runs/mnist --reference-loss 1e-8 --output-dir output/incoming
```

The tool generalizes the original `prepare_mnist1024_incoming_f.py` workflow. It uses the **actual every-step trace**, with no fitted loss derivative and no interpolation across missing transitions. It accepts any logged depth, seed, schedule, and beta2. Each line is a separate run; it does not silently pool different models or produce confidence bands.

For raw Adam second moment `v_t`, define the diagnostic transformed buffer `W_t=beta2^(-t) v_t`. The positive incoming mass is

```text
D_t = ||W_(t+1) - W_t||_1
    = (1-beta2) * beta2^(-(t+1)) * ||g(theta_t)||_2^2.
```

The shadow evaluates its natural logarithm without forming the growing power or squaring tiny gradients. It observes ordinary Adam; it never replaces the training update.

With `x_t=-log10(loss(theta_t))`, the plotted transition average is

```text
Fbar_t = D_t / (ln(x_(t+1)) - ln(x_t)).
```

This is incoming mass divided by the observed change in log loss-clock over one transition. It is not a claim that a continuous derivative exists. The terminal row has no successor: its prospective `D_terminal` is excluded. Flat/increasing loss, zero incoming mass, and undefined positive loss-clocks remain gaps.

Each run is divided by its first valid `Fbar_t` beginning at or after the first loss crossing of `--reference-loss`. The x-coordinate is the midpoint of the two endpoint `log10(x)` values. Later reversals are shown as separate time-ordered segments, not collapsed to one branch.

The loader checks contiguous steps, the complete terminal state, required columns, trace/result signature agreement, endpoint loss, and both artifact hashes and byte sizes. Failed numerical runs are refused. A normal iteration-budget stop is usable if a valid transition remains after the selected reference threshold; reaching the final requested training loss is not required.

Outputs are full-resolution per-run NPZ arrays, PNG/PDF overlays, and a manifest identifying each run, normalization step, input artifact hashes, and analysis source hash. Display thinning preserves gap boundaries; `--max-plot-points` is therefore a soft budget that can be exceeded for fragmented curves. Full processed values remain available. Logarithms are kept as logarithms throughout, so exponentially large incoming masses need not be represented as ordinary numbers. The loss-clock increment is evaluated directly from the stored natural log losses with `log1p`, before changing the clock's base, so adjacent representable values are not collapsed by an unnecessary rescaling.
