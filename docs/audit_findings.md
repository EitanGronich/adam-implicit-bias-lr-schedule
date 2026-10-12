# Mathematical and numerical audit — 12 October 2026

This audit checks the discrete Adam recurrence, objective gradients, diagnostic
definitions, numerical failure handling, and extraction of the two requested
experiments. It does not certify an asymptotic theorem or repeat the complete
historical 2,662-trajectory, 400,000-step experiment. A 54-trajectory subset was
rerun for 400,000 steps, and the full 2,662-trajectory grid for 10,000 steps;
the precise validation scope and test totals are recorded in
[validation.md](validation.md).

Original source paths below are relative to the parent research workspace.
Unqualified original MNIST filenames are in `dataset_reconstruction_adam/`.
Corrections are confined to the extracted repository; the original experiment
source files and old result artifacts are retained.

## Confirmed consequence for historical directional experiments

The aggressive `rho=3`, constant `eta=1` rerun reached floating-point failure,
not an observed mathematical limiting direction. In
`adam_fast_regime_l3/repository_adam_rerun_20261006/summary.json`, all three beta
settings report `gradient_square_zero.affected_fraction = 1.0` and
`parameter_stagnation.affected_fraction = 1.0`. The first gradient-square
underflows occurred at update 2. For `(beta1,beta2)=(0.9,0.999)`, every trajectory
also encountered an exactly zero represented gradient, and the run stopped at
update 1000. Its endpoint cloud is a numerical endpoint cloud; it cannot be
used as evidence of the exact Adam directional limit.

The inherited zero-denominator convention aggravates this failure mode.
`dataset_reconstruction_adam/adam_dynamics.py:221` masks every zero denominator
to a zero update, including coordinates with nonzero gradients or momentum.
A direct float64 probe with `theta=1`, `g=1e-200`, `beta1=0.9`, `beta2=0.999`,
`epsilon=0`, and `eta=0.1` returned `theta=1`, `m≈1e-201`, and `v=0`. Exact
first-step Adam would return `theta=0.9`. This is different from a legitimate
coordinate whose entire gradient and moment history is zero.

The new shared optimizer rejects that failed proposal without advancing any
parameter, moment, or step counter. It also rejects nonfinite inputs and
proposals. The directional runner checks the loss-gradient calculation and
moment representability separately, so a positive previous moment cannot hide
a newly underflowed gradient square. Numerical stops are reported explicitly.

**This finding does not invalidate the later tuned run.** The saved
`repository_adam_tuned_20261006/final_validation.json` reports the selected
`rho=1`, `eta=0.01`, `(0.9,0.999)` run reached 400,000 updates without guard
rejection. Its minimum recorded squared gradient was approximately
`1.9266e-186`, and its minimum recorded positive second moment was
`1.2797e-180`; both are normal float64 numbers. This audit inspected that
evidence and reran the subset described in the validation record; it did not
repeat the complete historical grid for that horizon. Its finite-time
directional stability still does not establish an asymptotic limit.

## Confirmed defects with no demonstrated corruption of the historical MNIST runs

1. **Direction diagnostics could underflow while the inputs remained normal.**
   The original `vector_norms`, `safe_cosine`, and `scale_aligned_error`
   (`mnist1024_three_layer_adam_experiment.py:365`, `:436`, `:1013`) square
   unscaled vectors. In float64, `norm([1e-200,2e-200])` returned zero even
   though the true, representable norm is `2.2360679775e-200`. Raw-moment
   norms, cosines, or scale-aligned errors can therefore become zero or NaN
   before the moments themselves underflow. The new diagnostics normalize by
   a maximum absolute entry before computing norms or directions. This affects
   reported diagnostics, not Adam trajectories. No historical MNIST artifact
   has been established as materially affected by this audit.

2. **Absolute epsilons broke scale invariance in optional metric diagnostics.**
   Original frozen-metric normalization clamps its RMS to `1e-30`
   (`mnist1024_three_layer_adam_experiment.py:1041`). Original nuclear-norm
   estimation normalizes a raw gradient by `norm(gradient)+epsilon`
   (`muon_dynamics.py:58`, `:62`). For a one-row matrix the resulting estimate
   is `norm(G)^2/(norm(G)+epsilon)`, although its nuclear norm is exactly
   `norm(G)`. Both are now normalized relative to their own scale; row and
   column nuclear norms are exact. General matrix nuclear norms remain
   explicitly approximate polar-iteration diagnostics. The deterministic
   spectral-norm estimator is also now applied to a scale-normalized matrix;
   its former `1e-30` normalization floor could suppress tiny matrices.

   The MNIST runner passes `epsilon=1e-30` at its nuclear-norm call
   (`mnist1024_three_layer_adam_experiment.py:879–883`), overriding the helper's
   `1e-12` default. Consequently, the existence of this defect is **not**
   evidence of substantial MSP error at a `1e-20` loss target. Its relevance
   depends on the actual gradient matrix scale. These changes do not modify
   optimization.

3. **The reusable logistic helper allowed unintended broadcasting.**
   `adam_dynamics.py:18` multiplies labels directly by model output. An `(N,1)`
   output and `(N,)` labels form an `(N,N)` loss. The new helper checks one
   scalar logit and label per example and flattens both. The actual historical
   three-layer MNIST model already returns `(N,)`; this defect does not show
   that those historical trajectories used the wrong objective.

4. **The diagnostic moment preview had a different rounding path.**
   The original preview (`mnist1024_three_layer_adam_experiment.py:969–1003`)
   uses multiply/add expressions unless log-shadow mode is enabled, whereas
   the optimizer uses in-place `mul_`, `add_`, and `addcmul_`. They represent
   the same real-arithmetic recurrence, but need not agree bit-for-bit near
   representational limits. The new preview uses the optimizer's operation
   sequence consistently. This corrects prospective diagnostics, not updates.

5. **Same-named second-moment ratios had different denominators.**
   Original layer summaries `g2_over_v` and `v_over_g2` use the bias-corrected
   `v_hat` (`mnist1024_three_layer_adam_experiment.py:835–848`), whereas the
   coordinate panel uses raw `v`. At the first update with `beta2=0.999`, a
   nonzero coordinate gives `g^2/v_hat=1` but `g^2/v=1000`. The discrepancy
   shrinks as the bias correction tends to one; it is substantial at early
   steps. New layer and coordinate diagnostics consistently use raw `v`.
   Historical comparisons that treated these same-named fields as identical
   need the bias-correction factor restored. Optimization was unaffected.

6. **Continuation validation and dataset conversion needed hardening.**
   The original optimizer restore uses `zip` without checking moment-list
   lengths (`mnist1024_three_layer_adam_experiment.py:626–632`); a truncated
   list silently leaves unmatched buffers at zero. The extraction rejects
   malformed counts, shapes, dtypes, moments, and step indices before changing
   ordinary state. No corrupted historical checkpoint was discovered.

   Also, original `ToTensor` pixel conversion follows PyTorch's global default
   dtype, which the runner changes before loading MNIST. Requesting float64
   continuation from float32 can therefore change the dataset fingerprint and
   fail the compatibility check even with the dtype-promotion flag. The new
   loader always divides the original uint8 pixels by 255 in float64 before
   centering and final casting. It matches historical float64 preprocessing;
   it intentionally avoids the historical float32 pixel-rounding path. Source
   hashes now guard continuation even when an edit has not changed Git HEAD.

7. **A newly active coordinate could receive a spurious finite log-moment rate.**
   Original `coordinate_metrics` reads `second_previous` but computes the rate
   only from `g^2/v_next`
   (`mnist1024_three_layer_adam_experiment.py:1275–1314`). When the previous
   moment is zero, the transformed-log increment is undefined. Floating-point
   rounding can put `(1-beta2)*g^2/v_next` just below one and produce a large
   finite number instead; the extraction smoke exposed values around 3600 for
   `rho/(-log(beta2))`. The corrected diagnostic requires positive finite
   previous and next moments and evaluates the recurrence in log space. A
   zero current gradient with positive moment history gives exactly zero rate.
   This affects initial or newly activated coordinate diagnostics, not the
   optimizer path. No claim is made that a particular historical plotted
   curve contains this artifact.

8. **A singular frozen metric could produce a false perfect alignment.**
   The original frozen alignment uses only coordinates with positive frozen
   weight in its dual norm, but uses every coordinate in the numerator and
   then clips the ratio to `[-1,1]`
   (`mnist1024_three_layer_adam_experiment.py:1561–1570`). For frozen weights
   `(1,0)`, current gradient `(1,1)`, and descent action `(-1,-1)`, it reports
   alignment `1`. The gradient actually has no finite dual norm for that
   singular metric. Likewise, the frozen inverse-action comparisons silently
   zero newly active coordinates (`:1525–1526`). This can matter when a ReLU
   coordinate becomes active after an anchor was frozen.

   The new inverse-action diagnostics require the current numerator to lie
   on the positive frozen support. Frozen alignment is explicitly restricted
   to pairs supported there; unsupported or zero directions produce NaN.
   Its weighted cosine is evaluated with logarithmic scaling to avoid a
   second, numerical loss of tiny gradient norms. Ordinary cosine comparisons
   with saved anchor vectors remain distinct and do not require an inverse
   metric. The optimizer trajectory is unchanged. No affected historical
   MNIST artifact has been established.

   A targeted historical check examined all 30 saved loss snapshots across
   `runs/mnist1024_three_layer_relu_adam_eta3e4_local_extremes/`, seed
   `20260828`: `(beta2,p)=(0.9,0)`, `(0.99,0)`, and `(0.999,0.25)`, each with
   `eta0=0.0003`. Each saved anchor has 175,165 zero coordinates. At every
   checked snapshot at or after its anchor, both the gradient and first-moment
   numerator were exactly zero on those coordinates. These particular saved
   states do not exhibit the support defect; unsaved intermediate states and
   other runs were not covered by this check.

## Interpretation and scope

- MNIST uses the mean binary logistic loss, labels `-1/+1`, full-batch
  gradients, a bias-free ReLU network, and Adam bias correction. Joint positive
  scaling of all parameters gives homogeneity degree equal to the number of
  affine layers. The first learning rate is `eta_1=eta0`; the power schedule is
  `eta_k=eta0*k^(-p)`.
- MNIST diagnostic rows associate `theta_t` and its gradient with prospective
  moments that include that gradient and the proposed update at `eta_(t+1)`.
  Checkpoints retain the optimizer moments after the `t` updates actually
  applied. These are deliberately different states, and the output metadata
  names their timing.
- The directional objective is the sum of two exponential losses for the
  product model. Its analytic gradient is independently checked against
  automatic differentiation. Trajectories are actual discrete Adam iterates;
  no ODE solver, transformed-gradient replacement, or proved direction formula
  determines an update or exported endpoint direction.
- In a directional run with `sectors="both"`, swapping the first two parameter
  coordinates also swaps the input data coordinates. The combined atlas is
  two coordinate-equivalent datasets. It is not evidence that one fixed
  unswapped dataset realizes the combined set of directions. Use
  `sectors="original"` to study that single fixed dataset.
- A finite-horizon unit vector and small observed drift are empirical
  diagnostics. They are not proofs of convergence. Convex-hull volume describes
  the span of sampled points, not uniform filling or an open attainable set.
- The default beta policy is fixed. Optional MNIST `rescaled` beta policy
  retains the historical experiment's explicit rule
  `beta_i=reference_beta_i**(reference_rate/initial_rate)`; the resulting beta
  remains constant during a run. It is a different experiment and should not
  be mistaken for changing only the learning-rate schedule.

## Numerical regression evidence

The new `tests/test_mnist_numerics.py` checks representable norms at scales
`1e-200`, `1`, and `1e200`, cosine/projection invariance for mixed scales,
near-parallel residual accuracy, zero/nonfinite direction handling, exact
scalar-output nuclear norms, matrix scale invariance, and frozen-metric margin
scale invariance, and spectral estimates against a diagonal-matrix oracle.
The numerical-regression file has 22 cases, including a zero-row-sum matrix
that an all-ones power-iteration start would miss, a representable `1e-200`
projection residual, and a finite log increment whose ordinary ratio overflows.
Run together with the coordinate-diagnostic suite, all 34 tests and four
subtests passed. The shared
optimizer tests separately check a scalar recurrence, agreement with PyTorch
Adam, the zero-history convention, column-output logistic loss, and rejection
of a bad proposal before any state is committed. See the final validation
record for the completed repository-wide checks.
