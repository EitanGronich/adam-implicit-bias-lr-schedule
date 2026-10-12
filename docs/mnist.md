# MNIST learning-rate schedules

This experiment minimizes full-batch mean binary logistic loss on an equal number of even (-1) and odd (+1) MNIST training images. It follows the **actual discrete bias-corrected Adam updates**, with epsilon exactly zero, fixed betas within each run, and no weight decay. It records parameter directions, margins, moment geometry, and numerical limitations while changing the learning-rate schedule.

## Run and configure

From the repository root after installation:

```sh

python -m adam_lr.mnist --config configs/mnist/smoke.toml

python -m adam_lr.mnist --config configs/mnist/real_smoke.toml --data-dir /path/to/data

python -m adam_lr.mnist --config configs/mnist/three_layer_eta3e4.toml --seed 20260828 --beta2 0.999 --schedule p025_eta3e4

python -m adam_lr.mnist.plot runs/mnist_smoke --output-dir output/mnist_smoke

python -m adam_lr.mnist.plot runs/mnist_smoke --fields loss,linf_margin,a1em4_update_frozen_cosine --x loss-clock --anchor-cosines --coordinates

```

`smoke.toml` uses 16 synthetic examples and a small model; it tests execution, not a scientific conclusion. `real_smoke.toml` uses 16 real MNIST images. MNIST downloads are disabled by default; set `dataset.download=true` to download it explicitly. Data and output paths resolve against the working directory. `--data-dir`, `--output-dir`, `--device`, and `--max-steps` override the corresponding configuration choices; the horizon override applies to every selected schedule.

The production presets retain the original three-layer **784–1024–1024–1** architecture, 1,024 points, initialization seed convention, schedule identifiers, rate values, and per-schedule horizons at eta0=0.0003 and 0.0007. Each preset explicitly defines a serial sweep over three seeds and beta2 values 0.9, 0.99, and 0.999. They are long production experiments. Select a cell or use `--max-steps` for a short validation run. Outputs from sweeps are separated as `seed_<seed>/beta2_<beta2>/<schedule>/`.

The TOML sections are:

- `[experiment]`: seed, output directory, float32/float64, CPU/CUDA/auto, CPU thread count.

- `[dataset]`: kind (`mnist` or `synthetic`), `per_class` (total points = twice this), data directory, explicit download flag, centering, and synthetic input dimension/seed.

- `[model]`: `hidden_widths=[1024,1024]` gives three affine layers; `[64,32,16]` gives four; `[]` gives a linear model. Every layer is bias-free. Initialization is PyTorch's default per-layer uniform initialization or `normal` with configurable standard deviation, followed by `layer_scale` on every parameter.

- `[optimizer]`: initial learning rate, beta1, beta2, and `beta_policy`. `fixed` uses those betas directly. The optional historical `rescaled` policy uses beta_i(reference)^(reference_rate / initial_rate), **once per run**. Effective beta2 values rounded to zero or one are rejected.

- `[run]`: horizon, positive target loss, whether to stop at target, resume, and explicit target/dtype continuation flags.

- `[logging]`: scalar/coordinate/expensive diagnostic intervals, checkpoint intervals, fixed-coordinate panel size/selection threshold, optional log-second-moment shadow and per-step incoming trace, and persistent loss-checkpoint thresholds.

- `[[schedules]]`: a unique name, `power`, and optional rate and horizon overrides. For update k>=1, eta_k = eta0 * k^(-power). Zero gives a constant rate; positive powers decay; negative powers grow. Any finite power is accepted. Conditional theory annotations are marked unavailable outside [-1,1]; the discrete schedule still runs until its represented rate becomes unusable.

- Optional `[sweep]`: `seeds` and `beta2_values`. If absent, the single values in the other sections are used. `--seed` and `--beta2` select single values.

Unknown configuration keys are errors. Each schedule uses the same initial parameters for a fixed seed, independent of its rate or beta2. Initialization is constructed in float64 on CPU and then cast to the configured device/dtype. The saved metadata includes a hash of the complete initialization and a numerical homogeneity check. A depth-L ReLU network obeys f(c theta)=c^L f(theta) for c>0.

## Plotting and analysis

`--fields` accepts any comma-separated scalar column names in `trajectory_series.npz`, including all update/momentum/raw-v anchor diagnostics. `--x loss-clock` uses -ln(loss(theta_t)); no interpolation or implicit monotonicity assumption is made. `--anchor-cosines` adds the frozen-metric momentum-action and raw-second-moment cosine panels for all three anchors. `--coordinates` plots the saved fixed panels per layer; `--coordinate-fields` selects fields such as `raw_second,g2_over_v,v_over_g2,rho_over_c2,gradient,parameter`. `--max-coordinates` limits display traces with persistent/random categories interleaved, while preserving all saved values. Missing anchors are labeled rather than fabricated.

The optional incoming-log overview reads the JSONL file as a stream and displays at most `--max-incoming-points` decimated points, preserving endpoints. This is a display-only reduction; it does not change the raw file or replace full-resolution transformed-increment analysis. The incoming overview always uses update step on the x-axis because it is an every-step trace with different timing from the sampled scalar table.

## Dataset identity

Examples are the first requested number from each parity class in official MNIST order. Pixels are uint8/255 in float64, independently of PyTorch's default dtype, then optionally centered by the selected training set's per-pixel mean before casting. The uncentered image/label fingerprint and centering choice are recorded. This reproduces the historical float64 dataset conversion. Historical float32 loading rounded pixels before converting to float64; that behavior is deliberately corrected so dtype promotion retains the data fingerprint.

## State and diagnostic timing

The extracted runner uses diagnostic schema version 5. Its raw-moment ratio semantics and explicit timing fields should not be mixed with old version-4 tables without conversion.

`step=t` means **t completed Adam updates**. The model, loss, signed margins and gradients in that row are theta_t, loss(theta_t), and g_t. `moment_step=t+1` makes explicit that moment and action diagnostics are **prospective**: they include g_t and use eta_(t+1), even at the terminal row. The terminal prospective action is not applied. If the unused next learning rate is unrepresentable at a completed horizon or target, its logged value is undefined; this does not invalidate the completed updates.

Rolling `checkpoint.pt` stores theta_t and the optimizer buffers after t updates, before the current gradient has entered those buffers. It is the exact continuation artifact. A persistent loss checkpoint stores the evaluated theta_t/g_t, prospective moments/action, and the separate optimizer state before g_t; its timing fields describe the distinction. The moment preview uses the same mul/add/addcmul operation order as the actual optimizer.

The principal artifacts in each run directory are:

- `results.json`: resolved run signature, completion reason, initialization identity, code provenance, hardware, final evaluated loss, and artifact hashes.

- `trajectory_series.npz`: scalar records, sketches, norm/margin diagnostics, layer balance, update-gradient cosines, and moment/update cosines against loss anchors.

- `coordinate_series.npz`: a fixed coordinate panel per layer, chosen at the selection-loss crossing from persistently active coordinates plus a seeded sample. Completely inactive layers get explicitly labeled inactive panels, with undefined ratios retained as NaN.

- `checkpoint.pt`: atomic rolling model/optimizer/logging/RNG checkpoint.

- `frozen_metrics.pt`: raw second moments and other metric/action anchors at loss thresholds 1e-4, 1e-8, 1e-12. Missing crossings remain absent, not inferred.

- Optional `loss_checkpoints/` and manifest: persistent analysis snapshots at the first evaluated crossing of each requested threshold.

- Optional `log_second_moment_shadow.jsonl`: log-domain second-moment diagnostics, evaluated alongside ordinary Adam without changing its updates.

- Optional `incoming_second_moment.jsonl`: every-step natural logs of the prospective transformed second-moment increment, total and by layer. The terminal row is prospective. Its metadata records exact indexing and recovery behavior.

`linf_margin` is min_i y_i f(theta;x_i) / ||theta||_infinity^L. The Euclidean margin uses ||theta||_2^L. The layer balance is the largest layer max-norm divided by the smallest. Both layer and coordinate `g2_over_v` and `v_over_g2` use raw prospective v, consistently. The historical layer summaries instead used bias-corrected v_hat under the same name; this semantic error is corrected. The optional frozen-norm-efficiency plot reconstructs the own-anchor norm as (raw margin / frozen normalized margin)^(1/L), using the configured degree L, and plots its growth after the loss-1e-4 anchor. Matrix spectral and nuclear norms, and consequently MSP diagnostics, are **iterative estimates**, not exact singular-value calculations. These diagnostics use scale-normalized matrices to avoid arbitrary absolute regularization at tiny gradient magnitudes. Because these norms are approximate, derived alignment ratios may slightly exceed exact theoretical bounds; they are diagnostics, not certificates.

The shadow computes log(v_(t+1)) by logaddexp(log(beta2)+log(v_t), log(1-beta2)+2log|g_t|). It compares against the ordinary second moment and ordinary first-moment numerator. It diagnoses arithmetic in the second moment; it does not certify backpropagation, the first moment, or exact real-arithmetic Adam. `*_direct_*` quantities use represented buffer differences and may hit cancellation floors; `*_recurrence_*` use the positive-increment identity. Coordinate rho uses log1p((1-beta2)*g_t^2/(beta2*v_t))/(-log(beta2)), evaluated from the previous raw buffer in log space. Zero previous histories remain undefined; a zero current gradient with positive history has rho=0. This avoids a spurious large first-step rho from reconstructing a singular denominator through a rounded updated-buffer ratio. It reports the exact-arithmetic recurrence increment from represented previous v and g, rather than the literal log difference of rounded stored buffers. NaN means an undefined diagnostic; JSON represents nonfinite diagnostic entries as null, while NPZ preserves NaN/inf.

Every scalar row includes counts of nonzero gradients whose square underflows and of subnormal prospective second moments, even when the shadow is disabled. Summary counts cover sampled rows only. The optimizer stops atomically if a nonzero gradient/momentum has an unusable zero denominator or an update is nonfinite. Nonfinite loss/gradient and a positive logistic loss rounded to zero also stop the run. The last accepted state is saved, `completion_reason="numerical_failure"` and a reason are written, and the CLI exits with status 2 after writing the experiment summary. Such a run must not be interpreted as convergence.

Frozen inverse/action and dual-alignment diagnostics are NaN for incompatible zero-metric support or undefined zero-vector directions. Requiring action support is conservative; the own-anchor margin can still use a weighted seminorm when metric entries are zero. See [incoming_analysis.md](incoming_analysis.md) for the every-step transformed-moment analysis.

## Continuation and provenance

Resume checks the data, architecture, initialization, dynamics, diagnostic settings, and source hashes. Corrupt moment tensor counts, shapes, dtypes, negative second moments, and nonfinite buffers are rejected before ordinary optimizer state changes. Increasing the horizon is supported. A smaller target requires the explicit target-extension flag; float32-to-float64 continuation requires its explicit flag and is a numerical continuation, not a claim of an identical float64 trajectory. The log shadow currently requires float64 and cannot be reconstructed from an ordinary-only checkpoint.

New runs refuse to overwrite existing checkpoints unless resume is enabled. Atomic replacement protects checkpoint writes; loss snapshots include size/hash checks, and RNG state is saved. A signal requests a checkpoint at the next evaluation boundary. Source hashes and commit information make changes reviewable; editing the runner, model, data preparation, optimizer, or enabled shadow code requires a new output directory rather than silently changing a trajectory on resume.

The optional theory metadata labels schedule regions and conditional asymptotic guides. It does not prove parameter-direction convergence, a positive limiting margin, the manuscript's regularity assumptions, or that a finite run reached an asymptotic regime.

A different Git commit is allowed on resume only when the nonempty computational source-hash maps match exactly; both commit IDs are recorded in `resume_migrations`. Documentation or plotting-only commits can therefore extend a run, while changed computational code and legacy checkpoints without hashes retain strict rejection.
