# Validation performed on 12 October 2026

This records checks performed during the one-hour extraction and audit. The
tests verify particular recurrences, identities, failure cases, and executions;
they do not certify an asymptotic directional limit. The full MNIST production
sweeps were not rerun.

## Environment and automated checks

Local execution used Python 3.12.14, PyTorch 2.13.0+cu130, torchvision
0.28.0+cu130, NumPy 2.3.5, Matplotlib 3.11.1, and pytest 9.1.1 on Windows.
CPU numerical runs used float64 and one thread. The full-width GPU smoke used
an NVIDIA RTX 500 Ada laptop GPU. The GitHub workflow additionally installs
the package and runs the tests with CPU PyTorch on Linux.

The test suite covers the discrete Adam recurrence against PyTorch and an
independent scalar oracle; changing learning rates; atomic failure rejection;
network homogeneity; analytic product-model gradients against autograd and
finite differences; layer-general initialization identities; coordinate
permutation symmetry; configuration validation; checkpoint and resume identity;
log-space moment recurrences; diagnostic scale invariance; frozen-metric
support; incoming-trace integrity; and plotting integration.

Run the suite with `python -m pytest -q`. The final local suite passed:
**162 tests and 13 subtests**, in 65.75 seconds. An installable wheel was built
and installed into an isolated target directory; both experiment modules ran
from that installed package without importing the original research folders.
The [Linux CI workflow](https://github.com/EitanGronich/adam-implicit-bias-lr-schedule/actions) also installs the package and runs this suite. The initial extraction passed all 158 then-current tests and 13 subtests on Linux in 5.48 seconds; the final resume-only additions are covered by the subsequent run.

## MNIST executions

| Check | Result |
|---|---|
| Extraction regression: 32 real MNIST points, two hidden layers of width 12, 100 updates for each power 0, 0.25, 0.7 | Parameters, losses, logits, and both moment buffers were bitwise identical to the original implementation on identical supplied inputs. This comparison used a common pixel-minus-0.5 input array; canonical loader preprocessing was tested separately. |
| Real-MNIST execution: 16 examples, 8 CPU updates | Completed; final mean logistic loss 0.6921136584. |
| Production-shape execution: 1,024 examples, 784–1024–1024–1, float64 GPU, 1,000 updates, eta0=0.0003, betas=(0.9,0.999), initialization scale 0.01 | Completed in 105 seconds; final loss 2.3769154936e-6. The 1e-4 anchor was reached at update 330. Sampled gradient-square underflow and second-moment subnormal counts were zero. This is a partial training validation, not the full production horizon. |
| Loss-anchor exercise: 32 real examples, 784–128–128–1, eta=0.003, betas=(0.9,0.99), initialization scale 1 | Reached loss 9.9711980628e-13 at update 3,034. Crossed the three fixed anchors at updates 17, 43, and 3,034. Exercised coordinate/scalar logging, log-moment shadow, every-step incoming trace, persistent loss checkpoints, and plots. |

The smaller anchor exercise is a diagnostic validation configuration; it is
not a substitute for the 1,024-example research experiment. The two production
presets each expose 45 schedule/seed/beta2 cells and their original budgets.
Those long sweeps remain to be run with the corrected diagnostic definitions.

## Actual discrete directional simulations

All checks below ran ordinary bias-corrected Adam. No ODE integration or
predicted-limit formula produced their parameter trajectories or directions.

| Check | Result |
|---|---|
| Original-source recurrence comparison, L=3, 54 trajectories, 100 updates | Initialization, gradients, parameters, and first/second moments matched bitwise. |
| Full 11-point control grid in both coordinate sectors, L=3, 2,662 trajectories, 10,000 updates | Completed without numerical guard rejection; coordinate-permutation error zero. Final support margins ranged from approximately 15.47 to 17.65. |
| Full selected time horizon, L=3, 54 grid trajectories, 400,000 updates, rho=1, eta=0.01, betas=(0.9,0.999) | Completed in 496 seconds without numerical guard rejection or parameter stagnation; permutation error zero. Final support margins 214.2195531–215.6621123. |
| General-depth run, L=4, 256 paired random trajectories, 1,000 updates, schedule power 0.25 | Completed with finite states, no guard rejection, and permutation error zero. |

For the 400,000-update check, matching grid points were selected from the
original saved 2,662-trajectory tuned run. Initial and final parameters, final
raw second moments, current gradients, and both margins matched **bitwise**.
Unit-vector differences were at most 2.23e-16 due to stable normalization.
The original comparison archive SHA-256 is
`ee980c9bbf5df0c221817cd50481ad0207201bf25793616cb4326c4b23e80814`.

Observed angular drift between updates 200,000 and 400,000 had median
1.003195 degrees and maximum 3.388183 degrees. These nonzero finite-time
drifts must not be labeled a certified limiting direction. Both-sector runs
combine coordinate-permuted datasets, as explained in [directional.md](directional.md).

The full 2,662-by-400,000 simulation was not repeated during this audit.
Existing artifacts were inspected separately to distinguish the failed
aggressive run from the later tuned run; see [audit_findings.md](audit_findings.md).

## Provenance and retained material

Original source files are identified by relative path and SHA-256 in
[provenance.json](provenance.json). New runs record source hashes, configuration,
software versions, and numerical stop reasons. The source comparison and
validation outputs remain in the local ignored `runs/`, `output/`, and `tmp/`
directories. They are deliberately excluded from this code repository, along
with datasets and checkpoints.
