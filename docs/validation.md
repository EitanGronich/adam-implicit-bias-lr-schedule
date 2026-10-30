# Tests

Run the numerical and integration tests from the repository root:

```sh
python -m pytest -q
```

The tests cover Adam's recurrence, gradients, initialization, configuration,
checkpoint resume, trace timing, statistical preprocessing, and figure
exports. Small synthetic models exercise training on CPU. Full MNIST sweeps
are run through the same command described in [MNIST experiments](mnist.md).
