"""Resume rejects changed scientific settings or corrupted checkpoint prefixes."""
import pytest
from adam_lr.mnist import runner
from test_mnist_training import arguments, dataset, SPEC


@pytest.mark.parametrize("change", ["epsilon", "bias", "target_loss", "source"])
def test_resume_rejects_changed_run_without_modifying_checkpoint(tmp_path, monkeypatch, change):
    x, y = dataset()
    runner.STOP_REQUESTED = False
    runner.run_one(args=arguments(tmp_path), spec=SPEC, x=x, y=y)
    checkpoint = tmp_path / "trial/checkpoint.pt"
    trace = tmp_path / "trial/trace.bin"
    before = checkpoint.read_bytes(), trace.read_bytes()
    args = arguments(tmp_path, steps=7, resume=True)
    if change == "source":
        previous = runner.source_metadata
        def changed():
            result = previous()
            result["module_sha256"]["optim/adam.py"] = "changed"
            return result
        monkeypatch.setattr(runner, "source_metadata", changed)
    else:
        setattr(args, change, {"epsilon": 1e-8, "bias": True, "target_loss": 1e-15}[change])
    with pytest.raises(RuntimeError, match="signature mismatch"):
        runner.run_one(args=args, spec=SPEC, x=x, y=y)
    assert (checkpoint.read_bytes(), trace.read_bytes()) == before


def test_resume_rejects_corrupt_committed_trace_without_truncating(tmp_path):
    x, y = dataset()
    runner.STOP_REQUESTED = False
    runner.run_one(args=arguments(tmp_path), spec=SPEC, x=x, y=y)
    checkpoint = tmp_path / "trial/checkpoint.pt"
    trace = tmp_path / "trial/trace.bin"
    damaged = bytearray(trace.read_bytes())
    damaged[12] ^= 1
    trace.write_bytes(damaged)
    before = checkpoint.read_bytes(), trace.read_bytes()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runner.run_one(args=arguments(tmp_path, steps=7, resume=True), spec=SPEC, x=x, y=y)
    assert (checkpoint.read_bytes(), trace.read_bytes()) == before


def test_disabled_resume_and_reduced_budget_do_not_overwrite(tmp_path):
    x, y = dataset()
    runner.STOP_REQUESTED = False
    runner.run_one(args=arguments(tmp_path), spec=SPEC, x=x, y=y)
    checkpoint = tmp_path / "trial/checkpoint.pt"
    before = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_one(args=arguments(tmp_path), spec=SPEC, x=x, y=y)
    with pytest.raises(ValueError, match="budget is below"):
        runner.run_one(args=arguments(tmp_path, steps=3, resume=True), spec=SPEC, x=x, y=y)
    assert checkpoint.read_bytes() == before
