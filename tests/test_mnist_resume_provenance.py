"""A docs-only commit may resume; changed or unverified code may not."""
import copy
import pytest
from adam_lr.mnist.runner import validate_resume_signature


def test_identical_source_hashes_allow_commit_and_horizon_change():
    saved={"git_commit":"old", "source_sha256":{"mnist/runner.py":"a"*64}, "max_steps":10}
    requested=copy.deepcopy(saved)
    requested.update(git_commit="new",max_steps=20)
    migrations=validate_resume_signature(saved,requested,allow_target_extension=False)
    assert migrations == ("git_commit:old->new:identical_computational_sources", "max_steps:10->20")
    assert saved["git_commit"] == "old" and requested["git_commit"] == "new"


def test_changed_computational_hash_still_rejects_different_commit():
    saved={"git_commit":"old", "source_sha256":{"mnist/runner.py":"a"*64}}
    requested={"git_commit":"new", "source_sha256":{"mnist/runner.py":"b"*64}}
    with pytest.raises(RuntimeError,match="signature mismatch"):
        validate_resume_signature(saved,requested,allow_target_extension=False)


@pytest.mark.parametrize("hashes", [None,{}])
def test_unhashed_legacy_signature_retains_exact_commit_requirement(hashes):
    saved={"git_commit":"old"}
    requested={"git_commit":"new"}
    if hashes is not None:
        saved["source_sha256"]=hashes
        requested["source_sha256"]=hashes
    with pytest.raises(RuntimeError,match="signature mismatch"):
        validate_resume_signature(saved,requested,allow_target_extension=False)
