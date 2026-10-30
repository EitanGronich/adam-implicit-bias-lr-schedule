"""Trace integrity and interrupted-write recovery use the checkpoint prefix."""
import json

import numpy as np
import pytest

from adam_lr.mnist.traces import METRICS, ROW_BYTES, TraceWriter, load_trace


VALUES = (1., 2., 3., .1, .2, .01, 1.)


def completed(path):
    path.mkdir()
    writer = TraceWriter(path, {"test": 1}, buffer_rows=2)
    writer.append(0, 1., True, VALUES)
    writer.append(1, .9, True, VALUES)
    writer.append(2, .8, False, VALUES)
    state = writer.commit(2, "max_steps")
    (path / "results.json").write_text(json.dumps(dict(signature={"test": 1}, final_step=2,
                                                       completion_reason="max_steps")))
    return state


def test_buffer_size_does_not_change_saved_observations(tmp_path):
    first = tmp_path / "first"
    completed(first)
    second = tmp_path / "second"
    second.mkdir()
    writer = TraceWriter(second, {"test": 1}, buffer_rows=100)
    for step, loss in enumerate((1., .9, .8)):
        writer.append(step, loss, step < 2, VALUES)
    writer.commit(2, "max_steps")
    (second / "results.json").write_bytes((first / "results.json").read_bytes())
    assert (first / "trace.bin").read_bytes() == (second / "trace.bin").read_bytes()
    left, right = load_trace(first), load_trace(second)
    assert left["metric_columns"] == METRICS
    np.testing.assert_array_equal(left["values"], right["values"])


def test_resume_drops_terminal_row_and_uncommitted_tail(tmp_path):
    path = tmp_path / "trace"
    saved = completed(path)
    prefix = (path / "trace.bin").read_bytes()[:2 * ROW_BYTES]
    with (path / "trace.bin").open("ab") as stream:
        stream.write(b"interrupted write")
    writer = TraceWriter(path, {"test": 1}, step=2, saved_state=saved)
    writer.append(2, .8, True, VALUES)
    writer.append(3, .7, False, VALUES)
    writer.commit(3, "max_steps")
    (path / "results.json").write_text(json.dumps(dict(signature={"test": 1}, final_step=3,
                                                       completion_reason="max_steps")))
    assert (path / "trace.bin").read_bytes().startswith(prefix)
    assert load_trace(path)["update_applied"].tolist() == [True, True, True, False]


@pytest.mark.parametrize("damage", ["truncate", "replace", "metadata"])
def test_loader_rejects_corrupt_or_mismatched_data(tmp_path, damage):
    path = tmp_path / "trace"
    completed(path)
    binary = path / "trace.bin"
    if damage == "truncate":
        binary.write_bytes(binary.read_bytes()[:-8])
    elif damage == "replace":
        values = bytearray(binary.read_bytes())
        values[25] ^= 1
        binary.write_bytes(values)
    else:
        data = json.loads((path / "trace.json").read_text())
        data["signature"] = {"test": 2}
        (path / "trace.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mismatch|differs"):
        load_trace(path)
