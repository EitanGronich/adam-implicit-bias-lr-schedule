"""Buffered per-update observations with checkpoint-bound integrity checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


METRICS = ("raw_margin", "theta_linf", "theta_l2", "margin_linf", "margin_l2",
           "epsilon_update_change_squared", "epsilon_update_change_squared_valid")
FIELDS = ("step", "loss", "update_applied", *METRICS)
ROW_BYTES = 8 * len(FIELDS)
FORMAT = "adam_mnist_trace_v1"


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prefix_hash(path, byte_count):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = byte_count
        while remaining:
            block = stream.read(min(8 * 1024**2, remaining))
            if not block:
                raise ValueError("The trace is shorter than its committed checkpoint prefix")
            digest.update(block)
            remaining -= len(block)
    return digest


class TraceWriter:
    """Append observations; only a saved checkpoint commits a recoverable prefix."""
    def __init__(self, run_dir, signature, *, buffer_rows=1024, step=0, saved_state=None):
        if type(buffer_rows) is not int or buffer_rows < 1:
            raise ValueError("trace_buffer_rows must be positive")
        self.path = Path(run_dir) / "trace.bin"
        self.metadata = Path(run_dir) / "trace.json"
        self.signature, self.buffer_rows = signature, buffer_rows
        self.buffer = []
        if saved_state is None:
            if self.path.exists() or self.metadata.exists():
                raise FileExistsError(f"Trace already exists: {self.path}")
            self.path.touch()
            self.rows, self.digest = 0, hashlib.sha256()
        else:
            rows = saved_state.get("rows")
            if type(rows) is not int or rows not in (step, step + 1):
                raise ValueError("Checkpoint trace count disagrees with its optimizer step")
            digest = _prefix_hash(self.path, rows * ROW_BYTES)
            if digest.hexdigest() != saved_state.get("sha256"):
                raise ValueError("Checkpoint trace SHA256 mismatch")
            # A terminal observation at theta_t is regenerated when training resumes.
            self.rows, self.digest = step, _prefix_hash(self.path, step * ROW_BYTES)
            with self.path.open("r+b") as stream:
                stream.truncate(step * ROW_BYTES)

    def append(self, step, loss, applied, values):
        if step != self.rows + len(self.buffer) or len(values) != len(METRICS):
            raise ValueError("Trace rows must be contiguous with the expected metric columns")
        self.buffer.append((step, loss, float(applied), *values))
        if len(self.buffer) >= self.buffer_rows:
            self.flush()

    def flush(self):
        if self.buffer:
            data = np.asarray(self.buffer, dtype="<f8").tobytes()
            with self.path.open("ab") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            self.digest.update(data)
            self.rows += len(self.buffer)
            self.buffer.clear()

    def state(self):
        self.flush()
        return {"rows": self.rows, "sha256": self.digest.hexdigest()}

    def commit(self, final_step, completion_reason):
        state = self.state()
        _write_json(self.metadata, dict(format=FORMAT, fields=FIELDS, dtype="<f8",
                    row_bytes=ROW_BYTES, signature=self.signature, final_step=final_step,
                    completion_reason=completion_reason, **state))
        return state


def load_trace(run_dir, *, signature=None, final_step=None, completion_reason=None):
    """Read the complete committed trace for plotting or figure preparation."""
    root = Path(run_dir)
    if signature is None or final_step is None or completion_reason is None:
        result = json.loads((root / "results.json").read_text(encoding="utf-8"))
        signature = result["signature"] if signature is None else signature
        final_step = result["final_step"] if final_step is None else final_step
        completion_reason = result["completion_reason"] if completion_reason is None else completion_reason
    if completion_reason not in ("target_loss", "max_steps", "signal"):
        raise ValueError("Cannot analyze an unfinished or numerically failed experiment")
    meta = json.loads((root / "trace.json").read_text(encoding="utf-8"))
    if (meta.get("format") != FORMAT or tuple(meta.get("fields", ())) != FIELDS
            or meta.get("dtype") != "<f8" or meta.get("row_bytes") != ROW_BYTES
            or meta.get("signature") != signature or meta.get("final_step") != final_step
            or meta.get("completion_reason") != completion_reason
            or meta.get("rows") != final_step + 1):
        raise ValueError("Trace metadata differs from the experiment results")
    path = root / "trace.bin"
    size = meta["rows"] * ROW_BYTES
    if path.stat().st_size != size or _prefix_hash(path, size).hexdigest() != meta.get("sha256"):
        raise ValueError("Trace size or SHA256 mismatch")
    rows = np.fromfile(path, dtype="<f8").reshape(meta["rows"], len(FIELDS))
    if (not np.array_equal(rows[:, 0], np.arange(final_step + 1))
            or not np.isfinite(rows[:, 1]).all() or (rows[:, 1] <= 0).any()
            or not np.all(rows[:-1, 2] == 1) or rows[-1, 2] != 0):
        raise ValueError("Trace observations or applied-update flags are inconsistent")
    return dict(step=rows[:, 0].astype(np.int64), loss=rows[:, 1].copy(),
                update_applied=rows[:, 2].astype(bool), metric_columns=METRICS,
                values=rows[:, 3:].copy())
