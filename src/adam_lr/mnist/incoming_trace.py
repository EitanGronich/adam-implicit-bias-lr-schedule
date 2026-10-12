"""Buffered per-step scalar trace of incoming transformed second moments.

Loss and gradient refer to theta_t; D_t is prospective, including at the
terminal row.  The final row must not be added to the moment at that row.
Tiny detached tensors stay on their source device until a batch is flushed.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from collections.abc import Sequence

import torch


class BufferedIncomingTrace:
    """Append contiguous scalar rows, recovering exactly at a rolling checkpoint."""

    def __init__(
        self, run_dir: Path, *, signature: dict, layer_names: Sequence[str],
        resume_step: int | None = None, buffer_rows: int = 1024,
        saved_signature: dict | None = None,
    ):
        self.path = Path(run_dir) / "incoming_second_moment.jsonl"
        self.metadata_path = Path(run_dir) / "incoming_second_moment.meta.json"
        self.layer_names = tuple(layer_names)
        if not self.layer_names or len(set(self.layer_names)) != len(self.layer_names):
            raise ValueError("incoming trace layer names must be nonempty and distinct")
        if buffer_rows <= 0:
            raise ValueError("incoming trace buffer_rows must be positive")
        self.buffer_rows = int(buffer_rows)
        self._buffer: list[tuple[int, float, torch.Tensor]] = []
        self.columns = ("step", "log_loss", "log_D_total", *(
            f"log_D_{layer}" for layer in self.layer_names
        ))
        self.metadata = {
            "version": 1, "signature": signature,
            "layer_names": list(self.layer_names), "columns": list(self.columns),
            "logarithm": "natural; null represents exact zero magnitude (negative infinity)",
            "alignment": "row t: loss(theta_t), gradient g_t, prospective transition t -> t+1",
            "log_D_formula": "log(1-beta2)-(t+1)*log(beta2)+logsumexp(2*log(abs(g_t)))",
            "terminal_row": "included; prospective D_terminal is not part of V_terminal",
            "buffer_rows": self.buffer_rows,
            "resume": "retain contiguous rows below checkpoint step; recompute checkpoint row",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume_step is None:
            if self.path.exists() or self.metadata_path.exists():
                raise RuntimeError("incoming trace already exists; explicit checkpoint resume is required")
            self.path.touch()
            self.next_step = 0
            self.rows_written = 0
        else:
            if resume_step < 0 or not self.path.exists() or not self.metadata_path.exists():
                raise RuntimeError("incoming trace resume requires its existing trace and metadata")
            old = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if (old.get("version") != 1 or old.get("layer_names") != list(self.layer_names)
                    or old.get("columns") != list(self.columns)
                    or not isinstance(old.get("signature"), dict)
                    or old.get("signature") not in (signature, saved_signature)):
                raise RuntimeError("incoming trace metadata/signature mismatch")
            # Scan only the required prefix.  Torn writes and speculative rows
            # after the checkpoint are deliberately discarded, never trusted.
            with self.path.open("r+b") as stream:
                for expected in range(resume_step):
                    line = stream.readline()
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError) as error:
                        raise RuntimeError("incoming trace has a missing/corrupt checkpoint prefix") from error
                    if (not line.endswith(b"\n") or tuple(row) != self.columns
                            or row.get("step") != expected):
                        raise RuntimeError("incoming trace prefix is incomplete or out of order")
                    for column in self.columns[1:]:
                        self._clean_log(row[column])
                stream.truncate(stream.tell())
                stream.flush()
                os.fsync(stream.fileno())
            self.next_step = int(resume_step)
            self.rows_written = int(resume_step)
        self._write_metadata()

    @staticmethod
    def _clean_log(value: float | None) -> float | None:
        if value is None or value == -math.inf:
            return None
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("incoming trace logarithms cannot contain NaN or positive infinity")
        return value

    def _write_metadata(self) -> None:
        payload = {**self.metadata, "rows_written": self.rows_written,
                   "last_step_written": self.rows_written - 1 if self.rows_written else None}
        temporary = self.metadata_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.metadata_path)

    def append(
        self, step: int, log_loss: float, log_increments: Sequence[float] | torch.Tensor,
    ) -> None:
        if step != self.next_step:
            raise ValueError(f"incoming trace expected step {self.next_step}, got {step}")
        self._clean_log(log_loss)
        values = (log_increments.detach() if torch.is_tensor(log_increments)
                  else torch.tensor(log_increments, dtype=torch.float64))
        if values.ndim != 1 or values.numel() != 1 + len(self.layer_names):
            raise ValueError("incoming trace log increments must contain total followed by each layer")
        self._buffer.append((int(step), float(log_loss), values))
        self.next_step += 1
        if len(self._buffer) >= self.buffer_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        # One device transfer for the whole small scalar batch, not one per step.
        values = torch.stack([row[2] for row in self._buffer]).cpu().tolist()
        lines = []
        for (step, log_loss, _), increments in zip(self._buffer, values):
            row = dict(zip(self.columns, (step, self._clean_log(log_loss), *(
                self._clean_log(value) for value in increments
            ))))
            lines.append(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        self.rows_written += len(self._buffer)
        self._buffer.clear()
        self._write_metadata()
