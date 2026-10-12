"""Record the code actually executed, including uncommitted source changes."""
from __future__ import annotations

import hashlib
from pathlib import Path
import platform
import subprocess


def source_metadata() -> dict:
    package = Path(__file__).resolve().parent
    hashes = {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(package.rglob("*.py"))}
    # This works in an editable checkout. Installed wheels still carry the
    # exact module hashes, even if there is no Git executable or checkout.
    root = package.parent.parent
    commit, dirty = None, None
    if (root / ".git").exists():
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                    check=True, capture_output=True, text=True, timeout=5).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "pyproject.toml"],
                                        cwd=root, check=True, capture_output=True, text=True, timeout=5).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return {"python_version": platform.python_version(), "git_commit": commit,
            "git_dirty": dirty, "module_sha256": hashes}
