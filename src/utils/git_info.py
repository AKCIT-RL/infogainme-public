"""Helpers to capture git state at runtime for experiment reproducibility."""

from __future__ import annotations

import os
import subprocess
import threading
from functools import lru_cache


def _run(args: list[str]) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_git_info_locked() -> dict[str, str | bool | None]:
    # Prefer env vars injected by the launcher (e.g. inside Singularity where
    # git is not installed). GIT_COMMIT / GIT_BRANCH / GIT_DIRTY are set by
    # the dgx/ shell scripts before entering the container.
    commit = os.environ.get("GIT_COMMIT") or _run(["git", "rev-parse", "HEAD"])
    branch = os.environ.get("GIT_BRANCH") or _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty_env = os.environ.get("GIT_DIRTY")
    if dirty_env is not None:
        dirty: bool | None = dirty_env.lower() in ("1", "true", "yes")
    else:
        # --untracked-files=no skips walking large untracked dirs like outputs/
        status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
        dirty = bool(status) if status is not None else None
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def get_git_info() -> dict[str, str | bool | None]:
    """Return current repo git state: commit, branch, dirty flag.

    Reads GIT_COMMIT / GIT_BRANCH / GIT_DIRTY env vars first (set by the
    dgx/ launchers before entering the Singularity container where git is
    unavailable), then falls back to running git subprocesses directly.

    Cached per-process. Thread-safe on first call — benchmarks fan out games
    via ThreadPoolExecutor, so without the lock multiple workers would race
    and fork parallel git subprocesses. Returns None values outside a git repo.
    """
    with _lock:
        return _get_git_info_locked()
