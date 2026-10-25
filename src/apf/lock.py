"""Test-set guard for Stage D.

Phase 1 (selection) may read only rows before the first test cutoff. Phase 2 (test) may read test
rows only after the protocol and the selected configurations are locked: the lock file records the
protocol's SHA-256, and the lock file, protocol and code must be committed and unchanged.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from .config import PROJECT_DIR
from .splits import TEST_CUTOFFS

PROTOCOL = PROJECT_DIR / "reports" / "stage_d_protocol.md"
LOCK = PROJECT_DIR / "outputs" / "stage_d_locked_config.json"
GUARDED = ("src", "scripts", "reports/stage_d_protocol.md", "outputs/stage_d_locked_config.json")


class LockError(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_before_test(table):
    """The only rows phase 1 may use: every cutoff before the first test cutoff."""
    out = table[table["cutoff"] < min(TEST_CUTOFFS)]
    assert not out["cutoff"].isin(TEST_CUTOFFS).any()
    return out


def lock_contents(selection_path, head, protocol=PROTOCOL, previous=None, reason=None, previous_tag=None):
    """The `selected` part of a lock, built from a Phase 1 selection file. With `previous` (the lock being
    superseded) a reason is required and the configurations must be identical to the previous lock's."""
    sel = json.loads(Path(selection_path).read_text())
    if sel["protocol_sha256_at_run"] != sha256(protocol):
        raise LockError("protocol changed after Phase 1 ran")
    out = {"configurations": sel["selected"], "data_md5": sel["data_md5"], "environment": sel["environment"],
           "seeds": {**sel["seeds"], "bootstrap": 20261024, "bootstrap_draws": 1000},
           "phase1_selection_file": Path(selection_path).name,
           "phase1_selection_sha256": sha256(selection_path), "phase1_commit": head}
    if previous is not None:
        if not reason:
            raise LockError("a superseding lock needs a reason")
        if previous["selected"]["configurations"] != out["configurations"]:
            raise LockError("configurations differ from the superseded lock")
        out["supersedes"] = {"protocol_sha256": previous["protocol_sha256"], "written": previous["written"],
                             "phase1_selection_sha256": previous["selected"]["phase1_selection_sha256"],
                             "phase1_commit": previous["selected"]["phase1_commit"], "tag": previous_tag,
                             "reason": reason}
    return out


def write_lock(selected, protocol=PROTOCOL, lock=LOCK):
    payload = {"protocol_file": str(Path(protocol).name), "protocol_sha256": sha256(protocol),
               "written": time.strftime("%Y-%m-%d %H:%M:%S"), "selected": selected}
    Path(lock).write_text(json.dumps(payload, indent=1, default=str))
    return payload


def _git_dirty(repo, paths):
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
                       cwd=repo, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def require_lock(protocol=PROTOCOL, lock=LOCK, repo=PROJECT_DIR, guarded=GUARDED):
    if not Path(lock).exists():
        raise LockError("no lock file: run model selection and commit the lock first")
    payload = json.loads(Path(lock).read_text())
    if payload["protocol_sha256"] != sha256(protocol):
        raise LockError("protocol changed after the configuration was locked")
    dirty = _git_dirty(repo, guarded)
    if dirty:
        raise LockError(f"uncommitted changes in guarded paths:\n{dirty}")
    return payload


def release_test_rows(table, **kw):
    """Test-cutoff rows, released only under a valid lock."""
    payload = require_lock(**kw)
    return table[table["cutoff"].isin(TEST_CUTOFFS)], payload
