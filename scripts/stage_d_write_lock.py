"""Writes outputs/stage_d_locked_config.json from the committed Phase 1 selection (protocol §7).

Refuses if a lock already exists, if the protocol changed since Phase 1 ran, or if the guarded
paths (src, scripts, protocol) have uncommitted changes.
Usage: .venv/bin/python scripts/stage_d_write_lock.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apf import config as C                                   # noqa: E402
from apf.lock import LOCK, PROTOCOL, _git_dirty, sha256, write_lock  # noqa: E402

SELECTION = C.OUTPUT_DIR / "stage_d_phase1_selection.json"


def main():
    if LOCK.exists():
        sys.exit(f"refusing: {LOCK.name} already exists")
    dirty = _git_dirty(C.PROJECT_DIR, ("src", "reports/stage_d_protocol.md", str(SELECTION.relative_to(C.PROJECT_DIR))))
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    sel = json.loads(SELECTION.read_text())
    if sel["protocol_sha256_at_run"] != sha256(PROTOCOL):
        sys.exit("refusing: protocol changed after Phase 1 ran")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.PROJECT_DIR, capture_output=True,
                          text=True, check=True).stdout.strip()
    payload = write_lock({
        "configurations": sel["selected"],
        "data_md5": sel["data_md5"],
        "environment": sel["environment"],
        "seeds": {**sel["seeds"], "bootstrap": 20261024, "bootstrap_draws": 1000},
        "phase1_selection_file": SELECTION.name,
        "phase1_selection_sha256": sha256(SELECTION),
        "phase1_commit": head,
    })
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
