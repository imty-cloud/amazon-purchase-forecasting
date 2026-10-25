"""Writes outputs/stage_d_locked_config.json from the committed Phase 1 selection (protocol §7).

Refuses if the protocol changed since Phase 1 ran, if guarded paths have uncommitted changes, or if a
lock exists and --supersede is not given. --supersede REASON replaces an existing lock and records
what it supersedes (the old lock stays in git history at its tag). --dry-run prints the contents only.
Usage: .venv/bin/python scripts/stage_d_write_lock.py [--supersede "reason"] [--dry-run]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apf import config as C                                          # noqa: E402
from apf.lock import LOCK, PROTOCOL, _git_dirty, lock_contents, sha256, write_lock  # noqa: E402

SELECTION = C.OUTPUT_DIR / "stage_d_phase1_selection.json"
PREVIOUS_TAG = "stage-d-lock"


def main(supersede=None, dry_run=False):
    if LOCK.exists() and not supersede:
        sys.exit(f"refusing: {LOCK.name} exists; use --supersede \"<reason>\"")
    if supersede and not LOCK.exists():
        sys.exit("refusing: --supersede given but there is no lock to supersede")
    rel = lambda p: str(Path(p).relative_to(C.PROJECT_DIR))
    dirty = _git_dirty(C.PROJECT_DIR, ("src", "scripts", rel(PROTOCOL), rel(SELECTION), rel(LOCK)))
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.PROJECT_DIR, capture_output=True,
                          text=True, check=True).stdout.strip()
    previous = json.loads(LOCK.read_text()) if supersede else None
    selected = lock_contents(SELECTION, head, previous=previous, reason=supersede,
                             previous_tag=PREVIOUS_TAG if supersede else None)
    if dry_run:
        print(json.dumps({"protocol_file": PROTOCOL.name, "protocol_sha256": sha256(PROTOCOL),
                          "written": f"<at write time; dry run {time.strftime('%Y-%m-%d %H:%M:%S')}>",
                          "selected": selected}, indent=1))
        return
    print(json.dumps(write_lock(selected), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--supersede", metavar="REASON")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.supersede, a.dry_run)
