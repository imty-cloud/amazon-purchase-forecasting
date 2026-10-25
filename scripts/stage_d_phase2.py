"""Stage D Phase 2: refit and evaluate both test sets once (reports/stage_d_protocol.md §5-§8).

Refuses to run unless the lock, protocol and code are committed and unchanged, and refuses if
results already exist. Usage: .venv/bin/python scripts/stage_d_phase2.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apf.phase2 import main  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    res = main()
    print(res["comparisons"].query("metric == 'ap_y0'").to_string(index=False))
    print(f"Phase 2 finished in {time.time() - t0:.0f} s")
