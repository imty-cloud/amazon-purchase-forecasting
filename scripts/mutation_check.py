"""Plants known boundary bugs in a temporary copy of src/apf and checks that the test suite fails.

Usage: .venv/bin/python scripts/mutation_check.py   (writes outputs/stage_b_mutation_check.json)
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = [  # (name, file, original, mutated)
    ("window end c+89", "target.py", "c + pd.Timedelta(days=horizon)", "c + pd.Timedelta(days=horizon - 1)"),
    ("window includes cutoff day", "target.py", "return c + pd.Timedelta(days=1), ", "return c, "),
    ("history excludes cutoff day", "target.py", "days[days[DATE] <= pd.Timestamp(cutoff)]",
     "days[days[DATE] < pd.Timestamp(cutoff)]"),
    ("completeness guard off by one", "target.py", "if end > observed_through:",
     "if end > observed_through + pd.Timedelta(days=1):"),
    ("no START filter", "target.py", "tx.loc[tx[DATE] >= START, [ID, DATE]]", "tx[[ID, DATE]]"),
    ("duplicates keyed on 7 columns", "data.py", "a, b = _row_keys(ch)", "a, b = _row_keys(ch.drop(columns=[TITLE]))"),
    ("gift title rule case-sensitive", "data.py", "case=False, na=False", "case=True, na=False"),
    ("survey guard disabled", "target.py", "    if bad:", "    if False:"),
    ("panel keeps input columns", "target.py", "t = tx.loc[tx[DATE] >= START, [ID, DATE]]",
     "t = tx.loc[tx[DATE] >= START]"),
]


def run_tests(pythonpath):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "-o", "pythonpath=", "-m", "not realdata"],
                       cwd=ROOT, env={"PYTHONPATH": str(pythonpath), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()[-1]


def main():
    res = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn, a, b in [("none (control)", None, None, None)] + MUTANTS:
            dst = Path(tmp) / "src"
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(ROOT / "src", dst)
            if fn:
                p = dst / "apf" / fn
                s = p.read_text()
                assert s.count(a) == 1, (name, s.count(a))
                p.write_text(s.replace(a, b))
            code, last = run_tests(dst)
            res.append({"mutant": name, "tests_failed": code != 0, "pytest": last})
            print(f"{name:35s} {'CAUGHT' if code else 'not caught'}  {last}")
    ok = (not res[0]["tests_failed"]) and all(r["tests_failed"] for r in res[1:])
    (ROOT / "outputs" / "stage_b_mutation_check.json").write_text(
        json.dumps({"all_mutants_caught_and_control_passes": ok, "results": res}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
