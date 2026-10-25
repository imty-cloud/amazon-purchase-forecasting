"""Plants known boundary bugs in a temporary copy of src/apf and checks that the test suite fails.

Usage: .venv/bin/python scripts/mutation_check.py   (writes outputs/mutation_check.json)
"""
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
    # Stage C
    ("features read all data, not history", "features.py", "h = history_until(store.days, c)",
     "h = store.days"),
    ("trailing window one day too long", "features.py",
     'F[f"days_{w}"] = _count_in(h, c - pd.Timedelta(days=w - 1), c, ids)',
     'F[f"days_{w}"] = _count_in(h, c - pd.Timedelta(days=w), c, ids)'),
    ("categories_365 one day too long", "features.py", "cd[DATE] >= c - pd.Timedelta(days=364)",
     "cd[DATE] >= c - pd.Timedelta(days=365)"),
    ("categories counted after cutoff", "features.py", "cd = history_until(store.cat_days, c)",
     "cd = store.cat_days"),
    ("previous window includes c_prev", "features.py",
     "active = _count_in(h, cp + pd.Timedelta(days=1), prev_end, ids) > 0",
     "active = _count_in(h, cp, prev_end, ids) > 0"),
    ("missing prior outcome set to 0", "features.py",
     'F["prev_90d_active"] = active.astype(float).where(first <= cp)',
     'F["prev_90d_active"] = active.astype(float)'),
    ("P(y=1) column returned as P(y=0)", "evaluation.py", "classes.index(0)", "classes.index(1)"),
    ("event coded as purchase", "evaluation.py", "return (y == 0).astype(int)", "return (y == 1).astype(int)"),
    ("row bootstrap instead of respondent", "evaluation.py",
     "        idx = np.concatenate([rows_of[k] for k in r.integers(0, len(rows_of), len(rows_of))])\n        out.append",
     "        idx = r.integers(0, len(y), len(y))\n        out.append"),
    ("hold-out respondents in train", "splits.py",
     'p["in_train"] = p["cutoff"].isin(TRAIN_CUTOFFS) & fit_ok', 'p["in_train"] = p["cutoff"].isin(TRAIN_CUTOFFS)'),
    ("unseeded hold-out", "splits.py", "np.random.default_rng(seed)", "np.random.default_rng()"),
    # Stage D preparation
    ("top-k ties broken by row order", "evaluation.py",
     "tp = (e[above] == 1).sum() + take * (e[at] == 1).sum()",
     "tp = (e[np.argsort(-s, kind='mergesort')[:k]] == 1).sum()"),
    ("bootstrap not paired across models", "evaluation.py",
     "        for m, (p, s) in preds.items():\n            for name, fn in metric_fns.items():",
     "        for m, (p, s) in preds.items():\n            idx = np.concatenate([rows_of[k] for k in r.integers(0, len(rows_of), len(rows_of))])\n            for name, fn in metric_fns.items():"),
    ("early-stopping metric targets y = 1", "models.py",
     "average_precision_score(1 - np.asarray(y_true), -np.asarray(y_pred))",
     "average_precision_score(np.asarray(y_true), np.asarray(y_pred))"),
    ("refit ignores selected iterations", "models.py",
     '"n_estimators": int(n_estimators)}', '"n_estimators": LGBM_FIXED["n_estimators"]}'),
    ("lock ignores protocol changes", "lock.py",
     'if payload["protocol_sha256"] != sha256(protocol):', "if False:"),
    ("lock ignores uncommitted code", "lock.py", "    if dirty:\n", "    if False:\n"),
    ("phase 1 can read test rows", "lock.py",
     'out = table[table["cutoff"] < min(TEST_CUTOFFS)]', 'out = table[table["cutoff"] <= max(TEST_CUTOFFS)]'),
    # top-k estimand (protocol v1.1 §6)
    ("top-k ranked over pooled cutoffs", "evaluation.py",
     "parts = [top_fraction(y[g == v], s[g == v], frac) for v in pd.unique(g)]",
     "parts = [top_fraction(y, s, frac)]"),
    ("top-k uses floor instead of ceiling", "evaluation.py", "k = int(np.ceil(frac * len(e)))",
     "k = max(int(np.floor(frac * len(e))), 1)"),
    ("top-k averages per-cutoff precision", "evaluation.py",
     '"precision_y0": float(tp / flagged), "recall_y0": float(tp / max(events, 1))}',
     '"precision_y0": float(np.mean([p["precision_y0"] for p in parts])), "recall_y0": float(tp / max(events, 1))}'),
    # Phase 2
    ("phase 2 loads data before the lock", "phase2.py",
     "    payload = require_lock()                                   # 1. lock first, before any data",
     "    (loader or load_table)()\n    payload = require_lock()"),
    ("phase 2 refits on all pre-test rows", "phase2.py",
     'refit, train = dev[dev["in_refit"]], dev[dev["in_train"]]', 'refit, train = dev, dev[dev["in_train"]]'),
    ("phase 2 run-once guard removed", "phase2.py",
     "    if (out_dir / RESULT_FILE).exists():", "    if False:"),
    # class orientation and preprocessing
    ("calibration intercept sign flipped", "evaluation.py",
     "if np.mean(1 / (1 + np.exp(-(mid + lp)))) < e.mean():", "if np.mean(1 / (1 + np.exp(-(mid + lp)))) > e.mean():"),
    ("LR imputes by mean, not median", "models.py",
     'blocks.append((f, Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),\n                                    ("log1p"',
     'blocks.append((f, Pipeline([("impute", SimpleImputer(strategy="mean", add_indicator=True)),\n                                    ("log1p"'),
    ("reliability bins split ties by row order", "evaluation.py",
     'd["bin"] = d["p0"].rank(method="dense").astype(int) - 1',
     'd["bin"] = pd.qcut(d["p0"].rank(method="first"), n_bins, labels=False)'),
    # protocol v1.2 reporting
    ("Bonferroni level left at 95 %", "phase2.py",
     "BONF_LO, BONF_HI = 100 * 0.05 / N_PRIMARY / 2, 100 - 100 * 0.05 / N_PRIMARY / 2", "BONF_LO, BONF_HI = 2.5, 97.5"),
    ("P3 compares LightGBM, not the primary model", "phase2.py",
     '("P3", "M2_lr_full", "B2_recency", "primary")', '("P3", "M3_lgbm_full", "B2_recency", "primary")'),
    ("per-cutoff AP CI uses all cutoffs", "phase2.py", "                k = cut == c\n",
     "                k = np.ones(len(cut), bool)\n"),
    ("AP lift computed as a difference", "phase2.py",
     'st["ap_lift"] = st["ap_y0"] / st["prevalence_y0"]', 'st["ap_lift"] = st["ap_y0"] - st["prevalence_y0"]'),
    # superseding lock
    ("superseding lock accepts changed configurations", "lock.py",
     'if previous["selected"]["configurations"] != out["configurations"]:', "if False:"),
    ("superseding lock needs no reason", "lock.py", "        if not reason:\n", "        if False:\n"),
]


def run_tests(pythonpath):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "-o", "pythonpath=", "-m", "not realdata"],
                       cwd=ROOT, env={"PYTHONPATH": str(pythonpath), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()[-1]


def run_one(tmp, i, name, fn, a, b):
    dst = Path(tmp) / f"m{i:02d}" / "src"                     # one copy per mutant (runs in parallel)
    shutil.copytree(ROOT / "src", dst)
    if fn:
        p = dst / "apf" / fn
        s = p.read_text()
        assert s.count(a) == 1, (name, s.count(a))
        p.write_text(s.replace(a, b))
    code, last = run_tests(dst)
    return {"mutant": name, "tests_failed": code != 0, "pytest": last}


def main():
    jobs = [("none (control)", None, None, None)] + MUTANTS
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda ij: run_one(tmp, ij[0], *ij[1]), enumerate(jobs)))
    for r in res:
        print(f"{r['mutant']:37s} {'CAUGHT' if r['tests_failed'] else 'not caught'}  {r['pytest']}")
    ok = (not res[0]["tests_failed"]) and all(r["tests_failed"] for r in res[1:])
    (ROOT / "outputs" / "mutation_check.json").write_text(
        json.dumps({"all_mutants_caught_and_control_passes": ok, "results": res}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
