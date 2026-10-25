"""Stage D Phase 1: model selection, fit on Train and score Tune (reports/stage_d_protocol.md §4).

Reads only non-hold-out rows at cutoffs before the first test cutoff. Labels are built only for
those cutoffs; hold-out membership comes from respondent IDs (history <= 2022-06-30), not labels.
Writes the candidate selection, not the lock (the lock needs approval, protocol §7).
Usage: .venv/bin/python scripts/stage_d_phase1.py [--out-dir DIR]
"""
import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apf import config as C                                   # noqa: E402
from apf import models as Mo                                  # noqa: E402
from apf.data import load_transactions, verify_file           # noqa: E402
from apf.evaluation import (PersistenceBaseline, PrevalenceBaseline, RecencyRule, metrics,  # noqa: E402
                            p_no_purchase, top_fraction_by_group)
from apf.features import build_feature_panel, make_daily_store  # noqa: E402
from apf.lock import PROTOCOL, rows_before_test, sha256       # noqa: E402
from apf.splits import (TEST_CUTOFFS, TRAIN_CUTOFFS, TUNE_CUTOFFS, assign, check_design,  # noqa: E402
                        holdout_respondents)
from apf.target import build_panel, eligible_respondents, purchase_days  # noqa: E402

ID = C.ID


def ds(t):
    return str(pd.Timestamp(t).date())


def tune_report(y, p0, score0, cut, p_fit):
    m = metrics(y, p0, score0, train_prevalence0=p_fit)
    s = p0 if score0 is None else score0
    m["per_cutoff_ap_y0"] = {ds(c): metrics(y[cut == c], p0[cut == c], s[cut == c])["ap_y0"]
                             for c in TUNE_CUTOFFS}
    for frac in (0.10, 0.20):
        m[f"top{int(frac * 100)}"] = top_fraction_by_group(y, s, cut, frac)
    return m


def main(out_dir):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    f = verify_file(C.PURCHASES_CSV)
    assert f["md5_match"], f
    tx, _ = load_transactions(C.PURCHASES_CSV)
    store, days = make_daily_store(tx), purchase_days(tx)

    # data: labels only for cutoffs before the first test cutoff
    dev = [c for c in C.CUTOFFS if c < min(TEST_CUTOFFS)]
    assert not set(dev) & set(TEST_CUTOFFS) and len(dev) == 13
    lab = build_panel(days, dev)[[ID, "cutoff", "y"]]
    tab = lab.merge(build_feature_panel(store, dev), on=[ID, "cutoff"], how="inner", validate="one_to_one")
    assert len(tab) == len(lab)
    tab = rows_before_test(tab)
    hold = holdout_respondents(eligible_respondents(days, C.CUTOFFS[-1]))      # IDs only
    tab = assign(tab, hold)
    check_design(tab)
    tab = tab[~tab["holdout"]]                                                  # never used in Phase 1
    tr, tu = tab[tab["in_train"]], tab[tab["in_tune"]]
    assert set(tr["cutoff"]) == set(TRAIN_CUTOFFS) and set(tu["cutoff"]) == set(TUNE_CUTOFFS)

    # reconcile with Stage C (outputs/stage_c_split_summary.csv)
    sc = pd.read_csv(C.OUTPUT_DIR / "stage_c_split_summary.csv").set_index("role")
    counts = {"train": (len(tr), int((tr["y"] == 0).sum())), "tune": (len(tu), int((tu["y"] == 0).sum()))}
    for role, key in (("in_train", "train"), ("in_tune", "tune")):
        assert counts[key] == (sc.loc[role, "obs"], sc.loc[role, "y0"]), (key, counts[key])
    assert len(hold) == 1002

    ytr, ytu, cut = tr["y"].to_numpy(), tu["y"].to_numpy(), tu["cutoff"].to_numpy()
    p_fit = float((ytr == 0).mean())
    grid, tune_metrics, warn_log, p0_sel, notes = [], {}, [], {}, {}

    # baselines (fitted on Train)
    for name, b in (("B0_prevalence", PrevalenceBaseline()), ("B1_persistence", PersistenceBaseline()),
                    ("B2_recency", RecencyRule())):
        b.fit(tr, ytr)
        tune_metrics[name] = tune_report(ytu, b.predict_p0(tu), b.score0(tu) if hasattr(b, "score0") else None,
                                         cut, p_fit)

    # logistic regression: C grid per feature set
    selected = {}
    for mid, fs in (("M1_lr_rfm", "rfm"), ("M2_lr_full", "full"), ("S1_lr_no_prev", "no_prev")):
        rows = []
        for Cv in Mo.LR_C_GRID:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                m = Mo.make_logreg(Mo.FEATURE_SETS[fs], Cv).fit(tr, ytr)
            conv = not any(issubclass(x.category, ConvergenceWarning) for x in w)
            warn_log += [{"model": mid, "C": Cv, "category": x.category.__name__, "message": str(x.message)[:200]}
                         for x in w]
            p0 = p_no_purchase(m, tu)
            r = {"model": mid, "feature_set": fs, "C": Cv, "converged": conv,
                 "n_iter": int(m.named_steps["lr"].n_iter_[0]), "tune_ap_y0": metrics(ytu, p0)["ap_y0"]}
            rows.append(r)
            grid.append(r)
        best = Mo.select([r for r in rows if r["converged"]], tiebreak=("C",))
        m = Mo.make_logreg(Mo.FEATURE_SETS[fs], best["C"]).fit(tr, ytr)
        selected[mid] = {"feature_set": fs, "C": best["C"]}
        p0_sel[mid] = p_no_purchase(m, tu)
        tune_metrics[mid] = tune_report(ytu, p0_sel[mid], None, cut, p_fit)
        notes[f"{mid}_mean_p0_train_vs_observed"] = (float(p_no_purchase(m, tr).mean()), p_fit)

    # LightGBM: 6 configurations, early stopping on Tune AP for y = 0
    for mid, fs in (("M3_lgbm_full", "full"), ("S2_lgbm_no_prev", "no_prev")):
        cols = Mo.FEATURE_SETS[fs]
        rows, fitted = [], {}
        for cfg in Mo.lgbm_configs():
            t1 = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                m = Mo.fit_lgbm_early_stopping(tr[cols], ytr, tu[cols], ytu, cfg)
            warn_log += [{"model": mid, **cfg, "category": x.category.__name__, "message": str(x.message)[:200]}
                         for x in w]
            p0 = p_no_purchase(m, tu[cols])
            ours = metrics(ytu, p0)["ap_y0"]
            lgb_best = m.best_score_["valid_0"]["ap_no_purchase"]
            r = {"model": mid, "feature_set": fs, **cfg, "best_iteration": int(m.best_iteration_),
                 "hit_cap": bool(m.best_iteration_ >= Mo.LGBM_FIXED["n_estimators"]),
                 "tune_ap_y0": ours, "lightgbm_best_score": float(lgb_best),
                 "metric_agreement": bool(abs(ours - lgb_best) < 1e-9),
                 "neg_min_child_samples": -cfg["min_child_samples"], "seconds": round(time.time() - t1, 1)}
            rows.append(r)
            grid.append(r)
            fitted[(cfg["num_leaves"], cfg["min_child_samples"])] = m
        best = Mo.select(rows, tiebreak=("num_leaves", "neg_min_child_samples"))
        cfg = {"num_leaves": best["num_leaves"], "min_child_samples": best["min_child_samples"]}
        m = fitted[(cfg["num_leaves"], cfg["min_child_samples"])]
        selected[mid] = {"feature_set": fs, **cfg, "best_iteration": best["best_iteration"]}
        p0_sel[mid] = p_no_purchase(m, tu[cols])
        tune_metrics[mid] = tune_report(ytu, p0_sel[mid], None, cut, p_fit)
        if fs == "full":
            imp = dict(zip(m.booster_.feature_name(),
                           m.booster_.feature_importance("split", iteration=m.best_iteration_).tolist()))
            notes["M3_splits_on_prev_90d_active"] = int(imp["prev_90d_active"])

    # Phase 1 notes (tune and train rows only)
    notes["M3_vs_S2_identical_tune_predictions"] = bool(np.array_equal(p0_sel["M3_lgbm_full"],
                                                                       p0_sel["S2_lgbm_no_prev"]))
    notes["M2_vs_S1_identical_tune_predictions"] = bool(np.array_equal(p0_sel["M2_lr_full"], p0_sel["S1_lr_no_prev"]))
    notes["tune_observed_y0_rate"] = float((ytu == 0).mean())
    for mid in ("M1_lr_rfm", "M2_lr_full", "S1_lr_no_prev", "M3_lgbm_full"):
        notes[f"{mid}_mean_p0_tune"] = float(p0_sel[mid].mean())

    import lightgbm, sklearn                                       # noqa: E401
    res = {
        "run_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_sha256_at_run": sha256(PROTOCOL), "data_md5": f["md5"],
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__},
        "seeds": {"holdout": 20261024, "lightgbm": Mo.SEED},
        "rows": {"train": counts["train"][0], "train_y0": counts["train"][1],
                 "tune": counts["tune"][0], "tune_y0": counts["tune"][1],
                 "holdout_respondents_excluded": int(len(hold))},
        "selected": selected,
        "tune_metrics": tune_metrics,
        "warnings": warn_log,
        "all_lightgbm_metric_agreement": all(r.get("metric_agreement", True) for r in grid),
        "any_lightgbm_hit_cap": any(r.get("hit_cap", False) for r in grid),
        "lr_nonconverged": [(r["model"], r["C"]) for r in grid if r.get("converged") is False],
        "notes": notes,
    }
    pd.DataFrame(grid).drop(columns=["neg_min_child_samples"], errors="ignore").to_csv(
        out_dir / "stage_d_tune_results.csv", index=False)
    res["runtime_seconds"] = round(time.time() - t0, 1)
    (out_dir / "stage_d_phase1_selection.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("rows", "selected", "all_lightgbm_metric_agreement",
                                         "any_lightgbm_hit_cap", "lr_nonconverged", "runtime_seconds")},
                     indent=1, default=str))
    print(f"warnings recorded: {len(warn_log)}")
    if out_dir == C.OUTPUT_DIR:
        write_report(res, pd.DataFrame(grid))


def write_report(res, grid):
    """reports/stage_d_phase1_results.md from this run's outputs (no hand-typed numbers)."""
    tm, n, sel = res["tune_metrics"], res["notes"], res["selected"]
    rows = "\n".join(
        f"| {m} | {v['ap_y0']:.3f} | {v['ap_no_skill']:.3f} | {v['roc_auc']:.3f} | {v['brier']:.4f} | "
        f"{v['brier_skill']:.3f} | {v['cal_slope']:.2f} / {v['cal_intercept']:+.2f} | "
        f"{v['top10']['precision_y0']:.3f} / {v['top10']['recall_y0']:.3f} | "
        f"{v['top20']['precision_y0']:.3f} / {v['top20']['recall_y0']:.3f} |" for m, v in tm.items())
    lgb = grid[grid["model"].str.contains("lgbm")]
    lgb_rows = "\n".join(f"| {r.model} | {int(r.num_leaves)} | {int(r.min_child_samples)} | {int(r.best_iteration)} | "
                          f"{r.tune_ap_y0:.4f} |" for r in lgb.itertuples())
    lr = grid[grid["model"].str.contains("lr")]
    lr_rows = "\n".join(f"| {r.model} | {r.C:g} | {r.converged} | {int(r.n_iter)} | {r.tune_ap_y0:.4f} |"
                         for r in lr.itertuples())
    md = f"""# Stage D Phase 1 — model selection results

**Run:** {res['run_started']} · {res['runtime_seconds']} s · protocol SHA-256 `{res['protocol_sha256_at_run'][:16]}…` ·
data MD5 `{res['data_md5']}`. Generated by `scripts/stage_d_phase1.py`.

Train: {res['rows']['train']:,} rows ({res['rows']['train_y0']:,} y = 0). Tune: {res['rows']['tune']:,} rows
({res['rows']['tune_y0']:,} y = 0). The {res['rows']['holdout_respondents_excluded']:,} hold-out respondents are
excluded. **No test-cutoff label was constructed.** The numbers below are *selection-set* results (optimistic,
especially for LightGBM) and are not performance estimates.

## Selected configurations

```json
{json.dumps(sel, indent=1)}
```

## Tune metrics (both tune cutoffs pooled)

| Model | AP (y = 0) | no-skill AP | ROC-AUC | Brier | Brier skill | calibration slope / intercept | P / R @10 % | P / R @20 % |
|---|---|---|---|---|---|---|---|---|
{rows}

## Grids

| Model | C | converged | iterations | tune AP |
|---|---|---|---|---|
{lr_rows}

| Model | num_leaves | min_child_samples | best iteration | tune AP |
|---|---|---|---|---|
{lgb_rows}

Checks:
- warnings recorded: {len(res['warnings'])};
- LightGBM's reported best score equals our AP in every configuration: {res['all_lightgbm_metric_agreement']};
- any configuration at the 3000-iteration cap: {res['any_lightgbm_hit_cap']};
- non-converged logistic regressions: {res['lr_nonconverged'] or 'none'}.

## Notes

1. **LightGBM did not use `prev_90d_active`.**
   - The selected full model makes {n['M3_splits_on_prev_90d_active']} splits on it.
   - The models with and without it give identical tune predictions: {n['M3_vs_S2_identical_tune_predictions']}.
   - This holds for these data, this grid and this training period. It does not show that the feature is
     redundant in general. The logistic regressions with and without it give identical tune predictions:
     {n['M2_vs_S1_identical_tune_predictions']}. The sensitivity
     model S2 is kept as specified.
2. **Calibration-in-the-large on tune.**
   - The observed rate of y = 0 is {n['tune_observed_y0_rate']:.4f}.
   - Mean predicted P(y = 0) is {n['M1_lr_rfm_mean_p0_tune']:.4f} (LR, RFM only), {n['M2_lr_full_mean_p0_tune']:.4f} (LR, full),
     {n['S1_lr_no_prev_mean_p0_tune']:.4f} (LR, without `prev_90d_active`) and {n['M3_lgbm_full_mean_p0_tune']:.4f} (LightGBM).
   - On the training rows, LR full reproduces the training rate
     ({n['M2_lr_full_mean_p0_train_vs_observed'][0]:.4f} vs {n['M2_lr_full_mean_p0_train_vs_observed'][1]:.4f}).
   - So the logistic regressions under-predict y = 0 in the tune period. This is documented; no recalibration
     is applied (protocol §6).
"""
    (C.PROJECT_DIR / "reports" / "stage_d_phase1_results.md").write_text(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=C.OUTPUT_DIR)
    main(ap.parse_args().out_dir)
