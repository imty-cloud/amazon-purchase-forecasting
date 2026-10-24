"""Stage C: features, leakage checks, split design checks, baseline smoke test (training period only).

No main model is trained. No metric is computed on tune or test labels.
Usage: .venv/bin/python scripts/stage_c_features.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apf import config as C                                   # noqa: E402
from apf import splits as S                                   # noqa: E402
from apf.data import data_version, load_transactions, verify_file  # noqa: E402
from apf.evaluation import PersistenceBaseline, PrevalenceBaseline, RecencyRule, metrics  # noqa: E402
from apf.features import (FEATURE_COLUMNS, FEATURES, build_feature_panel, features_at,  # noqa: E402
                          make_daily_store, previous_quarter_end)
from apf.target import build_panel, labels_at, purchase_days   # noqa: E402

OUT = C.OUTPUT_DIR
ID, DATE = C.ID, C.DATE
ROLES = ["in_train", "in_tune", "in_refit", "in_test_temporal", "in_test_holdout",
         "in_holdout_train_period"]


def ds(t):
    return str(pd.Timestamp(t).date())


def main():
    t0 = time.time()
    R = {"run_started": time.strftime("%Y-%m-%d %H:%M:%S")}
    assert verify_file(C.PURCHASES_CSV)["md5_match"]
    tx, _ = load_transactions(C.PURCHASES_CSV)

    # 1. features (primary: raw; sensitivity: no gift cards) and labels
    tables = {}
    for v, ycol in (("raw", "y"), ("nogift", "y_nogift")):
        t = data_version(tx, v)
        store = make_daily_store(t)
        Fp = build_feature_panel(store, C.CUTOFFS)
        lab = build_panel(purchase_days(t), C.CUTOFFS)[[ID, "cutoff", "y"]]
        m = lab.merge(Fp, on=[ID, "cutoff"], how="outer", validate="one_to_one", indicator=True)
        R[f"{v}_feature_label_pairs_mismatched"] = int((m["_merge"] != "both").sum())
        tables[v] = (m.drop(columns="_merge"), store, t)
    raw, store, _ = tables["raw"]
    assert R["raw_feature_label_pairs_mismatched"] == 0 and R["nogift_feature_label_pairs_mismatched"] == 0

    # 2. leakage on real data: features from data truncated at c == features from all data
    leak = {}
    for c in C.CUTOFFS:
        trunc = features_at(make_daily_store(tx[tx[DATE] <= c]), c)
        leak[ds(c)] = bool(trunc.equals(features_at(store, c)))
    R["leakage_truncation_check"] = {"per_cutoff": leak, "all_pass": all(leak.values())}

    # 3. prev_90d_active at c_k must equal the label y at c_(k-1)
    cons = {}
    for c in C.CUTOFFS:
        cp = previous_quarter_end(c)
        prev = labels_at(purchase_days(tx), cp).set_index(ID)["y"]
        f = raw.loc[raw["cutoff"] == c].set_index(ID)["prev_90d_active"]
        both = f.dropna()
        cons[ds(c)] = {"previous_cutoff": ds(cp), "n_with_prior": int(len(both)),
                       "n_without_prior": int(f.isna().sum()),
                       "ids_match": bool(set(both.index) == set(prev.index) & set(f.index)),
                       "values_match": bool((both == prev.reindex(both.index)).all())}
    R["prev_90d_active_vs_previous_label"] = {
        "per_cutoff": cons, "all_pass": all(x["ids_match"] and x["values_match"] for x in cons.values())}
    # alternative definition considered in review: any purchase in the full calendar quarter
    # (c_prev, c]. Uses history <= c only; counts rows where the two definitions disagree.
    alt = {}
    for c in C.CUTOFFS:
        cp = previous_quarter_end(c)
        f = raw.loc[raw["cutoff"] == c].set_index(ID)["prev_90d_active"].dropna()
        h = store.days[(store.days[DATE] > cp) & (store.days[DATE] <= c)]
        full_q = pd.Series(f.index.isin(h[ID]), index=f.index).astype(float)
        alt[ds(c)] = {"window_days_not_covered": int((c - (cp + pd.Timedelta(days=90))).days),
                      "rows_with_prior": int(len(f)), "rows_differ": int((full_q != f).sum()),
                      "differ_only_0_to_1": bool(((full_q == 1) | (f == 0))[full_q != f].all())}
    R["prev_quarter_definition_comparison"] = {
        "per_cutoff": alt, "rows_differ_total": sum(v["rows_differ"] for v in alt.values()),
        "rows_with_prior_total": sum(v["rows_with_prior"] for v in alt.values())}

    # 4. split design
    hold = S.holdout_respondents(raw[ID])
    raw = S.assign(raw, hold)
    S.check_design(raw)
    split_rows = []
    for r in ROLES:
        g = raw[raw[r]]
        split_rows.append({"role": r, "cutoffs": ", ".join(sorted({ds(c) for c in g["cutoff"]})),
                           "obs": len(g), "respondents": g[ID].nunique(),
                           "y0": int((g["y"] == 0).sum()), "pct_y0": 100 * (g["y"] == 0).mean(),
                           "prev_90d_active_missing": int(g["prev_90d_active"].isna().sum())})
    sp = pd.DataFrame(split_rows)
    sp.to_csv(OUT / "stage_c_split_summary.csv", index=False)
    ov = pd.DataFrame({a: {b: len(set(raw.loc[raw[a], ID]) & set(raw.loc[raw[b], ID])) for b in ROLES}
                       for a in ROLES})
    ov.to_csv(OUT / "stage_c_respondent_overlap.csv")
    per = raw.groupby(ID).agg(holdout=("holdout", "first"), k=("cutoff", "size"),
                              first_cut=("cutoff", "min"))
    R["holdout"] = {
        "seed": S.HOLDOUT_SEED, "fraction": S.HOLDOUT_FRACTION, "respondents": int(len(hold)),
        "of_total": int(raw[ID].nunique()),
        "share_at_all_15_cutoffs": {str(k): float(v) for k, v in
                                    per.groupby("holdout")["k"].apply(lambda s: (s == 15).mean()).items()},
        "mean_cutoffs_per_respondent": {str(k): float(v) for k, v in per.groupby("holdout")["k"].mean().items()},
        "pct_y0_training_period": {str(k): float(100 * (g["y"] == 0).mean()) for k, g in
                                   raw[raw["cutoff"].isin(S.TRAIN_CUTOFFS)].groupby("holdout")},
        "embargo_train_to_tune": S.check_embargo(S.TRAIN_CUTOFFS, S.TUNE_CUTOFFS),
        "embargo_refit_to_test": S.check_embargo(S.REFIT_CUTOFFS, S.TEST_CUTOFFS),
        "test_temporal_respondents_not_in_refit": int(len(
            set(raw.loc[raw["in_test_temporal"], ID]) - set(raw.loc[raw["in_refit"], ID]))),
    }

    # 5. feature descriptives: missingness at every cutoff (no labels); distributions and
    #    correlations on non-hold-out training rows only
    miss = raw.groupby("cutoff")[FEATURE_COLUMNS].apply(lambda g: g.isna().sum())
    miss = miss.loc[:, miss.any()]
    miss.index = [ds(c) for c in miss.index]
    miss.to_csv(OUT / "stage_c_missingness.csv")
    tr = raw[raw["in_train"]]
    desc = tr[FEATURE_COLUMNS].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc.to_csv(OUT / "stage_c_feature_distribution_train.csv")
    rho = tr[[c for c in FEATURE_COLUMNS if c != "cutoff_quarter"]].corr(method="spearman")
    rho.round(4).to_csv(OUT / "stage_c_feature_spearman_train.csv")
    hi = [(a, b, float(rho.loc[a, b])) for i, a in enumerate(rho) for b in rho.columns[i + 1:]
          if abs(rho.loc[a, b]) >= 0.9]
    R["high_correlations_train_abs_ge_0.9"] = hi

    # 6. baseline smoke test, training period only: fit on non-hold-out train cutoffs except the
    #    last, evaluate on the last train cutoff (2021-03-31). Not a result; checks orientation.
    fit = raw[raw["in_train"] & (raw["cutoff"] < S.TRAIN_CUTOFFS[-1])]
    ev = raw[raw["in_train"] & (raw["cutoff"] == S.TRAIN_CUTOFFS[-1])]
    p_train = float((fit["y"] == 0).mean())
    smoke = {}
    for name, b in (("prevalence", PrevalenceBaseline()), ("persistence", PersistenceBaseline()),
                    ("recency", RecencyRule())):
        b.fit(fit, fit["y"].to_numpy())
        p0 = b.predict_p0(ev)
        s0 = b.score0(ev) if hasattr(b, "score0") else None
        smoke[name] = metrics(ev["y"].to_numpy(), p0, s0, train_prevalence0=p_train)
    R["baseline_smoke_training_period"] = {"fit_cutoffs": f"{ds(S.TRAIN_CUTOFFS[0])} .. {ds(S.TRAIN_CUTOFFS[-2])}",
                                           "eval_cutoff": ds(S.TRAIN_CUTOFFS[-1]), "metrics": smoke}

    # 7. independent feature recomputation (standard library) on a seeded sample
    ind_path = OUT / "independent_features_sample.csv"
    subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("independent_feature_check.py")),
                    str(C.PURCHASES_CSV), str(ind_path), "300", "7"], check=True, capture_output=True)
    ind = pd.read_csv(ind_path, dtype={ID: str})
    ind["cutoff"] = pd.to_datetime(ind["cutoff"])
    cols = [c for c in ind.columns if c not in (ID, "cutoff")]
    mg = ind.merge(raw[[ID, "cutoff"] + cols], on=[ID, "cutoff"], how="left", suffixes=("_ind", ""),
                   indicator=True)
    res = {"rows_compared": int(len(mg)), "respondents": int(ind[ID].nunique()),
           "rows_missing_in_pipeline": int((mg["_merge"] != "both").sum()),
           "pipeline_rows_for_sample_missing_in_independent": int(len(
               raw[raw[ID].isin(set(ind[ID]))]) - len(ind))}
    worst = {}
    for c in cols:
        a, b = mg[c + "_ind"].astype(float), mg[c].astype(float)
        nan_mismatch = int((a.isna() != b.isna()).sum())
        diff = float(np.nanmax(np.abs(a - b))) if (a.notna() & b.notna()).any() else 0.0
        worst[c] = {"nan_mismatch": nan_mismatch, "max_abs_diff": diff}
    res["per_feature"] = worst
    res["all_match"] = bool(res["rows_missing_in_pipeline"] == 0
                            and res["pipeline_rows_for_sample_missing_in_independent"] == 0
                            and all(v["nan_mismatch"] == 0 and v["max_abs_diff"] <= 1e-6 for v in worst.values()))
    R["independent_feature_check"] = res

    # 8. outputs
    pd.DataFrame([{"feature": k, "definition": v[0], "lookback": v[1], "missing": v[2], "leakage_note": v[3]}
                  for k, v in FEATURES.items()]).to_csv(OUT / "stage_c_feature_definitions.csv", index=False)
    for v, (tab, _, _) in tables.items():
        tab = tab if v != "raw" else raw
        if v != "raw":
            tab = S.assign(tab, hold)
        out = tab.assign(cutoff=tab["cutoff"].dt.strftime("%Y-%m-%d"))
        out.to_csv(OUT / f"model_table_{v}.csv.gz", index=False, compression="gzip")
    R["model_table_rows"] = {v: int(len(t[0])) for v, t in tables.items()}
    R["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "stage_c_summary.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({k: R[k] for k in ("leakage_truncation_check", "prev_90d_active_vs_previous_label",
                                        "holdout", "high_correlations_train_abs_ge_0.9", "runtime_seconds")},
                     indent=1, default=str))
    print(sp.to_string(index=False))
    print(json.dumps(R["independent_feature_check"], indent=1))
    print(json.dumps(R["baseline_smoke_training_period"], indent=1))


if __name__ == "__main__":
    main()
