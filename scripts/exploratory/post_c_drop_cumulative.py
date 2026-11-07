"""C: drop cumulative variables; Train -> Tune ONLY (no Refit, no test sets). POST-HOC / exploratory.
LightGBM uses the locked num_leaves, min_child_samples and best_iteration (114, fixed, no early stopping)."""
from post_common import *
from apf import models as Mo
from apf.evaluation import metrics, p_no_purchase

DROP = ["frequency_days", "monetary_total", "categories_total", "tenure_days"]   # fixed before running
table = get_table()
tr, tu = table[table["in_train"]], table[table["in_tune"]]
assert not tr.holdout.any() and not tu.holdout.any()
assert not tu["cutoff"].isin(TEST_CUTOFFS).any() and not tr["cutoff"].isin(TEST_CUTOFFS).any()
full = list(Mo.FULL_FEATURES); assert len(full) == 19 and all(d in full for d in DROP)
reduced = [f for f in full if f not in DROP]
rows = []
for mid, kind in (("M2_lr_full", "lr"), ("M3_lgbm_full", "lgbm")):
    cf = CONF[mid]
    for label, cols in (("full_19", full), (f"drop4_{len(reduced)}", reduced)):
        if kind == "lr":
            m = Mo.make_logreg(cols, cf["C"]).fit(tr, tr["y"].to_numpy()); X = tu
        else:
            m = Mo.make_lgbm_refit({"num_leaves": cf["num_leaves"], "min_child_samples": cf["min_child_samples"]},
                                   cf["best_iteration"]).fit(tr[cols], tr["y"].to_numpy()); X = tu[cols]
        p0 = p_no_purchase(m, X)
        r = metrics(tu["y"].to_numpy(), p0, train_prevalence0=float((tr["y"] == 0).mean()))
        rows.append(dict(model=mid, config=label, n_features=len(cols), n_tune=r["n"], prevalence_y0=r["prevalence_y0"],
                         ap_y0=r["ap_y0"], roc_auc=r["roc_auc"], brier=r["brier"], cal_intercept=r["cal_intercept"],
                         cal_slope=r["cal_slope"], mean_pred_p0=float(np.mean(p0)), observed_rate_y0=r["prevalence_y0"],
                         mean_pred_over_observed=float(np.mean(p0) / r["prevalence_y0"])))
df = pd.DataFrame(rows)
write_csv(df, "post_c_drop_cumulative_tune.csv")
print(df.round(4).to_string())
