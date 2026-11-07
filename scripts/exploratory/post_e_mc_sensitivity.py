"""E (optional): Monte Carlo sensitivity of P1-P3: B=10000, different seed, Bonferroni 98.33%. POST-HOC / exploratory.
Point estimates are unchanged by construction (Refit models, same data); only the bootstrap draws differ."""
from post_common import *
from apf.evaluation import paired_cluster_bootstrap, percentile_ci
from apf.phase2 import _boot_fns, BONF_LO, BONF_HI

B, SEED = 10000, 20270101
assert SEED != SEEDS["bootstrap"]
P = [("P1", "M2_lr_full", "M1_lr_rfm"), ("P2", "M3_lgbm_full", "M2_lr_full"), ("P3", "M2_lr_full", "B2_recency")]
MODELS = ["B2_recency", "M1_lr_rfm", "M2_lr_full", "M3_lgbm_full"]
table = get_table(); fitted = fit_final(table, MODELS)
locked = pd.read_csv(OUTPUT_DIR / "stage_d_comparisons.csv")
rows = []
for sname in TEST_SETS:
    S = test_set(table, sname)
    preds = {m: (fitted[m].p0(S), fitted[m].score0(S)) for m in MODELS}
    d = paired_cluster_bootstrap(S["y"].to_numpy(), preds, S[ID].to_numpy(), _boot_fns(), B=B, seed=SEED)
    w = d.pivot_table(index=["draw", "metric"], columns="model", values="value")
    for cid, a, b in P:
        for metric in ("ap_y0", "roc_auc", "brier"):
            diff = (w[a] - w[b]).xs(metric, level="metric").to_numpy()
            lo, hi = percentile_ci(diff); blo, bhi = percentile_ci(diff, BONF_LO, BONF_HI)
            L = locked[(locked.set == sname) & (locked.comparison == cid) & (locked.metric == metric)].iloc[0]
            rows.append(dict(set=sname, comparison=cid, metric=metric, estimate_locked=L.estimate,
                             bonf_low_B1000=L.bonf_ci_low, bonf_high_B1000=L.bonf_ci_high,
                             bonf_low_B10000=blo, bonf_high_B10000=bhi,
                             ci95_low_B10000=lo, ci95_high_B10000=hi,
                             excludes0_B1000=bool(L.bonf_ci_excludes_0), excludes0_B10000=bool(blo > 0 or bhi < 0),
                             lower_sign_changed=bool(np.sign(L.bonf_ci_low) != np.sign(blo)),
                             upper_sign_changed=bool(np.sign(L.bonf_ci_high) != np.sign(bhi)),
                             conclusion_changed=bool(L.bonf_ci_excludes_0) != bool(blo > 0 or bhi < 0), B=B, seed=SEED))
    print(sname, "done", flush=True)
df = pd.DataFrame(rows); write_csv(df, "post_e_mc_sensitivity.csv"); print(df.round(4).to_string())
