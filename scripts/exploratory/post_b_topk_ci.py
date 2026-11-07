"""B: precision/recall at top 10%/20% with respondent-cluster bootstrap CI (Refit models). POST-HOC / exploratory."""
from post_common import *
from apf.evaluation import _clusters, top_fraction_by_group, percentile_ci

B, SEED = SEEDS["bootstrap_draws"], SEEDS["bootstrap"]
MODELS = ["B2_recency", "M1_lr_rfm", "M2_lr_full", "M3_lgbm_full"]
FR = (0.10, 0.20)
table = get_table()
fitted = fit_final(table, MODELS)
locked = pd.read_csv(OUTPUT_DIR / "stage_d_test_metrics.csv")
rows, drows = [], []


def topk(y, s, cut):
    out = {}
    for f in FR:
        t = top_fraction_by_group(y, s, cut, f)
        out[f"precision_top{int(f*100)}"], out[f"recall_top{int(f*100)}"] = t["precision_y0"], t["recall_y0"]
    return out


for sname in TEST_SETS:
    S = test_set(table, sname)
    y, cut, ids = S["y"].to_numpy(), S["cutoff"].to_numpy(), S[ID].to_numpy()
    sc = {m: fitted[m].score0(S) for m in MODELS}
    point = {m: topk(y, sc[m], cut) for m in MODELS}
    for m in MODELS:                                  # consistency with the locked table
        L = locked[(locked.set == sname) & (locked.model == m) & (locked.scope == "pooled")].iloc[0]
        for k, v in point[m].items():
            assert abs(v - L[k]) < 1e-9, (sname, m, k, v, L[k])
    rows_of = _clusters(ids)
    rng = np.random.default_rng(SEED)
    draws = []
    for b in range(B):
        idx = np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), len(rows_of))])
        d = {m: topk(y[idx], sc[m][idx], cut[idx]) for m in MODELS}
        draws.append(d)
    for m in MODELS:
        for k in point[m]:
            v = np.array([d[m][k] for d in draws]); lo, hi = percentile_ci(v)
            rows.append(dict(set=sname, model=m, metric=k, estimate=point[m][k], ci_low=lo, ci_high=hi, B=B, seed=SEED))
    for name, a, b_ in (("M2-B2", "M2_lr_full", "B2_recency"), ("M3-M2", "M3_lgbm_full", "M2_lr_full")):
        for k in ("precision_top20", "recall_top20"):
            v = np.array([d[a][k] - d[b_][k] for d in draws]); lo, hi = percentile_ci(v)
            drows.append(dict(set=sname, comparison=name, metric=k, estimate=point[a][k] - point[b_][k],
                              ci_low=lo, ci_high=hi, ci_excludes_0=bool(lo > 0 or hi < 0), B=B, seed=SEED))
r, d = pd.DataFrame(rows), pd.DataFrame(drows)
write_csv(r, "post_b_topk_ci.csv"); write_csv(d, "post_b_topk_paired_diff.csv")
print(r.round(4).to_string()); print(d.round(4).to_string())
