"""Step 0: rebuild features + Refit models under the locked config; compare pooled AP/ROC-AUC/Brier
with outputs/stage_d_test_metrics.csv to 4 decimals. POST-HOC / exploratory."""
import sys
from post_common import *
from apf.evaluation import metrics

table = get_table()
models = fit_final(table, ["B0_prevalence", "B1_persistence", "B2_recency", "M1_lr_rfm", "M2_lr_full",
                           "M3_lgbm_full", "S1_lr_no_prev", "S2_lgbm_no_prev"])
locked = pd.read_csv(OUTPUT_DIR / "stage_d_test_metrics.csv")
rows = []
for sname in TEST_SETS:
    S = test_set(table, sname)
    for m, mod in models.items():
        r = metrics(S["y"].to_numpy(), mod.p0(S), mod.score0(S), train_prevalence0=mod.p_fit)
        L = locked[(locked.set == sname) & (locked.model == m) & (locked.scope == "pooled")].iloc[0]
        for k in ("ap_y0", "roc_auc", "brier"):
            rows.append(dict(set=sname, model=m, metric=k, rebuilt=r[k], locked=L[k],
                             abs_diff=abs(r[k] - L[k]), match4=round(r[k], 4) == round(L[k], 4)))
df = pd.DataFrame(rows)
write_csv(df, "post_step0_check.csv")
print(df.to_string())
print("ALL MATCH (4 d.p.):", bool(df.match4.all()), "| max abs diff:", df.abs_diff.max())
sys.exit(0 if df.match4.all() else 1)
