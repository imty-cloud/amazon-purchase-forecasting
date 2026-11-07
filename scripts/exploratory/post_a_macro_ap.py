"""A: unweighted mean of the two test-cutoff APs vs pooled AP. POST-HOC / exploratory. No training."""
from post_common import *
st = pd.read_csv(OUTPUT_DIR / "stage_d_temporal_stability.csv")
met = pd.read_csv(OUTPUT_DIR / "stage_d_test_metrics.csv")
rows = []
for s in TEST_SETS:
    for m in ["B2_recency", "M1_lr_rfm", "M2_lr_full", "M3_lgbm_full"]:
        a = st[(st.set == s) & (st.model == m)].sort_values("cutoff")
        assert len(a) == 2
        pooled = met[(met.set == s) & (met.model == m) & (met.scope == "pooled")].ap_y0.item()
        rows.append(dict(set=s, model=m, ap_2022_03_31=a.ap_y0.iloc[0], ap_2022_06_30=a.ap_y0.iloc[1],
                         macro_ap=a.ap_y0.mean(), pooled_ap=pooled, macro_minus_pooled=a.ap_y0.mean() - pooled))
df = pd.DataFrame(rows)
write_csv(df, "post_a_macro_ap.csv")
print(df.round(4).to_string())
