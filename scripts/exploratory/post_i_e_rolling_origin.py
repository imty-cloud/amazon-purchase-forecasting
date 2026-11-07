# POST-HOC, không dùng để đổi kết luận P1–P3
"""E (THĂM DÒ): rolling-origin với siêu tham số khóa. Mốc đánh giá: 8 mốc cuối trước tập kiểm định có nhãn đầy đủ
(2020-03-31 .. 2021-12-31, đều thuộc Train/Tune/Refit). Cửa sổ huấn luyện mở rộng: mọi mốc trước mốc đánh giá.
Chỉ người ngoài hold-out. M2: C khóa. M3: num_leaves, min_child_samples khóa, 114 cây cố định, không dừng sớm.
B2 chỉ để tham chiếu. Không đọc dòng nào của tập kiểm định."""
from post_i_common import *
from apf.splits import check_embargo

LABEL = "thăm dò"
EVAL = list(CUTOFFS[5:13])
assert len(EVAL) == 8 and not set(EVAL) & set(TEST_CUTOFFS)
table = get_table()
dev = table[~table["holdout"] & table["cutoff"].isin(CUTOFFS[:13])]
assert not dev["cutoff"].isin(TEST_CUTOFFS).any()


def one(c, mid):
    def run():
        fit_cuts = [x for x in CUTOFFS if x < c]
        assert check_embargo(fit_cuts, [c])
        F, S = dev[dev["cutoff"].isin(fit_cuts)], dev[dev["cutoff"] == c]
        m = locked(mid).fit(F, F["y"].to_numpy())
        return score_row(S, m.p0(S), m.score0(S), m.p_fit, label=LABEL, cutoff=str(c.date()), model=mid,
                         n_fit_cutoffs=len(fit_cuts), n_fit=len(F))
    return ckpt(f"e_{mid}_{c.date()}", run)


rows = []
for c in EVAL:
    for mid in ("B2_recency", "M2_lr_full", "M3_lgbm_full"):
        rows.append(one(c, mid))
    log("xong mốc", c.date())
df = pd.DataFrame(rows)
write_csv(df, "post_i_e_rolling_origin.csv")
agg = (df.groupby("model")[["ap_y0", "roc_auc", "brier", "cal_intercept", "cal_slope"]].agg(["mean", "std", "min", "max"]))
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index().assign(label=LABEL, n_origins=len(EVAL))
write_csv(agg, "post_i_e_rolling_origin_agg.csv")
w = df.pivot(index="cutoff", columns="model", values="ap_y0")
w["M3_minus_M2"] = w["M3_lgbm_full"] - w["M2_lr_full"]
w["M2_minus_B2"] = w["M2_lr_full"] - w["B2_recency"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(df[["label", "cutoff", "model", "n_fit_cutoffs", "n_fit", "n", "prevalence_y0", "ap_y0", "brier", "cal_intercept", "cal_slope", "mean_pred_p0"]].round(4).to_string())
print(agg.round(4).T.to_string())
print(w.round(4).to_string())
print("số mốc M3 > M2:", int((w["M3_minus_M2"] > 0).sum()), "/", len(w), "; M2 > B2:", int((w["M2_minus_B2"] > 0).sum()), "/", len(w))
log("XONG E")
