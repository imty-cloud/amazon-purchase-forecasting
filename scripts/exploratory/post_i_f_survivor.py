# POST-HOC, không dùng để đổi kết luận P1–P3
"""F: giả thuyết cạnh tranh "mẫu chỉ gồm người còn dùng Amazon vào cuối 2022". Chốt trước khi chạy; không chọn gì bằng tập kiểm định.
(1) Hiệu chuẩn theo nhóm, ở Tune và hai tập kiểm định: tam phân vị tenure_days tính trên Train; số mốc có mặt (15 / 10-14 / 1-9).
    Số mốc có mặt tính trên cả 15 mốc nhưng chỉ phụ thuộc ngày mua đầu tiên, không phụ thuộc nhãn; chỉ dùng để chia nhóm chẩn đoán.
(2) Biến thể M2 chuẩn hóa theo thâm niên (C khóa = 0.01, không tinh chỉnh lại):
    frequency_days -> frequency_days/(tenure_days+1); monetary_total -> monetary_total/(tenure_days+1);
    categories_total -> categories_total/frequency_days. Hai dạng: giữ hoặc bỏ tenure_days.
(3) Cửa sổ trượt 4 và 6 quý gần nhất cho M2 và M3 (siêu tham số khóa, 114 cây): k mốc cuối của Train -> Tune; k mốc cuối của Refit -> hai tập kiểm định.
Hiệu số so với mô hình khóa: bootstrap cụm ghép cặp B = 1000 (mô tả, không hiệu chỉnh bội)."""
from post_i_common import *
from apf.evaluation import calibration_intercept_slope

table = get_table().copy()
table["freq_per_day"] = table["frequency_days"] / (table["tenure_days"] + 1)
table["spend_per_day"] = table["monetary_total"] / (table["tenure_days"] + 1)
table["categories_per_purchase_day"] = table["categories_total"] / table["frequency_days"]
RATE = {"frequency_days": "freq_per_day", "monetary_total": "spend_per_day", "categories_total": "categories_per_purchase_day"}
FULL = list(Mo.FULL_FEATURES)
KEEP = [RATE.get(f, f) for f in FULL]
DROP = [f for f in KEEP if f != "tenure_days"]
assert len(KEEP) == 19 and len(DROP) == 18
C2 = CONF["M2_lr_full"]["C"]
cf3 = CONF["M3_lgbm_full"]


class Window:
    """Chỉ giữ k mốc cắt gần nhất của tập huấn luyện được đưa vào."""

    def __init__(self, inner, k):
        self.inner, self.k = inner, k

    def fit(self, F, y):
        cuts = sorted(F["cutoff"].unique())[-self.k:]
        assert len(cuts) == self.k
        G = F[F["cutoff"].isin(cuts)]
        self.m = self.inner().fit(G, G["y"].to_numpy())
        self.p_fit = self.m.p_fit
        return self

    def p0(self, S):
        return self.m.p0(S)

    def score0(self, S):
        return self.m.score0(S)


builders = {
    "M2_lr_full": lambda: locked("M2_lr_full"),
    "M3_lgbm_full": lambda: locked("M3_lgbm_full"),
    "M2_tenure_norm_keep": lambda: SklearnLR(lambda: Mo.make_logreg(KEEP, C2)),
    "M2_tenure_norm_drop": lambda: SklearnLR(lambda: Mo.make_logreg(DROP, C2)),
}
for k in (4, 6):
    builders[f"M2_window{k}q"] = (lambda k=k: Window(lambda: locked("M2_lr_full"), k))
    builders[f"M3_window{k}q"] = (lambda k=k: Window(lambda: locked("M3_lgbm_full"), k))

keep = {}
res = evaluate_models(table, builders, keep=keep)
write_csv(res, "post_i_f_variants.csv")

tr, tu, rf, tt, ho = frames(table)
SETS = {"tune": tu, "temporal_test": tt, "holdout_test": ho}
q1, q2 = tr["tenure_days"].quantile([1 / 3, 2 / 3]).to_numpy()
log("ngưỡng tam phân vị tenure_days trên Train:", q1, q2)
present = table.groupby(ID)["cutoff"].transform("size")


def groups(S):
    pr = present.loc[S.index]
    return {"tenure": np.select([S["tenure_days"] <= q1, S["tenure_days"] <= q2], [f"T1 (<={q1:.0f}d)", f"T2 (<={q2:.0f}d)"], f"T3 (>{q2:.0f}d)"),
            "n_cutoffs_present": np.select([pr == 15, pr >= 10], ["15", "10-14"], "1-9")}


rows = []
for (sname, m), (p0, s0) in keep.items():
    S = SETS[sname]
    y = S["y"].to_numpy()
    for by, lab in groups(S).items():
        for g in sorted(set(lab)):
            k = lab == g
            ev = int((y[k] == 0).sum())
            r = dict(set=sname, model=m, by=by, group=g, n=int(k.sum()), n_event_y0=ev, prevalence_y0=float((y[k] == 0).mean()),
                     mean_pred_p0=float(p0[k].mean()))
            if 0 < ev < k.sum():
                r.update(metrics(y[k], p0[k], s0[k]))
                r = {x: r[x] for x in ("set", "model", "by", "group", "n", "n_event_y0", "prevalence_y0", "mean_pred_p0",
                                       "ap_y0", "roc_auc", "brier", "cal_intercept", "cal_slope")} | {"mean_pred_p0": float(p0[k].mean())}
            rows.append(r)
grp = pd.DataFrame(rows)
write_csv(grp, "post_i_f_calibration_by_group.csv")

fns = dict(boot_fns(), cal_intercept=intercept_fn)
fns = {k: fns[k] for k in ("ap_y0", "brier", "cal_intercept")}
dfs = []
for sname, S in (("temporal_test", tt), ("holdout_test", ho), ("tune", tu)):
    P = {m: keep[(sname, m)] for m in builders}
    pairs = [("M2_tenure_norm_keep", "M2_lr_full"), ("M2_tenure_norm_drop", "M2_lr_full"), ("M2_window4q", "M2_lr_full"),
             ("M2_window6q", "M2_lr_full"), ("M3_window4q", "M3_lgbm_full"), ("M3_window6q", "M3_lgbm_full")]
    dfs.append(paired_diff(S, P, pairs, fns).assign(set=sname))
    log("bootstrap xong", sname)
diff = pd.concat(dfs, ignore_index=True)
write_csv(diff, "post_i_f_paired_diff.csv")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
print(res[["set", "model", "n", "prevalence_y0", "ap_y0", "roc_auc", "brier", "cal_intercept", "cal_slope", "mean_pred_p0"]].round(4).to_string())
print(grp[grp["model"].isin(["M2_lr_full", "M3_lgbm_full"])].round(4).to_string())
print(grp.pivot_table(index=["model", "by", "group"], columns="set", values="cal_intercept")[["tune", "temporal_test", "holdout_test"]].round(3).to_string())
print(diff.round(4).to_string())
log("XONG F")
