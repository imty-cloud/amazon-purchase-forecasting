# POST-HOC, không dùng để đổi kết luận P1–P3
"""C: baseline còn thiếu, so với B2, M1, M2, M3 khóa trong cùng một bảng (Tune; temporal_test; holdout_test).
Tune: huấn luyện Train. Hai tập kiểm định: huấn luyện Refit. Chốt trước khi chạy:
 C1a  quy tắc theo recency_to_median_gap; C1b  quy tắc theo median_gap_days (đẳng hướng, NaN có ô riêng, xem GapRule).
 C2   hồi quy logistic chỉ gồm spline bậc 3 của log1p(recency_days), 6 nút theo phân vị của dữ liệu huấn luyện,
      ngoại suy tuyến tính; C chọn theo AP trên Tune (Train -> Tune) trong lưới 0.001..1000, hòa thì chọn C nhỏ hơn.
 C3   BG/NBD, Pareto/NBD trên (x, t_x, T) tính từ lịch sử <= mốc cắt; đơn vị tuần; tau = 90 ngày. Không dùng nhãn.
      Tham số ước lượng trên người ngoài hold-out. Hai biến thể:
        _roll  ước lượng lại tại đúng từng mốc đánh giá, từ lịch sử đến mốc đó (đúng nghĩa yêu cầu);
        _fixed ước lượng một lần tại mốc huấn luyện cuối (2021-03-31 cho Tune, 2021-12-31 cho kiểm định)."""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer

from post_i_common import *
import post_i_clv as clv

TAU = 90 / 7
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
table = get_table()
tr, tu, rf, tt, ho = frames(table)


def xtT(F):
    T = F["tenure_days"].to_numpy(float)
    tx = T - F["recency_days"].to_numpy(float)
    assert (tx >= 0).all()
    return F["frequency_days"].to_numpy(float) - 1, tx / 7, T / 7


PARAMS = []


def clv_params(kind, c):
    """Tham số tại mốc c, từ người ngoài hold-out; chỉ đọc đặc trưng <= c, không đọc nhãn."""
    def run():
        F = table[(table["cutoff"] == c) & ~table["holdout"]]
        x, tx, T = xtT(F)
        t0 = time.time()
        p, ll = clv.fit(kind, x, tx, T)
        A = table[table["cutoff"] == c]        # cả hold-out, tại cả T và T + tau
        xa, txa, Ta = xtT(A)
        bad = int((~np.isfinite(clv.LL[kind](p, xa, txa, Ta)) | ~np.isfinite(clv.LL[kind](p, xa, txa, Ta + TAU))).sum())
        return dict(kind=kind, cutoff=str(pd.Timestamp(c).date()), n=len(F), p1=p[0], p2=p[1], p3=p[2], p4=p[3],
                    mean_loglik=ll, n_nonfinite_ll=bad, fit_seconds=time.time() - t0)
    r = ckpt(f"c_clv_{kind}_{pd.Timestamp(c).date()}", run)
    if r not in PARAMS:
        PARAMS.append(r)
        log(r)
    return np.array([r["p1"], r["p2"], r["p3"], r["p4"]])


class CLV:
    def __init__(self, kind, mode):
        self.kind, self.mode = kind, mode
        self.p_fit = None

    def fit(self, F, y):                     # y không được dùng: mô hình này không học từ nhãn
        self.c_fit = F["cutoff"].max()
        return self

    def p0(self, S):
        out = np.full(len(S), np.nan)
        cuts = S["cutoff"].to_numpy()
        for c in np.unique(cuts):
            k = cuts == c
            par = clv_params(self.kind, c if self.mode == "roll" else self.c_fit)
            out[k] = clv.p_no_purchase(self.kind, par, *xtT(S[k]), TAU)
        assert np.isfinite(out).all()
        return out

    def score0(self, S):
        return self.p0(S)


def spline_pipe(C):
    return Pipeline([("log1p", FunctionTransformer(np.log1p)),
                     ("spline", SplineTransformer(n_knots=6, degree=3, knots="quantile", extrapolation="linear")),
                     ("lr", LogisticRegression(C=C, max_iter=5000))])


# ---- chọn C chỉ bằng Train -> Tune ----
sel = []
for C in C_GRID:
    m = spline_pipe(C).fit(tr[["recency_days"]], tr["y"].to_numpy())
    sel.append(dict(C=C, tune_ap_y0=average_precision_score(1 - tu["y"], p_no_purchase(m, tu[["recency_days"]]))))
sel = pd.DataFrame(sel)
best_C = float(sel.sort_values(["tune_ap_y0", "C"], ascending=[False, True]).iloc[0]["C"])
sel["selected"] = sel["C"] == best_C
write_csv(sel, "post_i_c_spline_selection.csv")
print(sel.round(5).to_string())
log("spline C chốt =", best_C)

builders = {
    "B2_recency": lambda: locked("B2_recency"),
    "M1_lr_rfm": lambda: locked("M1_lr_rfm"),
    "M2_lr_full": lambda: locked("M2_lr_full"),
    "M3_lgbm_full": lambda: locked("M3_lgbm_full"),
    "C1a_rule_rec_to_gap": lambda: GapRule("recency_to_median_gap"),
    "C1b_rule_median_gap": lambda: GapRule("median_gap_days"),
    "C2_spline_recency": lambda: SklearnLR(lambda: spline_pipe(best_C), cols=["recency_days"]),
    "C3a_bgnbd_roll": lambda: CLV("bgnbd", "roll"),
    "C3b_pareto_roll": lambda: CLV("pareto", "roll"),
    "C3a_bgnbd_fixed": lambda: CLV("bgnbd", "fixed"),
    "C3b_pareto_fixed": lambda: CLV("pareto", "fixed"),
}
keep = {}
res = evaluate_models(table, builders, keep=keep)
write_csv(res, "post_i_c_baselines.csv")
write_csv(pd.DataFrame(PARAMS).sort_values(["kind", "cutoff"]), "post_i_c_clv_params.csv")

# ---- kiểm tra: bốn mô hình khóa dựng lại phải trùng bảng khóa trên mọi chỉ số chung ----
L = pd.read_csv(OUTPUT_DIR / "stage_d_test_metrics.csv")
L = L[L["scope"] == "pooled"]
cols = ["ap_y0", "roc_auc", "brier", "cal_intercept", "cal_slope", "precision_top10", "recall_top10",
        "precision_top20", "recall_top20"]
for _, r in res[res["model"].isin(["B2_recency", "M1_lr_rfm", "M2_lr_full", "M3_lgbm_full"]) & (res["set"] != "tune")].iterrows():
    q = L[(L["set"] == r["set"]) & (L["model"] == r["model"])].iloc[0]
    for k in cols:
        assert round(r[k], 4) == round(q[k], 4), (r["set"], r["model"], k, r[k], q[k])
log("bốn mô hình khóa khớp bảng khóa đến 4 chữ số trên", len(cols), "chỉ số")

# ---- khoảng bootstrap cụm ghép cặp cho hiệu số AP/Brier so với B2 và M2 (mô tả, không hiệu chỉnh bội) ----
new = [m for m in builders if m[:2] in ("C1", "C2", "C3")]
dfs = []
for sname, S in (("temporal_test", tt), ("holdout_test", ho)):
    P = {m: keep[(sname, m)] for m in builders}
    pairs = [(m, "B2_recency") for m in new] + [(m, "M2_lr_full") for m in new]
    dfs.append(paired_diff(S, P, pairs, boot_fns()).assign(set=sname))
    log("bootstrap xong", sname)
diff = pd.concat(dfs, ignore_index=True)
write_csv(diff, "post_i_c_paired_diff.csv")

show = ["set", "model", "n", "prevalence_y0", "ap_y0", "roc_auc", "brier", "cal_intercept", "cal_slope", "mean_pred_p0",
        "precision_top10", "recall_top10", "precision_top20", "recall_top20"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
print(res[show].round(4).to_string())
print(pd.DataFrame(PARAMS).round(4).to_string())
print(diff[diff["metric"] == "ap_y0"].round(4).to_string())
log("XONG C")
