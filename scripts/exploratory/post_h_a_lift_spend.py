"""POST-HOC, không dùng để đổi kết luận P1–P3.
A: lift/gain tại top 10% và 20% và recall của y = 0 có trọng số chi tiêu, trên hai tập kiểm định.

Mô hình: Refit đã khóa (B2, M1, M2, M3). Điểm xếp hạng = score0 như trong giao thức (§6): xếp hạng trong từng mốc cắt,
hòa điểm chia theo kỳ vọng. Trọng số = spend_365 (chi tiêu 365 ngày trước mốc cắt, biết tại thời điểm cắt).
Với dòng y = 0 thì chi tiêu trong cửa sổ nhãn bằng 0 theo định nghĩa, nên không thể dùng chi tiêu tương lai làm trọng số;
spend_365 là thước đo giá trị khách hàng có thể bị mất.
Bootstrap cụm theo người tham gia, B = 1000, hạt giống 20261024; so sánh có ghép cặp."""
from post_common import *
from apf.evaluation import _clusters, top_fraction_by_group

B, SEED = 1000, 20261024
MODELS = ["B2_recency", "M1_lr_rfm", "M2_lr_full", "M3_lgbm_full"]
FRACS = (0.10, 0.20)

table = get_table()
fitted = fit_final(table, MODELS)


def incl(e, s, frac):
    """Xác suất mỗi dòng nằm trong top `frac` (hòa điểm chia theo kỳ vọng); cùng quy tắc với evaluation.top_fraction."""
    k = int(np.ceil(frac * len(e)))
    t = np.sort(s)[::-1][k - 1]
    above, at = s > t, s == t
    w = np.zeros(len(e))
    w[above] = 1.0
    w[at] = (k - above.sum()) / at.sum()
    return w, k


def stats(y, s, g, spend, frac):
    """Gộp qua các mốc cắt: precision, recall, lift, recall có trọng số chi tiêu."""
    e = (np.asarray(y) == 0).astype(float)
    tp = ktot = ev = n = wtp = wev = 0.0
    for v in pd.unique(g):
        m = g == v
        w, k = incl(e[m], s[m], frac)
        tp += (w * e[m]).sum(); ktot += k; ev += e[m].sum(); n += m.sum()
        wtp += (w * e[m] * spend[m]).sum(); wev += (e[m] * spend[m]).sum()
    prec, rec = tp / ktot, tp / max(ev, 1)
    return dict(precision=prec, recall=rec, lift=prec / (ev / n), spend_recall=wtp / max(wev, 1e-12))


res, draws = [], {}
for set_name in TEST_SETS:
    S = test_set(table, set_name)
    y, g, rid = S["y"].to_numpy(), S["cutoff"].to_numpy(), S["Survey ResponseID"].to_numpy()
    spend = S["spend_365"].fillna(0).to_numpy().astype(float)
    sc = {m: fitted[m].score0(S) for m in MODELS}
    # kiểm tra tái lập: precision/recall khớp bảng khóa
    locked = pd.read_csv(OUT.parent / "stage_d_test_metrics.csv", comment="#")
    for m in MODELS:
        for f in FRACS:
            mine = stats(y, sc[m], g, spend, f)
            row = locked[(locked.set == set_name) & (locked.model == m) & (locked.scope == "pooled")].iloc[0]
            assert abs(mine["precision"] - row[f"precision_top{int(f*100)}"]) < 1e-9, (set_name, m, f)
            assert abs(mine["recall"] - row[f"recall_top{int(f*100)}"]) < 1e-9, (set_name, m, f)
    rows_of = _clusters(rid)
    rng = np.random.default_rng(SEED)
    D = {(m, f): [] for m in MODELS for f in FRACS}
    for b in range(B):
        idx = np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), len(rows_of))])
        for m in MODELS:
            for f in FRACS:
                D[(m, f)].append(stats(y[idx], sc[m][idx], g[idx], spend[idx], f))
    for m in MODELS:
        for f in FRACS:
            pt = stats(y, sc[m], g, spend, f)
            arr = pd.DataFrame(D[(m, f)])
            for met in ("precision", "recall", "lift", "spend_recall"):
                lo, hi = np.percentile(arr[met], [2.5, 97.5])
                res.append(dict(set=set_name, model=m, top=int(f * 100), metric=met, estimate=pt[met], ci_low=lo, ci_high=hi,
                                n=len(y), events_y0=int((y == 0).sum())))
    # so sánh ghép cặp (khoảng Bonferroni 98,33% như các so sánh P1–P3: 3 so sánh)
    for a, b_ in (("M2_lr_full", "B2_recency"), ("M3_lgbm_full", "M2_lr_full"), ("M2_lr_full", "M1_lr_rfm")):
        for f in FRACS:
            for met in ("recall", "spend_recall"):
                d = np.array([x[met] for x in D[(a, f)]]) - np.array([x[met] for x in D[(b_, f)]])
                pt = stats(y, sc[a], g, spend, f)[met] - stats(y, sc[b_], g, spend, f)[met]
                lo, hi = np.percentile(d, [0.833, 99.167])
                res.append(dict(set=set_name, model=f"{a} - {b_}", top=int(f * 100), metric=met + "_diff", estimate=pt,
                                ci_low=lo, ci_high=hi, n=len(y), events_y0=int((y == 0).sum())))

df = pd.DataFrame(res)
write_csv(df, "post_h_a_lift_spend.csv")
pd.set_option("display.width", 200)
print(df[df.metric.isin(["lift", "recall", "spend_recall"])].round(3).to_string())
print(df[df.metric.str.endswith("_diff")].round(4).to_string())
