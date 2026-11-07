"""POST-HOC, không dùng để đổi kết luận P1–P3.
B: tầm quan trọng THEO NHÓM đặc trưng (không theo từng đặc trưng vì có 14 cặp |rho| >= 0,9).
 (1) bỏ cả nhóm khỏi mô hình đầy đủ, huấn luyện Train -> chấm Tune (không đụng tập kiểm định);
 (2) bỏ cả nhóm / thêm một nhóm vào RFM, huấn luyện Refit -> chấm hai tập kiểm định, kèm khoảng bootstrap cụm theo người
     tham gia cho hiệu số AP (mô tả, không hiệu chỉnh bội);
 (3) permutation importance theo nhóm trên mô hình Refit đầy đủ (xáo trộn cả khối cột của nhóm).
Siêu tham số khóa; LightGBM giữ cố định num_leaves, min_child_samples và 114 cây (không dừng sớm)."""
from post_common import *
from apf import models as Mo
from apf.evaluation import metrics, p_no_purchase, _clusters
from sklearn.metrics import average_precision_score

B, SEED, R = 1000, 20261024, 20
GROUPS = {
    "RFM": ["recency_days", "frequency_days", "monetary_total"],
    "Cường độ chi tiêu": ["spend_per_purchase_day", "spend_90", "spend_365"],
    "Cửa sổ hoạt động": ["days_30", "days_90", "days_180", "days_365", "active_months_12", "days_same_window_last_year"],
    "Độ đa dạng": ["categories_total", "categories_365"],
    "Thời gian": ["tenure_days", "median_gap_days", "recency_to_median_gap"],
    "Kết cục trước": ["prev_90d_active"],
    "Lịch": ["cutoff_quarter"],
}
FULL = list(Mo.FULL_FEATURES)
assert sorted(sum(GROUPS.values(), [])) == sorted(FULL) and len(FULL) == 19
table = get_table()
tr, tu = table[table["in_train"]], table[table["in_tune"]]
refit = table[table["in_refit"]]
for fr in (tr, tu, refit):
    assert not fr["holdout"].any()
assert not tu["cutoff"].isin(TEST_CUTOFFS).any() and not refit["cutoff"].isin(TEST_CUTOFFS).any()
MIDS = (("M2_lr_full", "lr"), ("M3_lgbm_full", "lgbm"))


def fit(kind, cf, cols, F):
    if kind == "lr":
        return Mo.make_logreg(cols, cf["C"]).fit(F, F["y"].to_numpy())
    return Mo.make_lgbm_refit({"num_leaves": cf["num_leaves"], "min_child_samples": cf["min_child_samples"]},
                              cf["best_iteration"]).fit(F[cols], F["y"].to_numpy())


def p0_of(m, kind, X, cols):
    return p_no_purchase(m, X if kind == "lr" else X[cols])


def evalm(y, p0, ref):
    r = metrics(y, p0, train_prevalence0=ref)
    return dict(ap_y0=r["ap_y0"], roc_auc=r["roc_auc"], brier=r["brier"], cal_intercept=r["cal_intercept"], cal_slope=r["cal_slope"])


# ---- (1) Train -> Tune, bỏ nhóm ----
rows1 = []
for mid, kind in MIDS:
    cf = CONF[mid]
    variants = [("full_19", FULL)] + [(f"bỏ {g}", [c for c in FULL if c not in cols]) for g, cols in GROUPS.items()]
    for lab, cols in variants:
        m = fit(kind, cf, cols, tr)
        r = evalm(tu["y"].to_numpy(), p0_of(m, kind, tu, cols), float((tr["y"] == 0).mean()))
        rows1.append(dict(model=mid, variant=lab, n_features=len(cols), **r))
d1 = pd.DataFrame(rows1)
for mid, _ in MIDS:
    base = d1[(d1.model == mid) & (d1.variant == "full_19")].iloc[0]
    for k in ("ap_y0", "roc_auc", "brier"):
        d1.loc[d1.model == mid, f"d_{k}"] = d1.loc[d1.model == mid, k] - base[k]
write_csv(d1, "post_h_b_drop_group_tune.csv")
print("== (1) Train -> Tune"); print(d1.round(4).to_string())

# ---- (2) Refit -> test: bỏ nhóm và thêm nhóm vào RFM ----
rows2, rowsP = [], []
locked = pd.read_csv(OUT.parent / "stage_d_test_metrics.csv", comment="#")
for set_name in TEST_SETS:
    S = test_set(table, set_name)
    y = S["y"].to_numpy(); rid = S["Survey ResponseID"].to_numpy()
    e = (y == 0).astype(int)
    rows_of = _clusters(rid)
    rng = np.random.default_rng(SEED)
    boot_idx = [np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), len(rows_of))]) for _ in range(B)]
    for mid, kind in MIDS:
        cf = CONF[mid]
        preds, fitted = {}, {}
        variants = [("full_19", FULL)]
        variants += [(f"bỏ {g}", [c for c in FULL if c not in cols]) for g, cols in GROUPS.items()]
        variants += [("chỉ RFM", GROUPS["RFM"])]
        variants += [(f"RFM + {g}", GROUPS["RFM"] + cols) for g, cols in GROUPS.items() if g != "RFM"]
        for lab, cols in variants:
            m = fit(kind, cf, cols, refit); fitted[lab] = (m, cols)
            preds[lab] = p0_of(m, kind, S, cols)
        # kiểm tra tái lập
        lk = locked[(locked.set == set_name) & (locked.model == mid) & (locked.scope == "pooled")].iloc[0]
        assert abs(average_precision_score(e, preds["full_19"]) - lk["ap_y0"]) < 1e-9, (set_name, mid)
        APb = {lab: np.array([average_precision_score(e[i], p[i]) for i in boot_idx]) for lab, p in preds.items()}
        ref = float((refit["y"] == 0).mean())
        for lab, p in preds.items():
            r = evalm(y, p, ref)
            comp = "full_19" if (lab == "full_19" or lab.startswith("bỏ")) else "chỉ RFM"
            if lab in ("full_19", "chỉ RFM"):
                comp = None
            dd = APb[lab] - APb[comp] if comp else None
            rows2.append(dict(set=set_name, model=mid, variant=lab, n_features=len(fitted[lab][1]), **r,
                              vs=comp or "", d_ap=(r["ap_y0"] - evalm(y, preds[comp], ref)["ap_y0"]) if comp else np.nan,
                              d_ap_lo=np.percentile(dd, 2.5) if comp else np.nan, d_ap_hi=np.percentile(dd, 97.5) if comp else np.nan))
        # ---- (3) permutation importance theo nhóm (mô hình đầy đủ) ----
        m_full, cols_full = fitted["full_19"]
        base_ap = average_precision_score(e, preds["full_19"])
        base_br = float(np.mean((preds["full_19"] - e) ** 2))
        rp = np.random.default_rng(SEED + 1)
        for g, cols in GROUPS.items():
            dap, dbr = [], []
            for _ in range(R):
                X = S.copy()
                perm = rp.permutation(len(X))
                X[cols] = S[cols].to_numpy()[perm]
                p = p0_of(m_full, kind, X, cols_full)
                dap.append(average_precision_score(e, p) - base_ap); dbr.append(float(np.mean((p - e) ** 2)) - base_br)
            rowsP.append(dict(set=set_name, model=mid, group=g, n_features=len(cols), repeats=R, d_ap_mean=np.mean(dap),
                              d_ap_sd=np.std(dap, ddof=1), d_brier_mean=np.mean(dbr), d_brier_sd=np.std(dbr, ddof=1)))
d2, dP = pd.DataFrame(rows2), pd.DataFrame(rowsP)
write_csv(d2, "post_h_b_group_ablation_test.csv")
write_csv(dP, "post_h_b_group_permutation_test.csv")
pd.set_option("display.width", 220)
print("== (2) Refit -> test"); print(d2.drop(columns=["cal_slope"]).round(4).to_string())
print("== (3) permutation"); print(dP.round(4).to_string())
