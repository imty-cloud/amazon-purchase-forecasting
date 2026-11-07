# POST-HOC, không dùng để đổi kết luận P1–P3
"""D: tinh chỉnh LightGBM kỹ hơn. Chốt trước khi chạy:
Không gian (36 cấu hình ngẫu nhiên, hạt giống 20261024; chọn trong 36 cấu hình này, cấu hình khóa chỉ để so sánh):
  num_leaves log-đều nguyên [5, 63]; min_child_samples log-đều nguyên [20, 500]; learning_rate log-đều [0.01, 0.1];
  reg_lambda log-đều [0.1, 30]; colsample_bytree đều [0.5, 1]; subsample đều [0.5, 1] (subsample_freq = 1);
  monotone: 50 % có ràng buộc đơn điệu trên P(y=1): giảm theo recency_days, recency_to_median_gap; tăng theo frequency_days,
  days_30/90/180/365, active_months_12, prev_90d_active.
CV: rolling-origin 3 lần chia trên các mốc Train và Tune, chỉ người ngoài hold-out: đánh giá tại 2021-03-31, 2021-06-30, 2021-09-30,
  huấn luyện trên mọi mốc trước đó. Số cây: dừng sớm trong từng lần chia (tối đa 3000, kiên nhẫn 200, cùng chỉ số AP y=0 như giao thức).
Chọn: AP trung bình 3 lần chia cao nhất. Số cây cuối = trung vị best_iteration của 3 lần chia. Huấn luyện lại trên Refit; chấm MỘT lần
  trên hai tập kiểm định. Lưu ý: dừng sớm dùng chính mốc đánh giá nên AP CV lạc quan (như nhau giữa các cấu hình).
Giới hạn thời gian tìm kiếm: DEADLINE_MIN; quá hạn thì dừng lấy cấu hình mới và ghi số đã chạy."""
import json

import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

from post_i_common import *
from apf.splits import check_embargo

N_CFG, DEADLINE_MIN, WORKERS, THREADS = 36, 75, 3, 3
FOLDS = [TRAIN_CUTOFFS[-1], TUNE_CUTOFFS[0], TUNE_CUTOFFS[1]]
FULL = list(Mo.FULL_FEATURES)
MONO = {"recency_days": -1, "recency_to_median_gap": -1, "frequency_days": 1, "days_30": 1, "days_90": 1, "days_180": 1,
        "days_365": 1, "active_months_12": 1, "prev_90d_active": 1}
MONO_VEC = [MONO.get(f, 0) for f in FULL]


def sample_configs():
    r = np.random.default_rng(SEED)
    lu = lambda a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))
    out = []
    for i in range(N_CFG):
        out.append(dict(cfg_id=i, num_leaves=int(round(lu(5, 63))), min_child_samples=int(round(lu(20, 500))),
                        learning_rate=lu(0.01, 0.1), reg_lambda=lu(0.1, 30), colsample_bytree=float(r.uniform(0.5, 1)),
                        subsample=float(r.uniform(0.5, 1)), monotone=bool(r.integers(0, 2))))
    return out


def params_of(c):
    p = {**Mo.LGBM_FIXED, "n_jobs": THREADS, "num_leaves": c["num_leaves"], "min_child_samples": c["min_child_samples"],
         "learning_rate": c["learning_rate"], "reg_lambda": c["reg_lambda"], "colsample_bytree": c["colsample_bytree"],
         "subsample": c["subsample"]}
    if c["subsample"] < 1:
        p["subsample_freq"] = 1
    if c["monotone"]:
        p["monotone_constraints"] = MONO_VEC
    return p


def cv_one(dev, c, deadline, ckdir, n_fixed=None):
    path = ckdir / f"d_cfg_{c['cfg_id']}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    if time.time() > deadline:
        return None
    t0, aps, its = time.time(), [], []
    for val in FOLDS:
        fit_cuts = [x for x in TRAIN_CUTOFFS + TUNE_CUTOFFS if x < val]
        assert check_embargo(fit_cuts, [val])
        F, V = dev[dev["cutoff"].isin(fit_cuts)], dev[dev["cutoff"] == val]
        yF, yV = F["y"].to_numpy(), V["y"].to_numpy()
        if n_fixed is None:
            m = lgb.LGBMClassifier(**params_of(c), metric="None")
            m.fit(F[FULL], yF, eval_X=(V[FULL],), eval_y=(yV,), eval_metric=Mo.ap_no_purchase_eval,
                  callbacks=[lgb.early_stopping(Mo.EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)])
            its.append(int(m.best_iteration_))
        else:
            m = lgb.LGBMClassifier(**{**params_of(c), "n_estimators": n_fixed}).fit(F[FULL], yF)
            its.append(n_fixed)
        aps.append(float(average_precision_score(1 - yV, p_no_purchase(m, V[FULL]))))
    r = dict(c, cv_ap_mean=float(np.mean(aps)), cv_ap_sd=float(np.std(aps, ddof=1)), **{f"ap_fold{i}": a for i, a in enumerate(aps)},
             **{f"iter_fold{i}": n for i, n in enumerate(its)}, n_trees_final=int(np.median(its)), seconds=time.time() - t0)
    path.write_bytes(pickle.dumps(r))
    return r


table = get_table()
# ---- tìm kiếm: chỉ các dòng Train + Tune; không có dòng kiểm định trong khung dữ liệu này ----
dev = table[table["in_train"] | table["in_tune"]].copy()
assert not dev["holdout"].any() and not dev["cutoff"].isin(TEST_CUTOFFS).any() and not dev["cutoff"].isin([CUTOFFS[12]]).any()
assert set(FOLDS) <= set(dev["cutoff"])
cfgs = sample_configs()
deadline = T0 + DEADLINE_MIN * 60
res = Parallel(n_jobs=WORKERS, verbose=5)(delayed(cv_one)(dev, c, deadline, CKPT) for c in cfgs)
done = [r for r in res if r is not None]
log(f"đã chạy {len(done)}/{N_CFG} cấu hình")
LOCK_BASE = dict(num_leaves=CONF["M3_lgbm_full"]["num_leaves"], min_child_samples=CONF["M3_lgbm_full"]["min_child_samples"],
                 learning_rate=Mo.LGBM_FIXED["learning_rate"], reg_lambda=1.0, colsample_bytree=1.0, subsample=1.0, monotone=False)
ref_es = cv_one(dev, dict(LOCK_BASE, cfg_id="locked_early_stop"), T0 + 1e9, CKPT)
ref_fx = cv_one(dev, dict(LOCK_BASE, cfg_id="locked_114_trees"), T0 + 1e9, CKPT, n_fixed=CONF["M3_lgbm_full"]["best_iteration"])
search = pd.DataFrame(done).sort_values(["cv_ap_mean", "cfg_id"], ascending=[False, True]).reset_index(drop=True)
best = search.iloc[0].to_dict()
search["selected"] = search["cfg_id"] == best["cfg_id"]
out = pd.concat([search, pd.DataFrame([ref_es, ref_fx]).assign(selected=False)], ignore_index=True)
write_csv(out, "post_i_d_search.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 100)
print(out[["cfg_id", "num_leaves", "min_child_samples", "learning_rate", "reg_lambda", "colsample_bytree", "subsample", "monotone",
           "cv_ap_mean", "cv_ap_sd", "ap_fold0", "ap_fold1", "ap_fold2", "iter_fold0", "iter_fold1", "iter_fold2", "n_trees_final", "seconds"]].round(4).to_string())
# Chốt cấu hình TRƯỚC khi đọc tập kiểm định; nếu chạy lại, phải trùng cấu hình đã chốt.
sel = ckpt("d_selected", lambda: {k: best[k] for k in ("cfg_id", "num_leaves", "min_child_samples", "learning_rate", "reg_lambda",
                                                       "colsample_bytree", "subsample", "monotone", "n_trees_final", "cv_ap_mean")})
assert sel["cfg_id"] == best["cfg_id"], "cấu hình chọn đã đổi sau lần chốt"
log("CHỐT:", json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in sel.items()}))


def final():
    c = {k: sel[k] for k in ("num_leaves", "min_child_samples", "learning_rate", "reg_lambda", "colsample_bytree", "subsample", "monotone")}
    p = params_of(c)
    extra = {k: v for k, v in p.items() if k not in Mo.LGBM_FIXED or v != Mo.LGBM_FIXED[k]}
    cfg = {k: extra.pop(k) for k in list(extra) if k in ("num_leaves", "min_child_samples")}
    n = int(sel["n_trees_final"])
    keep = {}
    builders = {"D_best_lgbm": lambda: LGBM(FULL, cfg, n, **extra), "M3_lgbm_full": lambda: locked("M3_lgbm_full"),
                "M2_lr_full": lambda: locked("M2_lr_full")}
    tab = evaluate_models(table, builders, keep=keep)
    tr, tu, rf, tt, ho = frames(table)
    fns = dict(boot_fns(), cal_intercept=intercept_fn)
    diffs = []
    for sname, S in (("temporal_test", tt), ("holdout_test", ho)):
        P = {m: keep[(sname, m)] for m in builders}
        diffs.append(paired_diff(S, P, [("D_best_lgbm", "M3_lgbm_full"), ("D_best_lgbm", "M2_lr_full")], fns).assign(set=sname))
        log("bootstrap xong", sname)
    return tab, pd.concat(diffs, ignore_index=True)


tab, diff = ckpt("d_final", final)     # chấm một lần; lần chạy lại đọc đệm, không chấm lại
tab["note"] = np.where(tab["set"] == "tune", "Tune đã dùng để chọn cấu hình: lạc quan", "")
write_csv(tab, "post_i_d_final.csv")
write_csv(diff, "post_i_d_paired_diff.csv")
L = pd.read_csv(OUTPUT_DIR / "stage_d_test_metrics.csv")
for _, r in tab[(tab["model"] == "M3_lgbm_full") & (tab["set"] != "tune")].iterrows():
    q = L[(L["set"] == r["set"]) & (L["model"] == "M3_lgbm_full") & (L["scope"] == "pooled")].iloc[0]
    assert round(r["ap_y0"], 4) == round(q["ap_y0"], 4)
print(tab[["set", "model", "n", "ap_y0", "roc_auc", "brier", "cal_intercept", "cal_slope", "precision_top10", "recall_top10", "precision_top20", "recall_top20"]].round(4).to_string())
print(diff.round(4).to_string())
log("XONG D")
