# POST-HOC, không dùng để đổi kết luận P1–P3
"""Phần dùng chung cho nhóm post_i_* (C, D, E, F). Chỉ thêm file mới; dùng lại post_common.py và src/apf.
Quy ước: mọi mô hình trả P(y=0) (`p0`) và điểm xếp hạng cho y=0 (`score0`); y=1 là có mua.
Kết quả trung gian lưu trong thư mục đệm ngoài dự án (APF_POST_CACHE/post_i_ckpt) để chạy lại được khi bị ngắt."""
import time
import pickle

from post_common import *   # noqa: F401,F403  (np, pd, ID, CONF, SEEDS, OUT, write_csv, get_table, fit_final, ...)
from post_common import CACHE
from apf import models as Mo
from apf.evaluation import metrics, top_fraction_by_group, p_no_purchase, event
from apf.phase2 import _Model
from apf.splits import TEST_CUTOFFS, TRAIN_CUTOFFS, TUNE_CUTOFFS, REFIT_CUTOFFS
from apf.config import CUTOFFS

SEED = 20261024
CKPT = CACHE.parent / "post_i_ckpt"
CKPT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def ckpt(name, fn):
    """Chạy fn() một lần; lần sau đọc lại từ đệm (để chạy lại khi bị ngắt)."""
    p = CKPT / f"{name}.pkl"
    if p.exists():
        log("ckpt hit:", name)
        return pickle.loads(p.read_bytes())
    v = fn()
    p.write_bytes(pickle.dumps(v))
    return v


def frames(table):
    """Train, Tune, Refit (đều không có hold-out) + hai tập kiểm định. Có assert chống rò rỉ."""
    tr, tu, rf = table[table["in_train"]], table[table["in_tune"]], table[table["in_refit"]]
    for f in (tr, tu, rf):
        assert not f["holdout"].any()
        assert not f["cutoff"].isin(TEST_CUTOFFS).any()
    tt, ho = table[table["in_test_temporal"]], table[table["in_test_holdout"]]
    return tr, tu, rf, tt, ho


def score_row(S, p0, s0, p_fit, **tags):
    """Một dòng chỉ số: AP, ROC-AUC, Brier, hệ số chặn/độ dốc hiệu chuẩn, precision/recall top-10/20%
    (top-k theo từng mốc rồi gộp, như mô hình khóa)."""
    y, cut = S["y"].to_numpy(), S["cutoff"].to_numpy()
    p0, s0 = np.asarray(p0, float), np.asarray(s0, float)
    r = metrics(y, p0, s0, train_prevalence0=p_fit)
    out = dict(tags, n=r["n"], prevalence_y0=r["prevalence_y0"], ap_y0=r["ap_y0"], roc_auc=r["roc_auc"],
               brier=r["brier"], cal_intercept=r["cal_intercept"], cal_slope=r["cal_slope"],
               mean_pred_p0=float(p0.mean()))
    for frac in (0.10, 0.20):
        t = top_fraction_by_group(y, s0, cut, frac)
        out[f"precision_top{int(frac * 100)}"], out[f"recall_top{int(frac * 100)}"] = t["precision_y0"], t["recall_y0"]
    return out


# ---- các dạng mô hình có chung giao diện fit / p0 / score0 / p_fit ----------------------------------------

class SklearnLR:
    """Pipeline sklearn đã có (make_logreg hoặc spline) — dự báo P(y=0)."""

    def __init__(self, build, cols=None):
        self.build, self.cols = build, cols

    def fit(self, F, y):
        self.m = self.build().fit(F if self.cols is None else F[self.cols], y)
        self.p_fit = float(event(y).mean())
        return self

    def p0(self, F):
        return p_no_purchase(self.m, F if self.cols is None else F[self.cols])

    def score0(self, F):
        return self.p0(F)


class LGBM:
    def __init__(self, cols, cfg, n_trees, **extra):
        self.cols, self.cfg, self.n, self.extra = list(cols), cfg, n_trees, extra

    def fit(self, F, y):
        params = {**Mo.LGBM_FIXED, **self.cfg, **self.extra, "n_estimators": int(self.n)}
        import lightgbm as lgb
        self.m = lgb.LGBMClassifier(**params).fit(F[self.cols], y)
        self.p_fit = float(event(y).mean())
        return self

    def p0(self, F):
        return p_no_purchase(self.m, F[self.cols])

    def score0(self, F):
        return self.p0(F)


class GapRule:
    """Quy tắc xếp hạng theo một cột (recency_to_median_gap hoặc median_gap_days). Xác suất: hồi quy đẳng hướng
    trên Train/Refit (chiều tăng/giảm tự chọn bằng dữ liệu huấn luyện); ô riêng cho NaN (chỉ 1 ngày mua).
    score0 = giá trị cột (đổi dấu nếu giảm); với NaN, điểm là giá trị có xác suất đẳng hướng gần nhất tỉ lệ NaN ở Train."""

    def __init__(self, col):
        self.col = col

    def fit(self, F, y):
        from sklearn.isotonic import IsotonicRegression
        e = event(y)
        s = F[self.col].to_numpy(float)
        ok = ~np.isnan(s)
        self.iso = IsotonicRegression(increasing="auto", out_of_bounds="clip", y_min=0, y_max=1).fit(s[ok], e[ok])
        self.sign = 1.0 if self.iso.increasing_ else -1.0
        self.nan_rate = float(e[~ok].mean()) if (~ok).any() else float(e.mean())
        grid = np.unique(s[ok])
        self.nan_score = float(self.sign * grid[np.argmin(np.abs(self.iso.predict(grid) - self.nan_rate))])
        self.p_fit = float(e.mean())
        return self

    def p0(self, F):
        s = F[self.col].to_numpy(float)
        return np.where(np.isnan(s), self.nan_rate, self.iso.predict(np.where(np.isnan(s), 0.0, s)))

    def score0(self, F):
        s = F[self.col].to_numpy(float)
        return np.where(np.isnan(s), self.nan_score, self.sign * s)


def locked(mid):
    """Mô hình khóa (B2, M1, M2, M3, ...) dưới cấu hình khóa."""
    return _Model(*model_specs(CONF)[mid])


def evaluate_models(table, builders, tags=None, sets=("tune", "temporal_test", "holdout_test"), keep=None):
    """builders: {tên: hàm tạo mô hình mới}. Tune: huấn luyện Train; hai tập kiểm định: huấn luyện Refit.
    Mỗi mô hình được chấm đúng một lần trên mỗi tập (cấu hình đã chốt trước khi gọi)."""
    tr, tu, rf, tt, ho = frames(table)
    plan = {"tune": (tr, tu), "temporal_test": (rf, tt), "holdout_test": (rf, ho)}
    rows = []
    for name, mk in builders.items():
        fitted = {}
        for sname in sets:
            fit_df, S = plan[sname]
            key = "train" if sname == "tune" else "refit"
            if key not in fitted:
                fitted[key] = mk().fit(fit_df, fit_df["y"].to_numpy())
            m = fitted[key]
            p0, s0 = m.p0(S), m.score0(S)
            if keep is not None:
                keep[(sname, name)] = (p0, s0)
            rows.append(score_row(S, p0, s0, m.p_fit, set=sname, model=name, **(tags or {})))
        log("done", name)
    return pd.DataFrame(rows)


def paired_diff(S, preds, pairs, fns, B=1000, seed=SEED):
    """Khoảng bootstrap cụm ghép cặp theo người tham gia cho hiệu số a - b. preds: {tên: (p0, score0)};
    pairs: [(a, b)]; fns: {chỉ số: fn(y, p0, score0)}. Ước lượng điểm tính trên mẫu gốc."""
    from apf.evaluation import paired_cluster_bootstrap, percentile_ci
    y, g = S["y"].to_numpy(), S[ID].to_numpy()
    use = sorted({m for pair in pairs for m in pair})
    d = paired_cluster_bootstrap(y, {m: preds[m] for m in use}, g, fns, B=B, seed=seed)
    w = d.pivot_table(index=["draw", "metric"], columns="model", values="value")
    rows = []
    for a, b in pairs:
        for k, fn in fns.items():
            lo, hi = percentile_ci((w[a] - w[b]).xs(k, level="metric").to_numpy())
            est = fn(y, *preds[a]) - fn(y, *preds[b])
            rows.append(dict(model_a=a, model_b=b, metric=k, estimate=float(est), ci_low=lo, ci_high=hi,
                             ci_excludes_0=bool(lo > 0 or hi < 0), B=B))
    return pd.DataFrame(rows)


def boot_fns():
    from apf.phase2 import _boot_fns
    return _boot_fns()


def intercept_fn(y, p, s):
    # chỉ cần hệ số chặn; giải bằng phân đôi (không dùng độ dốc để nhanh hơn)
    e = event(y)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(q / (1 - q))
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(mid + lp)))) < e.mean():
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
