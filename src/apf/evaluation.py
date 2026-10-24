"""Metrics and baselines. The event of interest is y = 0 (no purchase day in the 90-day window).

Convention: every model and baseline returns `p_no_purchase` = P(y = 0). Metrics take y (1 = purchase)
and convert to the event indicator e = 1 - y internally, so a score can never be paired with the
wrong class by accident.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def event(y):
    y = np.asarray(y)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0/1")
    return (y == 0).astype(int)


def p_no_purchase(model, X):
    """P(y = 0) from a fitted sklearn-style classifier, selected by label, not by column position."""
    classes = list(model.classes_)
    if sorted(classes) != [0, 1]:
        raise ValueError(f"expected classes 0/1, got {classes}")
    return model.predict_proba(X)[:, classes.index(0)]


def metrics(y, p0, score0=None, train_prevalence0=None):
    """p0: probability of y = 0. score0: optional ranking score for y = 0 (defaults to p0)."""
    e = event(y)
    p0 = np.asarray(p0, float)
    s = p0 if score0 is None else np.asarray(score0, float)
    if not ((p0 >= 0) & (p0 <= 1)).all():
        raise ValueError("p0 must be a probability")
    out = {"n": int(len(e)), "n_event_y0": int(e.sum()), "prevalence_y0": float(e.mean()),
           "ap_no_skill": float(e.mean()),
           "ap_y0": float(average_precision_score(e, s)),
           "roc_auc": float(roc_auc_score(e, s)),
           "brier": float(brier_score_loss(e, p0))}
    ref = e.mean() if train_prevalence0 is None else train_prevalence0
    out["brier_reference"] = float(np.mean((e - ref) ** 2))
    out["brier_skill"] = 1 - out["brier"] / out["brier_reference"]
    out.update(calibration_intercept_slope(e, p0))
    return out


def calibration_intercept_slope(e, p0, eps=1e-6):
    """Logistic recalibration of e on logit(p0): slope 1, intercept 0 = well calibrated.
    Intercept is estimated with the slope fixed at 1 (calibration-in-the-large)."""
    lp = np.log(np.clip(p0, eps, 1 - eps) / (1 - np.clip(p0, eps, 1 - eps)))
    if np.ptp(lp) == 0:
        return {"cal_slope": np.nan, "cal_intercept": np.nan}
    slope = LogisticRegression(C=np.inf).fit(lp[:, None], e).coef_[0, 0]   # unpenalised
    # intercept with offset lp: solve mean(sigmoid(a + lp)) = mean(e) by bisection
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(mid + lp)))) < e.mean():
            lo = mid
        else:
            hi = mid
    return {"cal_slope": float(slope), "cal_intercept": float((lo + hi) / 2)}


def reliability_table(y, p0, n_bins=10):
    """Quantile bins of p0: mean predicted vs observed rate of y = 0."""
    e = event(y)
    d = pd.DataFrame({"p0": p0, "e": e})
    d["bin"] = pd.qcut(d["p0"].rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(n=("e", "size"), mean_p0=("p0", "mean"), observed_y0=("e", "mean"))


def top_fraction(y, score0, frac):
    """Flag the top `frac` of observations by score0; precision and recall for y = 0.

    Ties at the threshold are shared out in expectation (as if broken at random), so the result
    does not depend on row order. This matters for coarse scores (persistence has 3 levels)."""
    e = event(y)
    s = np.asarray(score0, float)
    k = int(np.ceil(frac * len(e)))
    t = np.sort(s)[::-1][k - 1]                     # score of the k-th flagged row
    above, at = s > t, s == t
    take = (k - above.sum()) / at.sum()             # share of tied rows that is flagged
    tp = (e[above] == 1).sum() + take * (e[at] == 1).sum()
    return {"fraction": frac, "flagged": k, "tied_at_threshold": int(at.sum()),
            "precision_y0": float(tp / k), "recall_y0": float(tp / max(e.sum(), 1))}


def _clusters(groups):
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    return np.split(order, np.cumsum(np.bincount(inv))[:-1])


def cluster_bootstrap(y, p0, groups, fn, B=1000, seed=0):
    """Respondent-cluster bootstrap: resample respondents with replacement, keep all their rows."""
    rows_of = _clusters(groups)
    r = np.random.default_rng(seed)
    y, p0 = np.asarray(y), np.asarray(p0)
    out = []
    for _ in range(B):
        idx = np.concatenate([rows_of[k] for k in r.integers(0, len(rows_of), len(rows_of))])
        out.append(fn(y[idx], p0[idx]))
    return np.asarray(out)


def paired_cluster_bootstrap(y, preds, groups, metric_fns, B=1000, seed=0):
    """Paired respondent-cluster bootstrap. Every model is scored on the *same* resampled
    respondents in each draw, so differences between models are paired.

    preds: {model: (p0, score0)}; metric_fns: {metric: fn(y, p0, score0) -> float}.
    Returns a long DataFrame: draw, model, metric, value."""
    rows_of = _clusters(groups)
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    preds = {m: (np.asarray(p, float), np.asarray(p if s is None else s, float)) for m, (p, s) in preds.items()}
    recs = []
    for b in range(B):
        idx = np.concatenate([rows_of[k] for k in r.integers(0, len(rows_of), len(rows_of))])
        for m, (p, s) in preds.items():
            for name, fn in metric_fns.items():
                recs.append((b, m, name, fn(y[idx], p[idx], s[idx])))
    return pd.DataFrame(recs, columns=["draw", "model", "metric", "value"])


def percentile_ci(draws, lo=2.5, hi=97.5):
    return float(np.nanpercentile(draws, lo)), float(np.nanpercentile(draws, hi))


# ---- baselines: fitted on training rows only; each returns P(y = 0) ------------------------

class PrevalenceBaseline:
    """Constant training rate of y = 0 (no ranking skill: AUC 0.5, AP = prevalence)."""

    def fit(self, F, y):
        self.p0_ = float(event(y).mean())
        return self

    def predict_p0(self, F):
        return np.full(len(F), self.p0_)


class PersistenceBaseline:
    """P(y = 0 | prev_90d_active), estimated on training rows; separate cell for missing
    (no purchase on or before the previous quarter-end)."""

    def fit(self, F, y):
        e = pd.Series(event(y), index=F.index)
        key = F["prev_90d_active"].fillna(-1)
        self.rate_ = e.groupby(key).mean().to_dict()
        self.default_ = float(e.mean())
        return self

    def predict_p0(self, F):
        key = F["prev_90d_active"].fillna(-1)
        return key.map(self.rate_).fillna(self.default_).to_numpy(float)


class RecencyRule:
    """Ranking score = recency_days (longer since last purchase -> more likely y = 0).
    Probabilities from a monotone (isotonic) map of recency fitted on training rows."""

    def fit(self, F, y):
        self.iso_ = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=1)
        self.iso_.fit(F["recency_days"].to_numpy(float), event(y))
        return self

    def score0(self, F):
        return F["recency_days"].to_numpy(float)

    def predict_p0(self, F):
        return self.iso_.predict(F["recency_days"].to_numpy(float))
