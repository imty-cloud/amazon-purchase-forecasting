"""Metric orientation (event = y = 0), baselines, cluster bootstrap and the split design."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from apf import config as C
from apf import splits as S
from apf.evaluation import (PersistenceBaseline, PrevalenceBaseline, RecencyRule, cluster_bootstrap,
                            event, metrics, p_no_purchase, top_fraction)

ID = C.ID


# ---- class orientation --------------------------------------------------------------------

def test_event_is_no_purchase():
    assert event([1, 0, 1]).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        event([0, 2])


class Reversed:
    """Classifier whose classes_ are ordered [1, 0]."""
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        p0 = np.asarray(X)[:, 0]
        return np.column_stack([1 - p0, p0])


def test_p_no_purchase_selects_class_by_label():
    X = np.array([[0.9], [0.2]])
    assert p_no_purchase(Reversed(), X).tolist() == [0.9, 0.2]
    r = np.random.default_rng(0)
    Xs = r.normal(size=(400, 1))
    y = (Xs[:, 0] + r.normal(size=400) > 0).astype(int)
    lr = LogisticRegression().fit(Xs, y)
    p0 = p_no_purchase(lr, Xs)
    assert np.allclose(p0, 1 - lr.predict_proba(Xs)[:, 1])
    assert metrics(y, p0)["roc_auc"] > 0.7


def test_metric_orientation_and_no_skill():
    y = np.array([1, 1, 1, 0, 1, 0, 1, 1])
    perfect = (y == 0).astype(float) * 0.9 + 0.05
    m = metrics(y, perfect)
    assert m["ap_y0"] == 1 and m["roc_auc"] == 1 and m["prevalence_y0"] == 0.25
    wrong_class = metrics(y, 1 - perfect)          # passing P(y=1) by mistake
    assert wrong_class["roc_auc"] == 0 and wrong_class["ap_y0"] < m["ap_no_skill"] + 0.2
    const = metrics(y, np.full(len(y), 0.25))
    assert const["ap_y0"] == pytest.approx(const["ap_no_skill"]) and const["roc_auc"] == 0.5
    assert const["brier"] == pytest.approx(np.mean(((y == 0) - 0.25) ** 2))
    assert const["brier_skill"] == pytest.approx(0)
    with pytest.raises(ValueError):
        metrics(y, perfect * 2)


def test_calibration_of_well_calibrated_predictions():
    r = np.random.default_rng(1)
    p0 = r.uniform(0.02, 0.6, 200_000)
    y = np.where(r.uniform(size=p0.size) < p0, 0, 1)
    m = metrics(y, p0)
    assert m["cal_slope"] == pytest.approx(1, abs=0.03) and m["cal_intercept"] == pytest.approx(0, abs=0.03)
    shifted = metrics(y, np.clip(p0 * 1.5, 0, 1))
    assert shifted["cal_intercept"] < -0.2           # over-prediction gives a negative intercept


def test_top_fraction():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 1])
    s = np.array([10, 9, 8, 1, 1, 7, 1, 1, 1, 1], float)
    t = top_fraction(y, s, 0.3)
    assert t["flagged"] == 3 and t["precision_y0"] == pytest.approx(2 / 3)
    assert t["recall_y0"] == pytest.approx(2 / 3)


def test_cluster_bootstrap_resamples_respondents():
    groups = np.array(["a"] + ["b"] * 9)
    sizes = cluster_bootstrap(np.ones(10, int), np.zeros(10), groups, lambda y, p: len(y), B=200)
    assert set(np.unique(sizes)) <= {2, 10, 18} and len(np.unique(sizes)) > 1


# ---- baselines ----------------------------------------------------------------------------

def test_baselines_fit_on_training_rows_only():
    F = pd.DataFrame({"prev_90d_active": [1, 1, 0, 0, np.nan, 1], "recency_days": [1, 5, 50, 80, 3, 2]})
    y = np.array([1, 1, 0, 1, 1, 0])
    pb = PersistenceBaseline().fit(F, y)
    assert pb.rate_ == {1.0: pytest.approx(1 / 3), 0.0: 0.5, -1.0: 0.0}
    assert pb.predict_p0(pd.DataFrame({"prev_90d_active": [0.0, np.nan]})).tolist() == [0.5, 0.0]
    assert PrevalenceBaseline().fit(F, y).predict_p0(F).tolist() == [1 / 3] * 6
    rr = RecencyRule().fit(F, y)
    p = rr.predict_p0(pd.DataFrame({"recency_days": [0, 10, 100, 1000]}))
    assert (np.diff(p) >= 0).all() and ((p >= 0) & (p <= 1)).all()
    assert rr.score0(F).tolist() == F["recency_days"].tolist()


# ---- split design ---------------------------------------------------------------------------

def _panel(n=200):
    ids = [f"R_{i:015d}" for i in range(n)]
    return pd.DataFrame([(i, c, 1) for i in ids for c in C.CUTOFFS], columns=[ID, "cutoff", "y"])


def test_holdout_is_deterministic_and_order_free():
    ids = [f"R_{i:015d}" for i in range(1000)]
    a = S.holdout_respondents(ids)
    b = S.holdout_respondents(list(reversed(ids)) + ids[:10])
    assert a.equals(b) and len(a) == 200
    assert not a.equals(S.holdout_respondents(ids, seed=1))


def test_roles_and_holdout_isolation():
    p = _panel()
    h = S.holdout_respondents(p[ID])
    a = S.assign(p, h)
    S.check_design(a)
    assert not a.loc[a["holdout"], ["in_train", "in_tune", "in_refit", "in_test_temporal"]].any().any()
    assert a.groupby(ID)["holdout"].nunique().eq(1).all()        # whole respondent in one group
    assert a.loc[a["in_test_holdout"], ID].isin(h).all()
    assert set(a.loc[a["in_train"], "cutoff"]) == set(S.TRAIN_CUTOFFS)
    bad = a.copy()
    bad.loc[bad.index[bad["holdout"]][0], "in_train"] = True
    with pytest.raises(AssertionError):
        S.check_design(bad)


def test_embargo():
    assert S.check_embargo(S.TRAIN_CUTOFFS, S.TUNE_CUTOFFS)
    assert S.check_embargo(S.REFIT_CUTOFFS, S.TEST_CUTOFFS)
    assert S.check_embargo(C.CUTOFFS[:11], C.CUTOFFS[11:12])         # label ends 2021-09-28 <= 2021-09-30
    assert not S.check_embargo(C.CUTOFFS[:12], C.CUTOFFS[11:12])     # label ends 2021-12-29 > 2021-09-30
    assert not S.check_embargo(C.CUTOFFS[:14], S.TEST_CUTOFFS)      # 2022-03-31 label overlaps test
    assert S.TRAIN_CUTOFFS[-1] == pd.Timestamp("2021-03-31")
    assert S.TUNE_CUTOFFS == [pd.Timestamp("2021-06-30"), pd.Timestamp("2021-09-30")]
    assert S.TEST_CUTOFFS == [pd.Timestamp("2022-03-31"), pd.Timestamp("2022-06-30")]
