"""Stage D building blocks on synthetic data only: model factories, early stopping, refit,
paired bootstrap, tie-safe operating points, test-set lock, prev_90d_active provenance."""
import ast
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apf import config as C
from apf import lock as L
from apf import models as Mo
from apf.evaluation import metrics, p_no_purchase, paired_cluster_bootstrap, top_fraction
from apf.features import FEATURE_COLUMNS, previous_quarter_end
from apf.splits import TEST_CUTOFFS


def synth(n=3000, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f: r.gamma(2, 5, n) for f in FEATURE_COLUMNS})
    X["cutoff_quarter"] = r.integers(1, 5, n).astype(float)
    X["prev_90d_active"] = r.integers(0, 2, n).astype(float)
    for f in ("median_gap_days", "recency_to_median_gap", "prev_90d_active"):
        X.loc[r.uniform(size=n) < 0.05, f] = np.nan
    logit = 2.5 - 0.15 * X["recency_days"] + 0.8 * X["prev_90d_active"].fillna(0)
    y = (r.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


# ---- feature sets and logistic regression -------------------------------------------------

def test_feature_sets():
    assert Mo.FULL_FEATURES == FEATURE_COLUMNS and len(Mo.FULL_FEATURES) == 19
    assert "prev_90d_active" not in Mo.NO_PREV_FEATURES and len(Mo.NO_PREV_FEATURES) == 18
    assert Mo.RFM_FEATURES == ["recency_days", "frequency_days", "monetary_total"]


@pytest.mark.parametrize("fs", ["rfm", "full", "no_prev"])
def test_logreg_handles_missing_and_points_at_no_purchase(fs):
    X, y = synth()
    m = Mo.make_logreg(Mo.FEATURE_SETS[fs], C=1.0).fit(X, y)
    p0 = p_no_purchase(m, X)
    assert np.isfinite(p0).all()
    assert metrics(y, p0)["roc_auc"] > 0.7                 # recency drives y = 0 in the simulation
    n_in = m.named_steps["prep"].transform(X.head()).shape[1]
    expected = {"rfm": 3, "full": 19 - 4 + 2 * 2 + 2 + 3, "no_prev": 18 - 3 + 2 * 2 + 3}[fs]
    assert n_in == expected


def test_logreg_preprocessing_fitted_on_fitting_rows_only():
    X, y = synth()
    m = Mo.make_logreg(Mo.FULL_FEATURES, 1.0).fit(X.iloc[:1000], y[:1000])
    imp = m.named_steps["prep"].named_transformers_["median_gap_days"].named_steps["impute"]
    assert imp.statistics_[0] == pytest.approx(X["median_gap_days"].iloc[:1000].median())
    sc = m.named_steps["prep"].named_transformers_["log"].named_steps["scale"]
    assert sc.n_samples_seen_ == 1000


def test_logreg_rejects_unknown_quarter():
    X, y = synth()
    m = Mo.make_logreg(Mo.FULL_FEATURES, 1.0).fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X.head(1).assign(cutoff_quarter=5.0))


# ---- LightGBM ----------------------------------------------------------------------------

def test_lgbm_grid_is_small_and_documented():
    cfgs = Mo.lgbm_configs()
    assert len(cfgs) == 6 and {"num_leaves": 7, "min_child_samples": 50} in cfgs


def test_ap_eval_metric_targets_no_purchase():
    y = np.array([1, 1, 0, 1, 0])
    p1 = np.array([0.9, 0.8, 0.1, 0.7, 0.2])           # P(y=1): low for the two y = 0 rows
    assert Mo.ap_no_purchase_eval(y, p1) == ("ap_no_purchase", 1.0, True)


def test_early_stopping_then_refit_with_selected_iterations():
    X, y = synth(4000, 1)
    Xv, yv = synth(1500, 2)
    cfg = {"num_leaves": 7, "min_child_samples": 50}
    m = Mo.fit_lgbm_early_stopping(X, y, Xv, yv, cfg)
    best = m.best_iteration_
    assert 1 <= best < Mo.LGBM_FIXED["n_estimators"]
    # the metric LightGBM optimised equals our AP for y = 0 at the best iteration
    p0 = p_no_purchase(m, Xv)
    assert m.best_score_["valid_0"]["ap_no_purchase"] == pytest.approx(metrics(yv, p0)["ap_y0"])
    r = Mo.make_lgbm_refit(cfg, best).fit(X, y)
    assert r.n_estimators == best and r.booster_.current_iteration() == best
    r2 = Mo.make_lgbm_refit(cfg, best).fit(X, y)
    assert np.array_equal(p_no_purchase(r, Xv), p_no_purchase(r2, Xv))   # deterministic


def test_selection_rule():
    res = [{"cfg": "a", "tune_ap_y0": 0.40, "num_leaves": 31},
           {"cfg": "b", "tune_ap_y0": 0.42, "num_leaves": 31},
           {"cfg": "c", "tune_ap_y0": 0.42, "num_leaves": 7}]
    assert Mo.select(res, tiebreak=("num_leaves",))["cfg"] == "c"


# ---- operating points and paired bootstrap ----------------------------------------------

def test_top_fraction_ties_do_not_depend_on_row_order():
    y = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1])
    s = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], float)     # 4 tied at the top, k = 2
    a = top_fraction(y, s, 0.2)
    perm = np.random.default_rng(0).permutation(10)
    b = top_fraction(y[perm], s[perm], 0.2)
    assert a == b and a["precision_y0"] == pytest.approx(0.25) and a["recall_y0"] == pytest.approx(0.25)
    assert a["tied_at_threshold"] == 4


def test_paired_bootstrap_is_paired_and_by_respondent():
    r = np.random.default_rng(3)
    groups = np.repeat([f"R{i}" for i in range(300)], 2)
    y = np.where(r.uniform(size=600) < 0.15, 0, 1)
    p_a = np.clip((y == 0) * 0.3 + r.uniform(0, 0.5, 600), 0, 1)
    fns = {"ap": lambda y, p, s: metrics(y, p, s)["ap_y0"]}
    d = paired_cluster_bootstrap(y, {"A": (p_a, None), "A_copy": (p_a.copy(), None)}, groups, fns, B=50)
    w = d.pivot_table(index="draw", columns="model", values="value")
    assert (w["A"] - w["A_copy"]).abs().max() == 0             # same draws for every model
    assert w["A"].std() > 0
    sizes = paired_cluster_bootstrap(y, {"A": (p_a, None)}, groups,
                                     {"n": lambda y, p, s: len(y)}, B=20)["value"]
    assert (sizes == 600).all()                                 # 300 respondents x 2 rows


# ---- test-set lock ---------------------------------------------------------------------

def test_rows_before_test_excludes_test_cutoffs():
    t = pd.DataFrame({"cutoff": C.CUTOFFS})
    assert not L.rows_before_test(t)["cutoff"].isin(TEST_CUTOFFS).any()
    assert len(L.rows_before_test(t)) == 13


def _git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)


def test_lock_requires_committed_unchanged_protocol(tmp_path):
    repo = tmp_path
    (repo / "reports").mkdir(); (repo / "outputs").mkdir(); (repo / "src").mkdir()
    proto, lock = repo / "reports" / "stage_d_protocol.md", repo / "outputs" / "stage_d_locked_config.json"
    proto.write_text("protocol v1"); (repo / "src" / "m.py").write_text("x = 1")
    kw = dict(protocol=proto, lock=lock, repo=repo)
    with pytest.raises(L.LockError):
        L.require_lock(**kw)                                    # no lock yet
    _git(repo, "init", "-q"); _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    L.write_lock({"lgbm": {"num_leaves": 7}}, protocol=proto, lock=lock)
    with pytest.raises(L.LockError):
        L.require_lock(**kw)                                    # not committed
    _git(repo, "add", "-A"); _git(repo, "commit", "-qm", "lock")
    assert L.require_lock(**kw)["selected"] == {"lgbm": {"num_leaves": 7}}
    table = pd.DataFrame({"cutoff": C.CUTOFFS})
    assert len(L.release_test_rows(table, **kw)[0]) == 2
    proto.write_text("protocol v2")
    with pytest.raises(L.LockError):
        L.require_lock(**kw)                                    # protocol edited after locking
    _git(repo, "commit", "-qam", "edit protocol after locking")
    with pytest.raises(L.LockError, match="protocol changed"):
        L.require_lock(**kw)                                    # ... even when the edit is committed
    proto.write_text("protocol v1"); _git(repo, "commit", "-qam", "revert protocol")
    assert L.require_lock(**kw)
    (repo / "src" / "m.py").write_text("x = 2")
    with pytest.raises(L.LockError, match="uncommitted"):
        L.require_lock(**kw)                                    # code edited after locking


# ---- prev_90d_active provenance ----------------------------------------------------

def test_prev_quarter_window_is_point_in_time_for_every_cutoff():
    for c in C.CUTOFFS:
        cp = previous_quarter_end(c)
        assert cp < c and (c - cp).days in (90, 91, 92)
        assert cp + pd.Timedelta(days=90) <= c                 # fully observed by the cutoff


def test_features_module_does_not_use_target_labels():
    src = Path(C.PROJECT_DIR / "src" / "apf" / "features.py").read_text()
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    names |= {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {"labels_at", "build_panel", "check_window_complete"}
