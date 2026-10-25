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
from apf.evaluation import (metrics, p_no_purchase, paired_cluster_bootstrap, top_fraction,
                            top_fraction_by_group)
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


def test_top_fraction_by_group_flags_within_each_cutoff():
    # each cutoff has one y = 0 row, ranked first within its cutoff; all of B's scores are below A's
    y = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    s = np.r_[np.linspace(10, 9, 10), np.linspace(5, 4, 10)]
    g = np.array(["A"] * 10 + ["B"] * 10)
    r = top_fraction_by_group(y, s, g, 0.1)
    assert r["flagged"] == 2                                   # one per cutoff
    assert r["precision_y0"] == pytest.approx(1.0) and r["recall_y0"] == pytest.approx(1.0)
    pooled = top_fraction(y, s, 0.1)                           # one ranking flags A's top two
    assert pooled["precision_y0"] == pytest.approx(0.5) and pooled["recall_y0"] == pytest.approx(0.5)


@pytest.mark.parametrize("n,frac,k", [(5000, 0.1, 500), (5000, 0.2, 1000), (3935, 0.1, 394),
                                      (3935, 0.2, 787), (10, 0.1, 1), (7, 0.2, 2)])
def test_top_fraction_k_is_exact_ceiling(n, frac, k):
    y = np.ones(n, int); y[0] = 0
    assert top_fraction(y, np.arange(n, dtype=float), frac)["flagged"] == k


def test_top_fraction_by_group_pools_counts_not_rates():
    # A: 10 rows, k=1, its top row is y=0 (precision 1). B: 30 rows, k=3, one y=0 among its top 3.
    yA = np.r_[0, np.ones(9, int)]
    yB = np.r_[1, 0, 1, np.ones(27, int)]
    y, s = np.r_[yA, yB], np.r_[np.arange(10, 0, -1), np.arange(30, 0, -1)].astype(float)
    g = np.array(["A"] * 10 + ["B"] * 30)
    r = top_fraction_by_group(y, s, g, 0.1)
    assert (r["flagged"], r["tp_expected"], r["events"]) == (4, 2.0, 2)
    assert r["precision_y0"] == pytest.approx(2 / 4)          # pooled; mean of per-cutoff rates = 2/3
    assert r["recall_y0"] == pytest.approx(1.0)


def test_top_fraction_by_group_ties_and_empty_events():
    # A: all 10 scores tied, 2 of them y=0, k=1 -> expected tp 0.2. B: no y=0 at all.
    y = np.r_[0, 0, np.ones(8, int), np.ones(10, int)]
    s = np.r_[np.full(10, 5.0), np.arange(10, dtype=float)]
    g = np.array(["A"] * 10 + ["B"] * 10)
    r = top_fraction_by_group(y, s, g, 0.1)
    assert r["tp_expected"] == pytest.approx(0.2) and r["flagged"] == 2 and r["events"] == 2
    assert r["precision_y0"] == pytest.approx(0.1) and r["recall_y0"] == pytest.approx(0.1)
    perm = np.random.default_rng(1).permutation(20)
    assert top_fraction_by_group(y[perm], s[perm], g[perm], 0.1) == r


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


# ---- class orientation and prevalence (event = y = 0) -------------------------------------

def test_calibration_sign_convention_for_event_y0():
    r = np.random.default_rng(5)
    p0 = r.uniform(0.05, 0.5, 100_000)
    y = np.where(r.uniform(size=p0.size) < p0, 0, 1)            # y = 0 occurs with probability p0
    under = metrics(y, p0 * 0.6)                                # P(y=0) too low -> intercept > 0
    over = metrics(y, np.clip(p0 * 1.5, 0, 1))                  # P(y=0) too high -> intercept < 0
    assert under["cal_intercept"] > 0.2 and over["cal_intercept"] < -0.2
    assert metrics(y, p0)["prevalence_y0"] == pytest.approx((y == 0).mean())


def test_reliability_table_observed_rate_is_y0():
    from apf.evaluation import reliability_table
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1] * 10)
    t = reliability_table(y, np.linspace(0.01, 0.99, 100), n_bins=5)
    assert (t["n"] * t["observed_y0"]).sum() == pytest.approx((y == 0).sum())


def test_logreg_applies_training_imputation_unchanged_to_new_data():
    X, y = synth()
    m = Mo.make_logreg(Mo.FULL_FEATURES, 1.0).fit(X.iloc[:2000], y[:2000])
    train_med = X["median_gap_days"].iloc[:2000].median()
    new = X.iloc[2000:2100].copy()
    new["median_gap_days"] = 999.0                              # new data with a very different median
    a = new.iloc[[0]].assign(median_gap_days=np.nan)            # missing -> must use the TRAIN median
    b = new.iloc[[0]].assign(median_gap_days=train_med)
    prep = m.named_steps["prep"]
    ta, tb = prep.transform(a), prep.transform(b)
    names = list(prep.get_feature_names_out())
    keep = [i for i, n in enumerate(names) if "missingindicator" not in n]
    assert np.allclose(ta[:, keep], tb[:, keep])
    sc = prep.named_transformers_["log"].named_steps["scale"]
    cols = prep.transformers_[0][2]
    assert np.allclose(sc.mean_, np.log1p(X[cols].iloc[:2000]).mean().to_numpy())


def test_no_class_weighting_anywhere():
    assert Mo.make_logreg(Mo.FULL_FEATURES, 0.01).named_steps["lr"].class_weight is None
    p = Mo.make_lgbm_refit({"num_leaves": 7, "min_child_samples": 200}, 114).get_params()
    assert p["class_weight"] is None and "is_unbalance" not in p and "scale_pos_weight" not in p


def test_reliability_table_keeps_ties_together_and_ignores_row_order():
    from apf.evaluation import reliability_table
    r = np.random.default_rng(2)
    p0 = r.choice([0.05, 0.2, 0.6], 3000)                       # coarse score, like persistence
    y = np.where(r.uniform(size=3000) < p0, 0, 1)
    t = reliability_table(y, p0)
    assert len(t) == 3 and sorted(t["mean_p0"].round(6)) == [0.05, 0.2, 0.6]
    perm = r.permutation(3000)
    pd.testing.assert_frame_equal(t, reliability_table(y[perm], p0[perm]))
    cont = reliability_table(y, r.uniform(size=3000))
    assert len(cont) == 10 and cont["n"].between(299, 301).all()


def test_phase1_script_builds_labels_and_features_for_pre_test_cutoffs_only():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "stage_d_phase1.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("build_panel", "build_feature_panel")]
    assert len(calls) == 2 and all(isinstance(c.args[1], ast.Name) and c.args[1].id == "dev" for c in calls)
    assert "dev = [c for c in C.CUTOFFS if c < min(TEST_CUTOFFS)]" in src
    assert "rows_before_test(tab)" in src and "release_test_rows" not in src
