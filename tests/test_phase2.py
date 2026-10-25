"""Phase 2 workflow on synthetic tables only: lock-first ordering, run-once guard, refit rows,
locked iteration counts, metric tables, paired comparisons, determinism."""
import json

import numpy as np
import pandas as pd
import pytest

from apf import config as C
from apf import phase2 as P2
from apf.features import FEATURE_COLUMNS
from apf.lock import LockError, rows_before_test, sha256
from apf.splits import TEST_CUTOFFS, assign, holdout_respondents

CONFIG = {"M1_lr_rfm": {"feature_set": "rfm", "C": 0.01}, "M2_lr_full": {"feature_set": "full", "C": 0.01},
          "S1_lr_no_prev": {"feature_set": "no_prev", "C": 0.01},
          "M3_lgbm_full": {"feature_set": "full", "num_leaves": 7, "min_child_samples": 50, "best_iteration": 37},
          "S2_lgbm_no_prev": {"feature_set": "no_prev", "num_leaves": 7, "min_child_samples": 50,
                              "best_iteration": 37}}


def synth_table(n_ids=300, seed=0):
    r = np.random.default_rng(seed)
    ids = [f"R_{i:015d}" for i in range(n_ids)]
    t = pd.DataFrame([(i, c) for i in ids for c in C.CUTOFFS], columns=[C.ID, "cutoff"])
    for f in FEATURE_COLUMNS:
        t[f] = r.gamma(2, 10, len(t))
    t["cutoff_quarter"] = t["cutoff"].dt.quarter.astype(float)
    t["prev_90d_active"] = r.integers(0, 2, len(t)).astype(float)
    t.loc[r.uniform(size=len(t)) < 0.05, ["median_gap_days", "recency_to_median_gap"]] = np.nan
    p0 = 1 / (1 + np.exp(-(-3 + 0.06 * t["recency_days"])))
    t["y"] = (r.uniform(size=len(t)) >= p0).astype(int)
    return assign(t, holdout_respondents(ids))


@pytest.fixture(scope="module")
def result():
    t = synth_table()
    return P2.run_phase2(rows_before_test(t), t[t["cutoff"].isin(TEST_CUTOFFS)], CONFIG, B=20, seed=1), t


# ---- guards: nothing is loaded before the lock passes ----------------------------------------

def _never_load():
    raise AssertionError("data loaded before the guards passed")


def test_main_checks_lock_before_loading(monkeypatch, tmp_path):
    def no_lock():
        raise LockError("no lock")
    monkeypatch.setattr(P2, "require_lock", no_lock)
    with pytest.raises(LockError):
        P2.main(out_dir=tmp_path, loader=_never_load)


def _fake_lock(monkeypatch, tmp_path, sel_sha=None):
    sel = tmp_path / "stage_d_phase1_selection.json"
    sel.write_text("{}")
    payload = {"protocol_sha256": "x", "selected": {
        "phase1_selection_file": sel.name, "phase1_selection_sha256": sel_sha or sha256(sel),
        "configurations": CONFIG, "seeds": {"bootstrap": 3}, "phase1_commit": "abc"}}
    monkeypatch.setattr(P2, "require_lock", lambda: payload)
    monkeypatch.setattr(P2, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(P2, "release_test_rows", lambda t: (t[t["cutoff"].isin(TEST_CUTOFFS)], payload))
    return payload


def test_main_refuses_if_phase1_selection_changed(monkeypatch, tmp_path):
    _fake_lock(monkeypatch, tmp_path, sel_sha="0" * 64)
    with pytest.raises(LockError, match="differs"):
        P2.main(out_dir=tmp_path / "out", loader=_never_load)


def test_main_refuses_if_results_exist(monkeypatch, tmp_path):
    _fake_lock(monkeypatch, tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / P2.RESULT_FILE).write_text("x")
    with pytest.raises(LockError, match="already"):
        P2.main(out_dir=tmp_path / "out", loader=_never_load)


def test_main_end_to_end_then_refuses_second_run(monkeypatch, tmp_path):
    _fake_lock(monkeypatch, tmp_path)
    out = tmp_path / "out"
    P2.main(out_dir=out, B=10, loader=lambda: synth_table(120, 3))
    for f in (P2.RESULT_FILE, "stage_d_comparisons.csv", "stage_d_temporal_vs_holdout.csv",
              "stage_d_reliability.csv", "stage_d_bootstrap_draws.csv.gz", "stage_d_predictions.csv.gz",
              "stage_d_phase2_meta.json", "stage_d_test_metric_cis.csv"):
        assert (out / f).exists(), f
    assert json.loads((out / "stage_d_phase2_meta.json").read_text())["B"] == 10
    with pytest.raises(LockError, match="already"):
        P2.main(out_dir=out, loader=_never_load)


# ---- the computation ------------------------------------------------------------------------

def test_refit_rows_and_locked_iterations(result):
    res, t = result
    meta = res["meta"]
    n_refit = int((t["in_refit"]).sum())
    n_train = int((t["in_train"]).sum())
    assert set(meta["fit_rows"]["temporal_test"].values()) == {n_refit}
    assert set(meta["fit_rows"]["holdout_test"].values()) == {n_refit}
    assert set(meta["fit_rows"]["holdout_train_period"].values()) == {n_train}
    for s in meta["lightgbm_trees"].values():
        assert s == {"M3_lgbm_full": 37, "S2_lgbm_no_prev": 37}


def test_evaluation_sets_and_metric_table(result):
    res, t = result
    m = res["metrics"]
    pooled = m[m["scope"] == "pooled"]
    assert len(pooled) == 3 * 8 and set(pooled["set"]) == set(P2.SETS)
    rows = res["meta"]["rows"]
    assert rows["temporal_test"] == int(t["in_test_temporal"].sum())
    assert rows["holdout_test"] == int(t["in_test_holdout"].sum())
    pr = res["predictions"]
    held = set(t.loc[t["holdout"], C.ID])
    assert not pr.loc[pr["set"] == "temporal_test", C.ID].isin(held).any()
    assert pr.loc[pr["set"] == "holdout_test", C.ID].isin(held).all()
    assert set(m.loc[m["set"] != "holdout_train_period", "scope"]) == {"pooled", "2022-03-31", "2022-06-30"}
    for col in ("ap_y0", "ap_no_skill", "roc_auc", "brier", "brier_skill",
                "precision_top10", "recall_top10", "precision_top20", "recall_top20"):
        assert pooled[col].notna().all(), col
    # calibration slope/intercept are undefined for the constant prevalence baseline only
    const = pooled["model"] == "B0_prevalence"
    assert pooled.loc[const, ["cal_slope", "cal_intercept"]].isna().all().all()
    assert pooled.loc[~const, ["cal_slope", "cal_intercept"]].notna().all().all()


def test_comparisons_are_paired_and_predefined(result):
    res, _ = result
    c, d = res["comparisons"], res["draws"]
    assert len(c) == 2 * 7 * 3 and set(c.loc[c["role"] == "primary", "comparison"]) == {"C1"}
    row = c[(c["set"] == "temporal_test") & (c["comparison"] == "C1") & (c["metric"] == "ap_y0")].iloc[0]
    w = d[(d["set"] == "temporal_test") & (d["metric"] == "ap_y0")].pivot(index="draw", columns="model", values="value")
    diff = w["M3_lgbm_full"] - w["M2_lr_full"]
    assert row["ci_low"] == pytest.approx(np.percentile(diff, 2.5))
    pooled = res["metrics"].query("set == 'temporal_test' and scope == 'pooled'").set_index("model")["ap_y0"]
    assert row["estimate"] == pytest.approx(pooled["M3_lgbm_full"] - pooled["M2_lr_full"])
    assert d.groupby(["set", "model", "metric"]).size().eq(20).all()


def test_run_phase2_rejects_holdout_in_refit():
    t = synth_table(100, 5)
    dev = rows_before_test(t).copy()
    dev.loc[dev.index[dev["holdout"]][0], "in_refit"] = True
    with pytest.raises(AssertionError, match="hold-out"):
        P2.run_phase2(dev, t[t["cutoff"].isin(TEST_CUTOFFS)], CONFIG, B=5)


def test_run_phase2_deterministic():
    t = synth_table(120, 7)
    a = P2.run_phase2(rows_before_test(t), t[t["cutoff"].isin(TEST_CUTOFFS)], CONFIG, B=10, seed=2)
    b = P2.run_phase2(rows_before_test(t), t[t["cutoff"].isin(TEST_CUTOFFS)], CONFIG, B=10, seed=2)
    pd.testing.assert_frame_equal(a["metrics"], b["metrics"])
    pd.testing.assert_frame_equal(a["comparisons"], b["comparisons"])
