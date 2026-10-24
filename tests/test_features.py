"""Feature lookback boundaries, cutoff-day inclusion, missing history and leakage (synthetic data)."""
import statistics

import numpy as np
import pandas as pd
import pytest

from apf import config as C
from apf.features import (FEATURE_COLUMNS, build_feature_panel, check_feature_frame, features_at,
                          make_daily_store)
from apf.target import eligible_respondents, purchase_days

ID, DATE, CAT = C.ID, C.DATE, C.CATEGORY
CUT = pd.Timestamp("2020-06-30")


def tx(rows):
    """rows: [(id, date, spend, category), ...]"""
    return pd.DataFrame({ID: [r[0] for r in rows], DATE: pd.to_datetime([r[1] for r in rows]),
                         "spend": [float(r[2]) for r in rows], CAT: [r[3] for r in rows]})


def feat(rows, cutoff=CUT):
    return features_at(make_daily_store(tx(rows)), cutoff)


def d(offset, base=CUT):
    return (base + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")


OLD = ("2018-02-01", 1.0, "X")   # an old purchase so the respondent is eligible


# ---- cutoff day and trailing-window boundaries --------------------------------------------

def test_purchase_on_cutoff_day_is_included():
    F = feat([("R_a", *OLD), ("R_a", d(0), 5.0, "PEN")])
    r = F.loc["R_a"]
    assert r.recency_days == 0 and r.days_30 == 1 and r.days_365 == 1 and r.spend_90 == 5.0
    assert r.categories_365 == 1


def test_purchase_after_cutoff_is_excluded():
    F = feat([("R_a", *OLD), ("R_a", d(1), 5.0, "PEN")])
    r = F.loc["R_a"]
    assert r.frequency_days == 1 and r.monetary_total == 1.0 and r.categories_total == 1


@pytest.mark.parametrize("w", [30, 90, 180, 365])
def test_trailing_window_edges(w):
    inside = feat([("R_a", *OLD), ("R_a", d(-(w - 1)), 2.0, "PEN")]).loc["R_a"]
    outside = feat([("R_a", *OLD), ("R_a", d(-w), 2.0, "PEN")]).loc["R_a"]
    assert inside[f"days_{w}"] == 1 and outside[f"days_{w}"] == 0
    if w in (90, 365):
        assert inside[f"spend_{w}"] == 2.0 and outside[f"spend_{w}"] == 0.0
    if w == 365:
        assert inside.categories_365 == 1 and outside.categories_365 == 0


def test_same_window_last_year_edges():
    rows = [("R_a", *OLD)] + [("R_a", d(o), 1.0, None) for o in (-365, -364, -275, -274)]
    assert feat(rows).loc["R_a"].days_same_window_last_year == 2


def test_active_months_12_edges():
    rows = [("R_a", "2019-06-30", 1, None), ("R_a", "2019-07-01", 1, None),
            ("R_a", "2019-07-20", 1, None), ("R_a", "2020-06-30", 1, None)]
    assert feat(rows).loc["R_a"].active_months_12 == 2       # 2019-07 and 2020-06


def test_prev_90d_active():
    # c = 2020-06-30, previous quarter-end 2020-03-31, window 2020-04-01 .. 2020-06-29
    F = feat([("R_in", "2020-01-05", 1, None), ("R_in", "2020-06-29", 1, None),
              ("R_cut", "2020-01-05", 1, None), ("R_cut", "2020-06-30", 1, None),
              ("R_start", "2020-01-05", 1, None), ("R_start", "2020-04-01", 1, None),
              ("R_new", "2020-04-15", 1, None)])
    assert F.loc["R_in"].prev_90d_active == 1
    assert F.loc["R_cut"].prev_90d_active == 0           # the cutoff day is not in that window
    assert F.loc["R_start"].prev_90d_active == 1
    assert np.isnan(F.loc["R_new"].prev_90d_active)      # no history at 2020-03-31


# ---- values, missing history ----------------------------------------------------------------

def test_rfm_gaps_and_spend_aggregation():
    rows = [("R_a", "2020-06-01", 2.0, "A"), ("R_a", "2020-06-01", 3.0, "B"),
            ("R_a", "2020-06-03", 1.0, None), ("R_a", "2020-06-10", 4.0, "A")]
    r = feat(rows).loc["R_a"]
    assert r.frequency_days == 3 and r.monetary_total == 10.0
    assert r.spend_per_purchase_day == pytest.approx(10 / 3)
    assert r.median_gap_days == 4.5                          # gaps 2 and 7
    assert r.recency_days == 20 and r.recency_to_median_gap == pytest.approx(20 / 4.5)
    assert r.categories_total == 2 and r.tenure_days == 29


def test_missing_history():
    F = feat([("R_one", "2019-05-05", 1, None), ("R_future", "2020-07-01", 1, "A")])
    assert list(F.index) == ["R_one"]                        # no history: no row
    r = F.loc["R_one"]
    assert np.isnan(r.median_gap_days) and np.isnan(r.recency_to_median_gap)
    assert r.categories_total == 0 and r.days_365 == 0
    assert F.drop(columns=["median_gap_days", "recency_to_median_gap"]).notna().all().all()


def test_rows_before_start_ignored():
    F = feat([("R_a", "2017-12-31", 100, "A"), ("R_a", "2019-01-01", 1, "B")])
    assert F.loc["R_a"].monetary_total == 1 and F.loc["R_a"].categories_total == 1


def test_population_equals_target_eligibility():
    t = _random_tx(3)
    s = make_daily_store(t)
    for c in C.CUTOFFS:
        assert features_at(s, c).index.equals(eligible_respondents(purchase_days(t), c))


def test_frame_guard_rejects_outcome_and_survey_columns():
    F = feat([("R_a", *OLD)])
    for bad in ("y", "target_purchase_days", "Q-amazon-use-how-oft"):
        with pytest.raises(ValueError):
            check_feature_frame(F.assign(**{bad: 0}))
    assert list(F.columns) == FEATURE_COLUMNS


# ---- leakage: randomised perturbation and brute-force oracle ---------------------------------

def _random_tx(seed, n_ids=40, n_rows=1500):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        ID: r.choice([f"R_{i:015d}" for i in range(n_ids)], n_rows),
        DATE: C.START + pd.to_timedelta(r.integers(-20, 1760, n_rows), unit="D"),
        "spend": r.gamma(2.0, 10.0, n_rows).round(2),
        CAT: pd.Series(r.choice(["A", "B", "C", "D", None], n_rows), dtype=object)})


@pytest.mark.parametrize("seed", range(4))
def test_features_unchanged_by_post_cutoff_rows(seed):
    t = _random_tx(seed)
    for c in C.CUTOFFS[::3]:
        base = features_at(make_daily_store(t), c)
        trunc = features_at(make_daily_store(t[t[DATE] <= c]), c)
        noise = _random_tx(seed + 50).assign(**{DATE: c + pd.Timedelta(days=1)})
        added = features_at(make_daily_store(pd.concat([t, noise])), c)
        pd.testing.assert_frame_equal(base, trunc)
        pd.testing.assert_frame_equal(base, added)


def oracle(t, c):
    """Straight-line reimplementation of every feature for one cutoff."""
    out = {}
    c = pd.Timestamp(c)
    for i, g in t[(t[DATE] >= C.START) & (t[DATE] <= c)].groupby(ID):
        spend_by_day = g.groupby(DATE)["spend"].sum()
        ds = sorted(spend_by_day.index)
        within = lambda lo, hi: [x for x in ds if lo <= (c - x).days <= hi]   # days before c
        cats = g.dropna(subset=[CAT])
        gaps = [(b - a).days for a, b in zip(ds, ds[1:])]
        med = statistics.median(gaps) if gaps else np.nan
        rec = (c - ds[-1]).days
        cp = c - pd.offsets.QuarterEnd(1)
        months = {(x.year, x.month) for x in ds}
        cm = c.year * 12 + c.month
        out[i] = {
            "recency_days": rec, "frequency_days": len(ds), "monetary_total": spend_by_day.sum(),
            "spend_per_purchase_day": spend_by_day.sum() / len(ds),
            **{f"days_{w}": len(within(0, w - 1)) for w in (30, 90, 180, 365)},
            **{f"spend_{w}": sum(spend_by_day[x] for x in within(0, w - 1)) for w in (90, 365)},
            "categories_total": cats[CAT].nunique(),
            "categories_365": cats.loc[(c - cats[DATE]).dt.days <= 364, CAT].nunique(),
            "tenure_days": (c - ds[0]).days, "median_gap_days": med,
            "recency_to_median_gap": rec / med if gaps else np.nan,
            "active_months_12": sum(1 for (y, m) in months if 0 <= cm - (y * 12 + m) <= 11),
            "days_same_window_last_year": len(within(275, 364)),
            "prev_90d_active": (float(any(cp < x <= cp + pd.Timedelta(days=90) for x in ds))
                                    if ds[0] <= cp else np.nan),
            "cutoff_quarter": c.quarter}
    return pd.DataFrame.from_dict(out, orient="index").astype(float)[FEATURE_COLUMNS]


@pytest.mark.parametrize("seed", range(3))
def test_features_match_oracle(seed):
    t = _random_tx(seed)
    s = make_daily_store(t)
    for c in C.CUTOFFS:
        got = features_at(s, c)
        want = oracle(t, c).rename_axis(ID)
        pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-9, atol=1e-9)


def test_feature_panel_keys_unique():
    P = build_feature_panel(make_daily_store(_random_tx(1)), C.CUTOFFS)
    assert not P.duplicated([ID, "cutoff"]).any()
