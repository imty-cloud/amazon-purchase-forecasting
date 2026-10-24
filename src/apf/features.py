"""Respondent-level features at a cutoff, from purchases dated on or before the cutoff only.

Trailing windows of W days end on the cutoff day and include it: days c-W+1 .. c.
All features are computed from `history_until(..., c)`; nothing after c is ever read.
Definitions, lookbacks, missing values and leakage notes: FEATURES below and
reports/stage_c_features_validation.md.

The RFM, spend-per-day, category and median-interval logic follows rfm_v3/rfm_study_v3.ipynb
cell 6 (`make_store`, `features_at`), with Tenure redefined as c - first purchase and trailing
windows added.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CATEGORY, DATE, ID, START
from .target import OUTCOME_COLUMNS, assert_no_survey_columns, history_until

WINDOWS = (30, 90, 180, 365)

# name: (definition, lookback, missing values, leakage note)
FEATURES = {
    "recency_days": ("days from the last purchase day to the cutoff (0 = bought on the cutoff day)",
                     "all history", "never missing",
        "uses the last day <= c only; tested at the cutoff day and day c+1"),
    "frequency_days": ("distinct purchase days", "all history", "never missing",
        "counts days <= c; duplicates collapse to one day"),
    "monetary_total": ("sum of price x quantity", "all history", "never missing",
        "includes gift cards and exact-duplicate rows (primary policy)"),
    "spend_per_purchase_day": ("monetary_total / frequency_days (proxy for order value: no order IDs)",
                               "all history", "never missing",
        "not an order value: a day can hold several orders"),
    "days_30": ("distinct purchase days in the trailing window", "30 d", "0 if none",
        "window c-29 .. c; edges tested"),
    "days_90": ("distinct purchase days in the trailing window", "90 d", "0 if none",
        "window c-89 .. c; edges tested"),
    "days_180": ("distinct purchase days in the trailing window", "180 d", "0 if none",
        "window c-179 .. c; edges tested"),
    "days_365": ("distinct purchase days in the trailing window", "365 d", "0 if none",
        "window c-364 .. c; edges tested"),
    "spend_90": ("sum of price x quantity in the trailing window", "90 d", "0 if none",
        "window c-89 .. c"),
    "spend_365": ("sum of price x quantity in the trailing window", "365 d", "0 if none",
        "window c-364 .. c"),
    "categories_total": ("distinct non-missing categories", "all history", "0 if all missing",
        "category first seen after c is not counted"),
    "categories_365": ("distinct non-missing categories in the trailing window", "365 d", "0 if none",
        "window c-364 .. c"),
    "tenure_days": ("days from the first observed purchase (>= 2018-01-01, left-truncated) to the cutoff",
                    "all history", "never missing",
        "left-truncated at 2018-01-01: observed span, not account age"),
    "median_gap_days": ("median gap between consecutive purchase days", "all history",
                        "NaN if frequency_days = 1",
        "gaps between days <= c only"),
    "recency_to_median_gap": ("recency_days / median_gap_days", "all history",
                              "NaN if frequency_days = 1",
        "both terms use days <= c"),
    "active_months_12": ("calendar months with >= 1 purchase among the 12 months ending with the cutoff month",
                         "12 months", "0 if none",
        "cutoffs are month-ends, so the 12 months end on c"),
    "days_same_window_last_year": ("distinct purchase days in c-364 .. c-275 (the target window shifted back 365 d)",
                                   "c-364 .. c-275", "0 if none",
        "ends 275 days before c"),
    "prev_90d_active": ("1 if >= 1 purchase day in the previous quarter-end's 90-day target window "
                            "(c_prev+1 .. c_prev+90, which ends on or before c)",
                            "c_prev+1 .. c_prev+90", "NaN if no purchase on or before c_prev",
        "previous label window ends <= c (asserted); equals the previous cutoff's y"),
    "cutoff_quarter": ("calendar quarter of the cutoff (1-4)", "-", "never missing",
        "calendar fact known in advance; it can encode period effects, not leakage"),
}
FEATURE_COLUMNS = list(FEATURES)


@dataclass
class DailyStore:
    """Pre-aggregated purchase history (no column refers to any particular cutoff)."""
    days: pd.DataFrame        # one row per (ID, day): spend
    cat_days: pd.DataFrame    # one row per (ID, Category, day), non-missing category only


def make_daily_store(tx):
    t = tx[tx[DATE] >= START]
    days = (t.groupby([ID, DATE], observed=True)["spend"].sum().reset_index())
    days[ID] = days[ID].astype(str)
    # gap computations rely on (ID, day) order
    days = days.sort_values([ID, DATE], kind="mergesort").reset_index(drop=True)
    cd = (t.loc[t[CATEGORY].notna(), [ID, CATEGORY, DATE]].astype({ID: str, CATEGORY: str})
          .drop_duplicates().reset_index(drop=True))
    return DailyStore(days=days, cat_days=cd)


def previous_quarter_end(c):
    return pd.Timestamp(c) - pd.offsets.QuarterEnd(1)


def _count_in(h, lo, hi, ids):
    """Distinct purchase days with lo <= day <= hi, per respondent (h has one row per day)."""
    w = h[(h[DATE] >= lo) & (h[DATE] <= hi)]
    return w.groupby(ID).size().reindex(ids, fill_value=0).astype(float)


def features_at(store, cutoff):
    c = pd.Timestamp(cutoff)
    h = history_until(store.days, c)
    g = h.groupby(ID, sort=True)
    ids = pd.Index(g.size().index, name=ID)
    first, last = g[DATE].min(), g[DATE].max()
    F = pd.DataFrame(index=ids)
    F["recency_days"] = (c - last).dt.days.astype(float)
    F["frequency_days"] = g.size().astype(float)
    F["monetary_total"] = g["spend"].sum()
    F["spend_per_purchase_day"] = F["monetary_total"] / F["frequency_days"]
    for w in WINDOWS:
        F[f"days_{w}"] = _count_in(h, c - pd.Timedelta(days=w - 1), c, ids)
    for w in (90, 365):
        hw = h[h[DATE] >= c - pd.Timedelta(days=w - 1)]
        F[f"spend_{w}"] = hw.groupby(ID)["spend"].sum().reindex(ids, fill_value=0.0)
    cd = history_until(store.cat_days, c)
    F["categories_total"] = cd.groupby(ID)[CATEGORY].nunique().reindex(ids, fill_value=0).astype(float)
    cd365 = cd[cd[DATE] >= c - pd.Timedelta(days=364)]
    F["categories_365"] = cd365.groupby(ID)[CATEGORY].nunique().reindex(ids, fill_value=0).astype(float)
    F["tenure_days"] = (c - first).dt.days.astype(float)
    gaps = h[DATE].diff().dt.days.where(h[ID].eq(h[ID].shift()))
    F["median_gap_days"] = gaps.groupby(h[ID]).median().reindex(ids)
    F["recency_to_median_gap"] = F["recency_days"] / F["median_gap_days"]
    m = h[DATE].dt.to_period("M")
    cm = c.to_period("M")
    F["active_months_12"] = (h[(m > cm - 12)].assign(_m=m).groupby(ID)["_m"].nunique()
                             .reindex(ids, fill_value=0).astype(float))
    F["days_same_window_last_year"] = _count_in(h, c - pd.Timedelta(days=364),
                                                c - pd.Timedelta(days=275), ids)
    cp = previous_quarter_end(c)
    prev_end = cp + pd.Timedelta(days=90)
    assert prev_end <= c, "previous target window must end on or before the cutoff"
    active = _count_in(h, cp + pd.Timedelta(days=1), prev_end, ids) > 0
    F["prev_90d_active"] = active.astype(float).where(first <= cp)
    F["cutoff_quarter"] = float(c.quarter)
    F = F[FEATURE_COLUMNS]
    check_feature_frame(F)
    return F


def check_feature_frame(F):
    assert_no_survey_columns(F.columns)
    bad = set(F.columns) & set(OUTCOME_COLUMNS)
    if bad:
        raise ValueError(f"outcome columns in a feature frame: {bad}")
    if not F.index.is_unique:
        raise ValueError("one row per respondent expected")


def build_feature_panel(store, cutoffs):
    parts = []
    for c in sorted(pd.Timestamp(x) for x in cutoffs):
        F = features_at(store, c).reset_index()
        F.insert(1, "cutoff", c)
        parts.append(F)
    P = pd.concat(parts, ignore_index=True)
    if P.duplicated([ID, "cutoff"]).any():
        raise AssertionError("respondent-cutoff pairs are not unique")
    return P
