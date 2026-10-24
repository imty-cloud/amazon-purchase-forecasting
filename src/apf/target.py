"""Respondent-cutoff panel: eligibility and the 90-day purchase target.

Definitions (dates are calendar days; `Order Date` carries no time of day):
  history at cutoff c   = purchase days d with START <= d <= c
  eligible at c         = >= 1 purchase day in the history at c
  target window at c    = (c, c + H], i.e. days c+1 .. c+H inclusive, H = 90
  y = 1                 = >= 1 eligible purchase day in the target window, else 0
  a window is usable    only if c + H <= OBSERVED_THROUGH (otherwise no label is produced)

Adapted from rfm_v3/rfm_study_v3.ipynb cell 6 (`make_store`, `features_at` population rule,
`target_after`): binary target, 90-day horizon, explicit completeness guard.
"""
import numpy as np
import pandas as pd

from .config import DATE, HORIZON_DAYS, ID, OBSERVED_THROUGH, START, SURVEY_COLUMN_PREFIX

OUTCOME_COLUMNS = ("y", "target_purchase_days")   # outcome information: never a feature


class IncompleteWindowError(ValueError):
    pass


def assert_no_survey_columns(columns):
    bad = [c for c in columns if str(c).startswith(SURVEY_COLUMN_PREFIX)]
    if bad:
        raise ValueError(f"survey variables (collected after every cutoff) present: {bad}")


def window(cutoff, horizon=HORIZON_DAYS):
    """First and last day of the target window (both inclusive)."""
    c = pd.Timestamp(cutoff)
    return c + pd.Timedelta(days=1), c + pd.Timedelta(days=horizon)


def check_window_complete(cutoff, horizon=HORIZON_DAYS, observed_through=OBSERVED_THROUGH):
    end = window(cutoff, horizon)[1]
    if end > observed_through:
        raise IncompleteWindowError(
            f"window of cutoff {pd.Timestamp(cutoff).date()} ends {end.date()}, after the last "
            f"fully observed day {observed_through.date()}: no label can be produced")
    if pd.Timestamp(cutoff) < START:
        raise ValueError("cutoff before the start of the data")


def purchase_days(tx):
    """One row per (respondent, purchase day) with day >= START."""
    t = tx.loc[tx[DATE] >= START, [ID, DATE]]
    days = t.drop_duplicates().sort_values([ID, DATE], kind="mergesort").reset_index(drop=True)
    days[ID] = days[ID].astype(str)
    return days


def history_until(days, cutoff):
    """The only access path to pre-cutoff behaviour: rows with day <= cutoff."""
    h = days[days[DATE] <= pd.Timestamp(cutoff)]
    assert h.empty or h[DATE].max() <= pd.Timestamp(cutoff)
    return h


def eligible_respondents(days, cutoff):
    return pd.Index(np.sort(history_until(days, cutoff)[ID].unique()), name=ID)


def labels_at(days, cutoff, horizon=HORIZON_DAYS, observed_through=OBSERVED_THROUGH):
    check_window_complete(cutoff, horizon, observed_through)
    ids = eligible_respondents(days, cutoff)
    w0, w1 = window(cutoff, horizon)
    win = days[(days[DATE] >= w0) & (days[DATE] <= w1)]
    n = win.groupby(ID).size().reindex(ids, fill_value=0).astype(int)
    return pd.DataFrame({ID: ids, "cutoff": pd.Timestamp(cutoff), "window_start": w0,
                         "window_end": w1, "target_purchase_days": n.to_numpy(),
                         "y": (n.to_numpy() > 0).astype(np.int8)})


def build_panel(days, cutoffs, horizon=HORIZON_DAYS, observed_through=OBSERVED_THROUGH):
    cutoffs = sorted(pd.Timestamp(c) for c in cutoffs)
    panel = pd.concat([labels_at(days, c, horizon, observed_through) for c in cutoffs],
                      ignore_index=True)
    if panel.duplicated([ID, "cutoff"]).any():
        raise AssertionError("respondent-cutoff pairs are not unique")
    return panel
