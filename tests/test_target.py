"""Target window, cutoff eligibility, uniqueness and leakage invariants (synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from apf import config as C
from apf.target import (IncompleteWindowError, assert_no_survey_columns, build_panel,
                        check_window_complete, eligible_respondents, history_until, labels_at,
                        purchase_days, window)

ID, DATE = C.ID, C.DATE
CUT = pd.Timestamp("2020-06-30")


def tx(rows):
    """rows: [(id, 'YYYY-MM-DD'), ...] -> minimal transaction frame."""
    return pd.DataFrame({ID: [r[0] for r in rows], DATE: pd.to_datetime([r[1] for r in rows])})


def label(rows, cutoff=CUT):
    lab = labels_at(purchase_days(tx(rows)), cutoff)
    return dict(zip(lab[ID], lab["y"]))


# ---- window boundaries -------------------------------------------------------------------

def test_window_is_days_c_plus_1_to_c_plus_90():
    w0, w1 = window(CUT)
    assert (w0, w1) == (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-09-28"))
    assert (w1 - w0).days + 1 == 90


@pytest.mark.parametrize("offset,expected", [(0, 0), (1, 1), (45, 1), (90, 1), (91, 0)])
def test_purchase_at_boundary(offset, expected):
    day = (CUT + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
    assert label([("R_a", "2019-01-01"), ("R_a", day)]) == {"R_a": expected}


def test_purchase_on_cutoff_day_is_history_not_target():
    lab = label([("R_a", "2020-06-30")])
    assert lab == {"R_a": 0}


# ---- cutoff eligibility --------------------------------------------------------------------

def test_eligibility_requires_history_on_or_before_cutoff():
    lab = label([("R_on", "2020-06-30"), ("R_after", "2020-07-01"), ("R_before", "2018-03-01"),
                 ("R_after", "2020-07-02")])
    assert set(lab) == {"R_on", "R_before"}           # first purchase after c: not eligible


def test_rows_before_start_are_not_history():
    lab = label([("R_old", "2017-12-31"), ("R_old", "2020-07-05")])
    assert lab == {}


def test_inactive_for_years_is_still_eligible_and_negative():
    assert label([("R_a", "2018-01-01")]) == {"R_a": 0}


# ---- incomplete windows ----------------------------------------------------------------------

def test_latest_admissible_cutoff():
    check_window_complete(pd.Timestamp("2022-08-02"))
    with pytest.raises(IncompleteWindowError):
        check_window_complete(pd.Timestamp("2022-08-03"))


def test_incomplete_window_produces_no_labels_rather_than_negatives():
    d = purchase_days(tx([("R_a", "2019-01-01")]))
    with pytest.raises(IncompleteWindowError):
        labels_at(d, pd.Timestamp("2022-09-30"))
    with pytest.raises(IncompleteWindowError):
        build_panel(d, [pd.Timestamp("2022-06-30"), pd.Timestamp("2022-09-30")])


def test_configured_cutoffs():
    cs = C.CUTOFFS
    assert len(cs) == 15 and cs[0] == pd.Timestamp("2018-12-31") and cs[-1] == pd.Timestamp("2022-06-30")
    assert all(c.is_quarter_end for c in cs)
    assert all(c + pd.Timedelta(days=C.HORIZON_DAYS) <= C.OBSERVED_THROUGH for c in cs)
    assert all((c - C.START).days >= 364 for c in cs)       # >= 12 months of history
    for a, b in zip(cs[:-1], cs[1:]):                       # target windows never overlap
        assert window(a)[1] < window(b)[0]


# ---- uniqueness ----------------------------------------------------------------------------

def test_repeat_rows_and_same_day_rows_count_as_one_purchase_day():
    rows = [("R_a", "2019-01-01")] + [("R_a", "2020-07-03")] * 3 + [("R_a", "2020-07-04")]
    lab = labels_at(purchase_days(tx(rows)), CUT)
    assert len(lab) == 1 and lab["target_purchase_days"].item() == 2


def test_same_day_rows_differing_in_other_columns_are_one_purchase_day():
    t = tx([("R_a", "2019-01-01"), ("R_a", "2020-07-03"), ("R_a", "2020-07-03")]).assign(
        Category=["PEN", "PEN", "BOOK"], spend=[1.0, 2.0, 3.0])
    d = purchase_days(t)
    assert list(d.columns) == [ID, DATE] and len(d) == 2
    assert labels_at(d, CUT)["target_purchase_days"].item() == 1


def test_panel_has_one_row_per_respondent_cutoff():
    rows = [("R_a", "2019-01-01"), ("R_a", "2019-01-01"), ("R_b", "2019-05-05"), ("R_b", "2020-02-02")]
    p = build_panel(purchase_days(tx(rows)), C.CUTOFFS)
    assert not p.duplicated([ID, "cutoff"]).any()
    # R_a first buys 2019-01-01 (eligible from 2019-03-31: 14 cutoffs); R_b first buys
    # 2019-05-05 (eligible from 2019-06-30: 13 cutoffs)
    assert p.groupby(ID).size().to_dict() == {"R_a": 14, "R_b": 13}


def test_panel_columns_are_keys_window_and_outcome_only():
    t = tx([("R_a", "2019-01-01")]).assign(**{"Q-demos-age": "35 - 44 years", "Category": "PEN"})
    p = build_panel(purchase_days(t), C.CUTOFFS)
    assert list(p.columns) == [ID, "cutoff", "window_start", "window_end", "target_purchase_days", "y"]


def test_survey_columns_rejected():
    with pytest.raises(ValueError):
        assert_no_survey_columns(["Recency", "Q-demos-age"])
    assert_no_survey_columns(["Recency", "y"])


# ---- leakage-sensitive construction --------------------------------------------------------

def test_history_until_never_returns_future_rows():
    d = purchase_days(tx([("R_a", "2020-06-30"), ("R_a", "2020-07-01"), ("R_b", "2021-01-01")]))
    h = history_until(d, CUT)
    assert h[DATE].max() == CUT and set(h[ID]) == {"R_a"}


def _random_tx(seed, n_ids=60, n_rows=1500):
    r = np.random.default_rng(seed)
    ids = [f"R_{i:015d}" for i in range(n_ids)]
    dates = C.START + pd.to_timedelta(r.integers(-30, 1800, n_rows), unit="D")
    return pd.DataFrame({ID: r.choice(ids, n_rows), DATE: dates})


@pytest.mark.parametrize("seed", range(5))
def test_eligibility_independent_of_post_cutoff_data(seed):
    t = _random_tx(seed)
    d = purchase_days(t)
    for c in C.CUTOFFS:
        future_changed = purchase_days(pd.concat([t[t[DATE] <= c], _random_tx(seed + 100)
                                                  .assign(**{DATE: lambda x: x[DATE] + pd.Timedelta(days=2000)})]))
        assert eligible_respondents(d, c).equals(eligible_respondents(future_changed, c))


@pytest.mark.parametrize("seed", range(5))
def test_labels_unchanged_by_data_after_window(seed):
    t = _random_tx(seed)
    for c in C.CUTOFFS:
        end = window(c)[1]
        base = labels_at(purchase_days(t), c)
        cut_off = labels_at(purchase_days(t[t[DATE] <= end]), c)
        noise = t.assign(**{DATE: end + pd.Timedelta(days=1)})
        added = labels_at(purchase_days(pd.concat([t, noise])), c)
        pd.testing.assert_frame_equal(base, cut_off)
        pd.testing.assert_frame_equal(base, added)


@pytest.mark.parametrize("seed", range(5))
def test_panel_matches_brute_force_oracle(seed):
    t = _random_tx(seed)
    p = build_panel(purchase_days(t), C.CUTOFFS)
    got = {(i, c): y for i, c, y in zip(p[ID], p["cutoff"], p["y"])}
    want = {}
    for i, g in t.groupby(ID):
        ds = [x for x in g[DATE] if x >= C.START]
        for c in C.CUTOFFS:
            if any(x <= c for x in ds):
                want[(i, c)] = int(any(c < x <= c + pd.Timedelta(days=90) for x in ds))
    assert got == want
