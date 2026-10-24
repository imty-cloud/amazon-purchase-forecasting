"""Checks on the real CSV (read in place). Skipped when the file is not available."""
import os

import pandas as pd
import pytest

from apf import config as C
from apf.data import load_transactions, md5sum
from apf.target import build_panel, purchase_days

pytestmark = [pytest.mark.realdata,
              pytest.mark.skipif(not C.PURCHASES_CSV.exists(), reason="raw CSV not available")]


@pytest.fixture(scope="module")
def loaded():
    st = os.stat(C.PURCHASES_CSV)
    tx, qc = load_transactions(C.PURCHASES_CSV)
    return tx, qc, st


def test_raw_file_is_official_and_untouched(loaded):
    _, _, st = loaded
    assert md5sum(C.PURCHASES_CSV) == C.OFFICIAL_MD5["amazon-purchases.csv"]
    st2 = os.stat(C.PURCHASES_CSV)
    assert (st2.st_size, st2.st_mtime_ns) == (st.st_size, st.st_mtime_ns)


def test_quality_counts(loaded):
    _, qc, _ = loaded
    assert qc["rows"] == 1_850_717 and qc["respondents"] == 5027
    assert qc["unparseable_dates"] == 0 and qc["ids_not_matching_pattern"] == 0
    assert qc["exact_duplicate_extra_copies"] == 11_624 and qc["gift_rows"] == 41_483


# (n, y=1, y=0) from reports/audit_and_feasibility.md §5; the first, a COVID-period and the last cutoff
@pytest.mark.parametrize("cutoff,expected", [("2018-12-31", (4281, 3558, 723)),
                                             ("2020-03-31", (4728, 4056, 672)),
                                             ("2022-06-30", (5008, 4384, 624))])
def test_counts_reconcile_with_stage_a(loaded, cutoff, expected):
    p = build_panel(purchase_days(loaded[0]), [pd.Timestamp(cutoff)])
    assert (len(p), int(p["y"].sum()), int((p["y"] == 0).sum())) == expected


# ---- Stage C ------------------------------------------------------------------------------

def test_features_ignore_post_cutoff_rows_real(loaded):
    from apf.features import features_at, make_daily_store
    tx = loaded[0]
    c = pd.Timestamp("2021-03-31")
    full = features_at(make_daily_store(tx), c)
    assert full.equals(features_at(make_daily_store(tx[tx[C.DATE] <= c]), c))
    assert len(full) == 4915


def test_prev_90d_active_equals_previous_label_real(loaded):
    from apf.features import features_at, make_daily_store
    tx = loaded[0]
    f = features_at(make_daily_store(tx), pd.Timestamp("2022-06-30"))["prev_90d_active"].dropna()
    prev = build_panel(purchase_days(tx), [pd.Timestamp("2022-03-31")]).set_index(C.ID)["y"]
    assert f.index.equals(prev.index) and (f == prev).all()


def test_holdout_real(loaded):
    from apf.splits import holdout_respondents
    ids = build_panel(purchase_days(loaded[0]), C.CUTOFFS)[C.ID]
    h = holdout_respondents(ids)
    assert len(h) == 1002 and h.equals(holdout_respondents(ids.sample(frac=1, random_state=1)))
